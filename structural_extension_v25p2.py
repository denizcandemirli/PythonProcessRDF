# -*- coding: utf-8 -*-
"""
structural_extension_v25p2.py

Purpose (v25p.2 minor patch):

Include M5 (LoadBearing/Shear/Moment/Bracing) shares in the motif heatmap.

Move the S3 system score to a density/role-based formula (no fixed baseline).

Apply proxy and weak topology penalties.

Single-file, standalone execution (no external dependencies other than rdflib, pandas, numpy).

Inputs:

--input-dir, --pattern : Locate RDF files.

Parameters:

--dual-thresh : Dual system threshold (both frame & wall are high).

--w-motif, --w-system : Fusion weights for structural similarity.

--alpha-m5 : Scaling factor for M5 shares in the motif vector.

--proxy-penalty : Multiplier (1 - penalty) if only type-based proxy was used.

--weak-topo-penalty : Multiplier (1 - penalty) if only weak topology (adjacentZone) is present.

--allow-type-only-proxy : Enable type-based proxy counts when no topology is available.

--func-all : Expanded function matching (hasFunction/hasQuality/… + label regex).

--emit-debug : Write debug CSVs.

Outputs (inside out-root/out-name folder):

struct_types_histogram.csv

struct_functions_histogram.csv

struct_functions_shares_wide.csv

struct_data_availability.csv

struct_motif_counts.csv

struct_motif_shares.csv (wide: model × [M2_, M3_, M4_, M2b_, M5_*])

struct_motif_shares_long.csv (long: model, motif, share)

struct_motif_densities_per100.csv

struct_system_scores.csv (frame, wall, dual, braced)

struct_score_components.csv (intermediate values: densities and shares)

struct_similarity_matrix.csv (S_struct = w_motif * cos + w_system * cos)

pairwise_structural_summary.csv

weights_used.json

motif_proxy_summary.csv

type_mapping_hits.csv / type_mapping_unknown.csv (type matches)
"""

import os, re, json, glob, argparse, math
from collections import defaultdict, Counter
import numpy as np
import pandas as pd
from rdflib import Graph, RDF, RDFS

# ---------------------------
# Yardımcılar
# ---------------------------

def localname(x: str) -> str:
    s = str(x)
    if "#" in s:
        s = s.split("#")[-1]
    if "/" in s:
        s = s.split("/")[-1]
    return s

def tokset_from_labels(labels):
    toks = set()
    for t in labels:
        t2 = re.sub(r"[^A-Za-z0-9]+", " ", str(t)).strip().lower()
        if t2:
            toks |= set(t2.split())
    return toks

def cosine(a: np.ndarray, b: np.ndarray) -> float:
    da = np.linalg.norm(a)
    db = np.linalg.norm(b)
    if da == 0 or db == 0:
        return 0.0
    return float(np.dot(a, b) / (da * db))

# ---------------------------
# Tip ve fonksiyon sözlükleri (regex + IFC kısa yolları)
# ---------------------------

TYPE_PATTERNS = {
    "Beam"   : r"(beam|ifcbeam)\b",
    "Column" : r"(column|ifccolumn)\b",
    "Slab"   : r"(slab|deck|floor|plate|ifcslab|ifcfloor)\b",
    "Wall"   : r"(wall|shear[- ]?wall|corewall|retainingwall|ifcwall)\b",
    "Brace"  : r"(brace|bracing|tie|strut|ifcstructuralcurvemember)\b",
    "Core"   : r"(core|shearcore)\b",
    "Foundation": r"(foundation|footing|pile)\b",
}

FUNC_PATTERNS = {
    "LB"      : r"(load\s*bearing|bearing)\b",
    "Shear"   : r"(shear)\b",
    "Moment"  : r"(moment|bending)\b",
    "Bracing" : r"(brace|bracing|tie|strut|stiffener|diaphragm)\b",
}

STRONG_TOPO = {"adjacentelement", "intersectingelement"}
WEAK_TOPO   = {"adjacentzone"}

# motif anahtar adları (rapor/ısı haritası için)
MOTIF_KEYS = ["M2_frameNode", "M3_wallSlab", "M4_core", "M2b_braceNode"]
M5_KEYS    = ["M5_LB", "M5_Shear", "M5_Moment", "M5_Bracing"]

# ---------------------------
# RDF Model Okuma
# ---------------------------

class Model:
    def __init__(self, name, graph: Graph):
        self.name = name
        self.g = graph
        self.types = defaultdict(set)     # node -> {type localnames}
        self.labels = defaultdict(set)    # node -> {labels}
        self.edges_by_pred = defaultdict(list)  # pred_local -> [(s,o),...]

        self._index()

    def _index(self):
        for s, p, o in self.g.triples((None, RDF.type, None)):
            self.types[s].add(localname(o))
        for s, p, o in self.g.triples((None, RDFS.label, None)):
            self.labels[s].add(str(o))

        # tüm predikatları topla (yerel ad ile)
        for s, p, o in self.g:
            pred = localname(p).lower()
            if pred in STRONG_TOPO or pred in WEAK_TOPO or pred in {
                "hascontinuantpart", "haspropercontinuantpart",
                "hasfunction", "hasquality", "hasrole", "hasstructuralfunction", "hasstructuralrole"
            }:
                self.edges_by_pred[pred].append((s, o))

def read_models(input_dir: str, pattern: str):
    paths = sorted(glob.glob(os.path.join(input_dir, pattern)))
    models = []
    for p in paths:
        name = os.path.basename(p)
        try:
            g = Graph()
            # çoğu RDF/XML – rdflib formatı otomatik algılıyor
            g.parse(p)
            models.append(Model(name, g))
        except Exception as e:
            print(f"[WARN] Parse error: {name}: {e}")
    if not models:
        raise RuntimeError("Hiç RDF yüklenemedi; input-dir/pattern kontrol edin.")
    return models

# ---------------------------
# Tip / fonksiyon eşleme
# ---------------------------

def compile_patterns(d: dict):
    return {k: re.compile(v, re.I) for k, v in d.items()}

TYPE_RX = compile_patterns(TYPE_PATTERNS)
FUNC_RX = compile_patterns(FUNC_PATTERNS)

def classify_node_types(types_set, labels_set):
    """Her node'u (Beam/Column/Slab/Wall/Brace/Core/...) kategorilerine eşler."""
    namebag = " ".join(sorted([t.lower() for t in types_set] + list(tokset_from_labels(labels_set))))
    cats = set()
    for cat, rx in TYPE_RX.items():
        if rx.search(namebag):
            cats.add(cat)
    return cats

def classify_functions(func_objs, labels_set, func_all=False):
    """
    Fonksiyon kategorileri: LB / Shear / Moment / Bracing
    - func_objs: hasFunction/hasQuality/hasRole hedef localnames
    - labels_set: node label/annotation (regex için)
    - func_all=True ise label anahtarlarından da türet
    """
    bag = " ".join([t.lower() for t in func_objs] + (list(tokset_from_labels(labels_set)) if func_all else []))
    cats = set()
    for cat, rx in FUNC_RX.items():
        if rx.search(bag):
            cats.add(cat)
    return cats

# ---------------------------
# S1 — Envanter (tip ve fonksiyonlar)
# ---------------------------

def s1_inventories(models, func_all=False):
    rows_types = []
    rows_funcs = []
    func_shares_rows = []   # model × LB/Shear/Moment/Bracing payları
    type_hits = Counter()
    type_unknown = Counter()

    data_av_rows = []       # model bazında veri sinyali (strong/weak/proxy)

    model_node2type = {}    # ileride motiflerde kullanacağız

    for m in models:
        # tip/fonksiyon çıkarımı
        node2cats = defaultdict(set)
        node2funcobjs = defaultdict(set)

        # hasFunction benzeri kenarlardan object localname topla
        for pred in ["hasfunction", "hasquality", "hasrole", "hasstructuralfunction", "hasstructuralrole"]:
            for s, o in m.edges_by_pred.get(pred, []):
                node2funcobjs[s].add(localname(o))

        # node bazında sınıflandırma
        for n in set(list(m.types.keys()) + list(m.labels.keys()) + list(node2funcobjs.keys())):
            cats = classify_node_types(m.types.get(n, set()), m.labels.get(n, set()))
            if cats:
                node2cats[n] |= cats
                for t in m.types.get(n, set()):
                    type_hits[t.lower()] += 1
            else:
                for t in m.types.get(n, set()):
                    type_unknown[t.lower()] += 1

            funs = classify_functions(list(node2funcobjs.get(n, set())),
                                      m.labels.get(n, set()),
                                      func_all=func_all)
            # fonksiyonları node'a yaz (ileride share hesaplayacağız)
            node2funcobjs[n] = funs

        model_node2type[m.name] = node2cats

        # Tip histogramı
        type_counts = Counter()
        for cats in node2cats.values():
            for c in cats:
                type_counts[c] += 1
        for cat, cnt in sorted(type_counts.items()):
            rows_types.append({"model": m.name, "type": cat, "count": int(cnt)})

        # Fonksiyon histogramı
        func_counts = Counter()
        for funs in node2funcobjs.values():
            for f in funs:
                func_counts[f] += 1
        for fcat in ["LB", "Shear", "Moment", "Bracing"]:
            rows_funcs.append({"model": m.name, "function": fcat, "count": int(func_counts.get(fcat, 0))})

        # Fonksiyon payları (modeldeki yapısal eleman sayısına göre normalize)
        n_struct = sum(1 for cats in node2cats.values() if cats & {"Beam","Column","Slab","Wall","Brace","Core","Foundation"})
        denom = max(1, n_struct)
        func_shares_rows.append({
            "model": m.name,
            "LB":      func_counts.get("LB", 0)      / denom,
            "Shear":   func_counts.get("Shear", 0)   / denom,
            "Moment":  func_counts.get("Moment", 0)  / denom,
            "Bracing": func_counts.get("Bracing", 0) / denom,
            "_n_struct": n_struct
        })

        # Veri sinyali (S2/S3 için bilgi): strong/weak topo var mı?
        strong_present = any(m.edges_by_pred.get(p, []) for p in STRONG_TOPO)
        weak_present   = any(m.edges_by_pred.get(p, []) for p in WEAK_TOPO)
        data_av_rows.append({
            "model": m.name,
            "has_strong_topo": int(bool(strong_present)),
            "has_weak_topo": int(bool(weak_present)),
        })

    df_types = pd.DataFrame(rows_types).sort_values(["model", "type"])
    df_funcs = pd.DataFrame(rows_funcs).sort_values(["model", "function"])
    df_func_wide = pd.DataFrame(func_shares_rows).sort_values("model")
    df_av = pd.DataFrame(data_av_rows).sort_values("model")

    # eşleşme listeleri
    hits_rows = [{"token": t, "count": c} for t, c in type_hits.items()]
    unk_rows  = [{"token": t, "count": c} for t, c in type_unknown.items()]
    df_type_hits = pd.DataFrame(hits_rows).sort_values("count", ascending=False)
    df_type_unknown = pd.DataFrame(unk_rows).sort_values("count", ascending=False)

    return df_types, df_funcs, df_func_wide, df_av, model_node2type, df_type_hits, df_type_unknown

# ---------------------------
# S2 — Motif sayımları (proxy/penalty desteği)
# ---------------------------

def s2_motifs(models, model_node2type, allow_type_only, proxy_penalty, weak_topo_penalty):
    rows_counts = []
    rows_dens   = []
    proxy_rows  = []

    def count_pairs(m, Aset, Bset, strong_first=True):
        """A–B arasında güçlü/zayıf topoloji ile eşleşen benzersiz çift sayısı ve hangi kanal kullanıldı."""
        pairs = set()
        used_strong = False
        used_weak = False

        # güçlü topoloji
        for pred in STRONG_TOPO:
            for s, o in m.edges_by_pred.get(pred, []):
                if s in Aset and o in Bset or s in Bset and o in Aset:
                    key = frozenset([s, o])
                    pairs.add(key)
                    used_strong = True

        # zayıf topoloji
        for pred in WEAK_TOPO:
            for s, o in m.edges_by_pred.get(pred, []):
                if s in Aset and o in Bset or s in Bset and o in Aset:
                    key = frozenset([s, o])
                    pairs.add(key)
                    used_weak = True

        return len(pairs), used_strong, used_weak

    for m in models:
        cats = model_node2type[m.name]
        E_beam   = {n for n,cs in cats.items() if "Beam" in cs}
        E_column = {n for n,cs in cats.items() if "Column" in cs}
        E_slab   = {n for n,cs in cats.items() if "Slab" in cs}
        E_wall   = {n for n,cs in cats.items() if "Wall" in cs}
        E_brace  = {n for n,cs in cats.items() if "Brace" in cs}
        E_core   = {n for n,cs in cats.items() if "Core" in cs}

        frame_members = E_beam | E_column
        n_struct = max(1, len(E_beam|E_column|E_slab|E_wall|E_brace|E_core))

        # M2: frame node (beam–column yakınlığı)
        c2, st2, wk2 = count_pairs(m, E_beam, E_column)
        used_proxy_2 = False
        if c2 == 0 and allow_type_only and (E_beam and E_column):
            c2 = min(len(E_beam), len(E_column))  # kaba proxy
            used_proxy_2 = True
        dens2 = (c2 / n_struct) * 100.0

        # M3: wall–slab
        c3, st3, wk3 = count_pairs(m, E_wall, E_slab)
        used_proxy_3 = False
        if c3 == 0 and allow_type_only and (E_wall and E_slab):
            c3 = min(len(E_wall), len(E_slab))
            used_proxy_3 = True
        dens3 = (c3 / n_struct) * 100.0

        # M2b: brace–frame
        c2b, st2b, wk2b = count_pairs(m, E_brace, frame_members)
        used_proxy_2b = False
        if c2b == 0 and allow_type_only and (E_brace and frame_members):
            c2b = min(len(E_brace), len(frame_members))
            used_proxy_2b = True
        dens2b = (c2b / n_struct) * 100.0

        # M4: core çevresinde slab kompozisyonu (yakınlıkla)
        # pratik: en az 2 slab komşusu olan core say
        core_good = 0
        st4 = False; wk4 = False
        if E_core and E_slab:
            # güçlü
            adj = set()
            for pred in STRONG_TOPO:
                for s,o in m.edges_by_pred.get(pred, []):
                    if s in E_core and o in E_slab:
                        adj.add(s); st4 = True
                    if o in E_core and s in E_slab:
                        adj.add(o); st4 = True
            # zayıf
            adj_w = set()
            for pred in WEAK_TOPO:
                for s,o in m.edges_by_pred.get(pred, []):
                    if s in E_core and o in E_slab:
                        adj_w.add(s); wk4 = True
                    if o in E_core and s in E_slab:
                        adj_w.add(o); wk4 = True
            core_good = len(adj|adj_w)
        used_proxy_4 = False
        if core_good == 0 and allow_type_only and E_core:
            core_good = len(E_core)
            used_proxy_4 = True
        dens4 = (core_good / n_struct) * 100.0

        # cezalar
        # proxy cezası motif bazında uygulanır
        def penalize(d, used_proxy, used_strong, used_weak):
            v = d
            if used_proxy:
                v *= (1.0 - proxy_penalty)
            # zayıf-topoloji tek başınaysa ceza
            if (not used_strong) and (used_weak or used_proxy):
                v *= (1.0 - weak_topo_penalty)
            return v

        dens2_p = penalize(dens2, used_proxy_2, st2, wk2)
        dens3_p = penalize(dens3, used_proxy_3, st3, wk3)
        dens2b_p= penalize(dens2b, used_proxy_2b, st2b, wk2b)
        dens4_p = penalize(dens4, used_proxy_4, st4, wk4)

        rows_counts.append({
            "model": m.name,
            "M2_frameNode": c2, "M3_wallSlab": c3, "M4_core": core_good, "M2b_braceNode": c2b,
            "_n_struct": n_struct
        })
        rows_dens.append({
            "model": m.name,
            "M2_frameNode": dens2_p, "M3_wallSlab": dens3_p, "M4_core": dens4_p, "M2b_braceNode": dens2b_p
        })

        proxy_rows.append({
            "model": m.name,
            "M2_proxy": int(used_proxy_2), "M3_proxy": int(used_proxy_3),
            "M2b_proxy": int(used_proxy_2b), "M4_proxy": int(used_proxy_4),
            "M2_strong": int(st2), "M2_weak": int(wk2),
            "M3_strong": int(st3), "M3_weak": int(wk3),
            "M2b_strong": int(st2b), "M2b_weak": int(wk2b),
            "M4_strong": int(st4), "M4_weak": int(wk4),
        })

    df_counts = pd.DataFrame(rows_counts).sort_values("model")
    df_dens   = pd.DataFrame(rows_dens).sort_values("model")
    df_proxy  = pd.DataFrame(proxy_rows).sort_values("model")
    return df_counts, df_dens, df_proxy

# ---------------------------
# S3 — Sistem skorları (yoğunluk/rol tabanlı)
# ---------------------------

def s3_system_scores(models, df_dens, df_func_wide, dual_thresh):
    # dens: per-100 ölçek
    dens = df_dens.set_index("model")[["M2_frameNode","M3_wallSlab","M4_core","M2b_braceNode"]]
    funs = df_func_wide.set_index("model")[["LB","Shear","Moment","Bracing"]]

    rows = []
    comp_rows = []

    for model in dens.index:
        d2  = float(dens.loc[model, "M2_frameNode"]) / 100.0
        d3  = float(dens.loc[model, "M3_wallSlab"]) / 100.0
        d2b = float(dens.loc[model, "M2b_braceNode"]) / 100.0

        fLB = float(funs.loc[model, "LB"])
        fSh = float(funs.loc[model, "Shear"])
        fMo = float(funs.loc[model, "Moment"])
        fBr = float(funs.loc[model, "Bracing"])

        frame  = 0.6*d2 + 0.4*fMo
        wall   = 0.6*d3 + 0.4*((fLB + fSh)/2.0)
        braced = 0.7*d2b + 0.3*fBr

        # dual: eşik kuralı
        if frame >= dual_thresh and wall >= dual_thresh:
            dual = 0.5*(frame + wall)
        else:
            dual = min(frame, wall)

        rows.append({
            "model": model,
            "frame": frame, "wall": wall, "dual": dual, "braced": braced
        })
        comp_rows.append({
            "model": model,
            "dens_M2": d2, "dens_M3": d3, "dens_M2b": d2b,
            "share_LB": fLB, "share_Shear": fSh, "share_Moment": fMo, "share_Bracing": fBr
        })

    df_scores = pd.DataFrame(rows).sort_values("model")
    df_comps  = pd.DataFrame(comp_rows).sort_values("model")
    return df_scores, df_comps

# ---------------------------
# S4 — Yapısal benzerlik (motif + sistem vektörü kosinüs füzyonu)
# ---------------------------

def s4_struct_similarity(df_dens, df_func_wide, df_scores, w_motif, w_system, alpha_m5):
    # motif vektörü = [M2, M3, M4, M2b, alpha*(LB,Shear,Moment,Bracing)]
    A = df_dens.set_index("model")[["M2_frameNode","M3_wallSlab","M4_core","M2b_braceNode"]].copy()
    A = A / 100.0  # 0-1 ölçeğe çek
    B = df_func_wide.set_index("model")[["LB","Shear","Moment","Bracing"]].copy()
    B = alpha_m5 * B
    V = pd.concat([A, B], axis=1)

    models = list(V.index)
    M = len(models)
    S_motif = np.zeros((M, M), dtype=float)
    for i in range(M):
        vi = V.iloc[i].values.astype(float)
        for j in range(M):
            vj = V.iloc[j].values.astype(float)
            S_motif[i,j] = cosine(vi, vj)

    # sistem vektörü = [frame, wall, dual, braced]
    S = df_scores.set_index("model")[["frame","wall","dual","braced"]].copy()
    Svals = S.values.astype(float)
    Ssys = np.zeros((M,M), dtype=float)
    for i in range(M):
        vi = Svals[i]
        for j in range(M):
            vj = Svals[j]
            Ssys[i,j] = cosine(vi, vj)

    Stotal = w_motif*S_motif + w_system*Ssys

    df_total = pd.DataFrame(Stotal, index=models, columns=models)
    return df_total

# ---------------------------
# Ana akış
# ---------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--pattern", default="*_DG.rdf")
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--out-name", required=True)

    ap.add_argument("--dual-thresh", type=float, default=0.25)
    ap.add_argument("--w-motif", type=float, default=0.5)
    ap.add_argument("--w-system", type=float, default=0.5)
    ap.add_argument("--alpha-m5", type=float, default=0.40)
    ap.add_argument("--proxy-penalty", type=float, default=0.70)
    ap.add_argument("--weak-topo-penalty", type=float, default=0.50)

    ap.add_argument("--allow-type-only-proxy", action="store_true")
    ap.add_argument("--func-all", action="store_true")
    ap.add_argument("--emit-debug", action="store_true")

    args = ap.parse_args()

    out_dir = os.path.join(args.out_root, args.out_name)
    os.makedirs(out_dir, exist_ok=True)

    # 1) RDF'leri yükle
    models = read_models(args.input_dir, args.pattern)
    print("[LOAD]")
    for m in models:
        print("  ", m.name)

    # 2) S1: tip ve fonksiyon envanteri
    df_types, df_funcs, df_func_wide, df_av, node2type, df_hits, df_unk = s1_inventories(
        models, func_all=args.func_all
    )

    # 3) S2: motif sayımları + cezalı yoğunluklar
    df_counts, df_dens, df_proxy = s2_motifs(
        models, node2type,
        allow_type_only=args.allow_type_only_proxy,
        proxy_penalty=args.proxy_penalty,
        weak_topo_penalty=args.weak_topo_penalty
    )

    # 4) S3: sistem skorları (yoğunluk/rol tabanlı)
    df_scores, df_comps = s3_system_scores(
        models, df_dens, df_func_wide, dual_thresh=args.dual_thresh
    )

    # 5) S4: yapısal benzerlik (motif + sistem kosinüs füzyonu)
    df_Sstruct = s4_struct_similarity(
        df_dens, df_func_wide, df_scores,
        w_motif=args.w_motif, w_system=args.w_system, alpha_m5=args.alpha_m5
    )

    # ------------ ÇIKTILAR ------------
    # S1
    df_types.to_csv(os.path.join(out_dir, "struct_types_histogram.csv"), index=False)
    df_funcs.to_csv(os.path.join(out_dir, "struct_functions_histogram.csv"), index=False)
    df_func_wide.to_csv(os.path.join(out_dir, "struct_functions_shares_wide.csv"), index=False)
    df_av.to_csv(os.path.join(out_dir, "struct_data_availability.csv"), index=False)
    df_hits.to_csv(os.path.join(out_dir, "type_mapping_hits.csv"), index=False)
    df_unk.to_csv(os.path.join(out_dir, "type_mapping_unknown.csv"), index=False)

    # S2
    df_counts.to_csv(os.path.join(out_dir, "struct_motif_counts.csv"), index=False)
    df_dens.to_csv(os.path.join(out_dir, "struct_motif_densities_per100.csv"), index=False)
    df_proxy.to_csv(os.path.join(out_dir, "motif_proxy_summary.csv"), index=False)

    # Isı haritası için: motif payları (0–1), ayrıca M5 paylarını ekle
    # Not: M2/M3/M4/M2b densiteleri per-100 idi → /100 ile paya çeviriyoruz
    wide_for_heat = df_dens.copy()
    for c in ["M2_frameNode","M3_wallSlab","M4_core","M2b_braceNode"]:
        wide_for_heat[c] = wide_for_heat[c] / 100.0
    # M5: doğrudan df_func_wide’dan
    m5 = df_func_wide[["model","LB","Shear","Moment","Bracing"]].copy()
    m5.rename(columns={
        "LB":"M5_LB","Shear":"M5_Shear","Moment":"M5_Moment","Bracing":"M5_Bracing"
    }, inplace=True)
    df_motif_shares = pd.merge(wide_for_heat, m5, on="model", how="left")
    df_motif_shares = df_motif_shares[["model"] + MOTIF_KEYS + M5_KEYS]
    df_motif_shares.to_csv(os.path.join(out_dir, "struct_motif_shares.csv"), index=False)

    # uzun form (model,motif,share)
    long_rows = []
    for _, row in df_motif_shares.iterrows():
        model = row["model"]
        for k in MOTIF_KEYS + M5_KEYS:
            long_rows.append({"model": model, "motif": k, "share": float(row[k])})
    df_long = pd.DataFrame(long_rows)
    df_long.to_csv(os.path.join(out_dir, "struct_motif_shares_long.csv"), index=False)

    # S3
    df_scores.to_csv(os.path.join(out_dir, "struct_system_scores.csv"), index=False)
    df_comps.to_csv(os.path.join(out_dir, "struct_score_components.csv"), index=False)

    # S4
    df_Sstruct.to_csv(os.path.join(out_dir, "struct_similarity_matrix.csv"))

    # pairwise özet
    models_list = list(df_Sstruct.index)
    rows_pw = []
    for i in range(len(models_list)):
        for j in range(i+1, len(models_list)):
            a = models_list[i]; b = models_list[j]
            rows_pw.append({"A": a, "B": b, "S_struct": float(df_Sstruct.loc[a,b])})
    df_pw = pd.DataFrame(rows_pw).sort_values("S_struct", ascending=False)
    df_pw.to_csv(os.path.join(out_dir, "pairwise_structural_summary.csv"), index=False)

    # weights / konfig
    meta = {
        "dual_thresh": args.dual_thresh,
        "w_motif": args.w_motif, "w_system": args.w_system,
        "alpha_m5": args.alpha_m5,
        "proxy_penalty": args.proxy_penalty,
        "weak_topo_penalty": args.weak_topo_penalty,
        "allow_type_only_proxy": bool(args.allow_type_only_proxy),
        "func_all": bool(args.func_all)
    }
    with open(os.path.join(out_dir, "weights_used.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    if args.emit_debug:
        # Kullanışlı birleştirilmiş csv (raporda hızlı referans için)
        dbg = pd.merge(df_motif_shares, df_scores, on="model", how="left")
        dbg.to_csv(os.path.join(out_dir, "struct_debug_bundle.csv"), index=False)

    print(f"\n[OK] Saved outputs under: {out_dir}")
    for fn in [
        "struct_types_histogram.csv",
        "struct_functions_histogram.csv",
        "struct_functions_shares_wide.csv",
        "struct_data_availability.csv",
        "struct_motif_counts.csv",
        "struct_motif_shares.csv",
        "struct_motif_shares_long.csv",
        "struct_motif_densities_per100.csv",
        "struct_system_scores.csv",
        "struct_score_components.csv",
        "struct_similarity_matrix.csv",
        "pairwise_structural_summary.csv",
        "weights_used.json",
        "motif_proxy_summary.csv",
        "type_mapping_hits.csv",
        "type_mapping_unknown.csv",
        "struct_debug_bundle.csv" if args.emit_debug else None
    ]:
        if fn:
            print(" -", fn)

if __name__ == "__main__":
    main()
