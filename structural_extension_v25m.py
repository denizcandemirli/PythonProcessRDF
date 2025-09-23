# -*- coding: utf-8 -*-
"""
Structural Extension v25m
-------------------------
v25l → v25m iyileştirmeleri:
 - M5 (LoadBearing payı) motif kanalına α katsayısıyla enjekte edilir (M5_structRole).
 - Wall–Slab motifleri için gerçek "proxy" sayımı (cezalı) eklendi; zone ve intersect varyantları desteklenir.
 - "model" sütunu tüm kritik CSV'lerde garanti edilir; boş DataFrame durumları güvenli.
 - weights_used.json çıktısı genişletildi.

Girdi: *_DG.rdf dosyaları (Design Graph) + ontology referansı (opsiyonel).
Çıktılar (out-name klasörü altında):
 - struct_types_histogram.csv
 - struct_functions_histogram.csv
 - struct_functions_shares_wide.csv
 - struct_data_availability.csv
 - struct_motif_counts.csv
 - struct_motif_shares.csv
 - struct_motif_densities_per100.csv
 - struct_system_scores.csv
 - struct_score_components.csv
 - struct_similarity_matrix.csv
 - pairwise_structural_summary.csv
 - weights_used.json
 - (emit-debug) motif_proxy_edges.csv, type_mapping_hits.csv, type_mapping_unknown.csv

Bağımlılıklar: pandas, numpy, rdflib, networkx
"""

import os, re, json, argparse, itertools, math
from collections import defaultdict, Counter

import pandas as pd
import numpy as np
import rdflib as rdf
import networkx as nx

# --------- Namespaces (kısa sabitler)
RDF  = rdf.RDF
RDFS = rdf.RDFS
BOT  = rdf.Namespace("https://w3id.org/bot#")
CORE = rdf.Namespace("https://w3id.org/builtforms/core#")
BFO  = rdf.Namespace("http://purl.obolibrary.org/obo/")

# --------- Argümanlar
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--pattern", default="*_DG.rdf")
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--out-name", required=True)
    ap.add_argument("--dual-thresh", type=float, default=0.25)
    ap.add_argument("--w-motif", type=float, default=0.5)
    ap.add_argument("--w-system", type=float, default=0.5)
    ap.add_argument("--alpha-m5", type=float, default=0.40, help="M5 LB payının motif kanalına enjekte katsayısı")
    ap.add_argument("--proxy-penalty", type=float, default=0.50, help="Proxy motif sayımında ceza katsayısı")
    ap.add_argument("--allow-type-only-proxy", action="store_true")
    ap.add_argument("--func-all", action="store_true", help="Fonksiyon teşhisi: core:hasFunction yoksa tip/label'dan yaklaştır")
    ap.add_argument("--emit-debug", action="store_true")
    return ap.parse_args()

# --------- Yardımcılar
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path

def list_files(input_dir, pattern):
    import glob
    return sorted(glob.glob(os.path.join(input_dir, pattern)))

def safe_df(df, cols=None):
    df = df.copy()
    if cols:
        for c in cols:
            if c not in df.columns:
                df[c] = np.nan
        df = df[cols]
    return df

def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))

# --------- Tip & fonksiyon eşleme sözlükleri
TYPE_MAP_REGEX = [
    # (regex, canonical)
    (r"\bIfcWall\b|(^|\W)Wall(s)?\b|Shear[- ]?Wall\b|CoreWall\b|RetainingWall\b|ExteriorWall\b|InteriorWall\b", "Wall"),
    (r"\bIfcSlab\b|\bIfcFloor\b|(^|\W)(Slab|Deck|Floor|Plate)(s)?\b", "Slab"),
    (r"\bIfcColumn\b|(^|\W)(Column|Pier)(s)?\b", "Column"),
    (r"\bIfcBeam\b|(^|\W)(Beam|Girder)(s)?\b", "Beam"),
    (r"\bIfcStructuralCurveMember\b|(^|\W)(Brace|Bracing|Strut|Tie)(s)?\b", "Brace"),
    (r"\bCore\b|LiftCore\b|StairCore\b", "Core"),
    (r"\bFoundation\b|Footing\b|Pile\b", "Foundation"),
]

FUNC_MAP_REGEX = [
    (r"Load[- ]?Bearing", "LoadBearing"),
    (r"Moment|Bending|RigidFrame|MomentFrame", "Moment"),
    (r"Shear|Diaphragm|Stiffener", "Shear"),
]

# --------- RDF → ham çıkarım
class ModelData:
    def __init__(self, name, graph: rdf.Graph):
        self.name = name
        self.g = graph
        self.labels = {}      # node -> lower label string
        self.types  = defaultdict(set) # node -> set of local type names
        self.funcs  = defaultdict(set) # node -> set of functions (local)
        # edges
        self.adjE = set()     # (e1,e2)
        self.intE = set()
        self.part = set()     # (e,p) continuant part
        self.adjZ = set()     # (e,z)
        self.zone_nodes = set()

def localname(uri):
    try:
        s = str(uri)
        if "#" in s: return s.split("#")[-1]
        return s.rsplit("/", 1)[-1]
    except Exception:
        return str(uri)

def load_model(path):
    g = rdf.Graph()
    g.parse(path)
    md = ModelData(os.path.basename(path), g)
    # labels
    for s, p, o in g.triples((None, RDFS.label, None)):
        if isinstance(o, rdf.term.Literal):
            md.labels[s] = str(o).strip().lower()
    # types
    for s, p, o in g.triples((None, RDF.type, None)):
        md.types[s].add(localname(o))
    # functions
    for s, p, o in g.triples((None, CORE.hasFunction, None)):
        md.funcs[s].add(localname(o))
    # edges
    for s, p, o in g.triples((None, BOT.adjacentElement, None)):
        md.adjE.add((s, o))
        md.adjE.add((o, s))
    for s, p, o in g.triples((None, BOT.intersectingElement, None)):
        md.intE.add((s, o))
        md.intE.add((o, s))
    for s, p, o in g.triples((None, BFO.term("BFO_0000178"), None)):  # hasContinuantPart
        md.part.add((s, o))
    for s, p, o in g.triples((None, BOT.adjacentZone, None)):
        md.adjZ.add((s, o))
        md.zone_nodes.add(o)
    return md

def map_type_of(node, md: ModelData):
    lbl = md.labels.get(node, "")
    ts  = [t.lower() for t in md.types.get(node, [])]
    namecand = " ".join(ts + [lbl]).strip()
    for pat, canon in TYPE_MAP_REGEX:
        if re.search(pat, namecand, re.IGNORECASE):
            return canon
    return None

def has_function(node, md: ModelData, func_all=False):
    # Önce explicit
    fset = {localname(f) for f in md.funcs.get(node, set())}
    for pat, out in FUNC_MAP_REGEX:
        # explicit eşleşme
        if any(re.search(pat, localname(f), re.IGNORECASE) for f in fset):
            if out == "LoadBearing": return "LoadBearing"
            if out == "Moment": return "Moment"
            if out == "Shear":  return "Shear"
    # Yaklaştırma (func-all)
    if func_all:
        lbl = md.labels.get(node, "")
        ts  = [localname(t) for t in md.types.get(node, [])]
        blob = (lbl + " " + " ".join(ts)).lower()
        if re.search(r"load[- ]?bearing", blob, re.IGNORECASE):
            return "LoadBearing"
        if re.search(r"moment|rigidframe|momentframe|bending", blob, re.IGNORECASE):
            return "Moment"
        if re.search(r"shear|diaphragm|stiffener", blob, re.IGNORECASE):
            return "Shear"
    return None

# --------- S1: tip/fonksiyon envanteri + availability
STRUCT_TYPES = ["Wall","Slab","Column","Beam","Brace","Core","Foundation"]

def stage_S1(models, overrides=None, func_all=False, emit_debug=False, out_dir=None):
    rows_t, rows_f = [], []
    hits_rows, unk_rows = [], []

    node2type = {m.name: {} for m in models}
    # type mapping
    for m in models:
        for n in m.types.keys():
            canon = map_type_of(n, m)
            if canon:
                node2type[m.name][n] = canon
                if emit_debug:
                    hits_rows.append({"model": m.name, "node": str(n), "mapped": canon})
            else:
                if emit_debug:
                    unk_rows.append({"model": m.name, "node": str(n), "localtypes": "|".join(sorted(m.types.get(n, []))), "label": m.labels.get(n,"")})

    # type histogram
    for m in models:
        c = Counter(node2type[m.name].values())
        total = sum(c.values()) if c else 0
        for t in STRUCT_TYPES:
            rows_t.append({"model": m.name, "type": t, "count": int(c.get(t,0)), "share": (c.get(t,0)/total if total>0 else 0.0)})

    # function histogram (explicit + optional heuristic)
    for m in models:
        fcount = Counter()
        for n in m.types.keys():
            f = has_function(n, m, func_all=func_all)
            if f: fcount[f] += 1
        total_f = sum(fcount.values())
        # üç fonksiyon için kolonlar
        for f in ["LoadBearing","Moment","Shear"]:
            rows_f.append({"model": m.name, "function": f, "count": int(fcount.get(f,0)), "share": (fcount.get(f,0)/total_f if total_f>0 else 0.0)})

    df_types = pd.DataFrame(rows_t)
    if df_types.empty:
        df_types = pd.DataFrame(columns=["model","type","count","share"])
    df_types = df_types.sort_values(["model","type"], ignore_index=True)

    df_funcs = pd.DataFrame(rows_f)
    if df_funcs.empty:
        df_funcs = pd.DataFrame(columns=["model","function","count","share"])
    df_funcs = df_funcs.sort_values(["model","function"], ignore_index=True)

    # functions shares wide
    df_func_wide = df_funcs.pivot_table(index="model", columns="function", values="share", fill_value=0.0).reset_index()
    for f in ["LoadBearing","Moment","Shear"]:
        if f not in df_func_wide.columns:
            df_func_wide[f] = 0.0

    # availability
    av_rows = []
    for m in models:
        av_rows.append({
            "model": m.name,
            "n_types_total": int(len(node2type[m.name])),
            "has_adjacentElement": int(len(m.adjE)>0),
            "has_intersectingElement": int(len(m.intE)>0),
            "has_part": int(len(m.part)>0),
            "has_adjacentZone": int(len(m.adjZ)>0),
            "n_functions_explicit": int(sum(1 for n in m.types.keys() if len(m.funcs.get(n,set()))>0)),
        })
    df_av = pd.DataFrame(av_rows).sort_values("model", ignore_index=True)

    # debug
    if emit_debug and out_dir:
        safe_df(pd.DataFrame(hits_rows), ["model","node","mapped"]).to_csv(os.path.join(out_dir,"type_mapping_hits.csv"), index=False)
        safe_df(pd.DataFrame(unk_rows),  ["model","node","localtypes","label"]).to_csv(os.path.join(out_dir,"type_mapping_unknown.csv"), index=False)

    return df_types, df_funcs, df_func_wide, df_av, node2type

# --------- S2: yapısal motif sayımı (gerçek + proxy)
def count_edges_between(md: ModelData, node2type, A, B, kind="adj"):
    """ A-B tipleri arasında edge sayısı (adjacentElement/intersectingElement) """
    cnt = 0
    if kind == "adj":
        pool = md.adjE
    elif kind == "int":
        pool = md.intE
    else:
        return 0
    # yönsüz kabul: (u,v) ve (v,u) ikili ekledik; tekil say
    seen = set()
    for u,v in pool:
        if (v,u) in seen: continue
        tu = node2type.get(u); tv = node2type.get(v)
        if tu == A and tv == B or tu == B and tv == A:
            cnt += 1
            seen.add((u,v))
    return cnt

def wall_slab_zone_variant(md: ModelData, node2type):
    """ Wall–Zone–Slab zinciri (proxy veya gerçek varyant). """
    # E_w --adjZ--> Z --adjZ--> E_s
    # basit sayım: tüm Z için Wall ve Slab komşuların min eşleşmesi
    z_to_w = defaultdict(int)
    z_to_s = defaultdict(int)
    for e,z in md.adjZ:
        t = node2type.get(e)
        if t == "Wall": z_to_w[z] += 1
        if t == "Slab": z_to_s[z] += 1
    pairs = 0
    for z in md.zone_nodes:
        w = z_to_w.get(z,0); s = z_to_s.get(z,0)
        if w>0 and s>0:
            pairs += min(w,s)
    return pairs

def stage_S2(models, node2type, df_types, df_funcs, func_wide, alpha_m5=0.4, proxy_penalty=0.5, allow_type_only=False, emit_debug=False, out_dir=None):
    rows_counts = []
    proxy_rows  = []

    for m in models:
        n_wall = sum(1 for n,t in node2type[m.name].items() if t=="Wall")
        n_slab = sum(1 for n,t in node2type[m.name].items() if t=="Slab")
        n_col  = sum(1 for n,t in node2type[m.name].items() if t=="Column")
        n_beam = sum(1 for n,t in node2type[m.name].items() if t=="Beam")
        n_br   = sum(1 for n,t in node2type[m.name].items() if t=="Brace")
        n_core = sum(1 for n,t in node2type[m.name].items() if t=="Core")

        # M2_frameNode: Beam–Column adjacency
        m2_frame = count_edges_between(m, node2type[m.name], "Beam","Column", kind="adj")

        # M2_braceNode: Brace–(Column|Beam) adjacency
        m2_brace = (count_edges_between(m, node2type[m.name], "Brace","Column", kind="adj")
                    + count_edges_between(m, node2type[m.name], "Brace","Beam", kind="adj"))

        # M3_wallSlab (adjacentElement + intersectingElement + zone varyantı)
        m3_adj  = count_edges_between(m, node2type[m.name], "Wall","Slab", kind="adj")
        m3_int  = count_edges_between(m, node2type[m.name], "Wall","Slab", kind="int")
        m3_zone = wall_slab_zone_variant(m, node2type[m.name])

        m3 = m3_adj + m3_int + m3_zone

        # Proxy: hiçbiri yoksa, tip-only proxy
        if m3 == 0 and allow_type_only and (n_wall>0 and n_slab>0):
            prox = int(min(n_wall, n_slab) * proxy_penalty)
            m3 += prox
            proxy_rows.append({"model": m.name, "motif": "M3_wallSlab_proxy", "count": prox})

        # M4_core: çekirdek kompozisyonu (basit: core sayısı)
        m4_core = n_core

        # M5_structRole: LB payını motif kanalına α ile enjekte edeceğiz (counts yerine pay)
        # burada count karşılığı 0 yaz, pay eklemeyi shares aşamasında yapacağız.
        m5_struct = 0

        rows_counts += [
            {"model": m.name, "motif": "M2_frameNode", "count": m2_frame},
            {"model": m.name, "motif": "M2_braceNode", "count": m2_brace},
            {"model": m.name, "motif": "M3_wallSlab",  "count": m3},
            {"model": m.name, "motif": "M4_core",      "count": m4_core},
            {"model": m.name, "motif": "M5_structRole","count": m5_struct},
        ]

    df_counts = pd.DataFrame(rows_counts)
    if df_counts.empty:
        df_counts = pd.DataFrame(columns=["model","motif","count"])
    df_counts = df_counts.sort_values(["model","motif"], ignore_index=True)

    # motif shares (satır normalize)
    if df_counts.empty:
        df_shares = pd.DataFrame(columns=["model","motif","share"])
    else:
        tot = df_counts.groupby("model")["count"].sum().rename("tot").reset_index()
        tmp = df_counts.merge(tot, on="model", how="left")
        tmp["share"] = np.where(tmp["tot"]>0, tmp["count"] / tmp["tot"], 0.0)
        df_shares = tmp[["model","motif","share"]].copy()

    # M5 enjeksiyonu: LoadBearing payını motif kanalına α ile ekle
    if not func_wide.empty:
        lb = func_wide[["model","LoadBearing"]].copy()
        lb = lb.rename(columns={"LoadBearing":"lb_share"})
        df_shares = df_shares.merge(lb, on="model", how="left")
        df_shares["lb_share"] = df_shares["lb_share"].fillna(0.0)
        # sadece M5 satırlarına α*lb_share ekle
        df_shares["share"] = np.where(
            df_shares["motif"]=="M5_structRole",
            df_shares["share"] + (alpha_m5 * df_shares["lb_share"]),
            df_shares["share"]
        )
        df_shares = df_shares.drop(columns=["lb_share"])

    # densities per 100 structural entities (yaklaşık ölçek bağımsızlaştırma)
    dens_rows = []
    nE_map = df_types.groupby("model")["count"].sum().to_dict()
    for _,r in df_counts.iterrows():
        denom = nE_map.get(r["model"], 0)
        val = (100.0 * r["count"] / denom) if denom>0 else 0.0
        dens_rows.append({"model": r["model"], "motif": r["motif"], "per100": val})
    df_dens = pd.DataFrame(dens_rows)
    if df_dens.empty:
        df_dens = pd.DataFrame(columns=["model","motif","per100"])

    # debug proxy
    if emit_debug and out_dir:
        safe_df(pd.DataFrame(proxy_rows), ["model","motif","count"]).to_csv(os.path.join(out_dir,"motif_proxy_edges.csv"), index=False)

    return df_counts, df_shares, df_dens

# --------- S3: sistem skorları (frame, wall, dual, braced)
def stage_S3(models, df_types, df_funcs, df_shares, dual_thresh=0.25):
    # girdiler güvenli
    types_w = df_types.pivot_table(index="model", columns="type", values="count", fill_value=0).reset_index()
    func_w  = df_funcs.pivot_table(index="model", columns="function", values="share", fill_value=0.0).reset_index()
    share_w = df_shares.pivot_table(index="model", columns="motif", values="share", fill_value=0.0).reset_index()

    for col in ["model"]:
        for df in [types_w, func_w, share_w]:
            if col not in df.columns:
                df[col] = ""
    # extract needed columns
    def get_col(df, name):
        return df[name] if name in df.columns else pd.Series([0.0]*len(df))

    # birleşik tablo
    models_list = sorted(set(types_w["model"]) | set(func_w["model"]) | set(share_w["model"]))
    rows = []
    for m in models_list:
        tw = types_w[types_w["model"]==m]
        fw = func_w[func_w["model"]==m]
        sw = share_w[share_w["model"]==m]

        n_col = int(tw["Column"].iloc[0]) if "Column" in tw.columns and not tw.empty else 0
        n_beam= int(tw["Beam"].iloc[0])   if "Beam"   in tw.columns and not tw.empty else 0
        n_wall= int(tw["Wall"].iloc[0])   if "Wall"   in tw.columns and not tw.empty else 0
        n_slab= int(tw["Slab"].iloc[0])   if "Slab"   in tw.columns and not tw.empty else 0
        lb_share = float(fw["LoadBearing"].iloc[0]) if "LoadBearing" in fw.columns and not fw.empty else 0.0

        m2_frame = float(sw["M2_frameNode"].iloc[0]) if "M2_frameNode" in sw.columns and not sw.empty else 0.0
        m3_wall  = float(sw["M3_wallSlab"].iloc[0])  if "M3_wallSlab"  in sw.columns and not sw.empty else 0.0
        m2_brace = float(sw["M2_braceNode"].iloc[0]) if "M2_braceNode" in sw.columns and not sw.empty else 0.0
        m5_role  = float(sw["M5_structRole"].iloc[0])if "M5_structRole" in sw.columns and not sw.empty else 0.0

        # basit kural tabanlı skorlar (0..1 ölçeğinde kalacak şekilde)
        frame_score = min(1.0, m2_frame + 0.5*m5_role)   # çerçeve: düğüm + moment/LB temsili
        wall_score  = min(1.0, m3_wall + 0.5*lb_share)   # duvar: wall-slab motif + LB
        braced_score= min(1.0, m2_brace)                 # brace düğümü varlığı
        dual_score  = 0.0
        if frame_score >= dual_thresh and wall_score >= dual_thresh:
            dual_score = (frame_score + wall_score)/2.0

        rows.append({
            "model": m,
            "frame": frame_score,
            "wall": wall_score,
            "dual": dual_score,
            "braced": braced_score,
            # bileşenleri raporla
            "m2_frameNode": m2_frame,
            "m3_wallSlab": m3_wall,
            "m2_braceNode": m2_brace,
            "m5_structRole": m5_role,
            "lb_share": lb_share,
            "n_col": n_col, "n_beam": n_beam, "n_wall": n_wall, "n_slab": n_slab
        })

    df_sys = pd.DataFrame(rows).sort_values("model", ignore_index=True)
    # ayrı bileşen dosyası
    comp_cols = ["model","m2_frameNode","m3_wallSlab","m2_braceNode","m5_structRole","lb_share","n_col","n_beam","n_wall","n_slab"]
    df_comp = df_sys[comp_cols].copy()
    return df_sys, df_comp

# --------- S4: yapısal benzerlik (motif + sistem füzyonu)
def stage_S4(df_shares, df_sys, w_motif=0.5, w_system=0.5):
    # motif tablosu → wide
    M = df_shares.pivot_table(index="model", columns="motif", values="share", fill_value=0.0)
    M = M.reindex(sorted(M.index), axis=0)
    # sistem tablosu → vektör
    S = df_sys.set_index("model")[["frame","wall","dual","braced"]].fillna(0.0)
    S = S.reindex(sorted(S.index), axis=0)

    models = list(M.index)
    n = len(models)
    Smotif = np.zeros((n,n))
    Ssys   = np.zeros((n,n))
    for i,a in enumerate(models):
        for j,b in enumerate(models):
            Smotif[i,j] = cosine(M.loc[a].values.astype(float), M.loc[b].values.astype(float))
            Ssys[i,j]   = cosine(S.loc[a].values.astype(float), S.loc[b].values.astype(float))
    Sfinal = w_motif*Smotif + w_system*Ssys

    # matrisi dataFrame olarak döndür
    df_final = pd.DataFrame(Sfinal, index=models, columns=models).reset_index().rename(columns={"index":"model"})
    df_motif = pd.DataFrame(Smotif, index=models, columns=models).reset_index().rename(columns={"index":"model"})
    df_sysm  = pd.DataFrame(Ssys,   index=models, columns=models).reset_index().rename(columns={"index":"model"})
    return df_final, df_motif, df_sysm

def pairwise_summary(df_final):
    # üst üçgeni uzun forma dök
    models = df_final["model"].tolist()
    mat = df_final.set_index("model").loc[models, models].values
    rows = []
    for i in range(len(models)):
        for j in range(i+1, len(models)):
            rows.append({"A": models[i], "B": models[j], "struct_similarity": float(mat[i,j])})
    return pd.DataFrame(rows).sort_values("struct_similarity", ascending=False, ignore_index=True)

# --------- main
def main():
    args = parse_args()
    out_dir = ensure_dir(os.path.join(args.out_root, args.out_name))

    # yükleme
    files = list_files(args.input_dir, args.pattern)
    if not files:
        print("[WARN] No files matched.")
        return
    print("[LOAD]")
    models = []
    for f in files:
        print("  ", os.path.basename(f))
        models.append(load_model(f))

    # S1
    df_types, df_funcs, df_func_wide, df_av, node2type = stage_S1(
        models, overrides=None, func_all=args.func_all,
        emit_debug=args.emit_debug, out_dir=out_dir
    )
    safe_df(df_types, ["model","type","count","share"]).to_csv(os.path.join(out_dir,"struct_types_histogram.csv"), index=False)
    safe_df(df_funcs, ["model","function","count","share"]).to_csv(os.path.join(out_dir,"struct_functions_histogram.csv"), index=False)
    safe_df(df_func_wide, ["model","LoadBearing","Moment","Shear"]).to_csv(os.path.join(out_dir,"struct_functions_shares_wide.csv"), index=False)
    safe_df(df_av, ["model","n_types_total","has_adjacentElement","has_intersectingElement","has_part","has_adjacentZone","n_functions_explicit"]).to_csv(os.path.join(out_dir,"struct_data_availability.csv"), index=False)

    # S2
    df_counts, df_shares, df_dens = stage_S2(
        models, node2type, df_types, df_funcs, df_func_wide,
        alpha_m5=args.alpha_m5,
        proxy_penalty=args.proxy_penalty,
        allow_type_only=args.allow_type_only_proxy,
        emit_debug=args.emit_debug, out_dir=out_dir
    )
    safe_df(df_counts, ["model","motif","count"]).to_csv(os.path.join(out_dir,"struct_motif_counts.csv"), index=False)
    safe_df(df_shares, ["model","motif","share"]).to_csv(os.path.join(out_dir,"struct_motif_shares.csv"), index=False)
    safe_df(df_dens,   ["model","motif","per100"]).to_csv(os.path.join(out_dir,"struct_motif_densities_per100.csv"), index=False)

    # S3
    df_sys, df_comp = stage_S3(models, df_types, df_funcs, df_shares, dual_thresh=args.dual_thresh)
    safe_df(df_sys,  ["model","frame","wall","dual","braced"]).to_csv(os.path.join(out_dir,"struct_system_scores.csv"), index=False)
    safe_df(df_comp, ["model","m2_frameNode","m3_wallSlab","m2_braceNode","m5_structRole","lb_share","n_col","n_beam","n_wall","n_slab"]).to_csv(os.path.join(out_dir,"struct_score_components.csv"), index=False)

    # S4
    df_final, df_motifM, df_sysM = stage_S4(df_shares, df_sys, w_motif=args.w_motif, w_system=args.w_system)
    safe_df(df_final, ["model"] + sorted([c for c in df_final.columns if c!="model"])).to_csv(os.path.join(out_dir,"struct_similarity_matrix.csv"), index=False)
    pw = pairwise_summary(df_final)
    safe_df(pw, ["A","B","struct_similarity"]).to_csv(os.path.join(out_dir,"pairwise_structural_summary.csv"), index=False)

    # weights & meta
    meta = {
        "w_motif": args.w_motif, "w_system": args.w_system,
        "alpha_m5": args.alpha_m5, "proxy_penalty": args.proxy_penalty,
        "allow_type_only_proxy": bool(args.allow_type_only_proxy),
        "dual_thresh": args.dual_thresh,
        "files": [os.path.basename(f) for f in files],
        "notes": "type-only proxy and M5 (LoadBearing) share injection applied" if args.allow_type_only_proxy else "M5 injection applied",
    }
    with open(os.path.join(out_dir,"weights_used.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("\n[OK] Saved outputs under:", out_dir)
    for fn in [
        "struct_types_histogram.csv",
        "struct_functions_histogram.csv",
        "struct_functions_shares_wide.csv",
        "struct_data_availability.csv",
        "struct_motif_counts.csv",
        "struct_motif_shares.csv",
        "struct_motif_densities_per100.csv",
        "struct_system_scores.csv",
        "struct_score_components.csv",
        "struct_similarity_matrix.csv",
        "pairwise_structural_summary.csv",
        "weights_used.json"
    ]:
        print(" -", fn)

if __name__ == "__main__":
    main()
