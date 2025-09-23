# -*- coding: utf-8 -*-
"""
Structural Extension v25h
Amaç (S1–S4):
- S1: Yapısal tip & fonksiyon envanteri (genişletilmiş regex + IFC kısa yolu + izlenebilirlik)
- S2: Yapısal motifler (M2/M3/M4/Brace/M5) — predicate varyantları + proxy’ler
- S3: Sistem skorları (frame / wall / braced / dual) — Eurocode/ASCE esinli kurallar + ceza
- S4: Yapısal benzerlik matrisi (motif payları + sistem vektörü, cosine) ve özet

YENİ (v25h) — v25g/v25c’e göre:
- Tip sözlüğü genişledi: Wall/Slab/Beam/Brace/Column/Core/Foundation için sinonimler + IFC short-circuit
- M3_proxy: Wall–Slab komşuluğu için adjacency’e ek olarak “shared zone” ve “part-of zinciri” (hasContinuantPart) varyantları
- Brace/Frame sinyali: Brace sinonimleri + IfcStructuralCurveMember(Type=BRACE), Moment/RigidFrame gibi frame etiketleri
- Dual kalibrasyonu: frame katkısı < threshold → dual=0 (cezalı)
- Debug/diagnostic dosyaları: type_mapping_hits/unknown + motif_proxy_edges + weights_used.json

Bağımlılıklar: rdflib, pandas, numpy
Koşum:
  python structural_extension_v25h.py \
    --input-dir "." \
    --pattern "*_DG.rdf" \
    --out-root ".\\repro_pack\\output" \
    --out-name "07 - Structural_Extension_v25h" \
    --func-all \
    --dual-thresh 0.25 \
    --emit-debug
"""

import argparse, os, re, json, math, itertools, collections
from pathlib import Path
import numpy as np
import pandas as pd
from rdflib import Graph, URIRef, RDF, RDFS, Namespace

# --- Namespaces (esnek)
BOT = Namespace("https://w3id.org/bot#")
CORE = Namespace("https://example.org/core#")  # senin RDF’lerdeki core prefix’ine uyacak şekilde esnek bırakıldı
BFO  = Namespace("http://purl.obolibrary.org/obo/")  # BFO_0000178 hasContinuantPart
# Not: CORE ve diğer prefix’lerde farklı IRI’lar olabilir; predicate’leri sadece localname eşlemesiyle de yakalıyoruz.

# -------- Yardımcılar

def localname(uri):
    s = str(uri)
    if "#" in s: return s.rsplit("#",1)[1]
    return s.rstrip("/").rsplit("/",1)[-1]

def ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)
    return p

def cosine(a, b):
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    da = np.dot(a, a); db = np.dot(b, b)
    if da == 0 or db == 0: return 0.0
    return float(np.dot(a, b) / math.sqrt(da*db))

def read_rdf(path):
    g = Graph()
    # format'ı otomatik algılamaya bırakıyoruz; RDF/XML ağırlıklı.
    g.parse(path)
    return g

# -------- v25h: Genişletilmiş tip/IFC sözlüğü

COARSE_TYPES = {
    "Wall": [
        r"(^|[^A-Za-z])(Wall|Shear[- ]?Wall|CoreWall|RetainingWall|ExteriorWall|InteriorWall)([^A-Za-z]|$)",
        r"(^|[^A-Za-z])(Partition)([^A-Za-z]|$)"  # opsiyonel; yanlış-pozitifler için dikkatli kullanılacak
    ],
    "Slab": [
        r"(^|[^A-Za-z])(Slab|Floor|Deck|Plate|Panel)([^A-Za-z]|$)"
    ],
    "Beam": [
        r"(^|[^A-Za-z])(Beam|Girder|Joist|Rafter|Spandrel)([^A-Za-z]|$)",
        r"(^IfcBeam)"
    ],
    "Column": [
        r"(^|[^A-Za-z])(Column|Pier|Pillar)([^A-Za-z]|$)",
        r"(^IfcColumn)"
    ],
    "Brace": [
        r"(^|[^A-Za-z])(Brace|Bracing|Tie|Strut)([^A-Za-z]|$)",
        r"(^IfcStructuralCurveMember).*BRACE"
    ],
    "Core": [
        r"(^|[^A-Za-z])(Core|CoreWall)([^A-Za-z]|$)"
    ],
    "Foundation": [
        r"(^|[^A-Za-z])(Foundation|Footing|Pile|MatFoundation|Raft)([^A-Za-z]|$)"
    ],
}

# IFC doğrudan sınıf adları (short-circuit)
IFC_CLASS_PREFIXES = {
    "Wall": ["IfcWall", "IfcWallStandardCase"],
    "Slab": ["IfcSlab", "IfcFloor"],
    "Beam": ["IfcBeam"],
    "Column":["IfcColumn"],
    "Brace": ["IfcStructuralCurveMember"],  # type=BRACE kontrolü altta label’da regex ile
    "Core":  ["IfcBuildingElementProxy"],    # core/liftshaft gibi label ile yakalanabilir (esnek)
    "Foundation": ["IfcFooting", "IfcPile", "IfcFoundation"]
}

STRUCT_WHITE = set(COARSE_TYPES.keys())

# Fonksiyon/rol eşlemesi (etiket & tip tabanlı)
FUNC_MAP = {
    "LoadBearing": [r"Load[- ]?Bearing", r"bearing"],
    "Shear":       [r"Shear", r"shear"],
    "Moment":      [r"Moment|RigidFrame|MomentFrame", r"moment"]
}

# Predicate localname eşlemeleri (esnek, prefix bağımsız)
PRED_EQUIV = {
    "adjacentElement": {"adjacentElement"},
    "adjacentZone":    {"adjacentZone"},
    "intersectingElement": {"intersectingElement"},
    "hasContinuantPart": {"BFO_0000178", "hasContinuantPart"},
    "hasFunction": {"hasFunction", "hasStructuralFunction", "hasRole"}
}

def pred_key(p):
    return localname(p)

def label_of(g, n):
    lbl = None
    for o in g.objects(n, RDFS.label):
        try:
            lbl = str(o)
            break
        except: pass
    return lbl

def match_any(regex_list, text):
    if text is None: return False
    for pat in regex_list:
        if re.search(pat, text, flags=re.IGNORECASE):
            return True
    return False

# -------- S1: Tip & fonksiyon envanteri (izlenebilirlik dahil)

class Model:
    def __init__(self, name, graph):
        self.name = name
        self.g = graph
        self.nodes = set()
        self.types = collections.defaultdict(set)  # node -> {local types}
        self.labels = {}                           # node -> rdfs:label (varsa)
        self.funcs = collections.defaultdict(set)  # element -> {func class}
        self.adj_edges = []       # (a,b,"adjacentElement"), vs.
        self.zone_edges = []      # (a,b,"adjacentZone")
        self.inters_edges = []    # (a,b,"intersectingElement")
        self.part_edges = []      # (a,b,"hasContinuantPart")
        self.struct_nodes = set() # structural beyaz listeden biri atanmış düğümler

def build_model(path):
    g = read_rdf(path)
    m = Model(Path(path).name, g)

    # Tipler
    for s, o in g.subject_objects(RDF.type):
        m.nodes.add(s)
        m.types[s].add(localname(o))
    # Label
    for s in set(m.nodes):
        lbl = label_of(g, s)
        if lbl: m.labels[s] = lbl

    # Fonksiyon
    # ?e core:hasFunction ?f ; ?f rdf:type ?ft OR rdfs:label/lexical
    for p in g.predicates():
        lk = pred_key(p)
        if lk in PRED_EQUIV["hasFunction"]:
            for e, f in g.subject_objects(p):
                # Tip/label oku
                ftypes = [localname(t) for t in g.objects(f, RDF.type)]
                flabel = label_of(g, f)
                # Eşleme
                bucket = set()
                for F, regs in FUNC_MAP.items():
                    # tip listesi ya da label üzerinde regex
                    if any(match_any(regs, t) for t in ftypes) or match_any(regs, flabel):
                        bucket.add(F)
                if bucket:
                    m.funcs[e].update(bucket)

    # Kenarlar (predicate localname ile)
    for s,p,o in g:
        pk = pred_key(p)
        if pk in PRED_EQUIV["adjacentElement"]:
            m.adj_edges.append((s,o,"adjacentElement"))
            m.adj_edges.append((o,s,"adjacentElement"))
        elif pk in PRED_EQUIV["adjacentZone"]:
            m.zone_edges.append((s,o,"adjacentZone"))
            m.zone_edges.append((o,s,"adjacentZone"))
        elif pk in PRED_EQUIV["intersectingElement"]:
            m.inters_edges.append((s,o,"intersectingElement"))
            m.inters_edges.append((o,s,"intersectingElement"))
        elif pk in PRED_EQUIV["hasContinuantPart"]:
            m.part_edges.append((s,o,"hasContinuantPart"))

    return m

def source_hit(row_source, hits, node, coarse):
    hits.append({
        "model": row_source["model"],
        "node":  str(node),
        "coarse_type": coarse,
        "source": row_source["source"],
        "matched": row_source["matched"]
    })

def stage_S1(models, func_all=False, emit_debug=False, out_dir=None):
    """
    Tip eşleme sırası (v25h):
      1) IFC sınıf kısa yolu  (IfcWall*, IfcSlab*, IfcBeam*…)
      2) Ontoloji tip adı     (localname "Wall", "Slab", …)
      3) Label/regex eşleşmesi (ExteriorWall, Deck, Girder, Bracing…)
      4) (Opsiyonel) Partition → Wall (LB fonksiyonu ya da kalınlık/komşuluk eşiği varsa) [şimdilik LB varlığı koşullu]
    """
    rows_types = []
    rows_funcs = []
    hits = []
    unknown_rows = []
    node2type = {m.name:{} for m in models}

    for m in models:
        # S1a: Tip eşleme
        for n in m.nodes:
            # Topla aday metinler
            ln_types = m.types.get(n, set())
            lbl = m.labels.get(n, None)

            assigned = set()

            # (1) IFC short-circuit
            for coarse, ifc_list in IFC_CLASS_PREFIXES.items():
                for t in ln_types:
                    if any(t.startswith(pref) for pref in ifc_list):
                        assigned.add(coarse)
                        source_hit({"model":m.name,"source":"ifc_local","matched":t}, hits, n, coarse)

            # (2) Ontoloji localname
            for coarse, regs in COARSE_TYPES.items():
                for t in ln_types:
                    if match_any(regs, t):
                        assigned.add(coarse)
                        source_hit({"model":m.name,"source":"ontology","matched":t}, hits, n, coarse)

            # (3) Label/regex
            if lbl:
                for coarse, regs in COARSE_TYPES.items():
                    if match_any(regs, lbl):
                        assigned.add(coarse)
                        source_hit({"model":m.name,"source":"label_regex","matched":lbl}, hits, n, coarse)

            # (4) Partition → Wall koşulu (func_all ise koşulsuz; değilse LB varsa)
            if ("Partition" in (lbl or "")) or any("Partition" in t for t in ln_types):
                cond = func_all or ("LoadBearing" in m.funcs.get(n, set()))
                if cond:
                    assigned.add("Wall")
                    source_hit({"model":m.name,"source":"partition_promote","matched":lbl or ";".join(ln_types)}, hits, n, "Wall")

            # Struct beyaz listeye düştüyse kaydet
            assigned = {c for c in assigned if c in STRUCT_WHITE}
            if assigned:
                node2type[m.name][n] = assigned
            else:
                # unknown kaydı
                unknown_rows.append({"model":m.name, "node":str(n), "types":";".join(ln_types), "label":lbl or ""})

        # S1b: Tip histogram
        c = collections.Counter()
        for n,cs in node2type[m.name].items():
            for t in cs: c[t]+=1
        for t, cnt in sorted(c.items()):
            rows_types.append({"model":m.name, "type":t, "count":cnt})

        # S1c: Fonksiyon histogram
        if func_all:
            # element'e bağlı tüm fonksiyon bireylerini tip/label bazlı taradık; burada sadece ana üçlüye indirgeriz
            fcount = collections.Counter()
            for e,fs in m.funcs.items():
                for F in fs:
                    fcount[F]+=1
        else:
            # sadece beyaz listedeki (struct) düğümlerdeki fonksiyonları say
            fcount = collections.Counter()
            for e,fs in m.funcs.items():
                if e in node2type[m.name]:
                    for F in fs: fcount[F]+=1

        if fcount:
            for F, cnt in sorted(fcount.items()):
                rows_funcs.append({"model":m.name, "function":F, "count":cnt})

    df_types = pd.DataFrame(rows_types).sort_values(["model","type"]) if rows_types else pd.DataFrame(columns=["model","type","count"])
    df_funcs = pd.DataFrame(rows_funcs).sort_values(["model","function"]) if rows_funcs else pd.DataFrame(columns=["model","function","count"])

    # Fonksiyon paylarını geniş format
    wide = df_funcs.pivot(index="model", columns="function", values="count").fillna(0.0)
    wide_share = (wide.T / wide.sum(axis=1).replace(0, np.nan)).T.fillna(0.0).reset_index()

    if out_dir:
        ensure_dir(out_dir)
        df_types.to_csv(os.path.join(out_dir, "struct_types_histogram.csv"), index=False)
        df_funcs.to_csv(os.path.join(out_dir, "struct_functions_histogram.csv"), index=False)
        wide_share.to_csv(os.path.join(out_dir, "struct_functions_shares_wide.csv"), index=False)
        pd.DataFrame(hits).to_csv(os.path.join(out_dir, "type_mapping_hits.csv"), index=False)
        pd.DataFrame(unknown_rows).to_csv(os.path.join(out_dir, "type_mapping_unknown.csv"), index=False)

    return df_types, df_funcs, wide_share, node2type

# -------- S2: Motifler (M2/M3/M4/Brace/M5)

def stage_S2(models, node2type, out_dir=None, emit_debug=False):
    """
    Motifler:
      M2_frameNode: Beam–Column adjacency (adjacentElement)
      M3_wallSlabAdj: Wall–Slab adjacency (adjacentElement veya adjacentZone veya PROXY)
         PROXY-1: sameZone (Wall adjZone Z & Slab adjZone Z)
         PROXY-2: part-chain (Wall hasPart p ... AND p adjacentElement Slab) [yaklaşık]
      M4_core: Core çevresi (adjacentElement ile ≥2 Slab/Wall komşu)
      Brace motif: Brace komşuluk derecesi (adjacentElement)
      M5_structRole: E → F_(LB/Shear/Moment) (S1’den)
    """
    models_names = [m.name for m in models]
    # sayımlar
    motif_counts = {m.name: {"M2_frameNode":0, "M3_wallSlabAdj":0, "M4_core":0, "M_braceAdj":0, "M5_structRole":0} for m in models}
    proxy_rows = []

    # yardımcı indeksler
    zone_adj = {m.name: collections.defaultdict(set) for m in models}  # node -> {zones}
    elem_adj = {m.name: collections.defaultdict(set) for m in models}  # node -> {elems}

    for m in models:
        # adjacency indeksleri
        for a,b,_ in m.adj_edges:
            elem_adj[m.name][a].add(b)
        for a,b,_ in m.zone_edges:
            zone_adj[m.name][a].add(b)

    def isType(mname, n, coarse):
        return (n in node2type[mname]) and (coarse in node2type[mname][n])

    # M5: fonksiyonlar (element -> funcs) say
    for m in models:
        for e,fs in m.funcs.items():
            if any(F in {"LoadBearing","Shear","Moment"} for F in fs):
                motif_counts[m.name]["M5_structRole"] += 1

    for m in models:
        # M2: Beam–Column adjacency
        seen_pairs = set()
        for a in list(elem_adj[m.name].keys()):
            for b in elem_adj[m.name][a]:
                if (b,a) in seen_pairs: continue
                if isType(m.name, a, "Beam") and isType(m.name, b, "Column"):
                    motif_counts[m.name]["M2_frameNode"] += 1
                    seen_pairs.add((a,b))
                elif isType(m.name, a, "Column") and isType(m.name, b, "Beam"):
                    motif_counts[m.name]["M2_frameNode"] += 1
                    seen_pairs.add((a,b))

        # M3: Wall–Slab adjacency + PROXY
        seen_ws = set()
        # Adjacency tabanlı
        for a in list(elem_adj[m.name].keys()):
            for b in elem_adj[m.name][a]:
                if (b,a) in seen_ws: continue
                if (isType(m.name, a,"Wall") and isType(m.name, b,"Slab")) or (isType(m.name, a,"Slab") and isType(m.name, b,"Wall")):
                    motif_counts[m.name]["M3_wallSlabAdj"] += 1
                    seen_ws.add((a,b))
        # PROXY-1: sameZone
        for w in list(zone_adj[m.name].keys()):
            if not isType(m.name, w, "Wall"): continue
            wz = zone_adj[m.name][w]
            if not wz: continue
            # slab’ler içinde aynı zone’a bağlı olanlar
            for s, Sz in zone_adj[m.name].items():
                if s == w: continue
                if not isType(m.name, s, "Slab"): continue
                if wz.intersection(Sz):
                    motif_counts[m.name]["M3_wallSlabAdj"] += 1
                    proxy_rows.append({"model":m.name, "proxy":"sameZone", "wall":str(w), "slab":str(s)})
        # PROXY-2: part-chain ~ wall hasPart p ... ve p adjacent slab (yaklaşık)
        part_of = collections.defaultdict(set)
        for a,b,_ in m.part_edges:
            part_of[a].add(b)
        for w in part_of.keys():
            if not isType(m.name, w, "Wall"): continue
            parts = part_of[w]
            for p in parts:
                for s in elem_adj[m.name].get(p, []):
                    if isType(m.name, s, "Slab"):
                        motif_counts[m.name]["M3_wallSlabAdj"] += 1
                        proxy_rows.append({"model":m.name, "proxy":"partChain", "wall":str(w), "slab":str(s), "part":str(p)})

        # M4: Core çevresi (≥2 slab OR wall komşu)
        for cnd in list(elem_adj[m.name].keys()):
            if not isType(m.name, cnd, "Core"): continue
            neigh = elem_adj[m.name][cnd]
            cnt = sum(1 for x in neigh if isType(m.name,x,"Slab") or isType(m.name,x,"Wall"))
            if cnt >= 2:
                motif_counts[m.name]["M4_core"] += 1

        # Brace motif: Brace komşulukları
        for a in list(elem_adj[m.name].keys()):
            if isType(m.name, a, "Brace"):
                deg = len([b for b in elem_adj[m.name][a] if b in node2type[m.name]])
                if deg > 0:
                    motif_counts[m.name]["M_braceAdj"] += 1

    # Tabloya dök
    df_counts = pd.DataFrame([
        {"model":m, **motif_counts[m]} for m in motif_counts.keys()
    ]).sort_values("model")

    # Paylar (satır-normalize)
    df_vals = df_counts.set_index("model")
    shares = (df_vals.T / df_vals.sum(axis=1).replace(0, np.nan)).T.fillna(0.0).reset_index()

    # Densities (per 100 structural elements)
    dens = []
    for m in models:
        # structural node sayısı
        nE = sum(1 for n in node2type[m.name].keys() if node2type[m.name][n] & STRUCT_WHITE)
        denom = max(nE, 1)
        row = {"model":m.name}
        for col in df_vals.columns:
            row[col] = (float(df_vals.loc[m.name, col]) / denom) * 100.0
        dens.append(row)
    df_dens = pd.DataFrame(dens).sort_values("model")

    if out_dir:
        df_counts.to_csv(os.path.join(out_dir, "struct_motif_counts.csv"), index=False)
        shares.to_csv(os.path.join(out_dir, "struct_motif_shares.csv"), index=False)
        df_dens.to_csv(os.path.join(out_dir, "struct_motif_densities_per100.csv"), index=False)
        if emit_debug:
            pd.DataFrame(proxy_rows).to_csv(os.path.join(out_dir, "motif_proxy_edges.csv"), index=False)

    return df_counts, shares, df_dens

# -------- S3: Sistem skorları

def stage_S3(models, node2type, func_shares_wide, motif_dens, dual_thresh=0.25, out_dir=None):
    """
    frame ~ M2_frameNode_density + moment_share
    wall  ~ M3_wallSlabAdj_density + 0.5*LB_share + 0.5*shear_share
    braced~ braceAdj_density
    dual  ~ min(frame, wall), ancak frame < dual_thresh ise 0 (cezalı)
    """
    # payları oku
    func_w = func_shares_wide.set_index("model") if not func_shares_wide.empty else pd.DataFrame()
    dens   = motif_dens.set_index("model")
    # güvenli sütun erişimi
    def col(df, m, c):
        try:
            return float(df.loc[m, c]) if (c in df.columns and m in df.index) else 0.0
        except: return 0.0

    rows_comp = []
    rows_scores = []

    for m in dens.index:
        frame_motif = col(dens, m, "M2_frameNode")
        wall_m3     = col(dens, m, "M3_wallSlabAdj")
        brace_m     = col(dens, m, "M_braceAdj")
        core_m      = col(dens, m, "M4_core")

        LB_share    = col(func_w, m, "LoadBearing")
        shear_share = col(func_w, m, "Shear")
        moment_share= col(func_w, m, "Moment")

        # bileşenler
        frame_comp = 0.6*frame_motif + 0.4*moment_share*100.0  # moment share’ı yüzdelik ölçekle ağırlıklandırıldı
        wall_comp  = 0.6*wall_m3     + 0.4*((0.5*LB_share + 0.5*shear_share)*100.0)
        braced_comp= brace_m

        # skorlar (0-1’e ölçekleyelim; 100’lükten normalize)
        # normalize için kaba bir ölçek:  per100 zaten %; moment/LB paylarını 100 ile çarptık. 100’e bölerek 0–1’e getir.
        frame = min(frame_comp/100.0, 1.0)
        wall  = min(wall_comp/100.0, 1.0)
        braced= min(braced_comp/100.0, 1.0)

        dual_raw = min(frame, wall)
        dual = 0.0 if frame < dual_thresh else dual_raw

        rows_comp.append({
            "model":m,
            "frame_motif":frame_motif, "moment_share":moment_share,
            "wall_m3":wall_m3, "LB_share":LB_share, "shear_share":shear_share,
            "brace_m":brace_m, "core_m":core_m
        })
        rows_scores.append({
            "model":m,
            "frame":round(frame,4),
            "wall": round(wall,4),
            "braced":round(braced,4),
            "dual": round(dual,4)
        })

    df_comp   = pd.DataFrame(rows_comp).sort_values("model")
    df_scores = pd.DataFrame(rows_scores).sort_values("model")
    if out_dir:
        df_comp.to_csv(os.path.join(out_dir,"struct_score_components.csv"), index=False)
        df_scores.to_csv(os.path.join(out_dir,"struct_system_scores.csv"), index=False)
    return df_scores, df_comp

# -------- S4: Benzerlik ve özet

def stage_S4(models, motif_shares, system_scores, out_dir=None, w_motif=0.5, w_sys=0.5):
    # motif payları (model × motif)
    M = motif_shares.set_index("model")
    motifs = list(M.columns)

    # sistem vektörü
    S = system_scores.set_index("model")[["frame","wall","braced","dual"]]

    names = list(M.index)
    n = len(names)
    Smat_motif = np.zeros((n,n))
    Smat_sys   = np.zeros((n,n))
    Smat_final = np.zeros((n,n))
    rows_pair  = []

    for i,a in enumerate(names):
        for j,b in enumerate(names):
            va = M.loc[a, :].values
            vb = M.loc[b, :].values
            sa = S.loc[a, :].values if a in S.index else np.zeros(4)
            sb = S.loc[b, :].values if b in S.index else np.zeros(4)
            c1 = cosine(va, vb)
            c2 = cosine(sa, sb)
            Smat_motif[i,j] = c1
            Smat_sys[i,j]   = c2
            Smat_final[i,j] = w_motif*c1 + w_sys*c2
            if i<j:
                rows_pair.append({
                    "model_a":a,"model_b":b,
                    "S_motif":round(c1,4),
                    "S_system":round(c2,4),
                    "S_struct_total":round(Smat_final[i,j],4)
                })
    df_pair = pd.DataFrame(rows_pair).sort_values(["S_struct_total","S_motif","S_system"], ascending=False)
    df_mat  = pd.DataFrame(Smat_final, index=names, columns=names)
    if out_dir:
        df_mat.to_csv(os.path.join(out_dir,"struct_similarity_matrix.csv"))
        df_pair.to_csv(os.path.join(out_dir,"pairwise_structural_summary.csv"), index=False)
        with open(os.path.join(out_dir,"weights_used.json"),"w",encoding="utf-8") as f:
            json.dump({"w_motif":w_motif, "w_system":w_sys}, f, indent=2)
    return df_mat, df_pair

# -------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--pattern", default="*_DG.rdf")
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--out-name", required=True)
    ap.add_argument("--dual-thresh", type=float, default=0.25)
    ap.add_argument("--func-all", action="store_true")
    ap.add_argument("--emit-debug", action="store_true")
    args = ap.parse_args()

    in_dir = Path(args.input_dir)
    files = sorted([str(p) for p in in_dir.glob(args.pattern)])
    print("[LOAD]")
    for f in files: print("      ", Path(f).name)
    models = [build_model(f) for f in files]

    out_dir = ensure_dir(os.path.join(args.out_root, args.out_name))

    # S1
    df_types, df_funcs, df_func_wide, node2type = stage_S1(models, func_all=args.func_all, emit_debug=args.emit_debug, out_dir=out_dir)

    # S1: Data availability özeti
    rows_av = []
    for m in models:
        def count_type(t):
            return sum(1 for n in node2type[m.name].keys() if t in node2type[m.name][n])
        rows_av.append({
            "model":m.name,
            "n_Wall":count_type("Wall"),
            "n_Slab":count_type("Slab"),
            "n_Beam":count_type("Beam"),
            "n_Column":count_type("Column"),
            "n_Brace":count_type("Brace"),
            "n_Core":count_type("Core"),
            "n_Foundation":count_type("Foundation"),
            "n_adjacentElement":len(m.adj_edges),
            "n_adjacentZone":len(m.zone_edges),
            "n_intersectingElement":len(m.inters_edges),
            "n_hasContinuantPart":len(m.part_edges),
            "n_func_LoadBearing":int((df_funcs[(df_funcs["model"]==m.name)&(df_funcs["function"]=="LoadBearing")]["count"]).sum()) if not df_funcs.empty else 0,
            "n_func_Shear":int((df_funcs[(df_funcs["model"]==m.name)&(df_funcs["function"]=="Shear")]["count"]).sum()) if not df_funcs.empty else 0,
            "n_func_Moment":int((df_funcs[(df_funcs["model"]==m.name)&(df_funcs["function"]=="Moment")]["count"]).sum()) if not df_funcs.empty else 0,
        })
    pd.DataFrame(rows_av).sort_values("model").to_csv(os.path.join(out_dir, "struct_data_availability.csv"), index=False)

    # S2
    df_counts, df_shares, df_dens = stage_S2(models, node2type, out_dir=out_dir, emit_debug=args.emit_debug)

    # S3
    df_scores, df_comp = stage_S3(models, node2type, df_func_wide, df_dens, dual_thresh=args.dual_thresh, out_dir=out_dir)

    # S4
    df_mat, df_pair = stage_S4(models, df_shares, df_scores, out_dir=out_dir, w_motif=0.5, w_sys=0.5)

    print("\n[OK] Saved outputs under:", out_dir)
    for fn in [
        "struct_types_histogram.csv",
        "struct_functions_histogram.csv",
        "struct_functions_shares_wide.csv",
        "type_mapping_hits.csv",
        "type_mapping_unknown.csv",
        "struct_motif_counts.csv",
        "struct_motif_shares.csv",
        "struct_motif_densities_per100.csv",
        "struct_score_components.csv",
        "struct_system_scores.csv",
        "struct_data_availability.csv",
        "struct_similarity_matrix.csv",
        "pairwise_structural_summary.csv",
        "weights_used.json",
        "motif_proxy_edges.csv" if args.emit_debug else None
    ]:
        if fn:
            print(" -", fn)

if __name__ == "__main__":
    main()
