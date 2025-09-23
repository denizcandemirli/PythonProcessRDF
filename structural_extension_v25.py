# structural_extension_v25.py
# S1–S4 Structural Extension (v2.5 + M5 hotfix call in main):
# - Expanded proxies for Wall–Slab: strict intersect, adjacency, and zone-based adjacency
# - Aggressive IFC mapping for types (IfcSlab/IfcFloor/IfcWall/IfcStructuralCurveMember…)
# - Stronger brace/frame signals via function/type synonyms
# - Eurocode/ASCE-inspired dual calibration: dual=0 if frame_share < 0.25
#
# Outputs under out_root/out_name:
#   struct_types_histogram.csv
#   struct_functions_histogram.csv
#   struct_motif_counts.csv
#   struct_motif_shares.csv
#   struct_system_scores.csv   (includes dual_raw, dual)
#   struct_similarity_matrix.csv
#   pairwise_structural_summary.csv

import argparse, os, glob, re, collections
from typing import Dict, Set, Tuple, List
import pandas as pd
import numpy as np
from rdflib import Graph, RDF

# ---------- helpers ----------
def ensure_dir(p: str): os.makedirs(p, exist_ok=True)
def localname(uri) -> str:
    s = str(uri)
    if '#' in s: return s.split('#')[-1]
    return s.rstrip('/').split('/')[-1]

# ---------- expanded regex / IFC mapping ----------
# element classes (case-insensitive; search anywhere in localName)
RE_IFC_BEAM   = re.compile(r'(IfcBeam|(^|[^A-Za-z])Beam([^A-Za-z]|$)|Girder|Primary[Mm]ember|BeamSegment)', re.I)
RE_IFC_COLUMN = re.compile(r'(IfcColumn|(^|[^A-Za-z])Column([^A-Za-z]|$)|Pillar)', re.I)

# Slab family: Slab / Deck / Floor / Plate / IfcFloor / IfcSlab
RE_IFC_SLAB   = re.compile(r'(IfcSlab|IfcFloor|(^|[^A-Za-z])(Slab|Deck|Floor|Plate)([^A-Za-z]|$))', re.I)

# Wall family: Wall / ShearWall / CoreWall / RetainingWall / IfcWall
RE_IFC_WALL   = re.compile(r'(IfcWall|(^|[^A-Za-z])(Wall|Shear[- ]?Wall|CoreWall|RetainingWall)([^A-Za-z]|$))', re.I)

# Brace family: Brace / Bracing / X/K/Chevron brace / IfcStructuralCurveMember (used as brace in many exports)
RE_IFC_BRACE  = re.compile(r'(IfcStructuralCurveMember|(^|[^A-Za-z])(Brace|Bracing|X[- ]?brace|K[- ]?brace|Chevron)([^A-Za-z]|$))', re.I)

# Core family
RE_IFC_CORE   = re.compile(r'(Ifc)?(Core|CoreWall|ShearCore|Shaft|StairCore|LiftCore)', re.I)

# Foundation family
RE_IFC_FOUND  = re.compile(r'(Ifc)?(Foundation|Footing|Pad|Strip|Pile|Raft|Mat)', re.I)

# Zone/Space family (for adjacentZone proxy)
RE_ZONE       = re.compile(r'(Zone|Space|Room|IfcSpace)', re.I)

def is_struct_element(type_str: str) -> bool:
    t = type_str
    return any([
        RE_IFC_BEAM.search(t), RE_IFC_COLUMN.search(t), RE_IFC_SLAB.search(t),
        RE_IFC_WALL.search(t), RE_IFC_BRACE.search(t), RE_IFC_CORE.search(t),
        RE_IFC_FOUND.search(t)
    ])

def coarse_type(type_str: str) -> str:
    t = type_str
    if RE_IFC_BEAM.search(t):   return 'Beam'
    if RE_IFC_COLUMN.search(t): return 'Column'
    if RE_IFC_SLAB.search(t):   return 'Slab'
    if RE_IFC_WALL.search(t):   return 'Wall'
    if RE_IFC_BRACE.search(t):  return 'Brace'
    if RE_IFC_CORE.search(t):   return 'Core'
    if RE_IFC_FOUND.search(t):  return 'Foundation'
    return 'Other'

def is_zone_type(type_str: str) -> bool:
    return bool(RE_ZONE.search(type_str))

# function buckets (expanded)
RE_FUNC_LOAD   = re.compile(r'(load[_\- ]?bearing|[^A-Za-z]bearing([^A-Za-z]|$))', re.I)
RE_FUNC_SHEAR  = re.compile(r'(shear|shear[_\- ]?wall)', re.I)
RE_FUNC_MOMENT = re.compile(r'(moment|bending|lateral|rigid\s*frame|moment\s*frame|frame\s*action)', re.I)
RE_FUNC_DIAPH  = re.compile(r'(diaphragm)', re.I)
RE_FUNC_STIFF  = re.compile(r'(stiff|stiffener)', re.I)
RE_FUNC_BRACE  = re.compile(r'(brace|bracing|tie|strut)', re.I)

def func_bucket(f: str) -> str:
    if RE_FUNC_LOAD.search(f):   return 'LoadBearing'
    if RE_FUNC_SHEAR.search(f):  return 'Shear'
    if RE_FUNC_MOMENT.search(f): return 'Moment'
    if RE_FUNC_DIAPH.search(f):  return 'Diaphragm'
    if RE_FUNC_STIFF.search(f):  return 'Stiffener'
    if RE_FUNC_BRACE.search(f):  return 'Bracing'
    return 'Other'

# predicate localnames (namespace-agnostic)
PRED_ADJ_EL   = 'adjacentElement'
PRED_INT_EL   = 'intersectingElement'
PRED_ADJ_ZONE = 'adjacentZone'
PRED_PART     = 'BFO_0000178'   # hasContinuantPart
PRED_HASFUNC  = 'hasFunction'

# ---------- model container ----------
class ModelData:
    def __init__(self, name: str):
        self.name = name
        self.types: Dict[str, Set[str]] = collections.defaultdict(set)       # node -> set(local type names)
        self.functions: Dict[str, List[str]] = collections.defaultdict(list) # element -> function type strings
        self.adj_edges: Set[Tuple[str,str]] = set()     # undirected E–E
        self.int_edges: Set[Tuple[str,str]] = set()     # undirected E–E
        self.part_edges: Set[Tuple[str,str]] = set()    # directed E->P
        self.adj_zone:  Set[Tuple[str,str]] = set()     # directed E->Z

    def add_type(self, s, t): self.types[str(s)].add(localname(t))
    def add_func(self, e, ftype): self.functions[str(e)].append(localname(ftype))

    def add_undirected(self, store, a, b):
        a, b = str(a), str(b)
        if a == b: return
        u, v = (a,b) if a < b else (b,a)
        store.add((u,v))

    def add_directed(self, store, a, b): store.add((str(a), str(b)))

def load_model(path: str) -> ModelData:
    g = Graph(); g.parse(path, format='xml')
    md = ModelData(os.path.basename(path))
    for s,_,o in g.triples((None, RDF.type, None)):
        md.add_type(s,o)
    for s,p,o in g:
        lp = localname(p)
        if lp == PRED_HASFUNC:
            for _,_,ft in g.triples((o, RDF.type, None)):
                md.add_func(s, ft)
        elif lp == PRED_ADJ_EL:
            md.add_undirected(md.adj_edges, s, o)
        elif lp == PRED_INT_EL:
            md.add_undirected(md.int_edges, s, o)
        elif lp == PRED_PART:
            md.add_directed(md.part_edges, s, o)
        elif lp == PRED_ADJ_ZONE:
            md.add_directed(md.adj_zone, s, o)
    return md

# ---------- utilities ----------
def node_class(md: ModelData, n: str) -> str:
    ts = md.types.get(n, set())
    for t in ts:
        c = coarse_type(t)
        if c != 'Other': return c
    return 'Other'

def is_zone_node(md: ModelData, n: str) -> bool:
    for t in md.types.get(n, set()):
        if is_zone_type(t): return True
    return False

def minmax01(x: float, lo: float, hi: float) -> float:
    if hi <= lo: return 0.0
    v = (x - lo) / (hi - lo)
    return float(max(0.0, min(1.0, v)))

# ---------- S1 ----------
def s1_inventories(models: List[ModelData], out_dir: str):
    rows_types, rows_funcs = [], []

    for m in models:
        elem_nodes = [n for n, ts in m.types.items() if any(is_struct_element(t) for t in ts)]
        # type histogram
        tcounts = collections.Counter()
        for n in elem_nodes:
            cset = {coarse_type(t) for t in m.types[n]}
            cset.discard('Other')
            for ct in cset: tcounts[ct] += 1
        total_t = sum(tcounts.values()) or 1
        for ct, c in sorted(tcounts.items()):
            rows_types.append({"model": m.name, "coarse_type": ct, "count": c, "share": c/total_t})
        # function histogram (on structural elements only)
        fcounts = collections.Counter(); total_f = 0
        for e, flist in m.functions.items():
            if not any(is_struct_element(t) for t in m.types.get(e, [])): continue
            for f in flist:
                fb = func_bucket(f)
                fcounts[fb] += 1; total_f += 1
        total_f = total_f or 1
        for fb, c in sorted(fcounts.items()):
            rows_funcs.append({"model": m.name, "function": fb, "count": c, "share": c/total_f})

    df_types = (pd.DataFrame(rows_types)
                  if rows_types else pd.DataFrame(columns=["model","coarse_type","count","share"]))
    df_funcs = (pd.DataFrame(rows_funcs)
                  if rows_funcs else pd.DataFrame(columns=["model","function","count","share"]))
    if not df_types.empty: df_types = df_types.sort_values(["model","coarse_type"])
    if not df_funcs.empty: df_funcs = df_funcs.sort_values(["model","function"])
    df_types.to_csv(os.path.join(out_dir, "struct_types_histogram.csv"), index=False)
    df_funcs.to_csv(os.path.join(out_dir, "struct_functions_histogram.csv"), index=False)
    return df_types, df_funcs

# ---------- S2 (motifs) ----------
def s2_motifs(models: List[ModelData], out_dir: str):
    motif_cols = ["M2_frameNode","M3_wallSlab","M3_wallSlabAdj","M3_wallSlabZone","M4_core","M2_braceNode","M5_structRole"]
    rows = []
    for m in models:
        adj, inter = m.adj_edges, m.int_edges

        # frame node: Beam–Column adjacency
        frame_pairs = {(a,b) for (a,b) in adj if {'Beam','Column'} == {node_class(m,a), node_class(m,b)}}

        # wall–slab strict intersection
        wallslab_int = {(a,b) for (a,b) in inter if {'Wall','Slab'} == {node_class(m,a), node_class(m,b)}}

        # wall–slab adjacency (proxy)
        wallslab_adj = {(a,b) for (a,b) in adj if {'Wall','Slab'} == {node_class(m,a), node_class(m,b)}}

        # wall–slab via zone: Wall--Z and Slab--Z (same zone)
        zone_map = collections.defaultdict(lambda: {"Wall": set(), "Slab": set()})
        for e, z in m.adj_zone:
            if not is_zone_node(m, z): continue
            ec = node_class(m, e)
            if ec in ("Wall","Slab"):
                zone_map[z][ec].add(e)
        wallslab_zone_pairs = set()
        for z, d in zone_map.items():
            for w in d["Wall"]:
                for s in d["Slab"]:
                    u,v = (w,s) if w < s else (s,w)
                    wallslab_zone_pairs.add((u,v))

        # brace node: Brace–(Beam|Column) adjacency
        brace_pairs = {(a,b) for (a,b) in adj
                       if (node_class(m,a) == 'Brace' and node_class(m,b) in {'Beam','Column'}) or
                          (node_class(m,b) == 'Brace' and node_class(m,a) in {'Beam','Column'})}

        # core: core element with parts and at least one Slab neighbor
        parts_by_e = collections.defaultdict(set)
        for e,p in m.part_edges: parts_by_e[e].add(p)
        nbrs = collections.defaultdict(set)
        for a,b in adj:
            nbrs[a].add(b); nbrs[b].add(a)
        core_nodes = []
        for n, ts in m.types.items():
            if node_class(m, n) == 'Core' and len(parts_by_e.get(n,()))>0:
                if any(node_class(m, x) == 'Slab' for x in nbrs.get(n,())):
                    core_nodes.append(n)

        # structRole motif: E_* -> F_(load|shear|moment)
        role_edges = 0
        for e, flist in m.functions.items():
            if not any(is_struct_element(t) for t in m.types.get(e, [])): continue
            for f in flist:
                if func_bucket(f) in ('LoadBearing','Shear','Moment'):
                    role_edges += 1

        rows.append({
            "model": m.name,
            "M2_frameNode": len(frame_pairs),
            "M3_wallSlab": len(wallslab_int),
            "M3_wallSlabAdj": len(wallslab_adj),
            "M3_wallSlabZone": len(wallslab_zone_pairs),
            "M4_core": len(core_nodes),
            "M2_braceNode": len(brace_pairs),
            "M5_structRole": role_edges
        })

    df_counts = pd.DataFrame(rows).set_index("model")
    # ensure all motif columns exist (even if all zeros)
    df_counts = df_counts.reindex(columns=motif_cols, fill_value=0).sort_index()

    shares = df_counts.div(df_counts.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)

    df_counts.to_csv(os.path.join(out_dir, "struct_motif_counts.csv"))
    shares.to_csv(os.path.join(out_dir, "struct_motif_shares.csv"))
    return df_counts, shares

def inject_M5_from_s1(df_counts: pd.DataFrame, df_funcs: pd.DataFrame) -> pd.DataFrame:
    """
    S1'de güvenle saydığımız fonksiyon histogramından (LoadBearing/Shear/Moment)
    M5_structRole sayısını türetip motif tablosuna yazar. Böylece S2'de gating/reindex
    gibi nedenlerle 0'a düşen M5 sinyali garanti altına alınır.
    """
    if df_counts.empty:
        return df_counts

    # S1 fonksiyon histogramında bu üç fonksiyonu topla
    m5_map = {}
    for m in df_counts.index:
        sub = df_funcs.query("model == @m and function in ['LoadBearing','Shear','Moment']")
        m5_map[m] = int(sub['count'].sum()) if not sub.empty else 0

    # Sütun yoksa oluştur
    if 'M5_structRole' not in df_counts.columns:
        df_counts['M5_structRole'] = 0

    # Değerleri yaz
    for m, v in m5_map.items():
        if m in df_counts.index:
            df_counts.at[m, 'M5_structRole'] = int(v)
    return df_counts

# ---------- S3 (system scores with calibration) ----------
def s3_system_scores(models: List[ModelData],
                     motif_counts: pd.DataFrame,
                     types_hist: pd.DataFrame,
                     func_hist: pd.DataFrame,
                     out_dir: str,
                     dual_frame_share_threshold: float = 0.25):

    # structural element counts per model
    nE = {}
    for md in models:
        nE[md.name] = sum(1 for n,ts in md.types.items() if any(is_struct_element(t) for t in ts))

    # function share lookup
    def share_func(model: str, fname: str) -> float:
        df = func_hist.query("model == @model and function == @fname")
        return float(df["share"].sum()) if not df.empty else 0.0

    # densities (per 100 structural elements), float-safe
    dens = motif_counts.astype(float).copy()
    for m in dens.index:
        denom = max(1, nE.get(m,1))
        dens.loc[m,:] = (dens.loc[m,:] / denom) * 100.0

    # percentile min–max per motif column
    col_ranges = {}
    for col in dens.columns:
        arr = dens[col].astype(float).values
        lo, hi = np.nanpercentile(arr, 10), np.nanpercentile(arr, 90)
        col_ranges[col] = (float(lo), float(hi))

    def norm_col(m, col):
        lo, hi = col_ranges[col]
        return minmax01(float(dens.loc[m,col]), lo, hi)

    rows = []
    for m in dens.index:
        # frame: frame node + moment-related function
        frame = 0.7*norm_col(m, "M2_frameNode") + 0.3*share_func(m, "Moment")

        # wall: strict+adj+zone, then add function support
        wall_motif = 0.6*norm_col(m, "M3_wallSlab") + 0.3*norm_col(m, "M3_wallSlabAdj") + 0.1*norm_col(m, "M3_wallSlabZone")
        wall = 0.6*wall_motif + 0.4*((share_func(m,"LoadBearing")+share_func(m,"Shear"))/2.0)

        # braced
        braced = 0.8*norm_col(m, "M2_braceNode") + 0.2*share_func(m, "Bracing")

        # dual raw and calibrated
        denom = max(1e-6, (0.5*frame + 0.5*wall))
        dual_raw = (frame*wall) / denom
        frame_share = frame / max(1e-6, (frame + wall))
        dual = 0.0 if frame_share < dual_frame_share_threshold else dual_raw

        rows.append({"model": m, "frame": frame, "wall": wall, "dual_raw": dual_raw, "dual": dual, "braced": braced, "frame_share": frame_share})

    df_sys = pd.DataFrame(rows).set_index("model").sort_index()
    df_sys.to_csv(os.path.join(out_dir, "struct_system_scores.csv"))
    return df_sys

# ---------- S4 (structural similarity) ----------
def cosine_sim_matrix(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    norms = np.linalg.norm(X, axis=1, keepdims=True); norms[norms==0] = 1.0
    Y = X / norms
    S = Y @ Y.T
    return np.clip(S, 0.0, 1.0)

def s4_similarity(motif_shares: pd.DataFrame, sys_scores: pd.DataFrame,
                  out_dir: str, beta1: float = 0.7, beta2: float = 0.3):
    labels = sorted(set(motif_shares.index) & set(sys_scores.index))
    M = motif_shares.loc[labels, :]                  # model × motifs
    # Use calibrated system vector (frame, wall, dual, braced)
    sys_cols = [c for c in ["frame","wall","dual","braced"] if c in sys_scores.columns]
    Svec = sys_scores.loc[labels, sys_cols]

    S_motif = cosine_sim_matrix(M.values)
    S_sys   = cosine_sim_matrix(Svec.values)
    S_comb  = beta1*S_motif + beta2*S_sys

    df = pd.DataFrame(S_comb, index=labels, columns=labels)
    df.to_csv(os.path.join(out_dir, "struct_similarity_matrix.csv"))

    rows = []
    for i in range(len(labels)):
        for j in range(i+1, len(labels)):
            rows.append({
                "model_A": labels[i], "model_B": labels[j],
                "S_struct_total": float(df.iloc[i,j]),
                "S_motif": float(S_motif[i,j]), "S_system": float(S_sys[i,j])
            })
    pd.DataFrame(rows).sort_values("S_struct_total", ascending=False).to_csv(
        os.path.join(out_dir, "pairwise_structural_summary.csv"), index=False
    )

# ---------- orchestrator ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--pattern", default="*_DG.rdf")
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--out-name", default="07 - Structural_Extension_v25")
    ap.add_argument("--beta1", type=float, default=0.7)
    ap.add_argument("--beta2", type=float, default=0.3)
    ap.add_argument("--dual-thresh", type=float, default=0.25)  # Eurocode/ASCE-inspired
    args = ap.parse_args()

    out_dir = os.path.join(args.out_root, args.out_name); ensure_dir(out_dir)

    paths = sorted(glob.glob(os.path.join(args.input_dir, args.pattern)))
    if not paths: raise SystemExit(f"No RDF files matched: {args.input_dir}/{args.pattern}")

    print("[LOAD]")
    for p in paths: print("      ", os.path.basename(p))
    models = [load_model(p) for p in paths]

    # S1
    df_types, df_funcs = s1_inventories(models, out_dir)
    # S2
    motif_counts, motif_shares = s2_motifs(models, out_dir)

    # ---- M5 HOTFIX CALL (new) ----
    motif_counts = inject_M5_from_s1(motif_counts, df_funcs)
    motif_shares = motif_counts.div(motif_counts.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    motif_counts.to_csv(os.path.join(out_dir, "struct_motif_counts.csv"))
    motif_shares.to_csv(os.path.join(out_dir, "struct_motif_shares.csv"))
    # -------------------------------

    # S3
    sys_scores = s3_system_scores(models=models,
                                  motif_counts=motif_counts,
                                  types_hist=df_types,
                                  func_hist=df_funcs,
                                  out_dir=out_dir,
                                  dual_frame_share_threshold=args.dual_thresh)
    # S4
    s4_similarity(motif_shares, sys_scores, out_dir, beta1=args.beta1, beta2=args.beta2)

    print(f"\n[OK] Saved outputs under: {out_dir}")
    print(" - struct_types_histogram.csv")
    print(" - struct_functions_histogram.csv")
    print(" - struct_motif_counts.csv")
    print(" - struct_motif_shares.csv")
    print(" - struct_system_scores.csv")
    print(" - struct_similarity_matrix.csv")
    print(" - pairwise_structural_summary.csv")

if __name__ == "__main__":
    main()
