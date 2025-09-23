# structural_extension_v24.py
# S1–S4: Structural Extension (v2.4) — proxy motif + expanded regex + IFC names
# - Adds M3_wallSlabAdj (adjacentElement(Wall,Slab)) as proxy when intersectingElement is absent
# - Expands regex to catch Beam/Brace/Core/Foundation variants and Ifc* class names
# - Extends function buckets with synonyms (Moment/Bending/Lateral; Bracing/Tie/Strut; Stiffener; Diaphragm)
# - Uses float dtypes for densities to avoid pandas future warnings
#
# Outputs under out_root/out_name:
#   struct_types_histogram.csv
#   struct_functions_histogram.csv
#   struct_motif_counts.csv
#   struct_motif_shares.csv
#   struct_system_scores.csv
#   struct_similarity_matrix.csv
#   pairwise_structural_summary.csv

import argparse, os, glob, re, collections
from typing import Dict, Set, Tuple, List
import pandas as pd
import numpy as np
from rdflib import Graph, RDF

# -----------------------
# IO helpers
# -----------------------
def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def localname(uri) -> str:
    s = str(uri)
    if '#' in s:
        return s.split('#')[-1]
    return s.rstrip('/').split('/')[-1]

# -----------------------
# Regex dictionaries (expanded + IFC)
# -----------------------
# Elements (case-insensitive)
RE_IFC_BEAM   = re.compile(r'^(Ifc)?Beam|Girder|Primary(M|m)ember|Beam(Segment)?$', re.I)
RE_IFC_COLUMN = re.compile(r'^(Ifc)?Column|Pillar$', re.I)
RE_IFC_SLAB   = re.compile(r'^(Ifc)?Slab|Deck|Floor$', re.I)
RE_IFC_WALL   = re.compile(r'^(Ifc)?Wall|Shear[- ]?Wall$', re.I)
RE_IFC_BRACE  = re.compile(r'^(Ifc)?Brace|Bracing|X[- ]?brace|K[- ]?brace|Chevron$', re.I)
RE_IFC_CORE   = re.compile(r'^(Ifc)?(Core|CoreWall|ShearCore|Shaft|StairCore|LiftCore)$', re.I)
RE_IFC_FOUND  = re.compile(r'^(Ifc)?(Foundation|Footing|Pad|Strip|Pile|Raft|Mat)$', re.I)

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

# Function buckets (expanded)
RE_FUNC_LOAD   = re.compile(r'(load[_\- ]?bearing|bearing)', re.I)
RE_FUNC_SHEAR  = re.compile(r'(shear|shear[_\- ]?wall)', re.I)
RE_FUNC_MOMENT = re.compile(r'(moment|bending|lateral)', re.I)
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

# Predicate localnames
PRED_ADJ_EL  = 'adjacentElement'
PRED_INT_EL  = 'intersectingElement'
PRED_PART    = 'BFO_0000178'      # hasContinuantPart
PRED_HASFUNC = 'hasFunction'

# -----------------------
# Model container
# -----------------------
class ModelData:
    def __init__(self, name: str):
        self.name = name
        self.types: Dict[str, Set[str]] = collections.defaultdict(set)    # node -> set(local type names)
        self.functions: Dict[str, List[str]] = collections.defaultdict(list)  # element -> function type strings
        self.adj_edges: Set[Tuple[str,str]] = set()   # undirected E–E adjacencies
        self.int_edges: Set[Tuple[str,str]] = set()   # undirected E–E intersections
        self.part_edges: Set[Tuple[str,str]] = set()  # directed E->P

    def add_type(self, s, t):
        self.types[str(s)].add(localname(t))

    def add_func(self, e, ftype):
        self.functions[str(e)].append(localname(ftype))

    def add_undirected(self, store, a, b):
        a, b = str(a), str(b)
        if a == b: return
        u, v = (a,b) if a < b else (b,a)
        store.add((u,v))

    def add_directed(self, store, a, b):
        store.add((str(a), str(b)))

def load_model(path: str) -> ModelData:
    g = Graph()
    g.parse(path, format='xml')
    md = ModelData(os.path.basename(path))

    # collect rdf:type
    for s, _, o in g.triples((None, RDF.type, None)):
        md.add_type(s, o)

    # scan graph for key predicates (namespace-agnostic via localname)
    for s, p, o in g:
        lp = localname(p)
        if lp == PRED_HASFUNC:
            # attach function types (types of the function node) onto the element s
            for _,_,ft in g.triples((o, RDF.type, None)):
                md.add_func(s, ft)
        elif lp == PRED_ADJ_EL:
            md.add_undirected(md.adj_edges, s, o)
        elif lp == PRED_INT_EL:
            md.add_undirected(md.int_edges, s, o)
        elif lp == PRED_PART:
            md.add_directed(md.part_edges, s, o)

    return md

# -----------------------
# Utility
# -----------------------
def node_class(md: ModelData, n: str) -> str:
    ts = md.types.get(n, set())
    # choose first non-Other coarse class
    for t in ts:
        c = coarse_type(t)
        if c != 'Other': return c
    return 'Other'

def minmax01(x: float, lo: float, hi: float) -> float:
    if hi <= lo: return 0.0
    v = (x - lo) / (hi - lo)
    return float(max(0.0, min(1.0, v)))

# -----------------------
# S1 — Type/Function inventories
# -----------------------
def s1_inventories(models: List[ModelData], out_dir: str):
    rows_types, rows_funcs = [], []

    for m in models:
        # structural element nodes
        elem_nodes = [n for n, ts in m.types.items() if any(is_struct_element(t) for t in ts)]
        tcounts = collections.Counter()
        for n in elem_nodes:
            cset = {coarse_type(t) for t in m.types[n]}
            cset.discard('Other')
            for ct in cset:
                tcounts[ct] += 1
        total_t = sum(tcounts.values()) or 1
        for ct, c in sorted(tcounts.items()):
            rows_types.append({"model": m.name, "coarse_type": ct, "count": c, "share": c/total_t})

        # function histogram (only on structural elements)
        fcounts = collections.Counter()
        total_f = 0
        for e, flist in m.functions.items():
            if not any(is_struct_element(t) for t in m.types.get(e, [])):
                continue
            for f in flist:
                fb = func_bucket(f)
                fcounts[fb] += 1
                total_f += 1
        total_f = total_f or 1
        for fb, c in sorted(fcounts.items()):
            rows_funcs.append({"model": m.name, "function": fb, "count": c, "share": c/total_f})

    # ---- SAFE BUILD + SORT ----
    if rows_types:
        df_types = pd.DataFrame(rows_types)
        df_types = df_types.sort_values(["model","coarse_type"])
    else:
        df_types = pd.DataFrame(columns=["model","coarse_type","count","share"])

    if rows_funcs:
        df_funcs = pd.DataFrame(rows_funcs)
        df_funcs = df_funcs.sort_values(["model","function"])
    else:
        # create an empty, but well-formed table
        df_funcs = pd.DataFrame(columns=["model","function","count","share"])

    df_types.to_csv(os.path.join(out_dir, "struct_types_histogram.csv"), index=False)
    df_funcs.to_csv(os.path.join(out_dir, "struct_functions_histogram.csv"), index=False)
    return df_types, df_funcs

# -----------------------
# S2 — Structural micro-motifs (with proxy)
# -----------------------
def s2_motifs(models: List[ModelData], out_dir: str):
    motif_names = ["M2_frameNode", "M3_wallSlab", "M3_wallSlabAdj", "M4_core", "M2_braceNode", "M5_structRole"]
    rows = []

    for m in models:
        adj, inter = m.adj_edges, m.int_edges

        # frame node: Beam–Column adjacency
        frame_pairs = {(a,b) for (a,b) in adj
                       if {'Beam','Column'} == {node_class(m,a), node_class(m,b)}}

        # wall–slab intersections (strict)
        wallslab_int = {(a,b) for (a,b) in inter
                        if {'Wall','Slab'} == {node_class(m,a), node_class(m,b)}}

        # wall–slab adjacency (proxy)
        wallslab_adj = {(a,b) for (a,b) in adj
                        if {'Wall','Slab'} == {node_class(m,a), node_class(m,b)}}

        # brace node: Brace–(Beam|Column) adjacency
        brace_pairs = {(a,b) for (a,b) in adj
                       if (node_class(m,a) == 'Brace' and node_class(m,b) in {'Beam','Column'}) or
                          (node_class(m,b) == 'Brace' and node_class(m,a) in {'Beam','Column'})}

        # core motif: Core element with parts and at least one Slab neighbor
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
            if not any(is_struct_element(t) for t in m.types.get(e, [])):
                continue
            for f in flist:
                if func_bucket(f) in ('LoadBearing','Shear','Moment'):
                    role_edges += 1

        rows.append({
            "model": m.name,
            "M2_frameNode": len(frame_pairs),
            "M3_wallSlab": len(wallslab_int),
            "M3_wallSlabAdj": len(wallslab_adj),
            "M4_core": len(core_nodes),
            "M2_braceNode": len(brace_pairs),
            "M5_structRole": role_edges
        })

    df_counts = pd.DataFrame(rows).set_index("model").sort_index()
    # row-normalized shares (if row sum==0, stays 0)
    shares = df_counts.div(df_counts.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)

    df_counts.to_csv(os.path.join(out_dir, "struct_motif_counts.csv"))
    shares.to_csv(os.path.join(out_dir, "struct_motif_shares.csv"))
    return df_counts, shares

# -----------------------
# S3 — Macro system scores
# -----------------------
def s3_system_scores(models: List[ModelData],
                     motif_counts: pd.DataFrame,
                     types_hist: pd.DataFrame,
                     func_hist: pd.DataFrame,
                     out_dir: str):
    # structural element counts per model
    nE = {}
    for md in models:
        nE[md.name] = sum(1 for n,ts in md.types.items() if any(is_struct_element(t) for t in ts))

    # helper lookups from hist
    def share_func(model: str, fname: str) -> float:
        df = func_hist.query("model == @model and function == @fname")
        return float(df["share"].sum()) if not df.empty else 0.0

    # densities per 100 structural elements
    dens = motif_counts.astype(float).copy()
    for m in dens.index:
        denom = max(1, nE.get(m,1))
        dens.loc[m,:] = (dens.loc[m,:] / denom) * 100.0

    # percentile ranges per column
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
        # frame: frameNode + Moment function
        frame = 0.7*norm_col(m, "M2_frameNode") + 0.3*share_func(m, "Moment")

        # wall: strict intersection AND/OR adjacency proxy
        wall_raw = 0.6*norm_col(m, "M3_wallSlab") + 0.4*norm_col(m, "M3_wallSlabAdj")
        wall = 0.6*wall_raw + 0.4*((share_func(m,"LoadBearing")+share_func(m,"Shear"))/2.0)

        # braced: brace-node + bracing func
        braced = 0.8*norm_col(m, "M2_braceNode") + 0.2*share_func(m, "Bracing")

        # dual: high frame & high wall together
        denom = max(1e-6, (0.5*frame + 0.5*wall))
        dual = (frame*wall) / denom

        rows.append({"model": m, "frame": frame, "wall": wall, "dual": dual, "braced": braced})

    df_sys = pd.DataFrame(rows).set_index("model").sort_index()
    df_sys.to_csv(os.path.join(out_dir, "struct_system_scores.csv"))
    return df_sys

# -----------------------
# S4 — Structural similarity (motif ⊕ system)
# -----------------------
def cosine_sim_matrix(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms==0] = 1.0
    Y = X / norms
    S = Y @ Y.T
    return np.clip(S, 0.0, 1.0)

def s4_similarity(motif_shares: pd.DataFrame, sys_scores: pd.DataFrame,
                  out_dir: str, beta1: float = 0.7, beta2: float = 0.3):
    labels = sorted(set(motif_shares.index) & set(sys_scores.index))
    M = motif_shares.loc[labels, :]
    S = sys_scores.loc[labels, :]

    S_motif = cosine_sim_matrix(M.values)
    S_sys   = cosine_sim_matrix(S.values)
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

# -----------------------
# Orchestrator
# -----------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--pattern", default="*_DG.rdf")
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--out-name", default="07 - Structural_Extension")
    ap.add_argument("--beta1", type=float, default=0.7)
    ap.add_argument("--beta2", type=float, default=0.3)
    args = ap.parse_args()

    out_dir = os.path.join(args.out_root, args.out_name)
    ensure_dir(out_dir)

    paths = sorted(glob.glob(os.path.join(args.input_dir, args.pattern)))
    if not paths:
        raise SystemExit(f"No RDF files matched: {args.input_dir}/{args.pattern}")

    print("[LOAD]")
    for p in paths:
        print("      ", os.path.basename(p))
    models = [load_model(p) for p in paths]

    # S1
    df_types, df_funcs = s1_inventories(models, out_dir)
    # S2
    motif_counts, motif_shares = s2_motifs(models, out_dir)
    # S3
    sys_scores = s3_system_scores(models=models,
                                  motif_counts=motif_counts,
                                  types_hist=df_types,
                                  func_hist=df_funcs,
                                  out_dir=out_dir)
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
