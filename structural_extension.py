# structural_extension.py
# S1–S4: Structural Extension pipeline
# Deniz Can Demirli — BIM RDF design graphs
#
# What it does
#  - Parses RDF/XML with rdflib
#  - Extracts structural element/role inventories (S1)
#  - Detects structural micro-motifs with lightweight type filters (S2)
#  - Scores macro systems (frame / wall / dual / braced) rule-based (S3)
#  - Builds structural similarity:  S_struct_total = 0.7·cos(motif_shares) + 0.3·cos(system_scores)  (S4)
#
# Assumptions
#  - Predicates may come with different namespaces: we match by localname
#    ('adjacentElement', 'intersectingElement', 'BFO_0000178', 'hasFunction').
#  - Element types are detected by localname regex (Beam|Column|Slab|Wall|Brace|Core|Foundation|Footing|Pile|Girder).
#  - Functions detected by regex (load|bearing|shear|moment|diaphragm|stiff|brace).
#  - Undirected reading for adjacency/intersection.
#
# Outputs (under out_root/out_name):
#   struct_types_histogram.csv
#   struct_functions_histogram.csv
#   struct_motif_counts.csv
#   struct_motif_shares.csv
#   struct_system_scores.csv
#   struct_similarity_matrix.csv     (final structural similarity)
#   pairwise_structural_summary.csv  (top pairs & drivers)
#
# Usage example:
#   python structural_extension.py ^
#       --input-dir "." ^
#       --pattern "*_DG.rdf" ^
#       --out-root ".\repro_pack\output" ^
#       --out-name "07 - Structural_Extension"

import argparse, os, glob, math, re, json, itertools, collections
from typing import Dict, Set, Tuple, List

import pandas as pd
import numpy as np
from rdflib import Graph, RDF

# -----------------------
# Helpers: path & IO
# -----------------------
def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def save_csv(df: pd.DataFrame, path: str):
    df.to_csv(path, index=True if df.index.name is not None else False)

def localname(uri: str) -> str:
    s = str(uri)
    if '#' in s:
        return s.split('#')[-1]
    return s.rstrip('/').split('/')[-1]

# -----------------------
# Structural dictionaries
# -----------------------
# Element type regex groups (case-insensitive)
RE_BEAM   = re.compile(r'(beam|girder)', re.I)
RE_COLUMN = re.compile(r'(column|pillar)', re.I)
RE_SLAB   = re.compile(r'(slab|deck|floor)', re.I)
RE_WALL   = re.compile(r'(wall|shearwall)', re.I)
RE_BRACE  = re.compile(r'(brace|bracing)', re.I)
RE_CORE   = re.compile(r'(core|corewall|shearcore)', re.I)
RE_FOUND  = re.compile(r'(foundation|footing|pile|raft)', re.I)

# Any structural element (union)
def is_struct_element(t: str) -> bool:
    return any([
        RE_BEAM.search(t), RE_COLUMN.search(t), RE_SLAB.search(t),
        RE_WALL.search(t), RE_BRACE.search(t), RE_CORE.search(t),
        RE_FOUND.search(t)
    ])

# Map type string → coarse class
def coarse_type(t: str) -> str:
    if RE_BEAM.search(t):   return 'Beam'
    if RE_COLUMN.search(t): return 'Column'
    if RE_SLAB.search(t):   return 'Slab'
    if RE_WALL.search(t):   return 'Wall'
    if RE_BRACE.search(t):  return 'Brace'
    if RE_CORE.search(t):   return 'Core'
    if RE_FOUND.search(t):  return 'Foundation'
    return 'Other'

# Function roles
RE_FUNC_LOAD   = re.compile(r'(load[_\- ]?bearing|bearing)', re.I)
RE_FUNC_SHEAR  = re.compile(r'(shear|shear[_\- ]?wall)', re.I)
RE_FUNC_MOMENT = re.compile(r'(moment|bending)', re.I)
RE_FUNC_DIAPH  = re.compile(r'(diaphragm)', re.I)
RE_FUNC_STIFF  = re.compile(r'(stiff|stiffener)', re.I)
RE_FUNC_BRACE  = re.compile(r'(brace|bracing)', re.I)

def func_bucket(f: str) -> str:
    if RE_FUNC_LOAD.search(f):   return 'LoadBearing'
    if RE_FUNC_SHEAR.search(f):  return 'Shear'
    if RE_FUNC_MOMENT.search(f): return 'Moment'
    if RE_FUNC_DIAPH.search(f):  return 'Diaphragm'
    if RE_FUNC_STIFF.search(f):  return 'Stiffener'
    if RE_FUNC_BRACE.search(f):  return 'Bracing'
    return 'Other'

# Predicates by localname (namespace-agnostic)
PRED_ADJ_EL   = 'adjacentElement'
PRED_INT_EL   = 'intersectingElement'
PRED_PART     = 'BFO_0000178'          # hasContinuantPart
PRED_HASFUNC  = 'hasFunction'

# -----------------------
# RDF → model graph facts
# -----------------------
class ModelData:
    def __init__(self, name: str):
        self.name = name
        # node → set(types)
        self.types: Dict[str, Set[str]] = collections.defaultdict(set)
        # node → list(function type strings)
        self.functions: Dict[str, List[str]] = collections.defaultdict(list)
        # undirected structural edges among E nodes
        self.adj_edges: Set[Tuple[str, str]] = set()
        self.int_edges: Set[Tuple[str, str]] = set()
        # composition E→P
        self.part_edges: Set[Tuple[str, str]] = set()

    def add_type(self, s, t):
        self.types[str(s)].add(localname(t))

    def add_func(self, e, ftype):
        self.functions[str(e)].append(localname(ftype))

    def add_undirected(self, store: Set[Tuple[str,str]], a, b):
        a, b = str(a), str(b)
        if a == b: return
        u, v = (a, b) if a < b else (b, a)
        store.add((u, v))

    def add_directed(self, store: Set[Tuple[str,str]], a, b):
        store.add((str(a), str(b)))

def load_model(path: str) -> ModelData:
    g = Graph()
    g.parse(path, format='xml')
    md = ModelData(os.path.basename(path))

    # collect types
    for s, p, o in g.triples((None, RDF.type, None)):
        md.add_type(s, o)

    # scan all triples once for predicates by localname
    for s, p, o in g:
        lp = localname(p)
        if lp == PRED_HASFUNC:
            # s --hasFunction--> fnode ; get fnode type(s)
            # store function TYPES onto element s
            f_types = [localname(ft) for _,_,ft in g.triples((o, RDF.type, None))]
            for ft in f_types:
                md.add_func(s, ft)
        elif lp == PRED_ADJ_EL:
            md.add_undirected(md.adj_edges, s, o)
        elif lp == PRED_INT_EL:
            md.add_undirected(md.int_edges, s, o)
        elif lp == PRED_PART:
            md.add_directed(md.part_edges, s, o)

    return md

# -----------------------
# S1 — Type & Function inventories
# -----------------------
def s1_inventories(models: List[ModelData], out_dir: str):
    rows_types = []
    rows_funcs = []

    for m in models:
        # element-centric types
        elem_nodes = [n for n, ts in m.types.items() if any(is_struct_element(t) for t in ts)]
        counts = collections.Counter()
        for n in elem_nodes:
            # choose a coarse class per node (first matching wins)
            tset = {coarse_type(t) for t in m.types[n]}
            tset.discard('Other')
            if not tset:
                continue
            # allow multiple coarse classes if ambiguous
            for ct in tset:
                counts[ct] += 1
        total = sum(counts.values()) or 1
        for ct, c in sorted(counts.items()):
            rows_types.append({"model": m.name, "coarse_type": ct, "count": c, "share": c/total})

        # function buckets (element→functionType)
        fcounts = collections.Counter()
        total_f = 0
        for e, flist in m.functions.items():
            # only from structural elements (optional guard)
            if not any(is_struct_element(t) for t in m.types.get(e, [])):
                continue
            for f in flist:
                fb = func_bucket(f)
                fcounts[fb] += 1
                total_f += 1
        total_f = total_f or 1
        for fb, c in sorted(fcounts.items()):
            rows_funcs.append({"model": m.name, "function": fb, "count": c, "share": c/total_f})

    df_types = pd.DataFrame(rows_types).sort_values(["model","coarse_type"])
    df_funcs = pd.DataFrame(rows_funcs).sort_values(["model","function"])

    save_csv(df_types, os.path.join(out_dir, "struct_types_histogram.csv"))
    save_csv(df_funcs, os.path.join(out_dir, "struct_functions_histogram.csv"))
    return df_types, df_funcs

# -----------------------
# S2 — Structural micro-motifs
# -----------------------
def node_class(md: ModelData, n: str) -> str:
    # pick a single dominant coarse class for decisions
    ts = md.types.get(n, set())
    for t in ts:
        c = coarse_type(t)
        if c != 'Other': return c
    return 'Other'

def s2_motifs(models: List[ModelData], out_dir: str):
    # Motifs:
    # M2_frameNode: adjacentElement(E_beam – E_column)
    # M3_wallSlab : intersectingElement(E_wall – E_slab)
    # M4_core     : E_core -> P_*  AND exists adjacentElement(Core, SlabElement) or Part-of around
    # M2_braceNode: adjacentElement(E_brace – (E_beam|E_column))
    # M5_structRole: E_* -> F_(load|shear|moment)
    motif_names = ["M2_frameNode", "M3_wallSlab", "M4_core", "M2_braceNode", "M5_structRole"]

    counts = []
    for m in models:
        # Build quick lookups
        adj = m.adj_edges
        inter = m.int_edges

        # frameNode
        frame_pairs = {(a,b) for (a,b) in adj
                       if {'Beam','Column'} == {node_class(m,a), node_class(m,b)}}
        # wall–slab intersections
        wallslab_pairs = {(a,b) for (a,b) in inter
                          if {'Wall','Slab'} == {node_class(m,a), node_class(m,b)}}
        # brace–frame adjacency
        brace_pairs = {(a,b) for (a,b) in adj
                       if (node_class(m,a) == 'Brace' and node_class(m,b) in {'Beam','Column'}) or
                          (node_class(m,b) == 'Brace' and node_class(m,a) in {'Beam','Column'})}

        # core motif: core element that has parts and is adjacent to at least one slab element
        parts_by_e = collections.defaultdict(set)
        for e,p in m.part_edges:
            parts_by_e[e].add(p)
        # adjacency map to find neighbor slab
        nbrs = collections.defaultdict(set)
        for a,b in adj:
            nbrs[a].add(b); nbrs[b].add(a)

        core_nodes = []
        for n, ts in m.types.items():
            if node_class(m, n) == 'Core' and len(parts_by_e.get(n,()))>0:
                # neighbor slab?
                has_slab_nbr = any(node_class(m, x) == 'Slab' for x in nbrs.get(n,()))
                if has_slab_nbr:
                    core_nodes.append(n)

        # structRole: E_* → F_(load|shear|moment)
        role_edges = 0
        for e, flist in m.functions.items():
            if not any(is_struct_element(t) for t in m.types.get(e, [])):
                continue
            for f in flist:
                fb = func_bucket(f)
                if fb in ('LoadBearing','Shear','Moment'):
                    role_edges += 1

        # Counts
        c_frame = len(frame_pairs)
        c_wallslab = len(wallslab_pairs)
        c_core = len(core_nodes)
        c_brace = len(brace_pairs)

        counts.append({
            "model": m.name,
            "M2_frameNode": c_frame,
            "M3_wallSlab": c_wallslab,
            "M4_core": c_core,
            "M2_braceNode": c_brace,
            "M5_structRole": role_edges
        })

    df_counts = pd.DataFrame(counts).set_index("model").sort_index()
    # shares: row-normalize over listed motifs (avoid zero-div)
    shares = df_counts.div(df_counts.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    shares.index.name = "model"

    save_csv(df_counts, os.path.join(out_dir, "struct_motif_counts.csv"))
    save_csv(shares.reset_index(), os.path.join(out_dir, "struct_motif_shares.csv"))
    return df_counts, shares

# -----------------------
# S3 — Macro system scores
# -----------------------
def minmax01(x: float, lo: float, hi: float) -> float:
    if hi <= lo: return 0.0
    return max(0.0, min(1.0, (x - lo) / (hi - lo)))

def s3_system_scores(models: List[ModelData], motif_counts: pd.DataFrame,
                     types_hist: pd.DataFrame, func_hist: pd.DataFrame,
                     out_dir: str):
    # Prepare helper aggregates per model
    # nE: number of structural elements
    nE = {}
    for m in models:
        nE[m.name] = sum(1 for n,ts in m.types.items() if any(is_struct_element(t) for t in ts))
    # helpers
    def share_func(model: str, fname: str) -> float:
        df = func_hist.query("model == @model and function == @fname")
        return float(df["share"].sum()) if not df.empty else 0.0
    def count_type(model: str, tname: str) -> int:
        df = types_hist.query("model == @model and coarse_type == @tname")
        return int(df["count"].sum()) if not df.empty else 0

    # motif densities (per 100 structural elements)
    dens = motif_counts.copy()
    for m in motif_counts.index:
        denom = max(1, nE.get(m,1))
        dens.loc[m,:] = (motif_counts.loc[m,:] / denom) * 100.0

    # pick ranges for min-max (robust): percentiles across models
    def pctl(col):
        arr = dens[col].astype(float).values
        return np.nanpercentile(arr, 10), np.nanpercentile(arr, 90)

    p_frame_lo, p_frame_hi = pctl("M2_frameNode")
    p_wall_lo,  p_wall_hi  = pctl("M3_wallSlab")
    p_br_lo,    p_br_hi    = pctl("M2_braceNode")
    p_core_lo,  p_core_hi  = pctl("M4_core")

    rows = []
    for m in motif_counts.index:
        # frame macro: frameNode density + Moment function share
        frame_raw = minmax01(float(dens.loc[m,"M2_frameNode"]), p_frame_lo, p_frame_hi)
        frame = 0.7*frame_raw + 0.3*share_func(m, "Moment")

        # wall macro: wall-slab density + Load/Shear on walls (approx via function share)
        wall_raw = minmax01(float(dens.loc[m,"M3_wallSlab"]), p_wall_lo, p_wall_hi)
        wall = 0.6*wall_raw + 0.4*(share_func(m,"LoadBearing") + share_func(m,"Shear"))/2.0

        # braced macro: brace-node density (+ bracing func if present)
        br_raw = minmax01(float(dens.loc[m,"M2_braceNode"]), p_br_lo, p_br_hi)
        braced = 0.8*br_raw + 0.2*share_func(m,"Bracing")

        # dual macro: frame and wall both high → take harmonic-ish combination
        dual = (frame*wall) / max(1e-6, (0.5*frame + 0.5*wall))

        rows.append({
            "model": m,
            "frame": float(frame),
            "wall": float(wall),
            "dual": float(dual),
            "braced": float(braced)
        })

    df_sys = pd.DataFrame(rows).set_index("model").sort_index()
    save_csv(df_sys.reset_index(), os.path.join(out_dir, "struct_system_scores.csv"))
    return df_sys

# -----------------------
# S4 — Structural similarity
# -----------------------
def cosine_sim_matrix(X: np.ndarray) -> np.ndarray:
    # rows = models, columns = features
    X = np.asarray(X, dtype=float)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms==0] = 1.0
    Y = X / norms
    return np.clip(Y @ Y.T, 0.0, 1.0)

def s4_similarity(motif_shares: pd.DataFrame, sys_scores: pd.DataFrame, out_dir: str,
                  beta1: float = 0.7, beta2: float = 0.3):
    # align models
    labels = sorted(set(motif_shares.index) & set(sys_scores.index))
    M = motif_shares.loc[labels, :]
    S = sys_scores.loc[labels, :]

    S_motif = cosine_sim_matrix(M.values)
    S_sys   = cosine_sim_matrix(S.values)
    S_comb  = beta1*S_motif + beta2*S_sys

    df = pd.DataFrame(S_comb, index=labels, columns=labels)
    df.to_csv(os.path.join(out_dir, "struct_similarity_matrix.csv"), index=True)

    # small pairwise summary
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
    ap.add_argument("--beta1", type=float, default=0.7, help="weight for motif similarity within structural channel")
    ap.add_argument("--beta2", type=float, default=0.3, help="weight for system similarity within structural channel")
    args = ap.parse_args()

    out_dir = os.path.join(args.out_root, args.out_name)
    ensure_dir(out_dir)

    paths = sorted(glob.glob(os.path.join(args.input-dir if hasattr(args,'input-dir') else args.input_dir, args.pattern)))
    if not paths:
        raise SystemExit(f"No RDF files matched: {args.input_dir}/{args.pattern}")

    print("[LOAD]", *[os.path.basename(p) for p in paths], sep="\n       ")
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
    s4_similarity(motif_shares, sys_scores, out_dir,
              beta1=args.beta1, beta2=args.beta2)

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
