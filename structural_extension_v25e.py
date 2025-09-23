# structural_extension_v25e.py
# Structural Extension S1–S4 (v25d -> v25e)
# - Multilingual (EN/DE/TR) regex for types & functions
# - IFC shortcuts (IfcWall/IfcSlab/IfcBeam/IfcColumn/IfcStructuralCurveMember…)
# - rdfs:label enrichment for class & function types (lang-priority aware)
# - Keeps: predicate aliases, M5 injection AFTER S2, system+similarity logic
# - Diagnostics: struct_data_availability.csv (+ debug extras)
#
# Outputs under out_root/out_name:
#   struct_data_availability.csv
#   struct_types_histogram.csv
#   struct_functions_histogram.csv
#   struct_motif_counts.csv
#   struct_motif_shares.csv
#   struct_system_scores.csv
#   struct_similarity_matrix.csv
#   pairwise_structural_summary.csv
#   (if --emit-debug) struct_functions_shares_wide.csv, struct_motif_densities_per100.csv, struct_score_components.csv

import argparse, os, glob, re, collections, json
from typing import Dict, Set, Tuple, List
import pandas as pd
import numpy as np
from rdflib import Graph, RDF
from rdflib.namespace import RDFS

VERSION = "v25e"

def ensure_dir(p: str): os.makedirs(p, exist_ok=True)
def localname(uri) -> str:
    s = str(uri)
    if '#' in s: return s.split('#')[-1]
    return s.rstrip('/').split('/')[-1]

# ---------------- multilingual IFC/type regex (EN/DE/TR) ----------------
# Note: We run these on both localName and rdfs:label texts.
RE_IFC_BEAM   = re.compile(r'(IfcBeam|(^|[^A-Za-z])Beam([^A-Za-z]|$)|Girder|Tr(ä|ae)ger|Kir(i|ı)s)', re.I)
RE_IFC_COLUMN = re.compile(r'(IfcColumn|(^|[^A-Za-z])Column([^A-Za-z]|$)|Pillar|St(ü|ue)tze|S(ü|u)tun|Kolon)', re.I)
RE_IFC_SLAB   = re.compile(r'(IfcSlab|IfcFloor|(^|[^A-Za-z])(Slab|Deck|Floor|Plate|Decke|Boden(platte)?|Platte|D(ö|o)seme)([^A-Za-z]|$))', re.I)
RE_IFC_WALL   = re.compile(r'(IfcWall(StandardCase)?|(^|[^A-Za-z])(Wall|Shear[ -]?Wall|CoreWall|RetainingWall|Wand|Duvar)([^A-Za-z]|$))', re.I)
RE_IFC_BRACE  = re.compile(r'(IfcStructuralCurveMember|(^|[^A-Za-z])(Brace|Bracing|X[ -]?brace|K[ -]?brace|Chevron|Tie|Strut|Aussteif|Verband|[ÇC]apraz)([^A-Za-z]|$))', re.I)
RE_IFC_CORE   = re.compile(r'((Ifc)?(Core|CoreWall|ShearCore|Kern|[ÇC]ekirdek|Shaft))', re.I)
RE_IFC_FOUND  = re.compile(r'((Ifc)?(Foundation|Fundament|Footing|Pad|Strip|Pile|Raft|Mat|Temel))', re.I)
RE_ZONE       = re.compile(r'(Zone|Space|Room|IfcSpace|Raum)', re.I)

def is_struct_element(text: str) -> bool:
    t = text
    return any([
        RE_IFC_BEAM.search(t), RE_IFC_COLUMN.search(t), RE_IFC_SLAB.search(t),
        RE_IFC_WALL.search(t), RE_IFC_BRACE.search(t), RE_IFC_CORE.search(t),
        RE_IFC_FOUND.search(t)
    ])

def coarse_type(text: str) -> str:
    t = text
    if RE_IFC_BEAM.search(t):   return 'Beam'
    if RE_IFC_COLUMN.search(t): return 'Column'
    if RE_IFC_SLAB.search(t):   return 'Slab'
    if RE_IFC_WALL.search(t):   return 'Wall'
    if RE_IFC_BRACE.search(t):  return 'Brace'
    if RE_IFC_CORE.search(t):   return 'Core'
    if RE_IFC_FOUND.search(t):  return 'Foundation'
    return 'Other'

def is_zone_text(text: str) -> bool:
    return bool(RE_ZONE.search(text))

# ---------------- multilingual function buckets (EN/DE/TR) ----------------
RE_FUNC_LOAD   = re.compile(r'(load[_\- ]?bearing|tragend|ta(s|ş)iyici)', re.I)
RE_FUNC_SHEAR  = re.compile(r'(shear|scher|kesme)', re.I)
RE_FUNC_MOMENT = re.compile(r'(moment|bending|rigid\s*frame|moment\s*frame|momenten|rijit|r(i|ı)jit)', re.I)
RE_FUNC_DIAPH  = re.compile(r'(diaphragm|deckenscheibe|diyafram)', re.I)
RE_FUNC_STIFF  = re.compile(r'(stiff|steif|sertle(s|ş)tir)', re.I)
RE_FUNC_BRACE  = re.compile(r'(brace|bracing|aussteif|verband|[çc]apraz|tie|strut)', re.I)

def func_bucket(text: str) -> str:
    t = text
    if RE_FUNC_LOAD.search(t):   return 'LoadBearing'
    if RE_FUNC_SHEAR.search(t):  return 'Shear'
    if RE_FUNC_MOMENT.search(t): return 'Moment'
    if RE_FUNC_DIAPH.search(t):  return 'Diaphragm'
    if RE_FUNC_STIFF.search(t):  return 'Stiffener'
    if RE_FUNC_BRACE.search(t):  return 'Bracing'
    return 'Other'

# ---------------- predicates (local names) ----------------
PRED_ADJ_CANDS   = {'adjacentElement','adjacentElements','adjacent'}
PRED_INT_CANDS   = {'intersectingElement','intersectsElement'}
PRED_ZONE_CANDS  = {'adjacentZone','hasSpace','adjacent_space'}
PRED_PART_CANDS  = {'BFO_0000178','hasContinuantPart','hasPart','partOf'}
PRED_HASFUNC     = 'hasFunction'

# ---------------- model container ----------------
class ModelData:
    def __init__(self, name: str):
        self.name = name
        # We store arbitrary "type texts" (localName + labels) for each node
        self.types: Dict[str, Set[str]] = collections.defaultdict(set)       # node -> set(texts)
        self.functions: Dict[str, List[str]] = collections.defaultdict(list) # element -> list of function "texts"
        self.adj_edges: Set[Tuple[str,str]] = set()     # undirected E–E
        self.int_edges: Set[Tuple[str,str]] = set()     # undirected E–E
        self.part_edges: Set[Tuple[str,str]] = set()    # directed E->P
        self.adj_zone:  Set[Tuple[str,str]] = set()     # directed E->Z

    def add_type_text(self, s, text): self.types[str(s)].add(str(text))
    def add_func_text(self, e, text): self.functions[str(e)].append(str(text))

    def add_undirected(self, store, a, b):
        a, b = str(a), str(b)
        if a == b: return
        u, v = (a,b) if a < b else (b,a)
        store.add((u,v))

    def add_directed(self, store, a, b): store.add((str(a), str(b)))

# label helpers
def all_labels(g: Graph, node, lang_priority: List[str]) -> List[str]:
    vals = []
    for lit in g.objects(node, RDFS.label):
        try:
            txt = str(lit)
            lang = (lit.language or "").lower()
        except Exception:
            txt, lang = str(lit), ""
        vals.append((lang, txt))
    if not vals: return []
    # sort by preferred language order
    order = {lang:i for i,lang in enumerate([x.lower() for x in lang_priority])}
    vals.sort(key=lambda t: order.get(t[0], 999))
    return [v for _,v in vals]

def load_model(path: str, lang_priority: List[str]) -> ModelData:
    g = Graph(); g.parse(path, format='xml')
    md = ModelData(os.path.basename(path))

    # --- collect rdf:types with localName + labels of the CLASS ---
    for s,_,o in g.triples((None, RDF.type, None)):
        # localName of class
        md.add_type_text(s, localname(o))
        # all labels of the class (EN/DE/TR prioritized)
        for lab in all_labels(g, o, lang_priority):
            md.add_type_text(s, lab)

    # --- edges & functions ---
    for s,p,o in g:
        lp = localname(p)
        if lp == PRED_HASFUNC:
            # take rdf:type of function individual
            f_texts = []
            for _,_,ft in g.triples((o, RDF.type, None)):
                f_texts.append(localname(ft))
                for lab in all_labels(g, ft, lang_priority): f_texts.append(lab)
            # if no type found, try label on function node
            if not f_texts:
                for lab in all_labels(g, o, lang_priority): f_texts.append(lab)
            if not f_texts:
                f_texts = [localname(o)]
            for txt in f_texts:
                md.add_func_text(s, txt)
        elif lp in PRED_ADJ_CANDS:
            md.add_undirected(md.adj_edges, s, o)
        elif lp in PRED_INT_CANDS:
            md.add_undirected(md.int_edges, s, o)
        elif lp in PRED_PART_CANDS:
            md.add_directed(md.part_edges, s, o)
        elif lp in PRED_ZONE_CANDS:
            md.add_directed(md.adj_zone, s, o)
    return md

# ---------------- utilities ----------------
def node_class(md: ModelData, n: str) -> str:
    for t in md.types.get(n, set()):
        c = coarse_type(t)
        if c != 'Other': return c
    return 'Other'

def is_zone_node(md: ModelData, n: str) -> bool:
    for t in md.types.get(n, set()):
        if is_zone_text(t): return True
    return False

def minmax01(x: float, lo: float, hi: float) -> float:
    if hi <= lo: return 0.0
    v = (x - lo) / (hi - lo)
    return float(max(0.0, min(1.0, v)))

# ---------------- S1: inventories ----------------
def s1_inventories(models: List[ModelData], out_dir: str, func_all: bool, emit_debug: bool):
    rows_types, rows_funcs, avail_rows = [], [], []

    for m in models:
        # structural element nodes
        elem_nodes = [n for n, ts in m.types.items() if any(is_struct_element(t) for t in ts)]
        tcounts = collections.Counter()
        for n in elem_nodes:
            cset = {coarse_type(t) for t in m.types[n]}
            cset.discard('Other')
            for ct in cset: tcounts[ct] += 1
        total_t = sum(tcounts.values()) or 1
        for ct, c in sorted(tcounts.items()):
            rows_types.append({"model": m.name, "coarse_type": ct, "count": c, "share": c/total_t})

        # function histogram
        fcounts = collections.Counter(); total_f = 0
        for e, flist in m.functions.items():
            if (not func_all) and (e not in elem_nodes):  # gate to structural elems
                continue
            for f in flist:
                fb = func_bucket(f)
                fcounts[fb] += 1; total_f += 1
        total_f = total_f or 1
        for fb, c in sorted(fcounts.items()):
            rows_funcs.append({"model": m.name, "function": fb, "count": c, "share": c/total_f})

        # availability
        avail_rows.append({
            "model": m.name,
            "n_struct_nodes": len(elem_nodes),
            "n_types_total": sum(len(v) for v in m.types.values()),
            "n_type_Beam": tcounts.get("Beam", 0),
            "n_type_Column": tcounts.get("Column", 0),
            "n_type_Slab": tcounts.get("Slab", 0),
            "n_type_Wall": tcounts.get("Wall", 0),
            "n_type_Brace": tcounts.get("Brace", 0),
            "n_type_Core": tcounts.get("Core", 0),
            "n_type_Foundation": tcounts.get("Foundation", 0),
            "n_funcs_total": sum(fcounts.values()),
            "n_func_LB": fcounts.get("LoadBearing",0),
            "n_func_Shear": fcounts.get("Shear",0),
            "n_func_Moment": fcounts.get("Moment",0),
            "n_adj_edges": len(m.adj_edges),
            "n_int_edges": len(m.int_edges),
            "n_part_edges": len(m.part_edges),
            "n_adj_zone": len(m.adj_zone),
            "has_adj": int(len(m.adj_edges)>0),
            "has_int": int(len(m.int_edges)>0),
            "has_zone": int(len(m.adj_zone)>0)
        })

    df_types = (pd.DataFrame(rows_types)
                if rows_types else pd.DataFrame(columns=["model","coarse_type","count","share"]))
    df_funcs = (pd.DataFrame(rows_funcs)
                if rows_funcs else pd.DataFrame(columns=["model","function","count","share"]))
    if not df_types.empty: df_types = df_types.sort_values(["model","coarse_type"])
    if not df_funcs.empty: df_funcs = df_funcs.sort_values(["model","function"])

    df_types.to_csv(os.path.join(out_dir, "struct_types_histogram.csv"), index=False)
    df_funcs.to_csv(os.path.join(out_dir, "struct_functions_histogram.csv"), index=False)

    if emit_debug and not df_funcs.empty:
        pivot = df_funcs.pivot_table(index="model", columns="function", values="share", aggfunc="sum").fillna(0.0)
        pivot.to_csv(os.path.join(out_dir, "struct_functions_shares_wide.csv"))

    pd.DataFrame(avail_rows).to_csv(os.path.join(out_dir, "struct_data_availability.csv"), index=False)
    return df_types, df_funcs

# ---------------- S2: structural motifs ----------------
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

        # wall–slab via common zone
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

        # core: core with parts and at least one Slab neighbor
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

        rows.append({
            "model": m.name,
            "M2_frameNode": len(frame_pairs),
            "M3_wallSlab": len(wallslab_int),
            "M3_wallSlabAdj": len(wallslab_adj),
            "M3_wallSlabZone": len(wallslab_zone_pairs),
            "M4_core": len(core_nodes),
            "M2_braceNode": len(brace_pairs),
            "M5_structRole": 0  # will be injected from S1
        })

    df_counts = pd.DataFrame(rows).set_index("model")
    df_counts = df_counts.reindex(columns=motif_cols, fill_value=0).sort_index()
    return df_counts

def inject_M5_from_s1(df_counts: pd.DataFrame, df_funcs: pd.DataFrame) -> pd.DataFrame:
    if df_counts.empty or df_funcs.empty:
        return df_counts
    m5_map = {}
    for m in df_counts.index:
        sub = df_funcs.query("model == @m and function in ['LoadBearing','Shear','Moment']")
        m5_map[m] = int(sub['count'].sum()) if not sub.empty else 0
    if 'M5_structRole' not in df_counts.columns:
        df_counts['M5_structRole'] = 0
    for m, v in m5_map.items():
        if m in df_counts.index:
            df_counts.at[m, 'M5_structRole'] = int(v)
    return df_counts

# ---------------- S3: system scores ----------------
def s3_system_scores(models: List[ModelData],
                     motif_counts: pd.DataFrame,
                     func_hist: pd.DataFrame,
                     out_dir: str,
                     dual_frame_share_threshold: float = 0.25,
                     emit_debug: bool = False):

    # structural element counts per model
    nE = {}
    for md in models:
        nE[md.name] = sum(1 for n,ts in md.types.items() if any(is_struct_element(t) for t in ts))

    # function share lookup
    def share_func(model: str, fname: str) -> float:
        df = func_hist.query("model == @model and function == @fname")
        return float(df["share"].sum()) if not df.empty else 0.0

    # densities per 100 elements
    dens = motif_counts.astype(float).copy()
    for m in dens.index:
        denom = max(1, nE.get(m,1))
        dens.loc[m,:] = (dens.loc[m,:] / denom) * 100.0

    # percentile min–max per motif
    col_ranges = {}
    for col in dens.columns:
        arr = dens[col].astype(float).values
        lo, hi = np.nanpercentile(arr, 10), np.nanpercentile(arr, 90)
        col_ranges[col] = (float(lo), float(hi))

    def norm_col(m, col):
        lo, hi = col_ranges[col]
        return minmax01(float(dens.loc[m,col]), lo, hi)

    rows = []
    rows_comp = []
    for m in dens.index:
        frame_motif = norm_col(m, "M2_frameNode")
        frame_func  = share_func(m, "Moment")
        frame = 0.7*frame_motif + 0.3*frame_func

        wall_m3 = 0.6*norm_col(m, "M3_wallSlab") + 0.3*norm_col(m, "M3_wallSlabAdj") + 0.1*norm_col(m, "M3_wallSlabZone")
        wall_func = ((share_func(m,"LoadBearing")+share_func(m,"Shear"))/2.0)
        wall = 0.6*wall_m3 + 0.4*wall_func

        braced = 0.8*norm_col(m, "M2_braceNode") + 0.2*share_func(m, "Bracing")

        denom = max(1e-6, (0.5*frame + 0.5*wall))
        dual_raw = (frame*wall) / denom
        frame_share = frame / max(1e-6, (frame + wall))
        dual = 0.0 if frame_share < dual_frame_share_threshold else dual_raw

        rows.append({"model": m, "frame": frame, "wall": wall, "dual_raw": dual_raw, "dual": dual, "braced": braced, "frame_share": frame_share})
        rows_comp.append({"model": m, "frame_motif": frame_motif, "frame_func": frame_func, "wall_m3": wall_m3, "wall_func": wall_func})

    df_sys = pd.DataFrame(rows).set_index("model").sort_index()
    df_sys.to_csv(os.path.join(out_dir, "struct_system_scores.csv"))

    if emit_debug:
        dens.to_csv(os.path.join(out_dir, "struct_motif_densities_per100.csv"))
        pd.DataFrame(rows_comp).set_index("model").sort_index().to_csv(os.path.join(out_dir, "struct_score_components.csv"))

    return df_sys

# ---------------- S4: structural similarity ----------------
def cosine_sim_matrix(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    norms = np.linalg.norm(X, axis=1, keepdims=True); norms[norms==0] = 1.0
    Y = X / norms
    S = Y @ Y.T
    return np.clip(S, 0.0, 1.0)

def s4_similarity(motif_shares: pd.DataFrame, sys_scores: pd.DataFrame,
                  out_dir: str, beta1: float = 0.7, beta2: float = 0.3):
    labels = sorted(set(motif_shares.index) & set(sys_scores.index))
    if not labels:
        raise SystemExit("No common labels between motif_shares and sys_scores.")
    M = motif_shares.loc[labels, :]  # model × motifs
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

# ---------------- orchestrator ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--pattern", default="*_DG.rdf")
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--out-name", default="07 - Structural_Extension_v25e")
    ap.add_argument("--beta1", type=float, default=0.7)  # motif
    ap.add_argument("--beta2", type=float, default=0.3)  # system
    ap.add_argument("--dual-thresh", type=float, default=0.25)
    ap.add_argument("--func-all", action="store_true", help="Count functions on ALL elements in S1.")
    ap.add_argument("--emit-debug", action="store_true", help="Write additional diagnostics.")
    ap.add_argument("--lang-priority", default="en,de,tr", help="Comma-separated preferred label languages (lowercase).")
    args = ap.parse_args()

    out_dir = os.path.join(args.out_root, args.out_name); ensure_dir(out_dir)
    input_dir = getattr(args, "input_dir", getattr(args, "input-dir", None))
    if input_dir is None: raise SystemExit("--input-dir is required")
    paths = sorted(glob.glob(os.path.join(input_dir, args.pattern)))
    if not paths: raise SystemExit(f"No RDF files matched: {input_dir}/{args.pattern}")

    langs = [x.strip().lower() for x in args.lang_priority.split(",") if x.strip()]
    print(f"[LOAD] (Structural Extension {VERSION})  label-langs={langs}")
    for p in paths: print("      ", os.path.basename(p))
    models = [load_model(p, langs) for p in paths]

    # S1
    df_types, df_funcs = s1_inventories(models, out_dir, func_all=args.func_all, emit_debug=args.emit_debug)

    # S2 - counts (M5=0 here, will inject)
    motif_counts = s2_motifs(models, out_dir)

    # ---- M5 INJECTION (from S1) ----
    motif_counts = inject_M5_from_s1(motif_counts, df_funcs)
    # shares AFTER injection (M5 dahil edilmezse: sadece M2-M4-… ile normalize etmek isterseniz burada kolonları filtreleyebilirsiniz)
    motif_shares = motif_counts.div(motif_counts.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    motif_counts.to_csv(os.path.join(out_dir, "struct_motif_counts.csv"))
    motif_shares.to_csv(os.path.join(out_dir, "struct_motif_shares.csv"))

    # S3
    sys_scores = s3_system_scores(models=models,
                                  motif_counts=motif_counts,
                                  func_hist=df_funcs,
                                  out_dir=out_dir,
                                  dual_frame_share_threshold=args.dual_thresh,
                                  emit_debug=args.emit_debug)

    # S4
    s4_similarity(motif_shares, sys_scores, out_dir, beta1=args.beta1, beta2=args.beta2)

    print(f"\n[OK] Saved outputs under: {out_dir}")
    print(" - struct_data_availability.csv")
    print(" - struct_types_histogram.csv")
    print(" - struct_functions_histogram.csv")
    print(" - struct_motif_counts.csv")
    print(" - struct_motif_shares.csv")
    print(" - struct_system_scores.csv")
    print(" - struct_similarity_matrix.csv")
    print(" - pairwise_structural_summary.csv")
    if args.emit_debug:
        print(" - struct_functions_shares_wide.csv")
        print(" - struct_motif_densities_per100.csv")
        print(" - struct_score_components.csv")

if __name__ == "__main__":
    main()
