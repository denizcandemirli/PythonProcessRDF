# -*- coding: utf-8 -*-
"""
Structural Extension v25n
- Patch-1: Neighborhood proxy v2 for M2 (frame) & M3 (wall-slab)
- Patch-2: Expanded regex/IFC dictionaries for Brace/Moment & Wall/Slab variants
- Patch-3: Similarity mixing: Cosine ⊕ Hellinger (to avoid 0.99+ collapse)

Outputs (under out_root/out_name):
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
 - motif_proxy_edges.csv                 (debug)
 - type_mapping_hits.csv / unknown.csv   (debug)
"""
import os, re, json, argparse, glob, math
from collections import defaultdict, Counter
import numpy as np
import pandas as pd
from rdflib import Graph, URIRef, Literal, Namespace, RDF, RDFS

# -------------------------
# Helpers
# -------------------------
def localname(x):
    s = str(x)
    if '#' in s:
        return s.rsplit('#',1)[1]
    return s.rstrip('/').rsplit('/',1)[-1]

def lc(s):
    return (s or "").lower()

def hellinger_sim(p, q, eps=1e-12):
    p = np.clip(np.asarray(p, dtype=float), 0, None)
    q = np.clip(np.asarray(q, dtype=float), 0, None)
    ps = p.sum(); qs = q.sum()
    if ps <= 0 and qs <= 0: return 1.0
    if ps <= 0: p = np.ones_like(p) / len(p)
    else: p = p / ps
    if qs <= 0: q = np.ones_like(q) / len(q)
    else: q = q / qs
    d = np.sqrt(0.5 * np.sum((np.sqrt(p+eps) - np.sqrt(q+eps))**2))
    return float(max(0.0, 1.0 - d))

def cosine_sim(a, b):
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    if na == 0 and nb == 0: return 1.0
    if na == 0 or nb == 0: return 0.0
    return float(np.dot(a, b) / (na * nb))

def mix_sim(v1, v2):
    # Average of cosine and Hellinger similarity
    return 0.5 * cosine_sim(v1, v2) + 0.5 * hellinger_sim(v1, v2)

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

# -------------------------
# Dictionaries (regex)
# -------------------------
RE_BEAM       = re.compile(r'\b(beam|girder)\b', re.I)
RE_COLUMN     = re.compile(r'\b(column|pillar)\b', re.I)
RE_SLAB       = re.compile(r'\b(slab|deck|floor|plate)\b', re.I)
RE_WALL       = re.compile(r'\b(wall|shear[- ]?wall|corewall|retainingwall)\b', re.I)
RE_BRACE      = re.compile(r'\b(brace|bracing|x[- ]?brace|tie|strut|diagonal)\b', re.I)
RE_CORE       = re.compile(r'\b(core(?!wall))\b', re.I)
RE_FOUNDATION = re.compile(r'\b(foundation|footing|pilecap|pile|raft|mat)\b', re.I)

# IFC fast paths
IFC_BEAM   = re.compile(r'\bIfcBeam\b', re.I)
IFC_COL    = re.compile(r'\bIfcColumn\b', re.I)
IFC_SLAB   = re.compile(r'\b(IfcSlab|IfcFloor)\b', re.I)
IFC_WALL   = re.compile(r'\bIfcWall\b', re.I)
IFC_BRACE1 = re.compile(r'\bIfcStructuralCurveMember\b', re.I)
IFC_BRACE2 = re.compile(r'\bIfcMember\b', re.I)

# Functions (roles)
RE_LOADBEARING = re.compile(r'load[- ]?bearing|bearing', re.I)
RE_SHEAR       = re.compile(r'\bshear\b', re.I)
RE_MOMENT      = re.compile(r'\bmoment(frame)?\b|\brigid[- ]?frame\b|\b(smrf|imrf)\b', re.I)

# Predicates (robust by suffix)
def pred_is(p, name):
    return lc(localname(p)) == lc(name)

def pred_like(p, name):
    return lc(localname(p)) == lc(name) or lc(str(p)).endswith(name)

# BOT/BFO/common predicate names
P_ADJ_EL   = "adjacentElement"
P_INT_EL   = "intersectingElement"
P_ADJ_ZONE = "adjacentZone"
P_HAS_PART = "BFO_0000178"  # hasContinuantPart
P_HASFUNC  = "hasFunction"

# -------------------------
# Model load
# -------------------------
class Model:
    def __init__(self, name, g):
        self.name = name
        self.g = g
        self.labels = defaultdict(set)      # node -> {labels}
        self.types  = defaultdict(set)      # node -> {rdf:type localnames}
        self.pred_edges = defaultdict(list) # pred_localname -> [(s,o), ...]
        self.node2zones = defaultdict(set)  # node -> {zone ids}

def load_models(input_dir, pattern):
    files = sorted(glob.glob(os.path.join(input_dir, pattern)))
    models = []
    for f in files:
        g = Graph()
        g.parse(f)
        m = Model(os.path.basename(f), g)
        # labels
        for s, p, o in g.triples((None, RDFS.label, None)):
            if isinstance(o, Literal):
                m.labels[s].add(str(o))
        # types
        for s, p, o in g.triples((None, RDF.type, None)):
            m.types[s].add(localname(o))
        # predicate edges
        for s, p, o in g:
            pl = localname(p)
            if pred_like(p, P_ADJ_EL) or pred_like(p, P_INT_EL) or pred_like(p, P_ADJ_ZONE) or P_HAS_PART in str(p) or pred_like(p, P_HASFUNC):
                m.pred_edges[pl].append((s, o))
            # zones
            if pred_like(p, P_ADJ_ZONE):
                m.node2zones[s].add(o)
        models.append(m)
    return models

# -------------------------
# S1: type/function inventory
# -------------------------
STRUCT_CLASSES = ["Beam","Column","Slab","Wall","Brace","Core","Foundation"]

def detect_struct_class(name_or_label):
    s = lc(name_or_label)
    if IFC_BEAM.search(s) or RE_BEAM.search(s): return "Beam"
    if IFC_COL.search(s) or RE_COLUMN.search(s): return "Column"
    if IFC_SLAB.search(s) or RE_SLAB.search(s): return "Slab"
    if IFC_WALL.search(s) or RE_WALL.search(s): return "Wall"
    if IFC_BRACE1.search(s) or IFC_BRACE2.search(s) or RE_BRACE.search(s): return "Brace"
    if RE_CORE.search(s): return "Core"
    if RE_FOUNDATION.search(s): return "Foundation"
    return None

def detect_function(labels_or_types):
    txt = " ".join(lc(x) for x in labels_or_types)
    if RE_MOMENT.search(txt): return "Moment"
    if RE_SHEAR.search(txt):  return "Shear"
    if RE_LOADBEARING.search(txt): return "LoadBearing"
    return None

def s1_inventory(models, func_all=False):
    rows_types = []
    rows_funcs = []
    type_hits = []
    type_unknown = []
    func_counter = defaultdict(lambda: Counter())  # model -> Counter(LB/Shear/Moment)
    availability = []

    for m in models:
        # availability (per model)
        av = {
            "model": m.name,
            "has_adjacentElement": int(len(m.pred_edges.get(P_ADJ_EL, []))>0),
            "has_intersectingElement": int(any(pred_like(p,P_INT_EL) for p in m.pred_edges.keys())),
            "has_hasContinuantPart": int(len([1 for k in m.pred_edges.keys() if P_HAS_PART in k or P_HAS_PART in lc(k)])>0),
            "has_adjacentZone": int(len(m.pred_edges.get(P_ADJ_ZONE, []))>0),
            "has_hasFunction": int(len(m.pred_edges.get(P_HASFUNC, []))>0),
        }
        availability.append(av)

        # types
        for n, tset in m.types.items():
            # collect candidate strings: localnames of types + labels of node
            cands = set(tset) | set(localname(n))
            cands |= set().union(*[set(localname(t)) for t in tset]) if tset else set()
            cands |= m.labels.get(n, set())
            found = None
            for c in cands:
                cls = detect_struct_class(c)
                if cls:
                    found = cls; break
            if found:
                rows_types.append({"model": m.name, "node": str(n), "type": found})
                type_hits.append({"model": m.name, "node": str(n), "hit": found})
            else:
                type_unknown.append({"model": m.name, "node": str(n), "unknown": "|".join(sorted(cands))[:500]})

        # functions
        # 1) explicit core:hasFunction targets
        func_map = defaultdict(set)  # node -> {LB/Shear/Moment}
        for s, o in m.pred_edges.get(P_HASFUNC, []):
            # try types/labels on 'o'
            ftxt = set([localname(o)]) | m.labels.get(o, set())
            f = detect_function(ftxt)
            if f: func_map[s].add(f)

        # 2) optionally broaden with func_all (regex on node labels/types)
        if func_all:
            for n, tset in m.types.items():
                base = set([localname(n)]) | m.labels.get(n, set()) | set(tset)
                f = detect_function(base)
                if f: func_map[n].add(f)

        # aggregate per model
        for n, fs in func_map.items():
            for f in fs:
                func_counter[m.name][f] += 1

    # build dataframes
    df_types = pd.DataFrame(rows_types) if rows_types else pd.DataFrame(columns=["model","node","type"])
    df_types_hist = df_types.groupby(["model","type"]).size().reset_index(name="count")

    rows_f = []
    for model, cnt in func_counter.items():
        total = sum(cnt.values()) if sum(cnt.values())>0 else 1
        for f in ["LoadBearing","Shear","Moment"]:
            c = cnt.get(f,0)
            rows_f.append({"model": model, "function": f, "count": c, "share": c/total})

    df_funcs = pd.DataFrame(rows_f) if rows_f else pd.DataFrame(columns=["model","function","count","share"])
    df_funcs_hist = df_funcs[["model","function","count"]].copy()
    df_funcs_wide = df_funcs.pivot_table(index="model", columns="function", values="share", fill_value=0.0).reset_index()

    df_av = pd.DataFrame(availability)
    df_hits = pd.DataFrame(type_hits) if type_hits else pd.DataFrame(columns=["model","node","hit"])
    df_unknown = pd.DataFrame(type_unknown) if type_unknown else pd.DataFrame(columns=["model","node","unknown"])

    return df_types, df_types_hist, df_funcs, df_funcs_hist, df_funcs_wide, df_av, df_hits, df_unknown

# -------------------------
# S2: motif counts & shares  (with proxy v2)
# -------------------------
def s2_motifs(models, df_types, df_funcs_wide, proxy_penalty=0.5, allow_type_only_proxy=False, out_dir=None):
    # index types per model
    model_nodes_by_type = defaultdict(lambda: defaultdict(set))  # model -> class -> nodes
    node_types_lookup = defaultdict(dict)                        # model -> node -> class or None

    for _, row in df_types.iterrows():
        model = row["model"]; node = row["node"]; t = row["type"]
        model_nodes_by_type[model][t].add(node)
        node_types_lookup[model][node] = t

    # build shares for M5 (from df_funcs_wide)
    funcs_map = {}  # model -> dict shares
    for _, r in df_funcs_wide.iterrows():
        model = r["model"]
        lb = float(r.get("LoadBearing", 0.0))
        sh = float(r.get("Shear", 0.0))
        mo = float(r.get("Moment", 0.0))
        funcs_map[model] = {"LB": lb, "Shear": sh, "Moment": mo}

    # helpers: collect edges per model
    by_model_edges = {}
    by_model_zones = {}
    for m in models:
        edges = {
            "adj": set((str(s), str(o)) for (s,o) in m.pred_edges.get(P_ADJ_EL, [])),
            "int": set((str(s), str(o)) for (s,o) in m.pred_edges.get(P_INT_EL, [])),
            "part": set((str(s), str(o)) for (s,o) in m.pred_edges.items() if P_HAS_PART in lc(str(s)) or P_HAS_PART in lc(str(s))) # fallback
        }
        # hasContinuantPart robust get
        part_edges = set()
        for pl, lst in m.pred_edges.items():
            if P_HAS_PART in pl or P_HAS_PART in lc(pl):
                part_edges |= set((str(s), str(o)) for (s,o) in lst)
        edges["part"] = part_edges

        zones = defaultdict(set)
        for (s,o) in m.pred_edges.get(P_ADJ_ZONE, []):
            zones[str(s)].add(str(o))
        by_model_edges[m.name] = edges
        by_model_zones[m.name] = zones

    # proxy bookkeeping
    proxy_rows = []

    # count motifs per model
    rows_counts = []
    rows_density = []  # densities per 100 struct elements
    for m in models:
        name = m.name
        edges = by_model_edges[name]
        zones = by_model_zones[name]
        typeset = model_nodes_by_type[name]

        beams = typeset["Beam"]; columns = typeset["Column"]; slabs = typeset["Slab"]; walls = typeset["Wall"]
        braces = typeset["Brace"]; cores = typeset["Core"]

        # helper: check adjacency/intersection
        def hard_adj(u,v):
            return (u,v) in edges["adj"] or (v,u) in edges["adj"] or (u,v) in edges["int"] or (v,u) in edges["int"]

        def share_zone(u,v):
            return len(zones.get(u,set()) & zones.get(v,set()))>0

        def part_degree(u):
            # out-degree wrt hasContinuantPart
            return sum(1 for (s,o) in edges["part"] if s==u)

        # ---- M2_frameNode (Beam–Column)
        cnt_m2_hard = 0; cnt_m2_proxy = 0
        for b in beams:
            for c in columns:
                if hard_adj(b,c):
                    cnt_m2_hard += 1
                    proxy_rows.append({"model": name, "motif":"M2_frameNode", "u": b, "v": c, "evidence":"hard", "reason":"adjacent/intersect"})
                else:
                    if share_zone(b,c):
                        # proxy v2: both have any adjacency or any zone co-membership already checked; we add weak condition “has any neighbor via adj edges”
                        has_b_nei = any(b==x or b==y for (x,y) in edges["adj"])
                        has_c_nei = any(c==x or c==y for (x,y) in edges["adj"])
                        if has_b_nei or has_c_nei or allow_type_only_proxy:
                            cnt_m2_proxy += 1
                            proxy_rows.append({"model": name, "motif":"M2_frameNode", "u": b, "v": c, "evidence":"proxy", "reason":"shareZone(+adj-any or type-only)"})

        # ---- M3_wallSlab
        cnt_m3_hard = 0; cnt_m3_proxy = 0
        for w in walls:
            for s in slabs:
                if hard_adj(w,s):
                    cnt_m3_hard += 1
                    proxy_rows.append({"model": name, "motif":"M3_wallSlab", "u": w, "v": s, "evidence":"hard", "reason":"adjacent/intersect"})
                else:
                    if share_zone(w,s):
                        # proxy v2: both have material parts (hasContinuantPart) or allow_type_only_proxy
                        if part_degree(w)>=1 and part_degree(s)>=1:
                            cnt_m3_proxy += 1
                            proxy_rows.append({"model": name, "motif":"M3_wallSlab", "u": w, "v": s, "evidence":"proxy", "reason":"shareZone+hasPart>=1"})
                        elif allow_type_only_proxy:
                            cnt_m3_proxy += 1
                            proxy_rows.append({"model": name, "motif":"M3_wallSlab", "u": w, "v": s, "evidence":"proxy", "reason":"shareZone(type-only)"})

        # ---- M4_core (core with ≥2 slab neighbors hard/proxy)
        cnt_m4 = 0
        for c in cores:
            n = 0
            for s in slabs:
                if hard_adj(c,s) or (share_zone(c,s) and (part_degree(s)>=1 or allow_type_only_proxy)):
                    n += 1
            if n >= 2:
                cnt_m4 += 1

        # ---- M2b_braceNode (brace near frame members)
        cnt_m2b_hard = 0; cnt_m2b_proxy = 0
        for r in braces:
            # brace with beam or column
            for x in beams | columns:
                if hard_adj(r,x):
                    cnt_m2b_hard += 1
                    proxy_rows.append({"model": name, "motif":"M2b_braceNode", "u": r, "v": x, "evidence":"hard", "reason":"adjacent/intersect"})
                else:
                    if share_zone(r,x):
                        cnt_m2b_proxy += 1
                        proxy_rows.append({"model": name, "motif":"M2b_braceNode", "u": r, "v": x, "evidence":"proxy", "reason":"shareZone"})

        # penalty apply
        cnt_m2 = cnt_m2_hard + proxy_penalty * cnt_m2_proxy
        cnt_m3 = cnt_m3_hard + proxy_penalty * cnt_m3_proxy
        cnt_m2b= cnt_m2b_hard + proxy_penalty * cnt_m2b_proxy

        # counts row
        rows_counts.append({
            "model": name,
            "M2_frameNode": cnt_m2,
            "M3_wallSlab": cnt_m3,
            "M4_core": cnt_m4,
            "M2b_braceNode": cnt_m2b,
            "M5_LB": funcs_map.get(name,{}).get("LB",0.0),
            "M5_Shear": funcs_map.get(name,{}).get("Shear",0.0),
            "M5_Moment": funcs_map.get(name,{}).get("Moment",0.0),
            "M3_hard": cnt_m3_hard, "M3_proxy": cnt_m3_proxy,
            "M2_hard": cnt_m2_hard, "M2_proxy": cnt_m2_proxy,
            "M2b_hard":cnt_m2b_hard,"M2b_proxy":cnt_m2b_proxy
        })

        # densities per 100 structural elements
        n_struct = len(typeset["Beam"]|typeset["Column"]|typeset["Slab"]|typeset["Wall"]|typeset["Brace"]|typeset["Core"]|typeset["Foundation"])
        scale = 100.0 / max(1, n_struct)
        rows_density.append({
            "model": name,
            "M2_frameNode": cnt_m2*scale,
            "M3_wallSlab": cnt_m3*scale,
            "M4_core": cnt_m4*scale,
            "M2b_braceNode": cnt_m2b*scale
        })

    df_counts = pd.DataFrame(rows_counts).fillna(0.0)
    df_density = pd.DataFrame(rows_density).fillna(0.0)

    # shares for count-based motifs (M2/M3/M4/M2b)
    cols_counts = ["M2_frameNode","M3_wallSlab","M4_core","M2b_braceNode"]
    shares_rows = []
    for _, r in df_counts.iterrows():
        model = r["model"]
        x = np.array([float(r[c]) for c in cols_counts], dtype=float)
        s = x.sum()
        if s<=0: sc = np.zeros_like(x)
        else: sc = x / s
        # append M5 shares (as-is)
        m5 = np.array([float(r["M5_LB"]), float(r["M5_Shear"]), float(r["M5_Moment"])], dtype=float)
        shares_rows.append({
            "model": model,
            "M2_frameNode": sc[0],
            "M3_wallSlab": sc[1],
            "M4_core": sc[2],
            "M2b_braceNode": sc[3],
            "M5_LB": m5[0], "M5_Shear": m5[1], "M5_Moment": m5[2]
        })
    df_shares = pd.DataFrame(shares_rows)

    # debug proxy edges
    df_proxy = pd.DataFrame(proxy_rows) if proxy_rows else pd.DataFrame(columns=["model","motif","u","v","evidence","reason"])

    # save debug if requested by caller
    if out_dir:
        df_proxy.to_csv(os.path.join(out_dir, "motif_proxy_edges.csv"), index=False)

    return df_counts, df_shares, df_density

# -------------------------
# S3: system scores (rule-based macro)
# -------------------------
def s3_system_scores(df_shares, df_counts, df_funcs_wide, dual_thresh=0.25):
    # components
    comp_rows = []
    rows = []
    for _, r in df_shares.iterrows():
        model = r["model"]
        m2 = float(r["M2_frameNode"])
        m3 = float(r["M3_wallSlab"])
        m4 = float(r["M4_core"])
        m2b= float(r["M2b_braceNode"])
        lb = float(r.get("M5_LB",0.0)); sh = float(r.get("M5_Shear",0.0)); mo = float(r.get("M5_Moment",0.0))

        # frame ≈ M2 + Moment
        frame = 0.6*m2 + 0.4*mo
        # wall ≈ M3 + (Shear/LB)
        wall = 0.7*m3 + 0.3*((sh + lb)/2.0)
        # braced ≈ M2b
        braced = min(1.0, m2b)

        # dual if both above threshold
        if frame >= dual_thresh and wall >= dual_thresh:
            dual = min(1.0, 0.5*(frame+wall))
        else:
            dual = 0.0

        rows.append({"model": model, "frame": frame, "wall": wall, "dual": dual, "braced": braced})
        comp_rows.append({
            "model": model,
            "frame_m2": m2, "frame_moment": mo,
            "wall_m3": m3, "wall_role": 0.5*(sh+lb),
            "braced_m2b": m2b
        })
    df_sys = pd.DataFrame(rows)
    df_comp = pd.DataFrame(comp_rows)
    # clip [0,1]
    for c in ["frame","wall","dual","braced"]:
        df_sys[c] = df_sys[c].clip(lower=0.0, upper=1.0)
    return df_sys, df_comp

# -------------------------
# S4: structural similarity
# -------------------------
def pairwise_matrix(models, df_sys, df_shares, alpha_m5=0.40, w_motif=0.5, w_system=0.5):
    # build motif vectors (mix of count-shares and M5)
    motif_cols_counts = ["M2_frameNode","M3_wallSlab","M4_core","M2b_braceNode"]
    m5_cols = ["M5_LB","M5_Shear","M5_Moment"]
    order = [c for c in motif_cols_counts] + [c for c in m5_cols]

    motif_vecs = {}
    for _, r in df_shares.iterrows():
        model = r["model"]
        a = np.array([float(r[c]) for c in motif_cols_counts], dtype=float)
        b = np.array([float(r[c]) for c in m5_cols], dtype=float)
        v = np.concatenate([(1.0-alpha_m5)*a, alpha_m5*b], axis=0)
        s = v.sum()
        if s>0: v = v / s
        motif_vecs[model] = v

    # build system vectors
    sys_cols = ["frame","wall","dual","braced"]
    sys_vecs = {}
    for _, r in df_sys.iterrows():
        model = r["model"]
        v = np.array([float(r[c]) for c in sys_cols], dtype=float)
        s = v.sum()
        if s>0: v = v/s
        sys_vecs[model] = v

    models_list = list(motif_vecs.keys())
    n = len(models_list)
    M = np.zeros((n,n), dtype=float)

    for i in range(n):
        for j in range(n):
            mi = motif_vecs[models_list[i]]; mj = motif_vecs[models_list[j]]
            si = sys_vecs[models_list[i]];   sj = sys_vecs[models_list[j]]
            smotif = mix_sim(mi, mj)
            ssys   = mix_sim(si, sj)
            M[i,j] = w_motif*smotif + w_system*ssys

    df_mat = pd.DataFrame(M, index=models_list, columns=models_list)
    # pairs
    rows = []
    for i in range(n):
        for j in range(i+1, n):
            rows.append({
                "A": models_list[i], "B": models_list[j],
                "similarity": float(df_mat.iloc[i,j])
            })
    df_pairs = pd.DataFrame(rows).sort_values("similarity", ascending=False)
    return df_mat, df_pairs

# -------------------------
# Write helpers
# -------------------------
def write_csvs(out_dir, **dfs):
    for name, df in dfs.items():
        df.to_csv(os.path.join(out_dir, f"{name}.csv"), index=False)

# -------------------------
# Main
# -------------------------
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
    ap.add_argument("--proxy-penalty", type=float, default=0.50)
    ap.add_argument("--allow-type-only-proxy", action="store_true")
    ap.add_argument("--func-all", action="store_true")
    ap.add_argument("--emit-debug", action="store_true")
    args = ap.parse_args()

    models = load_models(args.input_dir, args.pattern)
    if not models:
        print("[ERR] No RDF files matched.")
        return

    print("[LOAD]")
    for m in models: print("  ", m.name)

    out_dir = os.path.join(args.out_root, args.out_name)
    ensure_dir(out_dir)

    # S1
    df_types, df_types_hist, df_funcs, df_funcs_hist, df_funcs_wide, df_av, df_hits, df_unknown = s1_inventory(
        models, func_all=args.func_all
    )
    # persist S1
    df_types_hist.to_csv(os.path.join(out_dir, "struct_types_histogram.csv"), index=False)
    df_funcs_hist.to_csv(os.path.join(out_dir, "struct_functions_histogram.csv"), index=False)
    df_funcs_wide.to_csv(os.path.join(out_dir, "struct_functions_shares_wide.csv"), index=False)
    df_av.to_csv(os.path.join(out_dir, "struct_data_availability.csv"), index=False)
    if args.emit_debug:
        df_hits.to_csv(os.path.join(out_dir, "type_mapping_hits.csv"), index=False)
        df_unknown.to_csv(os.path.join(out_dir, "type_mapping_unknown.csv"), index=False)

    # S2
    df_counts, df_shares, df_density = s2_motifs(
        models, df_types, df_funcs_wide,
        proxy_penalty=args.proxy_penalty,
        allow_type_only_proxy=args.allow_type_only_proxy,
        out_dir=out_dir if args.emit_debug else None
    )
    df_counts.to_csv(os.path.join(out_dir, "struct_motif_counts.csv"), index=False)
    df_shares.to_csv(os.path.join(out_dir, "struct_motif_shares.csv"), index=False)
    df_density.to_csv(os.path.join(out_dir, "struct_motif_densities_per100.csv"), index=False)

    # S3
    df_sys, df_comp = s3_system_scores(df_shares, df_counts, df_funcs_wide, dual_thresh=args.dual_thresh)
    df_sys.to_csv(os.path.join(out_dir, "struct_system_scores.csv"), index=False)
    df_comp.to_csv(os.path.join(out_dir, "struct_score_components.csv"), index=False)

    # S4
    df_mat, df_pairs = pairwise_matrix(models, df_sys, df_shares, alpha_m5=args.alpha_m5, w_motif=args.w_motif, w_system=args.w_system)
    df_mat.to_csv(os.path.join(out_dir, "struct_similarity_matrix.csv"))
    df_pairs.to_csv(os.path.join(out_dir, "pairwise_structural_summary.csv"), index=False)

    # weights
    weights = {
        "dual_thresh": args.dual_thresh,
        "w_motif": args.w_motif,
        "w_system": args.w_system,
        "alpha_m5": args.alpha_m5,
        "proxy_penalty": args.proxy_penalty,
        "allow_type_only_proxy": bool(args.allow_type_only_proxy),
        "func_all": bool(args.func_all)
    }
    with open(os.path.join(out_dir, "weights_used.json"), "w", encoding="utf-8") as f:
        json.dump(weights, f, indent=2)

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
    if args.emit_debug:
        print(" - motif_proxy_edges.csv")
        print(" - type_mapping_hits.csv")
        print(" - type_mapping_unknown.csv")

if __name__ == "__main__":
    main()
