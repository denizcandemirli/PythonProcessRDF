#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File: structural_similarity.py
Purpose: Compute motif-based structural similarity between RDF Design Graphs.
Input: RDF model directory path, merged ontology (0000_Merged.rdf)
Output: struct_similarity_matrix.csv, struct_motif_counts.csv, radar plots

This script combines the best features from:
- structural_extension_v25p2.py (latest structural analysis)
- subgraph_similarity_v2.py (canonical motif detection)

Author: Deniz Demirli
Supervisor: Dr. Chao Li (TUM)
Version: 2025.10
License: SPDX-License-Identifier: CC-BY-4.0
"""

__version__ = "2025.10"
__author__ = "Deniz Demirli"
__supervisor__ = "Dr. Chao Li (TUM)"

import os
import re
import json
import glob
import argparse
import math
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Set
import numpy as np
import pandas as pd

# Dependencies
try:
    import rdflib
    from rdflib import Graph, RDF, RDFS
except ImportError:
    raise SystemExit("rdflib is required. pip install rdflib")

try:
    import networkx as nx
    from networkx.algorithms import isomorphism as iso
except ImportError:
    raise SystemExit("networkx is required. pip install networkx")

# ---------------------------
# Helper Functions
# ---------------------------

def localname(x: str) -> str:
    """Extract local name from URI."""
    s = str(x)
    if "#" in s:
        s = s.split("#")[-1]
    if "/" in s:
        s = s.split("/")[-1]
    return s

def tokset_from_labels(labels):
    """Extract tokens from labels for pattern matching."""
    toks = set()
    for t in labels:
        t2 = re.sub(r"[^A-Za-z0-9]+", " ", str(t)).strip().lower()
        if t2:
            toks |= set(t2.split())
    return toks

def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    da = np.linalg.norm(a)
    db = np.linalg.norm(b)
    if da == 0 or db == 0:
        return 0.0
    return float(np.dot(a, b) / (da * db))

def ensure_dir(path: str):
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)

def norm_path(p: str) -> str:
    """Normalize path."""
    return os.path.normpath(p)

def qname(uri: str, nsmap: Dict[str, str]) -> str:
    """Convert URI to qualified name."""
    for pre, ns in nsmap.items():
        if uri.startswith(ns):
            return f"{pre}:{uri[len(ns):]}"
    return uri

def load_rdf(path: str):
    """Load RDF graph and return graph with namespace mapping."""
    g = rdflib.Graph()
    try:
        g.parse(path, format="xml")
    except Exception:
        g.parse(path)
    nsmap = {}
    for pre, ns in g.namespace_manager.namespaces():
        nsmap[str(pre)] = str(ns)
    return g, nsmap

def contains_any(s: str, toks: List[str]) -> bool:
    """Check if string contains any of the tokens."""
    s = s.lower()
    return any(t in s for t in toks)

# ---------------------------
# Canonicalization
# ---------------------------

def canon_pred(qname_str: str) -> str:
    """
    Canonicalize variety of predicate names:
      ...:AdjacentComponent / ...:adjacentElement / ...:isAdjacentTo  -> adjacentelement
      ...:adjacentZone                                             -> adjacentzone
      ...:BFO_0000178                                              -> bfo_0000178
      ...:hasFunction                                              -> hasfunction
      ...:hasQuality                                               -> hasquality
      ...:intersects* / ...:overlaps* / ...:crosses*               -> intersectingelement
      others -> local name lower-case
    """
    local = qname_str.split(":", 1)[-1]
    s = local.lower()

    if "bfo_0000178" in s:
        return "bfo_0000178"
    if "hasfunction" in s or ("function" in s and "has" in s):
        return "hasfunction"
    if "hasquality" in s or ("quality" in s and "has" in s):
        return "hasquality"
    if "adjacent" in s and "zone" in s:
        return "adjacentzone"
    if ("adjacent" in s and any(tok in s for tok in ["element","component","to"])) or "adjoin" in s or "touch" in s:
        return "adjacentelement"
    if any(tok in s for tok in ["intersect","overlap","cross","cut","penetrat"]):
        return "intersectingelement"
    return s  # fallback

# ---------------------------
# Type and Function Dictionaries
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

# Motif key names (for reports/heatmaps)
MOTIF_KEYS = ["M2_frameNode", "M3_wallSlab", "M4_core", "M2b_braceNode"]
M5_KEYS    = ["M5_LB", "M5_Shear", "M5_Moment", "M5_Bracing"]

def compile_patterns(d: dict):
    """Compile regex patterns."""
    return {k: re.compile(v, re.I) for k, v in d.items()}

TYPE_RX = compile_patterns(TYPE_PATTERNS)
FUNC_RX = compile_patterns(FUNC_PATTERNS)

def classify_node_types(types_set, labels_set):
    """Classify each node into (Beam/Column/Slab/Wall/Brace/Core/...) categories."""
    namebag = " ".join(sorted([t.lower() for t in types_set] + list(tokset_from_labels(labels_set))))
    cats = set()
    for cat, rx in TYPE_RX.items():
        if rx.search(namebag):
            cats.add(cat)
    return cats

def classify_functions(func_objs, labels_set, func_all=False):
    """
    Function categories: LB / Shear / Moment / Bracing
    - func_objs: hasFunction/hasQuality/hasRole target localnames
    - labels_set: node label/annotation (for regex)
    - func_all=True then derive from label keys too
    """
    bag = " ".join([t.lower() for t in func_objs] + (list(tokset_from_labels(labels_set)) if func_all else []))
    cats = set()
    for cat, rx in FUNC_RX.items():
        if rx.search(bag):
            cats.add(cat)
    return cats

# ---------------------------
# RDF Model Reading
# ---------------------------

class Model:
    """RDF model wrapper with indexing."""
    def __init__(self, name, graph: Graph):
        self.name = name
        self.g = graph
        self.types = defaultdict(set)     # node -> {type localnames}
        self.labels = defaultdict(set)    # node -> {labels}
        self.edges_by_pred = defaultdict(list)  # pred_local -> [(s,o),...]

        self._index()

    def _index(self):
        """Index the RDF graph."""
        for s, p, o in self.g.triples((None, RDF.type, None)):
            self.types[s].add(localname(o))
        for s, p, o in self.g.triples((None, RDFS.label, None)):
            self.labels[s].add(str(o))

        # collect all predicates (by local name)
        for s, p, o in self.g:
            pred = localname(p).lower()
            if pred in STRONG_TOPO or pred in WEAK_TOPO or pred in {
                "hascontinuantpart", "haspropercontinuantpart",
                "hasfunction", "hasquality", "hasrole", "hasstructuralfunction", "hasstructuralrole"
            }:
                self.edges_by_pred[pred].append((s, o))

def read_models(input_dir: str, pattern: str):
    """Read RDF models from directory."""
    paths = sorted(glob.glob(os.path.join(input_dir, pattern)))
    models = []
    for p in paths:
        name = os.path.basename(p)
        try:
            g = Graph()
            # most RDF/XML – rdflib format auto-detection
            g.parse(p)
            models.append(Model(name, g))
        except Exception as e:
            print(f"[WARN] Parse error: {name}: {e}")
    if not models:
        raise RuntimeError("No RDF files loaded; check input-dir/pattern.")
    return models

# ---------------------------
# S1 — Inventory (types and functions)
# ---------------------------

def s1_inventories(models, func_all=False):
    """Generate type and function inventories."""
    rows_types = []
    rows_funcs = []
    func_shares_rows = []   # model × LB/Shear/Moment/Bracing shares
    type_hits = Counter()
    type_unknown = Counter()

    data_av_rows = []       # model-level data signal (strong/weak/proxy)

    model_node2type = {}    # for later motif use

    for m in models:
        # type/function extraction
        node2cats = defaultdict(set)
        node2funcobjs = defaultdict(set)

        # collect object localnames from hasFunction-like edges
        for pred in ["hasfunction", "hasquality", "hasrole", "hasstructuralfunction", "hasstructuralrole"]:
            for s, o in m.edges_by_pred.get(pred, []):
                node2funcobjs[s].add(localname(o))

        # classify by node
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
            # write functions to node (for later share calculation)
            node2funcobjs[n] = funs

        model_node2type[m.name] = node2cats

        # Type histogram
        type_counts = Counter()
        for cats in node2cats.values():
            for c in cats:
                type_counts[c] += 1
        for cat, cnt in sorted(type_counts.items()):
            rows_types.append({"model": m.name, "type": cat, "count": int(cnt)})

        # Function histogram
        func_counts = Counter()
        for funs in node2funcobjs.values():
            for f in funs:
                func_counts[f] += 1
        for fcat in ["LB", "Shear", "Moment", "Bracing"]:
            rows_funcs.append({"model": m.name, "function": fcat, "count": int(func_counts.get(fcat, 0))})

        # Function shares (normalized by structural element count in model)
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

        # Data signal (for S2/S3 info): strong/weak topo present?
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

    # mapping lists
    hits_rows = [{"token": t, "count": c} for t, c in type_hits.items()]
    unk_rows  = [{"token": t, "count": c} for t, c in type_unknown.items()]
    df_type_hits = pd.DataFrame(hits_rows).sort_values("count", ascending=False)
    df_type_unknown = pd.DataFrame(unk_rows).sort_values("count", ascending=False)

    return df_types, df_funcs, df_func_wide, df_av, model_node2type, df_type_hits, df_type_unknown

# ---------------------------
# S2 — Motif counts (with proxy/penalty support)
# ---------------------------

def s2_motifs(models, model_node2type, allow_type_only, proxy_penalty, weak_topo_penalty):
    """Generate motif counts with proxy and penalty support."""
    rows_counts = []
    rows_dens   = []
    proxy_rows  = []

    def count_pairs(m, Aset, Bset, strong_first=True):
        """Count unique A–B pairs connected by strong/weak topology and which channel was used."""
        pairs = set()
        used_strong = False
        used_weak = False

        # strong topology
        for pred in STRONG_TOPO:
            for s, o in m.edges_by_pred.get(pred, []):
                if s in Aset and o in Bset or s in Bset and o in Aset:
                    key = frozenset([s, o])
                    pairs.add(key)
                    used_strong = True

        # weak topology
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

        # M2: frame node (beam–column proximity)
        c2, st2, wk2 = count_pairs(m, E_beam, E_column)
        used_proxy_2 = False
        if c2 == 0 and allow_type_only and (E_beam and E_column):
            c2 = min(len(E_beam), len(E_column))  # rough proxy
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

        # M4: core surrounded by slab composition (by proximity)
        # practical: count cores with at least 2 slab neighbors
        core_good = 0
        st4 = False; wk4 = False
        if E_core and E_slab:
            # strong
            adj = set()
            for pred in STRONG_TOPO:
                for s,o in m.edges_by_pred.get(pred, []):
                    if s in E_core and o in E_slab:
                        adj.add(s); st4 = True
                    if o in E_core and s in E_slab:
                        adj.add(o); st4 = True
            # weak
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

        # penalties
        # proxy penalty applied motif-wise
        def penalize(d, used_proxy, used_strong, used_weak):
            v = d
            if used_proxy:
                v *= (1.0 - proxy_penalty)
            # weak-topology-only penalty
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
# S3 — System scores (density/role-based)
# ---------------------------

def s3_system_scores(models, df_dens, df_func_wide, dual_thresh):
    """Generate system scores based on density and role."""
    # dens: per-100 scale
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

        # dual: threshold rule
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
# S4 — Structural similarity (motif + system vector cosine fusion)
# ---------------------------

def s4_struct_similarity(df_dens, df_func_wide, df_scores, w_motif, w_system, alpha_m5):
    """Generate structural similarity using motif + system vector cosine fusion."""
    # motif vector = [M2, M3, M4, M2b, alpha*(LB,Shear,Moment,Bracing)]
    A = df_dens.set_index("model")[["M2_frameNode","M3_wallSlab","M4_core","M2b_braceNode"]].copy()
    A = A / 100.0  # scale to 0-1
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

    # system vector = [frame, wall, dual, braced]
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
# Main Function
# ---------------------------

def main():
    """Main execution function."""
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

    # 1) Load RDFs
    models = read_models(args.input_dir, args.pattern)
    print("[LOAD]")
    for m in models:
        print("  ", m.name)

    # 2) S1: type and function inventory
    df_types, df_funcs, df_func_wide, df_av, node2type, df_hits, df_unk = s1_inventories(
        models, func_all=args.func_all
    )

    # 3) S2: motif counts + penalized densities
    df_counts, df_dens, df_proxy = s2_motifs(
        models, node2type,
        allow_type_only=args.allow_type_only_proxy,
        proxy_penalty=args.proxy_penalty,
        weak_topo_penalty=args.weak_topo_penalty
    )

    # 4) S3: system scores (density/role-based)
    df_scores, df_comps = s3_system_scores(
        models, df_dens, df_func_wide, dual_thresh=args.dual_thresh
    )

    # 5) S4: structural similarity (motif + system cosine fusion)
    df_Sstruct = s4_struct_similarity(
        df_dens, df_func_wide, df_scores,
        w_motif=args.w_motif, w_system=args.w_system, alpha_m5=args.alpha_m5
    )

    # ------------ OUTPUTS ------------
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

    # For heatmap: motif shares (0–1), also add M5 shares
    # Note: M2/M3/M4/M2b densities were per-100 → convert to shares with /100
    wide_for_heat = df_dens.copy()
    for c in ["M2_frameNode","M3_wallSlab","M4_core","M2b_braceNode"]:
        wide_for_heat[c] = wide_for_heat[c] / 100.0
    # M5: directly from df_func_wide
    m5 = df_func_wide[["model","LB","Shear","Moment","Bracing"]].copy()
    m5.rename(columns={
        "LB":"M5_LB","Shear":"M5_Shear","Moment":"M5_Moment","Bracing":"M5_Bracing"
    }, inplace=True)
    df_motif_shares = pd.merge(wide_for_heat, m5, on="model", how="left")
    df_motif_shares = df_motif_shares[["model"] + MOTIF_KEYS + M5_KEYS]
    df_motif_shares.to_csv(os.path.join(out_dir, "struct_motif_shares.csv"), index=False)

    # long form (model,motif,share)
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

    # pairwise summary
    models_list = list(df_Sstruct.index)
    rows_pw = []
    for i in range(len(models_list)):
        for j in range(i+1, len(models_list)):
            a = models_list[i]; b = models_list[j]
            rows_pw.append({"A": a, "B": b, "S_struct": float(df_Sstruct.loc[a,b])})
    df_pw = pd.DataFrame(rows_pw).sort_values("S_struct", ascending=False)
    df_pw.to_csv(os.path.join(out_dir, "pairwise_structural_summary.csv"), index=False)

    # weights / config
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
        # Useful combined csv (for quick reference in reports)
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
