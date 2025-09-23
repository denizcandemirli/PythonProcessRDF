# -*- coding: utf-8 -*-
"""
Subgraph/Motif-based structural similarity for BIM Design Graph RDFs.

Outputs go to:  repro_pack/output/05 - Subgraph_Similarity/

What it does
------------
1) Parse RDF (rdflib) -> build directed NetworkX graph per model
   - nodes: attributes => {"types": set([...]), "category": one of {zone, element, function, quality, part, other}}
   - edges: attributes => {"pred": "prefix:local"}  (subject -> object)
   - For symmetric predicates (adjacentElement, adjacentZone, intersectingElement) we add both directions.

2) Define a small motif library (8 motifs):
   M1: Z --adjacentZone--> Z
   M2: E --adjacentElement--> E
   M3: E --intersectingElement--> E
   M4: E --BFO_0000178(hasContinuantPart)--> P
   M5: E --core:hasFunction--> F
   M6: E --core:hasQuality-->  Q
   M7: E1 --adjacentElement--> E2  and  E1 --core:hasFunction--> F
   M8: E1 --adjacentElement--> E2  and  E1 --core:hasQuality-->  Q

   (Z=zone, E=element, F=function, Q=quality, P=part)

3) Count subgraph-isomorphisms per motif & per model (NetworkX DiGraphMatcher).
   - Node match: category equality
   - Edge match: predicate (pred) equality
   - Symmetry-safe dedup: canonicalize matched node-id tuple

4) Build a model x motif count matrix and save motif-level instances (top-K examples).

5) Compute pairwise similarities:
   - Cosine similarity on raw motif-count vectors
   - Overlap score per motif: min(countA, countB)/max(countA, countB), weighted average
   Save as CSV matrices.

Usage
-----
# From project root (after activating venv)
# pip install networkx rdflib pandas numpy
python subgraph_similarity.py ^
  --input-dir "." ^
  --pattern "*_DG.rdf" ^
  --out-root ".\\repro_pack\\output" ^
  --out-name "05 - Subgraph_Similarity" ^
  --max-samples 300

Author: you + ChatGPT
"""

import os, glob, json, math, argparse, itertools, collections
from typing import Dict, List, Tuple, Set
import pandas as pd
import numpy as np

# deps
try:
    import rdflib
except Exception as e:
    raise SystemExit("rdflib is required. pip install rdflib\n"+str(e))

import networkx as nx
from networkx.algorithms import isomorphism as iso


# -------------------------------
# Helpers
# -------------------------------

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def norm_path(p: str) -> str:
    return os.path.normpath(p)

def qname(uri: str, nsmap: Dict[str, str]) -> str:
    for pre, ns in nsmap.items():
        if uri.startswith(ns):
            return f"{pre}:{uri[len(ns):]}"
    return uri

def load_rdf(path: str):
    g = rdflib.Graph()
    try:
        g.parse(path, format="xml")
    except Exception:
        g.parse(path)
    # namespace map
    nsmap = {}
    for pre, ns in g.namespace_manager.namespaces():
        nsmap[str(pre)] = str(ns)
    return g, nsmap

def graph_from_rdf(path: str) -> Tuple[nx.DiGraph, Dict[str,str]]:
    """Return (DiGraph, nsmap) from RDF/XML."""
    g_rdf, nsmap = load_rdf(path)
    DG = nx.DiGraph()

    # collect node types
    subj_types: Dict[str, Set[str]] = collections.defaultdict(set)
    for s, _, o in g_rdf.triples((None, rdflib.RDF.type, None)):
        subj_types[str(s)].add(qname(str(o), nsmap))

    # add nodes
    for s in set(map(str, g_rdf.subjects())):
        DG.add_node(s, types=subj_types.get(s, set()))

    # add edges
    for s,p,o in g_rdf:
        s = str(s); p = str(p); pred = qname(p, nsmap)
        if isinstance(o, rdflib.term.Literal):
            continue
        o = str(o)
        DG.add_edge(s, o, pred=pred)

        # For symmetric predicates, add reverse direction as well
        if pred.endswith("adjacentElement") or pred.endswith("adjacentZone") or pred.endswith("intersectingElement"):
            DG.add_edge(o, s, pred=pred)

    # node categories (heuristic)
    #  - we infer 'part' also from being target of BFO_0000178
    #  - we keep lightweight, motif matching uses 'category' only
    incoming_from_pred = collections.defaultdict(set)
    for u, v, data in DG.edges(data=True):
        incoming_from_pred[v].add(data.get("pred", ""))

    for n, d in DG.nodes(data=True):
        tps = d.get("types", set())
        tps_lc = {t.lower() for t in tps}

        cat = "other"
        # zones (bot:Space or contains :Space)
        if any(":space" in t or t.endswith(":Space") for t in tps_lc):
            cat = "zone"
        # elements (bd:* or contains typical construction element tokens)
        if any(t.startswith("bd:") or any(tok in t for tok in [":wall",":slab",":door",":window",":column",":beam",":floor",":roof"]) for t in tps_lc):
            cat = "element"
        # function / quality
        if any(t.endswith("function") or t.endswith(":Function") for t in tps_lc):
            cat = "function"
        if any(t.endswith("quality") or t.endswith(":Quality") for t in tps_lc):
            cat = "quality"
        # part by incoming BFO_0000178
        if "BFO_0000178" in "|".join(incoming_from_pred.get(n, set())):
            cat = "part"

        DG.nodes[n]["category"] = cat

    return DG, nsmap


# -------------------------------
# Motif Library
# -------------------------------

def build_motif_library() -> Dict[str, nx.DiGraph]:
    """Define small directed motif patterns with node categories and edge predicates."""
    def dg():
        return nx.DiGraph()

    motifs: Dict[str, nx.DiGraph] = {}

    # helper to add nodes with category labels
    def add_nodes(G, labels):
        # labels: dict like {"a":"zone","b":"zone"}
        for k,v in labels.items():
            G.add_node(k, category=v)

    # M1: Z -adjacentZone- Z
    G = dg(); add_nodes(G, {"a":"zone","b":"zone"}); G.add_edge("a","b", pred="bot:adjacentZone"); 
    motifs["M1_adjacentZone_ZZ"] = G

    # M2: E -adjacentElement- E
    G = dg(); add_nodes(G, {"a":"element","b":"element"}); G.add_edge("a","b", pred="bot:adjacentElement");
    motifs["M2_adjacentElement_EE"] = G

    # M3: E -intersectingElement- E
    G = dg(); add_nodes(G, {"a":"element","b":"element"}); G.add_edge("a","b", pred="bot:intersectingElement");
    motifs["M3_intersectingElement_EE"] = G

    # M4: E -hasContinuantPart- P (BFO_0000178)
    G = dg(); add_nodes(G, {"a":"element","p":"part"}); G.add_edge("a","p", pred="BFO_0000178");
    motifs["M4_hasContinuantPart_EP"] = G

    # M5: E -core:hasFunction- F
    G = dg(); add_nodes(G, {"a":"element","f":"function"}); G.add_edge("a","f", pred="core:hasFunction");
    motifs["M5_hasFunction_EF"] = G

    # M6: E -core:hasQuality- Q
    G = dg(); add_nodes(G, {"a":"element","q":"quality"}); G.add_edge("a","q", pred="core:hasQuality");
    motifs["M6_hasQuality_EQ"] = G

    # M7: E1 -adjacentElement- E2  and  E1 -hasFunction- F
    G = dg(); add_nodes(G, {"a":"element","b":"element","f":"function"})
    G.add_edge("a","b", pred="bot:adjacentElement")
    G.add_edge("a","f", pred="core:hasFunction")
    motifs["M7_adjacentElement_plus_Function"] = G

    # M8: E1 -adjacentElement- E2  and  E1 -hasQuality- Q
    G = dg(); add_nodes(G, {"a":"element","b":"element","q":"quality"})
    G.add_edge("a","b", pred="bot:adjacentElement")
    G.add_edge("a","q", pred="core:hasQuality")
    motifs["M8_adjacentElement_plus_Quality"] = G

    return motifs


# -------------------------------
# Matching & Counting
# -------------------------------

def node_matcher(n1_attrs, n2_attrs):
    return n1_attrs.get("category") == n2_attrs.get("category")

def edge_matcher(e1_attrs, e2_attrs):
    return e1_attrs.get("pred") == e2_attrs.get("pred")

def unique_mapping_signature(mapping: Dict[str, str]) -> Tuple[str,...]:
    """Canonical signature to deduplicate symmetric matches within a model."""
    return tuple(sorted(mapping.values()))

def count_motif_instances(DG: nx.DiGraph, motif: nx.DiGraph, max_samples: int = 100000) -> Tuple[int, List[Dict[str,str]]]:
    """Return (count, sample_mappings)."""
    GM = iso.DiGraphMatcher(DG, motif, node_match=node_matcher, edge_match=edge_matcher)
    seen = set()
    samples = []
    n = 0
    for mapping in GM.subgraph_isomorphisms_iter():
        sig = unique_mapping_signature(mapping)
        if sig in seen:
            continue
        seen.add(sig)
        n += 1
        if len(samples) < max_samples:
            samples.append(mapping)
    return n, samples


# -------------------------------
# Similarities
# -------------------------------

def cosine_sim_matrix(M: np.ndarray) -> np.ndarray:
    # rows = models, cols = motif counts
    # cosine = (A.B) / (||A||*||B||)
    norms = np.linalg.norm(M, axis=1, keepdims=True)
    norms[norms==0] = 1.0
    X = M / norms
    return np.clip(X @ X.T, 0.0, 1.0)

def overlap_weighted_matrix(M: np.ndarray, weights: np.ndarray) -> np.ndarray:
    # s_ij = sum_w min(a_k, b_k)/max(a_k, b_k) / sum_w  (for max>0; if both 0 => 1)
    m = M.shape[0]
    S = np.zeros((m,m), dtype=float)
    wsum = weights.sum() if weights.sum()>0 else 1.0
    for i in range(m):
        for j in range(m):
            num = 0.0
            for k in range(M.shape[1]):
                a, b = M[i,k], M[j,k]
                if a==0 and b==0:
                    contrib = 1.0
                else:
                    contrib = min(a,b)/max(a,b)
                num += weights[k]*contrib
            S[i,j] = num / wsum
    return np.clip(S, 0.0, 1.0)


# -------------------------------
# Main pipeline
# -------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", default=".", help="Folder with RDFs")
    ap.add_argument("--pattern", default="*_DG.rdf", help="Glob for model RDFs")
    ap.add_argument("--out-root", default=os.path.join(".","repro_pack","output"))
    ap.add_argument("--out-name", default="05 - Subgraph_Similarity")
    ap.add_argument("--max-samples", type=int, default=300, help="Max sample matches per motif & model to save")
    ap.add_argument("--weights-json", default="", help='Optional motif weights as JSON dict {"M1":1.0,...}')
    args = ap.parse_args()

    input_dir = norm_path(args.input_dir)
    out_dir = norm_path(os.path.join(args.out_root, args.out_name))
    ensure_dir(out_dir)

    # 1) Load RDFs -> graphs
    rdf_paths = sorted(glob.glob(os.path.join(input_dir, args.pattern)))
    if not rdf_paths:
        print("[WARN] No RDFs matched:", os.path.join(input_dir, args.pattern))
    models = []
    graphs = {}
    for p in rdf_paths:
        name = os.path.basename(p)
        print(f"[LOAD] {name}")
        DG, nsmap = graph_from_rdf(p)
        models.append(name)
        graphs[name] = DG

    # 2) Build motif library
    motifs = build_motif_library()
    motif_ids = list(motifs.keys())

    # Optional weights
    weights = np.ones(len(motif_ids), dtype=float)
    if args.weights_json.strip():
        try:
            wd = json.loads(args.weights_json)
            weights = np.array([float(wd.get(mid, 1.0)) for mid in motif_ids], dtype=float)
        except Exception as e:
            print("[WARN] weights-json could not be parsed; using all 1.0. Err:", e)

    # 3) Count instances per model
    rows = []
    sample_rows = []
    for m in models:
        DG = graphs[m]
        for mid in motif_ids:
            n, samples = count_motif_instances(DG, motifs[mid], max_samples=args.max_samples)
            rows.append({"model": m, "motif": mid, "count": n})
            # save sample mappings (node ids)
            for idx, mp in enumerate(samples):
                if idx >= args.max_samples: break
                # invert mapping: motif node -> model node
                # store as comma-joined "nodeId"
                sample_rows.append({
                    "model": m, "motif": mid, **{f"{k}": str(v) for k,v in mp.items()}
                })

    df_counts = pd.DataFrame(rows).pivot_table(index="model", columns="motif", values="count", fill_value=0).reset_index()
    df_counts.to_csv(os.path.join(out_dir, "motif_counts_all.csv"), index=False)

    df_samples = pd.DataFrame(sample_rows)
    df_samples.to_csv(os.path.join(out_dir, "motif_match_samples.csv"), index=False)

    # persist motif library for transparency
    lib_rows = []
    for mid, G in motifs.items():
        lib_rows.append({
            "motif_id": mid,
            "nodes": ";".join([f"{n}:{G.nodes[n].get('category')}" for n in G.nodes()]),
            "edges": ";".join([f"{u}->{v}:{G.edges[u,v]['pred']}" for u,v in G.edges()]),
            "size_nodes": G.number_of_nodes(),
            "size_edges": G.number_of_edges(),
        })
    pd.DataFrame(lib_rows).to_csv(os.path.join(out_dir, "motif_library.csv"), index=False)

    # 4) Pairwise similarities
    # matrix M (models x motifs)
    M = df_counts.drop(columns=["model"]).values.astype(float)
    model_names = df_counts["model"].tolist()

    # 4a) Cosine
    S_cos = cosine_sim_matrix(M)
    pd.DataFrame(S_cos, index=model_names, columns=model_names)\
      .to_csv(os.path.join(out_dir, "similarity_motif_cosine.csv"))

    # 4b) Overlap weighted
    S_ov = overlap_weighted_matrix(M, weights)
    pd.DataFrame(S_ov, index=model_names, columns=model_names)\
      .to_csv(os.path.join(out_dir, "similarity_motif_overlap_weighted.csv"))

    # 4c) Final (default α=0.5)
    alpha = 0.5
    S_final = alpha*S_cos + (1-alpha)*S_ov
    pd.DataFrame(S_final, index=model_names, columns=model_names)\
      .to_csv(os.path.join(out_dir, "similarity_structural_final.csv"))

    # Also export a long-form pairwise table for quick reading
    rows_pairs = []
    for i,ai in enumerate(model_names):
        for j,aj in enumerate(model_names):
            if i >= j:  # upper triangle only
                continue
            rows_pairs.append({
                "model_A": ai, "model_B": aj,
                "cosine": S_cos[i,j],
                "overlap_weighted": S_ov[i,j],
                "final": S_final[i,j],
            })
    pd.DataFrame(rows_pairs).sort_values("final", ascending=False)\
      .to_csv(os.path.join(out_dir, "pairwise_structural_summary.csv"), index=False)

    print("\n[OK] Saved outputs under:", out_dir)
    print(" - motif_library.csv")
    print(" - motif_counts_all.csv")
    print(" - motif_match_samples.csv (node ids per motif)")
    print(" - similarity_motif_cosine.csv")
    print(" - similarity_motif_overlap_weighted.csv")
    print(" - similarity_structural_final.csv")
    print(" - pairwise_structural_summary.csv")


if __name__ == "__main__":
    main()
