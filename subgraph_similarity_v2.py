# -*- coding: utf-8 -*-
"""
Subgraph/Motif-based structural similarity (v2.3)
- Canonicalized predicates
- Enriched node categories (zone/element heuristics broadened; IFC-like names)
- ABox-focused motif subgraph (only relevant predicates, non-'other' nodes)
- NEW motif variants:
    * M2b_adjacentElement_EZ (element <-> zone adjacency)
    * M3b_intersectingElement_EZ (element <-> zone intersection)

OUTPUT (default folder):
  repro_pack/output/05b - Subgraph_Similarity_Canon/
    - motif_library.csv
    - motif_counts_all.csv
    - motif_match_samples.csv
    - similarity_motif_cosine.csv
    - similarity_motif_overlap_weighted.csv
    - similarity_structural_final.csv
    - pairwise_structural_summary.csv
  (--emit-extras)
    - debug_predicate_map.csv
    - debug_category_mix.csv
    - debug_motif_graph_pred_counts.csv
"""

import os, glob, json, argparse, collections
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
    nsmap = {}
    for pre, ns in g.namespace_manager.namespaces():
        nsmap[str(pre)] = str(ns)
    return g, nsmap

def contains_any(s: str, toks: List[str]) -> bool:
    s = s.lower()
    return any(t in s for t in toks)


# ---------- Canonicalization ----------

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


# -------------------------------
# RDF -> NetworkX DiGraph
# -------------------------------

def graph_from_rdf(path: str) -> Tuple[nx.DiGraph, Dict[str,str], Dict[str, int]]:
    """Return (DiGraph, nsmap, raw_pred_counter) from RDF."""
    g_rdf, nsmap = load_rdf(path)
    DG = nx.DiGraph()
    raw_pred_counter = collections.Counter()

    # collect node types
    subj_types: Dict[str, Set[str]] = collections.defaultdict(set)
    for s, _, o in g_rdf.triples((None, rdflib.RDF.type, None)):
        subj_types[str(s)].add(qname(str(o), nsmap))

    # add nodes (subjects + objects as node shells)
    for s in set(map(str, g_rdf.subjects())):
        DG.add_node(s, types=subj_types.get(s, set()))
    for o in set(map(str, [x for x in g_rdf.objects() if not isinstance(x, rdflib.term.Literal)])):
        if o not in DG:
            DG.add_node(o, types=set())

    # add edges (canonicalized predicates)
    for s,p,o in g_rdf:
        s = str(s)
        if isinstance(o, rdflib.term.Literal):
            continue
        o = str(o)
        raw = qname(str(p), nsmap)
        pred_key = canon_pred(raw)
        raw_pred_counter[raw] += 1

        DG.add_edge(s, o, pred=pred_key)

        # Symmetric relationships -> add reverse
        if pred_key in {"adjacentelement","adjacentzone","intersectingelement"}:
            DG.add_edge(o, s, pred=pred_key)

    # ---------- v2.3: node categories (heuristic enriched by types + in/out edges) ----------
    in_pred = collections.defaultdict(set)
    out_pred = collections.defaultdict(set)
    for u, v, data in DG.edges(data=True):
        p = data.get("pred", "")
        out_pred[u].add(p)
        in_pred[v].add(p)

    # IFC-like element keywords
    IFC_ELEMS = [
        "ifcwall","ifcslab","ifcdoor","ifcwindow","ifccolumn","ifcbeam","ifcstair","ifcroof","ifcmember",
        "ifcrailing","ifcramp","ifcfooting","ifcplate","ifccurtainwall","ifcgrid","buildingelement"
    ]

    for n, d in DG.nodes(data=True):
        tps = {t.lower() for t in d.get("types", set())}
        cat = "other"

        # part (incoming hasContinuantPart)
        if "bfo_0000178" in in_pred.get(n, set()):
            cat = "part"

        # function / quality
        if cat == "other" and ("hasfunction" in in_pred.get(n, set()) or any(t.endswith(":function") or t.endswith("function") for t in tps)):
            cat = "function"
        if cat == "other" and ("hasquality" in in_pred.get(n, set()) or any(t.endswith(":quality") or t.endswith("quality") for t in tps)):
            cat = "quality"

        # zone (type includes zone/space/room) OR adjacentzone context
        if cat == "other" and (
            any(any(tok in t for tok in ["zone",":space","space","room"]) for t in tps)
            or "adjacentzone" in in_pred.get(n, set())
            or "adjacentzone" in out_pred.get(n, set())
        ):
            cat = "zone"

        # element (broadened):
        #  - type startswith bd:/bot: (not zone/space)
        #  - type contains common building element tokens or IFC element class names
        #  - out semantic/part/adjacent/intersect OR in adjacent/intersect
        if cat == "other" and (
            any(t.startswith("bd:") or (t.startswith("bot:") and not contains_any(t, [":zone",":space"])) for t in tps)
            or any(contains_any(t, [":wall",":slab",":door",":window",":column",":beam",":floor",":roof","element"]) for t in tps)
            or any(contains_any(t, IFC_ELEMS) for t in tps)
            or any(p in out_pred.get(n, set()) for p in ["hasfunction","hasquality","bfo_0000178","adjacentelement","intersectingelement"])
            or any(p in in_pred.get(n, set())  for p in ["adjacentelement","intersectingelement"])
        ):
            cat = "element"

        DG.nodes[n]["category"] = cat

    return DG, nsmap, raw_pred_counter


# -------------------------------
# Build "motif graph" (ABox-only, relevant preds, non-'other' nodes)
# -------------------------------

ALLOWED_PREDS = {"adjacentelement","adjacentzone","intersectingelement","bfo_0000178","hasfunction","hasquality"}

def build_motif_graph(DG: nx.DiGraph) -> Tuple[nx.DiGraph, collections.Counter]:
    MG = nx.DiGraph()
    pred_use = collections.Counter()

    # keep only nodes with assigned category (!= 'other')
    keep_nodes = [n for n, d in DG.nodes(data=True) if d.get("category","other") != "other"]
    MG.add_nodes_from((n, DG.nodes[n]) for n in keep_nodes)

    for u, v, data in DG.edges(data=True):
        if u not in MG or v not in MG:
            continue
        p = data.get("pred", "")
        if p in ALLOWED_PREDS:
            MG.add_edge(u, v, pred=p)
            pred_use[p] += 1

    return MG, pred_use


# -------------------------------
# Motif Library (canonical preds)
# -------------------------------

def build_motif_library() -> Dict[str, nx.DiGraph]:
    def dg(): return nx.DiGraph()
    motifs: Dict[str, nx.DiGraph] = {}

    def add_nodes(G, labels: Dict[str,str]):
        for k,v in labels.items(): G.add_node(k, category=v)

    # M1: zone -adjacentzone- zone
    G = dg(); add_nodes(G, {"a":"zone","b":"zone"}); G.add_edge("a","b", pred="adjacentzone")
    motifs["M1_adjacentZone_ZZ"] = G

    # M1b: element -adjacentzone- zone
    G = dg(); add_nodes(G, {"e":"element","z":"zone"}); G.add_edge("e","z", pred="adjacentzone")
    motifs["M1b_adjacentZone_EZ"] = G

    # M2: element -adjacentelement- element
    G = dg(); add_nodes(G, {"a":"element","b":"element"}); G.add_edge("a","b", pred="adjacentelement")
    motifs["M2_adjacentElement_EE"] = G

    # M2b: element -adjacentelement- zone  (NEW)
    G = dg(); add_nodes(G, {"e":"element","z":"zone"}); G.add_edge("e","z", pred="adjacentelement")
    motifs["M2b_adjacentElement_EZ"] = G

    # M3: element -intersectingelement- element
    G = dg(); add_nodes(G, {"a":"element","b":"element"}); G.add_edge("a","b", pred="intersectingelement")
    motifs["M3_intersectingElement_EE"] = G

    # M3b: element -intersectingelement- zone  (NEW)
    G = dg(); add_nodes(G, {"e":"element","z":"zone"}); G.add_edge("e","z", pred="intersectingelement")
    motifs["M3b_intersectingElement_EZ"] = G

    # M4: element -bfo_0000178- part
    G = dg(); add_nodes(G, {"a":"element","p":"part"}); G.add_edge("a","p", pred="bfo_0000178")
    motifs["M4_hasContinuantPart_EP"] = G

    # M5: element -hasfunction- function
    G = dg(); add_nodes(G, {"a":"element","f":"function"}); G.add_edge("a","f", pred="hasfunction")
    motifs["M5_hasFunction_EF"] = G

    # M6: element -hasquality- quality
    G = dg(); add_nodes(G, {"a":"element","q":"quality"}); G.add_edge("a","q", pred="hasquality")
    motifs["M6_hasQuality_EQ"] = G

    # M7: element -adjacentelement- element + element -hasfunction- function
    G = dg(); add_nodes(G, {"a":"element","b":"element","f":"function"})
    G.add_edge("a","b", pred="adjacentelement"); G.add_edge("a","f", pred="hasfunction")
    motifs["M7_adjacentElement_plus_Function"] = G

    # M8: element -adjacentelement- element + element -hasquality- quality
    G = dg(); add_nodes(G, {"a":"element","b":"element","q":"quality"})
    G.add_edge("a","b", pred="adjacentelement"); G.add_edge("a","q", pred="hasquality")
    motifs["M8_adjacentElement_plus_Quality"] = G

    return motifs


# -------------------------------
# Matching & Counting
# -------------------------------

def node_matcher(n1_attrs, n2_attrs):
    return n1_attrs.get("category") == n2_attrs.get("category")

def edge_matcher(e1_attrs, e2_attrs):
    return e1_attrs.get("pred") == e2_attrs.get("pred")

def unique_mapping_signature(mapping: Dict[str, str]) -> tuple:
    return tuple(sorted(mapping.values()))

def count_motif_instances(DG: nx.DiGraph, motif: nx.DiGraph, max_samples: int = 100000):
    GM = iso.DiGraphMatcher(DG, motif, node_match=node_matcher, edge_match=edge_matcher)
    seen, samples, n = set(), [], 0
    for mapping in GM.subgraph_isomorphisms_iter():
        sig = unique_mapping_signature(mapping)
        if sig in seen:
            continue
        seen.add(sig); n += 1
        if len(samples) < max_samples:
            samples.append(mapping)
    return n, samples


# -------------------------------
# Similarities
# -------------------------------

def cosine_sim_matrix(M: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(M, axis=1, keepdims=True)
    norms[norms==0] = 1.0
    X = M / norms
    return np.clip(X @ X.T, 0.0, 1.0)

def overlap_weighted_matrix(M: np.ndarray, weights: np.ndarray) -> np.ndarray:
    m = M.shape[0]
    S = np.zeros((m, m), dtype=float)
    w = np.asarray(weights, dtype=float)
    for i in range(m):
        for j in range(m):
            num = 0.0; den = 0.0
            for a, b, wi in zip(M[i], M[j], w):
                mx = max(a, b)
                if mx > 0:            # only-present motifs contribute
                    num += wi * (min(a, b) / mx)
                    den += wi
            S[i, j] = (num / den) if den > 0 else 0.0
    np.fill_diagonal(S, 1.0)
    return S


# -------------------------------
# Main
# -------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", default=".")
    ap.add_argument("--pattern", default="*_DG.rdf")
    ap.add_argument("--out-root", default=os.path.join(".","repro_pack","output"))
    ap.add_argument("--out-name", default="05b - Subgraph_Similarity_Canon")
    ap.add_argument("--max-samples", type=int, default=300)
    ap.add_argument("--weights-json", default="")
    ap.add_argument("--emit-extras", action="store_true", help="debug_* dosyalarını üret")
    args = ap.parse_args()

    input_dir = norm_path(args.input_dir)
    out_dir = norm_path(os.path.join(args.out_root, args.out_name))
    ensure_dir(out_dir)

    # 1) Load RDFs -> graphs
    rdf_paths = sorted(glob.glob(os.path.join(input_dir, args.pattern)))
    if not rdf_paths:
        print("[WARN] No RDFs matched:", os.path.join(input_dir, args.pattern))
    models, graphs = [], {}
    pred_maps = []  # extras: raw->canonical counts per model

    for p in rdf_paths:
        name = os.path.basename(p)
        print(f"[LOAD] {name}")
        DG, nsmap, raw_pred_counter = graph_from_rdf(p)
        models.append(name)
        graphs[name] = DG
        if args.emit_extras:
            for raw, cnt in raw_pred_counter.items():
                pred_maps.append({"model": name, "raw_predicate": raw, "canonical": canon_pred(raw), "count": cnt})

    # 2) Build motif library
    motifs = build_motif_library()
    motif_ids = list(motifs.keys())
    weights = np.ones(len(motif_ids), dtype=float)
    if args.weights_json.strip():
        try:
            wd = json.loads(args.weights_json)
            weights = np.array([float(wd.get(mid, 1.0)) for mid in motif_ids], dtype=float)
        except Exception as e:
            print("[WARN] weights-json parse edilemedi; 1.0 kullanıldı. Err:", e)

    # 3) Build motif subgraphs (ABox-focused)
    motif_graphs = {}
    mg_pred_rows = []
    cat_rows = []
    for m in models:
        DG = graphs[m]
        MG, pred_use = build_motif_graph(DG)
        motif_graphs[m] = MG

        if args.emit_extras:
            for cat, cnt in collections.Counter([DG.nodes[n].get("category","other") for n in DG.nodes]).items():
                cat_rows.append({"model": m, "category": cat, "count": cnt})
            for k,v in pred_use.items():
                mg_pred_rows.append({"model": m, "pred": k, "count": v})

    # 4) Count motif instances
    rows, sample_rows = [], []
    for m in models:
        MG = motif_graphs[m]
        for mid in motif_ids:
            n, samples = count_motif_instances(MG, motifs[mid], max_samples=args.max_samples)
            rows.append({"model": m, "motif": mid, "count": n})
            for idx, mp in enumerate(samples):
                if idx >= args.max_samples:
                    break
                sample_rows.append({"model": m, "motif": mid, **{f"{k}": str(v) for k,v in mp.items()}})

    df_counts = pd.DataFrame(rows).pivot_table(index="model", columns="motif", values="count", fill_value=0).reset_index()
    df_counts.to_csv(os.path.join(out_dir, "motif_counts_all.csv"), index=False)

    pd.DataFrame(sample_rows).to_csv(os.path.join(out_dir, "motif_match_samples.csv"), index=False)

    # motif library açıklaması
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

    # 5) Similarities
    M = df_counts.drop(columns=["model"]).values.astype(float)
    model_names = df_counts["model"].tolist()

    S_cos = cosine_sim_matrix(M)
    pd.DataFrame(S_cos, index=model_names, columns=model_names)\
      .to_csv(os.path.join(out_dir, "similarity_motif_cosine.csv"))

    S_ov = overlap_weighted_matrix(M, weights)
    pd.DataFrame(S_ov, index=model_names, columns=model_names)\
      .to_csv(os.path.join(out_dir, "similarity_motif_overlap_weighted.csv"))

    alpha = 0.5
    S_final = alpha*S_cos + (1-alpha)*S_ov
    pd.DataFrame(S_final, index=model_names, columns=model_names)\
      .to_csv(os.path.join(out_dir, "similarity_structural_final.csv"))

    rows_pairs = []
    for i, ai in enumerate(model_names):
        for j, aj in enumerate(model_names):
            if i >= j:
                continue
            rows_pairs.append({
                "model_A": ai, "model_B": aj,
                "cosine": S_cos[i,j],
                "overlap_weighted": S_ov[i,j],
                "final": S_final[i,j],
            })
    pd.DataFrame(rows_pairs).sort_values("final", ascending=False)\
      .to_csv(os.path.join(out_dir, "pairwise_structural_summary.csv"), index=False)

    # extras
    if args.emit_extras:
        if pred_maps:
            pd.DataFrame(pred_maps).sort_values(["model","count"], ascending=[True, False])\
                .to_csv(os.path.join(out_dir, "debug_predicate_map.csv"), index=False)
        if cat_rows:
            pd.DataFrame(cat_rows).sort_values(["model","category"])\
                .to_csv(os.path.join(out_dir, "debug_category_mix.csv"), index=False)
        if mg_pred_rows:
            pd.DataFrame(mg_pred_rows).sort_values(["model","count"], ascending=[True, False])\
                .to_csv(os.path.join(out_dir, "debug_motif_graph_pred_counts.csv"), index=False)

    print("\n[OK] Saved outputs under:", out_dir)
    print(" - motif_library.csv")
    print(" - motif_counts_all.csv")
    print(" - motif_match_samples.csv   (node ids per motif)")
    print(" - similarity_motif_cosine.csv")
    print(" - similarity_motif_overlap_weighted.csv")
    print(" - similarity_structural_final.csv")
    print(" - pairwise_structural_summary.csv")
    if args.emit_extras:
        print(" - debug_predicate_map.csv")
        print(" - debug_category_mix.csv")
        print(" - debug_motif_graph_pred_counts.csv")

if __name__ == "__main__":
    main()
