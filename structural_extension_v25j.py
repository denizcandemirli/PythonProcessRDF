#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Structural Extension v25h
- S1: type/function inventories
- S2: structural motifs (micro)
- S3: system scores (macro)
- S4: structural similarity (final)  [GUARDED COSINE/JACCARD]

Changes vs v25g:
  * Added safe_cosine / safe_jaccard (zero-vector / empty-set -> 0.0)
  * S4 uses safe_* to compute S_motif and S_system
  * CLI weights for S_struct_total: --w-motif, --w-system (defaults 0.5/0.5)

This script preserves v25g behavior for S1-S3 and only hardens S4.
"""

import os, re, json, argparse, glob
from collections import defaultdict, Counter
import numpy as np
import pandas as pd
from rdflib import Graph, Namespace, URIRef, RDF, RDFS, Literal

# ---------------------------
# Namespaces (same as v25g)
# ---------------------------
CORE = Namespace("http://example.org/core#")
BD   = Namespace("http://example.org/bd#")
BOT  = Namespace("https://w3id.org/bot#")
BFO  = Namespace("http://purl.obolibrary.org/obo/")
QUDT = Namespace("http://qudt.org/schema/qudt/")

# ---------------------------------
# Utility: filesystem & formatting
# ---------------------------------
def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def to_local_name(u):
    s = str(u)
    if "#" in s:
        return s.rsplit("#", 1)[1]
    if "/" in s:
        return s.rsplit("/", 1)[1]
    return s

# ---------------------------------
# NEW: similarity guards (v25h)
# ---------------------------------
def safe_cosine(a, b):
    """Cosine with zero-vector guard: if either vector has 0-norm, return 0.0"""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))

def safe_jaccard(A, B):
    """Jaccard with empty-set guard: if both empty, return 0.0; if one empty, 0.0"""
    A = set(A); B = set(B)
    if not A and not B:
        return 0.0
    if not A or not B:
        return 0.0
    return float(len(A & B) / len(A | B))

# ---------------------------------
# S1: load RDF & build inventories
# (identical logic to v25g)
# ---------------------------------
TYPE_PATTERNS = {
    "Beam": r"(?:IfcBeam\b|Beam\b)",
    "Column": r"(?:IfcColumn\b|Column\b)",
    "Slab": r"(?:IfcSlab\b|Slab\b|Deck\b|Floor\b|Plate\b)",
    "Wall": r"(?:IfcWall\b|Wall\b|Shear[- ]?Wall\b|CoreWall\b|RetainingWall\b|ExteriorWall\b|InteriorWall\b|PartitionWall\b)",
    "Brace": r"(?:Brace\b|Bracing\b|Tie\b|Strut\b|IfcStructuralCurveMember.*BRACE)",
    "Core": r"(?:Core\b|StairCore\b|LiftCore\b)",
    "Foundation": r"(?:Foundation\b|Footing\b|PileCap\b)"
}
TYPE_REGEX = {k: re.compile(v, re.IGNORECASE) for k, v in TYPE_PATTERNS.items()}

FUNC_PATTERNS = {
    "LoadBearing": r"(?:LoadBearing\b|Bearing\b)",
    "Shear": r"(?:Shear\b)",
    "Moment": r"(?:Moment\b|RigidFrame\b|MomentFrame\b)"
}
FUNC_REGEX = {k: re.compile(v, re.IGNORECASE) for k, v in FUNC_PATTERNS.items()}

class Model:
    def __init__(self, name, graph):
        self.name = name
        self.g = graph
        self.types = defaultdict(set)      # node -> set(type_local_names)
        self.funcs = defaultdict(set)      # node -> set(function_labels)
        self.type_counts = Counter()
        self.func_counts = Counter()
        self._index()

    def _index(self):
        # rdf:type
        for s, p, o in self.g.triples((None, RDF.type, None)):
            tln = to_local_name(o)
            self.types[s].add(tln)

        # functions via core:hasFunction (and label/type of function node)
        for s, p, o in self.g.triples((None, CORE.hasFunction, None)):
            # function node 'o' may have types or labels
            labels = set()
            for _, _, lbl in self.g.triples((o, RDFS.label, None)):
                if isinstance(lbl, Literal):
                    labels.add(str(lbl))
            # fallback to type local names
            if not labels:
                for _, _, ft in self.g.triples((o, RDF.type, None)):
                    labels.add(to_local_name(ft))
            # classify function names via regex
            mapped = set()
            for lab in labels:
                for key, rx in FUNC_REGEX.items():
                    if rx.search(lab):
                        mapped.add(key)
            if not mapped and labels:
                # may still keep raw labels if needed
                pass
            for m in mapped:
                self.funcs[s].add(m)

        # counts
        for s, tset in self.types.items():
            # match structural super types
            hit_any = False
            for typeln in tset:
                for key, rx in TYPE_REGEX.items():
                    if rx.search(typeln):
                        self.type_counts[key] += 1
                        hit_any = True
            if not hit_any:
                # not counted in structural bucket
                pass

        for s, fset in self.funcs.items():
            for f in fset:
                self.func_counts[f] += 1

def load_models(input_dir, pattern):
    paths = sorted(glob.glob(os.path.join(input_dir, pattern)))
    models = []
    for p in paths:
        name = os.path.basename(p)
        print(f"       {name}")
        g = Graph()
        g.parse(p)  # rdflib guesses RDF/XML etc.
        models.append(Model(name, g))
    return models

# ---------------------------------
# S2: structural motifs (micro)
# (kept simple; topological motifs depend on explicit edges)
# ---------------------------------
MOTIF_COLS = ["M2_frameNode", "M3_wallSlabAdj", "M4_coreComp", "M_braceAdj", "M5_structRole"]

def count_motifs(model: Model):
    """
    Minimal motif set:
      - M5_structRole: count of element->function assignments (any of LoadBearing/Shear/Moment)
      - Others require explicit adjacency/intersect/part-whole edges; set to 0 if not present
    """
    m = {k: 0 for k in MOTIF_COLS}
    # M5: E -> F links
    m["M5_structRole"] = sum(len(fs) for fs in model.funcs.values())
    # NOTE: If you later enrich adjacency/intersection/part-whole in RDF,
    # implement the counts for M2/M3/M4/M_braceAdj here.
    return m

# ---------------------------------
# S3: system scores (macro; rule-based)
# ---------------------------------
def system_scores_for_model(model: Model, motif_row: dict, func_shares: dict, dual_thresh=0.25):
    """
    frame ~ M2 + Moment
    wall  ~ M3 + Wall with LB/Shear presence
    dual  ~ (frame >= dual_thresh) and (wall >= dual_thresh) -> combine
    braced~ brace motif
    """
    # normalize motif contributions to [0,1] by simple presence
    has_M2 = 1.0 if motif_row.get("M2_frameNode", 0) > 0 else 0.0
    has_M3 = 1.0 if motif_row.get("M3_wallSlabAdj", 0) > 0 else 0.0
    has_Mb = 1.0 if motif_row.get("M_braceAdj", 0) > 0 else 0.0

    # function shares
    lb = float(func_shares.get("LoadBearing", 0.0))
    sh = float(func_shares.get("Shear", 0.0))
    mo = float(func_shares.get("Moment", 0.0))

    frame = 0.5*has_M2 + 0.5*min(1.0, mo)
    wall  = 0.5*has_M3 + 0.5*min(1.0, lb + sh)  # crude but conservative
    braced= has_Mb

    dual = 0.0
    if frame >= dual_thresh and wall >= dual_thresh:
        dual = 0.5*frame + 0.5*wall

    return dict(frame=frame, wall=wall, dual=dual, braced=braced)

# ---------------------------------
# S1 writer helpers (same outputs as v25g)
# ---------------------------------
def write_s1_outputs(models, out_dir):
    # types
    rows_t = []
    for m in models:
        for t, c in m.type_counts.items():
            rows_t.append({"model": m.name, "type": t, "count": int(c)})
    df_types = pd.DataFrame(rows_t).sort_values(["model", "count"], ascending=[True, False])
    df_types.to_csv(os.path.join(out_dir, "struct_types_histogram.csv"), index=False)

    # functions
    rows_f = []
    for m in models:
        for f, c in m.func_counts.items():
            rows_f.append({"model": m.name, "function": f, "count": int(c)})
    df_funcs = pd.DataFrame(rows_f).sort_values(["model", "function"])
    df_funcs.to_csv(os.path.join(out_dir, "struct_functions_histogram.csv"), index=False)

    # function shares (wide)
    if not df_funcs.empty:
        total_by_model = df_funcs.groupby("model")["count"].sum()
        df_wide = df_funcs.pivot(index="model", columns="function", values="count").fillna(0.0)
        for col in df_wide.columns:
            df_wide[col] = df_wide[col] / total_by_model
        df_wide.to_csv(os.path.join(out_dir, "struct_functions_shares_wide.csv"))

    # data availability (very light)
    rows_av = []
    for m in models:
        rows_av.append({
            "model": m.name,
            # we don’t have explicit edges here; keep 0 to reflect data lack
            "n_adjacentElement": 0, "n_adjacentZone": 0, "n_intersectingElement": 0,
            "n_hasContinuantPart": 0,
            "n_Wall": int(m.type_counts.get("Wall", 0)),
            "n_Slab": int(m.type_counts.get("Slab", 0)),
            "n_Beam": int(m.type_counts.get("Beam", 0)),
            "n_Column": int(m.type_counts.get("Column", 0)),
            "n_Brace": int(m.type_counts.get("Brace", 0)),
            "n_Core": int(m.type_counts.get("Core", 0)),
            "n_Foundation": int(m.type_counts.get("Foundation", 0)),
        })
    pd.DataFrame(rows_av).to_csv(os.path.join(out_dir, "struct_data_availability.csv"), index=False)

    return df_types, (df_funcs if 'df_funcs' in locals() else pd.DataFrame())

# ---------------------------------
# S2 writer
# ---------------------------------
def write_s2_outputs(models, out_dir):
    rows = []
    for m in models:
        motifs = count_motifs(m)
        d = {"model": m.name}
        d.update(motifs)
        rows.append(d)
    df_counts = pd.DataFrame(rows).sort_values("model")
    df_counts.to_csv(os.path.join(out_dir, "struct_motif_counts.csv"), index=False)

    # row-normalized shares
    df_share = df_counts.set_index("model").astype(float)
    row_sums = df_share.sum(axis=1).replace(0.0, np.nan)
    df_share = df_share.div(row_sums, axis=0).fillna(0.0)
    df_share.to_csv(os.path.join(out_dir, "struct_motif_shares.csv"))

    # densities per 100 elements (optional, defaulting element count to type sum)
    nE = {}
    for m in models:
        nE[m.name] = int(sum(m.type_counts.values()))
    dens = df_counts.set_index("model").astype(float)
    for r in dens.index:
        denom = max(1, nE.get(r, 1))
        dens.loc[r, :] = (dens.loc[r, :] / denom) * 100.0
    dens.to_csv(os.path.join(out_dir, "struct_motif_densities_per100.csv"))

    # empty proxy edges
    pd.DataFrame(columns=["model","proxy","src","dst"]).to_csv(os.path.join(out_dir,"motif_proxy_edges.csv"), index=False)

    return df_counts

# ---------------------------------
# S3 writer
# ---------------------------------
def write_s3_outputs(models, out_dir, df_motif_counts, dual_thresh=0.25):
    # function shares (wide) from disk (S1 wrote it)
    p_func_wide = os.path.join(out_dir, "struct_functions_shares_wide.csv")
    if os.path.exists(p_func_wide):
        df_fw = pd.read_csv(p_func_wide).set_index("model")
    else:
        df_fw = pd.DataFrame()

    # system scores per model
    rows = []
    comps = []  # score components (diagnostic)
    for m in models:
        motif_row = df_motif_counts[df_motif_counts["model"] == m.name]
        motif_row = motif_row.drop(columns=["model"]).iloc[0].to_dict() if not motif_row.empty else {}
        func_shares = df_fw.loc[m.name].to_dict() if (not df_fw.empty and m.name in df_fw.index) else {}
        sc = system_scores_for_model(m, motif_row, func_shares, dual_thresh=dual_thresh)
        rows.append({"model": m.name, **sc})
        comps.append({"model": m.name, **motif_row, **{f"func_{k}": v for k, v in func_shares.items()}})
    df_sys = pd.DataFrame(rows).set_index("model").sort_index()
    df_sys.to_csv(os.path.join(out_dir, "struct_system_scores.csv"))
    pd.DataFrame(comps).to_csv(os.path.join(out_dir, "struct_score_components.csv"), index=False)
    return df_sys

# ---------------------------------
# S4 writer (GUARDED)
# ---------------------------------
def write_s4_outputs(out_dir, w_motif=0.5, w_system=0.5):
    # motif shares (rows: models)
    df_ms = pd.read_csv(os.path.join(out_dir, "struct_motif_shares.csv")).set_index("model")
    # system scores (rows: models)
    df_sys = pd.read_csv(os.path.join(out_dir, "struct_system_scores.csv")).set_index("model")

    models = list(df_sys.index)
    n = len(models)

    # pairwise matrices
    S_motif = np.zeros((n, n), dtype=float)
    S_system = np.zeros((n, n), dtype=float)
    S_total = np.zeros((n, n), dtype=float)

    # compute
    for i, a in enumerate(models):
        va_m = df_ms.loc[a].values
        va_s = df_sys.loc[a].values
        for j, b in enumerate(models):
            vb_m = df_ms.loc[b].values
            vb_s = df_sys.loc[b].values
            sm = safe_cosine(va_m, vb_m)     # << guarded
            ss = safe_cosine(va_s, vb_s)     # << guarded
            S_motif[i, j] = sm
            S_system[i, j] = ss
            S_total[i, j] = w_motif*sm + w_system*ss

    # write matrices
    dfM = pd.DataFrame(S_total, index=models, columns=models)
    dfM.to_csv(os.path.join(out_dir, "struct_similarity_matrix.csv"))

    # pairwise summary (upper triangle)
    rows = []
    for i in range(n):
        for j in range(i+1, n):
            rows.append({
                "model_a": models[i],
                "model_b": models[j],
                "S_motif": S_motif[i, j],
                "S_system": S_system[i, j],
                "S_struct_total": S_total[i, j],
            })
    df_pair = pd.DataFrame(rows).sort_values("S_struct_total", ascending=False)
    df_pair.to_csv(os.path.join(out_dir, "pairwise_structural_summary.csv"), index=False)

    # weights used
    with open(os.path.join(out_dir, "weights_used.json"), "w", encoding="utf-8") as f:
        json.dump({"w_motif": w_motif, "w_system": w_system}, f)

# ---------------------------------
# MAIN
# ---------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--pattern", default="*_DG.rdf")
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--out-name", default="07 - Structural_Extension_v25h")
    ap.add_argument("--dual-thresh", type=float, default=0.25)
    ap.add_argument("--w-motif", type=float, default=0.5)
    ap.add_argument("--w-system", type=float, default=0.5)
    ap.add_argument("--emit-debug", action="store_true")
    args = ap.parse_args()

    out_dir = os.path.join(args.out_root, args.out_name)
    ensure_dir(out_dir)

    print("[LOAD]")
    models = load_models(args.input_dir, args.pattern)

    # S1
    df_types, df_funcs = write_s1_outputs(models, out_dir)

    # S2
    df_motif_counts = write_s2_outputs(models, out_dir)

    # S3
    df_sys = write_s3_outputs(models, out_dir, df_motif_counts, dual_thresh=args.dual_thresh)

    # S4 (GUARDED)
    write_s4_outputs(out_dir, w_motif=args.w_motif, w_system=args.w_system)

    print("\n[OK] Saved outputs under:", out_dir)
    print(" - struct_types_histogram.csv")
    print(" - struct_functions_histogram.csv")
    print(" - struct_functions_shares_wide.csv")
    print(" - struct_motif_counts.csv")
    print(" - struct_motif_shares.csv")
    print(" - struct_motif_densities_per100.csv")
    print(" - motif_proxy_edges.csv")
    print(" - struct_score_components.csv")
    print(" - struct_system_scores.csv")
    print(" - struct_similarity_matrix.csv")
    print(" - pairwise_structural_summary.csv")
    print(" - weights_used.json")

if __name__ == "__main__":
    main()
