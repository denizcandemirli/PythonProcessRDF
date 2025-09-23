# -*- coding: utf-8 -*-
"""
Total similarity visualization (robust to new/old column names)

Produces:
 - total_heatmap.png
 - total_dendrogram.png
 - total_contrib_*.png / .csv  (weighted component bars for selected or top-3 pairs)

Accepts pairwise_total_summary.csv with either:
 - NEW:  A,B,S_content,S_typed,S_edge,S_struct,S_total
 - OLD:  model_A,model_B,content_cos,typed_edge_cos,edge_sets_jaccard,structural_similarity,total
"""

import os, argparse, json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform

FUSION_W_DEFAULT = {"w_content":0.30, "w_typed":0.20, "w_edge":0.10, "w_struct":0.40}

def safe_mkdir(p): os.makedirs(p, exist_ok=True)

def read_square(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0)
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.reindex(index=df.index, columns=df.index)
    M = df.values.astype(float)
    M = (M + M.T) / 2.0
    np.fill_diagonal(M, 1.0)
    M = np.clip(M, 0.0, 1.0)
    return pd.DataFrame(M, index=df.index, columns=df.columns)

def heatmap_total(S: pd.DataFrame, out_png: str):
    fig = plt.figure(figsize=(6.5, 5.2))
    ax = fig.add_subplot(111)
    im = ax.imshow(S.values, vmin=0, vmax=1, aspect="equal")
    ax.set_xticks(np.arange(S.shape[1]))
    ax.set_yticks(np.arange(S.shape[0]))
    ax.set_xticklabels(S.columns, rotation=45, ha="right")
    ax.set_yticklabels(S.index)
    ax.set_title("Total Similarity Heatmap (0–1)")
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("S_total", rotation=90)
    fig.tight_layout(); fig.savefig(out_png, dpi=200); plt.close(fig)

def dendro_total(S: pd.DataFrame, out_png: str):
    D = 1.0 - np.clip(S.values.astype(float), 0.0, 1.0)
    np.fill_diagonal(D, 0.0)
    if np.allclose(D, 0.0):
        D = D + np.eye(D.shape[0]) * 1e-9
        np.fill_diagonal(D, 0.0)
    cond = squareform(D, checks=False)
    Z = linkage(cond, method="average")
    fig = plt.figure(figsize=(max(6, 0.6*len(S.index)), 4.8))
    ax = fig.add_subplot(111)
    dendrogram(Z, labels=list(S.index), leaf_rotation=90, ax=ax)
    ax.set_title("Hierarchical Clustering (distance = 1 − S_total)")
    ax.set_ylabel("distance")
    fig.tight_layout(); fig.savefig(out_png, dpi=200); plt.close(fig)

def _canon_pairwise_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Map old/new columns to canonical: A,B,S_content,S_typed,S_edge,S_struct,S_total."""
    cmap = {c.lower(): c for c in df.columns}
    def has(k): return k in cmap
    out = pd.DataFrame()
    # A/B
    A = cmap["a"] if has("a") else cmap.get("model_a") or cmap.get("left") or cmap.get("source")
    B = cmap["b"] if has("b") else cmap.get("model_b") or cmap.get("right") or cmap.get("target")
    if not A or not B:
        raise RuntimeError("Cannot find A/B columns in pairwise_total_summary.csv")
    out["A"] = df[[A,B]].min(axis=1)
    out["B"] = df[[A,B]].max(axis=1)
    # components
    def pick(*cands):
        for c in cands:
            if c and c.lower() in cmap: return cmap[c.lower()]
        return None
    C = pick("S_content","content_cos","content_cosine","content")
    T = pick("S_typed","typed_edge_cos","typed_edge_cosine","typed_cos","typed")
    E = pick("S_edge","edge_sets_jaccard","edge_jaccard","edge")
    R = pick("S_struct","structural_similarity","struct","struct_final","motif_final")
    TOT = pick("S_total","total","final_similarity","score_total")
    if not all([C,T,E,R]): raise RuntimeError("Missing component columns in pairwise_total_summary.csv")
    out["S_content"] = df[C]; out["S_typed"] = df[T]; out["S_edge"] = df[E]; out["S_struct"] = df[R]
    if TOT: out["S_total"] = df[TOT]
    return out

def _load_weights_guess(in_dir: str) -> dict:
    # Try a weights JSON next to the matrix; fallback to authoritative defaults.
    p = os.path.join(in_dir, "weights_used.json")
    if os.path.exists(p):
        try:
            Wraw = json.load(open(p, "r"))
            # Accept nested ("weights_normalized") or flat
            src = Wraw.get("weights_normalized", Wraw)
            return {
                "w_content": float(src.get("w_content", FUSION_W_DEFAULT["w_content"])),
                "w_typed":   float(src.get("w_typed",   FUSION_W_DEFAULT["w_typed"])),
                "w_edge":    float(src.get("w_edge",    FUSION_W_DEFAULT["w_edge"])),
                "w_struct":  float(src.get("w_struct",  FUSION_W_DEFAULT["w_struct"])),
            }
        except Exception:
            pass
    return dict(FUSION_W_DEFAULT)

def parse_pairs_arg(s: str):
    pairs = []
    s = (s or "").strip()
    if not s: return pairs
    for token in s.split(";"):
        tok = token.strip()
        if not tok: continue
        parts = [x.strip() for x in tok.split(",")]
        if len(parts) == 2:
            # normalize ordering later; here just keep tuple
            pairs.append((parts[0], parts[1]))
    return pairs

def component_bars(in_dir: str, out_dir: str, pairs: list | None):
    W = _load_weights_guess(in_dir)
    df = pd.read_csv(os.path.join(in_dir, "pairwise_total_summary.csv"))
    dfC = _canon_pairwise_cols(df)

    # If no pairs provided: choose top-3 by S_total (if no S_total then by w·S sum)
    if not pairs:
        if "S_total" in dfC.columns:
            top = dfC.sort_values("S_total", ascending=False).head(3)
        else:
            tmp = dfC.copy()
            tmp["S_total_est"] = (W["w_content"]*tmp["S_content"] +
                                  W["w_typed"]  *tmp["S_typed"]   +
                                  W["w_edge"]   *tmp["S_edge"]    +
                                  W["w_struct"] *tmp["S_struct"])
            top = tmp.sort_values("S_total_est", ascending=False).head(3)
        pairs = list(zip(top["A"], top["B"]))

    for (A, B) in pairs:
        # locate the canonical row for this unordered pair
        row = dfC[((dfC["A"]==min(A,B)) & (dfC["B"]==max(A,B)))]
        if row.empty:
            print(f"[WARN] pair not found in summary: {A} vs {B}"); continue
        r = row.iloc[0]

        comps = {
            "content": W["w_content"] * float(r["S_content"]),
            "typed":   W["w_typed"]   * float(r["S_typed"]),
            "edge":    W["w_edge"]    * float(r["S_edge"]),
            "struct":  W["w_struct"]  * float(r["S_struct"]),
        }
        total = float(r["S_total"]) if "S_total" in row.index else sum(comps.values())
        share = {k: (v/total if total>0 else 0.0) for k,v in comps.items()}

        # CSV
        out_csv = os.path.join(out_dir, f"total_contrib_{A.replace('.rdf','')}__{B.replace('.rdf','')}.csv")
        pd.DataFrame([
            {"component":k, "weighted_contribution":v, "share_of_total":share[k]}
            for k,v in comps.items()
        ]).sort_values("weighted_contribution", ascending=False).to_csv(out_csv, index=False)

        # Bar plot
        fig = plt.figure(figsize=(7.4,4.4))
        ax = fig.add_subplot(111)
        ks = list(comps.keys()); vs = [comps[k] for k in ks]
        ax.bar(ks, vs)
        ax.set_ylabel("Weighted contribution")
        ax.set_title(f"Component contributions • {A} vs {B}")
        fig.tight_layout()
        out_png = os.path.join(out_dir, f"total_contrib_{A.replace('.rdf','')}__{B.replace('.rdf','')}.png")
        fig.savefig(out_png, dpi=200); plt.close(fig)
        # Avoid non-ASCII to be safe on Windows consoles
        print(f"[OK] Contribution bars for {A} vs {B} -> {out_png}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir",  required=True, help="...\\repro_pack\\output\\06 - Total_Similarity")
    ap.add_argument("--out-dir", required=True, help="...\\repro_pack\\output\\06b - Total_Similarity_Visuals")
    ap.add_argument("--pairs", default="", help="A,B;C,D; ...  (optional)")
    args = ap.parse_args()

    safe_mkdir(args.out_dir)

    # Heatmap + Dendrogram from total matrix
    S = read_square(os.path.join(args.in_dir, "total_similarity_matrix.csv"))
    heatmap_total(S, os.path.join(args.out_dir, "total_heatmap.png")); print("[OK] Heatmap saved.")
    dendro_total(S, os.path.join(args.out_dir, "total_dendrogram.png")); print("[OK] Dendrogram saved.")

    # Component contribution bars (top-3 or user-specified)
    pairs = parse_pairs_arg(args.pairs)
    component_bars(args.in_dir, args.out_dir, pairs)
    print("[DONE] Visuals written to:", args.out_dir)

if __name__ == "__main__":
    main()
