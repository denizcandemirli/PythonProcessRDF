# -*- coding: utf-8 -*-
"""
Motif (subgraph) similarity visuals for v2.3 outputs:
  1) Heatmap of row-normalized motif shares (model x motif)
  2) Dendrogram from final structural similarity (distance = 1 - S)
  3) Top-K motif contribution bars for selected pairs
Inputs (from 05b - Subgraph_Similarity_Canon):
  - motif_counts_all.csv
  - similarity_structural_final.csv
  - (optional) pairwise_structural_summary.csv
Usage (PowerShell example):
  python visualize_motif_similarity.py ^
    --in-dir ".\repro_pack\output\05b - Subgraph_Similarity_Canon" ^
    --out-dir ".\repro_pack\output\05c - Subgraph_Similarity_Visuals" ^
    --pairs "Building_05_DG.rdf,Building_06_DG.rdf;Option03_Revising_DG.rdf,Option04_Rev03_DG.rdf;Building_05_DG.rdf,Option04_Rev03_DG.rdf" ^
    --topk 15 --alpha 0.5
"""

import os, argparse, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform

def read_square(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # first column likely labels
    if df.shape[1] >= 2:
        df = df.set_index(df.columns[0])
    # drop unnamed junk
    drop_cols = [c for c in df.columns if str(c).lower().startswith("unnamed")]
    if drop_cols:
        df = df.drop(columns=drop_cols)
    # clean to numeric where possible
    df = df.apply(pd.to_numeric, errors="coerce")
    # symmetrize and clamp to [0,1]
    M = df.values.astype(float)
    M = (M + M.T) / 2.0
    np.fill_diagonal(M, 1.0)
    M = np.clip(M, 0.0, 1.0)
    return pd.DataFrame(M, index=df.index, columns=df.columns)

def safe_mkdir(p):
    os.makedirs(p, exist_ok=True)

def heatmap_counts_rowshare(counts_csv: str, out_png: str, out_csv: str):
    counts = pd.read_csv(counts_csv)
    assert "model" in counts.columns, "motif_counts_all.csv must contain a 'model' column"
    models = counts["model"].tolist()
    X = counts.drop(columns=["model"]).copy()
    # row-normalize → shares
    row_sums = X.sum(axis=1).replace(0, 1.0)
    shares = X.div(row_sums, axis=0)
    shares.insert(0, "model", models)
    shares.to_csv(out_csv, index=False)

    # plot heatmap (matplotlib only)
    fig = plt.figure(figsize=(max(8, 0.25*len(X.columns)), max(4, 0.5*len(models))))
    ax = fig.add_subplot(111)
    im = ax.imshow(shares.drop(columns=["model"]).values, aspect="auto", interpolation="nearest")
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models, fontsize=8)
    ax.set_xticks(range(len(X.columns)))
    ax.set_xticklabels(list(X.columns), rotation=90, fontsize=7)
    ax.set_title("Motif Shares (row-normalized)", fontsize=11)
    cb = fig.colorbar(im)
    cb.ax.tick_params(labelsize=8)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)

def dendro_from_similarity(sim_csv: str, out_png: str):
    S = read_square(sim_csv)
    # distance = 1 - similarity
    D = 1.0 - S.values
    np.fill_diagonal(D, 0.0)
    # condensed distance for linkage
    D_cond = squareform(D, checks=False)
    Z = linkage(D_cond, method="average")  # UPGMA/average
    # plot dendrogram
    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(111)
    dendrogram(Z, labels=S.index.tolist(), leaf_rotation=90, leaf_font_size=8, ax=ax)
    ax.set_title("Hierarchical Clustering (distance = 1 - S_final_motif)")
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)

def _pair_key(a,b): return f"{a}__{b}"

def motif_contributions_for_pair(counts_csv: str, a: str, b: str, alpha: float=0.5):
    """
    Returns per-motif contribution for pair (a,b):
      contrib_k = alpha * (a_k b_k)/(||a|| ||b||) + (1-alpha) * s_k / K
      where s_k = 1 if a_k=b_k=0 else min/max; K = #motifs
    """
    counts = pd.read_csv(counts_csv)
    assert "model" in counts.columns
    X = counts.set_index("model")
    if a not in X.index or b not in X.index:
        raise ValueError(f"Models not found in motif_counts_all.csv: {a}, {b}")

    va = X.loc[a].astype(float).values
    vb = X.loc[b].astype(float).values
    mot = list(X.columns)
    K = len(mot)

    na = math.sqrt(float((va*va).sum()))
    nb = math.sqrt(float((vb*vb).sum()))
    if na == 0 or nb == 0:
        cos_comp = np.zeros(K, dtype=float)
    else:
        cos_comp = (va*vb)/(na*nb)

    s = np.zeros(K, dtype=float)
    for k in range(K):
        ak, bk = va[k], vb[k]
        if ak==0 and bk==0:
            s[k] = 1.0
        else:
            m = max(ak,bk)
            s[k] = (min(ak,bk)/m) if m>0 else 0.0

    overlap_comp = s / float(K)
    contrib = alpha * cos_comp + (1.0 - alpha) * overlap_comp

    df = pd.DataFrame({
        "motif": mot,
        "cosine_component": cos_comp,
        "overlap_component": overlap_comp,
        "contribution": contrib
    }).sort_values("contribution", ascending=False)
    return df

def bar_topk_contrib(df: pd.DataFrame, title: str, out_png: str, topk: int=15):
    top = df.head(topk)
    fig = plt.figure(figsize=(10, max(3, 0.35*len(top))))
    ax = fig.add_subplot(111)
    ax.barh(top["motif"][::-1], top["contribution"][::-1])
    ax.set_xlabel("Contribution (α·cos + (1−α)·overlap/K)")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)

def parse_pairs_arg(pairs_arg: str):
    """
    "A,B;C,D;E,F" -> [(A,B),(C,D),(E,F)]
    Whitespace trimmed.
    """
    out = []
    for chunk in pairs_arg.split(";"):
        if not chunk.strip(): continue
        parts = [p.strip() for p in chunk.split(",")]
        if len(parts) != 2:
            raise ValueError(f"Bad pair spec: {chunk}")
        out.append((parts[0], parts[1]))
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", required=True, help="Folder of v2.3 motif outputs (05b - Subgraph_Similarity_Canon)")
    ap.add_argument("--out-dir", required=True, help="Where to save visuals")
    ap.add_argument("--pairs", default="", help='Pairs like "A,B;C,D;E,F" (model file names)')
    ap.add_argument("--topk", type=int, default=15)
    ap.add_argument("--alpha", type=float, default=0.5, help="Weight for cosine in contribution (rest is overlap)")
    args = ap.parse_args()

    safe_mkdir(args.out_dir)

    counts_csv = os.path.join(args.in_dir, "motif_counts_all.csv")
    sim_final_csv = os.path.join(args.in_dir, "similarity_structural_final.csv")
    manifest_csv = os.path.join(args.in_dir, "pairwise_structural_summary.csv")

    # 1) Heatmap (row shares)
    heatmap_png = os.path.join(args.out_dir, "motif_heatmap_shares.png")
    shares_csv  = os.path.join(args.out_dir, "motif_shares_row_normalized.csv")
    heatmap_counts_rowshare(counts_csv, heatmap_png, shares_csv)
    print("[OK] Heatmap saved:", heatmap_png)

    # 2) Dendrogram
    dendro_png = os.path.join(args.out_dir, "dendrogram_motif_final.png")
    dendro_from_similarity(sim_final_csv, dendro_png)
    print("[OK] Dendrogram saved:", dendro_png)

    # 3) Contribution bars for selected pairs
    pairs = []
    if args.pairs:
        pairs = parse_pairs_arg(args.pairs)
    elif os.path.exists(manifest_csv):
        # take top-3 by final score from manifest if pairs not specified
        mf = pd.read_csv(manifest_csv)
        if {"model_A","model_B","final"}.issubset(mf.columns):
            mf = mf.sort_values("final", ascending=False)
            for _,r in mf.head(3).iterrows():
                pairs.append((str(r["model_A"]), str(r["model_B"])))
    else:
        print("[WARN] No --pairs and no manifest found; skipping contribution bars.")

    for (A,B) in pairs:
        df = motif_contributions_for_pair(counts_csv, A, B, alpha=args.alpha)
        base = _pair_key(A,B).replace(".rdf","").replace(" ","")
        csv_path = os.path.join(args.out_dir, f"motif_contrib_{base}.csv")
        png_path = os.path.join(args.out_dir, f"motif_contrib_{base}.png")
        df.to_csv(csv_path, index=False)
        bar_topk_contrib(df, f"Top-{args.topk} Motif Contributions: {A} vs {B}", png_path, topk=args.topk)
        print(f"[OK] Saved contributions for {A} vs {B} →", png_path)

    print("[DONE] Visuals written to:", args.out_dir)

if __name__ == "__main__":
    main()
