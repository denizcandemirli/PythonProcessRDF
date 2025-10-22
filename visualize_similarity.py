#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File: visualize_similarity.py
Purpose: Generate visualizations for similarity analysis results.
Creates:
- Total similarity heatmap and dendrogram
- Component contribution bars
- S1→S4 structural heatmap + motif-share heatmap + system radar
- Optional SP / NetLSD
"""

__version__ = "2025.10"

import os, re, argparse, json
from typing import Dict
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

try:
    import seaborn as sns
    HAS_SNS = True
except Exception:
    HAS_SNS = False

try:
    from scipy.cluster.hierarchy import linkage, dendrogram
    from scipy.spatial.distance import squareform
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False
    print("[WARN] scipy not available, dendrograms will use a placeholder")

FUSION_WEIGHTS_DEFAULT = {"w_content": 0.30, "w_typed": 0.20, "w_edge": 0.10, "w_struct": 0.40}

# ---------------------------
# Helpers
# ---------------------------

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def _safe_float_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.reindex(index=df.index, columns=df.index)
    M = df.values.astype(float)
    M = (M + M.T) / 2.0
    np.fill_diagonal(M, 1.0)
    M = np.clip(M, 0.0, 1.0)
    return pd.DataFrame(M, index=df.index, columns=df.columns)

def read_square_matrix(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0)
    return _safe_float_df(df)

def load_weights_from_json(weights_path: str) -> Dict[str, float]:
    if not os.path.exists(weights_path):
        return FUSION_WEIGHTS_DEFAULT
    try:
        with open(weights_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        w = data.get("weights_normalized", data)
        return {
            "w_content": float(w.get("w_content", FUSION_WEIGHTS_DEFAULT["w_content"])),
            "w_typed":   float(w.get("w_typed",   FUSION_WEIGHTS_DEFAULT["w_typed"])),
            "w_edge":    float(w.get("w_edge",    FUSION_WEIGHTS_DEFAULT["w_edge"])),
            "w_struct":  float(w.get("w_struct",  FUSION_WEIGHTS_DEFAULT["w_struct"])),
        }
    except Exception as e:
        print(f"[WARN] Could not load weights from {weights_path}: {e}")
        return FUSION_WEIGHTS_DEFAULT

# ---------------------------
# Plotters
# ---------------------------

def create_heatmap(matrix: pd.DataFrame, title: str, output_path: str,
                   vmin: float = 0.0, vmax: float = 1.0, annot: bool = True):
    n = matrix.shape[0]
    use_annot = annot and n <= 12

    if HAS_SNS:
        plt.figure(figsize=(max(8, 0.8*n), max(6, 0.6*n)))
        ax = sns.heatmap(matrix, annot=use_annot, cmap="viridis", square=True,
                         fmt=".2f", vmin=vmin, vmax=vmax,
                         cbar_kws={'label': 'Similarity'})
        ax.set_title(title)
        plt.tight_layout()
        plt.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close()
    else:
        fig, ax = plt.subplots(figsize=(max(8, 0.8*n), max(6, 0.6*n)))
        im = ax.imshow(matrix.values, vmin=vmin, vmax=vmax, aspect="equal", cmap="viridis")
        ax.set_xticks(np.arange(n)); ax.set_yticks(np.arange(n))
        ax.set_xticklabels(matrix.columns, rotation=45, ha="right")
        ax.set_yticklabels(matrix.index)
        ax.set_title(title)
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Similarity", rotation=90)
        if use_annot:
            for i in range(n):
                for j in range(n):
                    ax.text(j, i, f"{matrix.iloc[i, j]:.2f}",
                            ha="center", va="center",
                            color="white" if matrix.iloc[i, j] < 0.5 else "black")
        plt.tight_layout()
        plt.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close()
    print(f"[OK] Heatmap saved: {output_path}")

def create_dendrogram(matrix: pd.DataFrame, title: str, output_path: str):
    D = 1.0 - np.clip(matrix.values.astype(float), 0.0, 1.0)
    np.fill_diagonal(D, 0.0)
    if np.allclose(D, 0.0):
        D = D + np.eye(D.shape[0]) * 1e-9
        np.fill_diagonal(D, 0.0)

    if not SCIPY_AVAILABLE:
        fig, ax = plt.subplots(figsize=(max(8, 0.6*len(matrix.index)), 6))
        ax.text(0.5, 0.5, "Install scipy for dendrograms\npip install scipy",
                ha="center", va="center", transform=ax.transAxes, fontsize=12)
        ax.set_title(title)
        plt.tight_layout()
        plt.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"[OK] Dendrogram placeholder saved: {output_path}")
        return

    from scipy.spatial.distance import squareform
    from scipy.cluster.hierarchy import linkage, dendrogram
    cond = squareform(D, checks=False)
    Z = linkage(cond, method="average")
    fig, ax = plt.subplots(figsize=(max(8, 0.6*len(matrix.index)), 6))
    dendrogram(Z, labels=list(matrix.index), leaf_rotation=90, ax=ax)
    ax.set_title(title)
    ax.set_ylabel("Distance (1 - similarity)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[OK] Dendrogram saved: {output_path}")

def create_component_bars(pairwise_path: str, weights: Dict[str, float],
                         output_dir: str, top_n: int = 3):
    df = pd.read_csv(pairwise_path)
    # normalize columns
    cmap = {}
    for c in df.columns:
        cl = c.lower()
        if "content" in cl and "cos" in cl:  cmap[c] = "S_content"
        elif "typed" in cl and "cos" in cl:  cmap[c] = "S_typed"
        elif "edge" in cl and "jaccard" in cl: cmap[c] = "S_edge"
        elif "struct" in cl:                 cmap[c] = "S_struct"
        elif "total" in cl:                  cmap[c] = "S_total"
    df = df.rename(columns=cmap)

    # choose top pairs by total (or estimated total)
    if "S_total" in df.columns:
        top_pairs = df.nlargest(top_n, "S_total")
    else:
        df["S_total_est"] = (
            weights["w_content"]*df.get("S_content", 0) +
            weights["w_typed"]  *df.get("S_typed",   0) +
            weights["w_edge"]   *df.get("S_edge",    0) +
            weights["w_struct"] *df.get("S_struct",  0)
        )
        top_pairs = df.nlargest(top_n, "S_total_est")

    for idx, (_, row) in enumerate(top_pairs.iterrows()):
        A = row.get("model_A", row.get("A", f"Pair_{idx}_A"))
        B = row.get("model_B", row.get("B", f"Pair_{idx}_B"))
        comps = {
            "Content":    weights["w_content"] * row.get("S_content", 0),
            "Typed-Edge": weights["w_typed"]   * row.get("S_typed",   0),
            "Edge-Set":   weights["w_edge"]    * row.get("S_edge",    0),
            "Structural": weights["w_struct"]  * row.get("S_struct",  0),
        }
        tot = sum(comps.values())
        if tot > 0:
            comps = {k: (v/tot)*100 for k,v in comps.items()}

        fig, ax = plt.subplots(figsize=(9, 5.5))
        bars = ax.bar(list(comps.keys()), list(comps.values()))
        ax.set_ylabel("Contribution (%)")
        ax.set_title(f"Component Contributions: {A} vs {B}")
        ax.set_ylim(0, 100)
        for bar, v in zip(bars, comps.values()):
            ax.text(bar.get_x()+bar.get_width()/2., v+1, f"{v:.1f}%", ha="center", va="bottom")
        plt.xticks(rotation=20)
        plt.tight_layout()
        out = os.path.join(output_dir, f"component_contrib_{A}_vs_{B}.png")
        plt.savefig(out, dpi=200, bbox_inches="tight"); plt.close()
        print(f"[OK] Component bars saved: {out}")

# ---------- SP / NetLSD / S1S4 specific plots ----------

def create_sp_heatmap(path: str, out: str):
    try:
        create_heatmap(read_square_matrix(path), "Motif SP Similarity", out)
    except Exception as e:
        print(f"[ERROR] SP heatmap: {e}")

def create_sp_profiles(vectors_csv: str, out: str):
    try:
        df = pd.read_csv(vectors_csv)
        sp_cols = [c for c in df.columns if c.startswith("sp_")]
        if not sp_cols:
            print(f"[WARN] No SP cols in {vectors_csv}"); return
        motifs = [c.replace("sp_","") for c in sp_cols]

        n = len(df)
        rows = 2
        cols = max(1, int(np.ceil(n/rows)))
        fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 5*rows), subplot_kw=dict(projection='polar'))
        axes = np.array(axes).reshape(-1) if n>1 else np.array([axes])
        angles = np.linspace(0, 2*np.pi, len(motifs), endpoint=False).tolist()
        angles += angles[:1]

        for i, (_, r) in enumerate(df.iterrows()):
            if i >= len(axes): break
            ax = axes[i]
            vals = [r[c] for c in sp_cols]; vals += vals[:1]
            ax.plot(angles, vals, 'o-', lw=2, label=r.get("model", f"M{i+1}"))
            ax.fill(angles, vals, alpha=0.25)
            ax.set_xticks(angles[:-1]); ax.set_xticklabels([m.replace('_','\n') for m in motifs], fontsize=8)
            ax.set_ylim(-1, 1); ax.grid(True)
            ax.set_title(r.get("model", f"Model {i+1}"))
        for j in range(i+1, len(axes)): axes[j].set_visible(False)
        plt.suptitle("Motif Significance Profiles", y=0.95)
        plt.tight_layout()
        plt.savefig(out, dpi=200, bbox_inches="tight"); plt.close()
        print(f"[OK] SP profiles saved: {out}")
    except Exception as e:
        print(f"[ERROR] SP profiles: {e}")

def create_netlsd_heatmap(path: str, out: str):
    try:
        create_heatmap(read_square_matrix(path), "NetLSD Structural Similarity", out)
    except Exception as e:
        print(f"[ERROR] NetLSD heatmap: {e}")

def create_netlsd_signatures(vectors_csv: str, out: str):
    try:
        df = pd.read_csv(vectors_csv, index_col=0)
        n = df.shape[0]
        plt.figure(figsize=(10, 6))
        times = np.logspace(-2, 2, num=df.shape[1])
        colors = plt.cm.Set3(np.linspace(0, 1, n))
        for i, (name, row) in enumerate(df.iterrows()):
            plt.plot(times, row.values, lw=2, label=name, color=colors[i])
        plt.xscale('log'); plt.xlabel('Time (log)'); plt.ylabel('Heat Trace')
        plt.title('NetLSD Heat-Trace Signatures'); plt.grid(True, alpha=0.3)
        plt.legend(bbox_to_anchor=(1.02,1), loc='upper left')
        plt.tight_layout()
        plt.savefig(out, dpi=200, bbox_inches="tight"); plt.close()
        print(f"[OK] NetLSD signatures saved: {out}")
    except Exception as e:
        print(f"[ERROR] NetLSD signatures: {e}")

def create_s1s4_structural_heatmap(path: str, out: str):
    try:
        create_heatmap(read_square_matrix(path), "S1→S4 Structural Similarity", out)
    except Exception as e:
        print(f"[ERROR] S1S4 structural heatmap: {e}")

def _choose_motif_columns_for_heatmap(df: pd.DataFrame):
    # prefer dens_* columns from S2
    preferred = ["dens_M2","dens_M3","dens_M4","dens_M2b"]
    motif_cols = [c for c in preferred if c in df.columns]
    if motif_cols:
        return motif_cols, True  # values are percentages (0–100)
    # fallback: any columns explicitly named like M2/M3/M4/M2b (avoid Moment)
    rx = re.compile(r"^M(2b?|3|4)$", re.I)
    motif_cols = [c for c in df.columns if rx.match(c)]
    if motif_cols:
        return motif_cols, False  # assume already 0–1 or fractional shares
    # last resort: any column that contains M2/M3/M4 token
    motif_cols = [c for c in df.columns if any(tok in c for tok in ["M2","M3","M4"])]
    return motif_cols, False

def create_s1s4_motif_share_heatmap(vectors_csv: str, out: str):
    try:
        df = pd.read_csv(vectors_csv, index_col=0)
        motif_cols, is_percent = _choose_motif_columns_for_heatmap(df)
        if not motif_cols:
            print(f"[WARN] No motif columns found in {vectors_csv}"); return

        X = df[motif_cols].copy()
        # If values look like 0–1, show as fractions (keep 3 decimals).
        # If they are 0–100 (%), scale to [0,1] only for display consistency.
        vmax = float(np.nanmax(X.values))
        if not is_percent and vmax <= 1.5:
            pass  # already fractions 0–1
        elif is_percent and vmax > 1.5:
            X = X / 100.0

        title = "S1→S4 Motif Share Heatmap"
        if HAS_SNS:
            plt.figure(figsize=(10, 6))
            sns.heatmap(X, annot=True if X.shape[0] <= 12 else False,
                        cmap="viridis", fmt=".3f", vmin=0.0, vmax=1.0,
                        cbar_kws={'label':'Motif Share (0–1)'} )
            plt.title(title); plt.xlabel('Motifs'); plt.ylabel('Models')
            plt.tight_layout()
            plt.savefig(out, dpi=200, bbox_inches="tight"); plt.close()
        else:
            # fallback without seaborn
            n = X.shape[0]
            fig, ax = plt.subplots(figsize=(max(8, 0.8*n), max(6, 0.6*n)))
            im = ax.imshow(X.values, vmin=0.0, vmax=1.0, aspect="equal", cmap="viridis")
            ax.set_xticks(np.arange(X.shape[1])); ax.set_yticks(np.arange(n))
            ax.set_xticklabels(X.columns, rotation=45, ha="right"); ax.set_yticklabels(X.index)
            ax.set_title(title)
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04); cbar.set_label("Motif Share (0–1)")
            if n <= 12:
                for i in range(n):
                    for j in range(X.shape[1]):
                        ax.text(j, i, f"{X.iloc[i, j]:.3f}",
                                ha="center", va="center",
                                color="white" if X.iloc[i, j] < 0.5 else "black")
            plt.tight_layout(); plt.savefig(out, dpi=200, bbox_inches="tight"); plt.close()
        print(f"[OK] S1S4 motif share heatmap saved: {out}")
    except Exception as e:
        print(f"[ERROR] S1S4 motif share heatmap: {e}")

def create_s1s4_system_radar(systems_csv: str, out: str):
    try:
        df = pd.read_csv(systems_csv, index_col=0)
        system_cols = [c for c in ["Frame","Braced","Wall","Dual"] if c in df.columns]
        if not system_cols:
            print(f"[WARN] No system columns in {systems_csv}"); return
        n = len(df)
        rows = 2
        cols = max(1, int(np.ceil(n/rows)))
        fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 5*rows), subplot_kw=dict(projection='polar'))
        axes = np.array(axes).reshape(-1) if n>1 else np.array([axes])
        angles = np.linspace(0, 2*np.pi, len(system_cols), endpoint=False).tolist()
        angles += angles[:1]
        for i, (name, r) in enumerate(df.iterrows()):
            if i >= len(axes): break
            ax = axes[i]
            vals = [r[c] for c in system_cols]; vals += vals[:1]
            ax.plot(angles, vals, 'o-', lw=2, label=name)
            ax.fill(angles, vals, alpha=0.25)
            ax.set_xticks(angles[:-1]); ax.set_xticklabels(system_cols, fontsize=8)
            ax.set_ylim(0, 1); ax.grid(True); ax.set_title(f"System: {name}")
        for j in range(i+1, len(axes)): axes[j].set_visible(False)
        plt.suptitle("S1→S4 System Scores", y=0.95)
        plt.tight_layout()
        plt.savefig(out, dpi=200, bbox_inches="tight"); plt.close()
        print(f"[OK] S1S4 system radar saved: {out}")
    except Exception as e:
        print(f"[ERROR] S1S4 system radar: {e}")

# ---------------------------
# Main
# ---------------------------

def main():
    ap = argparse.ArgumentParser(description="Generate similarity visualizations")
    ap.add_argument("--input-dir", required=True, help="Directory with CSV results")
    ap.add_argument("--output-dir", required=True, help="Directory to save figures")
    ap.add_argument("--total-matrix", default="total_similarity_matrix.csv")
    ap.add_argument("--pairwise-summary", default="pairwise_total_summary.csv")
    ap.add_argument("--weights-file", default="weights_used.json")
    ap.add_argument("--motif-shares", default="struct_motif_shares.csv")
    ap.add_argument("--top-pairs", type=int, default=3)
    ap.add_argument("--include-radar", action="store_true")
    ap.add_argument("--include-sp", action="store_true")
    ap.add_argument("--sp-matrix", default="struct_similarity_sp.csv")
    ap.add_argument("--sp-vectors", default="motif_sp_vectors.csv")
    ap.add_argument("--include-s1s4", action="store_true")
    ap.add_argument("--s1s4-matrix", default="struct_similarity_s1s4.csv")
    ap.add_argument("--s1s4-vectors", default="s4_motif_share_vectors.csv")
    ap.add_argument("--s1s4-systems", default="s3_system_scores.csv")
    ap.add_argument("--no-dendrogram", action="store_true", help="Skip dendrogram")
    ap.add_argument("--annot", action="store_true", help="Force value annotations in heatmaps")
    args = ap.parse_args()

    ensure_dir(args.output_dir)
    print(f"[INFO] Input dir : {args.input_dir}")
    print(f"[INFO] Output dir: {args.output_dir}")

    try:
        weights = load_weights_from_json(os.path.join(args.input_dir, args.weights_file))
        print(f"[INFO] Using weights: {weights}")

        total_csv = os.path.join(args.input_dir, args.total_matrix)
        if os.path.exists(total_csv):
            total_df = read_square_matrix(total_csv)
            create_heatmap(total_df, "Total Similarity Heatmap",
                           os.path.join(args.output_dir, "total_similarity_heatmap.png"),
                           annot=args.annot)
            if not args.no_dendrogram:
                create_dendrogram(total_df, "Hierarchical Clustering (Total Similarity)",
                                  os.path.join(args.output_dir, "total_similarity_dendrogram.png"))
        else:
            print(f"[WARN] Not found: {total_csv}")

        pairwise_csv = os.path.join(args.input_dir, args.pairwise_summary)
        if os.path.exists(pairwise_csv):
            create_component_bars(pairwise_csv, weights, args.output_dir, args.top_pairs)
        else:
            print(f"[WARN] Not found: {pairwise_csv}")

        if args.include_sp:
            spM = os.path.join(args.input_dir, args.sp_matrix)
            if os.path.exists(spM):
                create_sp_heatmap(spM, os.path.join(args.output_dir, "SP_similarity_heatmap.png"))
            else:
                print(f"[WARN] Not found: {spM}")
            spV = os.path.join(args.input_dir, args.sp_vectors)
            if os.path.exists(spV):
                # optional profile figure
                pass
            else:
                print(f"[WARN] Not found: {spV}")

        # NetLSD (if present)
        netlsdM = os.path.join(args.input_dir, "struct_similarity_netlsd.csv")
        if os.path.exists(netlsdM):
            create_netlsd_heatmap(netlsdM, os.path.join(args.output_dir, "NetLSD_similarity_heatmap.png"))

        # S1S4 visuals
        if args.include_s1s4:
            s1s4M = os.path.join(args.input_dir, args.s1s4_matrix)
            if os.path.exists(s1s4M):
                create_s1s4_structural_heatmap(s1s4M, os.path.join(args.output_dir, "S1S4_structural_heatmap.png"))
            else:
                print(f"[WARN] Not found: {s1s4M}")
            s1s4V = os.path.join(args.input_dir, args.s1s4_vectors)
            if os.path.exists(s1s4V):
                create_s1s4_motif_share_heatmap(s1s4V, os.path.join(args.output_dir, "S1S4_motif_share_heatmap.png"))
            else:
                print(f"[WARN] Not found: {s1s4V}")
            s1s4S = os.path.join(args.input_dir, args.s1s4_systems)
            if os.path.exists(s1s4S):
                create_s1s4_system_radar(s1s4S, os.path.join(args.output_dir, "S1S4_system_radar.png"))
            else:
                print(f"[WARN] Not found: {s1s4S}")

        print("[OK] Visualizations completed successfully!")
        print(f"[OK] Output directory: {args.output_dir}")
    except Exception as e:
        print(f"[ERROR] Visualization failed: {e}")
        raise

if __name__ == "__main__":
    main()
