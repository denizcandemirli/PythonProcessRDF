#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Structural Visuals Bundle:

Heatmap (motif shares): Prefers struct_motif_shares_long_m5.csv, falls back to *_long.csv.

Dendrogram: Structural similarity matrix.

Radar Charts: System scores.

Options:

--normalize {0,1} : Normalize in radar charts (1) or use absolute values (0).

--force-overwrite : Silently overwrite existing PNG files (default=1).

--clean : Deletes old PNGs in the output directory.

--tag TAG : Adds a tag to titles/filenames (e.g., v25p2).

This version ensures the similarity matrix:

Strips headers/indices,

Reindexes squarely based on the union of row/column names,

Enforces symmetry (A↔B),

Cleans NaN/Inf values, sets the diagonal to 1.0, and clips to the [0,1] range.
"""

import os, json, argparse, glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform

# ---------- yardımcılar ----------

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def read_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return pd.read_csv(path)

def save_csv(df: pd.DataFrame, path: str):
    df.to_csv(path, index=False)

# ---------- veri okuyucular ----------

def read_motif_shares_long(in_dir: str) -> pd.DataFrame:
    """Öncelik: *_long_m5 → *_long → wide (melt). Beklenen kolonlar: model,motif,share"""
    c1 = os.path.join(in_dir, "struct_motif_shares_long_m5.csv")
    c2 = os.path.join(in_dir, "struct_motif_shares_long.csv")
    c3 = os.path.join(in_dir, "struct_motif_shares.csv")

    if os.path.exists(c1):
        df = read_csv(c1)
    elif os.path.exists(c2):
        df = read_csv(c2)
    elif os.path.exists(c3):
        wide = read_csv(c3)
        motif_cols = [c for c in wide.columns if c != "model"]
        df = wide.melt(id_vars="model", value_vars=motif_cols,
                       var_name="motif", value_name="share")
    else:
        raise FileNotFoundError("Motif shares dosyası bulunamadı (m5/long/wide).")

    df.columns = [c.strip() for c in df.columns]
    need = {"model", "motif", "share"}
    if not need.issubset(set(df.columns)):
        raise ValueError(f"Motif shares 'long' beklenen kolonlara sahip değil. Algılanan: {list(df.columns)}")
    df["share"] = pd.to_numeric(df["share"], errors="coerce").fillna(0.0)
    return df

def _strip_all(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [str(c).strip() for c in df.columns]
    if df.index.name:
        df.index = df.index.map(lambda x: str(x).strip())
    else:
        df.index = pd.Index([str(x).strip() for x in df.index], name=df.index.name)
    return df

def read_similarity_matrix(in_dir: str) -> tuple[pd.DataFrame, list]:
    """struct_similarity_matrix_used.csv varsa onu, yoksa struct_similarity_matrix.csv okur."""
    cands = [
        os.path.join(in_dir, "struct_similarity_matrix_used.csv"),
        os.path.join(in_dir, "struct_similarity_matrix.csv"),
    ]
    for p in cands:
        if os.path.exists(p):
            raw = read_csv(p)
            break
    else:
        raise FileNotFoundError("struct_similarity_matrix(.csv/_used.csv) bulunamadı.")

    raw = _strip_all(raw)

    # 'model' sütunu varsa indeks yap
    if "model" in raw.columns:
        raw = raw.set_index("model")

    raw = _strip_all(raw)

    row_names = [str(i).strip() for i in raw.index.tolist()]
    col_names = [str(c).strip() for c in raw.columns.tolist()]
    all_names = list(dict.fromkeys(row_names + col_names))  # sırayı koru

    sim = raw.apply(pd.to_numeric, errors="coerce")
    sim = sim.reindex(index=all_names, columns=all_names)
    sim = sim.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    sim = (sim + sim.T) / 2.0
    np.fill_diagonal(sim.values, 1.0)
    sim = sim.clip(0.0, 1.0)

    labels = list(sim.index)
    return sim, labels

def read_system_scores(in_dir: str) -> pd.DataFrame:
    """struct_system_scores_used.csv varsa onu, yoksa struct_system_scores.csv okur."""
    cands = [
        os.path.join(in_dir, "struct_system_scores_used.csv"),
        os.path.join(in_dir, "struct_system_scores.csv"),
    ]
    for p in cands:
        if os.path.exists(p):
            df = read_csv(p)
            break
    else:
        raise FileNotFoundError("struct_system_scores(.csv/_used.csv) bulunamadı.")

    df.columns = [c.strip() for c in df.columns]
    need = {"model", "frame", "wall", "braced", "dual"}
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise ValueError(f"System scores eksik kolonlar: {miss}")

    for c in ["frame", "wall", "braced", "dual"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
        df[c] = df[c].clip(lower=0.0)
    return df

# ---------- çizimler ----------

def plot_heatmap(df_long: pd.DataFrame, out_png: str, dpi=150, tag_note=None):
    piv = df_long.pivot_table(index="model", columns="motif",
                              values="share", fill_value=0.0, aggfunc="mean")

    def motif_key(s: str):
        s = str(s)
        try:
            if s.startswith("M2b"): return (2.1, s)
            if s.startswith("M"): return (float(s[1]), s)
            return (9.0, s)
        except Exception:
            return (9.0, s)

    piv = piv.reindex(sorted(piv.columns, key=motif_key), axis=1)

    fig, ax = plt.subplots(figsize=(1.2 + 0.5*piv.shape[1], 0.9 + 0.35*piv.shape[0]))
    im = ax.imshow(piv.values, aspect="auto", cmap="viridis")
    ax.set_yticks(range(len(piv.index))); ax.set_yticklabels(piv.index)
    ax.set_xticks(range(len(piv.columns)))
    ax.set_xticklabels(piv.columns, rotation=45, ha="right")
    title = "Structural Motif Shares"
    if tag_note: title += f" ({tag_note})"
    ax.set_title(title)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04); cb.set_label("share (0–1)")
    fig.tight_layout(); fig.savefig(out_png, dpi=dpi, bbox_inches="tight"); plt.close(fig)

def plot_dendrogram(sim_df: pd.DataFrame, labels, out_png: str, dpi=150, tag_note=None):
    dist = 1.0 - sim_df.values
    dist = np.where(np.isfinite(dist), dist, 0.0)
    condensed = squareform(dist, checks=False)

    fig, ax = plt.subplots(figsize=(9, 4.8))
    Z = linkage(condensed, method="average")
    dendrogram(Z, labels=labels, orientation="left", ax=ax, color_threshold=0.0)
    title = "Structural Similarity (matrix) – Dendrogram"
    if tag_note: title += f" ({tag_note})"
    ax.set_title(title)
    fig.tight_layout(); fig.savefig(out_png, dpi=dpi, bbox_inches="tight"); plt.close(fig)

def plot_radars(df_scores: pd.DataFrame, out_dir: str, normalize=1, dpi=150, tag_note=None):
    ensure_dir(os.path.join(out_dir, "radars"))
    cats = ["frame","wall","braced","dual"]
    theta = np.linspace(0, 2*np.pi, len(cats), endpoint=False)
    theta = np.concatenate([theta, theta[:1]])

    # overlay
    fig_o, ax_o = plt.subplots(subplot_kw={"projection":"polar"}, figsize=(9,6.6))
    ax_o.set_xticks(theta[:-1]); ax_o.set_xticklabels(cats)
    ax_o.set_ylim(0,1); ax_o.set_yticks([0.25,0.5,0.75,1.0]); ax_o.set_yticklabels(["0.25","0.50","0.75","1.00"])
    ax_o.set_title("Structural System Profiles (overlay)" + (f" – {tag_note}" if tag_note else ""))

    for _, row in df_scores.iterrows():
        vals = [row[c] for c in cats]
        if normalize:
            mx = max(1e-9, max(vals)); vals = [v/mx for v in vals]
        vals = np.clip(vals, 0.0, 1.0); vals = np.concatenate([vals, vals[:1]])
        ax_o.plot(theta, vals, linewidth=1.6, alpha=0.9, label=row["model"])
        ax_o.fill(theta, vals, alpha=0.08)

        fig, ax = plt.subplots(subplot_kw={"projection":"polar"}, figsize=(8,8))
        ax.set_xticks(theta[:-1]); ax.set_xticklabels(cats)
        ax.set_ylim(0,1); ax.set_yticks([0.25,0.5,0.75,1.0]); ax.set_yticklabels(["0.25","0.50","0.75","1.00"])
        ax.set_title(f"Structural System Profile\n{row['model']}" + (f" ({tag_note})" if tag_note else ""))
        ax.plot(theta, vals, linewidth=1.8); ax.fill(theta, vals, alpha=0.1)
        out_single = os.path.join(out_dir, "radars", f"radar_{row['model'].replace(os.sep,'_')}.png")
        fig.tight_layout(); fig.savefig(out_single, dpi=dpi, bbox_inches="tight"); plt.close(fig)

    ax_o.legend(loc="center left", bbox_to_anchor=(1.05, 0.5))
    fig_o.tight_layout()
    out_overlay = os.path.join(out_dir, "radar_all_models.png")
    fig_o.savefig(out_overlay, dpi=dpi, bbox_inches="tight"); plt.close(fig_o)

# ---------- ana akış ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--normalize", type=int, default=1)
    ap.add_argument("--force-overwrite", type=int, default=1)
    ap.add_argument("--clean", action="store_true")
    ap.add_argument("--tag", type=str, default="")
    ap.add_argument("--dpi", type=int, default=150)
    args = ap.parse_args()

    ensure_dir(args.out_dir)
    if args.clean:
        for p in glob.glob(os.path.join(args.out_dir, "*.png")):
            try: os.remove(p)
            except: pass
        for p in glob.glob(os.path.join(args.out_dir, "radars", "*.png")):
            try: os.remove(p)
            except: pass

    tag_note = args.tag if args.tag else None

    # 1) motif shares (long)
    df_long = read_motif_shares_long(args.in_dir)
    save_csv(df_long, os.path.join(args.out_dir, "motif_shares_long_used.csv"))

    # 2) similarity matrix (kare/simetrik hale getiriliyor)
    sim_df, labels = read_similarity_matrix(args.in_dir)
    save_csv(sim_df.reset_index().rename(columns={"index":"model"}),
             os.path.join(args.out_dir, "struct_similarity_matrix_used.csv"))

    # 3) system scores
    df_scores = read_system_scores(args.in_dir)
    save_csv(df_scores, os.path.join(args.out_dir, "struct_system_scores_used.csv"))

    # çizimler
    plot_heatmap(df_long, os.path.join(args.out_dir, "struct_motif_heatmap.png"),
                 dpi=args.dpi, tag_note=tag_note)
    plot_dendrogram(sim_df, labels, os.path.join(args.out_dir, "struct_dendrogram.png"),
                    dpi=args.dpi, tag_note=tag_note)
    plot_radars(df_scores, args.out_dir, normalize=args.normalize,
                dpi=args.dpi, tag_note=tag_note)

    with open(os.path.join(args.out_dir, "visual_summary.json"), "w", encoding="utf-8") as f:
        json.dump({
            "in_dir": args.in_dir,
            "out_dir": args.out_dir,
            "normalize": args.normalize,
            "tag": args.tag,
            "files_written": [
                "struct_motif_heatmap.png", "struct_dendrogram.png",
                "radar_all_models.png", "radars/*",
                "motif_shares_long_used.csv",
                "struct_similarity_matrix_used.csv",
                "struct_system_scores_used.csv"
            ]
        }, f, indent=2, ensure_ascii=False)

    print(f"[OK] Visuals written to: {args.out_dir}")

if __name__ == "__main__":
    main()
