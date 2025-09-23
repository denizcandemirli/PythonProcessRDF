#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Minimal Structural Visuals: HEATMAP + DENDROGRAM

Heatmap: struct_motif_shares_long_m5.csv -> struct_motif_shares_long.csv -> struct_motif_shares.csv (melted)

Dendrogram: struct_similarity_matrix_used.csv -> struct_similarity_matrix.csv

File Cleaning/Update Guarantee:

If a PNG file already exists, it is deleted first and then re-saved (reduces Windows file lock errors).

Ensures the similarity matrix is square/symmetric, cleans NaN/Inf values, sets the diagonal to 1.0, and clips values to the [0,1] range.

Writes the subsets of data used into the OUT folder as *_used.csv (for easier diagnostics).

Usage (PowerShell):

python .\visualize_struct_matrix_heatmap.py `
  --in-dir ".\repro_pack\output\07 - Structural_Extension_v25p2" `
  --out-dir ".\repro_pack\output\07c - Structural_Visuals" `
  --tag v25p2 `
  --clean
"""

import os, argparse, glob
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

def write_png_force(fig, out_png: str, dpi=150):
    # eski dosya varsa sil (Windows'ta viewer açıkken overwrite hatası için)
    try:
        if os.path.exists(out_png):
            os.remove(out_png)
    except Exception:
        pass
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

def motif_key(m: str):
    s = str(m)
    try:
        if s.startswith("M2b"): return (2.1, s)
        if s.startswith("M"):   return (float(s[1]), s)
        return (9.0, s)
    except Exception:
        return (9.0, s)

# ---------- veri okuyucular ----------

def read_motif_shares_long(in_dir: str) -> pd.DataFrame:
    c1 = os.path.join(in_dir, "struct_motif_shares_long_m5.csv")
    c2 = os.path.join(in_dir, "struct_motif_shares_long.csv")
    c3 = os.path.join(in_dir, "struct_motif_shares.csv")

    if os.path.exists(c1):
        df = read_csv(c1)
        source = os.path.basename(c1)
    elif os.path.exists(c2):
        df = read_csv(c2)
        source = os.path.basename(c2)
    elif os.path.exists(c3):
        wide = read_csv(c3)
        if "model" not in wide.columns:
            raise ValueError("struct_motif_shares.csv beklenen 'model' kolonu içermiyor.")
        motif_cols = [c for c in wide.columns if c != "model"]
        df = wide.melt(id_vars="model", value_vars=motif_cols,
                       var_name="motif", value_name="share")
        source = os.path.basename(c3) + " (melt)"
    else:
        raise FileNotFoundError("Motif shares dosyası bulunamadı (m5/long/wide).")

    df.columns = [c.strip() for c in df.columns]
    need = {"model","motif","share"}
    if not need.issubset(df.columns):
        raise ValueError(f"Motif shares 'long' beklenen kolonlara sahip değil. Algılanan: {list(df.columns)}")
    df["share"] = pd.to_numeric(df["share"], errors="coerce").fillna(0.0)
    # Model/başlık temizliği
    df["model"] = df["model"].astype(str).str.strip()
    df["motif"] = df["motif"].astype(str).str.strip()
    return df, source

def read_similarity_matrix(in_dir: str) -> tuple[pd.DataFrame, list, str]:
    cands = [
        os.path.join(in_dir, "struct_similarity_matrix_used.csv"),
        os.path.join(in_dir, "struct_similarity_matrix.csv"),
    ]
    for p in cands:
        if os.path.exists(p):
            raw = read_csv(p)
            src = os.path.basename(p)
            break
    else:
        raise FileNotFoundError("struct_similarity_matrix(.csv/_used.csv) bulunamadı.")

    # kolon/indeks strip
    raw.columns = [str(c).strip() for c in raw.columns]
    if "model" in raw.columns:
        raw = raw.set_index("model")
    raw.index = pd.Index([str(i).strip() for i in raw.index], name="model")

    # kare reindex + simetri
    rows = raw.index.tolist()
    cols = [str(c).strip() for c in raw.columns.tolist()]
    names = list(dict.fromkeys(rows + cols))
    sim = raw.apply(pd.to_numeric, errors="coerce")
    sim = sim.reindex(index=names, columns=names)
    sim = sim.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    sim = (sim + sim.T) / 2.0
    np.fill_diagonal(sim.values, 1.0)
    sim = sim.clip(0.0, 1.0)
    labels = list(sim.index)
    return sim, labels, src

# ---------- çizimler ----------

def make_heatmap(df_long: pd.DataFrame, out_png: str, tag_note=None, dpi=150):
    piv = df_long.pivot_table(index="model", columns="motif",
                              values="share", aggfunc="mean", fill_value=0.0)
    piv = piv.reindex(sorted(piv.columns, key=motif_key), axis=1)

    # Dinamik boyut (çok büyük olmasın diye ufak sınırlama)
    h = 0.9 + 0.35*len(piv.index)
    w = 1.2 + 0.5*len(piv.columns)
    h = min(max(h, 3.5), 20.0)
    w = min(max(w, 4.0), 24.0)

    fig, ax = plt.subplots(figsize=(w, h))
    im = ax.imshow(piv.values, aspect="auto", cmap="viridis")
    ax.set_yticks(range(len(piv.index))); ax.set_yticklabels(piv.index)
    ax.set_xticks(range(len(piv.columns))); ax.set_xticklabels(piv.columns, rotation=45, ha="right")
    title = "Structural Motif Shares"
    if tag_note: title += f" ({tag_note})"
    ax.set_title(title)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04); cb.set_label("share (0–1)")
    fig.tight_layout()
    write_png_force(fig, out_png, dpi=dpi)

def make_dendrogram(sim_df: pd.DataFrame, labels, out_png: str, tag_note=None, dpi=150):
    dist = 1.0 - sim_df.values
    dist = np.where(np.isfinite(dist), dist, 0.0)
    condensed = squareform(dist, checks=False)

    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    Z = linkage(condensed, method="average")
    dendrogram(Z, labels=labels, orientation="left", ax=ax, color_threshold=0.0)
    title = "Structural Similarity (matrix) – Dendrogram"
    if tag_note: title += f" ({tag_note})"
    ax.set_title(title)
    fig.tight_layout()
    write_png_force(fig, out_png, dpi=dpi)

# ---------- ana akış ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--tag", type=str, default="")
    ap.add_argument("--dpi", type=int, default=150)
    ap.add_argument("--clean", action="store_true")
    args = ap.parse_args()

    ensure_dir(args.out_dir)
    if args.clean:
        for p in glob.glob(os.path.join(args.out_dir, "struct_*.png")):
            try: os.remove(p)
            except: pass

    tag_note = args.tag if args.tag else None

    # 1) motif shares (long)
    df_long, src_long = read_motif_shares_long(args.in_dir)
    df_long.to_csv(os.path.join(args.out_dir, "motif_shares_long_used.csv"), index=False)

    # 2) similarity matrix
    sim_df, labels, src_sim = read_similarity_matrix(args.in_dir)
    sim_df.reset_index().rename(columns={"index":"model"}) \
         .to_csv(os.path.join(args.out_dir, "struct_similarity_matrix_used.csv"), index=False)

    # 3) çizimler
    make_heatmap(df_long, os.path.join(args.out_dir, "struct_motif_heatmap.png"),
                 tag_note=tag_note, dpi=args.dpi)
    make_dendrogram(sim_df, labels, os.path.join(args.out_dir, "struct_dendrogram.png"),
                    tag_note=tag_note, dpi=args.dpi)

    print("[INFO] motif source :", src_long)
    print("[INFO] matrix source:", src_sim)
    print(f"[OK] Heatmap & dendrogram written to: {args.out_dir}")

if __name__ == "__main__":
    main()
