#!/usr/bin/env python3
"""
Structural system radar charts.
Reads struct_system_scores.csv (WIDE or LONG) and saves:
 - radar_all_models.png
 - radar_<model>.png (one per model)
 - struct_system_scores_used.csv
"""

import os, argparse, math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

SYS_COLS = ["frame","wall","dual","braced"]

def read_scores(in_dir: str) -> pd.DataFrame:
    path = os.path.join(in_dir, "struct_system_scores.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Cannot find {path}")

    df = pd.read_csv(path)
    # Try LONG: columns like ["model","system","score"]
    long_ok = set(["model","system","score"]).issubset(set(c.lower() for c in df.columns))
    if long_ok:
        # normalize header
        df.columns = [c.lower() for c in df.columns]
        piv = df.pivot_table(index="model", columns="system", values="score", aggfunc="mean").reset_index()
        piv.columns.name = None
        # ensure system columns exist
        for c in SYS_COLS:
            if c not in piv.columns:
                piv[c] = 0.0
        return piv[["model"]+SYS_COLS].copy()

    # Try WIDE: first col model, others system columns
    # Heuristic: first column is model id (stringy), rest numeric
    # Normalize headers
    df2 = df.copy()
    # if first col is unnamed, name it model
    if df2.columns[0].lower() not in ("model","models"):
        df2 = df2.rename(columns={df2.columns[0]:"model"})
    # lower-case known system columns
    newcols = []
    for c in df2.columns:
        lc = c.strip().lower()
        # some CSVs may carry "score_frame" style; normalize
        for s in SYS_COLS:
            if lc==s or lc.endswith("_"+s):
                lc = s
                break
        if c=="model": lc="model"
        newcols.append(lc)
    df2.columns = newcols
    for s in SYS_COLS:
        if s not in df2.columns:
            df2[s] = 0.0
        else:
            df2[s] = pd.to_numeric(df2[s], errors="coerce").fillna(0.0)
    return df2[["model"]+SYS_COLS].copy()

def radar_plot(df: pd.DataFrame, out_path: str, title: str, overlay=False):
    labels = SYS_COLS
    angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]  # close loop

    fig = plt.figure(figsize=(10,8))
    ax = plt.subplot(111, polar=True)
    ax.set_theta_offset(np.pi/2)
    ax.set_theta_direction(-1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=12)

    ax.set_rlabel_position(0)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25","0.50","0.75","1.00"], fontsize=10)
    ax.set_ylim(0, 1)

    if overlay:
        for _, row in df.iterrows():
            vals = [row[c] for c in labels]
            vals += vals[:1]
            ax.plot(angles, vals, linewidth=2, alpha=0.85, label=str(row["model"]))
            ax.fill(angles, vals, alpha=0.10)
        ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.10), fontsize=10)
    else:
        row = df.iloc[0]
        vals = [row[c] for c in labels]
        vals += vals[:1]
        ax.plot(angles, vals, linewidth=3, alpha=0.9)
        ax.fill(angles, vals, alpha=0.15)
        ax.set_title(str(row["model"]), y=1.08, fontsize=14)

    plt.title(title, y=1.10, fontsize=16)
    plt.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--normalize", type=int, default=0, help="1 => min-max per column to [0,1]")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    df = read_scores(args.in_dir)

    # Optional normalization (column-wise)
    if args.normalize:
        base = df[SYS_COLS].values.astype(float)
        mn = base.min(axis=0)
        mx = base.max(axis=0)
        rng = np.where((mx-mn)==0, 1.0, (mx-mn))
        norm = (base - mn)/rng
        df_norm = df.copy()
        df_norm[SYS_COLS] = norm
        df_used = df_norm
    else:
        # clip to [0,1] just in case
        df_used = df.copy()
        for c in SYS_COLS:
            df_used[c] = np.clip(pd.to_numeric(df_used[c], errors="coerce").fillna(0.0), 0.0, 1.0)

    # Save the table we used
    df_used.to_csv(os.path.join(args.out_dir, "struct_system_scores_used.csv"), index=False)

    # Overlay for all models
    radar_plot(df_used, os.path.join(args.out_dir, "radar_all_models.png"),
               title="Structural System Profile (all models)", overlay=True)

    # One radar per model
    for i in range(len(df_used)):
        radar_plot(df_used.iloc[[i]], os.path.join(args.out_dir, f"radar_{df_used.iloc[i]['model'].replace('/','_')}.png"),
                   title="Structural System Profile")

if __name__ == "__main__":
    main()
