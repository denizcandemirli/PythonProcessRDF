# -*- coding: utf-8 -*-
"""
Combine multiple similarity matrices into a single Total Similarity.

Inputs (CSV) - tolerate both WIDE (square matrix) and LONG (pairwise) forms:
 - similarity_content_cosine.csv
 - similarity_typed_edge_cosine.csv
 - similarity_edge_sets_jaccard.csv
 - similarity_structural_final.csv  (from subgraph_similarity_v2.py)

Outputs (default): repro_pack/output/06 - Total_Similarity/
 - total_similarity_matrix.csv
 - pairwise_total_summary.csv
 - weights_used.json
"""

import os, argparse, json
import pandas as pd
import numpy as np

def _clean_label(x: str) -> str:
    if not isinstance(x, str):
        x = str(x)
    x = x.strip().strip('"').strip("'")
    return x

def _is_long_form(df: pd.DataFrame):
    cols = {c.lower() for c in df.columns}
    return (("model_a" in cols and "model_b" in cols) or
            ("a" in cols and "b" in cols) or
            ("source" in cols and "target" in cols))

def _pivot_long(df: pd.DataFrame) -> pd.DataFrame:
    # normalize column names
    cols = {c.lower(): c for c in df.columns}
    def pick(*names):
        for n in names:
            if n in cols: return cols[n]
        return None

    ca = pick("model_a","a","source")
    cb = pick("model_b","b","target")
    if ca is None or cb is None:
        raise SystemExit("[pivot] Could not find model_a/model_b columns in long-form CSV")

    # pick a numeric value column (prefer common names)
    pref = [ "cosine","overlap","jaccard","final","score","value" ]
    cand = [c for c in df.columns if c not in [ca,cb]]
    valcol = None
    for p in pref:
        for c in cand:
            if c.lower() == p:
                valcol = c; break
        if valcol: break
    if valcol is None:
        # choose the last numeric column
        num_cols = [c for c in cand if pd.to_numeric(df[c], errors="coerce").notna().any()]
        if not num_cols:
            raise SystemExit("[pivot] No numeric value column found in long-form CSV")
        valcol = num_cols[-1]

    tmp = df[[ca,cb,valcol]].copy()
    tmp[ca] = tmp[ca].map(_clean_label)
    tmp[cb] = tmp[cb].map(_clean_label)
    tmp[valcol] = pd.to_numeric(tmp[valcol], errors="coerce")
    mat = tmp.pivot_table(index=ca, columns=cb, values=valcol, aggfunc="mean")

    # symmetrize to square with union of labels
    idx = sorted(set(map(_clean_label, mat.index)) | set(map(_clean_label, mat.columns)))
    mat = mat.reindex(index=idx, columns=idx)
    # fill symmetric entries
    M = mat.values.astype(float)
    # mirror where one side exists
    M = np.where(np.isnan(M), np.transpose(M), M)
    # set diagonal to 1, clamp
    np.fill_diagonal(M, 1.0)
    M = np.nan_to_num(M, nan=0.0)
    M = (M + M.T) / 2.0
    M = np.clip(M, 0.0, 1.0)
    return pd.DataFrame(M, index=idx, columns=idx)

def read_sim(path: str) -> pd.DataFrame:
    # load raw
    df_raw = pd.read_csv(path)
    # if looks like long form (pairwise), pivot it
    if _is_long_form(df_raw):
        M = _pivot_long(df_raw)
        print(f"[read_sim] Parsed LONG form → {path} shape={M.shape}")
        return M

    # else treat as WIDE
    df = df_raw.copy()
    # set first column as index if it looks like labels (non-numeric)
    if df.shape[1] >= 2:
        df.iloc[:,0] = df.iloc[:,0].map(_clean_label)
        df = df.set_index(df.columns[0])

    # drop 'Unnamed' junk
    drop_cols = [c for c in df.columns if str(c).lower().startswith("unnamed")]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    # clean labels
    df.index = [ _clean_label(i) for i in df.index ]
    df.columns = [ _clean_label(c) for c in df.columns ]

    # force numeric
    df_num = df.apply(pd.to_numeric, errors="coerce")
    # drop all-NaN rows/cols
    df_num = df_num.dropna(axis=0, how="all").dropna(axis=1, how="all")

    # try align by intersection of labels
    idx = set(map(_clean_label, df_num.index))
    cols = set(map(_clean_label, df_num.columns))
    common = sorted(idx & cols)

    if common:
        df_num = df_num.loc[common, common]
    else:
        # fallback: if square numeric but labels don't match, assume order=order
        if df_num.shape[0] == df_num.shape[1] and df_num.shape[0] > 0:
            print(f"[read_sim] No label intersection; falling back to ORDER-BASED labeling for {path}")
            labels = [ _clean_label(x) for x in df_num.index ]
            df_num.columns = labels
        else:
            raise SystemExit(f"[read_sim] Could not align labels in: {path}")

    # symmetrize and clamp
    M = df_num.values.astype(float)
    M = (M + M.T) / 2.0
    np.fill_diagonal(M, 1.0)
    M = np.clip(M, 0.0, 1.0)
    out = pd.DataFrame(M, index=df_num.index, columns=df_num.columns)
    print(f"[read_sim] Parsed WIDE form → {path} shape={out.shape}")
    return out

def align_mats(mats):
    sets = [set(m.index) for m in mats]
    common = sorted(set.intersection(*sets))
    if not common:
        raise SystemExit("No common model names across matrices. Check your CSVs.")
    out = [m.loc[common, common].copy() for m in mats]
    print("[align] common labels:", common)
    return out, common

def convex_combine(mats, weights):
    w = np.array(weights, dtype=float)
    if (w < 0).any():
        raise ValueError("Weights must be >= 0.")
    if w.sum() == 0:
        raise ValueError("Sum of weights must be > 0.")
    w = w / w.sum()
    acc = np.zeros_like(mats[0].values, dtype=float)
    for Mi, wi in zip(mats, w):
        acc += wi * Mi.values
    acc = np.clip(acc, 0.0, 1.0)
    return pd.DataFrame(acc, index=mats[0].index, columns=mats[0].columns), w.tolist()

def pairwise_table(S: pd.DataFrame, comps: dict):
    rows = []
    models = S.index.tolist()
    for i in range(len(models)):
        for j in range(i+1, len(models)):
            A, B = models[i], models[j]
            row = {"model_A": A, "model_B": B, "total": float(S.loc[A,B])}
            for k, M in comps.items():
                row[k] = float(M.loc[A,B])
            rows.append(row)
    return pd.DataFrame(rows).sort_values("total", ascending=False)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--content-cos", required=True)
    ap.add_argument("--typed-cos", required=True)
    ap.add_argument("--edge-jaccard", required=True)
    ap.add_argument("--motif-final", required=True)
    ap.add_argument("--w-content", type=float, default=0.30)
    ap.add_argument("--w-typed", type=float, default=0.20)
    ap.add_argument("--w-edge", type=float, default=0.10)
    ap.add_argument("--w-motif", type=float, default=0.40)
    ap.add_argument("--out-dir", default=os.path.join(".","repro_pack","output","06 - Total_Similarity"))
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    S_content = read_sim(args.content_cos)
    S_typed   = read_sim(args.typed_cos)
    S_edge    = read_sim(args.edge_jaccard)
    S_motif   = read_sim(args.motif_final)

    mats, common = align_mats([S_content, S_typed, S_edge, S_motif])
    S_content, S_typed, S_edge, S_motif = mats

    weights = [args.w_content, args.w_typed, args.w_edge, args.w_motif]
    S_total, w_norm = convex_combine([S_content, S_typed, S_edge, S_motif], weights)

    # save
    p_mat = os.path.join(args.out_dir, "total_similarity_matrix.csv")
    S_total.to_csv(p_mat, index=True)

    comps = {
        "content_cos": S_content,
        "typed_edge_cos": S_typed,
        "edge_sets_jaccard": S_edge,
        "motif_final": S_motif
    }
    p_pairs = os.path.join(args.out_dir, "pairwise_total_summary.csv")
    pairwise_table(S_total, comps).to_csv(p_pairs, index=False)

    meta = {
        "weights_input": {
            "w_content": args.w_content,
            "w_typed": args.w_typed,
            "w_edge": args.w_edge,
            "w_motif": args.w_motif
        },
        "weights_normalized": {
            "w_content": w_norm[0],
            "w_typed": w_norm[1],
            "w_edge": w_norm[2],
            "w_motif": w_norm[3]
        },
        "models": common
    }
    with open(os.path.join(args.out_dir, "weights_used.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("[OK] Saved:")
    print(" -", p_mat)
    print(" -", p_pairs)
    print(" -", os.path.join(args.out_dir, "weights_used.json"))

if __name__ == "__main__":
    main()
