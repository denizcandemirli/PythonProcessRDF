#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File: total_similarity.py
Purpose: Combine multiple similarity matrices into a single Total Similarity.
Input: CSV files from content, typed-edge, edge-set, and structural similarity
Output: total_similarity_matrix.csv, pairwise_total_summary.csv

This script combines similarity matrices from:
- Content similarity (cosine)
- Typed-edge similarity (cosine)
- Edge-set similarity (Jaccard)
- Structural similarity (motif-based)

Uses validated weight combination: {0.30, 0.20, 0.10, 0.40}

Author: Deniz Demirli
Supervisor: Dr. Chao Li (TUM)
Version: 2025.10
License: SPDX-License-Identifier: CC-BY-4.0
"""

__version__ = "2025.10"
__author__ = "Deniz Demirli"
__supervisor__ = "Dr. Chao Li (TUM)"

import os
import argparse
import json
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np

# ---------------------------
# Configuration
# ---------------------------

# Validated fusion weights
FUSION_WEIGHTS = {
    "content": 0.30,
    "typed": 0.20,
    "edge": 0.10,
    "struct": 0.40
}

# ---------------------------
# Helper Functions
# ---------------------------

def ensure_dir(path: str):
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)

def _clean_label(x: str) -> str:
    """Clean and normalize label strings."""
    if not isinstance(x, str):
        x = str(x)
    x = x.strip().strip('"').strip("'")
    return x

def _is_long_form(df: pd.DataFrame) -> bool:
    """Check if DataFrame is in long (pairwise) format."""
    cols = {c.lower() for c in df.columns}
    return (("model_a" in cols and "model_b" in cols) or
            ("a" in cols and "b" in cols) or
            ("source" in cols and "target" in cols))

def _pivot_long(df: pd.DataFrame) -> pd.DataFrame:
    """Convert long-form DataFrame to wide-form matrix."""
    # normalize column names
    cols = {c.lower(): c for c in df.columns}
    def pick(*names):
        for n in names:
            if n in cols: 
                return cols[n]
        return None

    ca = pick("model_a", "a", "source")
    cb = pick("model_b", "b", "target")
    if ca is None or cb is None:
        raise SystemExit("[pivot] Could not find model_a/model_b columns in long-form CSV")

    # pick a numeric value column (prefer common names)
    pref = ["cosine", "overlap", "jaccard", "final", "score", "value"]
    cand = [c for c in df.columns if c not in [ca, cb]]
    valcol = None
    for p in pref:
        for c in cand:
            if c.lower() == p:
                valcol = c
                break
        if valcol: 
            break
    if valcol is None:
        # choose the last numeric column
        num_cols = [c for c in cand if pd.to_numeric(df[c], errors="coerce").notna().any()]
        if not num_cols:
            raise SystemExit("[pivot] No numeric value column found in long-form CSV")
        valcol = num_cols[-1]

    tmp = df[[ca, cb, valcol]].copy()
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

def read_similarity_matrix(path: str) -> pd.DataFrame:
    """
    Read similarity matrix from CSV file.
    Handles both wide (matrix) and long (pairwise) formats.
    """
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
        df.iloc[:, 0] = df.iloc[:, 0].map(_clean_label)
        df = df.set_index(df.columns[0])

    # drop 'Unnamed' junk
    drop_cols = [c for c in df.columns if str(c).lower().startswith("unnamed")]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    # clean labels
    df.index = [_clean_label(i) for i in df.index]
    df.columns = [_clean_label(c) for c in df.columns]

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
            labels = [_clean_label(x) for x in df_num.index]
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

def align_matrices(matrices: List[pd.DataFrame]) -> Tuple[List[pd.DataFrame], List[str]]:
    """Align multiple similarity matrices to common model set."""
    sets = [set(m.index) for m in matrices]
    common = sorted(set.intersection(*sets))
    if not common:
        raise SystemExit("No common model names across matrices. Check your CSVs.")
    out = [m.loc[common, common].copy() for m in matrices]
    print("[align] common labels:", common)
    return out, common

def convex_combine(matrices: List[pd.DataFrame], weights: List[float]) -> Tuple[pd.DataFrame, List[float]]:
    """Combine matrices using convex combination with given weights."""
    w = np.array(weights, dtype=float)
    if (w < 0).any():
        raise ValueError("Weights must be >= 0.")
    if w.sum() == 0:
        raise ValueError("Sum of weights must be > 0.")
    w = w / w.sum()  # normalize weights
    acc = np.zeros_like(matrices[0].values, dtype=float)
    for Mi, wi in zip(matrices, w):
        acc += wi * Mi.values
    acc = np.clip(acc, 0.0, 1.0)
    return pd.DataFrame(acc, index=matrices[0].index, columns=matrices[0].columns), w.tolist()

def build_pairwise_summary(total_matrix: pd.DataFrame, component_matrices: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Build pairwise summary table from total and component matrices."""
    rows = []
    models = total_matrix.index.tolist()
    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            A, B = models[i], models[j]
            row = {"model_A": A, "model_B": B, "total": float(total_matrix.loc[A, B])}
            for k, M in component_matrices.items():
                row[k] = float(M.loc[A, B])
            rows.append(row)
    return pd.DataFrame(rows).sort_values("total", ascending=False)

# ---------------------------
# Main Function
# ---------------------------

def main():
    """Main execution function."""
    ap = argparse.ArgumentParser(description="Combine similarity matrices into total similarity")
    ap.add_argument("--content-cos", required=True,
                   help="Path to content cosine similarity CSV")
    ap.add_argument("--typed-cos", required=True,
                   help="Path to typed-edge cosine similarity CSV")
    ap.add_argument("--edge-jaccard", required=True,
                   help="Path to edge-set Jaccard similarity CSV")
    ap.add_argument("--struct-sim", required=True,
                   help="Path to structural similarity CSV")
    ap.add_argument("--struct-source", choices=["counts", "sp", "hybrid", "netlsd", "s1s4"], default="counts",
                   help="Structural similarity source: counts (motif counts), sp (significance profiles), hybrid (blend), netlsd (NetLSD heat-trace), s1s4 (S1→S4 structural channel)")
    ap.add_argument("--struct-sp", 
                   help="Path to SP-based structural similarity CSV (required if struct-source is 'sp' or 'hybrid')")
    ap.add_argument("--hybrid-alpha", type=float, default=0.5,
                   help="Blending weight for hybrid mode: alpha*SP + (1-alpha)*counts (default: 0.5)")
    ap.add_argument("--w-content", type=float, default=FUSION_WEIGHTS["content"],
                   help=f"Weight for content similarity (default: {FUSION_WEIGHTS['content']})")
    ap.add_argument("--w-typed", type=float, default=FUSION_WEIGHTS["typed"],
                   help=f"Weight for typed-edge similarity (default: {FUSION_WEIGHTS['typed']})")
    ap.add_argument("--w-edge", type=float, default=FUSION_WEIGHTS["edge"],
                   help=f"Weight for edge-set similarity (default: {FUSION_WEIGHTS['edge']})")
    ap.add_argument("--w-struct", type=float, default=FUSION_WEIGHTS["struct"],
                   help=f"Weight for structural similarity (default: {FUSION_WEIGHTS['struct']})")
    ap.add_argument("--output-dir", 
                   default=os.path.join(".", "repro_pack", "output", "06 - Total_Similarity"),
                   help="Directory to save results")
    
    args = ap.parse_args()
    
    # Ensure output directory exists
    ensure_dir(args.output_dir)
    
    print(f"[INFO] Combining similarity matrices...")
    print(f"[INFO] Content similarity: {args.content_cos}")
    print(f"[INFO] Typed-edge similarity: {args.typed_cos}")
    print(f"[INFO] Edge-set similarity: {args.edge_jaccard}")
    print(f"[INFO] Structural similarity: {args.struct_sim}")
    print(f"[INFO] Structural source: {args.struct_source}")
    if args.struct_source in ["sp", "hybrid"]:
        print(f"[INFO] SP similarity: {args.struct_sp}")
        if args.struct_source == "hybrid":
            print(f"[INFO] Hybrid alpha: {args.hybrid_alpha}")
    print(f"[INFO] Output directory: {args.output_dir}")
    
    try:
        # Read similarity matrices
        S_content = read_similarity_matrix(args.content_cos)
        S_typed = read_similarity_matrix(args.typed_cos)
        S_edge = read_similarity_matrix(args.edge_jaccard)
        
        # Handle structural similarity based on source type
        if args.struct_source == "counts":
            S_struct = read_similarity_matrix(args.struct_sim)
        elif args.struct_source == "sp":
            if not args.struct_sp:
                raise ValueError("--struct-sp is required when struct-source is 'sp'")
            S_struct = read_similarity_matrix(args.struct_sp)
        elif args.struct_source == "hybrid":
            if not args.struct_sp:
                raise ValueError("--struct-sp is required when struct-source is 'hybrid'")
            S_struct_counts = read_similarity_matrix(args.struct_sim)
            S_struct_sp = read_similarity_matrix(args.struct_sp)
            # Blend the two matrices
            S_struct = args.hybrid_alpha * S_struct_sp + (1 - args.hybrid_alpha) * S_struct_counts
            # Ensure values are in [0,1] range
            S_struct = np.clip(S_struct, 0.0, 1.0)
            print(f"[INFO] Hybrid structural similarity: {args.hybrid_alpha}*SP + {1-args.hybrid_alpha}*counts")
        elif args.struct_source == "netlsd":
            # Use NetLSD structural similarity
            netlsd_path = "struct_similarity_netlsd.csv"
            if not os.path.exists(netlsd_path):
                raise ValueError(f"NetLSD similarity matrix not found: {netlsd_path}. Run structural_signature_netlsd.py first.")
            S_struct = read_similarity_matrix(netlsd_path)
            print(f"[INFO] Using NetLSD structural similarity: {netlsd_path}")
        elif args.struct_source == "s1s4":
            # Use S1→S4 structural similarity
            s1s4_path = "struct_similarity_s1s4.csv"
            if not os.path.exists(s1s4_path):
                raise ValueError(f"S1S4 similarity matrix not found: {s1s4_path}. Run run_s1s4_struct.py first.")
            S_struct = read_similarity_matrix(s1s4_path)
            print(f"[INFO] Using S1→S4 structural similarity: {s1s4_path}")
        
        # Align matrices to common model set
        matrices, common_models = align_matrices([S_content, S_typed, S_edge, S_struct])
        S_content, S_typed, S_edge, S_struct = matrices
        
        # Combine using convex combination
        weights = [args.w_content, args.w_typed, args.w_edge, args.w_struct]
        S_total, normalized_weights = convex_combine([S_content, S_typed, S_edge, S_struct], weights)
        
        # Save results
        matrix_path = os.path.join(args.output_dir, "total_similarity_matrix.csv")
        S_total.to_csv(matrix_path, index=True)
        
        # Build pairwise summary
        component_matrices = {
            "content_cos": S_content,
            "typed_edge_cos": S_typed,
            "edge_sets_jaccard": S_edge,
            "struct_sim": S_struct
        }
        pairwise_df = build_pairwise_summary(S_total, component_matrices)
        pairwise_path = os.path.join(args.output_dir, "pairwise_total_summary.csv")
        pairwise_df.to_csv(pairwise_path, index=False)
        
        # Save metadata
        meta = {
            "weights_input": {
                "w_content": args.w_content,
                "w_typed": args.w_typed,
                "w_edge": args.w_edge,
                "w_struct": args.w_struct
            },
            "weights_normalized": {
                "w_content": normalized_weights[0],
                "w_typed": normalized_weights[1],
                "w_edge": normalized_weights[2],
                "w_struct": normalized_weights[3]
            },
            "structural_source": {
                "type": args.struct_source,
                "struct_sim_file": args.struct_sim,
                "struct_sp_file": args.struct_sp if args.struct_source in ["sp", "hybrid"] else None,
                "hybrid_alpha": args.hybrid_alpha if args.struct_source == "hybrid" else None
            },
            "models": common_models
        }
        meta_path = os.path.join(args.output_dir, "weights_used.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        
        print(f"[OK] Total similarity computed successfully!")
        print(f"[OK] Similarity matrix: {matrix_path}")
        print(f"[OK] Pairwise summary: {pairwise_path}")
        print(f"[OK] Metadata: {meta_path}")
        print(f"[OK] Models: {len(common_models)} ({', '.join(common_models)})")
        print(f"[OK] Weights: content={normalized_weights[0]:.3f}, typed={normalized_weights[1]:.3f}, "
              f"edge={normalized_weights[2]:.3f}, struct={normalized_weights[3]:.3f}")
        
        # Print summary statistics
        total_similarities = S_total.values[np.triu_indices_from(S_total.values, k=1)]
        print(f"[STATS] Total similarity range: {total_similarities.min():.3f} - {total_similarities.max():.3f}")
        print(f"[STATS] Mean total similarity: {total_similarities.mean():.3f}")
        print(f"[STATS] Std deviation: {total_similarities.std():.3f}")
        
    except Exception as e:
        print(f"[ERROR] Failed to compute total similarity: {e}")
        raise

if __name__ == "__main__":
    main()
