#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CLI wrapper for the S1→S4 structural channel (standards-aligned)

- Runs s1s4_struct.run_s1s4(models_dir, out_dir)
- Reads struct_similarity_s1s4.csv and prints/saves off-diagonal stats
- If a total_similarity_matrix.csv already exists in out_dir, also reports its stats
- Writes S1S4_run_summary.json in out_dir
"""

import argparse
import json
import os
from typing import Dict, Any

import numpy as np
import pandas as pd

from s1s4_struct import run_s1s4


def _mean_min_max_offdiag(df: pd.DataFrame) -> Dict[str, float]:
    """Return mean/min/max of the off-diagonal entries of a square similarity matrix."""
    A = df.values.astype(float)
    n = A.shape[0]
    if n == 0:
        return {"mean": float("nan"), "min": float("nan"), "max": float("nan")}
    off = A[~np.eye(n, dtype=bool)]
    if off.size == 0:
        return {"mean": float("nan"), "min": float("nan"), "max": float("nan")}
    return {
        "mean": float(off.mean()),
        "min": float(off.min()),
        "max": float(off.max()),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Run S1→S4 structural channel and report stats.")
    ap.add_argument("--models_dir", default="data/RDF_models", help="Folder with input RDF models")
    ap.add_argument("--out_dir",    default=".",               help="Where to write CSVs/figures/summary")
    args = ap.parse_args()

    # 1) Run S1→S4 and get file paths from the module
    res = run_s1s4(args.models_dir, args.out_dir)

    # 2) Load structural matrix and compute stats
    s_path = os.path.join(args.out_dir, "struct_similarity_s1s4.csv")
    if not os.path.isfile(s_path):
        raise FileNotFoundError(f"Expected structural matrix not found: {s_path}")

    S = pd.read_csv(s_path, index_col=0).astype(float)
    s_stats = _mean_min_max_offdiag(S)

    # 3) If a total matrix exists already, report its stats too (nice for one-shot runs)
    total_path = os.path.join(args.out_dir, "total_similarity_matrix.csv")
    t_stats: Dict[str, Any] = {}
    if os.path.isfile(total_path):
        T = pd.read_csv(total_path, index_col=0).astype(float)
        tm = _mean_min_max_offdiag(T)
        t_stats = {
            "total_mean_offdiag": round(tm["mean"], 4),
            "total_min_offdiag":  round(tm["min"], 4),
            "total_max_offdiag":  round(tm["max"], 4),
        }

    out = {
        "mean_offdiag_struct": round(s_stats["mean"], 4),
        "min_offdiag_struct":  round(s_stats["min"], 4),
        "max_offdiag_struct":  round(s_stats["max"], 4),
        "files": res.get("files", {}),
        **t_stats,
    }

    # 4) Persist + print
    sum_path = os.path.join(args.out_dir, "S1S4_run_summary.json")
    with open(sum_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
    