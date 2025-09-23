import os
import pandas as pd
import numpy as np


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
TOTAL_DIR = os.path.join(ROOT, "repro_pack", "output", "06 - Total_Similarity")


def simulate_missing(df: pd.DataFrame, frac: float, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    out = df.copy()
    n = len(out)
    k = int(frac * n)
    idx = rng.choice(n, size=k, replace=False)
    # zero-out S_content and S_typed to mimic missing semantic features
    out.loc[idx, ["S_content","S_typed"]] = 0.0
    out["S_total_missing"] = 0.30*out["S_content"] + 0.20*out["S_typed"] + 0.10*out["S_edge"] + 0.40*out["S_struct"]
    return out


def main():
    p = os.path.join(TOTAL_DIR, "pairwise_total_summary.csv")
    df = pd.read_csv(p)
    df = df.rename(columns={
        "model_A":"A","model_B":"B","total":"S_total",
        "content_cos":"S_content","typed_edge_cos":"S_typed",
        "edge_sets_jaccard":"S_edge",
        "structural_similarity":"S_struct",
    })
    rows = []
    for frac in [0.10, 0.30]:
        M = simulate_missing(df, frac)
        rows.append({
            "missing_frac": frac,
            "mean_S_total": float(df["S_total"].mean()),
            "mean_S_total_missing": float(M["S_total_missing"].mean()),
            "delta_mean": float(M["S_total_missing"].mean() - df["S_total"].mean()),
        })
    out_csv = os.path.join(ROOT, "missingness_sensitivity.csv")
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print("WROTE", out_csv)


if __name__ == "__main__":
    main()


