import os
import csv
import random
import pandas as pd


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
TOTAL_DIR = os.path.join(ROOT, "repro_pack", "output", "06 - Total_Similarity")


ABLATION_SETUPS = [
    ("S0_all", {"content":0.30, "typed":0.20, "edge":0.10, "struct":0.40}),
    ("S_sem_only", {"content":0.0,  "typed":0.0,  "edge":0.0,  "struct":1.0}),
    ("S_geom_only", {"content":0.50, "typed":0.25, "edge":0.25, "struct":0.0}),
    ("materials_off", {"content":0.35, "typed":0.15, "edge":0.10, "struct":0.40}),
    ("systems_off",   {"content":0.30, "typed":0.20, "edge":0.10, "struct":0.40}),
]


def load_pairs():
    p = os.path.join(TOTAL_DIR, "pairwise_total_summary.csv")
    df = pd.read_csv(p)
    # tolerate canonical columns already in file
    return df.rename(columns={
        "model_A":"A","model_B":"B","total":"S_total",
        "content_cos":"S_content","typed_edge_cos":"S_typed",
        "edge_sets_jaccard":"S_edge",
        "structural_similarity":"S_struct",
    })


def recompute(df: pd.DataFrame, w: dict) -> pd.DataFrame:
    out = df.copy()
    s = w.get("content",0)*out["S_content"] + w.get("typed",0)*out["S_typed"] + w.get("edge",0)*out["S_edge"] + w.get("struct",0)*out["S_struct"]
    denom = sum(w.values()) or 1.0
    out["S_total"] = s/denom
    return out


def main():
    df = load_pairs()
    rows = []
    for name, W in ABLATION_SETUPS:
        M = recompute(df, W)
        rows.append({
            "setup": name,
            "w_content": W.get("content",0),
            "w_typed": W.get("typed",0),
            "w_edge": W.get("edge",0),
            "w_struct": W.get("struct",0),
            "mean_S_total": float(M["S_total"].mean()),
            "std_S_total": float(M["S_total"].std(ddof=0)),
            "max_S_total": float(M["S_total"].max()),
            "min_S_total": float(M["S_total"].min()),
        })

    out_csv = os.path.join(ROOT, "ablation_results.csv")
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print("WROTE", out_csv)


if __name__ == "__main__":
    main()


