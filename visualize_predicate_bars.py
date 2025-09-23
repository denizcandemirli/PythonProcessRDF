
# visualize_predicate_bars.py
# Usage examples (PowerShell, venv active):
#   python visualize_predicate_bars.py --pred-csv ".\repro_pack\output\04 - Pairwise_Diffs\Typed_Edge\typededge_predicate_contrib_Building_05_DG.rdf__Building_06_DG.rdf.csv" --out ".\repro_pack\output\04 - Pairwise_Diffs\Typed_Edge" --tag "05-06" --topk 8
#   python visualize_predicate_bars.py --pred-csv ".\repro_pack\output\04 - Pairwise_Diffs\Typed_Edge\typededge_predicate_contrib_Option03_Revising_DG.rdf__Option04_Rev03_DG.rdf.csv" --out ".\repro_pack\output\04 - Pairwise_Diffs\Typed_Edge" --tag "03-04" --topk 8
#   python visualize_predicate_bars.py --pred-csv ".\repro_pack\output\04 - Pairwise_Diffs\Typed_Edge\typededge_predicate_contrib_Building_05_DG.rdf__Option04_Rev03_DG.rdf.csv" --out ".\repro_pack\output\04 - Pairwise_Diffs\Typed_Edge" --tag "05-04" --topk 8
#
import os, argparse, pandas as pd
import matplotlib.pyplot as plt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-csv", required=True, help="typededge_predicate_contrib_*.csv path")
    ap.add_argument("--out", default=".", help="Output folder for PNGs")
    ap.add_argument("--tag", default="pair", help="Short label used in figure titles and filenames (e.g., 05-06)")
    ap.add_argument("--topk", type=int, default=8, help="Top-k predicates to plot")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    df = pd.read_csv(args.pred_csv)
    for col in ["product_contrib_share","abs_pct_diff_sum"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Top by contribution share
    df1 = df.sort_values("product_contrib_share", ascending=False).head(args.topk)
    plt.figure(figsize=(8,5))
    plt.bar(df1["pred"], df1["product_contrib_share"])
    plt.title(f"{args.tag} • Predicate contribution share (top {args.topk})")
    plt.ylabel("product_contrib_share")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    f1 = os.path.join(args.out, f"predicate_share_{args.tag}.png")
    plt.savefig(f1, dpi=150)
    plt.close()

    # Top by difference sum
    df2 = df.sort_values("abs_pct_diff_sum", ascending=False).head(args.topk)
    plt.figure(figsize=(8,5))
    plt.bar(df2["pred"], df2["abs_pct_diff_sum"])
    plt.title(f"{args.tag} • Predicate difference sum (top {args.topk})")
    plt.ylabel("abs_pct_diff_sum")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    f2 = os.path.join(args.out, f"predicate_diffsum_{args.tag}.png")
    plt.savefig(f2, dpi=150)
    plt.close()

    print("Saved:", f1)
    print("Saved:", f2)

if __name__ == "__main__":
    main()
