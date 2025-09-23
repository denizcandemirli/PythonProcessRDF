
# pairwise_diffs_content.py
# Stage 1: For selected model pairs, analyze CONTENT features:
# - Features: rdf:type + hasFunction TYPEs + hasQuality TYPEs (counts from Feature Extraction CSVs)
# - Outputs per pair:
#     1) content_top_contributors_[A]__[B].csv    (biggest cosine numerator contributions a_i*b_i)
#     2) content_top_differences_[A]__[B].csv     (largest percent differences |pA - pB|)
#     3) content_summary_[A]__[B].csv             (cosine score, feature counts, key overlaps)
#
# Usage (from project root, venv active):
#   python pairwise_diffs_content.py
#     --feat-dir "repro_pack/output/01 - Building_Information - Feature Extraction"
#     --sim-dir  "repro_pack/output/02 - Similarity"
#     --out-dir  "repro_pack/output/04 - Pairwise_Diffs"
#     [--topk 20] [--pairs "ModelA.rdf,ModelB.rdf;ModelC.rdf,ModelD.rdf"]
#
# If --pairs is omitted, script will auto-pick top-3 pairs by final_similarity from similarity_combined_weighted.csv

import os, argparse, pandas as pd, numpy as np
from collections import Counter

def iri_label(iri: str) -> str:
    if not isinstance(iri, str): return str(iri)
    if '#' in iri:
        return iri.rsplit('#', 1)[-1]
    return iri.rstrip('/').rsplit('/', 1)[-1]

def read_hist(path, key_col, val_col):
    if not os.path.exists(path): 
        return Counter()
    try:
        df = pd.read_csv(path)
    except Exception:
        return Counter()
    if key_col not in df.columns or val_col not in df.columns:
        return Counter()
    cnt = Counter()
    for _, r in df.iterrows():
        try:
            k = str(r[key_col]); v = int(r[val_col])
            cnt[k] += v
        except Exception:
            continue
    return cnt

def build_content_vector(feat_dir, model_file):
    base = os.path.join(feat_dir, f"{model_file}_type_distribution.csv")
    type_cnt = read_hist(base, "type_iri", "count")

    base = os.path.join(feat_dir, f"{model_file}_function_type_histogram.csv")
    func_cnt = read_hist(base, "function_type_iri", "count")

    base = os.path.join(feat_dir, f"{model_file}_quality_type_histogram.csv")
    qual_cnt = read_hist(base, "quality_type_iri", "count")

    total = Counter(); total.update(type_cnt); total.update(func_cnt); total.update(qual_cnt)
    return total, {"type": type_cnt, "function_type": func_cnt, "quality_type": qual_cnt}

def cosine_from_counters(a: Counter, b: Counter) -> float:
    keys = set(a) | set(b)
    if not keys: return 0.0
    va = np.array([a.get(k,0) for k in keys], dtype=float)
    vb = np.array([b.get(k,0) for k in keys], dtype=float)
    dot = float(np.dot(va, vb))
    na = float(np.linalg.norm(va)); nb = float(np.linalg.norm(vb))
    if na == 0 or nb == 0: return 0.0
    return dot / (na * nb)

def pair_tables(modelA, modelB, vA, vB, topk, out_dir):
    keys = sorted(set(vA) | set(vB))
    sA = sum(vA.values()) if vA else 0
    sB = sum(vB.values()) if vB else 0

    rows = []
    for k in keys:
        a = vA.get(k, 0); b = vB.get(k, 0)
        pA = (a / sA) if sA else 0.0
        pB = (b / sB) if sB else 0.0
        prod = a * b
        rows.append({
            "feature_iri": k,
            "feature": iri_label(k),
            "count_A": a, "count_B": b,
            "pct_A": pA, "pct_B": pB,
            "abs_pct_diff": abs(pA - pB),
            "product_contrib": prod
        })
    df = pd.DataFrame(rows)

    top_contrib = df.sort_values(["product_contrib","count_A","count_B"], ascending=False).head(topk)
    top_diffs = df.sort_values(["abs_pct_diff","count_A","count_B"], ascending=False).head(topk)

    safe = lambda s: s.replace(os.sep, "_")
    pair_tag = f"{safe(modelA)}__{safe(modelB)}"
    out1 = os.path.join(out_dir, f"content_top_contributors_{pair_tag}.csv")
    out2 = os.path.join(out_dir, f"content_top_differences_{pair_tag}.csv")

    top_contrib.to_csv(out1, index=False)
    top_diffs.to_csv(out2, index=False)

    cos = cosine_from_counters(vA, vB)
    key_overlap = len([k for k in keys if (vA.get(k,0)>0 and vB.get(k,0)>0)])
    summ = pd.DataFrame([{
        "model_a": modelA, "model_b": modelB,
        "content_cosine_recomputed": cos,
        "features_nonzero_A": len([k for k in vA if vA[k]>0]),
        "features_nonzero_B": len([k for k in vB if vB[k]>0]),
        "features_overlap": key_overlap,
        "sum_counts_A": sA, "sum_counts_B": sB
    }])
    out3 = os.path.join(out_dir, f"content_summary_{pair_tag}.csv")
    summ.to_csv(out3, index=False)

    return out1, out2, out3

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat-dir", default=os.path.join(".", "repro_pack", "output", "01 - Building_Information - Feature Extraction"),
                    help="Feature Extraction CSV klasörü")
    ap.add_argument("--sim-dir",  default=os.path.join(".", "repro_pack", "output", "02 - Similarity"),
                    help="Similarity CSV klasörü (top-3 çift seçimi için)")
    ap.add_argument("--out-dir",  default=os.path.join(".", "repro_pack", "output", "04 - Pairwise_Diffs"),
                    help="Çıktı klasörü")
    ap.add_argument("--topk", type=int, default=20, help="Tablolarda gösterilecek üst öğe sayısı")
    ap.add_argument("--pairs", type=str, default="",
                    help="Model çiftleri; format: 'A.rdf,B.rdf;C.rdf,D.rdf' (boşsa top-3 otomatik seçilir)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    pairs = []
    if args.pairs.strip():
        for seg in args.pairs.split(";"):
            if not seg.strip(): continue
            a,b = [x.strip() for x in seg.split(",")]
            pairs.append((a,b))
    else:
        sim_csv = os.path.join(args.sim_dir, "similarity_combined_weighted.csv")
        if not os.path.exists(sim_csv):
            raise FileNotFoundError(f"Bulunamadı: {sim_csv}. --pairs ile çiftleri elle verebilirsiniz.")
        sim = pd.read_csv(sim_csv)
        sim = sim.sort_values("final_similarity", ascending=False).head(3)
        for _, r in sim.iterrows():
            pairs.append((r["model_a"], r["model_b"]))

    manifest_rows = []
    for (A,B) in pairs:
        vA, _ = build_content_vector(args.feat_dir, A)
        vB, _ = build_content_vector(args.feat_dir, B)
        o1,o2,o3 = pair_tables(A,B,vA,vB,args.topk,args.out_dir)
        manifest_rows.append({"pair": f"{A}__{B}", "top_contributors": o1, "top_differences": o2, "summary": o3})
        print(f"[OK] Wrote: {o1}\n     {o2}\n     {o3}")

    pd.DataFrame(manifest_rows).to_csv(os.path.join(args.out_dir, "content_pairwise_manifest.csv"), index=False)
    print("✅ Tamamlandı. Çıktılar:", args.out_dir)

if __name__ == "__main__":
    main()
