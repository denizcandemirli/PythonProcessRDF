# pairwise_diffs_typed_edge.py
# Stage 2: Typed-edge (semantic) pairwise differences.
# (subject_type, predicate, object_type) -> count profillerini karşılaştırır.
# Çıktılar (her çift için):
#   1) typededge_top_contributors_[A]__[B].csv
#      - a_i*b_i (cosine dot-product katkısı) büyük olan kalıplar
#   2) typededge_top_differences_[A]__[B].csv
#      - |pA - pB| (normalize pay farkı) büyük olan kalıplar
#   3) typededge_predicate_contrib_[A]__[B].csv
#      - predicate düzeyi katkı (sum(product_contrib), share, sum(abs_pct_diff))
#   4) typededge_summary_[A]__[B].csv
#      - recomputed typed_edge_cosine & typed_edge_jaccard, overlap sayıları
#   5) typededge_pairwise_manifest.csv (tüm üretilen dosyaların listesi)
#
# Kullanım (venv açık, proje kökünde):
#   python pairwise_diffs_typed_edge.py
#     --feat-dir "repro_pack/output/01 - Building_Information - Feature Extraction"
#     --sim-dir  "repro_pack/output/02 - Similarity"
#     --out-dir  "repro_pack/output/04 - Pairwise_Diffs/Typed_Edge"
#     [--topk 40]
#     [--pairs "A.rdf,B.rdf;C.rdf,D.rdf"]  # boşsa final_similarity'e göre top-3 seçer

import os, argparse, glob
import pandas as pd
import numpy as np
from collections import Counter

# Olası sütun adları
SUBJ_CAND = ["subject_type", "subj_type", "source_type", "subject_type_iri", "subj_type_iri"]
PRED_CAND = ["predicate", "predicate_iri", "pred_iri"]
OBJ_CAND  = ["object_type", "obj_type", "target_type", "object_type_iri", "obj_type_iri"]
CNT_CAND  = ["count", "n", "freq", "edge_count", "triple_count"]

def iri_label(iri: str) -> str:
    if not isinstance(iri, str): return str(iri)
    if "#" in iri: return iri.rsplit("#", 1)[-1]
    return iri.rstrip("/").rsplit("/", 1)[-1]

def find_col(df, candidates):
    for c in candidates:
        if c in df.columns: return c
    # bazı CSV'lerde whitespace / case farkı olabilir
    lower = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lower: return lower[c.lower()]
    return None

def resolve_profile_path(feat_dir: str, model_file: str) -> str:
    # 1) Tam ad: <model>_typed_edge_profile.csv
    p1 = os.path.join(feat_dir, f"{model_file}_typed_edge_profile.csv")
    if os.path.exists(p1): return p1
    # 2) Glob: model adını içeren ve _typed_edge_profile.csv ile biten
    patt = os.path.join(feat_dir, f"*{model_file}*_typed_edge_profile.csv")
    hits = glob.glob(patt)
    if hits: return hits[0]
    # 3) Uzantısız dene
    base = model_file.replace(".rdf", "")
    patt2 = os.path.join(feat_dir, f"*{base}*_typed_edge_profile.csv")
    hits2 = glob.glob(patt2)
    if hits2: return hits2[0]
    # 4) Son çare: klasördeki tüm *_typed_edge_profile.csv dosyalarından
    # model adındaki ayırt edici parçayı içeren ilkini seç
    all_prof = glob.glob(os.path.join(feat_dir, "*_typed_edge_profile.csv"))
    for q in [model_file, base]:
        for p in all_prof:
            if q in os.path.basename(p):
                return p
    # Bulunamadı
    return p1  # ilk denenen path (hata mesajında gözüksün)

def read_typed_profile(path):
    """CSV -> Counter[(subj_iri,pred_iri,obj_iri)] = count"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Typed-edge profile not found: {path}")
    df = pd.read_csv(path)
    cs, cp, co, cc = (find_col(df, SUBJ_CAND),
                      find_col(df, PRED_CAND),
                      find_col(df, OBJ_CAND),
                      find_col(df, CNT_CAND))
    if not (cs and cp and co and cc):
        raise ValueError(
            f"Column(s) missing in {path}. "
            f"Need subj in {SUBJ_CAND}, pred in {PRED_CAND}, obj in {OBJ_CAND}, count in {CNT_CAND}"
        )
    cnt = Counter()
    for _, r in df.iterrows():
        s = str(r[cs]); p = str(r[cp]); o = str(r[co])
        try:
            n = int(r[cc])
        except Exception:
            try: n = int(float(r[cc]))
            except Exception: n = 0
        if n < 0: continue
        cnt[(s, p, o)] += n
    return cnt

def cosine_from_counters(a: Counter, b: Counter) -> float:
    keys = set(a) | set(b)
    if not keys: return 0.0
    va = np.array([a.get(k, 0) for k in keys], dtype=float)
    vb = np.array([b.get(k, 0) for k in keys], dtype=float)
    dot = float(np.dot(va, vb))
    na = float(np.linalg.norm(va)); nb = float(np.linalg.norm(vb))
    if na == 0 or nb == 0: return 0.0
    return dot / (na * nb)

def jaccard_from_counters(a: Counter, b: Counter) -> float:
    A = {k for k, v in a.items() if v > 0}
    B = {k for k, v in b.items() if v > 0}
    if not (A or B): return 0.0
    return len(A & B) / len(A | B)

def build_pair_tables(modelA, modelB, vA: Counter, vB: Counter, topk: int, out_dir: str):
    keys = sorted(set(vA) | set(vB))
    sA = sum(vA.values()) if vA else 0
    sB = sum(vB.values()) if vB else 0
    rows = []
    for k in keys:
        a = vA.get(k, 0); b = vB.get(k, 0)
        pA = (a / sA) if sA else 0.0
        pB = (b / sB) if sB else 0.0
        rows.append({
            "subj_type_iri": k[0],
            "pred_iri": k[1],
            "obj_type_iri": k[2],
            "subj": iri_label(k[0]),
            "pred": iri_label(k[1]),
            "obj": iri_label(k[2]),
            "count_A": a,
            "count_B": b,
            "pct_A": pA,
            "pct_B": pB,
            "abs_pct_diff": abs(pA - pB),
            "product_contrib": a * b
        })
    df = pd.DataFrame(rows)

    # 1) En büyük cosine katkıları
    top_contrib = df.sort_values(["product_contrib", "count_A", "count_B"], ascending=False).head(topk)

    # 2) En büyük normalize farklar
    top_diffs = df.sort_values(["abs_pct_diff", "count_A", "count_B"], ascending=False).head(topk)

    # 3) Predicate düzeyi özet
    if len(df):
        pred_group = df.groupby(["pred_iri", "pred"], as_index=False).agg(
            product_contrib_sum=("product_contrib", "sum"),
            abs_pct_diff_sum=("abs_pct_diff", "sum"),
            countA=("count_A", "sum"),
            countB=("count_B", "sum"),
        )
        total_dot = float((df["count_A"] * df["count_B"]).sum())
        pred_group["product_contrib_share"] = (
            pred_group["product_contrib_sum"] / total_dot if total_dot > 0 else 0.0
        )
        pred_group = pred_group.sort_values(["product_contrib_sum", "abs_pct_diff_sum"], ascending=False)
    else:
        pred_group = pd.DataFrame(columns=["pred_iri","pred","product_contrib_sum","abs_pct_diff_sum","product_contrib_share","countA","countB"])

    # 4) Özet metrikler
    cos = cosine_from_counters(vA, vB)
    jac = jaccard_from_counters(vA, vB)
    overlap = sum(1 for k in keys if vA.get(k,0)>0 and vB.get(k,0)>0)

    summ = pd.DataFrame([{
        "model_a": modelA, "model_b": modelB,
        "typed_edge_cosine_recomputed": cos,
        "typed_edge_jaccard_recomputed": jac,
        "triples_nonzero_A": len([k for k in vA if vA[k] > 0]),
        "triples_nonzero_B": len([k for k in vB if vB[k] > 0]),
        "triples_overlap": overlap,
        "sum_counts_A": sA, "sum_counts_B": sB
    }])

    # Yaz
    safe = lambda s: s.replace(os.sep, "_")
    tag = f"{safe(modelA)}__{safe(modelB)}"
    out1 = os.path.join(out_dir, f"typededge_top_contributors_{tag}.csv")
    out2 = os.path.join(out_dir, f"typededge_top_differences_{tag}.csv")
    out3 = os.path.join(out_dir, f"typededge_predicate_contrib_{tag}.csv")
    out4 = os.path.join(out_dir, f"typededge_summary_{tag}.csv")

    top_contrib.to_csv(out1, index=False)
    top_diffs.to_csv(out2, index=False)
    pred_group.to_csv(out3, index=False)
    summ.to_csv(out4, index=False)

    return out1, out2, out3, out4

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat-dir", default=os.path.join(".", "repro_pack", "output", "01 - Building_Information - Feature Extraction"),
                    help="Typed-edge profil CSV'lerinin olduğu klasör")
    ap.add_argument("--sim-dir",  default=os.path.join(".", "repro_pack", "output", "02 - Similarity"),
                    help="Top-3 çifti seçmek için similarity klasörü")
    ap.add_argument("--out-dir",  default=os.path.join(".", "repro_pack", "output", "04 - Pairwise_Diffs", "Typed_Edge"),
                    help="Çıktı klasörü")
    ap.add_argument("--topk", type=int, default=40, help="Her listede tutulacak satır sayısı")
    ap.add_argument("--pairs", type=str, default="",
                    help="Çiftler: 'A.rdf,B.rdf;C.rdf,D.rdf'. Boşsa final_similarity'e göre top-3 alınır.")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Çift listesi
    pairs = []
    if args.pairs.strip():
        for seg in args.pairs.split(";"):
            if not seg.strip(): continue
            a, b = [x.strip() for x in seg.split(",")]
            pairs.append((a, b))
    else:
        sim_csv = os.path.join(args.sim_dir, "similarity_combined_weighted.csv")
        if not os.path.exists(sim_csv):
            raise FileNotFoundError(f"Bulunamadı: {sim_csv}. --pairs ile çiftleri elle verin.")
        sim = pd.read_csv(sim_csv).sort_values("final_similarity", ascending=False).head(3)
        for _, r in sim.iterrows():
            pairs.append((r["model_a"], r["model_b"]))

    manifest = []
    for (A, B) in pairs:
        fA = resolve_profile_path(args.feat_dir, A)
        fB = resolve_profile_path(args.feat_dir, B)
        if not os.path.exists(fA):
            raise FileNotFoundError(f"{A} için typed-edge profili bulunamadı. Denenen yol: {fA}")
        if not os.path.exists(fB):
            raise FileNotFoundError(f"{B} için typed-edge profili bulunamadı. Denenen yol: {fB}")

        vA = read_typed_profile(fA)
        vB = read_typed_profile(fB)

        o1, o2, o3, o4 = build_pair_tables(A, B, vA, vB, args.topk, args.out_dir)
        manifest.append({"pair": f"{A}__{B}",
                         "top_contributors": o1, "top_differences": o2,
                         "predicate_contrib": o3, "summary": o4})
        print(f"[OK] {A} vs {B}\n  -> {o1}\n  -> {o2}\n  -> {o3}\n  -> {o4}")

    man_path = os.path.join(args.out_dir, "typededge_pairwise_manifest.csv")
    pd.DataFrame(manifest).to_csv(man_path, index=False)
    print("✅ Tamamlandı. Çıktılar:", args.out_dir)

if __name__ == "__main__":
    main()
