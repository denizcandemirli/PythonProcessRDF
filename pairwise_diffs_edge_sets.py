# pairwise_diffs_edge_sets.py
# Stage 3: Edge-set / topology pairwise differences per predicate.
# Inputs: *_adjacentElement_edges.csv, *_adjacentZone_edges.csv,
#         *_intersectingElement_edges.csv, *_hasContinuantPart_edges.csv (or *_BFO_0000178_edges.csv)
# Outputs: per-predicate Jaccard summary + top shared edges + top unique edges.

import os, glob, argparse
import pandas as pd
from collections import Counter

PRED_KEYS = {
    "adjacentElement": ["adjacentElement"],
    "adjacentZone": ["adjacentZone"],
    "intersectingElement": ["intersectingElement"],
    "hasContinuantPart": ["hasContinuantPart", "BFO_0000178"],
}

SUBJ_CAND = ["subject","source","s","subj","start"]
OBJ_CAND  = ["object","target","o","obj","end"]

def find_edge_file(feat_dir: str, model: str, pred_key: str) -> str:
    base = model.replace(".rdf","")
    candidates = []
    for token in PRED_KEYS[pred_key]:
        patt1 = os.path.join(feat_dir, f"*{model}*_{token}_edges.csv")
        patt2 = os.path.join(feat_dir, f"*{base}*_{token}_edges.csv")
        candidates += glob.glob(patt1) + glob.glob(patt2)
    return candidates[0] if candidates else ""

def pick_col(df, cand):
    cols = {c.lower(): c for c in df.columns}
    for k in cand:
        if k in df.columns:
            return k
        if k.lower() in cols:
            return cols[k.lower()]
    return None

def canon_edge(u: str, v: str) -> tuple:
    a, b = str(u), str(v)
    return (a, b) if a <= b else (b, a)

def load_edges(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    cs = pick_col(df, SUBJ_CAND)
    co = pick_col(df, OBJ_CAND)
    if not cs or not co:
        raise ValueError(f"Edge list columns not found in {path} (need one of {SUBJ_CAND} / {OBJ_CAND})")
    out = df[[cs, co]].rename(columns={cs:"u", co:"v"}).dropna()
    out["u"] = out["u"].astype(str)
    out["v"] = out["v"].astype(str)
    uv = out.apply(lambda r: canon_edge(r["u"], r["v"]), axis=1)
    out["a"] = [t[0] for t in uv]
    out["b"] = [t[1] for t in uv]
    out = out[["a","b"]].drop_duplicates()
    return out

def degree_from_edges(df_edges: pd.DataFrame) -> Counter:
    deg = Counter()
    for _, r in df_edges.iterrows():
        deg[r["a"]] += 1
        deg[r["b"]] += 1
    return deg

def jaccard(A:set, B:set) -> float:
    if not (A or B):
        return 0.0
    return len(A & B) / len(A | B)

def empty_shared_df():
    return pd.DataFrame(columns=["u","v","deg_u_A","deg_v_A","deg_u_B","deg_v_B","score"])

def empty_unique_df():
    return pd.DataFrame(columns=["u","v","deg_u","deg_v","score"])

def build_for_pair(feat_dir, out_dir, modelA, modelB, topk=25):
    os.makedirs(out_dir, exist_ok=True)
    manifest = []
    for pred in PRED_KEYS.keys():
        pA = find_edge_file(feat_dir, modelA, pred)
        pB = find_edge_file(feat_dir, modelB, pred)
        if not pA or not pB:
            manifest.append({
                "pair": f"{modelA}__{modelB}", "predicate": pred,
                "summary": "", "shared": "", "uniqueA": "", "uniqueB": "",
                "note": f"edge file missing ({pA or 'None'} | {pB or 'None'})"
            })
            continue

        eA = load_edges(pA)
        eB = load_edges(pB)

        degA = degree_from_edges(eA)
        degB = degree_from_edges(eB)

        setA = set((r.a, r.b) for _, r in eA.iterrows())
        setB = set((r.a, r.b) for _, r in eB.iterrows())

        inter = setA & setB
        onlyA = setA - setB
        onlyB = setB - setA
        jac  = jaccard(setA, setB)

        # --- Top shared
        rows_shared = []
        for (u,v) in inter:
            score = (degA[u]+degA[v]) + (degB[u]+degB[v])
            rows_shared.append({
                "u":u, "v":v,
                "deg_u_A":degA[u], "deg_v_A":degA[v],
                "deg_u_B":degB[u], "deg_v_B":degB[v],
                "score":score
            })
        df_shared = pd.DataFrame(rows_shared) if rows_shared else empty_shared_df()
        if "score" in df_shared.columns:
            df_shared = df_shared.sort_values("score", ascending=False).head(topk)

        # --- Top uniques (A\B, B\A)
        def unique_rows(S, deg):
            if not S:
                return empty_unique_df()
            rows = []
            for (u,v) in S:
                score = deg[u] + deg[v]
                rows.append({"u":u, "v":v, "deg_u":deg[u], "deg_v":deg[v], "score":score})
            dfu = pd.DataFrame(rows)
            return dfu.sort_values("score", ascending=False).head(topk)

        df_uniqueA = unique_rows(onlyA, degA)
        df_uniqueB = unique_rows(onlyB, degB)

        # degree stats
        def deg_stats(deg: Counter):
            if not deg:
                return (0,0,0,0)
            vals = list(deg.values())
            import numpy as np
            return (int(len(vals)), float(sum(vals)/len(vals)), float(np.median(vals)), int(max(vals)))

        nA, meanA, medA, maxA = deg_stats(degA)
        nB, meanB, medB, maxB = deg_stats(degB)

        tag = f"{modelA.replace(os.sep,'_')}__{modelB.replace(os.sep,'_')}"
        outS = os.path.join(out_dir, f"edge_summary_{tag}_{pred}.csv")
        outC = os.path.join(out_dir, f"edge_top_shared_{tag}_{pred}.csv")
        outUA= os.path.join(out_dir, f"edge_top_uniqueA_{tag}_{pred}.csv")
        outUB= os.path.join(out_dir, f"edge_top_uniqueB_{tag}_{pred}.csv")

        pd.DataFrame([{
            "model_a": modelA, "model_b": modelB, "predicate": pred,
            "jaccard": jac,
            "edges_A": len(setA), "edges_B": len(setB),
            "edges_intersection": len(inter),
            "edges_onlyA": len(onlyA), "edges_onlyB": len(onlyB),
            "nodes_A": nA, "deg_mean_A": round(meanA,3), "deg_med_A": round(medA,3), "deg_max_A": maxA,
            "nodes_B": nB, "deg_mean_B": round(meanB,3), "deg_med_B": round(medB,3), "deg_max_B": maxB,
        }]).to_csv(outS, index=False)

        df_shared.to_csv(outC, index=False)
        df_uniqueA.to_csv(outUA, index=False)
        df_uniqueB.to_csv(outUB, index=False)

        note = "" if len(inter)>0 else "no shared edges"
        manifest.append({
            "pair": f"{modelA}__{modelB}", "predicate": pred,
            "summary": outS, "shared": outC, "uniqueA": outUA, "uniqueB": outUB,
            "note": note
        })
    return manifest

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat-dir",
                    default=os.path.join(".", "repro_pack", "output", "01 - Building_Information - Feature Extraction"),
                    help="Edge-list CSV folder")
    ap.add_argument("--out-dir",
                    default=os.path.join(".", "repro_pack", "output", "04 - Pairwise_Diffs", "Edge_Sets"),
                    help="Output folder")
    ap.add_argument("--pairs", default="",
                    help="A.rdf,B.rdf;C.rdf,D.rdf;... (required)")
    ap.add_argument("--topk", type=int, default=25)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    if not args.pairs.strip():
        raise SystemExit('Please provide --pairs like: "A.rdf,B.rdf;C.rdf,D.rdf"')

    pairs = []
    for seg in args.pairs.split(";"):
        seg = seg.strip()
        if not seg:
            continue
        a,b = [x.strip() for x in seg.split(",")]
        pairs.append((a,b))

    manifest = []
    for (A,B) in pairs:
        print(f"[PAIR] {A} vs {B}")
        manifest += build_for_pair(args.feat_dir, args.out_dir, A, B, topk=args.topk)

    man_path = os.path.join(args.out_dir, "edge_pair_manifest.csv")
    pd.DataFrame(manifest).to_csv(man_path, index=False)
    print("✅ Done. Manifest:", man_path)

if __name__ == "__main__":
    main()
