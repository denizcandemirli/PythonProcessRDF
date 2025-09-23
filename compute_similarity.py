# compute_similarity.py  —  fused total writer (authoritative weights)
import os, argparse
import numpy as np
import pandas as pd

FUSION_W = {"content": 0.30, "typed": 0.20, "edge": 0.10, "struct": 0.40}

def pick(cols, names):
    m = {c.lower(): c for c in cols}
    for n in names:
        if n.lower() in m: return m[n.lower()]
    return None

def canon_pairs(df):
    a = pick(df.columns, ["model_a","model_A","A","left","source","model1","name_a","file_a"])
    b = pick(df.columns, ["model_b","model_B","B","right","target","model2","name_b","file_b"])
    if not a or not b:
        raise ValueError("model A/B columns not found")
    out = df.copy()
    out["A"] = out[[a,b]].min(axis=1)
    out["B"] = out[[a,b]].max(axis=1)
    return out

def build_matrix_from_pairs(df, val="S_total"):
    names = sorted(set(df["A"]).union(df["B"]))
    M = pd.DataFrame(np.eye(len(names)), index=names, columns=names, dtype=float)
    for _, r in df.iterrows():
        M.loc[r["A"], r["B"]] = r[val]
        M.loc[r["B"], r["A"]] = r[val]
    return M

def main(total_dir, struct_dir):
    p_total = os.path.join(total_dir, "pairwise_total_summary.csv")
    p_struct= os.path.join(struct_dir, "pairwise_structural_summary.csv")
    if not os.path.exists(p_total):  raise FileNotFoundError(p_total)
    if not os.path.exists(p_struct): raise FileNotFoundError(p_struct)

    P = pd.read_csv(p_total)
    S = pd.read_csv(p_struct)

    P = canon_pairs(P)
    S = canon_pairs(S)

    # tolerate both old and new column names
    c = pick(P.columns, ["S_content","content_cos","content_cosine","content"])
    t = pick(P.columns, ["S_typed","typed_edge_cos","typed_edge_cosine","typed"])
    e = pick(P.columns, ["S_edge","edge_sets_jaccard","edge_jaccard_combined","edge"])
    s = pick(S.columns, ["S_struct","structural_similarity","struct","final"])
    if not all([c,t,e,s]):
        raise RuntimeError("Required component columns missing (content/typed/edge + structural).")

    M = pd.merge(P[["A","B",c,t,e]],
                 S[["A","B",s]],
                 on=["A","B"], how="inner").rename(columns={
                     c:"S_content", t:"S_typed", e:"S_edge", s:"S_struct"
                 })

    M["S_total"] = (
        FUSION_W["content"]*M["S_content"] +
        FUSION_W["typed"]  *M["S_typed"]   +
        FUSION_W["edge"]   *M["S_edge"]    +
        FUSION_W["struct"] *M["S_struct"]
    )

    # yaz: pairwise_total_summary.csv (standart kolonlar)
    cols = ["A","B","S_content","S_typed","S_edge","S_struct","S_total"]
    M[cols].to_csv(p_total, index=False)

    # yaz: total_similarity_matrix.csv
    mat = build_matrix_from_pairs(M, "S_total")
    mat.to_csv(os.path.join(total_dir, "total_similarity_matrix.csv"))

    print("WROTE:", p_total)
    print("WROTE:", os.path.join(total_dir, "total_similarity_matrix.csv"))
    print(f"Weights -> content={FUSION_W['content']}, typed={FUSION_W['typed']}, edge={FUSION_W['edge']}, struct={FUSION_W['struct']}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--total-dir",  default=r".\repro_pack\output\06 - Total_Similarity")
    ap.add_argument("--struct-dir", default=r".\repro_pack\output\07 - Structural_Extension_v25p2")
    args = ap.parse_args()
    main(args.total_dir, args.struct_dir)
