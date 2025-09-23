# verify_repro_pass.py  —  Robust repro checker for A–D channels, S_struct, and S_total
import os, argparse, glob
import numpy as np
import pandas as pd

W = {"content": 0.30, "typed": 0.20, "edge": 0.10, "struct": 0.40}

def pick(df, names):
    """Return first matching column (case-insensitive) from DataFrame df."""
    if df is None or df.empty: return None
    cmap = {c.strip().lower(): c for c in df.columns}
    for n in names:
        if n and n.strip().lower() in cmap:
            return cmap[n.strip().lower()]
    return None

def check_matrix_symmetry(df, name):
    if df is None or df.empty:
        return False, f"{name}: MISSING"
    if df.index.tolist() != df.columns.tolist():
        return False, f"{name}: index/columns mismatch"
    A = df.values.astype(float)
    sym_ok  = np.allclose(A, A.T, atol=1e-8)
    diag_ok = np.allclose(np.diag(A), 1.0, atol=1e-8)
    rng_ok  = (A.min() >= -1e-8) and (A.max() <= 1+1e-8)
    ok = sym_ok and diag_ok and rng_ok
    return ok, f"{name}: sym={sym_ok}, diag1={diag_ok}, rangeOK={rng_ok}"

def canon_pairs(df):
    """Create canonical pair columns A,B regardless of original names."""
    if df is None or df.empty: return None
    a = pick(df, ["A","model_a","left","source","file_a","name_a","model1"])
    b = pick(df, ["B","model_b","right","target","file_b","name_b","model2"])
    if not a or not b: return None
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

def main(root):
    out = []
    T = os.path.join(root, "06 - Total_Similarity")
    S = os.path.join(root, "07 - Structural_Extension_v25p2")

    # Load matrices
    mat_total  = pd.read_csv(os.path.join(T, "total_similarity_matrix.csv"), index_col=0) \
                 if os.path.exists(os.path.join(T, "total_similarity_matrix.csv")) else None
    mat_struct = pd.read_csv(os.path.join(S, "struct_similarity_matrix.csv"), index_col=0) \
                 if os.path.exists(os.path.join(S, "struct_similarity_matrix.csv")) else None

    # Pairwise csvs
    pair_total  = pd.read_csv(os.path.join(T, "pairwise_total_summary.csv")) \
                  if os.path.exists(os.path.join(T, "pairwise_total_summary.csv")) else None
    pair_struct = pd.read_csv(os.path.join(S, "pairwise_structural_summary.csv")) \
                  if os.path.exists(os.path.join(S, "pairwise_structural_summary.csv")) else None

    # Optional channel matrices
    candA = glob.glob(os.path.join(T, "*content*.csv")) or []
    candB = glob.glob(os.path.join(T, "*typed*.csv"))   or []
    candC = glob.glob(os.path.join(T, "*edge*jac*.csv")) or []
    chA = pd.read_csv(candA[0], index_col=0) if candA else None
    chB = pd.read_csv(candB[0], index_col=0) if candB else None
    chC = pd.read_csv(candC[0], index_col=0) if candC else None

    # Matrix sanity
    for df,name in [(mat_total,"total_similarity_matrix"),
                    (mat_struct,"struct_similarity_matrix"),
                    (chA,"similarity_content_cosine"),
                    (chB,"similarity_typed_edge_cosine"),
                    (chC,"similarity_edge_sets_jaccard")]:
        if df is not None:
            ok,msg = check_matrix_symmetry(df,name); out.append(msg)

    # === Recompute fused total from components (robust to old/new column names)
    fused_done = False
    if pair_total is not None:
        P = canon_pairs(pair_total)
        if P is not None:
            # component columns (prefer new canonical names)
            c = pick(P, ["S_content","content_cos","content_cosine","content"])
            t = pick(P, ["S_typed","typed_edge_cos","typed_edge_cosine","typed_cos","typed"])
            e = pick(P, ["S_edge","edge_sets_jaccard","edge_jaccard","edge"])
            # struct column may live in pair_total (new) or pair_struct (old)
            s = pick(P, ["S_struct","struct","struct_final","motif_final","structural_similarity"])

            if not s and pair_struct is not None:
                S_pair = canon_pairs(pair_struct)
                ss = pick(S_pair, ["S_struct","structural_similarity","struct"])
                if all([c,t,e,ss]):
                    M = pd.merge(P[["A","B",c,t,e]],
                                 S_pair[["A","B",ss]],
                                 on=["A","B"], how="inner")
                    M = M.rename(columns={c:"S_content", t:"S_typed", e:"S_edge", ss:"S_struct"})
                else:
                    M = None
            else:
                # everything in pair_total
                if all([c,t,e,s]):
                    M = P.rename(columns={c:"S_content", t:"S_typed", e:"S_edge", s:"S_struct"})
                else:
                    M = None

            if M is not None and not M.empty:
                M["S_total_fused"] = (
                    W["content"]*M["S_content"] +
                    W["typed"]  *M["S_typed"]   +
                    W["edge"]   *M["S_edge"]    +
                    W["struct"] *M["S_struct"]
                )
                fused_done = True
                if mat_total is not None and not mat_total.empty:
                    diffs = []
                    for _,row in M.iterrows():
                        a,b,st = row["A"], row["B"], row["S_total_fused"]
                        if a in mat_total.index and b in mat_total.columns:
                            diffs.append(abs(st - float(mat_total.loc[a,b])))
                    if diffs:
                        dmax, dmean = float(np.max(diffs)), float(np.mean(diffs))
                        out.append(f"fusion_vs_matrix: max|Δ|={dmax:.6g}, mean|Δ|={dmean:.6g}")
                        out.append("fusion_vs_matrix: PASS" if dmax<=1e-4 else "fusion_vs_matrix: FAIL (>1e-4)")
            # else: fall through to pairwise-vs-matrix comparison below

    # If recompute couldn't run, at least compare written S_total vs matrix
    if not fused_done and pair_total is not None and mat_total is not None and not mat_total.empty:
        P2 = canon_pairs(pair_total)
        if P2 is not None:
            total_col = pick(P2, ["S_total","total","final_similarity","score_total"])
            if total_col:
                diffs = []
                for _,row in P2.iterrows():
                    a,b,st = row["A"], row["B"], float(row[total_col])
                    if a in mat_total.index and b in mat_total.columns:
                        diffs.append(abs(st - float(mat_total.loc[a,b])))
                if diffs:
                    dmax, dmean = float(np.max(diffs)), float(np.mean(diffs))
                    out.append(f"pairwise_vs_matrix: max|Δ|={dmax:.6g}, mean|Δ|={dmean:.6g}")
                    out.append("pairwise_vs_matrix: PASS" if dmax<=1e-4 else "pairwise_vs_matrix: FAIL (>1e-4)")
            else:
                out.append("pairwise_vs_matrix: missing S_total column")

    # S3 radar: range & dual rule
    scores = pd.read_csv(os.path.join(S,"struct_system_scores.csv")) \
              if os.path.exists(os.path.join(S,"struct_system_scores.csv")) else None
    if scores is not None and not scores.empty:
        rng_flags = []
        for axis in ["frame","wall","braced","dual"]:
            if axis in scores.columns:
                rng_flags.append(((scores[axis]>=-1e-9)&(scores[axis]<=1+1e-9)).all())
                out.append(f"S3:{axis} rangeOK={rng_flags[-1]}")
        if set(["frame","wall"]).issubset(scores.columns) and "dual" in scores.columns:
            ok_dual = (scores["dual"] <= scores[["frame","wall"]].min(axis=1)+1e-9).all()
            out.append(f"S3:dual<=min(frame,wall) = {ok_dual}")

    print("\n".join(out))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Path to repro_pack\\output")
    args = ap.parse_args()
    main(args.root)
