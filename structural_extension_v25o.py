#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Structural Extension v25o
- Wider regex/IFC mapping (Wall/Slab/Brace/Column/Beam/Core)
- More predicates for motifs (adjacentZone & intersectingElement)
- Brace & Moment enrichment (labels and IFC types)
- Proxy penalty default 0.70 + proxy summary CSV
- Outputs compatible with v25n/v25m readers
"""

import os, re, argparse, json
import pandas as pd
import numpy as np
from collections import defaultdict
import networkx as nx
from rdflib import Graph, URIRef, RDF, RDFS, Literal

# ---------------------------
# Config: type & function maps
# ---------------------------

TYPE_REGEX = {
    "Wall": r"(?:IfcWall|Wall|Shear[- ]?Wall|CoreWall|ExteriorWall|InteriorWall|RetainingWall)",
    "Slab": r"(?:IfcSlab|Slab|Deck|Floor|Plate|FloorPlate)",
    "Column": r"(?:IfcColumn|Column|Pier)",
    "Beam": r"(?:IfcBeam|Beam|Girder)",
    "Brace": r"(?:IfcMember|IfcStructuralCurveMember|Brace|Bracing|Cross[- ]?Brace|X[- ]?Brace|Tie|Strut|Diagonal)",
    "Core": r"(?:Core|CoreWall|LiftCore|StairCore)",
}

FUNC_REGEX = {
    "LoadBearing": r"(Load\s*Bearing|LoadBearing|Bearing|Gravity|Primary)",
    "Shear":       r"(Shear|Lateral|WallAction)",
    "Moment":      r"(Moment|RigidFrame|MomentFrame|FrameAction)"
}

# motif predicates (both directions tolerated)
PRED_ADJ = ["adjacentElement","adjacentZone"]
PRED_INT = ["intersectingElement"]  # extend if needed

# ---------------------------
# Helpers
# ---------------------------

def norm_text(x:str) -> str:
    return (x or "").strip()

def hit(regex: str, text: str) -> bool:
    return bool(re.search(regex, text or "", flags=re.I))

def map_type(localname:str, label:str) -> str|None:
    for k, rx in TYPE_REGEX.items():
        if hit(rx, localname) or hit(rx, label):
            return k
    return None

def map_function(label:str) -> str|None:
    for k, rx in FUNC_REGEX.items():
        if hit(rx, label):
            return k
    return None

# ---------------------------
# Load RDF → in-memory model
# ---------------------------

class Model:
    def __init__(self, name:str, path:str):
        self.name = name
        self.path = path
        self.g = Graph()
        self.g.parse(path)
        self.types = defaultdict(list)        # node -> [types]
        self.labels = {}                      # node -> label
        self.edges = []                       # (subj,pred,obj)
        self.functions = defaultdict(list)    # node -> [function labels]
        self._init_index()

    def _init_index(self):
        g = self.g
        # labels
        for s, _, o in g.triples((None, RDFS.label, None)):
            self.labels[str(s)] = str(o)
        # rdf:type
        for s, _, o in g.triples((None, RDF.type, None)):
            s_id = str(s)
            lname = str(o).split("#")[-1].split("/")[-1]
            lab = self.labels.get(s_id, "")
            mapped = map_type(lname, lab)
            if mapped:
                self.types[s_id].append(mapped)
        # functions (hasFunction or label-based fallback)
        for s, p, o in g:
            pred = str(p).split("#")[-1].split("/")[-1]
            if pred.lower().endswith("hasfunction"):
                lab = self.labels.get(str(o), str(o))
                mapped = map_function(lab)
                if mapped:
                    self.functions[str(s)].append(mapped)
        # store all edges (stringified)
        for s,p,o in g:
            self.edges.append((str(s), str(p).split("#")[-1].split("/")[-1], str(o)))

    def iter_edges(self, predicates:list[str]):
        for s,p,o in self.edges:
            if p in predicates:
                yield s,p,o

# ---------------------------
# S1 — inventories
# ---------------------------

def s1_inventories(models:list[Model], func_all:bool):
    rows_t, rows_f = [], []
    for m in models:
        # types
        counter = defaultdict(int)
        for _, tlist in m.types.items():
            for t in tlist: counter[t]+=1
        for t,c in counter.items():
            rows_t.append({"model":m.name,"type":t,"count":c})

        # functions
        fcounter = defaultdict(int)
        if func_all:
            # harvest labels too
            for n, lab in m.labels.items():
                mapped = map_function(lab)
                if mapped: fcounter[mapped]+=1
        for n, fl in m.functions.items():
            for f in fl: fcounter[f]+=1
        for f,c in fcounter.items():
            rows_f.append({"model":m.name,"function":f,"count":c})

    df_types = pd.DataFrame(rows_t).sort_values(["model","type"]).reset_index(drop=True)
    df_funcs = pd.DataFrame(rows_f).sort_values(["model","function"]).reset_index(drop=True)
    return df_types, df_funcs

# ---------------------------
# S2 — motifs (M2/M2b/M3/M4 + M5)
# ---------------------------

def s2_motifs(models, df_types, df_funcs,
              allow_type_only=True, proxy_penalty=0.70, out_dir=None):
    rows_counts = []
    proxy_rows = []

    for m in models:
        name = m.name
        # Helper sets
        nodes_wall  = {n for n,ts in m.types.items() if "Wall"   in ts}
        nodes_slab  = {n for n,ts in m.types.items() if "Slab"   in ts}
        nodes_col   = {n for n,ts in m.types.items() if "Column" in ts}
        nodes_beam  = {n for n,ts in m.types.items() if "Beam"   in ts}
        nodes_brace = {n for n,ts in m.types.items() if "Brace"  in ts}
        nodes_core  = {n for n,ts in m.types.items() if "Core"   in ts}

        # ---- M3 wall-slab (true edges)
        m3_true = 0
        for s,p,o in m.iter_edges(PRED_ADJ+PRED_INT):
            if (s in nodes_wall and o in nodes_slab) or (s in nodes_slab and o in nodes_wall):
                m3_true += 1
        # ---- M2 frame node (beam-column adjacency)
        m2_true = 0
        for s,p,o in m.iter_edges(PRED_ADJ):
            if (s in nodes_beam and o in nodes_col) or (s in nodes_col and o in nodes_beam):
                m2_true += 1
        # ---- M2b brace node (brace to frame member)
        m2b_true = 0
        for s,p,o in m.iter_edges(PRED_ADJ):
            if (s in nodes_brace and (o in nodes_beam or o in nodes_col)) or \
               (o in nodes_brace and (s in nodes_beam or s in nodes_col)):
                m2b_true += 1
        # ---- M4 core (simple proxy: presence of core + slabs adjacency)
        m4_true = 0
        if nodes_core:
            # count any core-slab adjacency as 1 evidence
            hit_core = False
            for s,p,o in m.iter_edges(PRED_ADJ+PRED_INT):
                if (s in nodes_core and o in nodes_slab) or (o in nodes_core and s in nodes_slab):
                    hit_core = True; break
            m4_true = 1 if hit_core else 0

        # ---- M5 role counts (from df_funcs)
        dfF = df_funcs[df_funcs["model"]==name]
        m5_lb = int(dfF[dfF["function"]=="LoadBearing"]["count"].sum())
        m5_sh = int(dfF[dfF["function"]=="Shear"]["count"].sum())
        m5_mo = int(dfF[dfF["function"]=="Moment"]["count"].sum())

        # ---- Proxy fallback if needed
        used_proxy = False
        if allow_type_only and (m3_true+m2_true+m2b_true+m4_true)==0:
            # minimal proxy: if wall & slab exist → m3_proxy = 1; if core exists → m4_proxy = 1
            used_proxy = True
            m3_true = int(bool(nodes_wall and nodes_slab))
            m4_true = int(bool(nodes_core))
            # keep m2/m2b as is (0 unless edges exist)

        rows_counts.append({
            "model": name,
            "M2_frameNode": m2_true,
            "M2b_braceNode": m2b_true,
            "M3_wallSlab": m3_true,
            "M4_core": m4_true,
            "M5_LB": m5_lb,
            "M5_Shear": m5_sh,
            "M5_Moment": m5_mo,
            "used_proxy": int(used_proxy)
        })
        if used_proxy:
            proxy_rows.append({"model":name,"M3_proxy":m3_true,"M4_proxy":m4_true})

    df_counts = pd.DataFrame(rows_counts)
    # Shares (row-normalized over motif family; avoid division by zero)
    motif_cols = ["M2_frameNode","M2b_braceNode","M3_wallSlab","M4_core","M5_LB","M5_Shear","M5_Moment"]
    df_shares = df_counts[["model"]+motif_cols].copy()
    df_shares[motif_cols] = df_shares[motif_cols].astype(float)
    row_sum = df_shares[motif_cols].sum(axis=1).replace(0, 1.0)
    for c in motif_cols:
        df_shares[c] = df_shares[c] / row_sum

    # densities per 100 elements (optional)
    df_dens = df_shares.copy()
    for c in motif_cols:
        df_dens[c] = df_counts[c]  # raw counts here; caller may scale

    if out_dir:
        df_counts.to_csv(os.path.join(out_dir,"struct_motif_counts.csv"), index=False)
        df_shares.to_csv(os.path.join(out_dir,"struct_motif_shares.csv"), index=False)
        df_dens.to_csv(os.path.join(out_dir,"struct_motif_densities_per100.csv"), index=False)
        if proxy_rows:
            pd.DataFrame(proxy_rows).to_csv(os.path.join(out_dir,"motif_proxy_summary.csv"), index=False)

    return df_counts, df_shares, df_dens

# ---------------------------
# S3 — system scores (rule-based)
# ---------------------------

def s3_system_scores(models, motif_shares, df_types, df_funcs, dual_thresh=0.25):
    # Ensure WIDE
    if "model" not in motif_shares.columns:
        raise ValueError("motif_shares must be WIDE with 'model' column.")
    ms = motif_shares.set_index("model")
    rows = []
    for m in ms.index:
        M2 = ms.loc[m, "M2_frameNode"] if "M2_frameNode" in ms.columns else 0.0
        M3 = ms.loc[m, "M3_wallSlab"]  if "M3_wallSlab"  in ms.columns else 0.0
        M2b= ms.loc[m, "M2b_braceNode"]if "M2b_braceNode"in ms.columns else 0.0
        LB = ms.loc[m, "M5_LB"]        if "M5_LB"        in ms.columns else 0.0
        SH = ms.loc[m, "M5_Shear"]     if "M5_Shear"     in ms.columns else 0.0
        MO = ms.loc[m, "M5_Moment"]    if "M5_Moment"    in ms.columns else 0.0

        # simple formulas (same spirit as v25n)
        score_wall  = 0.6*M3 + 0.4*(LB+SH)/max(1e-9,1.0)   # rely on M3 + roles
        score_frame = 0.6*M2 + 0.4*MO
        score_braced= M2b
        score_dual  = 0.0
        # dual if both wall & frame contributions present and frame >= threshold
        if score_frame >= dual_thresh and score_wall >= dual_thresh:
            score_dual = min(1.0, (score_frame+score_wall)/2.0)

        rows.append({"model":m,
                     "frame":float(np.clip(score_frame,0,1)),
                     "wall": float(np.clip(score_wall,0,1)),
                     "dual": float(np.clip(score_dual,0,1)),
                     "braced":float(np.clip(score_braced,0,1))})
    df_sys = pd.DataFrame(rows)
    return df_sys

# ---------------------------
# S4 — structural similarity (motif + system)
# ---------------------------

def cosine(a,b):
    aa = np.linalg.norm(a); bb = np.linalg.norm(b)
    if aa==0 or bb==0: return 0.0
    return float(np.dot(a,b)/(aa*bb))

def s4_similarity(models, motif_shares, sys_scores, w_motif=0.5, w_system=0.5, alpha_m5=0.40):
    # motif vector: [M2,M2b,M3,M4, α*M5_LB, α*M5_Shear, α*M5_Moment]
    mot = motif_shares.set_index("model").copy()
    cols = ["M2_frameNode","M2b_braceNode","M3_wallSlab","M4_core","M5_LB","M5_Shear","M5_Moment"]
    for c in cols:
        if c not in mot.columns: mot[c]=0.0
        mot[c]=mot[c].astype(float)
    mot[["M5_LB","M5_Shear","M5_Moment"]] *= alpha_m5

    sys = sys_scores.set_index("model")[["frame","wall","dual","braced"]].astype(float)

    models = list(mot.index)
    S = pd.DataFrame(0.0, index=models, columns=models)
    for i,mi in enumerate(models):
        for j,mj in enumerate(models):
            if j<i: continue
            vi = mot.loc[mi, cols].values; vj = mot.loc[mj, cols].values
            si = sys.loc[mi].values;       sj = sys.loc[mj].values
            s_m = cosine(vi, vj); s_s = cosine(si, sj)
            s = w_motif*s_m + w_system*s_s
            S.loc[mi,mj] = S.loc[mj,mi] = s
    return S

# ---------------------------
# Main
# ---------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--pattern", default="*_DG.rdf")
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--out-name", required=True)
    ap.add_argument("--dual-thresh", type=float, default=0.25)
    ap.add_argument("--w-motif", type=float, default=0.5)
    ap.add_argument("--w-system", type=float, default=0.5)
    ap.add_argument("--alpha-m5", type=float, default=0.40)
    ap.add_argument("--proxy-penalty", type=float, default=0.70)
    ap.add_argument("--allow-type-only-proxy", action="store_true")
    ap.add_argument("--func-all", action="store_true")
    ap.add_argument("--emit-debug", action="store_true")
    args = ap.parse_args()

    # collect models
    files = [f for f in os.listdir(args.input_dir) if re.fullmatch(args.pattern.replace("*",".*"), f)]
    models = [Model(f, os.path.join(args.input_dir, f)) for f in files]

    out_dir = os.path.join(args.out_root, args.out_name)
    os.makedirs(out_dir, exist_ok=True)
    print("[LOAD]"); [print("  ", m.name) for m in models]

    # S1
    df_types, df_funcs = s1_inventories(models, func_all=args.func_all)
    df_types.to_csv(os.path.join(out_dir,"struct_types_histogram.csv"), index=False)
    df_funcs.to_csv(os.path.join(out_dir,"struct_functions_histogram.csv"), index=False)
    # convenience wide
    piv = df_funcs.pivot_table(index="model", columns="function", values="count", aggfunc="sum").fillna(0)
    piv.to_csv(os.path.join(out_dir,"struct_functions_shares_wide.csv"))

    # S2
    counts, shares, dens = s2_motifs(models, df_types, df_funcs,
                                     allow_type_only=args.allow_type_only_proxy,
                                     proxy_penalty=args.proxy_penalty,
                                     out_dir=out_dir)

    # S3
    sys_scores = s3_system_scores(models, shares, df_types, df_funcs, dual_thresh=args.dual_thresh)
    sys_scores.to_csv(os.path.join(out_dir,"struct_system_scores.csv"), index=False)
    # also components (for debug/teze ek görsel)
    sys_scores.to_csv(os.path.join(out_dir,"struct_score_components.csv"), index=False)

    # S4
    S = s4_similarity(models, shares, sys_scores,
                      w_motif=args.w_motif, w_system=args.w_system, alpha_m5=args.alpha_m5)
    S.to_csv(os.path.join(out_dir, "struct_similarity_matrix.csv"))
    # pairwise long summary
    rows=[]
    for i,a in enumerate(S.index):
        for j,b in enumerate(S.columns):
            if j<=i: continue
            rows.append({"A":a,"B":b,"similarity":S.loc[a,b]})
    pd.DataFrame(rows).to_csv(os.path.join(out_dir,"pairwise_structural_summary.csv"), index=False)

    # weights record
    with open(os.path.join(out_dir,"weights_used.json"),"w",encoding="utf-8") as f:
        json.dump({
            "dual_thresh": args.dual_thresh,
            "w_motif": args.w_motif,
            "w_system": args.w_system,
            "alpha_m5": args.alpha_m5,
            "proxy_penalty": args.proxy_penalty,
            "allow_type_only_proxy": bool(args.allow_type_only_proxy),
            "func_all": bool(args.func_all)
        }, f, indent=2)

    print(f"\n[OK] Saved outputs under: {out_dir}")
    print(" - struct_types_histogram.csv")
    print(" - struct_functions_histogram.csv")
    print(" - struct_functions_shares_wide.csv")
    print(" - struct_motif_counts.csv")
    print(" - struct_motif_shares.csv")
    print(" - struct_motif_densities_per100.csv")
    print(" - struct_system_scores.csv")
    print(" - struct_score_components.csv")
    print(" - struct_similarity_matrix.csv")
    print(" - pairwise_structural_summary.csv")
    print(" - weights_used.json")
    if os.path.exists(os.path.join(out_dir,"motif_proxy_summary.csv")):
        print(" - motif_proxy_summary.csv  (proxy evidence logged)")

if __name__=="__main__":
    main()
