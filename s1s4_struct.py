#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
S1→S4 Structural Channel (standards-aligned, thesis-defensible)
- S1: Element + function inventory, data availability
- S2: Motifs (M2 frame node, M3 wall-slab, M2b brace-frame, M4 core/slab),
      with relaxed/proxy counts + penalties (weak topology, type-only proxy)
- S3: System scores from motif densities + role shares (Frame, Wall, Braced, Dual)
- S4: Structural similarity = w_motif * cos([dens_M2,M3,M4,M2b] + α·[LB,Shear,Moment,Bracing])
                             + w_system * cos([Frame,Wall,Dual,Braced])

Outputs (pipeline-compatible names):
  s1_inventory.csv
  s2_motifs.csv                          # contains cnt_* and dens_* side-by-side
  s3_system_scores.csv
  s4_motif_share_vectors.csv             # “motif+M5” vector used by S4
  struct_similarity_s1s4_motif.csv
  struct_similarity_s1s4_system.csv
  struct_similarity_s1s4.csv
  s1s4_meta.json
"""

from __future__ import annotations
import os, re, json
from collections import defaultdict, Counter
from typing import Dict, List

import numpy as np
import pandas as pd
from rdflib import Graph, RDF, RDFS

# ---------------- Config ----------------

ELEMENT_RX = {
    "Beam"   : r"(?:\b|_)(beam|ifcbeam)(?:\b|_)",
    "Column" : r"(?:\b|_)(column|ifccolumn)(?:\b|_)",
    "Slab"   : r"(?:\b|_)(slab|deck|floor|plate|ifcslab|ifcfloor|ifcplate|ifccovering)(?:\b|_)",
    "Wall"   : r"(?:\b|_)(wall|shear[- ]?wall|ifcwall|ifcwallstandardcase)(?:\b|_)",
    "Brace"  : r"(?:\b|_)(brace|bracing|tie|strut|ifcstructuralcurvemember|ifcmember)(?:\b|_)",
    "Core"   : r"(?:\b|_)(core|shearcore|liftcore)(?:\b|_)",
}

FUNC_RX = {
    "LB"      : r"(?:\b|_)(load\s*bearing|bearing)(?:\b|_)",
    "Shear"   : r"(?:\b|_)(shear)(?:\b|_)",
    "Moment"  : r"(?:\b|_)(moment|bending)(?:\b|_)",
    "Bracing" : r"(?:\b|_)(brace|bracing|tie|strut|diaphragm|stiffener)(?:\b|_)",
}

STRONG_TOPO = {
    "adjacentelement", "intersectingelement", "connectedto",
    "isconnectedto", "relateselement", "relatedelements",
    "relconnectselements", "hasstructuralmember"
}
WEAK_TOPO = {"adjacentzone"}

DEFAULTS = {
    "dual_thresh"       : 0.25,
    "w_motif"           : 0.5,
    "w_system"          : 0.5,
    "alpha_m5"          : 0.40,  # weight for role shares in the motif vector
    "proxy_penalty"     : 0.70,  # multiplicative penalty when using type-only proxy
    "weak_topo_penalty" : 0.50,  # penalty when only weak topology exists
}

# ---------------- helpers ----------------

def _local(x: str) -> str:
    s = str(x)
    if "#" in s: s = s.split("#")[-1]
    if "/" in s: s = s.split("/")[-1]
    return s

def _norm_text_bag(types_set, labels_set):
    toks = []
    for t in types_set:
        toks.append(re.sub(r"[^A-Za-z0-9]+"," ",str(t)).lower())
    for l in labels_set:
        toks.append(re.sub(r"[^A-Za-z0-9]+"," ",str(l)).lower())
    return " ".join(toks)

def _compile(d: Dict[str, str]): return {k: re.compile(v, re.I) for k,v in d.items()}
E_RX = _compile(ELEMENT_RX)
F_RX = _compile(FUNC_RX)

# ---------------- S1: read & index ----------------

class Model:
    def __init__(self, name: str, g: Graph):
        self.name = name
        self.g = g
        self.types = defaultdict(set)       # node -> {type localnames}
        self.labels = defaultdict(set)      # node -> {labels}
        self.edges_by_pred = defaultdict(list)  # pred_local -> [(s,o),...]
        self._index()

    def _index(self):
        for s,p,o in self.g.triples((None, RDF.type, None)):
            self.types[s].add(_local(o))
        for s,p,o in self.g.triples((None, RDFS.label, None)):
            self.labels[s].add(str(o))
        for s,p,o in self.g:
            pred = _local(p).lower()
            if pred in STRONG_TOPO or pred in WEAK_TOPO or pred in {
                "hasfunction","hasquality","hasrole","hasstructuralfunction","hasstructuralrole"
            }:
                self.edges_by_pred[pred].append((s,o))

def _read_models(models_dir: str) -> List[Model]:
    out = []
    for fn in sorted(os.listdir(models_dir)):
        if not fn.lower().endswith(".rdf"): continue
        path = os.path.join(models_dir, fn)
        try:
            g = Graph(); g.parse(path)
            out.append(Model(fn, g))
        except Exception as e:
            print(f"[WARN] parse failed {fn}: {e}")
    if not out: raise RuntimeError("No RDF models parsed.")
    return out

def _classify_types(types_set, labels_set):
    bag = _norm_text_bag(types_set, labels_set)
    cats=set()
    for k,rx in E_RX.items():
        if rx.search(bag): cats.add(k)
    return cats

def _classify_funcs(func_objs, labels_set, func_all=True):
    bag = " ".join(
        [re.sub(r"[^A-Za-z0-9]+"," ",t).lower() for t in func_objs] +
        ([re.sub(r"[^A-Za-z0-9]+"," ",l).lower() for l in labels_set] if func_all else [])
    )
    cats=set()
    for k,rx in F_RX.items():
        if rx.search(bag): cats.add(k)
    return cats

def s1(models: List[Model]):
    rows_inv, func_wide_rows, data_av_rows = [], [], []
    node_type_map: Dict[str,Dict] = {}

    for m in models:
        node2types = defaultdict(set)
        node2funcs = defaultdict(set)

        # gather function target localnames
        func_targets = defaultdict(set)
        for pred in ["hasfunction","hasquality","hasrole","hasstructuralfunction","hasstructuralrole"]:
            for s,o in m.edges_by_pred.get(pred, []):
                func_targets[s].add(_local(o))

        # classify
        for n in set(list(m.types.keys()) + list(m.labels.keys()) + list(func_targets.keys())):
            tcats = _classify_types(m.types.get(n,set()), m.labels.get(n,set()))
            fcats = _classify_funcs(list(func_targets.get(n,set())), m.labels.get(n,set()), func_all=True)
            if tcats: node2types[n] |= tcats
            if fcats: node2funcs[n] |= fcats

        node_type_map[m.name] = node2types

        # inventory row (counts of element types)
        counts = Counter()
        for cs in node2types.values():
            for c in cs: counts[c]+=1
        rows_inv.append({"model": m.name, **{k:int(counts.get(k,0)) for k in ["Beam","Column","Slab","Wall","Brace","Core"]}})

        # function shares (normalize by structural element count)
        n_struct = max(1, sum(1 for cs in node2types.values() if cs & {"Beam","Column","Slab","Wall","Brace","Core"}))
        fcounts = Counter()
        for fs in node2funcs.values():
            for f in fs: fcounts[f]+=1
        func_wide_rows.append({
            "model": m.name,
            "LB": fcounts.get("LB",0)/n_struct,
            "Shear": fcounts.get("Shear",0)/n_struct,
            "Moment": fcounts.get("Moment",0)/n_struct,
            "Bracing": fcounts.get("Bracing",0)/n_struct,
            "_n_struct": n_struct
        })

        strong_present = any(m.edges_by_pred.get(p, []) for p in STRONG_TOPO)
        weak_present   = any(m.edges_by_pred.get(p, []) for p in WEAK_TOPO)
        data_av_rows.append({"model":m.name,"has_strong_topo":int(bool(strong_present)),"has_weak_topo":int(bool(weak_present))})

    inv_df   = pd.DataFrame(rows_inv).set_index("model").sort_index()
    func_df  = pd.DataFrame(func_wide_rows).set_index("model").sort_index()
    avail_df = pd.DataFrame(data_av_rows).set_index("model").sort_index()
    return inv_df, func_df, avail_df, node_type_map

# ---------------- S2: motifs w/ relaxed proxy + penalties ----------------

def s2(models: List[Model], node_type_map,
       allow_type_proxy=True,
       proxy_penalty=DEFAULTS["proxy_penalty"],
       weak_topo_penalty=DEFAULTS["weak_topo_penalty"]):

    def count_pairs(m: Model, Aset, Bset):
        pairs=set(); used_str=False; used_wk=False
        for pred in STRONG_TOPO:
            for s,o in m.edges_by_pred.get(pred, []):
                if (s in Aset and o in Bset) or (s in Bset and o in Aset):
                    pairs.add(frozenset([s,o])); used_str=True
        for pred in WEAK_TOPO:
            for s,o in m.edges_by_pred.get(pred, []):
                if (s in Aset and o in Bset) or (s in Bset and o in Aset):
                    pairs.add(frozenset([s,o])); used_wk=True
        return len(pairs), used_str, used_wk

    rows_counts, rows_dens = [], []

    for m in models:
        cats = node_type_map[m.name]
        E_beam   = {n for n,cs in cats.items() if "Beam" in cs}
        E_col    = {n for n,cs in cats.items() if "Column" in cs}
        E_slab   = {n for n,cs in cats.items() if "Slab" in cs}
        E_wall   = {n for n,cs in cats.items() if "Wall" in cs}
        E_brace  = {n for n,cs in cats.items() if "Brace" in cs}
        E_core   = {n for n,cs in cats.items() if "Core" in cs}
        frame    = E_beam | E_col
        denom    = max(1,len(E_beam|E_col|E_slab|E_wall|E_brace|E_core))

        # M2: beam—column
        c2, st2, wk2 = count_pairs(m, E_beam, E_col); p2=False
        if c2==0 and allow_type_proxy and E_beam and E_col:
            c2=min(len(E_beam),len(E_col)); p2=True
        d2 = (c2/denom)*100.0

        # M3: wall—slab
        c3, st3, wk3 = count_pairs(m, E_wall, E_slab); p3=False
        if c3==0 and allow_type_proxy and E_wall and E_slab:
            c3=min(len(E_wall),len(E_slab)); p3=True
        d3 = (c3/denom)*100.0

        # M2b: brace—frame
        c2b, st2b, wk2b = count_pairs(m, E_brace, frame); p2b=False
        if c2b==0 and allow_type_proxy and E_brace and frame:
            c2b=min(len(E_brace), len(frame)); p2b=True
        d2b = (c2b/denom)*100.0

        # M4: core—slab adjacency (≥1)
        core_good=0; st4=False; wk4=False
        if E_core and E_slab:
            adj=set()
            for pred in STRONG_TOPO:
                for s,o in m.edges_by_pred.get(pred, []):
                    if (s in E_core and o in E_slab): adj.add(s); st4=True
                    if (o in E_core and s in E_slab): adj.add(o); st4=True
            if not adj:
                for pred in WEAK_TOPO:
                    for s,o in m.edges_by_pred.get(pred, []):
                        if (s in E_core and o in E_slab): adj.add(s); wk4=True
                        if (o in E_core and s in E_slab): adj.add(o); wk4=True
            core_good=len(adj)
        p4=False
        if core_good==0 and allow_type_proxy and E_core:
            core_good=len(E_core); p4=True
        d4 = (core_good/denom)*100.0

        # penalties
        def penalize(d, proxy, strong, weak):
            v=d
            if proxy: v *= (1.0 - proxy_penalty)
            if (not strong) and (weak or proxy): v *= (1.0 - weak_topo_penalty)
            return v

        d2 = penalize(d2,p2,st2,wk2)
        d3 = penalize(d3,p3,st3,wk3)
        d2b= penalize(d2b,p2b,st2b,wk2b)
        d4 = penalize(d4,p4,st4,wk4)

        rows_counts.append({"model":m.name,"cnt_M2":c2,"cnt_M3":c3,"cnt_M4":core_good,"cnt_M2b":c2b,"_den":denom})
        rows_dens.append({"model":m.name,"dens_M2":d2,"dens_M3":d3,"dens_M4":d4,"dens_M2b":d2b})

    cnt_df = pd.DataFrame(rows_counts).set_index("model").sort_index()
    den_df = pd.DataFrame(rows_dens).set_index("model").sort_index()
    return cnt_df, den_df

# ---------------- S3: system scores ----------------

def s3(den_df: pd.DataFrame, func_df: pd.DataFrame, dual_thresh: float):
    # densities to 0–1
    A = den_df[["dens_M2","dens_M3","dens_M4","dens_M2b"]]/100.0
    B = func_df[["LB","Shear","Moment","Bracing"]]

    rows=[]; comps=[]
    for model in A.index:
        d2=float(A.loc[model,"dens_M2"]); d3=float(A.loc[model,"dens_M3"]); d2b=float(A.loc[model,"dens_M2b"])
        fLB=float(B.loc[model,"LB"]); fSh=float(B.loc[model,"Shear"]); fMo=float(B.loc[model,"Moment"]); fBr=float(B.loc[model,"Bracing"])

        frame  = 0.6*d2 + 0.4*fMo
        wall   = 0.6*d3 + 0.4*((fLB+fSh)/2.0)
        braced = 0.7*d2b + 0.3*fBr
        dual   = 0.5*(frame+wall) if (frame>=dual_thresh and wall>=dual_thresh) else min(frame,wall)

        rows.append({"model":model,"Frame":frame,"Wall":wall,"Dual":dual,"Braced":braced})
        comps.append({"model":model,"dens_M2":d2,"dens_M3":d3,"dens_M2b":d2b,"share_LB":fLB,"share_Shear":fSh,"share_Moment":fMo,"share_Bracing":fBr})

    sys_df   = pd.DataFrame(rows).set_index("model").sort_index()
    comps_df = pd.DataFrame(comps).set_index("model").sort_index()
    return sys_df, comps_df

# ---------------- S4: similarity (motif+M5 & systems) ----------------

def _pairwise_cos(df: pd.DataFrame) -> pd.DataFrame:
    idx=list(df.index); V=df.values.astype(float); n=len(idx)
    M=np.zeros((n,n),float)
    for i in range(n):
        ai=V[i]; nai=np.linalg.norm(ai)
        for j in range(n):
            bj=V[j]; nbj=np.linalg.norm(bj)
            c=0.0 if nai==0 or nbj==0 else float(np.dot(ai,bj)/(nai*nbj))
            # map [-1,1] → [0,1]
            M[i,j]=min(1.0,max(0.0,(c+1.0)/2.0))
    return pd.DataFrame(M,index=idx,columns=idx)

def s4(den_df: pd.DataFrame, func_df: pd.DataFrame, sys_df: pd.DataFrame,
       w_motif=DEFAULTS["w_motif"], w_system=DEFAULTS["w_system"], alpha_m5=DEFAULTS["alpha_m5"]):
    # motif+M5 vector (0–1)
    A = den_df[["dens_M2","dens_M3","dens_M4","dens_M2b"]]/100.0
    B = alpha_m5 * func_df[["LB","Shear","Moment","Bracing"]]
    V = pd.concat([A,B],axis=1)

    S_motif  = _pairwise_cos(V)
    S_system = _pairwise_cos(sys_df[["Frame","Wall","Dual","Braced"]])

    S_struct = w_motif*S_motif.values + w_system*S_system.values
    S_struct = np.clip(S_struct,0.0,1.0)
    S_struct = pd.DataFrame(S_struct, index=V.index, columns=V.index)
    return S_motif, S_system, S_struct, V

# ---------------- public API ----------------

def run_s1s4(models_dir: str, out_dir: str) -> Dict:
    os.makedirs(out_dir, exist_ok=True)

    models = _read_models(models_dir)
    # filter helper models (exclude 0000_Merged)
    models = [m for m in models if "0000_merged.rdf" not in m.name.lower()]

    inv_df, func_df, avail_df, node_type_map = s1(models)
    cnt_df, den_df = s2(models, node_type_map)
    sys_df, comps_df = s3(den_df, func_df, DEFAULTS["dual_thresh"])
    S_motif, S_system, S_struct, motif_share_df = s4(den_df, func_df, sys_df)

    # ---- save with pipeline-compatible names ----
    inv_df.reset_index().to_csv(os.path.join(out_dir,"s1_inventory.csv"), index=False)

    # S2 motifs table (counts + densities) — avoid shadowing function name
    motifs_table = pd.concat([cnt_df.add_prefix("cnt_"), den_df.add_prefix("dens_")], axis=1)
    motifs_table.reset_index().to_csv(os.path.join(out_dir,"s2_motifs.csv"), index=False)

    # s3 system scores
    sys_df.reset_index().to_csv(os.path.join(out_dir,"s3_system_scores.csv"), index=False)

    # s4 motif share vectors (our V)
    motif_share_df.reset_index().to_csv(os.path.join(out_dir,"s4_motif_share_vectors.csv"), index=False)

    # similarity matrices
    S_motif.to_csv(os.path.join(out_dir,"struct_similarity_s1s4_motif.csv"))
    S_system.to_csv(os.path.join(out_dir,"struct_similarity_s1s4_system.csv"))
    S_struct.to_csv(os.path.join(out_dir,"struct_similarity_s1s4.csv"))

    # meta
    meta = {"DEFAULTS":DEFAULTS,"ELEMENT_RX":ELEMENT_RX,"FUNC_RX":FUNC_RX,
            "STRONG_TOPO":list(STRONG_TOPO),"WEAK_TOPO":list(WEAK_TOPO),
            "models":[m.name for m in models]}
    with open(os.path.join(out_dir,"s1s4_meta.json"),"w",encoding="utf-8") as f:
        json.dump(meta,f,indent=2)

    return {"files":{
        "inv":"s1_inventory.csv","motifs":"s2_motifs.csv","systems":"s3_system_scores.csv",
        "vectors":"s4_motif_share_vectors.csv","S_motif":"struct_similarity_s1s4_motif.csv",
        "S_system":"struct_similarity_s1s4_system.csv","S_struct":"struct_similarity_s1s4.csv","meta":"s1s4_meta.json"
    }}
