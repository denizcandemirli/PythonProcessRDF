#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
structural_extension_v25l.py

S1  Tip/rol envanteri
S2  Yapısal motif sayımları + paylar (+isteğe bağlı tip-proxy)
S3  Sistem skorları (frame/wall/brace/dual)
S4  Yapısal benzerlik (motif + sistem vektörü)

Bu sürüm:
- S3 girişini sağlamlaştırır: motif_shares wide/long olabilir.
- allow_type_only_proxy akışını düzeltir.
- Eksik sütunlar için güvenli toplama yapar.

Gereksinimler: rdflib, pandas, numpy
"""

import os, re, json, argparse, math, itertools
from dataclasses import dataclass, field
from typing import Dict, Set, Tuple, List, Iterable

import numpy as np
import pandas as pd
from rdflib import Graph, URIRef, RDF, RDFS, Literal

# ----------------------------- yardımcılar -----------------------------

def local_name(x) -> str:
    if isinstance(x, URIRef):
        s = str(x)
        for sep in ['#', '/', ':']:
            if sep in s:
                s = s.split(sep)[-1]
        return s
    if isinstance(x, Literal):
        return str(x)
    return str(x)

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def ensure_numeric(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0)
    return out

def row_normalize(df: pd.DataFrame) -> pd.DataFrame:
    arr = df.values.astype(float)
    row_sum = arr.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1.0
    return pd.DataFrame(arr/row_sum, index=df.index, columns=df.columns)

def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    if na == 0 or nb == 0: return 0.0
    return float(np.dot(a,b)/(na*nb))

def safe_sum(df: pd.DataFrame, cols: Iterable[str]) -> pd.Series:
    cols = [c for c in cols if c in df.columns]
    if not cols:
        return pd.Series(0.0, index=df.index)
    return df[cols].sum(axis=1)

# ----------------------------- model -----------------------------

@dataclass
class Model:
    name: str
    g: Graph
    types: Dict[str, Set[str]] = field(default_factory=dict)
    labels: Dict[str, str] = field(default_factory=dict)
    functions: Dict[str, Set[str]] = field(default_factory=dict)

# ----------------------------- S0: yükleme -----------------------------

TYPE_REGEX = {
    "Beam":   re.compile(r"(Ifc)?(Beam|Girder|Lintel)", re.I),
    "Column": re.compile(r"(Ifc)?(Column|Pier)", re.I),
    "Slab":   re.compile(r"(Ifc)?(Slab|Floor|Deck|Plate)", re.I),
    "Wall":   re.compile(r"(Ifc)?(Wall|Shear[- ]?Wall|CoreWall|RetainingWall|ExternalWall|InternalWall)", re.I),
    "Brace":  re.compile(r"(Ifc)?(Brace|Bracing|Strut|Tie|Diagonal)", re.I),
    "Core":   re.compile(r"(Core)", re.I),
    "Foundation": re.compile(r"(Foundation|Footing|Pile|Raft)", re.I),
    # Space/Zone algısı:
    "Zone":   re.compile(r"(Space|Zone|Room|Area)", re.I),
    # Quality / Property sinyalleri:
    "Quality": re.compile(r"(Quality|Property|Parameter)", re.I),
    "Function": re.compile(r"(LoadBearing|Shear|Moment|Diaphragm|Stiffener)", re.I),
}

FUNCTION_REGEX = {
    "LoadBearing": re.compile(r"Load\s*Bearing|LoadBearing", re.I),
    "Shear":       re.compile(r"Shear", re.I),
    "Moment":      re.compile(r"Moment", re.I),
    "Diaphragm":   re.compile(r"Diaphragm", re.I),
    "Stiffener":   re.compile(r"Stiffener", re.I),
}

PRED_HINTS = {
    "adjacentElement": re.compile(r"adjacent(Element)?", re.I),
    "intersectingElement": re.compile(r"intersect(ing)?Element", re.I),
    "hasContinuantPart": re.compile(r"hasContinuantPart|partOf|hasPart", re.I),
    "adjacentZone": re.compile(r"adjacentZone|inZone|hasZone|containedIn", re.I),
    "hasFunction": re.compile(r"hasFunction|FunctionOf|hasStructuralFunction", re.I),
    "hasQuality": re.compile(r"hasQuality|hasProperty|hasParameter", re.I),
}

def load_models(input_dir: str, pattern: str) -> List[Model]:
    import glob
    paths = sorted(glob.glob(os.path.join(input_dir, pattern)))
    print("[LOAD]")
    models = []
    for p in paths:
        name = os.path.basename(p)
        print(f"   {name}")
        g = Graph()
        g.parse(p)
        models.append(Model(name=name, g=g))
    return models

# ----------------------------- S1: tip/rol envanteri -----------------------------

def classify_types(g: Graph) -> Tuple[Dict[str,Set[str]], Dict[str,str]]:
    types: Dict[str, Set[str]] = {}
    labels: Dict[str, str] = {}
    for s, p, o in g.triples((None, RDF.type, None)):
        sid = local_name(s)
        types.setdefault(sid, set()).add(local_name(o))
    for s, p, o in g.triples((None, RDFS.label, None)):
        sid = local_name(s)
        labels[sid] = str(o)
    return types, labels

def harvest_functions(g: Graph) -> Dict[str, Set[str]]:
    funcs: Dict[str, Set[str]] = {}
    for s, p, o in g:
        pred = local_name(p)
        if PRED_HINTS["hasFunction"].search(pred):
            sid = local_name(s)
            f = local_name(o)
            funcs.setdefault(sid, set()).add(f)
    return funcs

def is_type(name: str, rx: re.Pattern) -> bool:
    return bool(rx.search(name))

def node_category(node_types: Set[str], node_label: str) -> str:
    # E/Z/P/F/Q kategorisi
    txt = " ".join(node_types) + " " + (node_label or "")
    if TYPE_REGEX["Zone"].search(txt): return "Z"
    if TYPE_REGEX["Quality"].search(txt): return "Q"
    if TYPE_REGEX["Function"].search(txt): return "F"
    # Element varsayılan
    return "E"

def s1_inventories(models: List[Model]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows_type = []
    rows_func = []
    rows_av   = []

    for m in models:
        m.types, m.labels = classify_types(m.g)
        m.functions       = harvest_functions(m.g)

        # tip histogram
        el_counts = {"Beam":0,"Column":0,"Slab":0,"Wall":0,"Brace":0,"Core":0,"Foundation":0,"OtherElement":0}
        zone_count=0
        for n, tset in m.types.items():
            txt = " ".join(tset) + " " + m.labels.get(n,"")
            if is_type(txt, TYPE_REGEX["Zone"]):
                zone_count += 1
                continue
            matched=False
            for k,rx in TYPE_REGEX.items():
                if k in ["Zone","Quality","Function"]: continue
                if rx.search(txt):
                    el_counts[k]+=1
                    matched=True
                    break
            if not matched:
                el_counts["OtherElement"] += 1
        total_E = sum(el_counts.values())

        for k,v in el_counts.items():
            share = (v/total_E) if total_E>0 else 0.0
            rows_type.append({"model": m.name, "type": k, "count": v, "share": share})

        # fonksiyon histogramı (hasFunction üzerinden)
        f_counts = {"LoadBearing":0,"Shear":0,"Moment":0,"Diaphragm":0,"Stiffener":0}
        for n, fset in m.functions.items():
            for f in fset:
                for fname, rx in FUNCTION_REGEX.items():
                    if rx.search(f):
                        f_counts[fname]+=1
        total_F = sum(f_counts.values())
        for k,v in f_counts.items():
            share = (v/total_F) if total_F>0 else 0.0
            rows_func.append({"model": m.name, "function": k, "count": v, "share": share})

        # veri bulunurluğu özet
        def count_pred(key: str) -> int:
            rx = PRED_HINTS[key]
            c = 0
            for _,p,_ in m.g:
                if rx.search(local_name(p)): c += 1
            return c

        rows_av.append({
            "model": m.name,
            "n_elements": total_E,
            "n_zones": zone_count,
            "edges_adjacentElement": count_pred("adjacentElement"),
            "edges_intersectingElement": count_pred("intersectingElement"),
            "edges_adjacentZone": count_pred("adjacentZone"),
            "edges_hasContinuantPart": count_pred("hasContinuantPart"),
            "edges_hasFunction": count_pred("hasFunction"),
            "edges_hasQuality": count_pred("hasQuality"),
            "n_functions": total_F
        })

    df_types = pd.DataFrame(rows_type).sort_values(["model","type"])
    df_funcs = pd.DataFrame(rows_func).sort_values(["model","function"])
    df_av    = pd.DataFrame(rows_av).sort_values("model")

    # fonksiyon paylarını wide da verelim (S3 kolaylığı)
    df_func_wide = df_funcs.pivot(index="model", columns="function", values="share").fillna(0.0)

    return df_types, df_funcs, df_func_wide, df_av

# ----------------------------- S2: motifler -----------------------------

MOTIF_LIST = [
    "M1_adjacentZone_ZZ",
    "M1b_adjacentZone_EZ",
    "M2_adjacentElement_EE",
    "M2b_adjacentElement_EZ",
    "M3_intersectingElement_EE",
    "M3b_intersectingElement_EZ",
    "M4_hasContinuantPart_EP",
    "M5_hasFunction_EF",
    "M6_hasQuality_EQ",
    # Basit birleşik sinyaller (mevcut veride zayıfsa 0 kalabilir)
    "M7_adjacentElement_plus_Function",
    "M8_adjacentElement_plus_Quality",
]

def s2_motifs(models: List[Model],
              df_types: pd.DataFrame,
              df_funcs: pd.DataFrame,
              allow_type_only: bool=False,
              penalty_if_fallback: float=0.5,
              out_dir: str=None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, bool]:
    """
    motif_counts: wide (index=model, cols=motif)
    motif_shares: wide (row-normalize counts)
    motif_dens:   motif sayıları / 100 eleman (ölçek etkisini görmek için)
    """
    used_fallback = False
    counts = pd.DataFrame(0, index=[m.name for m in models], columns=MOTIF_LIST, dtype=float)

    # hızlı kategorizasyon hazırlığı
    node_cat: Dict[Tuple[str,str], str] = {}  # (model,node)->E/Z/F/Q
    for m in models:
        for n, tset in m.types.items():
            node_cat[(m.name, n)] = node_category(tset, m.labels.get(n,""))

    # doğrudan predikatlara bakarak motif say
    for m in models:
        for s,p,o in m.g:
            pred = local_name(p)
            ps = local_name(s); po = local_name(o)
            cs = node_cat.get((m.name, ps), "E")
            co = node_cat.get((m.name, po), "E")

            if PRED_HINTS["adjacentZone"].search(pred):
                if cs=="Z" and co=="Z": counts.loc[m.name,"M1_adjacentZone_ZZ"] += 1
                elif (cs=="E" and co=="Z") or (cs=="Z" and co=="E"): counts.loc[m.name,"M1b_adjacentZone_EZ"] += 1

            if PRED_HINTS["adjacentElement"].search(pred):
                if cs=="E" and co=="E": counts.loc[m.name,"M2_adjacentElement_EE"] += 1
                elif (cs=="E" and co=="Z") or (cs=="Z" and co=="E"): counts.loc[m.name,"M2b_adjacentElement_EZ"] += 1

            if PRED_HINTS["intersectingElement"].search(pred):
                if cs=="E" and co=="E": counts.loc[m.name,"M3_intersectingElement_EE"] += 1
                elif (cs=="E" and co=="Z") or (cs=="Z" and co=="E"): counts.loc[m.name,"M3b_intersectingElement_EZ"] += 1

            if PRED_HINTS["hasContinuantPart"].search(pred) and cs=="E":
                # P olarak işaretleyemiyorsak yine de say
                counts.loc[m.name,"M4_hasContinuantPart_EP"] += 1

            if PRED_HINTS["hasFunction"].search(pred) and cs=="E":
                counts.loc[m.name,"M5_hasFunction_EF"] += 1

            if PRED_HINTS["hasQuality"].search(pred) and cs=="E":
                counts.loc[m.name,"M6_hasQuality_EQ"] += 1

        # birleşik (proxy) sinyaller: E-E adjacency + fonksiyon/quality bağı
        ee = counts.loc[m.name, "M2_adjacentElement_EE"]
        ef = counts.loc[m.name, "M5_hasFunction_EF"]
        eq = counts.loc[m.name, "M6_hasQuality_EQ"]
        counts.loc[m.name, "M7_adjacentElement_plus_Function"] += min(ee, ef)
        counts.loc[m.name, "M8_adjacentElement_plus_Quality"]  += min(ee, eq)

    # tip-proxy: veri çok seyrekse tip isimlerinden motif tahmini (cezalı)
    if allow_type_only:
        used_fallback = True
        # Wall-Slab kesişimi için kaba tahmin: her ikisi de varsa bir miktar puan
        for m in models:
            t = df_types[df_types["model"]==m.name].set_index("type")["count"]
            wall = float(t.get("Wall",0)); slab = float(t.get("Slab",0))
            add = penalty_if_fallback * min(wall, slab)
            counts.loc[m.name, "M3_intersectingElement_EE"] += add

    counts = ensure_numeric(counts)
    shares = row_normalize(counts)
    # 100 element başına yoğunluk
    elem_per_model = (df_types[df_types["type"].isin(["Beam","Column","Slab","Wall","Brace","Core","Foundation","OtherElement"])]
                      .groupby("model")["count"].sum())
    dens_rows = []
    for m in counts.index:
        denom = max(float(elem_per_model.get(m,0.0)), 1.0)
        dens_rows.append((m, list((counts.loc[m,:]/denom)*100.0)))
    dens = pd.DataFrame([r[1] for r in dens_rows], index=counts.index, columns=counts.columns)

    if out_dir:
        counts.to_csv(os.path.join(out_dir,"struct_motif_counts.csv"))
        shares.to_csv(os.path.join(out_dir,"struct_motif_shares.csv"))
        dens.to_csv(os.path.join(out_dir,"struct_motif_densities_per100.csv"))

    return counts, shares, dens, used_fallback

# ----------------------------- S3: sistem skorları -----------------------------

def standardize_motif_shares(motif_shares: pd.DataFrame) -> pd.DataFrame:
    """
    Giriş: long (model,motif,share) veya wide (index=model, cols=motif)
    Çıkış: wide (index=model, cols=motif), sayısal.
    """
    if {"model","motif","share"}.issubset(motif_shares.columns):
        wide = (motif_shares.pivot(index="model", columns="motif", values="share")
                          .fillna(0.0))
    else:
        wide = motif_shares.copy()
        if wide.index.name is None:
            wide.index.name = "model"
    return ensure_numeric(wide)

def s3_system_scores(models: List[Model],
                     motif_shares: pd.DataFrame,
                     df_types: pd.DataFrame,
                     df_funcs: pd.DataFrame,
                     dual_thresh: float = 0.25,
                     out_dir: str = None) -> pd.DataFrame:

    wide = standardize_motif_shares(motif_shares)
    # fonksiyon paylarını wide'a çevir
    if {"model","function","share"}.issubset(df_funcs.columns):
        func_wide = (df_funcs.pivot(index="model", columns="function", values="share")
                             .reindex(index=wide.index)
                             .fillna(0.0))
    else:
        func_wide = pd.DataFrame(0.0, index=wide.index, columns=["LoadBearing","Shear","Moment"])

    # sinyaller (sütun adlarınızı burada genişletebilirsiniz)
    frame_sig = safe_sum(wide, ["M2_adjacentElement_EE","M7_adjacentElement_plus_Function"])
    wall_sig  = safe_sum(wide, ["M3_intersectingElement_EE","M3b_intersectingElement_EZ","M4_hasContinuantPart_EP"])
    brace_sig = safe_sum(wide, ["M2b_adjacentElement_EZ"])  # varsa başka brace motiflerini ekleyin

    lb = func_wide.reindex(columns=["LoadBearing"], fill_value=0.0).squeeze()
    sh = func_wide.reindex(columns=["Shear"],       fill_value=0.0).squeeze()
    mo = func_wide.reindex(columns=["Moment"],      fill_value=0.0).squeeze()

    frame_score = (frame_sig + 0.5*mo).clip(lower=0)
    wall_score  = (wall_sig  + 0.5*(lb+sh)).clip(lower=0)
    brace_score = (brace_sig + 0.25*lb).clip(lower=0)

    tot = (frame_score + wall_score).replace(0, np.nan)
    frame_ratio = (frame_score / tot).fillna(0.0)
    dual_score  = ((frame_score>0)&(wall_score>0)&(frame_ratio>=dual_thresh)).astype(float)

    sys_scores = pd.DataFrame({
        "frame": frame_score,
        "wall":  wall_score,
        "brace": brace_score,
        "dual":  dual_score
    }, index=wide.index)

    if out_dir:
        sys_scores.to_csv(os.path.join(out_dir,"struct_system_scores.csv"))

        # bileşen dökümü (opsiyonel)
        comp = pd.DataFrame({
            "frame_sig": frame_sig, "wall_sig": wall_sig, "brace_sig": brace_sig,
            "LB": lb, "SH": sh, "MO": mo, "frame_ratio": frame_ratio
        }, index=wide.index)
        comp.to_csv(os.path.join(out_dir,"struct_score_components.csv"))

    return sys_scores

# ----------------------------- S4: yapısal benzerlik -----------------------------

def s4_structural_similarity(motif_shares_wide: pd.DataFrame,
                             sys_scores: pd.DataFrame,
                             w_motif: float = 0.5,
                             w_system: float = 0.5) -> Tuple[pd.DataFrame, pd.DataFrame]:
    models = list(motif_shares_wide.index)
    M = len(models)
    S_motif = np.zeros((M,M))
    S_sys   = np.zeros((M,M))
    for i,a in enumerate(models):
        va = motif_shares_wide.loc[a,:].to_numpy(dtype=float)
        sa = sys_scores.loc[a,:].to_numpy(dtype=float)
        for j,b in enumerate(models):
            vb = motif_shares_wide.loc[b,:].to_numpy(dtype=float)
            sb = sys_scores.loc[b,:].to_numpy(dtype=float)
            S_motif[i,j] = cosine(va,vb)
            S_sys[i,j]   = cosine(sa,sb)
    S_total = w_motif*S_motif + w_system*S_sys

    df_total = pd.DataFrame(S_total, index=models, columns=models)
    pairs=[]
    for i in range(M):
        for j in range(i+1,M):
            pairs.append({
                "A": models[i], "B": models[j],
                "S_motif": S_motif[i,j], "S_system": S_sys[i,j], "S_struct_total": S_total[i,j]
            })
    df_pairs = pd.DataFrame(pairs).sort_values("S_struct_total", ascending=False)
    return df_total, df_pairs

# ----------------------------- main -----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--pattern", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--out-name", required=True)
    ap.add_argument("--dual-thresh", type=float, default=0.25)
    ap.add_argument("--w-motif", type=float, default=0.5)
    ap.add_argument("--w-system", type=float, default=0.5)
    ap.add_argument("--allow-type-only-proxy", action="store_true")
    ap.add_argument("--emit-debug", action="store_true")
    args = ap.parse_args()

    out_dir = os.path.join(args.out_root, args.out_name)
    ensure_dir(out_dir)

    models = load_models(args.input_dir, args.pattern)

    # S1
    df_types, df_funcs, df_func_wide, df_av = s1_inventories(models)
    df_types.to_csv(os.path.join(out_dir,"struct_types_histogram.csv"), index=False)
    df_funcs.to_csv(os.path.join(out_dir,"struct_functions_histogram.csv"), index=False)
    df_func_wide.to_csv(os.path.join(out_dir,"struct_functions_shares_wide.csv"))
    df_av.to_csv(os.path.join(out_dir,"struct_data_availability.csv"), index=False)

    # S2
    allow_fallback = bool(args.allow_type_only_proxy)  # typo fix + argparse adıyla uyum
    motif_counts, motif_shares, motif_dens, used_fallback = s2_motifs(
        models=models, df_types=df_types, df_funcs=df_funcs,
        allow_type_only=allow_fallback, penalty_if_fallback=0.5, out_dir=out_dir
    )

    # S3
    sys_scores = s3_system_scores(
        models=models, motif_shares=motif_shares,
        df_types=df_types, df_funcs=df_funcs,
        dual_thresh=args.dual_thresh, out_dir=out_dir
    )

    # S4
    motif_wide = standardize_motif_shares(motif_shares)
    S_struct, pairs = s4_structural_similarity(motif_wide, sys_scores,
                                               w_motif=args.w_motif, w_system=args.w_system)
    S_struct.to_csv(os.path.join(out_dir,"struct_similarity_matrix.csv"))
    pairs.to_csv(os.path.join(out_dir,"pairwise_structural_summary.csv"), index=False)

    # Ağırlıklar
    with open(os.path.join(out_dir,"weights_used.json"),"w",encoding="utf-8") as f:
        json.dump({"w_motif": args.w_motif, "w_system": args.w_system,
                   "dual_thresh": args.dual_thresh,
                   "allow_type_only_proxy": allow_fallback}, f, indent=2)

    print("\n[OK] Saved outputs under:", out_dir)
    print(" - struct_types_histogram.csv")
    print(" - struct_functions_histogram.csv")
    print(" - struct_functions_shares_wide.csv")
    print(" - struct_data_availability.csv")
    print(" - struct_motif_counts.csv")
    print(" - struct_motif_shares.csv")
    print(" - struct_motif_densities_per100.csv")
    print(" - struct_system_scores.csv")
    print(" - struct_similarity_matrix.csv")
    print(" - pairwise_structural_summary.csv")
    print(" - weights_used.json")
    if used_fallback:
        print("   (info) type-only proxy used in S2 with penalty.")
    if args.emit_debug:
        # basit ek raporlar istenirse burada üretilebilir
        pass

if __name__ == "__main__":
    main()
