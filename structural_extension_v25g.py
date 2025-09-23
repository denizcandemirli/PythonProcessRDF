# structural_extension_v25g.py (hot-fix integrated)
# Structural Extension (v25g + hf1)
# - IFC/regex/label/part propagation for coarse type mapping
# - Unknown filter tightened (exclude OWL/RDFS meta + quality/function classes)
# - Manual overrides support (struct_type_overrides.json)
# - Element-like node gating (adj/zone/part/func endpoints)
# - Motifs: M2_frameNode, M2_braceNode, M3_wallSlab, M3_wallSlabAdj, M3_wallSlabZone, M4_core, M5_structRole
# - System scores (frame/wall/dual/braced) + structural similarity (S_motif, S_system, S_struct_total)
#
# Requirements: rdflib==6.3.2, pandas==2.2.2, numpy
# Usage:
#   (venv) python structural_extension_v25g.py ^
#       --input-dir "." ^
#       --pattern "*_DG.rdf" ^
#       --out-root ".\repro_pack\output" ^
#       --out-name "07 - Structural_Extension_v25g" ^
#       --func-all --dual-thresh 0.25 --emit-debug

import argparse, os, glob, json, re, math, itertools
from collections import defaultdict, Counter
import pandas as pd
import numpy as np
from rdflib import Graph, URIRef, BNode, Literal, RDF, RDFS

# -----------------------
# Configuration / Mapping
# -----------------------

EXCLUDE_CLASS_LOCALS = {
    # OWL/RDFS/Meta
    "Class","Restriction","ObjectProperty","AnnotationProperty","Axiom",
    "FunctionalProperty","SymmetricProperty","DatatypeProperty","Datatype",
    "AllDisjointClasses","InverseFunctionalProperty","AsymmetricProperty",
    "Ontology","OntologyProperty",
    # Non-struct domains
    "Quality","Function","StructuralFunction","LoadBearingFunction",
    "ThermalResistance","ThermalMass","HeatTransferCoefficient"
}

IFC_LOCAL_MAP = {
    "IfcWall":"Wall", "IfcWallStandardCase":"Wall", "IfcCoreWall":"Wall",
    "IfcSlab":"Slab", "IfcFloor":"Slab", "IfcPlate":"Slab",
    "IfcColumn":"Column", "IfcBeam":"Beam", "IfcMember":"Beam",
    "IfcStructuralCurveMember":"Brace",
    "IfcFooting":"Foundation", "IfcPile":"Foundation", "IfcPadFooting":"Foundation",
    "IfcMatFoundation":"Foundation", "IfcStripFooting":"Foundation"
}

TYPE_REGEX = {
    "Wall":       r"(wall|shear[- ]?wall|core[- ]?wall|wand|duvar)\b",
    "Slab":       r"(slab|deck|floor|platte|decke|döşeme|plate)\b",
    "Beam":       r"(beam|träger|kiriş|lintel|balken)\b",
    "Column":     r"(column|stütze|kolon|pilar|pillar|stützen)\b",
    "Brace":      r"(brace|bracing|diagonal|kreuzverband|çapraz|strut|tie)\b",
    "Core":       r"(core|kern|çekirdek)\b",
    "Foundation": r"(foundation|footing|pile|caisson|raft|mat|pad|strip|fundament)\b"
}

FUNC_REGEX = {
    "LoadBearing": r"load[- ]?bearing|tragend|yük( )?taşıy[ıi]c[ıi]",
    "Shear":       r"shear|schub|kesme",
    "Moment":      r"moment|momenten|moment[- ]?frame|rigid[- ]?frame"
}

# Predicates matched by localName (case-insensitive)
PRED_NAMES = {
    "adjacentElement": "adjacentElement",
    "adjacentZone": "adjacentZone",
    "intersectingElement": "intersectingElement",
    "hasPart": "BFO_0000178",      # hasContinuantPart
    "hasFunction": "hasFunction",
    "hasQuality": "hasQuality"
}

# -----------------------
# Helpers
# -----------------------

def local_name(x):
    if isinstance(x, (URIRef, BNode)):
        s = str(x)
        if "#" in s:
            return s.rsplit("#",1)[1]
        return s.rsplit("/",1)[-1]
    if isinstance(x, Literal):
        return str(x)
    return str(x)

def pred_local(p):
    return local_name(p).lower()

def label_texts(g, n):
    vals = []
    for _,_,lbl in g.triples((n, RDFS.label, None)):
        try:
            vals.append(str(lbl))
        except:
            pass
    # Also consider localName as a pseudo-label
    loc = local_name(n)
    if loc and loc not in vals:
        vals.append(loc)
    return vals

def is_predicate(p, key):
    """Check if predicate local name matches target (case-insensitive)."""
    return pred_local(p) == PRED_NAMES[key].lower()

def classify_by_text(text: str):
    if not text: return None
    t = text.lower()
    # IFC quick map
    for k,v in IFC_LOCAL_MAP.items():
        if k.lower() in t:
            return v
    # Regex buckets
    for coarse, pat in TYPE_REGEX.items():
        if re.search(pat, t, flags=re.I):
            return coarse
    return None

def cosine_sim(a: np.ndarray, b: np.ndarray):
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a,b) / (na*nb))

def _safe_sum(series) -> int:
    """Series boşsa 0, doluysa toplam (int) döndürür."""
    try:
        return int(pd.to_numeric(series, errors="coerce").fillna(0).sum())
    except Exception:
        return 0

# -----------------------
# Data model per RDF
# -----------------------

class ModelData:
    def __init__(self, name, graph):
        self.name = name
        self.g = graph
        self.types = defaultdict(set)     # node -> set(class)
        self.labels = defaultdict(list)   # node -> [labels]
        self.adj_edges = []               # (u,v)
        self.zone_edges = []              # (u,zone)
        self.int_edges = []               # (u,v)
        self.part_edges = []              # (e,part)
        self.func_edges = []              # (e,f)
        self.func_types = defaultdict(set) # f -> set(class)
        self.nodes = set()

    def build(self):
        g = self.g
        # types & labels
        for s,p,o in g.triples((None, RDF.type, None)):
            self.types[s].add(o); self.nodes.add(s)
        for n in set(self.nodes):
            lbls = label_texts(g, n)
            if lbls:
                self.labels[n] += lbls

        # edges of interest
        for s,p,o in g:
            pl = pred_local(p)
            if pl == PRED_NAMES["adjacentElement"].lower():
                self.adj_edges.append((s,o))
                self.nodes.add(s); self.nodes.add(o)
            elif pl == PRED_NAMES["adjacentZone"].lower():
                self.zone_edges.append((s,o))
                self.nodes.add(s); self.nodes.add(o)
            elif pl == PRED_NAMES["intersectingElement"].lower():
                self.int_edges.append((s,o))
                self.nodes.add(s); self.nodes.add(o)
            elif pl == PRED_NAMES["hasPart"].lower():
                self.part_edges.append((s,o))
                self.nodes.add(s); self.nodes.add(o)
            elif pl == PRED_NAMES["hasFunction"].lower():
                self.func_edges.append((s,o))
                self.nodes.add(s); self.nodes.add(o)
            elif pl == RDF.type and isinstance(o, URIRef):
                # already handled above
                pass

        # function node types
        for f,p,o in g.triples((None, RDF.type, None)):
            if f in {x for _,x in self.func_edges}:
                self.func_types[f].add(o)

    # iterator over element-like nodes
    def iter_element_like(self):
        seen = set()
        for (u,v) in self.adj_edges:
            for n in (u,v):
                if n not in seen: seen.add(n); yield n
        for (u,z) in self.zone_edges:
            for n in (u,z):
                if n not in seen: seen.add(n); yield n
        for (e,p) in self.part_edges:
            if e not in seen: seen.add(e); yield e
        for (e,f) in self.func_edges:
            if e not in seen: seen.add(e); yield e

# -----------------------
# Mapping & overrides
# -----------------------

def decide_coarse_type_for_node(n, md, overrides):
    # a) manual override
    loc = local_name(n)
    if loc and loc in overrides:
        return overrides[loc]

    # b) instance text (localName + labels)
    texts = [loc] + md.labels.get(n, [])
    for t in texts:
        ct = classify_by_text(t)
        if ct:
            return ct

    # c) parts → parent
    parts = [p for (e,p) in md.part_edges if e == n]
    ptexts = []
    for p in parts:
        ptexts += [local_name(p)] + md.labels.get(p, [])
        for cl in md.types.get(p, []):
            ptexts.append(local_name(cl))
    for t in ptexts:
        ct = classify_by_text(t)
        if ct:
            return ct

    # d) ontology types (filtered)
    for cl in md.types.get(n, []):
        clloc = local_name(cl)
        if clloc in EXCLUDE_CLASS_LOCALS:
            continue
        ct = classify_by_text(clloc)
        if ct:
            return ct

    return None

# -----------------------
# Pipeline
# -----------------------

def stage_S1(models, overrides, func_all=False, emit_debug=False, out_dir=None):
    """
    - Coarse type mapping per node (Ifc/regex/label/part/ontology; with EXCLUDE filter)
    - Function shares (LoadBearing/Shear/Moment/Other)
    - Diagnostics: type_mapping_hits.csv, type_mapping_unknown.csv
    Returns:
      type_counts_df, func_hist_df, func_shares_wide_df, availability_df, node2type (per model)
    """
    hits = []      # {model, source, coarse_type, evidence}
    unknown = []   # {model, class_local, count}
    node2type = {} # model -> {node: coarse}
    per_model_type_counts = defaultdict(Counter)

    # Map nodes
    for md in models:
        m = md.name
        n2t = {}
        # count unknown classes only for element-like nodes
        unknown_local = Counter()

        for n in md.iter_element_like():
            ct = decide_coarse_type_for_node(n, md, overrides)
            if ct:
                n2t[n] = ct
                per_model_type_counts[m][ct] += 1
            else:
                # collect unknown class locals (filtered)
                for cl in md.types.get(n, []):
                    clloc = local_name(cl)
                    if clloc not in EXCLUDE_CLASS_LOCALS:
                        unknown_local[clloc] += 1

            # hit provenance (best-effort)
            loc = local_name(n)
            decided = None
            if loc and loc in overrides:
                decided = ("override", overrides[loc])
            if not decided:
                texts = [loc] + md.labels.get(n, [])
                for t in texts:
                    c = classify_by_text(t)
                    if c:
                        src = "ifc_local" if any(k.lower() in (t or "").lower() for k in IFC_LOCAL_MAP.keys()) else "label_regex"
                        decided = (src, c); break
            if not decided:
                parts = [p for (e,p) in md.part_edges if e == n]
                for p in parts:
                    ptexts = [local_name(p)] + md.labels.get(p, [])
                    for t in ptexts:
                        c = classify_by_text(t)
                        if c:
                            decided = ("part_propagation", c)
                            break
                    if decided: break
            if not decided:
                for cl in md.types.get(n, []):
                    clloc = local_name(cl)
                    if clloc in EXCLUDE_CLASS_LOCALS:
                        continue
                    c = classify_by_text(clloc)
                    if c:
                        decided = ("ontology", c)
                        break
            if decided:
                hits.append({"model": m, "source": decided[0], "coarse_type": decided[1], "count_nodes": 1})

        node2type[m] = n2t
        for k,v in unknown_local.items():
            unknown.append({"model": m, "class_local": k, "count": v})

    # Type histogram (per model, with share)
    rows_types = []
    for m, cnts in per_model_type_counts.items():
        total = sum(cnts.values())
        for t,c in cnts.items():
            share = c/total if total>0 else 0.0
            rows_types.append({"model": m, "type": t, "count": c, "share": share})
    df_types = pd.DataFrame(rows_types).sort_values(["model","type"])

    # Function histogram (per model)
    func_rows = []
    func_wide = []
    for md in models:
        m = md.name
        struct_nodes = set(node2type[m].keys())
        lb = sh = mo = oth = 0
        total_func = 0
        for (e,f) in md.func_edges:
            if (not func_all) and (e not in struct_nodes):
                continue
            clocs = [local_name(c) for c in md.func_types.get(f, [])]
            joined = " ".join(clocs).lower()
            total_func += 1
            if re.search(FUNC_REGEX["LoadBearing"], joined, re.I):
                lb += 1
            elif re.search(FUNC_REGEX["Shear"], joined, re.I):
                sh += 1
            elif re.search(FUNC_REGEX["Moment"], joined, re.I):
                mo += 1
            else:
                oth += 1
        for label,val in [("LoadBearing",lb),("Shear",sh),("Moment",mo),("Other",oth)]:
            share = val/total_func if total_func>0 else 0.0
            func_rows.append({"model": m, "function": label, "count": val, "share": share})
        func_wide.append({"model": m,
                          "LoadBearing": lb/(total_func or 1.0),
                          "Shear": sh/(total_func or 1.0),
                          "Moment": mo/(total_func or 1.0),
                          "Other": oth/(total_func or 1.0)})
    df_funcs = pd.DataFrame(func_rows).sort_values(["model","function"])
    df_func_wide = pd.DataFrame(func_wide).sort_values("model")

    # Availability / diagnostics (hot-fix: safe sums)
    av_rows = []
    for md in models:
        m = md.name
        struct_nodes = set(node2type[m].keys())
        type_counts = Counter(node2type[m].values())

        lb_cnt = _safe_sum(df_funcs.loc[(df_funcs["model"]==m) & (df_funcs["function"]=="LoadBearing"), "count"])
        sh_cnt = _safe_sum(df_funcs.loc[(df_funcs["model"]==m) & (df_funcs["function"]=="Shear"), "count"])
        mo_cnt = _safe_sum(df_funcs.loc[(df_funcs["model"]==m) & (df_funcs["function"]=="Moment"), "count"])
        total_func = _safe_sum(df_funcs.loc[(df_funcs["model"]==m), "count"])

        av_rows.append({
            "model": m,
            "n_struct_nodes": len(struct_nodes),
            "n_type_Wall": type_counts.get("Wall",0),
            "n_type_Slab": type_counts.get("Slab",0),
            "n_type_Beam": type_counts.get("Beam",0),
            "n_type_Column": type_counts.get("Column",0),
            "n_type_Brace": type_counts.get("Brace",0),
            "n_type_Core": type_counts.get("Core",0),
            "n_type_Foundation": type_counts.get("Foundation",0),
            "n_funcs_total": total_func,
            "n_func_LB": lb_cnt,
            "n_func_Shear": sh_cnt,
            "n_func_Moment": mo_cnt,
            "n_adj_edges": len(md.adj_edges),
            "n_int_edges": len(md.int_edges),
            "n_part_edges": len(md.part_edges),
            "n_adj_zone": len(md.zone_edges),
            "has_adj": 1 if len(md.adj_edges)>0 else 0,
            "has_int": 1 if len(md.int_edges)>0 else 0,
            "has_zone": 1 if len(md.zone_edges)>0 else 0
        })
    df_av = pd.DataFrame(av_rows).sort_values("model")

    # Emit diagnostics (hot-fix: empty guards)
    if emit_debug and out_dir:
        if len(hits) > 0:
            (pd.DataFrame(hits)
               .groupby(["model","source","coarse_type"], as_index=False)["count_nodes"].sum()
               .sort_values(["model","count_nodes"], ascending=[True,False])
               ).to_csv(os.path.join(out_dir,"type_mapping_hits.csv"), index=False)
        else:
            pd.DataFrame(columns=["model","source","coarse_type","count_nodes"])\
              .to_csv(os.path.join(out_dir,"type_mapping_hits.csv"), index=False)

        if len(unknown) > 0:
            (pd.DataFrame(unknown)
               .groupby(["model","class_local"], as_index=False)["count"].sum()
               .sort_values(["model","count"], ascending=[True,False])
               ).to_csv(os.path.join(out_dir,"type_mapping_unknown.csv"), index=False)
        else:
            pd.DataFrame(columns=["model","class_local","count"])\
              .to_csv(os.path.join(out_dir,"type_mapping_unknown.csv"), index=False)

    return df_types, df_funcs, df_func_wide, df_av, node2type


def stage_S2(models, node2type, func_shares_wide):
    """
    Motif counts & densities per model.
    Returns: motif_counts_df, motif_shares_df, densities_per100_df
    """
    MOTIFS = ["M2_frameNode","M2_braceNode","M3_wallSlab","M3_wallSlabAdj","M3_wallSlabZone","M4_core","M5_structRole"]
    rows_counts = []
    rows_share = []
    rows_density = []

    for md in models:
        m = md.name
        n2t = node2type[m]
        adj = defaultdict(set)
        for u,v in md.adj_edges:
            adj[u].add(v); adj[v].add(u)
        zones = defaultdict(set) # zone -> elements
        for e,z in md.zone_edges:
            zones[z].add(e)

        # M2_frameNode: Beam-Column adjacency (unordered)
        m2_frame = 0
        seen_fc = set()
        for u,v in md.adj_edges:
            tu = n2t.get(u); tv = n2t.get(v)
            if not tu or not tv: continue
            pair = tuple(sorted([str(u),str(v)]))
            if pair in seen_fc: continue
            if (tu=="Beam" and tv=="Column") or (tu=="Column" and tv=="Beam"):
                m2_frame += 1; seen_fc.add(pair)

        # M2_braceNode: Brace adjacent to Frame member (Beam/Column)
        m2_brace = 0
        seen_b = set()
        for u,v in md.adj_edges:
            tu = n2t.get(u); tv = n2t.get(v)
            if not tu or not tv: continue
            pair = tuple(sorted([str(u),str(v)]))
            if pair in seen_b: continue
            if (tu=="Brace" and tv in {"Beam","Column"}) or (tv=="Brace" and tu in {"Beam","Column"}):
                m2_brace += 1; seen_b.add(pair)

        # M3_wallSlab: Wall–Slab intersecting
        m3_ws = 0
        seen_ws = set()
        for u,v in md.int_edges:
            tu = n2t.get(u); tv = n2t.get(v)
            if not tu or not tv: continue
            pair = tuple(sorted([str(u),str(v)]))
            if pair in seen_ws: continue
            if {"Wall","Slab"} == {tu,tv}:
                m3_ws += 1; seen_ws.add(pair)

        # M3_wallSlabAdj: Wall–Slab adjacency
        m3_ws_adj = 0
        seen_wsa = set()
        for u,v in md.adj_edges:
            tu = n2t.get(u); tv = n2t.get(v)
            if not tu or not tv: continue
            pair = tuple(sorted([str(u),str(v)]))
            if pair in seen_wsa: continue
            if {"Wall","Slab"} == {tu,tv}:
                m3_ws_adj += 1; seen_wsa.add(pair)

        # M3_wallSlabZone: Wall and Slab touching same zone (proxy)
        m3_ws_zone = 0
        for z, elems in zones.items():
            walls = [e for e in elems if node2type[m].get(e)=="Wall"]
            slabs = [e for e in elems if node2type[m].get(e)=="Slab"]
            m3_ws_zone += len(walls) * len(slabs)

        # M4_core: Core with slab adjacency
        m4_core = 0
        seen_core = set()
        for u,v in md.adj_edges:
            tu = n2t.get(u); tv = n2t.get(v)
            if not tu or not tv: continue
            if (tu=="Core" and tv=="Slab") or (tv=="Core" and tu=="Slab"):
                if tu=="Core":
                    core = u
                elif tv=="Core":
                    core = v
                else:
                    core = None
                if core and core not in seen_core:
                    seen_core.add(core); m4_core += 1

        # M5_structRole
        m5 = 0
        for (e,f) in md.func_edges:
            if e not in n2t: continue
            clocs = " ".join([local_name(c) for c in md.func_types.get(f, [])]).lower()
            if re.search(FUNC_REGEX["LoadBearing"], clocs, re.I) or \
               re.search(FUNC_REGEX["Shear"], clocs, re.I) or \
               re.search(FUNC_REGEX["Moment"], clocs, re.I):
                m5 += 1

        counts = {
            "M2_frameNode": m2_frame,
            "M2_braceNode": m2_brace,
            "M3_wallSlab": m3_ws,
            "M3_wallSlabAdj": m3_ws_adj,
            "M3_wallSlabZone": m3_ws_zone,
            "M4_core": m4_core,
            "M5_structRole": m5
        }

        total_cnt = sum(counts.values())
        structN = max(len(node2type[m].keys()), 1)

        rows_counts.append({"model": m, **counts})
        rows_share.append({"model": m, **({k:(counts[k]/total_cnt if total_cnt>0 else 0.0) for k in counts})})
        rows_density.append({"model": m, **({k:(100.0*counts[k]/structN if structN>0 else 0.0) for k in counts})})

    df_counts = pd.DataFrame(rows_counts).sort_values("model")
    df_share  = pd.DataFrame(rows_share).sort_values("model")
    df_density = pd.DataFrame(rows_density).sort_values("model")
    return df_counts, df_share, df_density


def stage_S3(models, motif_counts, func_shares_wide, dual_thresh=0.25):
    """
    Simple rule-based system scores (frame/wall/dual/braced).
    Uses motif counts + function shares.
    """
    rows = []
    comp_rows = []
    for md in models:
        m = md.name
        mc = motif_counts[motif_counts["model"]==m].iloc[0].to_dict()
        frame_signal = mc["M2_frameNode"] + 0.5*mc["M2_braceNode"]
        wall_signal = mc["M3_wallSlab"] + mc["M3_wallSlabAdj"] + 0.5*mc["M3_wallSlabZone"]
        braced_signal = mc["M2_braceNode"]

        fs = func_shares_wide[func_shares_wide["model"]==m]
        lb = float(fs["LoadBearing"].iloc[0]) if len(fs)>0 else 0.0
        sh = float(fs["Shear"].iloc[0]) if len(fs)>0 else 0.0
        mo = float(fs["Moment"].iloc[0]) if len(fs)>0 else 0.0

        def scale(x):
            return math.tanh(math.log1p(x))  # [0,~0.76]

        frame_base = 0.6*scale(frame_signal) + 0.4*mo
        wall_base  = 0.6*scale(wall_signal)  + 0.4*(lb+sh)/2.0
        braced_base= 0.7*scale(braced_signal) + 0.3*lb

        total_fw = frame_base + wall_base
        frame_share = frame_base/(total_fw or 1.0)
        dual = (frame_base>0.05 and wall_base>0.05 and frame_share>=dual_thresh)
        dual_score = 1.0 if dual else 0.0

        rows.append({
            "model": m,
            "frame": frame_base,
            "wall": wall_base,
            "dual": dual_score,
            "braced": braced_base
        })
        comp_rows.append({
            "model": m,
            "frame_signal": frame_signal,
            "wall_signal": wall_signal,
            "braced_signal": braced_signal,
            "func_LoadBearing": lb, "func_Shear": sh, "func_Moment": mo,
            "frame_share": frame_share
        })
    df_scores = pd.DataFrame(rows).sort_values("model")
    df_comp = pd.DataFrame(comp_rows).sort_values("model")
    return df_scores, df_comp


def stage_S4(models, motif_shares, system_scores, w_motif=0.5, w_system=0.5):
    """
    Pairwise structural similarity:
    - S_motif = cosine(motif share vectors)
    - S_system = cosine([frame,wall,dual,braced])
    - S_struct_total = w_motif*S_motif + w_system*S_system
    """
    models_list = [md.name for md in models]
    M = motif_shares.set_index("model").loc[models_list]
    S = system_scores.set_index("model").loc[models_list][["frame","wall","dual","braced"]]

    mtx_motif = np.zeros((len(models_list),len(models_list)))
    mtx_system= np.zeros_like(mtx_motif)
    mtx_total = np.zeros_like(mtx_motif)

    for i,a in enumerate(models_list):
        va = M.loc[a].values.astype(float)
        sa = S.loc[a].values.astype(float)
        for j,b in enumerate(models_list):
            vb = M.loc[b].values.astype(float)
            sb = S.loc[b].values.astype(float)
            sm = cosine_sim(va, vb)
            ss = cosine_sim(sa, sb)
            st = w_motif*sm + w_system*ss
            mtx_motif[i,j] = sm
            mtx_system[i,j]= ss
            mtx_total[i,j] = st

    df_total = pd.DataFrame(mtx_total, index=models_list, columns=models_list).reset_index().rename(columns={"index":"model"})
    rows = []
    for i in range(len(models_list)):
        for j in range(i+1, len(models_list)):
            a,b = models_list[i], models_list[j]
            rows.append({
                "A": a, "B": b,
                "S_motif": mtx_motif[i,j],
                "S_system": mtx_system[i,j],
                "S_struct_total": mtx_total[i,j]
            })
    df_pairs = pd.DataFrame(rows).sort_values("S_struct_total", ascending=False)
    return df_total, df_pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--pattern", default="*_DG.rdf")
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--out-name", required=True)
    ap.add_argument("--func-all", action="store_true", help="Count function shares without requiring element gating")
    ap.add_argument("--dual-thresh", type=float, default=0.25)
    ap.add_argument("--emit-debug", action="store_true")
    args = ap.parse_args()

    in_dir = args.input_dir
    files = sorted(glob.glob(os.path.join(in_dir, args.pattern)))
    print("[LOAD]")
    for f in files:
        print(f"       {os.path.basename(f)}")
    models = []
    for f in files:
        g = Graph()
        g.parse(f, format="xml")
        md = ModelData(os.path.basename(f), g)
        md.build()
        models.append(md)

    out_dir = os.path.join(args.out_root, args.out_name)
    os.makedirs(out_dir, exist_ok=True)

    # read overrides if present
    overrides = {}
    ov_path = os.path.join(os.getcwd(), "struct_type_overrides.json")
    if os.path.exists(ov_path):
        try:
            overrides = json.load(open(ov_path, "r", encoding="utf-8"))
        except:
            overrides = {}

    # S1
    df_types, df_funcs, df_func_wide, df_av, node2type = stage_S1(models, overrides, func_all=args.func_all, emit_debug=args.emit_debug, out_dir=out_dir)

    # S2
    df_counts, df_share, df_density = stage_S2(models, node2type, df_func_wide)

    # S3
    df_scores, df_comp = stage_S3(models, df_counts, df_func_wide, dual_thresh=args.dual_thresh)

    # S4
    df_struct_total, df_pairs = stage_S4(models, df_share, df_scores, w_motif=0.5, w_system=0.5)

    # Save outputs
    df_types.to_csv(os.path.join(out_dir, "struct_types_histogram.csv"), index=False)
    df_funcs.to_csv(os.path.join(out_dir, "struct_functions_histogram.csv"), index=False)
    df_func_wide.to_csv(os.path.join(out_dir, "struct_functions_shares_wide.csv"), index=False)

    df_counts.to_csv(os.path.join(out_dir, "struct_motif_counts.csv"), index=False)
    df_share.to_csv(os.path.join(out_dir, "struct_motif_shares.csv"), index=False)
    df_density.to_csv(os.path.join(out_dir, "struct_motif_densities_per100.csv"), index=False)

    df_scores.to_csv(os.path.join(out_dir, "struct_system_scores.csv"), index=False)
    df_struct_total.to_csv(os.path.join(out_dir, "struct_similarity_matrix.csv"), index=False)
    df_pairs.to_csv(os.path.join(out_dir, "pairwise_structural_summary.csv"), index=False)
    df_comp.to_csv(os.path.join(out_dir, "struct_score_components.csv"), index=False)
    df_av.to_csv(os.path.join(out_dir, "struct_data_availability.csv"), index=False)

    print(f"\n[OK] Saved outputs under: {out_dir}")
    print(" - struct_types_histogram.csv")
    print(" - struct_functions_histogram.csv")
    print(" - struct_functions_shares_wide.csv")
    print(" - struct_motif_counts.csv")
    print(" - struct_motif_shares.csv")
    print(" - struct_motif_densities_per100.csv")
    print(" - struct_system_scores.csv")
    print(" - struct_similarity_matrix.csv")
    print(" - pairwise_structural_summary.csv")
    print(" - struct_score_components.csv")
    print(" - struct_data_availability.csv")
    if args.emit_debug:
        print(" - type_mapping_hits.csv")
        print(" - type_mapping_unknown.csv")

if __name__ == "__main__":
    main()
