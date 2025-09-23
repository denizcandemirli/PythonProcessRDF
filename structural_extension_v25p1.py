# -*- coding: utf-8 -*-
"""
Structural Extension v25p.1
S1: Type / Function inventories
S2: Structural motifs (strong/weak topology + type-only proxy w/ penalties)
S3: System scores (frame/wall/dual/braced) from motif + function signals
S4: Structural similarity = w_motif*cos(M) + w_system*cos(S)

Improvements vs v25p:
- Expanded FUNCTION LEXICON for Moment/Bracing (AEC variants).
- Robust S1 pivot -> always has ['model','LB','Shear','Moment','Bracing'] (zeros if absent).
- Proxy/weak-topology detailed accounting -> motif_proxy_summary.csv
- Weights/params fully recorded to weights_used.json

Req: pandas, numpy, rdflib, regex (re)
"""

import os, json, argparse, re, math
from collections import defaultdict, Counter
import numpy as np
import pandas as pd
from rdflib import Graph, RDF, RDFS, URIRef, Literal

# -----------------------
# Helpers
# -----------------------

def localname(uri):
    s = str(uri)
    for sep in ['#','/']:
        if sep in s:
            s = s.split(sep)[-1]
    return s

def norm_token(s):
    return re.sub(r'[^A-Za-z0-9]+','', s or '').lower()

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def cosine(u, v, eps=1e-12):
    u = np.asarray(u, float)
    v = np.asarray(v, float)
    nu = np.linalg.norm(u)
    nv = np.linalg.norm(v)
    if nu < eps or nv < eps:
        return 0.0
    return float(np.dot(u, v) / (nu * nv))

# -----------------------
# S0: Model loader
# -----------------------

STRONG_TOPO_PRED = {
    'adjacentelement', 'intersectingelement'
}
WEAK_TOPO_PRED = {
    'touches', 'meets', 'hasspatialrelation', 'intersects', 'nearby'
}

TYPE_MAP_CANON = {
    # core structural
    'ifcwall': 'Wall', 'wall': 'Wall', 'shearwall': 'Wall', 'corewall': 'Wall', 'retainingwall': 'Wall',
    'ifcslab': 'Slab', 'slab': 'Slab', 'deck': 'Slab', 'floor': 'Slab', 'plate': 'Slab',
    'ifccolumn': 'Column', 'column': 'Column', 'pier': 'Column',
    'ifcbeam': 'Beam', 'beam': 'Beam', 'girder': 'Beam', 'joist': 'Beam',
    'ifcbracing': 'Brace', 'brace': 'Brace', 'bracing': 'Brace', 'tie': 'Brace', 'strut': 'Brace',
    'ifcstructuralcurvemember': 'Brace',  # often bracing is modeled under structural curve member
    'core': 'Core',

    # foundation family (for S1 reporting; not used directly in S3)
    'foundation': 'Foundation', 'footing': 'Foundation', 'pilecap': 'Foundation', 'ifcfoundation': 'Foundation'
}

# FUNCTION LEXICON (expanded)
FUNC_MAP = {
    'loadbearing': 'LB', 'bearing': 'LB', 'gravity': 'LB', 'axial': 'LB',
    'shear': 'Shear', 'lateral': 'Shear',
    'moment': 'Moment', 'bending': 'Moment', 'flexural': 'Moment',
    'momentframe': 'Moment', 'rigidframe': 'Moment', 'couplingbeam': 'Moment',
    'diaphragm': 'Moment', 'stiffener': 'Moment',
    'bracing': 'Bracing', 'brace': 'Bracing',
    'bucklingrestrainedbrace': 'Bracing', 'brb': 'Bracing', 'tie': 'Bracing', 'strut': 'Bracing'
}

FUNC_FIELDS = {'LB','Shear','Moment','Bracing'}

class Model:
    def __init__(self, name, g):
        self.name = name
        self.g = g
        self.types = defaultdict(set)      # node -> set of canonical types
        self.labels = {}                   # node -> label string
        self.func = defaultdict(set)       # node -> set of canonical functions
        self.edges_strong = set()          # (u,v) strong topo
        self.edges_weak = set()            # (u,v) weak topo
        self.nodes = set()

def load_models(input_dir, pattern):
    files = [f for f in os.listdir(input_dir) if re.fullmatch(pattern.replace('*','.*'), f)]
    files.sort()
    models = []
    print("[LOAD]")
    for f in files:
        print("  ", f)
        g = Graph()
        g.parse(os.path.join(input_dir, f))
        m = Model(f, g)
        # labels
        for s, p, o in g.triples((None, RDFS.label, None)):
            if isinstance(o, Literal):
                m.labels[s] = str(o)
        # types
        for s, p, o in g.triples((None, RDF.type, None)):
            ln = norm_token(localname(o))
            canon = TYPE_MAP_CANON.get(ln, None)
            if canon:
                m.types[s].add(canon)
            else:
                # also keep raw localname for later unknown mapping report
                m.types[s].add(localname(o))
            m.nodes.add(s)
        # functions: via hasFunction / hasQuality / role-like properties
        for s, p, o in g:
            pn = norm_token(localname(p))
            if 'hasfunction' in pn or 'hasquality' in pn or pn in {'function','role','structuralfunction'}:
                val = norm_token(localname(o) if isinstance(o, URIRef) else str(o))
                for key, canon in FUNC_MAP.items():
                    if key in val:
                        m.func[s].add(canon)
        # topo edges
        for s, p, o in g:
            pn = norm_token(localname(p))
            if pn in STRONG_TOPO_PRED and isinstance(o, URIRef):
                m.edges_strong.add((s,o))
            elif pn in WEAK_TOPO_PRED and isinstance(o, URIRef):
                m.edges_weak.add((s,o))
        models.append(m)
    return models

def is_type(node_types, wanted):
    return any(t == wanted for t in node_types if t in TYPE_MAP_CANON.values() or t in {'Foundation'})

# -----------------------
# S1: Inventories
# -----------------------

def s1_inventories(models, func_all=False):
    rows_types = []
    rows_funcs = []
    unknown_hits = Counter()
    type_hits = Counter()

    # Scan
    for m in models:
        # type histogram (canonical & unknown)
        for n, ts in m.types.items():
            # record canonical hits + unknowns
            mapped = False
            raw_norms = set(norm_token(t) for t in ts)
            for rn in raw_norms:
                if rn in TYPE_MAP_CANON:
                    type_hits[TYPE_MAP_CANON[rn]] += 1
                    rows_types.append({'model': m.name, 'type': TYPE_MAP_CANON[rn], 'count': 1})
                    mapped = True
            if not mapped:
                # collect unknown
                for t in ts:
                    unknown_hits[(m.name, t)] += 1

        # function histogram
        for n in m.nodes:
            fset = m.func.get(n, set())
            # func_all -> LB varsayılanı yok; sadece açık eşleşenleri say
            for f in fset:
                if f in FUNC_FIELDS:
                    rows_funcs.append({'model': m.name, 'function': f, 'count': 1})

    # Aggregate
    df_types = (pd.DataFrame(rows_types)
                  .groupby(['model','type'], as_index=False)['count'].sum()
                  if rows_types else pd.DataFrame(columns=['model','type','count']))

    df_funcs = (pd.DataFrame(rows_funcs)
                  .groupby(['model','function'], as_index=False)['count'].sum()
                  if rows_funcs else pd.DataFrame(columns=['model','function','count']))

    # Wide shares (robust)
    if df_funcs.empty:
        piv = pd.DataFrame({'model':[m.name for m in models]})
    else:
        piv = (df_funcs.pivot(index='model', columns='function', values='count')
                      .fillna(0.0)
                      .astype(float))
        piv['total'] = piv.sum(axis=1)
        for col in ['LB','Shear','Moment','Bracing']:
            if col not in piv.columns:
                piv[col] = 0.0
        piv['sum_known'] = piv[['LB','Shear','Moment','Bracing']].sum(axis=1)
        # shares among known funcs (avoid division by zero)
        for col in ['LB','Shear','Moment','Bracing']:
            piv[col] = np.where(piv['sum_known']>0, piv[col]/piv['sum_known'], 0.0)
        piv = piv[['LB','Shear','Moment','Bracing']]
        piv = piv.reset_index()

    # Availability quick flags
    rows_av = []
    for m in models:
        rows_av.append({
            'model': m.name,
            'has_strong_topology': int(len(m.edges_strong) > 0),
            'has_weak_topology': int(len(m.edges_weak) > 0),
            'has_function_labels': int(len(m.func) > 0)
        })
    df_av = pd.DataFrame(rows_av)

    # Unknown & hits tables
    rows_hits = [{'model':m, 'type':t, 'hits':c} for (m,t), c in unknown_hits.items()]
    df_unknown = pd.DataFrame(rows_hits).sort_values(['model','type']) if rows_hits else pd.DataFrame(columns=['model','type','hits'])

    rows_map_hits = [{'type':k, 'hits':v} for k,v in type_hits.items()]
    df_map_hits = pd.DataFrame(rows_map_hits).sort_values(['type']) if rows_map_hits else pd.DataFrame(columns=['type','hits'])

    return df_types, df_funcs, piv, df_av, df_map_hits, df_unknown

# -----------------------
# S2: Motifs
# -----------------------

def s2_motifs(models, proxy_penalty=0.7, weak_topo_penalty=0.5, allow_type_only=True):
    """
    Count motifs per model with strong/weak/proxy breakdown.
    Apply penalties to weak/proxy counts when converting to effective counts.
    """
    motif_names = ['M2_frameNode','M2b_braceNode','M3_wallSlab','M4_core','M5_LB','M5_Shear','M5_Moment']
    rows_counts = []
    proxy_rows = []

    for m in models:
        # Pre-index nodes by type
        T = defaultdict(list)
        for n, ts in m.types.items():
            if 'Wall' in ts: T['Wall'].append(n)
            if 'Slab' in ts: T['Slab'].append(n)
            if 'Column' in ts: T['Column'].append(n)
            if 'Beam' in ts: T['Beam'].append(n)
            if 'Brace' in ts: T['Brace'].append(n)
            if 'Core' in ts: T['Core'].append(n)

        strong = set(m.edges_strong)
        weak = set(m.edges_weak)

        # helper to count pair motif with topo/proxy split
        def count_pair(A, B):
            s=w=p=0
            Aset, Bset = set(A), set(B)
            if Aset and Bset:
                # strong edges among A-B
                for u,v in strong:
                    if u in Aset and v in Bset or u in Bset and v in Aset:
                        s += 1
                # weak edges
                for u,v in weak:
                    if u in Aset and v in Bset or u in Bset and v in Aset:
                        w += 1
                # proxy if none
                if s==0 and w==0 and allow_type_only:
                    p = min(len(Aset), len(Bset))
            return s,w,p

        # M2: frame node (Beam-Column)
        s2,w2,p2 = count_pair(T['Beam'], T['Column'])
        # M2b: brace-node (Brace to Beam/Column)
        s2b1,w2b1,p2b1 = count_pair(T['Brace'], T['Beam'])
        s2b2,w2b2,p2b2 = count_pair(T['Brace'], T['Column'])
        s2b,w2b,p2b = s2b1+s2b2, w2b1+w2b2, p2b1+p2b2
        # M3: wall-slab
        s3,w3,p3 = count_pair(T['Wall'], T['Slab'])
        # M4: core surrounded by slabs (approx: any core with any slab adj/weak; else proxy by type)
        s4=w4=p4=0
        if T['Core'] and T['Slab']:
            # if any strong/weak edge between any core and any slab
            has_s = any(((c,s) in strong or (s,c) in strong) for c in T['Core'] for s in T['Slab'])
            has_w = any(((c,s) in weak or (s,c) in weak) for c in T['Core'] for s in T['Slab'])
            s4 = int(has_s)
            w4 = int((not has_s) and has_w)
            if s4==0 and w4==0 and allow_type_only:
                p4 = min(len(T['Core']), len(T['Slab']))

        # M5: function-based counts (by fraction of structural nodes)
        nodes_all = [n for n,ts in m.types.items() if any(t in {'Wall','Slab','Column','Beam','Brace','Core'} for t in ts)]
        n_all = max(1, len(nodes_all))
        fLB = sum(1 for n in nodes_all if 'LB' in m.func.get(n,set()))
        fSh = sum(1 for n in nodes_all if 'Shear' in m.func.get(n,set()))
        fMo = sum(1 for n in nodes_all if 'Moment' in m.func.get(n,set()))
        # raw counts (we'll convert to shares later)
        c5LB, c5Sh, c5Mo = fLB, fSh, fMo

        # proxy summary rows (raw)
        proxy_rows += [
            {'model': m.name, 'motif': 'M2_frameNode', 'strong': s2, 'weak': w2, 'proxy': p2},
            {'model': m.name, 'motif': 'M2b_braceNode', 'strong': s2b, 'weak': w2b, 'proxy': p2b},
            {'model': m.name, 'motif': 'M3_wallSlab', 'strong': s3, 'weak': w3, 'proxy': p3},
            {'model': m.name, 'motif': 'M4_core', 'strong': s4, 'weak': w4, 'proxy': p4},
        ]

        # penalized effective counts
        eff_M2  = s2  + weak_topo_penalty*w2  + proxy_penalty*p2
        eff_M2b = s2b + weak_topo_penalty*w2b + proxy_penalty*p2b
        eff_M3  = s3  + weak_topo_penalty*w3  + proxy_penalty*p3
        eff_M4  = s4  + weak_topo_penalty*w4  + proxy_penalty*p4

        rows_counts += [
            {'model':m.name,'motif':'M2_frameNode','count':eff_M2},
            {'model':m.name,'motif':'M2b_braceNode','count':eff_M2b},
            {'model':m.name,'motif':'M3_wallSlab','count':eff_M3},
            {'model':m.name,'motif':'M4_core','count':eff_M4},
            {'model':m.name,'motif':'M5_LB','count':c5LB},
            {'model':m.name,'motif':'M5_Shear','count':c5Sh},
            {'model':m.name,'motif':'M5_Moment','count':c5Mo},
        ]

    df_counts = pd.DataFrame(rows_counts)
    if df_counts.empty:
        df_counts = pd.DataFrame(columns=['model','motif','count'])

    # shares (row-normalized motif mix per model)
    if df_counts.empty:
        df_shares = pd.DataFrame(columns=['model']+[ 'M2_frameNode','M2b_braceNode','M3_wallSlab','M4_core','M5_LB','M5_Shear','M5_Moment' ])
        df_dens = pd.DataFrame(columns=['model']+[ 'M2_frameNode','M2b_braceNode','M3_wallSlab','M4_core','M5_LB','M5_Shear','M5_Moment' ])
    else:
        piv = df_counts.pivot(index='model', columns='motif', values='count').fillna(0.0)
        # fill missing motif cols with zeros (stable schema)
        for mname in ['M2_frameNode','M2b_braceNode','M3_wallSlab','M4_core','M5_LB','M5_Shear','M5_Moment']:
            if mname not in piv.columns: piv[mname]=0.0
        row_sum = piv.sum(axis=1).replace(0, np.nan)
        shares = piv.div(row_sum, axis=0).fillna(0.0).reset_index()
        df_shares = shares.copy()

        # per-100 density -> use structural node count as base
        model2N = {}
        for m in models:
            N = sum(1 for n,ts in m.types.items() if any(t in {'Wall','Slab','Column','Beam','Brace','Core'} for t in ts))
            model2N[m.name] = max(1, N)
        dens = piv.copy()
        for idx in dens.index:
            dens.loc[idx,:] = 100.0 * dens.loc[idx,:] / model2N.get(idx,1)
        df_dens = dens.reset_index()

    df_proxy = pd.DataFrame(proxy_rows) if proxy_rows else pd.DataFrame(columns=['model','motif','strong','weak','proxy'])
    return df_counts, df_shares, df_dens, df_proxy

# -----------------------
# S3: System scores
# -----------------------

def s3_system_scores(motif_shares, alpha_m5=0.40, dual_thresh=0.25):
    # motif columns guaranteed by s2_motifs
    required = ['M2_frameNode','M3_wallSlab','M2b_braceNode','M5_LB','M5_Shear','M5_Moment']
    for c in required:
        if c not in motif_shares.columns:
            motif_shares[c] = 0.0
    rows = []
    rows_comp = []
    for _, r in motif_shares.iterrows():
        model = r['model']
        M2, M3, M2b = float(r['M2_frameNode']), float(r['M3_wallSlab']), float(r['M2b_braceNode'])
        M5LB, M5Sh, M5Mo = float(r['M5_LB']), float(r['M5_Shear']), float(r['M5_Moment'])

        frame = M2 + alpha_m5 * M5Mo
        wall  = M3 + (0.7*M5Sh + 0.3*M5LB)
        braced = M2b
        dual = 0.0
        if min(frame, wall) >= dual_thresh:
            dual = 0.5*(frame + wall)

        rows.append({'model':model,'frame':frame,'wall':wall,'dual':dual,'braced':braced})
        rows_comp.append({'model':model,'M2':M2,'M3':M3,'M2b':M2b,'M5LB':M5LB,'M5Sh':M5Sh,'M5Mo':M5Mo,'alpha_m5':alpha_m5})
    return pd.DataFrame(rows), pd.DataFrame(rows_comp)

# -----------------------
# S4: Structural similarity
# -----------------------

def s4_similarity(motif_shares, system_scores, w_motif=0.5, w_system=0.5):
    motif_cols = ['M2_frameNode','M2b_braceNode','M3_wallSlab','M4_core','M5_LB','M5_Shear','M5_Moment']
    sys_cols   = ['frame','wall','dual','braced']

    ms = motif_shares.set_index('model')[motif_cols].reindex(motif_shares['model']).fillna(0.0).values
    sv = system_scores.set_index('model')[sys_cols].reindex(system_scores['model']).fillna(0.0).values
    labels = list(motif_shares['model'])

    n = len(labels)
    S_motif = np.zeros((n,n))
    S_sys   = np.zeros((n,n))
    S_final = np.zeros((n,n))

    for i in range(n):
        for j in range(n):
            S_motif[i,j] = cosine(ms[i], ms[j])
            S_sys[i,j]   = cosine(sv[i], sv[j])
            S_final[i,j] = w_motif*S_motif[i,j] + w_system*S_sys[i,j]

    df_final = pd.DataFrame(S_final, index=labels, columns=labels).reset_index().rename(columns={'index':'model'})
    # pairwise summary
    pairs = []
    for i in range(n):
        for j in range(i+1, n):
            pairs.append({
                'model_a': labels[i],
                'model_b': labels[j],
                'sim_motif': S_motif[i,j],
                'sim_system': S_sys[i,j],
                'sim_struct_final': S_final[i,j]
            })
    df_pairs = pd.DataFrame(pairs).sort_values('sim_struct_final', ascending=False)
    return df_final, df_pairs

# -----------------------
# Main
# -----------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input-dir', required=True)
    ap.add_argument('--pattern', required=True)
    ap.add_argument('--out-root', required=True)
    ap.add_argument('--out-name', required=True)
    ap.add_argument('--dual-thresh', type=float, default=0.25)
    ap.add_argument('--w-motif', type=float, default=0.5)
    ap.add_argument('--w-system', type=float, default=0.5)
    ap.add_argument('--alpha-m5', type=float, default=0.40)
    ap.add_argument('--proxy-penalty', type=float, default=0.70)
    ap.add_argument('--weak-topo-penalty', type=float, default=0.50)
    ap.add_argument('--allow-type-only-proxy', action='store_true')
    ap.add_argument('--func-all', action='store_true')
    ap.add_argument('--emit-debug', action='store_true')
    args = ap.parse_args()

    out_dir = os.path.join(args.out_root, args.out_name)
    ensure_dir(out_dir)

    # Load models
    models = load_models(args.input_dir, args.pattern)

    # S1
    df_types, df_funcs, df_func_wide, df_av, df_map_hits, df_unknown = s1_inventories(models, func_all=args.func_all)
    df_types.to_csv(os.path.join(out_dir, 'struct_types_histogram.csv'), index=False)
    df_funcs.to_csv(os.path.join(out_dir, 'struct_functions_histogram.csv'), index=False)
    df_func_wide.to_csv(os.path.join(out_dir, 'struct_functions_shares_wide.csv'), index=False)
    df_av.to_csv(os.path.join(out_dir, 'struct_data_availability.csv'), index=False)
    df_map_hits.to_csv(os.path.join(out_dir, 'type_mapping_hits.csv'), index=False)
    df_unknown.to_csv(os.path.join(out_dir, 'type_mapping_unknown.csv'), index=False)

    # S2
    counts, shares, dens, proxy = s2_motifs(
        models,
        proxy_penalty=args.proxy_penalty,
        weak_topo_penalty=args.weak_topo_penalty,
        allow_type_only=args.allow_type_only_proxy
    )
    counts.to_csv(os.path.join(out_dir, 'struct_motif_counts.csv'), index=False)
    shares.to_csv(os.path.join(out_dir, 'struct_motif_shares.csv'), index=False)
    dens.to_csv(os.path.join(out_dir, 'struct_motif_densities_per100.csv'), index=False)
    proxy.to_csv(os.path.join(out_dir, 'motif_proxy_summary.csv'), index=False)

    # S3
    sys_scores, sys_comp = s3_system_scores(shares, alpha_m5=args.alpha_m5, dual_thresh=args.dual_thresh)
    sys_scores.to_csv(os.path.join(out_dir, 'struct_system_scores.csv'), index=False)
    sys_comp.to_csv(os.path.join(out_dir, 'struct_score_components.csv'), index=False)

    # S4
    simM, pairs = s4_similarity(shares, sys_scores, w_motif=args.w_motif, w_system=args.w_system)
    simM.to_csv(os.path.join(out_dir, 'struct_similarity_matrix.csv'), index=False)
    pairs.to_csv(os.path.join(out_dir, 'pairwise_structural_summary.csv'), index=False)

    # Weights / params
    weights = {
        'dual_thresh': args.dual_thresh,
        'w_motif': args.w_motif,
        'w_system': args.w_system,
        'alpha_m5': args.alpha_m5,
        'proxy_penalty': args.proxy_penalty,
        'weak_topo_penalty': args.weak_topo_penalty,
        'allow_type_only_proxy': bool(args.allow_type_only_proxy),
        'func_all': bool(args.func_all)
    }
    with open(os.path.join(out_dir, 'weights_used.json'), 'w', encoding='utf-8') as f:
        json.dump(weights, f, indent=2)

    print("\n[OK] Saved outputs under:", out_dir)
    for f in [
        'struct_types_histogram.csv',
        'struct_functions_histogram.csv',
        'struct_functions_shares_wide.csv',
        'struct_data_availability.csv',
        'struct_motif_counts.csv',
        'struct_motif_shares.csv',
        'struct_motif_densities_per100.csv',
        'struct_system_scores.csv',
        'struct_score_components.csv',
        'struct_similarity_matrix.csv',
        'pairwise_structural_summary.csv',
        'motif_proxy_summary.csv',
        'type_mapping_hits.csv',
        'type_mapping_unknown.csv',
        'weights_used.json'
    ]:
        print(" -", f)

if __name__ == '__main__':
    main()
