#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Structural Extension v25p
- S1: type/function inventory (IFC & regex map; PartitionWall -> Wall only if LB/Shear)
- S2: structural motifs (M2 frameNode, M2b braceNode, M3 wallSlab, M4 core, M5 LB/Shear/Moment)
      * weak-topology support: touches/meets/hasSpatialRelation/intersects (penalized)
      * type-only proxy fallback (penalized)
      * motif_proxy_summary.csv (direct / weak / proxy katkı)
- S3: system scores (frame / wall / dual / braced) + components dump
- S4: structural similarity (motif cosine ⊕ system cosine), weights saved to JSON
Outputs (WIDE):
  struct_types_histogram.csv
  struct_functions_histogram.csv
  struct_functions_shares_wide.csv
  struct_data_availability.csv
  struct_motif_counts.csv
  struct_motif_shares.csv
  struct_motif_densities_per100.csv
  struct_system_scores.csv
  struct_score_components.csv
  struct_similarity_matrix.csv
  pairwise_structural_summary.csv
  motif_proxy_summary.csv
  type_mapping_hits.csv
  type_mapping_unknown.csv
  weights_used.json
"""
# -*- coding: utf-8 -*-
"""
Structural Extension v25p (hardened)
"""
import os, re, json, argparse
from collections import defaultdict, Counter
import pandas as pd
import numpy as np

def ensure_dir(p): os.makedirs(p, exist_ok=True)
def localname(x:str) -> str:
    x = str(x)
    if '#' in x: return x.rsplit('#',1)[-1]
    if '/' in x: return x.rstrip('/').rsplit('/',1)[-1]
    return x

def cosine(a,b):
    na=np.linalg.norm(a); nb=np.linalg.norm(b)
    if na==0 or nb==0: return 0.0
    return float(np.dot(a,b)/(na*nb))

def parse_rdf(path: str):
    triples=[]
    try:
        import rdflib
        g=rdflib.Graph(); g.parse(path)
        for s,p,o in g: triples.append((str(s),str(p),str(o)))
    except Exception:
        with open(path,'r',encoding='utf-8',errors='ignore') as f:
            for ln in f:
                if 'http' not in ln: continue
                parts = re.findall(r'<([^>]*)>', ln)
                if len(parts)>=3: triples.append((parts[0],parts[1],parts[2]))
    return triples

class Model:
    def __init__(self, path:str):
        self.path = path
        self.name = os.path.basename(path)
        self.triples = parse_rdf(path)
        self.types = defaultdict(set)
        self.funcs = defaultdict(set)
        self.labels= defaultdict(set)
        self.edges_adj=set(); self.edges_int=set(); self.edges_weak=set()
        self._index()

    def _index(self):
        P_TYPE={'rdf:type','http://www.w3.org/1999/02/22-rdf-syntax-ns#type'}
        P_LABEL={'rdfs:label','http://www.w3.org/2000/01/rdf-schema#label','label','name'}
        P_FUNC={'hasFunction','http://example.org/hasFunction','hasQuality','http://example.org/hasQuality'}
        P_ADJ ={'adjacentElement','http://example.org/adjacentElement','adjacentZone','http://example.org/adjacentZone'}
        P_INT ={'intersectingElement','http://example.org/intersectingElement'}
        P_WEAK={'touches','meets','hasSpatialRelation','intersects',
                'http://example.org/touches','http://example.org/meets',
                'http://example.org/hasSpatialRelation','http://example.org/intersects'}
        for s,p,o in self.triples:
            ls,lp,lo = localname(s), localname(p), localname(o)
            if lp in P_TYPE:
                self.types[ls].add(lo); self.labels[ls].add(lo)
            elif lp in P_LABEL:
                self.labels[ls].add(lo)
            elif lp in P_FUNC:
                tok = lo.lower()
                if 'load' in tok or 'lb' in tok: self.funcs[ls].add('LB')
                if 'shear' in tok: self.funcs[ls].add('Shear')
                if 'moment' in tok or 'rigid' in tok: self.funcs[ls].add('Moment')
                if 'brace' in tok or 'bracing' in tok or 'diagonal' in tok: self.funcs[ls].add('Bracing')
            elif lp in P_ADJ:
                self.edges_adj.add((ls,localname(o)))
            elif lp in P_INT:
                self.edges_int.add((ls,localname(o)))
            elif lp in P_WEAK:
                self.edges_weak.add((ls,localname(o)))
        # undirect
        self.edges_adj |= {(v,u) for (u,v) in list(self.edges_adj)}
        self.edges_int |= {(v,u) for (u,v) in list(self.edges_int)}
        self.edges_weak |= {(v,u) for (u,v) in list(self.edges_weak)}

# ---------- S1
def build_type_mapper():
    WALL   = re.compile(r'(?:^|_)(Wall|Shear[- ]?Wall|CoreWall|ExteriorWall|InteriorWall|PartitionWall|IfcWall|IfcWallStandardCase)$', re.IGNORECASE)
    SLAB   = re.compile(r'(?:^|_)(Slab|Deck|Floor|Plate|IfcSlab|IfcFloor|IfcPlate)$', re.IGNORECASE)
    COLUMN = re.compile(r'(?:^|_)(Column|IfcColumn)$', re.IGNORECASE)
    BEAM   = re.compile(r'(?:^|_)(Beam|IfcBeam|IfcMember)$', re.IGNORECASE)
    BRACE  = re.compile(r'(?:^|_)(Brace|Bracing|X[- ]?brace|Strut|Tie|IfcStructuralCurveMember|IfcBrace)$', re.IGNORECASE)
    CORE   = re.compile(r'(?:^|_)(Core|ShearCore|LiftCore|CoreWall)$', re.IGNORECASE)
    FOUND  = re.compile(r'(?:^|_)(Foundation|Pile|Footing|Raft)$', re.IGNORECASE)
    def map_type(lbl: str):
        s = localname(lbl)
        if WALL.search(s): return 'Wall'
        if SLAB.search(s): return 'Slab'
        if COLUMN.search(s): return 'Column'
        if BEAM.search(s): return 'Beam'
        if BRACE.search(s): return 'Brace'
        if CORE.search(s): return 'Core'
        if FOUND.search(s): return 'Foundation'
        return None
    return map_type

def s1_inventories(models, func_all=True):
    map_type = build_type_mapper()
    rows_t, rows_f = [], []
    type_hits, type_unknown = [], []
    node2type_all = {}
    avail_rows=[]

    for m in models:
        node2type={}
        for n,tset in m.types.items():
            lbls = set(tset) | set(m.labels.get(n,[]))
            mapped=None
            for lab in lbls:
                tt = map_type(lab)
                if tt:
                    mapped=tt
                    type_hits.append({'model':m.name,'node':n,'label':lab,'mapped':tt})
                    break
            if not mapped:
                for lab in lbls:
                    type_unknown.append({'model':m.name,'node':n,'label':lab})
            if mapped:
                if re.search(r'partition', ' '.join(lbls), re.IGNORECASE):
                    fn = m.funcs.get(n,set())
                    node2type[n] = 'Wall' if (('LB' in fn) or ('Shear' in fn)) else 'Wall_NS'
                else:
                    node2type[n] = mapped
        node2type_all[m.name]=node2type

        ctr = Counter([t for t in node2type.values() if t!='Wall_NS'])
        total = sum(ctr.values()) or 1
        for t,c in ctr.items():
            rows_t.append({'model':m.name,'type':t,'count':c,'share':c/total})

        fctr=Counter()
        for n,fset in m.funcs.items():
            for f in fset: fctr[f]+=1
        ftotal = sum(fctr.values()) or 1
        for f,c in fctr.items():
            rows_f.append({'model':m.name,'function':f,'count':c,'share':c/ftotal})

        avail_rows.append({
            'model':m.name,
            'has_adjacent': int(len(m.edges_adj)>0),
            'has_intersect':int(len(m.edges_int)>0),
            'has_weak':     int(len(m.edges_weak)>0),
            'has_function': int(any(m.funcs.values()))
        })

    df_types = pd.DataFrame(rows_t).sort_values(['model','type']) if rows_t else pd.DataFrame(columns=['model','type','count','share'])
    df_funcs = pd.DataFrame(rows_f).sort_values(['model','function']) if rows_f else pd.DataFrame(columns=['model','function','count','share'])

    # ---- robust WIDE
    model_names = [m.name for m in models]
    if len(df_funcs):
        piv = df_funcs.pivot_table(index='model', columns='function', values='share', fill_value=0.0, aggfunc='sum')
        # ad name might be lost; force it:
        piv.index.name = 'model'
        piv = piv.reset_index()
        if 'model' not in piv.columns:
            # ultra edge-case: first column rename to model
            first_col = piv.columns[0]
            piv = piv.rename(columns={first_col:'model'})
        for col in ['LB','Shear','Moment','Bracing']:
            if col not in piv.columns: piv[col]=0.0
        df_func_wide = piv[['model','LB','Shear','Moment','Bracing']]
    else:
        # no functions -> zeros for all models
        df_func_wide = pd.DataFrame({
            'model': model_names,
            'LB': [0.0]*len(model_names),
            'Shear': [0.0]*len(model_names),
            'Moment':[0.0]*len(model_names),
            'Bracing':[0.0]*len(model_names)
        })

    df_av = pd.DataFrame(avail_rows)

    pd.DataFrame(type_hits).to_csv(os.path.join(OUT_DIR,'type_mapping_hits.csv'), index=False)
    pd.DataFrame(type_unknown).to_csv(os.path.join(OUT_DIR,'type_mapping_unknown.csv'), index=False)
    return df_types, df_funcs, df_func_wide, df_av, node2type_all

# ---------- S2
def count_pairs(A,B,edges):
    s=0
    for (u,v) in edges:
        if u in A and v in B: s+=1
    return s

def s2_motifs(models, df_types, df_funcs, node2type_all,
              allow_type_only=True, proxy_penalty=0.7, weak_topo_penalty=0.5, out_dir='.'):
    rows_counts=[]; proxy_rows=[]
    totals = {m.name:int(df_types[df_types['model']==m.name]['count'].sum()) for m in models}
    for m in models:
        node2type = node2type_all[m.name]
        BEAMS   = {n for n,t in node2type.items() if t=='Beam'}
        COLS    = {n for n,t in node2type.items() if t=='Column'}
        WALLS   = {n for n,t in node2type.items() if t=='Wall'}
        SLABS   = {n for n,t in node2type.items() if t=='Slab'}
        CORES   = {n for n,t in node2type.items() if t=='Core'}
        BRACES  = {n for n,t in node2type.items() if t=='Brace'}
        FRAME_MEM = BEAMS | COLS

        LBn    = {n for n,f in m.funcs.items() if 'LB' in f}
        Shearn = {n for n,f in m.funcs.items() if 'Shear' in f}
        Momentn= {n for n,f in m.funcs.items() if 'Moment' in f}

        # M2
        c_dir  = count_pairs(BEAMS, COLS, m.edges_adj) + count_pairs(COLS, BEAMS, m.edges_adj)
        c_weak = weak_topo_penalty*(count_pairs(BEAMS, COLS, m.edges_weak)+count_pairs(COLS, BEAMS, m.edges_weak))
        c = c_dir + c_weak
        used_proxy=False
        if c==0 and allow_type_only and len(BEAMS)>0 and len(COLS)>0:
            c = proxy_penalty * min(len(BEAMS),len(COLS)); used_proxy=True
        rows_counts.append({'model':m.name,'motif':'M2_frameNode','count':float(c)})
        proxy_rows.append({'model':m.name,'motif':'M2_frameNode','direct':float(c_dir),'weak':float(c_weak),'proxy':float(c-(c_dir+c_weak)) if used_proxy else 0.0})

        # M2b
        c2_dir  = count_pairs(BRACES,FRAME_MEM,m.edges_adj)+count_pairs(FRAME_MEM,BRACES,m.edges_adj)
        c2_weak = weak_topo_penalty*(count_pairs(BRACES,FRAME_MEM,m.edges_weak)+count_pairs(FRAME_MEM,BRACES,m.edges_weak))
        c2 = c2_dir + c2_weak
        used_proxy2=False
        if c2==0 and allow_type_only and len(BRACES)>0 and len(FRAME_MEM)>0:
            c2 = proxy_penalty * min(len(BRACES),len(FRAME_MEM)); used_proxy2=True
        rows_counts.append({'model':m.name,'motif':'M2b_braceNode','count':float(c2)})
        proxy_rows.append({'model':m.name,'motif':'M2b_braceNode','direct':float(c2_dir),'weak':float(c2_weak),'proxy':float(c2-(c2_dir+c2_weak)) if used_proxy2 else 0.0})

        # M3
        c3_dir  = count_pairs(WALLS,SLABS,m.edges_int)+count_pairs(SLABS,WALLS,m.edges_int)
        c3_weak = weak_topo_penalty*(count_pairs(WALLS,SLABS,m.edges_weak)+count_pairs(SLABS,WALLS,m.edges_weak))
        c3 = c3_dir + c3_weak
        used_proxy3=False
        if c3==0 and allow_type_only and len(WALLS)>0 and len(SLABS)>0:
            c3 = proxy_penalty * min(len(WALLS),len(SLABS)); used_proxy3=True
        rows_counts.append({'model':m.name,'motif':'M3_wallSlab','count':float(c3)})
        proxy_rows.append({'model':m.name,'motif':'M3_wallSlab','direct':float(c3_dir),'weak':float(c3_weak),'proxy':float(c3-(c3_dir+c3_weak)) if used_proxy3 else 0.0})

        # M4
        core_touch=0.0; used_proxy4=False
        for u in CORES:
            deg = sum([(u,v) in m.edges_adj and v in SLABS for v in SLABS])
            deg += weak_topo_penalty*sum([(u,v) in m.edges_weak and v in SLABS for v in SLABS])
            if deg>0: core_touch += 1.0
        if core_touch==0 and allow_type_only and len(CORES)>0 and len(SLABS)>0:
            core_touch = proxy_penalty * min(len(CORES),1); used_proxy4=True
        rows_counts.append({'model':m.name,'motif':'M4_core','count':float(core_touch)})
        proxy_rows.append({'model':m.name,'motif':'M4_core','direct':float(core_touch if not used_proxy4 else 0.0),'weak':0.0,'proxy':float(core_touch if used_proxy4 else 0.0)})

        # M5
        rows_counts.append({'model':m.name,'motif':'M5_LB',    'count':float(len(LBn))})
        rows_counts.append({'model':m.name,'motif':'M5_Shear', 'count':float(len(Shearn))})
        rows_counts.append({'model':m.name,'motif':'M5_Moment','count':float(len(Momentn))})
        proxy_rows += [
            {'model':m.name,'motif':'M5_LB','direct':float(len(LBn)),'weak':0.0,'proxy':0.0},
            {'model':m.name,'motif':'M5_Shear','direct':float(len(Shearn)),'weak':0.0,'proxy':0.0},
            {'model':m.name,'motif':'M5_Moment','direct':float(len(Momentn)),'weak':0.0,'proxy':0.0},
        ]

    df_counts = pd.DataFrame(rows_counts)
    if df_counts.empty:
        df_shares = pd.DataFrame(columns=['model','M2_frameNode','M2b_braceNode','M3_wallSlab','M4_core','M5_LB','M5_Shear','M5_Moment'])
        df_dens = df_shares.copy()
    else:
        piv = df_counts.pivot_table(index='model', columns='motif', values='count', aggfunc='sum', fill_value=0.0)
        for c in ['M2_frameNode','M2b_braceNode','M3_wallSlab','M4_core','M5_LB','M5_Shear','M5_Moment']:
            if c not in piv.columns: piv[c]=0.0
        X = piv.copy()
        row_sum = X.sum(axis=1).replace(0.0,1.0)
        df_shares = (X.div(row_sum,axis=0)).reset_index()
        df_shares = df_shares[['model','M2_frameNode','M2b_braceNode','M3_wallSlab','M4_core','M5_LB','M5_Shear','M5_Moment']]
        # per 100
        totals = {m.name:int(df_types[df_types['model']==m.name]['count'].sum()) or 1 for m in models}
        dens=[]
        for m in models:
            tot=max(1,totals[m.name])
            row={'model':m.name}
            for c in ['M2_frameNode','M2b_braceNode','M3_wallSlab','M4_core','M5_LB','M5_Shear','M5_Moment']:
                row[c]=100.0*float(X.loc[m.name,c])/tot
            dens.append(row)
        df_dens=pd.DataFrame(dens)

    pd.DataFrame(proxy_rows).to_csv(os.path.join(out_dir,'motif_proxy_summary.csv'), index=False)
    return df_counts, df_shares, df_dens

# ---------- S3
def s3_system_scores(models, motif_shares, motif_dens, df_types, df_funcs,
                     alpha_m5=0.40, dual_thresh=0.25, out_dir='.'):
    pivS = motif_shares.set_index('model')
    dens = motif_dens.set_index('model')
    dens_norm = dens.copy()
    for c in dens.columns:
        mx = max(1e-9, dens[c].max())
        dens_norm[c] = dens[c]/mx
    rows=[]; comp=[]
    for m in pivS.index:
        frame = 0.6*float(dens_norm.loc[m,'M2_frameNode']) + 0.4*float(pivS.loc[m,'M5_Moment'])*alpha_m5
        wall  = 0.6*float(dens_norm.loc[m,'M3_wallSlab']) + 0.4*(0.7*float(pivS.loc[m,'M5_Shear']) + 0.3*float(pivS.loc[m,'M5_LB']))
        braced= float(dens_norm.loc[m,'M2b_braceNode'])
        dual=0.0
        if frame>=dual_thresh and wall>=dual_thresh:
            dual=0.5*(frame+wall)
        rows.append({'model':m,'frame':frame,'wall':wall,'dual':dual,'braced':braced})
        comp.append({'model':m,
                     'frame_from_M2':float(dens_norm.loc[m,'M2_frameNode']),
                     'frame_from_M5M':float(pivS.loc[m,'M5_Moment'])*alpha_m5,
                     'wall_from_M3':float(dens_norm.loc[m,'M3_wallSlab']),
                     'wall_from_roles':float(0.7*pivS.loc[m,'M5_Shear']+0.3*pivS.loc[m,'M5_LB']),
                     'braced_from_M2b':float(dens_norm.loc[m,'M2b_braceNode'])})
    df_sys=pd.DataFrame(rows)
    pd.DataFrame(comp).to_csv(os.path.join(out_dir,'struct_score_components.csv'), index=False)
    return df_sys

# ---------- S4
def s4_similarity(models, motif_shares, sys_scores, w_motif=0.5, w_system=0.5, out_dir='.'):
    names=[m.name for m in models]
    M = motif_shares.set_index('model')[['M2_frameNode','M2b_braceNode','M3_wallSlab','M4_core','M5_LB','M5_Shear','M5_Moment']].values.astype(float)
    S = sys_scores.set_index('model')[['frame','wall','dual','braced']].values.astype(float)
    n=len(names); Sim=np.zeros((n,n),dtype=float)
    for i in range(n):
        for j in range(n):
            Sim[i,j]= w_motif*cosine(M[i],M[j]) + w_system*cosine(S[i],S[j])
    df=pd.DataFrame(Sim,index=names,columns=names)
    df.to_csv(os.path.join(out_dir,'struct_similarity_matrix.csv'))
    rows=[]
    for i in range(n):
        for j in range(i+1,n):
            rows.append({'A':names[i],'B':names[j],'similarity':float(df.iloc[i,j])})
    pd.DataFrame(rows).sort_values('similarity',ascending=False).to_csv(
        os.path.join(out_dir,'pairwise_structural_summary.csv'), index=False)
    return df

def write_histograms(df_types, df_funcs, df_func_wide, df_av, out_dir):
    df_types.to_csv(os.path.join(out_dir,'struct_types_histogram.csv'), index=False)
    df_funcs.to_csv(os.path.join(out_dir,'struct_functions_histogram.csv'), index=False)
    df_func_wide.to_csv(os.path.join(out_dir,'struct_functions_shares_wide.csv'), index=False)
    df_av.to_csv(os.path.join(out_dir,'struct_data_availability.csv'), index=False)

def write_motif_tables(df_counts, df_shares, df_dens, out_dir):
    df_counts.to_csv(os.path.join(out_dir,'struct_motif_counts.csv'), index=False)
    df_shares.to_csv(os.path.join(out_dir,'struct_motif_shares.csv'), index=False)
    df_dens.to_csv(os.path.join(out_dir,'struct_motif_densities_per100.csv'), index=False)

def write_sys_tables(sys_scores, out_dir):
    sys_scores.to_csv(os.path.join(out_dir,'struct_system_scores.csv'), index=False)

def main():
    parser=argparse.ArgumentParser(description='Structural Extension v25p hardened')
    parser.add_argument('--input-dir', required=True)
    parser.add_argument('--pattern', default='*_DG.rdf')
    parser.add_argument('--out-root', required=True)
    parser.add_argument('--out-name', default='07 - Structural_Extension_v25p')
    parser.add_argument('--func-all', action='store_true')
    parser.add_argument('--allow-type-only-proxy', action='store_true')
    parser.add_argument('--proxy-penalty', type=float, default=0.70)
    parser.add_argument('--weak-topo-penalty', type=float, default=0.50)
    parser.add_argument('--alpha-m5', type=float, default=0.40)
    parser.add_argument('--dual-thresh', type=float, default=0.25)
    parser.add_argument('--w-motif', type=float, default=0.50)
    parser.add_argument('--w-system', type=float, default=0.50)
    parser.add_argument('--emit-debug', action='store_true')
    args=parser.parse_args()

    global OUT_DIR
    OUT_DIR=os.path.join(args.out_root, args.out_name)
    ensure_dir(OUT_DIR)

    import glob
    paths=sorted(glob.glob(os.path.join(args.input_dir, args.pattern)))
    print("[LOAD]"); [print("  ", os.path.basename(p)) for p in paths]
    models=[Model(p) for p in paths]

    # S1
    df_types, df_funcs, df_func_wide, df_av, node2type_all = s1_inventories(models, func_all=args.func_all)
    write_histograms(df_types, df_funcs, df_func_wide, df_av, OUT_DIR)

    # S2
    df_counts, df_shares, df_dens = s2_motifs(
        models, df_types, df_funcs, node2type_all,
        allow_type_only=args.allow_type_only_proxy,
        proxy_penalty=args.proxy_penalty,
        weak_topo_penalty=args.weak_topo_penalty,
        out_dir=OUT_DIR)
    write_motif_tables(df_counts, df_shares, df_dens, OUT_DIR)

    # S3
    sys_scores = s3_system_scores(models, df_shares, df_dens, df_types, df_funcs,
                                  alpha_m5=args.alpha_m5, dual_thresh=args.dual_thresh, out_dir=OUT_DIR)
    write_sys_tables(sys_scores, OUT_DIR)

    # S4
    Smat = s4_similarity(models, df_shares, sys_scores, w_motif=args.w_motif, w_system=args.w_system, out_dir=OUT_DIR)

    info={'version':'v25p-hardened','dual_thresh':args.dual_thresh,
          'w_motif':args.w_motif,'w_system':args.w_system,'alpha_m5':args.alpha_m5,
          'proxy_penalty':args.proxy_penalty,'weak_topo_penalty':args.weak_topo_penalty,
          'allow_type_only_proxy':bool(args.allow_type_only_proxy),
          'func_all':bool(args.func_all),'models':[m.name for m in models]}
    with open(os.path.join(OUT_DIR,'weights_used.json'),'w',encoding='utf-8') as f:
        json.dump(info,f,indent=2)

    print("\n[OK] Saved outputs under:", OUT_DIR)
    for fn in [
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
        print(" -", fn)

if __name__=='__main__':
    main()
