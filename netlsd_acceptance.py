#!/usr/bin/env python3
import glob, json, os, numpy as np, pandas as pd

# Find latest report and validation directories
rep_dirs = sorted([d for d in glob.glob('data/results/report_struct_netlsd_*') if os.path.isdir(d)], key=os.path.getmtime)
rep = rep_dirs[-1] if rep_dirs else 'data/results'
val_dirs = sorted([d for d in glob.glob('data/results/validation_struct_netlsd_*') if os.path.isdir(d)], key=os.path.getmtime)
val = val_dirs[-1]

print(f"Using validation directory: {val}")
print(f"Using report directory: {rep}")

# Load NetLSD similarity matrix
S = pd.read_csv(os.path.join(val, 'struct_similarity_netlsd.csv'), index_col=0).astype(float)
A = S.values
n = A.shape[0]
off = A[~np.eye(n, dtype=bool)]

# Compute summary statistics
summary = {
    'sp_like_channel': 'NetLSD',
    'sp_mean_offdiag': round(off.mean(), 4),
    'sp_min_offdiag': round(off.min(), 4),
    'sp_max_offdiag': round(off.max(), 4),
}

# Find top pairs
pairs = []
for i, a in enumerate(S.index):
    for j, b in enumerate(S.columns):
        if i < j:
            pairs.append((float(S.iloc[i, j]), a, b))

summary['top_pairs'] = [{'sim': round(v, 4), 'a': a, 'b': b} for v, a, b in sorted(pairs, reverse=True)[:3]]

# Try to add total similarity range if present
ts_path = 'repro_pack/output/06 - Total_Similarity/total_similarity_matrix.csv'
if os.path.exists(ts_path):
    T = pd.read_csv(ts_path, index_col=0).astype(float)
    B = T.values
    m = B[~np.eye(B.shape[0], dtype=bool)]
    summary['total_mean_offdiag'] = round(m.mean(), 4)
    summary['total_min_offdiag'] = round(m.min(), 4)
    summary['total_max_offdiag'] = round(m.max(), 4)

# Create report directory and save notes
os.makedirs(rep, exist_ok=True)
with open(os.path.join(rep, 'REPORT_NOTES.md'), 'w', encoding='utf-8') as f:
    f.write('# NetLSD structural channel — report snapshot\n\n')
    for k, v in summary.items():
        f.write(f'- {k}: {v}\n')

# Save JSON summary
with open('NETLSD_run_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print("\n=== NetLSD Acceptance Summary ===")
print(json.dumps(summary, indent=2))

print(f"\n=== Files Created ===")
print(f"Report directory: {rep}")
print(f"Validation directory: {val}")
print(f"Summary JSON: NETLSD_run_summary.json")



