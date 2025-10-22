<# 
Validate + Package (PowerShell-safe)
- Recomputes S1→S4 structural channel
- Recomputes TOTAL similarity
- Generates visuals
- Prints acceptance JSON
- Packages to timestamped folders
#>

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# 1) Ensure we run from repo root
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
if ($here) { Set-Location $here | Out-Null }

# 2) Recompute S1→S4 (writes CSVs to current folder)
Write-Host "[STEP] S1→S4..." -ForegroundColor Cyan
python run_s1s4_struct.py --models_dir data/RDF_models --out_dir .

# 3) Locate Step-1 matrices (content/typed/edge)
function Find-One($pattern) {
  $f = Get-ChildItem -Recurse -File -Filter $pattern | Sort-Object LastWriteTime -Descending | Select-Object -ExpandProperty FullName -First 1
  if (-not $f) { throw "Required file not found: $pattern" }
  return $f
}
$CONTENT = Find-One "similarity_content_cosine.csv"
$TYPED   = Find-One "similarity_typed_edge_cosine.csv"
$EDGESET = Find-One "similarity_edge_sets_jaccard.csv"
Write-Host "[INFO] CONTENT: $CONTENT"
Write-Host "[INFO] TYPED  : $TYPED"
Write-Host "[INFO] EDGESET: $EDGESET"

# 4) TOTAL similarity (uses S1S4 as structural)
Write-Host "[STEP] TOTAL similarity..." -ForegroundColor Cyan
python total_similarity.py `
  --content-cos   "$CONTENT" `
  --typed-cos     "$TYPED" `
  --edge-jaccard  "$EDGESET" `
  --struct-sim    ".\struct_similarity_s1s4.csv" `
  --struct-source s1s4 `
  --output-dir    .

# 5) Visuals (S1S4 + totals; SP/NetLSD if present)
Write-Host "[STEP] Visuals..." -ForegroundColor Cyan
python visualize_similarity.py --input-dir . --output-dir . --include-s1s4 --include-sp --annot

# 6) Acceptance JSON (write tiny temp .py → PowerShell-safe)
$py = @'
import json, numpy as np, pandas as pd
S = pd.read_csv("struct_similarity_s1s4.csv", index_col=0).astype(float).values
n = S.shape[0]
off = S[~np.eye(n, dtype=bool)]
T = pd.read_csv("total_similarity_matrix.csv", index_col=0).astype(float).values
m = T[~np.eye(T.shape[0], dtype=bool)]
out = {
  "S1S4_mean_offdiag": round(float(off.mean()), 4),
  "S1S4_min_offdiag":  round(float(off.min()),  4),
  "S1S4_max_offdiag":  round(float(off.max()),  4),
  "TOTAL_mean_offdiag": round(float(m.mean()), 4),
  "TOTAL_min_offdiag":  round(float(m.min()),  4),
  "TOTAL_max_offdiag":  round(float(m.max()),  4),
}
print(json.dumps(out, indent=2))
with open("ACCEPTANCE_SUMMARY.json","w") as f:
  json.dump(out, f, indent=2)
'@
$tmp = Join-Path $env:TEMP ("s1s4_accept_" + [guid]::NewGuid().ToString() + ".py")
Set-Content -LiteralPath $tmp -Encoding UTF8 -Value $py
Write-Host "[STEP] Acceptance JSON..." -ForegroundColor Cyan
python $tmp
Remove-Item $tmp -Force

# 7) Package to timestamped folders
$stamp  = Get-Date -Format "yyyy-MM-dd_HH-mm"
$rep    = "data\results\report_S1S4_$stamp"
$viz    = "data\results\report_S1S4_${stamp}_viz"
$val    = "data\results\validation_S1S4_$stamp"
New-Item -ItemType Directory -Force -Path $rep,$viz,$val | Out-Null

# Numerics (validation)
Move-Item s1_inventory.csv,s2_motifs.csv,s3_system_scores.csv,s4_motif_share_vectors.csv,struct_similarity_s1s4*.csv,s1s4_meta.json,ACCEPTANCE_SUMMARY.json -Destination $val -ErrorAction SilentlyContinue

# Report tables
Copy-Item total_similarity_matrix.csv,pairwise_total_summary.csv,struct_similarity_s1s4.csv -Destination $rep -ErrorAction SilentlyContinue

# Figures
Move-Item S1S4_*png,total_similarity_*png,component_contrib_*.png -Destination $viz -ErrorAction SilentlyContinue

# 8) Print where things went
"`nREPORT  : $(Resolve-Path $rep)"
"FIGURES : $(Resolve-Path $viz)"
"VALIDATE: $(Resolve-Path $val)"
