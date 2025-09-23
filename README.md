# Ontology-Driven Building Similarity Comparison

> Total similarity (S_total) from Real-BIM RDF design graphs by combining content, typed-edge, edge-set, and structural motif components.
> Reproducible pipeline + reports + visuals.

## Features
- Canonical structural motif similarity + total fusion
- RDF profiling: classes/properties, namespaces
- Traceability matrix (features ↔ RDF predicates)
- Visuals: heatmap, dendrogram, contribution bars
- Ablation & missingness sensitivity
- SHACL scaffold for validation
- Reproducible CSV/MD artefacts

## Repository Map (highlights)
- `subgraph_similarity_v2.py` – structural similarity (canon)
- `compute_similarity.py` – component fusion → S_total
- `visualize_total_similarity.py` – heatmap/dendrogram/bars
- `tools/` – helpers (repo map, RDF profiling, etc.)
- Reports & data: `REPORT.md`, `parameters.csv`, `rb_classes.csv`,
  `rb_properties.csv`, `ontology_overview.md`, `trace_matrix.csv`,
  `evaluation.md`, `metrics.csv`, `ablation_results.csv`,
  `missingness_sensitivity.csv`, `static_analysis.md`, `shacl_report.md`,
  `risk_matrix.csv`

## Installation
```bash
python -m venv venv
# Windows
.\n+venv\Scripts\python -m pip install --upgrade pip
.
venv\Scripts\python -m pip install -r requirements.txt
# If requirements are missing:
.
venv\Scripts\python -m pip install rdflib pandas networkx numpy matplotlib scipy
```

## Reproduction
```powershell
# Structural (canonical)
.
venv\Scripts\python subgraph_similarity_v2.py

# Fusion (S_total)
.
venv\Scripts\python compute_similarity.py \
  --total-dir ".\repro_pack\output\06 - Total_Similarity" \
  --struct-dir ".\repro_pack\output\05b - Subgraph_Similarity_Canon"

# Visuals
.
venv\Scripts\python visualize_total_similarity.py \
  --in-dir ".\repro_pack\output\06 - Total_Similarity" \
  --out-dir ".\repro_pack\output\06b - Total_Similarity_Visuals"
```

## Key Outputs
- `repro_pack/output/05b - Subgraph_Similarity_Canon/*`
- `repro_pack/output/06 - Total_Similarity/*`
- `repro_pack/output/06b - Total_Similarity_Visuals/*`
- `REPORT.md`, `run_log.txt`, `parameters.csv`,
  `rb_classes.csv`, `rb_properties.csv`, `ontology_overview.md`,
  `trace_matrix.csv`, `evaluation.md`, `metrics.csv`,
  `ablation_results.csv`, `missingness_sensitivity.csv`,
  `static_analysis.md`, `shacl_report.md`, `risk_matrix.csv`

## Notes
- Windows console unicode quirk fixed in `visualize_total_similarity.py`.
- Column name tolerance extended in `compute_similarity.py`.
- Consider refining SHACL shapes and ontology alignment for future work.

## License
MIT — see `LICENSE`.


