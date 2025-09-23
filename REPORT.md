Executive Summary

This study produces total similarity (S_total) by combining content, typed edge, edge-set, and structural motif similarities derived from Real-BIM RDF design graphs. The repository was scanned, architecture and data flow were extracted, the parameter surface inventory was created, and the canonical motif-based structural similarity was reproduced. Then fusion and visualization steps were executed; all outputs were written to the repo root and under repro_pack/output. With the traceability matrix, RDF properties used were linked to code fragments. Critical minor issues were fixed (column name tolerance, Unicode on Windows console). File list and deep links are below.

Architecture and Data Flow

Text diagram: RDF (*.rdf) → loading via rdflib → graph/motif/feature extraction → component similarities (cosine/jaccard/overlap) → weighted convex combination → total matrix and visuals.
Details: see architecture.md.

Parameter Surface and Effects

Discovered CLI arguments and defaults: parameters.csv.

Real-BIM RDF Usage Validation

Class and property profiles: rb_classes.csv, rb_properties.csv. Namespace overview: ontology_overview.md.
Traceability matrix: trace_matrix.csv.

Similarity Metrics & Ablation

Metrics used: cosine on motif count vectors and overlap-weighted per motif; cosine/Jaccard for content/typed/edge; convex combination weights {0.30, 0.20, 0.10, 0.40}.
Ablation was run: ablation_results.csv.

Evaluation Metrics

No ground-truth labels exist in the repo; weakly supervised evaluation based on expert rules was proposed (see evaluation.md).
Summary metrics: metrics.csv.

Error Analysis and Proposed Patches

compute_similarity.py: extended to accept new column names.

visualize_total_similarity.py: replaced Unicode arrow with ASCII for Windows consoles.
Proposed diffs: see patches/ (next step) and issues.md.

Strengths / Weaknesses / Risks

Strengths: modular pipeline, explicit CSVs, visualizations, transparent weighting.

Weaknesses: no automatic SHACL validation, missing tests and ablation runs.

Risks: namespace canonicalization sensitive to ontology, scale alignment. (see risk_matrix.csv)

Limitations and Future Work

Ontology alignment, entity linking, SHACL shapes, and missing data sensitivity analysis.

How to Reproduce

Environment: use venv. If needed:

.\venv\Scripts\python -m pip install rdflib pandas networkx numpy matplotlib scipy


Structural (canonical) and fusion + visuals:

.\venv\Scripts\python subgraph_similarity_v2.py
.\venv\Scripts\python compute_similarity.py --total-dir ".\repro_pack\output\06 - Total_Similarity" --struct-dir ".\repro_pack\output\05b - Subgraph_Similarity_Canon"
.\venv\Scripts\python visualize_total_similarity.py --in-dir ".\repro_pack\output\06 - Total_Similarity" --out-dir ".\repro_pack\output\06b - Total_Similarity_Visuals"

Generated Files

repo_map.json, architecture.md, parameters.csv

rb_classes.csv, rb_properties.csv, ontology_overview.md, trace_matrix.csv

repro_pack/output/05b - Subgraph_Similarity_Canon/*

repro_pack/output/06 - Total_Similarity/*, repro_pack/output/06b - Total_Similarity_Visuals/*

run_log.txt