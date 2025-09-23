## Architecture overview
Data layers and key modules by role.

### Layers
- Input: RDF/XML models in project root ( *_DG.rdf ), ontology file 0000_Merged.rdf
- ETL/Feature extraction: extract_features.py, subgraph_similarity*.py, structural_extension*.py
- Similarity calculators: content/typed/edge Jaccard/Cosine (CSV under repro_pack/output/02), structural (05/05b/07)
- Fusion: combine_total_similarity.py, compute_similarity.py -> total_similarity_matrix.csv
- Visualization/Reporting: visualize_*.py

### Data flow (text)
RDF (*.rdf) -> parsing (rdflib) -> graphs/motifs/types -> component similarities (CSV) -> convex fusion (weights) -> total score CSV + visuals

### Files by role
- feature_extraction:
  - export_m5_to_long.py
  - extract_features.py
- misc:
  - pairwise_diffs_content.py
  - pairwise_diffs_edge_sets.py
  - pairwise_diffs_typed_edge.py
  - tools/gen_repo_map.py
  - verify_outputs.py
  - verify_repro_pass.py
- rdf_data:
  - 0000_Merged.rdf
  - Building_05_DG.rdf
  - Building_06_DG.rdf
  - Freiform_Haus_DG.rdf
  - Option03_Revising_DG.rdf
  - Option04_Rev03_DG.rdf
- similarity_fusion:
  - combine_total_similarity.py
  - compute_similarity.py
  - compute_similarity_legacy.py
- structural_similarity:
  - structural_extension.py
  - structural_extension_v24.py
  - structural_extension_v25.py
  - structural_extension_v25b.py
  - structural_extension_v25c.py
  - structural_extension_v25d.py
  - structural_extension_v25e.py
  - structural_extension_v25f.py
  - structural_extension_v25g.py
  - structural_extension_v25h.py
  - structural_extension_v25j.py
  - structural_extension_v25k.py
  - structural_extension_v25l.py
  - structural_extension_v25m.py
  - structural_extension_v25n.py
  - structural_extension_v25o.py
  - structural_extension_v25p.py
  - structural_extension_v25p1.py
  - structural_extension_v25p2.py
  - subgraph_similarity.py
  - subgraph_similarity_v2.py
- visualization:
  - visualize_motif_similarity.py
  - visualize_predicate_bars.py
  - visualize_similarity.py
  - visualize_struct_matrix_heatmap.py
  - visualize_structural_bundle.py
  - visualize_structural_extension.py
  - visualize_structural_radar.py
  - visualize_structural_report.py
  - visualize_total_similarity.py
