# Structural Similarity – S1→S4 Channel + Total Similarity

This repository implements a **standards-aligned** structural channel (S1→S4) for comparing building RDF models (IFC-ish) and fuses it with three additional channels (content, typed-edge, edge-set) to produce a **total similarity** score and visualizations.

## Contents
- [Quick Start](#quick-start)
- [Pipelines & Channels](#pipelines--channels)
- [S1→S4 Structural Channel](#s1s4-structural-channel)
- [Fusion (Total Similarity)](#fusion-total-similarity)
- [Outputs](#outputs)
- [Design Choices & Assumptions](#design-choices--assumptions)
- [Limitations](#limitations)
- [Repro/Packaging](#repropackaging)
- [License](#license)

---

## Quick Start

```powershell
# 0) activate venv
& .\venv\Scripts\Activate.ps1

# 1) run the validator + packager (recomputes S1→S4, totals, visuals)
.\scripts\validate_and_package.ps1
```

This writes CSVs/PNGs to the project root and also copies them into timestamped `data/results/...` folders.

---

## Pipelines & Channels

We compute similarity from **four** complementary channels:

1. **Content (cosine)**
   Cosine similarity between document-term vectors built from RDF labels, class names, and string literals.
   *File*: `similarity_content_cosine.csv`

2. **Typed-edge (cosine)**
   Cosine similarity between histograms of typed relations (e.g., Beam–Column, Wall–Slab).
   *File*: `similarity_typed_edge_cosine.csv`

3. **Edge-set (Jaccard)**
   Jaccard over canonicalized untyped edge sets (structure-only).
   *File*: `similarity_edge_sets_jaccard.csv`

4. **Structural S1→S4 (this repo)**
   Inventory → Motifs → System Scores → Structural Similarity.
   *Files*: see [Outputs](#outputs).

The first three matrices are produced by earlier pipeline steps and reused here. The **S1→S4** matrices are computed by this repo.

---

## S1→S4 Structural Channel

**S1 – Inventory**

* Parse RDF (`rdflib`) and build a **typed multigraph** (`networkx`).
* **Element typing** uses IFC-aligned patterns:
  `Wall` (IfcWall, IfcWallStandardCase), `Slab` (IfcSlab, IfcPlate, IfcCovering),
  `Beam` (IfcBeam), `Column` (IfcColumn),
  `Brace` (IfcStructuralCurveMember, IfcMember), `Core` (Core/ShearCore labels).
* **Relations** include `adjacentElement`, `AdjacentComponent`, `connectedTo`, `relatesElement`, `relatedElements`, `adjacentZone`, `intersectingElement`, `partOf/hasPart/isPartOf/containsElement`, `RelAggregates`.

**S2 – Motifs (counts, densities, penalties)**

* Motifs: **M2** (Beam–Column), **M2b** (Brace–Beam/Column), **M3** (Wall–Slab), **M4** (Core participation).
* Relaxed variants (≤1-hop) if strict counts are zero.
* Penalties: `weak_topo_penalty` (e.g., 0.5) for relaxed-only detections; `proxy_penalty` (e.g., 0.7) for proxy-typed elements.
* Densities normalized **per 100 edges**.

**S3 – System Scores**

* Linear evidence from motif densities + functional role shares (if present):
  `Frame ≈ 0.6·M2 + 0.4·M5_moment`,
  `Braced ≈ 0.7·M2b + 0.3·M5_bracing`,
  `Wall ≈ 0.7·M3 + 0.3·M5_shear`,
  `Dual = min(Frame, Wall)` if both exceed a floor.
* Role shares default to 0 in absence of explicit labels.

**S4 – Similarity**

* Build motif-share vectors (edge-normalized).
* Pairwise **cosine** on motif shares and on system scores.
* If vectors are near 1-D, we z-score features to avoid cosine collapse.
* Cosines are mapped **from [−1,1] to [0,1]** and clipped to [0,1].
* Structural similarity is the fusion: `S_struct = 0.6·S_motif + 0.4·S_system`.
* Helper models (e.g., `0000_Merged.rdf`) are excluded before pairwise sims.

*Implementation*: `s1s4_struct.py` (module), `run_s1s4_struct.py` (CLI wrapper).

---

## Fusion (Total Similarity)

`total_similarity.py` loads the four channel matrices (long- or wide-form) and fuses them:

```
S_total = 0.30*S_content + 0.20*S_typed + 0.10*S_edge + 0.40*S_struct
```

Weights are configurable on the CLI and are written to `weights_used.json`.

---

## Outputs

**CSVs**

* `s1_inventory.csv`
* `s2_motifs.csv`
* `s3_system_scores.csv`
* `s4_motif_share_vectors.csv`
* `struct_similarity_s1s4_motif.csv`
* `struct_similarity_s1s4_system.csv`
* `struct_similarity_s1s4.csv`
* `total_similarity_matrix.csv`
* `pairwise_total_summary.csv`
* `weights_used.json`
* `s1s4_meta.json`

**PNGs**

* `S1S4_structural_heatmap.png`
* `S1S4_motif_share_heatmap.png`
* `S1S4_system_radar.png`
* `total_similarity_heatmap.png`
* `total_similarity_dendrogram.png`
* `component_contrib_*.png`
* (Optional) `SP_*png`, `NetLSD_*png`

---

## Design Choices & Assumptions

* IFC-aligned **type mapping** and broad **relation vocab** for portability across exports.
* **Penalized relaxed motifs**: count signal even without explicit beam-column links, but with downweighting for weaker evidence.
* **Scale alignment**: all structural cosines mapped to [0,1] before fusion.
* **Helper model filtering**: non-design aggregates (e.g., `0000_Merged.rdf`) removed.

---

## Limitations

* If models share very similar structural strategies, S1→S4 off-diagonals will be high (expected).
* Missing or inconsistent typing can undercount motifs (we mitigate via relaxed detection + proxy penalties, but it's not magic).
* System scores are a **linear** evidence model—good for auditability; not a code-check or nonlinear classifier.
* Content/typed/edge matrices come from prior steps; their quality gates the fusion.

---

## Repro/Packaging

```powershell
# One command to recompute + package
.\scripts\validate_and_package.ps1
```

Outputs are copied into `data/results/report_S1S4_<timestamp>` (tables),
`..._viz` (figures), and `validation_S1S4_<timestamp>` (all numerics + meta).

---

## License

This project is licensed under the MIT License - see the LICENSE file for details.