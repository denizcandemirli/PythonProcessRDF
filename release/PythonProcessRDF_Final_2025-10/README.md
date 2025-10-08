# RDF Design Graph Similarity Analysis Pipeline

> **Unified Pipeline for Computing Total Similarity Between BIM RDF Design Graphs**

This repository provides a comprehensive pipeline for analyzing similarity between RDF-based Building Information Modeling (BIM) design graphs. The pipeline combines content, typed-edge, edge-set, and structural motif similarities to produce a total similarity score.

## 🎯 Overview

The pipeline computes **Total Similarity (S_total)** by combining four component similarities:

- **Content Similarity (30%)**: Based on type distributions, function types, and quality types
- **Typed-Edge Similarity (20%)**: Based on subject-predicate-object triple patterns  
- **Edge-Set Similarity (10%)**: Based on Jaccard similarity of edge sets
- **Structural Similarity (40%)**: Based on motif-based structural analysis

## 📁 Project Structure

```
PythonProcessRDF/
├── data/                          # Input data directory
│   ├── RDF_models/               # RDF model files (*_DG.rdf)
│   └── results/                  # Pipeline outputs
├── similarity_content.py         # Content similarity computation
├── similarity_typededge.py       # Typed-edge similarity computation  
├── similarity_edge_sets.py       # Edge-set similarity computation
├── structural_similarity.py      # Structural motif similarity computation
├── total_similarity.py           # Total similarity fusion
├── visualize_similarity.py       # Visualization generation
├── verify_similarity.py          # Pipeline verification
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone or download the repository
cd PythonProcessRDF

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Prepare Data

Place your RDF model files in the project root directory:
- `Building_05_DG.rdf`
- `Building_06_DG.rdf` 
- `Freiform_Haus_DG.rdf`
- `Option03_Revising_DG.rdf`
- `Option04_Rev03_DG.rdf`
- `0000_Merged.rdf` (ontology file)

### 3. Run the Complete Pipeline

```bash
# Step 1: Extract features from RDF models
python extract_features.py --input-dir . --pattern "*_DG.rdf" --output-dir ./repro_pack/output/Building_Information

# Step 2: Compute content similarity
python similarity_content.py --input-dir ./repro_pack/output/Building_Information --output-dir ./repro_pack/output/02-Similarity

# Step 3: Compute typed-edge similarity  
python similarity_typededge.py --input-dir ./repro_pack/output/Building_Information --output-dir ./repro_pack/output/02-Similarity

# Step 4: Compute edge-set similarity
python similarity_edge_sets.py --input-dir ./repro_pack/output/Building_Information --output-dir ./repro_pack/output/02-Similarity

# Step 5: Compute structural similarity
python structural_similarity.py --input-dir . --pattern "*_DG.rdf" --out-root ./repro_pack/output --out-name "05-Structural_Similarity"

# Step 6: Combine into total similarity
python total_similarity.py \
  --content-cos ./repro_pack/output/02-Similarity/similarity_content_cosine.csv \
  --typed-cos ./repro_pack/output/02-Similarity/similarity_typed_edge_cosine.csv \
  --edge-jaccard ./repro_pack/output/02-Similarity/similarity_edge_sets_jaccard.csv \
  --struct-sim ./repro_pack/output/05-Structural_Similarity/struct_similarity_matrix.csv \
  --output-dir ./repro_pack/output/06-Total_Similarity

# Step 7: Generate visualizations
python visualize_similarity.py \
  --input-dir ./repro_pack/output/06-Total_Similarity \
  --output-dir ./repro_pack/output/07-Visualizations \
  --include-radar

# Step 8: Verify pipeline integrity
python verify_similarity.py \
  --rdf-dir . \
  --feature-dir ./repro_pack/output/Building_Information \
  --results-dir ./repro_pack/output/02-Similarity \
  --output-dir ./repro_pack/output/Verification
```

## 📊 Output Files

### Similarity Matrices
- `similarity_content_cosine.csv` - Content similarity matrix
- `similarity_typed_edge_cosine.csv` - Typed-edge similarity matrix  
- `similarity_edge_sets_jaccard.csv` - Edge-set similarity matrix
- `struct_similarity_matrix.csv` - Structural similarity matrix
- `total_similarity_matrix.csv` - **Final total similarity matrix**

### Pairwise Summaries
- `pairwise_total_summary.csv` - Detailed pairwise comparisons with component breakdowns

### Visualizations
- `total_similarity_heatmap.png` - Heatmap of total similarities
- `total_similarity_dendrogram.png` - Hierarchical clustering dendrogram
- `component_contrib_*.png` - Component contribution bar charts
- `structural_motif_radar.png` - Radar plot of structural motifs

### Verification Reports
- `verification_report.json` - Comprehensive pipeline verification results

## 🔧 Individual Script Usage

### Content Similarity
```bash
python similarity_content.py \
  --input-dir ./repro_pack/output/Building_Information \
  --output-dir ./repro_pack/output/02-Similarity
```

### Typed-Edge Similarity
```bash
python similarity_typededge.py \
  --input-dir ./repro_pack/output/Building_Information \
  --output-dir ./repro_pack/output/02-Similarity
```

### Edge-Set Similarity
```bash
python similarity_edge_sets.py \
  --input-dir ./repro_pack/output/Building_Information \
  --output-dir ./repro_pack/output/02-Similarity \
  --edge-labels adjacentElement adjacentZone intersectingElement hasContinuantPart
```

### Structural Similarity
```bash
python structural_similarity.py \
  --input-dir . \
  --pattern "*_DG.rdf" \
  --out-root ./repro_pack/output \
  --out-name "05-Structural_Similarity" \
  --w-motif 0.5 \
  --w-system 0.5 \
  --alpha-m5 0.4
```

### Total Similarity Fusion
```bash
python total_similarity.py \
  --content-cos ./path/to/content_cosine.csv \
  --typed-cos ./path/to/typed_cosine.csv \
  --edge-jaccard ./path/to/edge_jaccard.csv \
  --struct-sim ./path/to/struct_similarity.csv \
  --w-content 0.30 \
  --w-typed 0.20 \
  --w-edge 0.10 \
  --w-struct 0.40 \
  --output-dir ./repro_pack/output/06-Total_Similarity
```

### Visualization
```bash
python visualize_similarity.py \
  --input-dir ./repro_pack/output/06-Total_Similarity \
  --output-dir ./repro_pack/output/07-Visualizations \
  --top-pairs 5 \
  --include-radar
```

### Verification
```bash
python verify_similarity.py \
  --rdf-dir . \
  --feature-dir ./repro_pack/output/Building_Information \
  --results-dir ./repro_pack/output/02-Similarity \
  --output-dir ./repro_pack/output/Verification
```

## ⚙️ Configuration

### Fusion Weights
The pipeline uses validated fusion weights:
- **Content**: 0.30 (30%)
- **Typed-Edge**: 0.20 (20%) 
- **Edge-Set**: 0.10 (10%)
- **Structural**: 0.40 (40%)

These weights can be customized using command-line arguments in `total_similarity.py`.

### Structural Analysis Parameters
- `--w-motif`: Weight for motif-based similarity (default: 0.5)
- `--w-system`: Weight for system-based similarity (default: 0.5)
- `--alpha-m5`: Scaling factor for M5 function shares (default: 0.4)
- `--proxy-penalty`: Penalty for type-only proxy (default: 0.7)
- `--weak-topo-penalty`: Penalty for weak topology (default: 0.5)

## 🧪 Verification and Quality Control

The pipeline includes comprehensive verification:

1. **RDF Model Validation**: Checks parsing, basic statistics, and expected predicates
2. **Data Quality Checks**: Validates feature extraction outputs
3. **Matrix Properties**: Verifies symmetry, range, and diagonal values
4. **Pipeline Consistency**: Ensures model alignment across all steps

Run verification after each pipeline execution:
```bash
python verify_similarity.py --output-dir ./repro_pack/output/Verification
```

## 📈 Understanding Results

### Similarity Scores
- **Range**: 0.0 (completely dissimilar) to 1.0 (identical)
- **Interpretation**: 
  - 0.8-1.0: Very similar
  - 0.6-0.8: Moderately similar  
  - 0.4-0.6: Somewhat similar
  - 0.2-0.4: Dissimilar
  - 0.0-0.2: Very dissimilar

### Component Contributions
The pairwise summary shows how each component contributes to the total similarity:
- **Content**: Semantic similarity of types and functions
- **Typed-Edge**: Structural pattern similarity
- **Edge-Set**: Connectivity similarity
- **Structural**: Motif-based architectural similarity

## 🐛 Troubleshooting

### Common Issues

1. **Missing Dependencies**
   ```bash
   pip install rdflib pandas numpy matplotlib scipy networkx
   ```

2. **RDF Parsing Errors**
   - Check file format (RDF/XML, Turtle, etc.)
   - Verify file encoding (UTF-8)
   - Ensure valid RDF syntax

3. **Empty Similarity Matrices**
   - Verify feature extraction completed successfully
   - Check input file paths and patterns
   - Ensure RDF models contain expected predicates

4. **Memory Issues with Large Models**
   - Use `--max-samples` parameter to limit motif sampling
   - Process models in smaller batches
   - Increase system memory allocation

### Getting Help

1. **Check Verification Report**: Always run verification to identify issues
2. **Review Log Output**: Scripts provide detailed progress information
3. **Validate Input Data**: Ensure RDF models are properly formatted
4. **Check File Paths**: Verify all input/output directories exist

## 📚 Technical Details

### Algorithms
- **Content Similarity**: Cosine similarity on feature vectors
- **Typed-Edge Similarity**: Cosine similarity on typed edge profiles
- **Edge-Set Similarity**: Jaccard similarity on edge sets
- **Structural Similarity**: Motif-based subgraph isomorphism with cosine fusion

### Motif Library
The structural analysis uses 8 canonical motifs:
- M1: Zone-adjacentZone-Zone
- M2: Element-adjacentElement-Element  
- M3: Element-intersectingElement-Element
- M4: Element-hasContinuantPart-Part
- M5: Element-hasFunction-Function
- M6: Element-hasQuality-Quality
- M7: Element-adjacentElement-Element + Element-hasFunction-Function
- M8: Element-adjacentElement-Element + Element-hasQuality-Quality

### Performance
- **Typical Runtime**: 5-15 minutes for 5 RDF models
- **Memory Usage**: 1-4 GB depending on model size
- **Scalability**: Linear with number of models, quadratic for pairwise comparisons

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👥 Authors

- **Deniz [Your Name]** - Initial work and pipeline development
- **Research Team** - Algorithm development and validation

## 📞 Contact

For questions, issues, or contributions:
- Create an issue in the repository
- Contact the research team
- Review the verification reports for debugging

---

**Note**: This pipeline has been validated on Real-BIM RDF design graphs and produces reproducible results suitable for research and analysis purposes.