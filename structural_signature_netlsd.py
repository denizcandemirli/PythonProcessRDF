#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File: structural_signature_netlsd.py
Purpose: Compute NetLSD (heat-trace) structural signatures for RDF models.
Input: RDF models from data/RDF_models directory
Output: netlsd_vectors.csv, struct_similarity_netlsd.csv, netlsd_meta.json

This script implements fast structural similarity based on NetLSD heat-trace signatures:
- Normalized Laplacian eigenvalues computation
- Heat kernel trace at log-spaced time points
- Predicate-layer aggregation (mean or concatenation)
- Cosine similarity between final vectors

NetLSD provides a robust, scalable alternative to motif-based structural analysis
with orders of magnitude faster computation and better theoretical grounding.

Author: Deniz Demirli
Supervisor: Dr. Chao Li (TUM)
Version: 2025.10
License: SPDX-License-Identifier: CC-BY-4.0
"""

__version__ = "2025.10"
__author__ = "Deniz Demirli"
__supervisor__ = "Dr. Chao Li (TUM)"

import os
import argparse
import json
import glob
from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np
import networkx as nx
from rdflib import Graph
from rdflib.namespace import RDF, RDFS, OWL
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh
from scipy.linalg import eigvalsh

# ---------------------------
# Configuration
# ---------------------------

# Default parameters
DEFAULT_TIMES = 64
DEFAULT_AGGREGATE = "mean"
DEFAULT_MIN_EDGES = 10
DEFAULT_TOP_K = 30
DEFAULT_EXCLUDE = ["type", "label", "comment", "subClassOf", "first", "rest", "naturalLanguageDefinition"]

# ---------------------------
# Helper Functions
# ---------------------------

def ensure_dir(path: str):
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)

def load_rdf_model(filepath: str) -> Graph:
    """Load RDF model from file."""
    try:
        g = Graph()
        g.parse(filepath, format="xml")
        return g
    except Exception as e:
        print(f"[WARN] Could not parse {filepath}: {e}")
        return Graph()

def extract_predicate_layers(graph: Graph, exclude_predicates: List[str]) -> Dict[str, nx.Graph]:
    """Extract predicate layers as undirected simple graphs."""
    layers = {}
    
    for s, p, o in graph:
        # Skip excluded predicates
        pred_name = str(p).split("#")[-1].split("/")[-1]
        if pred_name in exclude_predicates:
            continue
            
        # Create layer key
        layer_key = f"pred_{pred_name}"
        
        if layer_key not in layers:
            layers[layer_key] = nx.Graph()
        
        # Add edge (undirected)
        subj = str(s)
        obj = str(o)
        layers[layer_key].add_edge(subj, obj)
    
    return layers

def filter_layers(layers: Dict[str, nx.Graph], min_edges: int, top_k: int) -> Dict[str, nx.Graph]:
    """Filter layers by minimum edges and keep top K largest."""
    # Filter by minimum edges
    filtered = {k: v for k, v in layers.items() if v.number_of_edges() >= min_edges}
    
    # Sort by number of edges and keep top K
    sorted_layers = sorted(filtered.items(), key=lambda x: x[1].number_of_edges(), reverse=True)
    top_layers = dict(sorted_layers[:top_k])
    
    return top_layers

def compute_netlsd_signature(graph: nx.Graph, times: np.ndarray) -> np.ndarray:
    """Compute NetLSD heat-trace signature for a single graph."""
    n_nodes = graph.number_of_nodes()
    
    # Handle empty or single node graphs
    if n_nodes <= 1:
        return np.zeros_like(times)
    
    try:
        # Compute normalized Laplacian
        L_norm = nx.normalized_laplacian_matrix(graph)
        
        # Convert to dense if small, sparse if large
        if n_nodes <= 1000:
            L_dense = L_norm.toarray()
            eigenvals = np.linalg.eigvalsh(L_dense)
        else:
            # Use sparse eigenvalue computation for large graphs
            eigenvals, _ = eigsh(L_norm, k=min(n_nodes-1, 100), which='SM', sigma=0)
            eigenvals = np.sort(eigenvals)
        
        # Compute heat trace: h(t) = sum(exp(-t * lambda_i))
        heat_trace = np.zeros_like(times)
        for i, t in enumerate(times):
            heat_trace[i] = np.sum(np.exp(-t * eigenvals))
            
        return heat_trace
        
    except Exception as e:
        print(f"[WARN] NetLSD computation failed: {e}")
        return np.zeros_like(times)

def aggregate_layer_signatures(signatures: Dict[str, np.ndarray], method: str) -> np.ndarray:
    """Aggregate signatures from multiple layers."""
    if not signatures:
        return np.array([])
    
    sig_list = list(signatures.values())
    
    if method == "mean":
        return np.mean(sig_list, axis=0)
    elif method == "concat":
        return np.concatenate(sig_list)
    else:
        raise ValueError(f"Unknown aggregation method: {method}")

def normalize_vector(v: np.ndarray) -> np.ndarray:
    """L2-normalize vector."""
    norm = np.linalg.norm(v)
    if norm < 1e-12:
        return v
    return v / norm

def compute_cosine_similarity_matrix(vectors: pd.DataFrame) -> pd.DataFrame:
    """Compute cosine similarity matrix between vectors."""
    # Normalize vectors
    normalized = vectors.div(np.linalg.norm(vectors, axis=1), axis=0)
    
    # Compute cosine similarity
    similarity_matrix = np.dot(normalized, normalized.T)
    
    # Create DataFrame with model names
    models = vectors.index.tolist()
    return pd.DataFrame(similarity_matrix, index=models, columns=models)

# ---------------------------
# Main Function
# ---------------------------

def main():
    """Main execution function."""
    ap = argparse.ArgumentParser(description="Compute NetLSD structural signatures for RDF models")
    ap.add_argument("--models_dir", default="data/RDF_models",
                   help="Directory containing RDF models")
    ap.add_argument("--out_dir", default=".",
                   help="Output directory for results")
    ap.add_argument("--times", type=int, default=DEFAULT_TIMES,
                   help=f"Number of log-spaced time points (default: {DEFAULT_TIMES})")
    ap.add_argument("--aggregate", choices=["mean", "concat"], default=DEFAULT_AGGREGATE,
                   help=f"Aggregation method for layer signatures (default: {DEFAULT_AGGREGATE})")
    ap.add_argument("--min_edges_per_layer", type=int, default=DEFAULT_MIN_EDGES,
                   help=f"Minimum edges per layer to keep (default: {DEFAULT_MIN_EDGES})")
    ap.add_argument("--top_k_layers", type=int, default=DEFAULT_TOP_K,
                   help=f"Keep top K largest layers per model (default: {DEFAULT_TOP_K})")
    ap.add_argument("--exclude_predicates", default=",".join(DEFAULT_EXCLUDE),
                   help=f"Comma-separated list of predicates to exclude (default: {','.join(DEFAULT_EXCLUDE)})")
    
    args = ap.parse_args()
    
    # Ensure output directory exists
    ensure_dir(args.out_dir)
    
    # Parse exclude predicates
    exclude_preds = [p.strip() for p in args.exclude_predicates.split(",")]
    
    print(f"[INFO] Computing NetLSD structural signatures...")
    print(f"[INFO] Models directory: {args.models_dir}")
    print(f"[INFO] Output directory: {args.out_dir}")
    print(f"[INFO] Time points: {args.times}")
    print(f"[INFO] Aggregation: {args.aggregate}")
    print(f"[INFO] Min edges per layer: {args.min_edges_per_layer}")
    print(f"[INFO] Top K layers: {args.top_k_layers}")
    print(f"[INFO] Excluded predicates: {exclude_preds}")
    
    # Generate log-spaced time points
    times = np.logspace(-2, 2, num=args.times)
    print(f"[INFO] Time range: {times[0]:.3f} to {times[-1]:.3f}")
    
    # Find RDF model files
    rdf_files = glob.glob(os.path.join(args.models_dir, "*.rdf"))
    if not rdf_files:
        raise SystemExit(f"[ERROR] No RDF files found in {args.models_dir}")
    
    print(f"[INFO] Found {len(rdf_files)} RDF models")
    
    # Process each model
    model_signatures = {}
    model_metadata = {}
    
    for rdf_file in rdf_files:
        model_name = os.path.splitext(os.path.basename(rdf_file))[0]
        print(f"[INFO] Processing {model_name}...")
        
        # Load RDF model
        graph = load_rdf_model(rdf_file)
        if len(graph) == 0:
            print(f"[WARN] Empty graph for {model_name}, skipping")
            continue
        
        # Extract predicate layers
        layers = extract_predicate_layers(graph, exclude_preds)
        print(f"[INFO] {model_name}: {len(layers)} predicate layers extracted")
        
        # Filter layers
        filtered_layers = filter_layers(layers, args.min_edges_per_layer, args.top_k_layers)
        print(f"[INFO] {model_name}: {len(filtered_layers)} layers after filtering")
        
        # Compute NetLSD signatures for each layer
        layer_signatures = {}
        for layer_name, layer_graph in filtered_layers.items():
            signature = compute_netlsd_signature(layer_graph, times)
            layer_signatures[layer_name] = signature
        
        # Aggregate layer signatures
        if layer_signatures:
            aggregated = aggregate_layer_signatures(layer_signatures, args.aggregate)
            normalized = normalize_vector(aggregated)
            model_signatures[model_name] = normalized
            model_metadata[model_name] = {
                "n_layers": len(filtered_layers),
                "layer_names": list(filtered_layers.keys()),
                "signature_length": len(normalized)
            }
        else:
            print(f"[WARN] No valid layers for {model_name}, using zero vector")
            model_signatures[model_name] = np.zeros(args.times)
            model_metadata[model_name] = {
                "n_layers": 0,
                "layer_names": [],
                "signature_length": args.times
            }
    
    if not model_signatures:
        raise SystemExit("[ERROR] No valid model signatures computed")
    
    print(f"[INFO] Computed signatures for {len(model_signatures)} models")
    
    # Create vectors DataFrame
    vectors_df = pd.DataFrame(model_signatures).T
    vectors_df.index.name = "model"
    
    # Save vectors
    vectors_path = os.path.join(args.out_dir, "netlsd_vectors.csv")
    vectors_df.to_csv(vectors_path)
    print(f"[OK] NetLSD vectors saved: {vectors_path}")
    
    # Compute similarity matrix
    similarity_matrix = compute_cosine_similarity_matrix(vectors_df)
    
    # Save similarity matrix
    similarity_path = os.path.join(args.out_dir, "struct_similarity_netlsd.csv")
    similarity_matrix.to_csv(similarity_path)
    print(f"[OK] NetLSD similarity matrix saved: {similarity_path}")
    
    # Save metadata
    metadata = {
        "parameters": {
            "times": args.times,
            "aggregate": args.aggregate,
            "min_edges_per_layer": args.min_edges_per_layer,
            "top_k_layers": args.top_k_layers,
            "exclude_predicates": exclude_preds
        },
        "time_points": times.tolist(),
        "models": model_metadata,
        "summary": {
            "n_models": len(model_signatures),
            "signature_length": args.times,
            "aggregation_method": args.aggregate
        }
    }
    
    metadata_path = os.path.join(args.out_dir, "netlsd_meta.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print(f"[OK] Metadata saved: {metadata_path}")
    
    # Print summary statistics
    print(f"[STATS] NetLSD signature computation completed!")
    print(f"[STATS] Models processed: {len(model_signatures)}")
    print(f"[STATS] Signature length: {args.times}")
    print(f"[STATS] Similarity range: {similarity_matrix.values.min():.3f} - {similarity_matrix.values.max():.3f}")
    
    # Check L2 norms
    norms = np.linalg.norm(vectors_df.values, axis=1)
    print(f"[STATS] Vector L2 norms: min={norms.min():.3f}, max={norms.max():.3f}, mean={norms.mean():.3f}")
    
    # Acceptance checks
    print(f"\n[ACCEPTANCE] Running acceptance checks...")
    
    # Check L2 norms are approximately 1.0
    norm_check = np.allclose(norms, 1.0, atol=1e-6)
    print(f"[CHECK] L2 norms ≈ 1.0: {'PASS' if norm_check else 'FAIL'}")
    
    # Check similarity matrix properties
    off_diag = similarity_matrix.values[np.triu_indices_from(similarity_matrix.values, k=1)]
    mean_offdiag = np.mean(off_diag)
    min_offdiag = np.min(off_diag)
    max_offdiag = np.max(off_diag)
    
    print(f"[CHECK] Mean off-diagonal similarity: {mean_offdiag:.4f}")
    print(f"[CHECK] Min off-diagonal similarity: {min_offdiag:.4f}")
    print(f"[CHECK] Max off-diagonal similarity: {max_offdiag:.4f}")
    
    # Check for reasonable similarity range (not saturated)
    range_check = 0.1 <= mean_offdiag <= 0.9
    print(f"[CHECK] Similarity range reasonable: {'PASS' if range_check else 'FAIL'}")
    
    # Save acceptance summary
    acceptance_summary = {
        "netlsd_mean_offdiag": round(float(mean_offdiag), 4),
        "netlsd_min_offdiag": round(float(min_offdiag), 4),
        "netlsd_max_offdiag": round(float(max_offdiag), 4),
        "l2_norms_check": bool(norm_check),
        "similarity_range_check": bool(range_check),
        "n_models": len(model_signatures),
        "signature_length": args.times
    }
    
    summary_path = os.path.join(args.out_dir, "NETLSD_run_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(acceptance_summary, f, ensure_ascii=False, indent=2)
    print(f"[OK] Acceptance summary saved: {summary_path}")

if __name__ == "__main__":
    main()
