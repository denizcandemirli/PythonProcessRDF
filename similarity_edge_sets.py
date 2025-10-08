#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File: similarity_edge_sets.py
Purpose: Compute edge-set similarity between RDF Design Graphs.
Input: RDF model directory path, edge set files
Output: similarity_edge_sets_jaccard.csv, edge set similarity matrix

This script computes edge-set similarity based on:
- Jaccard similarity on edge sets for specific predicates
- Combined edge set similarity
- Support for adjacentElement, adjacentZone, intersectingElement, hasContinuantPart

Author: Deniz [Your Surname]
Supervisor: Dr. Chao Li (TUM)
Version: 2025.10
License: SPDX-License-Identifier: CC-BY-4.0
"""

__version__ = "2025.10"
__author__ = "Deniz [Your Surname]"
__supervisor__ = "Dr. Chao Li (TUM)"

import os
import glob
import argparse
from itertools import combinations
from typing import Dict, List, Tuple, Set
import pandas as pd
import numpy as np

# ---------------------------
# Helper Functions
# ---------------------------

def ensure_dir(path: str):
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)

def set_jaccard(A: Set, B: Set) -> float:
    """Compute Jaccard similarity between two sets."""
    union = A | B
    return (len(A & B) / len(union)) if union else 0.0

def build_similarity_matrix(pairwise_df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """Build symmetric similarity matrix from pairwise data."""
    models = sorted(set(pairwise_df["model_a"]).union(set(pairwise_df["model_b"])))
    n = len(models)
    matrix = np.eye(n)  # diagonal = 1.0
    
    # Fill matrix from pairwise data
    for _, row in pairwise_df.iterrows():
        a, b = row["model_a"], row["model_b"]
        i, j = models.index(a), models.index(b)
        similarity = float(row[value_col])
        matrix[i, j] = similarity
        matrix[j, i] = similarity
    
    return pd.DataFrame(matrix, index=models, columns=models)

# ---------------------------
# Edge Set Loading
# ---------------------------

def load_edge_set(file_path: str) -> Set[Tuple[str, str, str]]:
    """
    Load edge set from CSV file.
    
    Args:
        file_path: Path to edge CSV file
        
    Returns:
        Set of (subject, predicate, object) tuples
    """
    if not os.path.exists(file_path):
        return set()
    
    try:
        df = pd.read_csv(file_path)
        required_cols = ["subject", "predicate", "object"]
        if not set(required_cols).issubset(df.columns):
            print(f"[WARN] Missing required columns in {file_path}")
            return set()
        
        edge_set = set()
        for _, row in df.iterrows():
            edge = (str(row["subject"]), str(row["predicate"]), str(row["object"]))
            edge_set.add(edge)
        
        return edge_set
    except Exception as e:
        print(f"[WARN] Error reading {file_path}: {e}")
        return set()

# ---------------------------
# Edge Set Similarity Computation
# ---------------------------

def compute_edge_set_similarity(input_dir: str, output_dir: str, 
                               edge_labels: List[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute edge-set similarity between RDF models.
    
    Args:
        input_dir: Directory containing edge set files
        output_dir: Directory to save results
        edge_labels: List of edge types to analyze
        
    Returns:
        Tuple of (pairwise_df, similarity_matrix)
    """
    if edge_labels is None:
        edge_labels = ["adjacentElement", "adjacentZone", "intersectingElement", "hasContinuantPart"]
    
    # Find available models based on edge files
    edge_files = []
    for label in edge_labels:
        pattern = os.path.join(input_dir, f"*_{label}_edges.csv")
        edge_files.extend(glob.glob(pattern))
    
    if not edge_files:
        raise FileNotFoundError(f"No edge files found in {input_dir}")
    
    # Extract model names from file paths
    models = set()
    for file_path in edge_files:
        filename = os.path.basename(file_path)
        for label in edge_labels:
            if filename.endswith(f"_{label}_edges.csv"):
                model_name = filename.replace(f"_{label}_edges.csv", "")
                models.add(model_name)
                break
    
    models = sorted(list(models))
    print(f"[INFO] Found {len(models)} models: {models}")
    
    # Load edge sets for each model and each edge type
    edge_sets = {}
    for model in models:
        edge_sets[model] = {}
        for label in edge_labels:
            file_path = os.path.join(input_dir, f"{model}_{label}_edges.csv")
            edge_set = load_edge_set(file_path)
            edge_sets[model][label] = edge_set
            print(f"[INFO] Model {model}, {label}: {len(edge_set)} edges")
        
        # Create combined edge set
        combined = set()
        for label in edge_labels:
            combined |= edge_sets[model][label]
        edge_sets[model]["__combined__"] = combined
        print(f"[INFO] Model {model}, combined: {len(combined)} edges")
    
    # Compute pairwise similarities
    rows = []
    for a, b in combinations(models, 2):
        row = {"model_a": a, "model_b": b}
        
        # Compute Jaccard similarity for each edge type
        for label in edge_labels + ["__combined__"]:
            jaccard_sim = set_jaccard(edge_sets[a][label], edge_sets[b][label])
            row[f"jaccard_{label}"] = jaccard_sim
        
        rows.append(row)
    
    pairwise_df = pd.DataFrame(rows)
    
    # Build similarity matrix (using combined Jaccard as primary metric)
    similarity_matrix = build_similarity_matrix(pairwise_df, "jaccard___combined__")
    
    return pairwise_df, similarity_matrix

# ---------------------------
# Main Function
# ---------------------------

def main():
    """Main execution function."""
    ap = argparse.ArgumentParser(description="Compute edge-set similarity between RDF models")
    ap.add_argument("--input-dir", 
                   default=os.path.join(".", "repro_pack", "output", "Building_Information"),
                   help="Directory containing edge set files")
    ap.add_argument("--output-dir", 
                   default=os.path.join(".", "repro_pack", "output", "02 - Similarity"),
                   help="Directory to save similarity results")
    ap.add_argument("--edge-labels", nargs="+", 
                   default=["adjacentElement", "adjacentZone", "intersectingElement", "hasContinuantPart"],
                   help="Edge types to analyze")
    
    args = ap.parse_args()
    
    # Ensure output directory exists
    ensure_dir(args.output_dir)
    
    print(f"[INFO] Computing edge-set similarity...")
    print(f"[INFO] Input directory: {args.input_dir}")
    print(f"[INFO] Output directory: {args.output_dir}")
    print(f"[INFO] Edge labels: {args.edge_labels}")
    
    try:
        # Compute edge-set similarity
        pairwise_df, similarity_matrix = compute_edge_set_similarity(
            args.input_dir, args.output_dir, args.edge_labels
        )
        
        # Save results
        pairwise_path = os.path.join(args.output_dir, "similarity_edge_sets_jaccard.csv")
        matrix_path = os.path.join(args.output_dir, "edge_sets_similarity_matrix.csv")
        
        pairwise_df.to_csv(pairwise_path, index=False)
        similarity_matrix.to_csv(matrix_path)
        
        print(f"[OK] Edge-set similarity computed successfully!")
        print(f"[OK] Pairwise results: {pairwise_path}")
        print(f"[OK] Similarity matrix: {matrix_path}")
        print(f"[OK] Found {len(pairwise_df)} model pairs")
        
        # Print summary statistics for each edge type
        for label in args.edge_labels + ["__combined__"]:
            col_name = f"jaccard_{label}"
            if col_name in pairwise_df.columns:
                similarities = pairwise_df[col_name].values
                print(f"[STATS] {label} Jaccard range: {similarities.min():.3f} - {similarities.max():.3f}")
                print(f"[STATS] {label} mean: {similarities.mean():.3f}, std: {similarities.std():.3f}")
        
    except Exception as e:
        print(f"[ERROR] Failed to compute edge-set similarity: {e}")
        raise

if __name__ == "__main__":
    main()
