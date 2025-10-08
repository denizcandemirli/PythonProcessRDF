#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File: similarity_content.py
Purpose: Compute content-based similarity between RDF Design Graphs.
Input: RDF model directory path, feature extraction outputs
Output: similarity_content_cosine.csv, content similarity matrix

This script computes content similarity based on:
- Type distributions (RDF types)
- Function type histograms
- Quality type histograms

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
import math
import argparse
from collections import Counter
from itertools import combinations
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np

# Dependencies
try:
    import rdflib
    from rdflib import Graph, RDF, RDFS
except ImportError:
    raise SystemExit("rdflib is required. pip install rdflib")

# ---------------------------
# Helper Functions
# ---------------------------

def ensure_dir(path: str):
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)

def load_counter_csv(path: str, key_col: str, val_col: str) -> Counter:
    """Load CSV file as Counter dictionary."""
    if not os.path.exists(path):
        return Counter()
    try:
        df = pd.read_csv(path)
        if key_col not in df.columns or val_col not in df.columns:
            return Counter()
        return Counter(dict(zip(df[key_col], df[val_col])))
    except Exception:
        # empty/corrupted file etc.
        return Counter()

def cosine(a: Counter, b: Counter) -> float:
    """Compute cosine similarity between two Counter objects."""
    keys = set(a) | set(b)
    if not keys:
        return 0.0
    va = [a.get(k, 0) for k in keys]
    vb = [b.get(k, 0) for k in keys]
    dot = sum(x * y for x, y in zip(va, vb))
    na = math.sqrt(sum(x * x for x in va))
    nb = math.sqrt(sum(y * y for y in vb))
    return 0.0 if na == 0 or nb == 0 else dot / (na * nb)

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
# Content Similarity Computation
# ---------------------------

def compute_content_similarity(input_dir: str, output_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute content similarity between RDF models.
    
    Args:
        input_dir: Directory containing feature extraction outputs
        output_dir: Directory to save results
        
    Returns:
        Tuple of (pairwise_df, similarity_matrix)
    """
    # Find available models based on type distribution files
    type_files = glob.glob(os.path.join(input_dir, "*_type_distribution.csv"))
    if not type_files:
        raise FileNotFoundError(f"No type distribution files found in {input_dir}")
    
    models = sorted([os.path.basename(f).replace("_type_distribution.csv", "") for f in type_files])
    print(f"[INFO] Found {len(models)} models: {models}")
    
    # Load content vectors for each model
    content_vectors = {}
    for model in models:
        # Type distribution
        type_c = load_counter_csv(
            os.path.join(input_dir, f"{model}_type_distribution.csv"), 
            "type_iri", "count"
        )
        
        # Function type histogram
        func_c = load_counter_csv(
            os.path.join(input_dir, f"{model}_function_type_histogram.csv"), 
            "function_type_iri", "count"
        )
        
        # Quality type histogram
        qual_c = load_counter_csv(
            os.path.join(input_dir, f"{model}_quality_type_histogram.csv"), 
            "quality_type_iri", "count"
        )
        
        # Combine all content features
        combined = Counter()
        combined.update(type_c)
        combined.update(func_c)
        combined.update(qual_c)
        content_vectors[model] = combined
        
        print(f"[INFO] Model {model}: {len(combined)} unique content features")
    
    # Compute pairwise similarities
    rows = []
    for a, b in combinations(models, 2):
        similarity = cosine(content_vectors[a], content_vectors[b])
        rows.append({
            "model_a": a,
            "model_b": b,
            "content_cosine": similarity
        })
    
    pairwise_df = pd.DataFrame(rows)
    
    # Build similarity matrix
    similarity_matrix = build_similarity_matrix(pairwise_df, "content_cosine")
    
    return pairwise_df, similarity_matrix

# ---------------------------
# Main Function
# ---------------------------

def main():
    """Main execution function."""
    ap = argparse.ArgumentParser(description="Compute content similarity between RDF models")
    ap.add_argument("--input-dir", 
                   default=os.path.join(".", "repro_pack", "output", "Building_Information"),
                   help="Directory containing feature extraction outputs")
    ap.add_argument("--output-dir", 
                   default=os.path.join(".", "repro_pack", "output", "02 - Similarity"),
                   help="Directory to save similarity results")
    ap.add_argument("--pattern", default="*_type_distribution.csv",
                   help="Pattern to find type distribution files")
    
    args = ap.parse_args()
    
    # Ensure output directory exists
    ensure_dir(args.output_dir)
    
    print(f"[INFO] Computing content similarity...")
    print(f"[INFO] Input directory: {args.input_dir}")
    print(f"[INFO] Output directory: {args.output_dir}")
    
    try:
        # Compute content similarity
        pairwise_df, similarity_matrix = compute_content_similarity(args.input_dir, args.output_dir)
        
        # Save results
        pairwise_path = os.path.join(args.output_dir, "similarity_content_cosine.csv")
        matrix_path = os.path.join(args.output_dir, "content_similarity_matrix.csv")
        
        pairwise_df.to_csv(pairwise_path, index=False)
        similarity_matrix.to_csv(matrix_path)
        
        print(f"[OK] Content similarity computed successfully!")
        print(f"[OK] Pairwise results: {pairwise_path}")
        print(f"[OK] Similarity matrix: {matrix_path}")
        print(f"[OK] Found {len(pairwise_df)} model pairs")
        
        # Print summary statistics
        similarities = pairwise_df["content_cosine"].values
        print(f"[STATS] Similarity range: {similarities.min():.3f} - {similarities.max():.3f}")
        print(f"[STATS] Mean similarity: {similarities.mean():.3f}")
        print(f"[STATS] Std deviation: {similarities.std():.3f}")
        
    except Exception as e:
        print(f"[ERROR] Failed to compute content similarity: {e}")
        raise

if __name__ == "__main__":
    main()
