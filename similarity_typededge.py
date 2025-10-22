#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File: similarity_typededge.py
Purpose: Compute typed-edge similarity between RDF Design Graphs.
Input: RDF model directory path, typed edge profiles
Output: similarity_typed_edge_cosine.csv, typed edge similarity matrix

This script computes typed-edge similarity based on:
- Subject-predicate-object triple patterns
- Cosine similarity on typed edge profiles
- Jaccard similarity on edge pattern sets

Author: Deniz Demirli
Supervisor: Dr. Chao Li (TUM)
Version: 2025.10
License: SPDX-License-Identifier: CC-BY-4.0
"""

__version__ = "2025.10"
__author__ = "Deniz Demirli"
__supervisor__ = "Dr. Chao Li (TUM)"

import os
import glob
import math
import argparse
from collections import Counter
from itertools import combinations
from typing import Dict, List, Tuple, Set
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
# Typed Edge Similarity Computation
# ---------------------------

def compute_typed_edge_similarity(input_dir: str, output_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Compute typed-edge similarity between RDF models.
    
    Args:
        input_dir: Directory containing typed edge profiles
        output_dir: Directory to save results
        
    Returns:
        Tuple of (cosine_df, jaccard_df, cosine_matrix)
    """
    # Find available models based on typed edge profile files
    profile_files = glob.glob(os.path.join(input_dir, "*_typed_edge_profile.csv"))
    if not profile_files:
        raise FileNotFoundError(f"No typed edge profile files found in {input_dir}")
    
    models = sorted([os.path.basename(f).replace("_typed_edge_profile.csv", "") for f in profile_files])
    print(f"[INFO] Found {len(models)} models: {models}")
    
    # Load typed edge profiles for each model
    typed_profiles = {}
    typed_keysets = {}
    
    for model in models:
        path = os.path.join(input_dir, f"{model}_typed_edge_profile.csv")
        profile_counter = Counter()
        keys = set()
        
        if os.path.exists(path):
            try:
                df = pd.read_csv(path)
                required_cols = ["subject_type", "predicate_iri", "object_type", "count"]
                if set(required_cols).issubset(df.columns):
                    # Create counter with (subject_type, predicate_iri, object_type) as keys
                    for _, row in df.iterrows():
                        key = (row["subject_type"], row["predicate_iri"], row["object_type"])
                        profile_counter[key] = int(row["count"])
                    keys = set(profile_counter.keys())
                else:
                    print(f"[WARN] Missing required columns in {path}")
            except Exception as e:
                print(f"[WARN] Error reading {path}: {e}")
        else:
            print(f"[WARN] File not found: {path}")
        
        typed_profiles[model] = profile_counter
        typed_keysets[model] = keys
        
        print(f"[INFO] Model {model}: {len(profile_counter)} typed edge patterns")
    
    # Compute pairwise similarities
    cosine_rows = []
    jaccard_rows = []
    
    for a, b in combinations(models, 2):
        # Cosine similarity on typed edge profiles
        cosine_sim = cosine(typed_profiles[a], typed_profiles[b])
        cosine_rows.append({
            "model_a": a,
            "model_b": b,
            "typed_edge_cosine": cosine_sim
        })
        
        # Jaccard similarity on edge pattern sets
        jaccard_sim = set_jaccard(typed_keysets[a], typed_keysets[b])
        jaccard_rows.append({
            "model_a": a,
            "model_b": b,
            "typed_edge_jaccard": jaccard_sim
        })
    
    cosine_df = pd.DataFrame(cosine_rows)
    jaccard_df = pd.DataFrame(jaccard_rows)
    
    # Build similarity matrix (using cosine as primary metric)
    cosine_matrix = build_similarity_matrix(cosine_df, "typed_edge_cosine")
    
    return cosine_df, jaccard_df, cosine_matrix

# ---------------------------
# Main Function
# ---------------------------

def main():
    """Main execution function."""
    ap = argparse.ArgumentParser(description="Compute typed-edge similarity between RDF models")
    ap.add_argument("--input-dir", 
                   default=os.path.join(".", "repro_pack", "output", "Building_Information"),
                   help="Directory containing typed edge profiles")
    ap.add_argument("--output-dir", 
                   default=os.path.join(".", "repro_pack", "output", "02 - Similarity"),
                   help="Directory to save similarity results")
    ap.add_argument("--pattern", default="*_typed_edge_profile.csv",
                   help="Pattern to find typed edge profile files")
    
    args = ap.parse_args()
    
    # Ensure output directory exists
    ensure_dir(args.output_dir)
    
    print(f"[INFO] Computing typed-edge similarity...")
    print(f"[INFO] Input directory: {args.input_dir}")
    print(f"[INFO] Output directory: {args.output_dir}")
    
    try:
        # Compute typed-edge similarity
        cosine_df, jaccard_df, cosine_matrix = compute_typed_edge_similarity(args.input_dir, args.output_dir)
        
        # Save results
        cosine_path = os.path.join(args.output_dir, "similarity_typed_edge_cosine.csv")
        jaccard_path = os.path.join(args.output_dir, "similarity_typed_edge_jaccard.csv")
        matrix_path = os.path.join(args.output_dir, "typed_edge_similarity_matrix.csv")
        
        cosine_df.to_csv(cosine_path, index=False)
        jaccard_df.to_csv(jaccard_path, index=False)
        cosine_matrix.to_csv(matrix_path)
        
        print(f"[OK] Typed-edge similarity computed successfully!")
        print(f"[OK] Cosine results: {cosine_path}")
        print(f"[OK] Jaccard results: {jaccard_path}")
        print(f"[OK] Similarity matrix: {matrix_path}")
        print(f"[OK] Found {len(cosine_df)} model pairs")
        
        # Print summary statistics
        cosine_sims = cosine_df["typed_edge_cosine"].values
        jaccard_sims = jaccard_df["typed_edge_jaccard"].values
        
        print(f"[STATS] Cosine similarity range: {cosine_sims.min():.3f} - {cosine_sims.max():.3f}")
        print(f"[STATS] Cosine mean: {cosine_sims.mean():.3f}, std: {cosine_sims.std():.3f}")
        print(f"[STATS] Jaccard similarity range: {jaccard_sims.min():.3f} - {jaccard_sims.max():.3f}")
        print(f"[STATS] Jaccard mean: {jaccard_sims.mean():.3f}, std: {jaccard_sims.std():.3f}")
        
    except Exception as e:
        print(f"[ERROR] Failed to compute typed-edge similarity: {e}")
        raise

if __name__ == "__main__":
    main()
