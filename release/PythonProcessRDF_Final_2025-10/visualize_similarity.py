#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File: visualize_similarity.py
Purpose: Generate visualizations for similarity analysis results.
Input: Similarity matrices and pairwise data from the pipeline
Output: Heatmaps, dendrograms, contribution bars, radar plots

This script creates comprehensive visualizations:
- Total similarity heatmap and dendrogram
- Component contribution bars
- Structural motif radar plots
- Individual similarity matrix heatmaps

Author: Deniz [Your Surname]
Supervisor: Dr. Chao Li (TUM)
Version: 2025.10
License: SPDX-License-Identifier: CC-BY-4.0
"""

__version__ = "2025.10"
__author__ = "Deniz [Your Surname]"
__supervisor__ = "Dr. Chao Li (TUM)"

import os
import argparse
import json
from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Dependencies
try:
    from scipy.cluster.hierarchy import linkage, dendrogram
    from scipy.spatial.distance import squareform
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("[WARN] scipy not available, dendrograms will use fallback implementation")

# ---------------------------
# Configuration
# ---------------------------

# Default fusion weights (used if weights file not found)
FUSION_WEIGHTS_DEFAULT = {
    "w_content": 0.30,
    "w_typed": 0.20,
    "w_edge": 0.10,
    "w_struct": 0.40
}

# ---------------------------
# Helper Functions
# ---------------------------

def ensure_dir(path: str):
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)

def read_square_matrix(path: str) -> pd.DataFrame:
    """Read square similarity matrix from CSV."""
    df = pd.read_csv(path, index_col=0)
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.reindex(index=df.index, columns=df.index)
    M = df.values.astype(float)
    M = (M + M.T) / 2.0
    np.fill_diagonal(M, 1.0)
    M = np.clip(M, 0.0, 1.0)
    return pd.DataFrame(M, index=df.index, columns=df.columns)

def load_weights_from_json(weights_path: str) -> Dict[str, float]:
    """Load fusion weights from JSON file."""
    if not os.path.exists(weights_path):
        return FUSION_WEIGHTS_DEFAULT
    
    try:
        with open(weights_path, "r") as f:
            data = json.load(f)
        # Handle nested structure
        weights = data.get("weights_normalized", data)
        return {
            "w_content": float(weights.get("w_content", FUSION_WEIGHTS_DEFAULT["w_content"])),
            "w_typed": float(weights.get("w_typed", FUSION_WEIGHTS_DEFAULT["w_typed"])),
            "w_edge": float(weights.get("w_edge", FUSION_WEIGHTS_DEFAULT["w_edge"])),
            "w_struct": float(weights.get("w_struct", FUSION_WEIGHTS_DEFAULT["w_struct"])),
        }
    except Exception as e:
        print(f"[WARN] Could not load weights from {weights_path}: {e}")
        return FUSION_WEIGHTS_DEFAULT

# ---------------------------
# Visualization Functions
# ---------------------------

def create_heatmap(matrix: pd.DataFrame, title: str, output_path: str, 
                  vmin: float = 0.0, vmax: float = 1.0):
    """Create similarity heatmap."""
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(matrix.values, vmin=vmin, vmax=vmax, aspect="equal", cmap="viridis")
    
    ax.set_xticks(np.arange(matrix.shape[1]))
    ax.set_yticks(np.arange(matrix.shape[0]))
    ax.set_xticklabels(matrix.columns, rotation=45, ha="right")
    ax.set_yticklabels(matrix.index)
    ax.set_title(title)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Similarity", rotation=90)
    
    # Add text annotations
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            text = ax.text(j, i, f"{matrix.iloc[i, j]:.2f}",
                          ha="center", va="center", color="white" if matrix.iloc[i, j] < 0.5 else "black")
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[OK] Heatmap saved: {output_path}")

def create_dendrogram(matrix: pd.DataFrame, title: str, output_path: str):
    """Create hierarchical clustering dendrogram."""
    # Convert similarity to distance
    D = 1.0 - np.clip(matrix.values.astype(float), 0.0, 1.0)
    np.fill_diagonal(D, 0.0)

    # Handle case where all similarities are 1.0
    if np.allclose(D, 0.0):
        D = D + np.eye(D.shape[0]) * 1e-9
        np.fill_diagonal(D, 0.0)
    
    if SCIPY_AVAILABLE:
        # Use scipy for proper dendrogram
        cond = squareform(D, checks=False)
        Z = linkage(cond, method="average")
        
        fig, ax = plt.subplots(figsize=(max(8, 0.6 * len(matrix.index)), 6))
        dendrogram(Z, labels=list(matrix.index), leaf_rotation=90, ax=ax)
        ax.set_title(title)
        ax.set_ylabel("Distance (1 - similarity)")
    else:
        # Fallback implementation
        fig, ax = plt.subplots(figsize=(max(8, 0.6 * len(matrix.index)), 6))
        ax.text(0.5, 0.5, "Dendrogram requires scipy\nInstall with: pip install scipy", 
                ha="center", va="center", transform=ax.transAxes, fontsize=12)
        ax.set_title(title)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[OK] Dendrogram saved: {output_path}")

def create_component_bars(pairwise_path: str, weights: Dict[str, float], 
                         output_dir: str, top_n: int = 3):
    """Create component contribution bar charts."""
    df = pd.read_csv(pairwise_path)
    
    # Normalize column names
    col_mapping = {}
    for col in df.columns:
        col_lower = col.lower()
        if "content" in col_lower and "cos" in col_lower:
            col_mapping[col] = "S_content"
        elif "typed" in col_lower and "cos" in col_lower:
            col_mapping[col] = "S_typed"
        elif "edge" in col_lower and "jaccard" in col_lower:
            col_mapping[col] = "S_edge"
        elif "struct" in col_lower or "structural" in col_lower:
            col_mapping[col] = "S_struct"
        elif "total" in col_lower:
            col_mapping[col] = "S_total"
    
    df = df.rename(columns=col_mapping)
    
    # Get top N pairs by total similarity
    if "S_total" in df.columns:
        top_pairs = df.nlargest(top_n, "S_total")
    else:
        # Estimate total from components
        df["S_total_est"] = (
            weights["w_content"] * df.get("S_content", 0) +
            weights["w_typed"] * df.get("S_typed", 0) +
            weights["w_edge"] * df.get("S_edge", 0) +
            weights["w_struct"] * df.get("S_struct", 0)
        )
        top_pairs = df.nlargest(top_n, "S_total_est")
    
    # Create bar chart for each top pair
    for idx, (_, row) in enumerate(top_pairs.iterrows()):
        model_a = row.get("model_A", row.get("A", f"Model_{idx}_A"))
        model_b = row.get("model_B", row.get("B", f"Model_{idx}_B"))
        
        # Calculate weighted contributions
        components = {
            "Content": weights["w_content"] * row.get("S_content", 0),
            "Typed-Edge": weights["w_typed"] * row.get("S_typed", 0),
            "Edge-Set": weights["w_edge"] * row.get("S_edge", 0),
            "Structural": weights["w_struct"] * row.get("S_struct", 0)
        }
        
        total = sum(components.values())
        if total > 0:
            # Normalize to percentages
            components = {k: (v / total) * 100 for k, v in components.items()}
        
        # Create bar chart
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(components.keys(), components.values(), 
                     color=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"])
        
        ax.set_ylabel("Contribution (%)")
        ax.set_title(f"Component Contributions: {model_a} vs {model_b}")
        ax.set_ylim(0, 100)
        
        # Add value labels on bars
        for bar, value in zip(bars, components.values()):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                   f"{value:.1f}%", ha="center", va="bottom")
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        output_path = os.path.join(output_dir, f"component_contrib_{model_a}_vs_{model_b}.png")
        plt.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"[OK] Component bars saved: {output_path}")

def create_radar_plot(data_path: str, output_path: str):
    """Create radar plot for structural motif data."""
    try:
        df = pd.read_csv(data_path)
        
        # Check if this is motif shares data
        if "motif" in df.columns and "share" in df.columns:
            # Pivot to wide format
            pivot_df = df.pivot(index="model", columns="motif", values="share").fillna(0)
        else:
            # Assume it's already in wide format
            pivot_df = df.set_index("model") if "model" in df.columns else df
        
        # Create radar plot
        fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(projection='polar'))
        
        # Get motif categories
        motifs = [col for col in pivot_df.columns if col.startswith("M")]
        angles = np.linspace(0, 2 * np.pi, len(motifs), endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle
        
        # Plot each model
        colors = plt.cm.Set3(np.linspace(0, 1, len(pivot_df)))
        for idx, (model, row) in enumerate(pivot_df.iterrows()):
            values = [row[motif] for motif in motifs]
            values += values[:1]  # Complete the circle
            
            ax.plot(angles, values, 'o-', linewidth=2, label=model, color=colors[idx])
            ax.fill(angles, values, alpha=0.25, color=colors[idx])
        
        # Customize plot
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(motifs)
        ax.set_ylim(0, 1)
        ax.set_title("Structural Motif Similarity Radar Plot", size=16, pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        ax.grid(True)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"[OK] Radar plot saved: {output_path}")
        
    except Exception as e:
        print(f"[WARN] Could not create radar plot: {e}")

# ---------------------------
# Main Function
# ---------------------------

def main():
    """Main execution function."""
    ap = argparse.ArgumentParser(description="Generate similarity visualizations")
    ap.add_argument("--input-dir", required=True,
                   help="Directory containing similarity results")
    ap.add_argument("--output-dir", required=True,
                   help="Directory to save visualizations")
    ap.add_argument("--total-matrix", default="total_similarity_matrix.csv",
                   help="Total similarity matrix filename")
    ap.add_argument("--pairwise-summary", default="pairwise_total_summary.csv",
                   help="Pairwise summary filename")
    ap.add_argument("--weights-file", default="weights_used.json",
                   help="Weights configuration filename")
    ap.add_argument("--motif-shares", default="struct_motif_shares.csv",
                   help="Structural motif shares filename")
    ap.add_argument("--top-pairs", type=int, default=3,
                   help="Number of top pairs for component analysis")
    ap.add_argument("--include-radar", action="store_true",
                   help="Include radar plot for structural motifs")
    
    args = ap.parse_args()
    
    # Ensure output directory exists
    ensure_dir(args.output_dir)
    
    print(f"[INFO] Generating similarity visualizations...")
    print(f"[INFO] Input directory: {args.input_dir}")
    print(f"[INFO] Output directory: {args.output_dir}")
    
    try:
        # Load weights
        weights_path = os.path.join(args.input_dir, args.weights_file)
        weights = load_weights_from_json(weights_path)
        print(f"[INFO] Using weights: {weights}")
        
        # Create total similarity heatmap and dendrogram
        total_matrix_path = os.path.join(args.input_dir, args.total_matrix)
        if os.path.exists(total_matrix_path):
            matrix = read_square_matrix(total_matrix_path)
            
            # Heatmap
            heatmap_path = os.path.join(args.output_dir, "total_similarity_heatmap.png")
            create_heatmap(matrix, "Total Similarity Heatmap", heatmap_path)
            
            # Dendrogram
            dendro_path = os.path.join(args.output_dir, "total_similarity_dendrogram.png")
            create_dendrogram(matrix, "Hierarchical Clustering (Total Similarity)", dendro_path)
        else:
            print(f"[WARN] Total similarity matrix not found: {total_matrix_path}")
        
        # Create component contribution bars
        pairwise_path = os.path.join(args.input_dir, args.pairwise_summary)
        if os.path.exists(pairwise_path):
            create_component_bars(pairwise_path, weights, args.output_dir, args.top_pairs)
        else:
            print(f"[WARN] Pairwise summary not found: {pairwise_path}")
        
        # Create radar plot for structural motifs
        if args.include_radar:
            motif_path = os.path.join(args.input_dir, args.motif_shares)
            if os.path.exists(motif_path):
                radar_path = os.path.join(args.output_dir, "structural_motif_radar.png")
                create_radar_plot(motif_path, radar_path)
            else:
                print(f"[WARN] Motif shares file not found: {motif_path}")
        
        print(f"[OK] Visualizations completed successfully!")
        print(f"[OK] Output directory: {args.output_dir}")
        
    except Exception as e:
        print(f"[ERROR] Failed to generate visualizations: {e}")
        raise

if __name__ == "__main__":
    main()
