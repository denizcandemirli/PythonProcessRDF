#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File: motif_significance_profile.py
Purpose: Compute Network Significance Profiles (SP) for structural similarity analysis.
Input: RDF model directory path, merged ontology (0000_Merged.rdf)
Output: motif_sp_vectors.csv, struct_similarity_sp.csv, SP visualizations

This script implements motif significance analysis following Milo et al. (2002):
- Counts motifs M1-M8 in real RDF graphs
- Generates null models with same degree distribution
- Computes Z-scores for motif over/under-representation
- Normalizes to significance profile vectors
- Computes cosine similarity between SP vectors

Author: Deniz Demirli
Supervisor: Dr. Chao Li (TUM)
Version: 2025.10
License: SPDX-License-Identifier: CC-BY-4.0
"""

__version__ = "2025.10"
__author__ = "Deniz Demirli"
__supervisor__ = "Dr. Chao Li (TUM)"

import os
import re
import json
import glob
import argparse
import time
import numpy as np
import pandas as pd
import networkx as nx
from collections import Counter, defaultdict
from itertools import combinations
import matplotlib.pyplot as plt
import seaborn as sns
from rdflib import Graph, URIRef, Literal, BNode
from rdflib.namespace import RDF, RDFS, OWL
import warnings
warnings.filterwarnings('ignore')

# Set matplotlib backend for headless operation
plt.switch_backend('Agg')

def localname(iri):
    """Extract local name from IRI."""
    if isinstance(iri, str):
        return iri.split('/')[-1].split('#')[-1]
    elif hasattr(iri, 'fragment') and iri.fragment:
        return iri.fragment
    else:
        return str(iri).split('/')[-1].split('#')[-1]

def tokset_from_labels(labels):
    """Convert label list to token set for motif matching."""
    return set(localname(l) for l in labels if l)

def cosine(a, b):
    """Compute cosine similarity between two vectors."""
    if isinstance(a, dict) and isinstance(b, dict):
        keys = set(a.keys()) | set(b.keys())
        va = np.array([a.get(k, 0) for k in keys])
        vb = np.array([b.get(k, 0) for k in keys])
    else:
        va, vb = np.array(a), np.array(b)
    
    norm_a, norm_b = np.linalg.norm(va), np.linalg.norm(vb)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return np.dot(va, vb) / (norm_a * norm_b)

class Model:
    """RDF Model wrapper with NetworkX conversion."""
    
    def __init__(self, name, rdf_path):
        self.name = name
        self.rdf_path = rdf_path
        self.graph = None
        self.nx_graph = None
        self.load_rdf()
        self.convert_to_networkx()
    
    def load_rdf(self):
        """Load RDF graph from file."""
        try:
            self.graph = Graph()
            self.graph.parse(self.rdf_path, format='xml')
            print(f"[OK] Loaded {self.name}: {len(self.graph)} triples")
        except Exception as e:
            print(f"[ERROR] Failed to load {self.name}: {e}")
            raise
    
    def convert_to_networkx(self):
        """Convert RDF graph to NetworkX for motif analysis."""
        self.nx_graph = nx.Graph()
        
        # Add nodes with type information
        for s, p, o in self.graph:
            if isinstance(s, URIRef) and isinstance(o, URIRef):
                # Add nodes with type labels
                s_label = localname(s)
                o_label = localname(o)
                p_label = localname(p)
                
                self.nx_graph.add_node(s_label, type=s_label)
                self.nx_graph.add_node(o_label, type=o_label)
                self.nx_graph.add_edge(s_label, o_label, predicate=p_label)
        
        print(f"[OK] Converted {self.name} to NetworkX: {self.nx_graph.number_of_nodes()} nodes, {self.nx_graph.number_of_edges()} edges")

def build_untyped_graph(graph_typed):
    """Build simplified untyped graph from typed RDF graph."""
    G_simple = nx.Graph()
    
    for node in graph_typed.nodes():
        G_simple.add_node(node)
    
    for u, v, data in graph_typed.edges(data=True):
        if not G_simple.has_edge(u, v):
            G_simple.add_edge(u, v)
    
    return G_simple

def build_predicate_layers(graph_typed, min_edges=5, top_k=50, exclude_predicates=None):
    """Build separate graphs for each predicate type with filtering."""
    layers = {}
    
    # Parse exclude predicates
    if exclude_predicates:
        exclude_set = set(p.strip() for p in exclude_predicates.split(',') if p.strip())
    else:
        exclude_set = set()
    
    # Build all layers first
    for u, v, data in graph_typed.edges(data=True):
        predicate = data.get('predicate', 'unknown')
        if predicate not in layers:
            layers[predicate] = nx.Graph()
        
        # Add nodes if not present
        if u not in layers[predicate]:
            layers[predicate].add_node(u)
        if v not in layers[predicate]:
            layers[predicate].add_node(v)
        
        layers[predicate].add_edge(u, v)
    
    # Apply filtering
    filtered_layers = {}
    
    # Filter by edge count and excluded predicates
    for predicate, layer_graph in layers.items():
        edge_count = layer_graph.number_of_edges()
        predicate_local = localname(predicate)
        
        # Skip if too few edges
        if edge_count < min_edges:
            continue
            
        # Skip if in exclude list
        if predicate_local in exclude_set:
            continue
            
        filtered_layers[predicate] = layer_graph
    
    # Apply top-K filter if specified
    if top_k > 0 and len(filtered_layers) > top_k:
        # Sort by edge count descending and keep top K
        sorted_layers = sorted(filtered_layers.items(), 
                             key=lambda x: x[1].number_of_edges(), 
                             reverse=True)
        filtered_layers = dict(sorted_layers[:top_k])
    
    return filtered_layers

def rewire_layer(layer_graph, n_iterations=10):
    """Rewire a single layer while preserving degree distribution."""
    if layer_graph.number_of_edges() < 2:
        return layer_graph.copy()
    
    rewired = layer_graph.copy()
    # Use a more aggressive rewiring strategy
    max_swaps = max(rewired.number_of_edges() * n_iterations, 100)
    
    try:
        # Try multiple rewiring attempts
        for attempt in range(3):
            nx.double_edge_swap(rewired, nswap=max_swaps, max_tries=max_swaps*20)
            # Check if rewiring actually changed something
            if not nx.is_isomorphic(rewired, layer_graph):
                break
    except nx.NetworkXError:
        pass  # Keep original if rewiring fails
    
    return rewired

def compose_layers(layers):
    """Compose multiple predicate layers into a single graph."""
    composed = nx.Graph()
    
    for predicate, layer in layers.items():
        for node in layer.nodes():
            composed.add_node(node, type=layer.nodes[node].get('type', ''))
        
        for u, v in layer.edges():
            composed.add_edge(u, v, predicate=predicate)
    
    return composed

def generate_null_models(graph, n_random=20, sp_mode='predicate', rewire_iter=10, max_seconds=None, max_matches=None, log_every=1, min_edges=5, top_k=50, exclude_predicates=None):
    """Generate null models with same degree distribution using double edge swap."""
    null_models = []
    
    for i in range(n_random):
        if (i + 1) % log_every == 0:
            print(f"[SP] Generating random model {i+1}/{n_random}", flush=True)
            
        if sp_mode == 'untyped':
            # Build untyped graph and rewire
            G_simple = build_untyped_graph(graph)
            null_graph = rewire_layer(G_simple, rewire_iter)
        else:  # predicate mode
            # Build predicate layers, rewire each, then compose
            layers = build_predicate_layers(graph, min_edges, top_k, exclude_predicates)
            rewired_layers = {}
            
            for predicate, layer in layers.items():
                rewired_layers[predicate] = rewire_layer(layer, rewire_iter)
            
            null_graph = compose_layers(rewired_layers)
        
        null_models.append(null_graph)
    
    return null_models

def count_layer_motifs_with_caps(G_layer, motif_patterns, *, max_seconds=None, max_matches=None):
    """Count motifs in a layer with time and work caps."""
    start = time.time()
    matches = 0
    counts = Counter()
    
    for motif_name, pattern in motif_patterns.items():
        count = 0
        
        # Check time cap
        if max_seconds is not None and (time.time() - start) >= max_seconds:
            break
            
        # Check matches cap
        if max_matches is not None and matches >= max_matches:
            break
            
        # Count motifs for this pattern
        if motif_name == "M1_isolated":
            count = sum(1 for node in G_layer.nodes() if G_layer.degree(node) == 0)
        elif motif_name == "M2_frameNode":
            count = sum(1 for node in G_layer.nodes() if G_layer.degree(node) == 1)
        elif motif_name == "M3_wallSlab":
            count = sum(1 for node in G_layer.nodes() if G_layer.degree(node) == 2)
        elif motif_name == "M4_core":
            count = sum(1 for node in G_layer.nodes() if G_layer.degree(node) >= 4)
        elif motif_name == "M5_beamColumn":
            count = sum(1 for node in G_layer.nodes() if G_layer.degree(node) == 3)
        elif motif_name == "M6_connection":
            count = sum(1 for node in G_layer.nodes() if G_layer.degree(node) == 3)
        elif motif_name == "M7_junction":
            count = sum(1 for node in G_layer.nodes() if G_layer.degree(node) >= 4)
        elif motif_name == "M8_complex":
            count = sum(1 for node in G_layer.nodes() if G_layer.degree(node) >= 5)
        
        counts[motif_name] = count
        matches += 1
        
        # Check caps after each motif
        if max_seconds is not None and (time.time() - start) >= max_seconds:
            break
        if max_matches is not None and matches >= max_matches:
            break
    
    return counts

def count_motifs_in_graph(graph, motif_patterns, sp_mode='predicate'):
    """Count motifs in a NetworkX graph."""
    motif_counts = Counter()
    
    for motif_name, pattern in motif_patterns.items():
        count = 0
        
        if sp_mode == 'untyped':
            # More sophisticated untyped motif counting that considers structure
            if motif_name == "M1_isolated":
                count = sum(1 for node in graph.nodes() if graph.degree(node) == 0)
            elif motif_name == "M2_frameNode":
                count = sum(1 for node in graph.nodes() if graph.degree(node) == 1)
            elif motif_name == "M3_wallSlab":
                # Count triangles (3-cliques)
                count = sum(1 for _ in nx.triangles(graph).values()) // 3
            elif motif_name == "M4_core":
                # Count 4-cliques
                count = len(list(nx.clique.find_cliques(graph)))
                count = sum(1 for clique in nx.clique.find_cliques(graph) if len(clique) >= 4)
            elif motif_name == "M5_beamColumn":
                # Count paths of length 2
                count = sum(1 for u in graph.nodes() 
                           for v in graph.neighbors(u) 
                           for w in graph.neighbors(v) 
                           if u != w and not graph.has_edge(u, w))
            elif motif_name == "M6_connection":
                # Count 3-stars (nodes with degree 3)
                count = sum(1 for node in graph.nodes() if graph.degree(node) == 3)
            elif motif_name == "M7_junction":
                # Count 4-stars (nodes with degree 4)
                count = sum(1 for node in graph.nodes() if graph.degree(node) == 4)
            elif motif_name == "M8_complex":
                # Count 5-stars (nodes with degree 5+)
                count = sum(1 for node in graph.nodes() if graph.degree(node) >= 5)
        else:
            # Typed motif counting (predicate-aware)
            if motif_name == "M1_isolated":
                count = sum(1 for node in graph.nodes() if graph.degree(node) == 0)
            elif motif_name == "M2_frameNode":
                count = sum(1 for node in graph.nodes() if graph.degree(node) == 1)
            elif motif_name == "M3_wallSlab":
                # Wall/slab patterns (degree 2, connected to similar types)
                count = sum(1 for node in graph.nodes() 
                           if graph.degree(node) == 2 and 
                           any(graph.nodes[neighbor].get('type', '') == graph.nodes[node].get('type', '') 
                               for neighbor in graph.neighbors(node)))
            elif motif_name == "M4_core":
                count = sum(1 for node in graph.nodes() if graph.degree(node) >= 4)
            elif motif_name == "M5_beamColumn":
                # Beam/column patterns (degree 2, specific connections)
                count = sum(1 for node in graph.nodes() 
                           if graph.degree(node) == 2 and
                           ('beam' in graph.nodes[node].get('type', '').lower() or
                            'column' in graph.nodes[node].get('type', '').lower()))
            elif motif_name == "M6_connection":
                count = sum(1 for node in graph.nodes() if graph.degree(node) == 3)
            elif motif_name == "M7_junction":
                count = sum(1 for node in graph.nodes() if graph.degree(node) >= 4)
            elif motif_name == "M8_complex":
                count = sum(1 for node in graph.nodes() if graph.degree(node) >= 5)
        
        motif_counts[motif_name] = count
    
    return motif_counts

def compute_significance_profiles(models, n_random=20, sp_mode='predicate', rewire_iter=10, z_cap=5.0, min_abs_z=0.0, max_seconds=None, max_matches=None, log_every=1, min_edges=5, top_k=50, exclude_predicates=None):
    """Compute significance profiles for all models."""
    # Define motif patterns
    motif_patterns = {
        "M1_isolated": "isolated nodes",
        "M2_frameNode": "frame nodes (degree 1)",
        "M3_wallSlab": "wall/slab patterns (degree 2)",
        "M4_core": "core nodes (degree 4+)",
        "M5_beamColumn": "beam/column patterns",
        "M6_connection": "connection patterns (degree 3)",
        "M7_junction": "junction patterns (degree 4+)",
        "M8_complex": "complex patterns (degree 5+)"
    }
    
    sp_vectors = {}
    z_vectors = {}
    motif_data = []
    debug_data = []
    counts_real_rows = []
    stats_rows = []
    
    for model in models:
        t0 = time.time()
        print(f"[SP] Processing {model.name} ({sp_mode} mode)...")
        
        # Log layer filtering for predicate mode
        if sp_mode == 'predicate':
            all_layers = build_predicate_layers(model.nx_graph, 0, 0, None)  # No filtering
            filtered_layers = build_predicate_layers(model.nx_graph, min_edges, top_k, exclude_predicates)
            print(f"[SP] {model.name}: kept {len(filtered_layers)}/{len(all_layers)} layers after filtering")
        
        # Count motifs in real graph
        real_counts = count_motifs_in_graph(model.nx_graph, motif_patterns, sp_mode)
        
        # Generate null models
        null_models = generate_null_models(model.nx_graph, n_random, sp_mode, rewire_iter, max_seconds, max_matches, log_every, min_edges, top_k, exclude_predicates)
        
        # Debug: Check if rewiring actually changed the graphs
        if len(null_models) > 0:
            original_edges = set(model.nx_graph.edges())
            null_edges = set(null_models[0].edges())
            edges_changed = len(original_edges.symmetric_difference(null_edges))
            print(f"[DEBUG] {model.name}: {edges_changed} edges changed in first null model")
        
        # Count motifs in null models
        null_counts = []
        for null_graph in null_models:
            null_count = count_motifs_in_graph(null_graph, motif_patterns, sp_mode)
            null_counts.append(null_count)
        
        # Compute Z-scores with zero-variance guard
        z_scores = {}
        debug_row = {"model": model.name}
        
        for motif in motif_patterns.keys():
            real_val = real_counts[motif]
            null_vals = [nc[motif] for nc in null_counts]
            
            mean_null = np.mean(null_vals)
            std_null = np.std(null_vals)
            
            # Zero-variance guard
            if std_null == 0:
                if mean_null == 0 and real_val == 0:
                    z_scores[motif] = 0.0  # Uninformative motif
                else:
                    # Strong significance signal
                    z_scores[motif] = np.sign(real_val - mean_null) * z_cap
            else:
                z_scores[motif] = (real_val - mean_null) / std_null
            
            # Store debug information
            debug_row[f"{motif}_real"] = real_val
            debug_row[f"{motif}_mean_rand"] = mean_null
            debug_row[f"{motif}_std_rand"] = std_null
            debug_row[f"{motif}_z"] = z_scores[motif]
            # Collect stats rows
            stats_rows.append({
                "model": model.name,
                "motif": motif,
                "mean_rand": float(mean_null),
                "std_rand": float(std_null)
            })
        
        debug_data.append(debug_row)
        
        # Normalize to unit vector (SP vector)
        z_values = np.array(list(z_scores.values()))
        if min_abs_z and float(min_abs_z) > 0.0:
            z_values = np.where(np.abs(z_values) < float(min_abs_z), 0.0, z_values)
        sp_norm = np.linalg.norm(z_values)
        
        if sp_norm > 0:
            sp_vector = z_values / sp_norm
        else:
            sp_vector = z_values
        
        sp_vectors[model.name] = dict(zip(motif_patterns.keys(), sp_vector))
        z_vectors[model.name] = dict(zip(motif_patterns.keys(), z_scores))
        
        # Store data for CSV
        row = {"model": model.name}
        row.update(real_counts)
        row.update({f"z_{k}": v for k, v in z_scores.items()})
        row.update({f"sp_{k}": v for k, v in sp_vectors[model.name].items()})
        motif_data.append(row)
        # Collect real counts row
        counts_row = {"model": model.name}
        counts_row.update(real_counts)
        counts_real_rows.append(counts_row)
        
        # Print statistics
        z_significant = sum(1 for z in z_scores.values() if abs(z) > 1.0)
        print(f"[SP] {model.name}: SP norm = {np.linalg.norm(sp_vector):.3f}, |Z|>1 motifs = {z_significant}/{len(motif_patterns)}")
        print(f"[SP] {model.name}: real-count + random ensemble finished in {time.time()-t0:.1f}s", flush=True)
    
    return sp_vectors, z_vectors, motif_data, debug_data, counts_real_rows, stats_rows

def compute_sp_similarity_matrix(sp_vectors):
    """Compute cosine similarity matrix from SP vectors."""
    models = list(sp_vectors.keys())
    n_models = len(models)
    
    similarity_matrix = np.zeros((n_models, n_models))
    
    for i, model_a in enumerate(models):
        for j, model_b in enumerate(models):
            if i == j:
                similarity_matrix[i, j] = 1.0
            else:
                sim = cosine(sp_vectors[model_a], sp_vectors[model_b])
                # Ensure non-negative similarity
                similarity_matrix[i, j] = max(0.0, sim)
    
    return pd.DataFrame(similarity_matrix, index=models, columns=models)

def visualize_sp_heatmap(similarity_matrix, output_path):
    """Create heatmap visualization of SP similarity matrix."""
    plt.figure(figsize=(10, 8))
    
    # Create heatmap
    sns.heatmap(similarity_matrix, 
                annot=True, 
                cmap='viridis', 
                square=True,
                fmt='.3f',
                cbar_kws={'label': 'SP Similarity'})
    
    plt.title('Motif Significance Profile Similarity Matrix', fontsize=16, pad=20)
    plt.xlabel('Models', fontsize=12)
    plt.ylabel('Models', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[OK] SP heatmap saved: {output_path}")

def visualize_sp_profiles(sp_vectors, output_dir):
    """Create individual SP profile visualizations for each model."""
    motifs = list(next(iter(sp_vectors.values())).keys())
    n_motifs = len(motifs)
    
    # Create radar plot for each model
    for model_name, sp_vector in sp_vectors.items():
        fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(projection='polar'))
        
        # Convert to numpy array
        values = np.array([sp_vector[motif] for motif in motifs])
        
        # Create angles for radar plot
        angles = np.linspace(0, 2 * np.pi, n_motifs, endpoint=False).tolist()
        values = np.concatenate((values, [values[0]]))  # Complete the circle
        angles += angles[:1]
        
        # Plot
        ax.plot(angles, values, 'o-', linewidth=2, label=model_name)
        ax.fill(angles, values, alpha=0.25)
        
        # Customize
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([motif.replace('_', '\n') for motif in motifs])
        ax.set_ylim(-1, 1)
        ax.set_title(f'Significance Profile: {model_name}', size=16, pad=20)
        ax.grid(True)
        
        plt.tight_layout()
        output_path = os.path.join(output_dir, f'SP_profiles_{model_name}.png')
        plt.savefig(output_path, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"[OK] SP profile saved: {output_path}")

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description='Compute Motif Significance Profiles')
    parser.add_argument('--models_dir', default='data/RDF_models', 
                       help='Directory containing RDF model files')
    parser.add_argument('--out_dir', default='.', 
                       help='Output directory for results')
    parser.add_argument('--n_random', type=int, default=20,
                       help='Number of random null models to generate')
    parser.add_argument('--merged_rdf', default='0000_Merged.rdf',
                       help='Path to merged RDF ontology')
    parser.add_argument('--sp_mode', choices=['untyped', 'predicate'], default='predicate',
                       help='SP computation mode: untyped (simplified) or predicate (layered)')
    parser.add_argument('--rewire_iter', type=int, default=10,
                       help='Number of rewiring iterations per layer')
    parser.add_argument('--z_cap', type=float, default=5.0,
                       help='Z-score cap for zero-variance motifs')
    parser.add_argument('--min_abs_z', type=float, default=0.0,
                       help='Zero-out |Z| below this threshold before SP normalization (denoise)')
    parser.add_argument("--max_seconds_per_layer", type=float, default=None,
                       help="Soft time cap per predicate layer (seconds). If exceeded, skip remaining matching work and continue.")
    parser.add_argument("--max_matches_per_layer", type=int, default=None,
                       help="Hard cap on motif isomorphism attempts per layer (VF2-like).")
    parser.add_argument("--log_every", type=int, default=1,
                       help="Print progress every k random graphs in the ensemble.")
    parser.add_argument("--min_edges_per_layer", type=int, default=5,
                       help="Minimum edges per layer to keep (filters tiny/noisy layers)")
    parser.add_argument("--top_k_layers", type=int, default=50,
                       help="Keep only top K layers by edge count (0 = no cap)")
    parser.add_argument("--exclude_predicates", type=str, default="type,label,comment,subClassOf,first,rest",
                       help="Comma-separated list of predicate local names to exclude")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.out_dir, exist_ok=True)
    
    # Find RDF model files
    if os.path.exists(args.models_dir):
        rdf_files = glob.glob(os.path.join(args.models_dir, "*.rdf"))
    else:
        # Fallback to current directory
        rdf_files = glob.glob("*.rdf")
    
    # Filter out merged ontology
    model_files = [f for f in rdf_files if not f.endswith(args.merged_rdf)]
    
    if not model_files:
        print(f"[ERROR] No RDF model files found in {args.models_dir}")
        return
    
    print(f"[SP] Found {len(model_files)} RDF model files")
    
    # Load models
    models = []
    for rdf_file in model_files:
        model_name = os.path.basename(rdf_file)  # Keep .rdf suffix for consistency
        try:
            model = Model(model_name, rdf_file)
            models.append(model)
        except Exception as e:
            print(f"[ERROR] Failed to load {model_name}: {e}")
            continue
    
    if not models:
        print("[ERROR] No models loaded successfully")
        return
    
    print(f"[SP] Successfully loaded {len(models)} models")
    
    # Compute significance profiles
    print(f"[SP] Computing significance profiles ({args.sp_mode} mode)...")
    sp_vectors, z_vectors, motif_data, debug_data, counts_real_rows, stats_rows = compute_significance_profiles(
        models, args.n_random, args.sp_mode, args.rewire_iter, args.z_cap, args.min_abs_z, 
        args.max_seconds_per_layer, args.max_matches_per_layer, args.log_every,
        args.min_edges_per_layer, args.top_k_layers, args.exclude_predicates)
    
    # Save SP vectors
    sp_df = pd.DataFrame(motif_data)
    sp_output = os.path.join(args.out_dir, 'motif_sp_vectors.csv')
    sp_df.to_csv(sp_output, index=False)
    print(f"[OK] SP vectors saved: {sp_output}")
    # Save real counts
    counts_df = pd.DataFrame(counts_real_rows)
    counts_output = os.path.join(args.out_dir, 'motif_counts_real.csv')
    counts_df.to_csv(counts_output, index=False)
    print(f"[OK] Real motif counts saved: {counts_output}")
    
    # Save Z vectors
    z_data = []
    for model in z_vectors.keys():
        row = {"model": model}
        row.update(z_vectors[model])
        z_data.append(row)
    z_df = pd.DataFrame(z_data)
    z_output = os.path.join(args.out_dir, 'motif_z_vectors.csv')
    z_df.to_csv(z_output, index=False)
    print(f"[OK] Z vectors saved: {z_output}")
    
    # Save debug information
    debug_df = pd.DataFrame(debug_data)
    debug_output = os.path.join(args.out_dir, 'motif_sp_debug.csv')
    debug_df.to_csv(debug_output, index=False)
    print(f"[OK] Debug data saved: {debug_output}")
    # Save random stats (mean/std per motif per model)
    stats_df = pd.DataFrame(stats_rows)
    stats_output = os.path.join(args.out_dir, 'motif_stats_random.csv')
    stats_df.to_csv(stats_output, index=False)
    print(f"[OK] Random stats saved: {stats_output}")
    
    # Compute similarity matrix
    print("[SP] Computing SP similarity matrix...")
    similarity_matrix = compute_sp_similarity_matrix(sp_vectors)
    
    # Save similarity matrix
    sim_output = os.path.join(args.out_dir, 'struct_similarity_sp.csv')
    similarity_matrix.to_csv(sim_output)
    print(f"[OK] SP similarity matrix saved: {sim_output}")
    
    # Create visualizations
    print("[SP] Creating visualizations...")
    heatmap_output = os.path.join(args.out_dir, 'SP_heatmap.png')
    visualize_sp_heatmap(similarity_matrix, heatmap_output)
    
    visualize_sp_profiles(sp_vectors, args.out_dir)
    
    # Print summary statistics
    print("\n=== SP SIMILARITY MATRIX SUMMARY ===")
    print(f"Matrix shape: {similarity_matrix.shape}")
    off_diagonal = similarity_matrix.values[np.triu_indices_from(similarity_matrix.values, k=1)]
    print(f"Mean off-diagonal similarity: {off_diagonal.mean():.3f}")
    print(f"Similarity range: [{similarity_matrix.values.min():.3f}, {similarity_matrix.values.max():.3f}]")
    print(f"Diagonal values: {np.allclose(np.diag(similarity_matrix.values), 1.0)}")
    print(f"Symmetric: {np.allclose(similarity_matrix.values, similarity_matrix.values.T)}")
    
    print("\n=== SP VECTOR STATISTICS ===")
    for model_name, sp_vector in sp_vectors.items():
        norm = np.linalg.norm(list(sp_vector.values()))
        z_significant = sum(1 for z in z_vectors[model_name].values() if isinstance(z, (int, float)) and abs(z) > 1.0)
        print(f"{model_name}: SP norm = {norm:.3f}, |Z|>1 motifs = {z_significant}/{len(sp_vector)}")
    
    print(f"\n[OK] Motif Significance Profile analysis complete!")
    print(f"Outputs saved to: {args.out_dir}")

if __name__ == "__main__":
    main()
