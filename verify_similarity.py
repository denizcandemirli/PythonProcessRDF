#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File: verify_similarity.py
Purpose: Verify similarity computation results and pipeline integrity.
Input: RDF models, similarity matrices, and intermediate results
Output: Verification reports, data quality checks, consistency analysis

This script performs comprehensive verification:
- RDF model parsing validation
- Similarity matrix properties (symmetry, range, etc.)
- Data consistency checks
- Pipeline integrity verification

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
import argparse
from collections import Counter
from typing import Dict, List, Tuple, Set
import pandas as pd
import numpy as np

# Dependencies
try:
    import rdflib
    from rdflib import Graph, RDF, RDFS, URIRef
except ImportError:
    raise SystemExit("rdflib is required. pip install rdflib")

# ---------------------------
# Configuration
# ---------------------------

# Expected namespace mappings
EXPECTED_NS = {
    "bot": "https://w3id.org/bot#",
    "bfo": "http://purl.obolibrary.org/obo/",
    "core": "https://spec.industrialontologies.org/ontology/core/Core/",
}

# Expected predicates for validation
EXPECTED_PREDICATES = {
    "hasFunction": URIRef(EXPECTED_NS["core"] + "hasFunction"),
    "hasQuality": URIRef(EXPECTED_NS["core"] + "hasQuality"),
    "adjacentElement": URIRef(EXPECTED_NS["bot"] + "adjacentElement"),
    "adjacentZone": URIRef(EXPECTED_NS["bot"] + "adjacentZone"),
    "intersectingElement": URIRef(EXPECTED_NS["bot"] + "intersectingElement"),
    "hasContinuantPart": URIRef(EXPECTED_NS["bfo"] + "BFO_0000178"),
}

# ---------------------------
# Helper Functions
# ---------------------------

def ensure_dir(path: str):
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)

def safe_read_csv(path: str, expected_cols: List[str] = None) -> pd.DataFrame:
    """Safely read CSV file with error handling."""
    if not os.path.exists(path):
        return pd.DataFrame(columns=expected_cols or [])
    try:
        df = pd.read_csv(path)
        if expected_cols and not set(expected_cols).issubset(df.columns):
            return pd.DataFrame(columns=expected_cols)
        return df
    except Exception as e:
        print(f"[WARN] Error reading {path}: {e}")
        return pd.DataFrame(columns=expected_cols or [])

def parse_rdf_graph(path: str) -> Graph:
    """Parse RDF graph with multiple format support."""
    g = Graph()
    try:
        g.parse(path, format="xml")
    except Exception:
        try:
            g.parse(path, format="turtle")
        except Exception:
            g.parse(path)
    return g

# ---------------------------
# Verification Functions
# ---------------------------

def verify_rdf_models(input_dir: str, pattern: str = "*_DG.rdf") -> Dict[str, Dict]:
    """Verify RDF model parsing and basic properties."""
    print("[VERIFY] Checking RDF models...")
    
    rdf_files = sorted(glob.glob(os.path.join(input_dir, pattern)))
    if not rdf_files:
        print(f"[WARN] No RDF files found matching pattern: {pattern}")
        return {}
    
    results = {}
    for rdf_path in rdf_files:
        model_name = os.path.basename(rdf_path)
        print(f"[VERIFY] Checking {model_name}...")
        
        try:
            g = parse_rdf_graph(rdf_path)
            
            # Basic statistics
            num_triples = len(g)
            num_subjects = len(set(g.subjects()))
            num_objects = len(set([o for o in g.objects() if not isinstance(o, rdflib.term.Literal)]))
            num_predicates = len(set(g.predicates()))
            
            # Type distribution
            type_counts = Counter(str(o) for _, _, o in g.triples((None, RDF.type, None)))
            
            # Predicate distribution
            pred_counts = Counter(str(p) for s, p, o in g)
            
            # Check for expected predicates
            found_predicates = {}
            for pred_name, pred_uri in EXPECTED_PREDICATES.items():
                count = sum(1 for _ in g.triples((None, pred_uri, None)))
                found_predicates[pred_name] = count
            
            results[model_name] = {
                "status": "OK",
                "num_triples": num_triples,
                "num_subjects": num_subjects,
                "num_objects": num_objects,
                "num_predicates": num_predicates,
                "type_distribution": dict(type_counts.most_common(10)),
                "predicate_distribution": dict(pred_counts.most_common(10)),
                "found_predicates": found_predicates,
                "parsing_errors": []
            }
            
            print(f"[OK] {model_name}: {num_triples} triples, {num_subjects} subjects, {num_objects} objects")
            
        except Exception as e:
            results[model_name] = {
                "status": "ERROR",
                "parsing_errors": [str(e)],
                "num_triples": 0,
                "num_subjects": 0,
                "num_objects": 0,
                "num_predicates": 0
            }
            print(f"[ERROR] {model_name}: {e}")
    
    return results

def verify_similarity_matrix(matrix_path: str, matrix_name: str) -> Dict:
    """Verify similarity matrix properties."""
    print(f"[VERIFY] Checking {matrix_name}...")
    
    if not os.path.exists(matrix_path):
        return {"status": "ERROR", "error": "File not found"}
    
    try:
        df = pd.read_csv(matrix_path, index_col=0)
        
        # Convert to numeric
        df_numeric = df.apply(pd.to_numeric, errors="coerce")
        
        # Check for missing values
        missing_count = df_numeric.isnull().sum().sum()
        
        # Check symmetry
        is_symmetric = np.allclose(df_numeric.values, df_numeric.values.T, rtol=1e-10)
        
        # Check diagonal values
        diagonal_values = np.diag(df_numeric.values)
        diagonal_ok = np.allclose(diagonal_values, 1.0, rtol=1e-10)
        
        # Check value range
        min_val = df_numeric.values.min()
        max_val = df_numeric.values.max()
        range_ok = 0.0 <= min_val and max_val <= 1.0
        
        # Check for negative values
        has_negative = (df_numeric.values < 0).any()
        
        # Statistics
        upper_triangle = df_numeric.values[np.triu_indices_from(df_numeric.values, k=1)]
        mean_similarity = upper_triangle.mean()
        std_similarity = upper_triangle.std()
        
        return {
            "status": "OK" if all([is_symmetric, diagonal_ok, range_ok, not has_negative]) else "WARNING",
            "shape": df_numeric.shape,
            "missing_values": int(missing_count),
            "is_symmetric": is_symmetric,
            "diagonal_ok": diagonal_ok,
            "range_ok": range_ok,
            "has_negative": has_negative,
            "min_value": float(min_val),
            "max_value": float(max_val),
            "mean_similarity": float(mean_similarity),
            "std_similarity": float(std_similarity),
            "models": list(df_numeric.index)
        }
        
    except Exception as e:
        return {"status": "ERROR", "error": str(e)}

def verify_pipeline_consistency(results_dir: str) -> Dict:
    """Verify consistency across pipeline results."""
    print("[VERIFY] Checking pipeline consistency...")
    
    # Expected files
    expected_files = [
        "similarity_content_cosine.csv",
        "similarity_typed_edge_cosine.csv", 
        "similarity_edge_sets_jaccard.csv",
        "struct_similarity_matrix.csv",
        "total_similarity_matrix.csv",
        "pairwise_total_summary.csv"
    ]
    
    results = {"status": "OK", "missing_files": [], "inconsistent_models": []}
    
    # Check for missing files
    for filename in expected_files:
        filepath = os.path.join(results_dir, filename)
        if not os.path.exists(filepath):
            results["missing_files"].append(filename)
            results["status"] = "WARNING"
    
    # Check model consistency across matrices
    matrix_files = [f for f in expected_files if f.endswith("_matrix.csv")]
    model_sets = []
    
    for filename in matrix_files:
        filepath = os.path.join(results_dir, filename)
        if os.path.exists(filepath):
            try:
                df = pd.read_csv(filepath, index_col=0)
                model_sets.append(set(df.index))
            except Exception as e:
                print(f"[WARN] Could not read {filename}: {e}")
    
    # Check for model inconsistencies
    if model_sets:
        common_models = set.intersection(*model_sets)
        all_models = set.union(*model_sets)
        inconsistent = all_models - common_models
        
        if inconsistent:
            results["inconsistent_models"] = list(inconsistent)
            results["status"] = "WARNING"
    
    return results

def verify_data_quality(feature_dir: str) -> Dict:
    """Verify data quality of feature extraction outputs."""
    print("[VERIFY] Checking data quality...")
    
    # Find available models
    type_files = glob.glob(os.path.join(feature_dir, "*_type_distribution.csv"))
    models = [os.path.basename(f).replace("_type_distribution.csv", "") for f in type_files]
    
    results = {"status": "OK", "models": {}, "summary": {}}
    
    total_checks = 0
    passed_checks = 0
    
    for model in models:
        model_results = {"status": "OK", "checks": []}
        
        # Check type distribution
        type_path = os.path.join(feature_dir, f"{model}_type_distribution.csv")
        type_df = safe_read_csv(type_path, ["type_iri", "count"])
        if not type_df.empty:
            model_results["checks"].append("type_distribution: OK")
            passed_checks += 1
        else:
            model_results["checks"].append("type_distribution: MISSING")
            model_results["status"] = "WARNING"
        total_checks += 1
        
        # Check function histogram
        func_path = os.path.join(feature_dir, f"{model}_function_type_histogram.csv")
        func_df = safe_read_csv(func_path, ["function_type_iri", "count"])
        if not func_df.empty:
            model_results["checks"].append("function_histogram: OK")
            passed_checks += 1
        else:
            model_results["checks"].append("function_histogram: MISSING")
        total_checks += 1
        
        # Check quality histogram
        qual_path = os.path.join(feature_dir, f"{model}_quality_type_histogram.csv")
        qual_df = safe_read_csv(qual_path, ["quality_type_iri", "count"])
        if not qual_df.empty:
            model_results["checks"].append("quality_histogram: OK")
            passed_checks += 1
        else:
            model_results["checks"].append("quality_histogram: MISSING")
        total_checks += 1
        
        # Check typed edge profile
        typed_path = os.path.join(feature_dir, f"{model}_typed_edge_profile.csv")
        typed_df = safe_read_csv(typed_path, ["subject_type", "predicate_iri", "object_type", "count"])
        if not typed_df.empty:
            model_results["checks"].append("typed_edge_profile: OK")
            passed_checks += 1
        else:
            model_results["checks"].append("typed_edge_profile: MISSING")
        total_checks += 1
        
        results["models"][model] = model_results
    
    results["summary"] = {
        "total_checks": total_checks,
        "passed_checks": passed_checks,
        "pass_rate": passed_checks / total_checks if total_checks > 0 else 0
    }
    
    if results["summary"]["pass_rate"] < 1.0:
        results["status"] = "WARNING"
    
    return results

# ---------------------------
# Main Function
# ---------------------------

def main():
    """Main execution function."""
    ap = argparse.ArgumentParser(description="Verify similarity computation pipeline")
    ap.add_argument("--rdf-dir", default=".",
                   help="Directory containing RDF model files")
    ap.add_argument("--feature-dir", 
                   default=os.path.join(".", "repro_pack", "output", "Building_Information"),
                   help="Directory containing feature extraction outputs")
    ap.add_argument("--results-dir", 
                   default=os.path.join(".", "repro_pack", "output", "02 - Similarity"),
                   help="Directory containing similarity results")
    ap.add_argument("--output-dir", 
                   default=os.path.join(".", "repro_pack", "output", "Verification"),
                   help="Directory to save verification reports")
    ap.add_argument("--rdf-pattern", default="*_DG.rdf",
                   help="Pattern for RDF files")
    
    args = ap.parse_args()
    
    # Ensure output directory exists
    ensure_dir(args.output_dir)
    
    print(f"[INFO] Starting similarity pipeline verification...")
    print(f"[INFO] RDF directory: {args.rdf_dir}")
    print(f"[INFO] Feature directory: {args.feature_dir}")
    print(f"[INFO] Results directory: {args.results_dir}")
    print(f"[INFO] Output directory: {args.output_dir}")
    
    verification_results = {}
    
    try:
        # 1. Verify RDF models
        rdf_results = verify_rdf_models(args.rdf_dir, args.rdf_pattern)
        verification_results["rdf_models"] = rdf_results
        
        # 2. Verify data quality
        quality_results = verify_data_quality(args.feature_dir)
        verification_results["data_quality"] = quality_results
        
        # 3. Verify similarity matrices
        matrix_results = {}
        matrix_files = [
            ("similarity_content_cosine.csv", "Content Similarity"),
            ("similarity_typed_edge_cosine.csv", "Typed-Edge Similarity"),
            ("similarity_edge_sets_jaccard.csv", "Edge-Set Similarity"),
            ("struct_similarity_matrix.csv", "Structural Similarity"),
            ("total_similarity_matrix.csv", "Total Similarity")
        ]
        
        for filename, name in matrix_files:
            filepath = os.path.join(args.results_dir, filename)
            matrix_results[filename] = verify_similarity_matrix(filepath, name)
        
        verification_results["similarity_matrices"] = matrix_results
        
        # 4. Verify pipeline consistency
        consistency_results = verify_pipeline_consistency(args.results_dir)
        verification_results["pipeline_consistency"] = consistency_results
        
        # Save verification report
        report_path = os.path.join(args.output_dir, "verification_report.json")
        with open(report_path, "w") as f:
            import json
            json.dump(verification_results, f, indent=2, default=str)
        
        # Generate summary
        print(f"\n[SUMMARY] Verification Results:")
        print(f"  RDF Models: {len([m for m in rdf_results.values() if m['status'] == 'OK'])}/{len(rdf_results)} OK")
        print(f"  Data Quality: {quality_results['summary']['pass_rate']:.1%} pass rate")
        print(f"  Similarity Matrices: {len([m for m in matrix_results.values() if m['status'] == 'OK'])}/{len(matrix_results)} OK")
        print(f"  Pipeline Consistency: {consistency_results['status']}")
        
        if consistency_results['missing_files']:
            print(f"  Missing Files: {', '.join(consistency_results['missing_files'])}")
        
        if consistency_results['inconsistent_models']:
            print(f"  Inconsistent Models: {', '.join(consistency_results['inconsistent_models'])}")
        
        print(f"\n[OK] Verification completed successfully!")
        print(f"[OK] Report saved: {report_path}")
        
    except Exception as e:
        print(f"[ERROR] Verification failed: {e}")
        raise

if __name__ == "__main__":
    main()
