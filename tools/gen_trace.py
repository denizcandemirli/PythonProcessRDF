import os
import re
import csv
from typing import List, Tuple

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))

# Heuristic mapping of feature labels used across the repo to RDF predicates
PRED_HINTS = {
    "adjacentElement": re.compile(r"adjacent(Element|element)", re.I),
    "adjacentZone": re.compile(r"adjacent(Zone|zone)", re.I),
    "intersectingElement": re.compile(r"intersect(ing)?(Element|element)", re.I),
    "BFO_0000178": re.compile(r"BFO_0000178|hasContinuantPart", re.I),
    "hasFunction": re.compile(r"hasFunction", re.I),
    "hasQuality": re.compile(r"hasQuality", re.I),
}

FEATURES = [
    ("motif:M1_adjacentZone_ZZ", "adjacentZone", "structural motif"),
    ("motif:M2_adjacentElement_EE", "adjacentElement", "structural motif"),
    ("motif:M3_intersectingElement_EE", "intersectingElement", "structural motif"),
    ("motif:M4_hasContinuantPart_EP", "BFO_0000178", "structural motif"),
    ("motif:M5_hasFunction_EF", "hasFunction", "semantic motif"),
    ("typed_edges:hasFunction", "hasFunction", "typed-edge"),
    ("typed_edges:hasQuality", "hasQuality", "typed-edge"),
    ("edge_sets:adjacentElement", "adjacentElement", "edge-set"),
]

# Files to scan for evidence lines
SCAN_FILES = [
    "subgraph_similarity.py",
    "subgraph_similarity_v2.py",
    "structural_extension.py",
    "extract_features.py",
]


def find_occurrences(path: str, needle: str) -> List[Tuple[int, str]]:
    rows = []
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for i, ln in enumerate(f, 1):
                if needle in ln:
                    rows.append((i, ln.strip()))
    except Exception:
        pass
    return rows


def main():
    out_csv = os.path.join(ROOT, "trace_matrix.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "model_feature","rdf_property","source_files","extraction_code(paths:lines)",
            "transformations","weight/role","used_in_metric","evidence_link"
        ])
        for feat, prop, role in FEATURES:
            srcs = []
            refs = []
            for fname in SCAN_FILES:
                p = os.path.join(ROOT, fname)
                occ = find_occurrences(p, prop)
                if occ:
                    srcs.append(fname)
                    for (ln, txt) in occ[:5]:
                        refs.append(f"{fname}:{ln}:{txt[:120]}")
            w.writerow([
                feat,
                prop,
                ";".join(srcs),
                " | ".join(refs),
                "canon/localname heuristics",
                role,
                "cosine/jaccard/overlap",
                "",
            ])
    print("WROTE", out_csv)


if __name__ == "__main__":
    main()
