import os
import json


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))


def main():
    repo_map_path = os.path.join(ROOT, "repo_map.json")
    if not os.path.exists(repo_map_path):
        raise SystemExit("repo_map.json not found. Run tools/gen_repo_map.py first.")
    data = json.load(open(repo_map_path, "r", encoding="utf-8"))

    roles = {}
    for e in data:
        roles.setdefault(e.get("role","misc"), []).append(e["path"])

    lines = []
    lines.append("## Architecture overview\n")
    lines.append("Data layers and key modules by role.\n")
    lines.append("\n### Layers\n")
    lines.append("- Input: RDF/XML models in project root ( *_DG.rdf ), ontology file 0000_Merged.rdf\n")
    lines.append("- ETL/Feature extraction: extract_features.py, subgraph_similarity*.py, structural_extension*.py\n")
    lines.append("- Similarity calculators: content/typed/edge Jaccard/Cosine (CSV under repro_pack/output/02), structural (05/05b/07)\n")
    lines.append("- Fusion: combine_total_similarity.py, compute_similarity.py -> total_similarity_matrix.csv\n")
    lines.append("- Visualization/Reporting: visualize_*.py\n")
    lines.append("\n### Data flow (text)\n")
    lines.append("RDF (*.rdf) -> parsing (rdflib) -> graphs/motifs/types -> component similarities (CSV) -> convex fusion (weights) -> total score CSV + visuals\n")

    lines.append("\n### Files by role\n")
    for role, files in sorted(roles.items()):
        lines.append(f"- {role}:\n")
        for p in sorted(files):
            if p.endswith(".py") or p.endswith(".rdf"):
                lines.append(f"  - {p}\n")

    with open(os.path.join(ROOT, "architecture.md"), "w", encoding="utf-8") as f:
        f.writelines(lines)
    print("WROTE architecture.md")


if __name__ == "__main__":
    main()


