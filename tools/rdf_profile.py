import os
import csv
from collections import Counter
from rdflib import Graph, RDF


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))


def find_rdf_files():
    for name in os.listdir(ROOT):
        if name.lower().endswith(".rdf"):
            yield os.path.join(ROOT, name)


def parse_graph(path: str) -> Graph:
    g = Graph()
    try:
        g.parse(path, format="xml")
    except Exception:
        g.parse(path)
    return g


def main():
    classes = Counter()
    properties = Counter()
    prop_samples = {}
    nsmap = Counter()

    files = list(find_rdf_files())
    if not files:
        print("No RDF files found in project root.")

    for p in files:
        g = parse_graph(p)
        # namespaces
        for pre, ns in g.namespace_manager.namespaces():
            nsmap[str(pre)] += 1
        # class counts
        for s, _, o in g.triples((None, RDF.type, None)):
            classes[str(o)] += 1
        # property counts and sample values
        for s, pred, o in g:
            key = str(pred)
            properties[key] += 1
            if key not in prop_samples:
                prop_samples[key] = str(o)

    # write rb_classes.csv
    with open(os.path.join(ROOT, "rb_classes.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["class","count"])
        for k,v in sorted(classes.items(), key=lambda x: (-x[1], x[0])):
            w.writerow([k, v])

    # write rb_properties.csv
    with open(os.path.join(ROOT, "rb_properties.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["property","count","sample_values"])            
        for k,v in sorted(properties.items(), key=lambda x: (-x[1], x[0])):
            w.writerow([k, v, prop_samples.get(k, "")])

    # ontology_overview.md
    with open(os.path.join(ROOT, "ontology_overview.md"), "w", encoding="utf-8") as f:
        f.write("## Namespaces observed\n\n")
        for pre, cnt in sorted(nsmap.items(), key=lambda x: (-x[1], x[0])):
            f.write(f"- {pre}: used in {cnt} file(s)\n")
        f.write("\n## Notes\n- Domain/range inference not implemented. Use SHACL for validation.\n")

    print("WROTE rb_classes.csv, rb_properties.csv, ontology_overview.md")


if __name__ == "__main__":
    main()


