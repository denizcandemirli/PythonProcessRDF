import os, glob
from rdflib import Graph, RDF, URIRef
from collections import Counter
import pandas as pd

BASE_DIR = "."
OUT_DIR  = os.path.join(BASE_DIR, "repro_pack", "output", "Building_Information")
os.makedirs(OUT_DIR, exist_ok=True)


# Klasördeki TÜM tasarım grafı RDF'lerini otomatik topla (ontoloji hariç)
DG_FILES = sorted([os.path.basename(p) for p in glob.glob(os.path.join(BASE_DIR, "*_DG.rdf"))])
ONTO_FILE = "0000_Merged.rdf"  # referans amaçlı

NS = {
    "bot": "https://w3id.org/bot#",
    "bfo": "http://purl.obolibrary.org/obo/",
    "core": "https://spec.industrialontologies.org/ontology/core/Core/",
    "bd": "http://quantecton.com/kb/BuildingDesign#",
}

P = {
    "hasFunction": URIRef(NS["core"] + "hasFunction"),
    "hasQuality": URIRef(NS["core"] + "hasQuality"),
    "adjacentElement": URIRef(NS["bot"] + "adjacentElement"),
    "adjacentZone": URIRef(NS["bot"] + "adjacentZone"),
    "intersectingElement": URIRef(NS["bot"] + "intersectingElement"),
    "hasContinuantPart": URIRef(NS["bfo"] + "BFO_0000178"),
}

def parse_graph(path):
    g = Graph()
    try:
        g.parse(path, format="xml")
    except Exception:
        try:
            g.parse(path, format="turtle")
        except Exception:
            g.parse(path)
    return g

def type_distribution(g: Graph):
    types = Counter(str(o) for _,_,o in g.triples((None, RDF.type, None)))
    return pd.DataFrame(types.items(), columns=["type_iri", "count"]).sort_values("count", ascending=False)

def predicate_distribution(g: Graph):
    preds = Counter(str(p) for s,p,o in g)
    return pd.DataFrame(preds.items(), columns=["predicate_iri", "count"]).sort_values("count", ascending=False)

def function_type_histogram(g: Graph):
    func_types = Counter()
    for s, p, o in g.triples((None, P["hasFunction"], None)):
        for _, _, ft in g.triples((o, RDF.type, None)):
            func_types[str(ft)] += 1
    return pd.DataFrame(func_types.items(), columns=["function_type_iri", "count"]).sort_values("count", ascending=False)

def quality_type_histogram(g: Graph):
    qual_types = Counter()
    for s, p, o in g.triples((None, P["hasQuality"], None)):
        for _, _, qt in g.triples((o, RDF.type, None)):
            qual_types[str(qt)] += 1
    return pd.DataFrame(qual_types.items(), columns=["quality_type_iri", "count"]).sort_values("count", ascending=False)

def edge_list(g: Graph, pred: URIRef):
    rows = [{"subject": str(s), "predicate": str(pred), "object": str(o)} for s, p, o in g.triples((None, pred, None))]
    # Boş kalsa bile başlıklar yazılsın:
    return pd.DataFrame(rows, columns=["subject", "predicate", "object"])

def typed_edge_profile(g: Graph):
    def get_types(node):
        return [str(o) for _,_,o in g.triples((node, RDF.type, None))]
    SEL = [P["hasFunction"], P["hasQuality"], P["adjacentElement"], P["adjacentZone"], P["intersectingElement"], P["hasContinuantPart"]]
    prof = Counter()
    for s,p,o in g:
        if p in SEL:
            st = get_types(s) or ["__UNtyped__"]
            ot = get_types(o) or ["__UNtyped__"]
            for a in st:
                for b in ot:
                    prof[(a,str(p),b)] += 1
    rows = [{"subject_type": k[0], "predicate_iri": k[1], "object_type": k[2], "count": v} for k,v in prof.items()]
    return pd.DataFrame(rows).sort_values("count", ascending=False)

def main():
    for fname in DG_FILES:
        g = parse_graph(os.path.join(BASE_DIR, fname))

        type_distribution(g).to_csv(os.path.join(OUT_DIR, f"{fname}_type_distribution.csv"), index=False)
        predicate_distribution(g).to_csv(os.path.join(OUT_DIR, f"{fname}_predicate_distribution.csv"), index=False)
        function_type_histogram(g).to_csv(os.path.join(OUT_DIR, f"{fname}_function_type_histogram.csv"), index=False)
        quality_type_histogram(g).to_csv(os.path.join(OUT_DIR, f"{fname}_quality_type_histogram.csv"), index=False)

        for label, pred in [("adjacentElement", P["adjacentElement"]),
                            ("adjacentZone", P["adjacentZone"]),
                            ("intersectingElement", P["intersectingElement"]),
                            ("hasContinuantPart", P["hasContinuantPart"])]:
            edge_list(g, pred).to_csv(os.path.join(OUT_DIR, f"{fname}_{label}_edges.csv"), index=False)

        typed_edge_profile(g).to_csv(os.path.join(OUT_DIR, f"{fname}_typed_edge_profile.csv"), index=False)

if __name__ == "__main__":
    main()
