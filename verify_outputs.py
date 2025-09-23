import os, glob
from rdflib import Graph, RDF, URIRef
import pandas as pd
from collections import Counter

BASE_DIR = "."
IN_DIR   = os.path.join(BASE_DIR, "repro_pack", "output", "Building_Information")  # <-- CSV'LERİ BURADA ARA
os.makedirs(IN_DIR, exist_ok=True)

# RDF dosyalarını doğrudan klasörden topla
DG_FILES = sorted([os.path.basename(p) for p in glob.glob(os.path.join(BASE_DIR, "*_DG.rdf"))])
ONTO_FILE = "0000_Merged.rdf"

NS = {
    "bot": "https://w3id.org/bot#",
    "bfo": "http://purl.obolibrary.org/obo/",
    "core": "https://spec.industrialontologies.org/ontology/core/Core/",
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

def safe_read_csv(path, expected_cols=None):
    if not os.path.exists(path):
        return pd.DataFrame(columns=expected_cols or [])
    try:
        df = pd.read_csv(path)
        if expected_cols and not set(expected_cols).issubset(df.columns):
            return pd.DataFrame(columns=expected_cols)
        return df
    except Exception:
        return pd.DataFrame(columns=expected_cols or [])

def main():
    rows = []
    for f in DG_FILES:
        g = parse_graph(os.path.join(BASE_DIR, f))

        # type distribution
        types = Counter(str(o) for _,_,o in g.triples((None, RDF.type, None)))
        type_csv = safe_read_csv(os.path.join(IN_DIR, f"{f}_type_distribution.csv"), ["type_iri","count"])
        rows.append({"file": f, "check": "type_sum",
                     "raw": sum(types.values()), "csv": int(type_csv["count"].sum()) if "count" in type_csv.columns else 0,
                     "ok": int(sum(types.values())) == (int(type_csv["count"].sum()) if "count" in type_csv.columns else 0)})

        # predicate distribution
        preds = Counter(str(p) for s,p,o in g)
        pred_csv = safe_read_csv(os.path.join(IN_DIR, f"{f}_predicate_distribution.csv"), ["predicate_iri","count"])
        rows.append({"file": f, "check": "pred_sum",
                     "raw": sum(preds.values()), "csv": int(pred_csv["count"].sum()) if "count" in pred_csv.columns else 0,
                     "ok": int(sum(preds.values())) == (int(pred_csv["count"].sum()) if "count" in pred_csv.columns else 0)})

        # function types
        func = Counter()
        for s,p,o in g.triples((None, P["hasFunction"], None)):
            for _,_,ft in g.triples((o, RDF.type, None)):
                func[str(ft)] += 1
        f_csv = safe_read_csv(os.path.join(IN_DIR, f"{f}_function_type_histogram.csv"), ["function_type_iri","count"])
        rows.append({"file": f, "check": "function_sum",
                     "raw": sum(func.values()), "csv": int(f_csv["count"].sum()) if "count" in f_csv.columns else 0,
                     "ok": int(sum(func.values())) == (int(f_csv["count"].sum()) if "count" in f_csv.columns else 0)})

        # quality types
        qual = Counter()
        for s,p,o in g.triples((None, P["hasQuality"], None)):
            for _,_,qt in g.triples((o, RDF.type, None)):
                qual[str(qt)] += 1
        q_csv = safe_read_csv(os.path.join(IN_DIR, f"{f}_quality_type_histogram.csv"), ["quality_type_iri","count"])
        rows.append({"file": f, "check": "quality_sum",
                     "raw": sum(qual.values()), "csv": int(q_csv["count"].sum()) if "count" in q_csv.columns else 0,
                     "ok": int(sum(qual.values())) == (int(q_csv["count"].sum()) if "count" in q_csv.columns else 0)})

        # edges
        for label, pred in [("adjacentElement", P["adjacentElement"]),
                            ("adjacentZone", P["adjacentZone"]),
                            ("intersectingElement", P["intersectingElement"]),
                            ("hasContinuantPart", P["hasContinuantPart"])]:
            raw = sum(1 for _ in g.triples((None, pred, None)))
            cpath = os.path.join(IN_DIR, f"{f}_{label}_edges.csv")
            csv_df = safe_read_csv(cpath, ["subject","predicate","object"])
            csvc = len(csv_df)
            rows.append({"file": f, "check": f"{label}_edges",
                         "raw": raw, "csv": csvc, "ok": raw == csvc})

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(IN_DIR, "verification_report.csv"), index=False)
    print("Verification report written:", os.path.join(IN_DIR, "verification_report.csv"))

if __name__ == "__main__":
    main()
