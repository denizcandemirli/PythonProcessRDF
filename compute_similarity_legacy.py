import os, glob, math
import pandas as pd
from collections import Counter
from itertools import combinations

BASE_DIR = "."
IN_DIR   = os.path.join(BASE_DIR, "repro_pack", "output", "Building_Information")  # <-- GİRİŞLER BURADA
OUT_DIR  = os.path.join(BASE_DIR, "repro_pack", "output")                           # <-- ÇIKIŞLAR BURADA
os.makedirs(OUT_DIR, exist_ok=True)

# Özellik çıkarımı tamamlanmış modellere göre listeyi oluştur
DG_FILES = sorted({
    os.path.basename(p).replace("_typed_edge_profile.csv", "")
    for p in glob.glob(os.path.join(IN_DIR, "*_typed_edge_profile.csv"))
})

def load_counter_csv(path, key_col, val_col):
    if not os.path.exists(path):
        return Counter()
    try:
        df = pd.read_csv(path)
        if key_col not in df.columns or val_col not in df.columns:
            return Counter()
        return Counter(dict(zip(df[key_col], df[val_col])))
    except Exception:
        # boş/bozuk dosya vs.
        return Counter()

def cosine(a: Counter, b: Counter):
    keys = set(a) | set(b)
    if not keys:
        return 0.0
    va = [a.get(k,0) for k in keys]
    vb = [b.get(k,0) for k in keys]
    dot = sum(x*y for x,y in zip(va,vb))
    na  = math.sqrt(sum(x*x for x in va))
    nb  = math.sqrt(sum(y*y for y in vb))
    return 0.0 if na==0 or nb==0 else dot/(na*nb)

def set_jaccard(A:set, B:set):
    U = A | B
    return (len(A & B) / len(U)) if U else 0.0

# Content vectors = type + function + quality-type
content_vectors = {}
for f in DG_FILES:
    type_c = load_counter_csv(os.path.join(IN_DIR, f"{f}_type_distribution.csv"), "type_iri", "count")
    func_c = load_counter_csv(os.path.join(IN_DIR, f"{f}_function_type_histogram.csv"), "function_type_iri", "count")
    qual_c = load_counter_csv(os.path.join(IN_DIR, f"{f}_quality_type_histogram.csv"), "quality_type_iri", "count")
    v = Counter(); v.update(type_c); v.update(func_c); v.update(qual_c)
    content_vectors[f] = v

# Typed-edge profiles (as counters and as key sets)
typed_profiles = {}
typed_keysets = {}
for f in DG_FILES:
    path = os.path.join(IN_DIR, f"{f}_typed_edge_profile.csv")
    c = Counter()
    keys = set()
    if os.path.exists(path):
        try:
            df = pd.read_csv(path)
            if set(["subject_type","predicate_iri","object_type","count"]).issubset(df.columns):
                c = Counter({(r["subject_type"], r["predicate_iri"], r["object_type"]): int(r["count"]) for _,r in df.iterrows()})
                keys = set(c.keys())
        except Exception:
            pass
    typed_profiles[f] = c
    typed_keysets[f]  = keys

# Edge sets per predicate and combined
def edge_set(f, label):
    p = os.path.join(IN_DIR, f"{f}_{label}_edges.csv")
    if not os.path.exists(p):
        return set()
    try:
        df = pd.read_csv(p)
        if not set(["subject","predicate","object"]).issubset(df.columns):
            return set()
        return set((row["subject"], row["predicate"], row["object"]) for _,row in df.iterrows())
    except Exception:
        return set()

edge_labels = ["adjacentElement","adjacentZone","intersectingElement","hasContinuantPart"]
edge_sets = {f:{lab: edge_set(f, lab) for lab in edge_labels} for f in DG_FILES}
for f in DG_FILES:
    allc = set()
    for lab in edge_labels:
        allc |= edge_sets[f][lab]
    edge_sets[f]["__combined__"] = allc

# Compute similarities
rows_content = []
rows_typed_cos = []
rows_typed_jac = []
rows_edge_jac  = []
rows_combined  = []

for a,b in combinations(DG_FILES,2):
    cc = cosine(content_vectors[a], content_vectors[b])
    rows_content.append({"model_a":a,"model_b":b,"content_cosine":cc})

    tec = cosine(typed_profiles[a], typed_profiles[b])
    tej = set_jaccard(typed_keysets[a], typed_keysets[b])
    rows_typed_cos.append({"model_a":a,"model_b":b,"typed_edge_cosine":tec})
    rows_typed_jac.append({"model_a":a,"model_b":b,"typed_edge_jaccard":tej})

    row = {"model_a":a,"model_b":b}
    for lab in edge_labels+["__combined__"]:
        row[f"jaccard_{lab}"] = set_jaccard(edge_sets[a][lab], edge_sets[b][lab])
    rows_edge_jac.append(row)

    structure_score = 0.5*tec + 0.5*row["jaccard___combined__"]  # "__combined__" için üç alt çizgi doğru
    final = 0.5*cc + 0.5*structure_score
    rows_combined.append({
        "model_a":a,"model_b":b,
        "content_cosine":cc,
        "typed_edge_cosine":tec,
        "edge_jaccard_combined":row["jaccard___combined__"],
        "structure_score":structure_score,
        "final_similarity":final
    })

pd.DataFrame(rows_content).to_csv(os.path.join(OUT_DIR,"similarity_content_cosine.csv"), index=False)
pd.DataFrame(rows_typed_cos).to_csv(os.path.join(OUT_DIR,"similarity_typed_edge_cosine.csv"), index=False)
pd.DataFrame(rows_typed_jac).to_csv(os.path.join(OUT_DIR,"similarity_typed_edge_jaccard.csv"), index=False)
pd.DataFrame(rows_edge_jac).to_csv(os.path.join(OUT_DIR,"similarity_edge_sets_jaccard.csv"), index=False)
pd.DataFrame(rows_combined).sort_values("final_similarity", ascending=False).to_csv(os.path.join(OUT_DIR,"similarity_combined_weighted.csv"), index=False)
print("Similarity CSVs written to:", OUT_DIR)
