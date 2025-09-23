import os
import sys
import json
import re
from datetime import datetime, timezone


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
OUT_PATH = os.path.join(ROOT, "repo_map.json")

EXCLUDE_DIRS = {"venv", ".git", "__pycache__"}


def detect_role(path: str) -> str:
    p = path.replace("\\", "/").lower()
    name = os.path.basename(p)
    if name.endswith(".rdf"):
        return "rdf_data"
    if "visualize_" in p:
        return "visualization"
    if any(tok in p for tok in ["extract", "export"]):
        return "feature_extraction"
    if any(tok in p for tok in ["compute_similarity", "combine_total"]):
        return "similarity_fusion"
    if any(tok in p for tok in ["subgraph_similarity", "structural_extension"]):
        return "structural_similarity"
    return "misc"


def list_files(root: str):
    for dirpath, dirnames, filenames in os.walk(root):
        # prune excluded dirs in-place
        dirnames[:] = [d for d in dirnames if d not in EXCLUDE_DIRS]
        for fname in filenames:
            full = os.path.join(dirpath, fname)
            # skip inside excluded subpaths (defense)
            if any((f"{os.sep}{ex}{os.sep}" in full) for ex in EXCLUDE_DIRS):
                continue
            yield full


def read_text_safely(path: str):
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read().splitlines()
    except Exception:
        return []


def extract_imports_py(lines):
    imports = []
    pat = re.compile(r"^(?:from\s+\S+\s+import\s+\S+|import\s+\S+)")
    for ln in lines:
        m = pat.match(ln.strip())
        if m:
            imports.append(m.group(0))
    # deps = top-level module names
    deps = []
    for imp in imports:
        s = imp
        s = re.sub(r"^from\s+", "", s)
        s = re.sub(r"\s+import.*$", "", s)
        s = re.sub(r"^import\s+", "", s)
        s = s.split(",")[0].strip()
        if s:
            deps.append(s.split(".")[0])
    # unique
    deps = sorted(set(deps))
    return imports, deps


def main():
    entries = []
    for full in list_files(ROOT):
        rel = os.path.relpath(full, ROOT)
        ext = os.path.splitext(full)[1].lower()
        if ext == ".py":
            lang = "python"
        elif ext == ".rdf":
            lang = "rdfxml"
        elif ext == ".ttl":
            lang = "turtle"
        elif ext == ".csv":
            lang = "csv"
        elif ext == ".json":
            lang = "json"
        else:
            lang = "other"

        lines = []
        imports = []
        deps = []
        if lang in {"python", "rdfxml", "turtle", "csv", "json"}:
            lines = read_text_safely(full)
            loc = len(lines)
            if lang == "python":
                imports, deps = extract_imports_py(lines)
        else:
            try:
                loc = os.path.getsize(full)
            except Exception:
                loc = 0

        try:
            mtime = datetime.fromtimestamp(os.path.getmtime(full), tz=timezone.utc).isoformat()
        except Exception:
            mtime = None

        entries.append({
            "path": rel.replace("\\", "/"),
            "role": detect_role(rel),
            "lang": lang,
            "loc": loc,
            "last_modified": mtime,
            "imports": imports,
            "deps": deps,
        })

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2)
    print("WROTE", OUT_PATH)


if __name__ == "__main__":
    sys.exit(main())


