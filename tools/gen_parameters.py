import os
import re
import csv
from datetime import datetime


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))


def find_py_files():
    for dirpath, dirnames, filenames in os.walk(ROOT):
        if any(x in dirpath for x in (os.sep+"venv"+os.sep, os.sep+".git"+os.sep)):
            continue
        for f in filenames:
            if f.endswith(".py"):
                yield os.path.join(dirpath, f)


ARG_PATTERNS = [
    re.compile(r"add_argument\((?P<args>[^\)]*)\)"),
]


def parse_add_argument(lines):
    params = []
    for i, ln in enumerate(lines):
        if "add_argument(" not in ln:
            continue
        text = ln
        # join simple continuations on next lines
        j = i + 1
        while text.count("(") > text.count(")") and j < len(lines):
            text += lines[j]
            j += 1
        m = ARG_PATTERNS[0].search(text)
        if not m:
            continue
        args = m.group("args")
        # extract flag name(s)
        flags = re.findall(r"'(--[^']+)'|\"(--[^\"]+)\"", args)
        flags = [x[0] or x[1] for x in flags]
        if not flags:
            continue
        name = flags[-1].lstrip('-').replace('-', '_')
        # default
        mdef = re.search(r"default\s*=\s*([^,\)]+)", args)
        default = (mdef.group(1).strip() if mdef else "")
        mtype = re.search(r"type\s*=\s*([a-zA-Z_][a-zA-Z0-9_\.]*)", args)
        ptype = (mtype.group(1) if mtype else "str")
        choices = ",".join(re.findall(r"choices\s*=\s*\[([^\]]+)\]", args))
        where_defined = None
        params.append({
            "name": name,
            "type": ptype,
            "default": default,
            "choices": choices,
            "where_defined": None,
            "where_used": None,
        })
    return params


def main():
    rows = []
    for path in find_py_files():
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                lines = f.read().splitlines()
        except Exception:
            continue
        params = parse_add_argument(lines)
        for p in params:
            p["where_defined"] = path.replace(ROOT+os.sep, "")
            rows.append(p)

    out_csv = os.path.join(ROOT, "parameters.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["name","type","default","range/choices","where_defined","where_used(paths:lines)","effect_on_metrics","safety_checks"])
        for r in rows:
            w.writerow([r["name"], r["type"], r["default"], r["choices"], r["where_defined"], "", "", ""])
    print("WROTE", out_csv)


if __name__ == "__main__":
    main()


