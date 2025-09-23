import os
import ast
import csv


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))


def py_files():
    for dirpath, dirnames, filenames in os.walk(ROOT):
        if any(x in dirpath for x in (os.sep+"venv"+os.sep, os.sep+".git"+os.sep)):
            continue
        for f in filenames:
            if f.endswith('.py'):
                yield os.path.join(dirpath, f)


def complexity_of_node(node: ast.AST) -> int:
    # very rough cyclomatic: count branching nodes
    count = 1
    for n in ast.walk(node):
        if isinstance(n, (ast.If, ast.For, ast.While, ast.And, ast.Or, ast.Try, ast.With, ast.BoolOp, ast.ExceptHandler, ast.IfExp)):
            count += 1
    return count


def analyze(path: str):
    try:
        src = open(path, 'r', encoding='utf-8', errors='ignore').read()
        tree = ast.parse(src)
    except Exception as e:
        return []
    rows = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            name = node.name
            loc = (node.end_lineno or node.lineno) - node.lineno + 1 if hasattr(node, 'end_lineno') else 0
            cplx = complexity_of_node(node)
            rows.append((path.replace(ROOT+os.sep,''), name, loc, cplx))
    return rows


def main():
    rows = []
    for p in py_files():
        rows.extend(analyze(p))
    rows.sort(key=lambda x: (-x[3], -x[2]))
    out_md = os.path.join(ROOT, 'static_analysis.md')
    with open(out_md, 'w', encoding='utf-8') as f:
        f.write('### Complexity (approx)\n')
        f.write('| file | function | loc | approx_cyclomatic |\n')
        f.write('|---|---:|---:|---:|\n')
        for file, fn, loc, c in rows[:200]:
            f.write(f'| {file} | {fn} | {loc} | {c} |\n')
        f.write('\nNotes: heuristic only; no dead-code/N+1 analysis implemented.\n')
    print('WROTE', out_md)


if __name__ == '__main__':
    main()


