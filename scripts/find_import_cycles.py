# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build the static top-level import graph for in-repo modules and find SCCs.

Used to identify cycles introduced by lifting previously-local imports.
Prints any SCC (strongly connected component) with size > 1, listing member
modules and the files that define them.
"""

from __future__ import annotations

import ast
import os
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SKIP_DIRS = {".git", ".venv", "node_modules", "__pycache__", ".worktrees", "build", "dist"}

LIB_ROOTS = [
    ("iris", REPO / "lib/iris/src/iris"),
    ("fray", REPO / "lib/fray/src/fray"),
    ("zephyr", REPO / "lib/zephyr/src/zephyr"),
    ("levanter", REPO / "lib/levanter/src/levanter"),
    ("haliax", REPO / "lib/haliax/src/haliax"),
    ("marin", REPO / "lib/marin/src/marin"),
    ("finelog", REPO / "lib/finelog/src/finelog"),
    ("rigging", REPO / "lib/rigging/src/rigging"),
]

# IN_REPO_TOP = set of top-level package names we care about.
IN_REPO_TOP = {name for name, _ in LIB_ROOTS}


def file_to_module(top: str, root: Path, f: Path) -> str:
    rel = f.relative_to(root).with_suffix("")
    parts = [top, *list(rel.parts)]
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def collect() -> tuple[dict[str, Path], dict[str, set[str]]]:
    module_file: dict[str, Path] = {}
    edges: dict[str, set[str]] = defaultdict(set)
    for top, root in LIB_ROOTS:
        if not root.exists():
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
            for fn in filenames:
                if not fn.endswith(".py"):
                    continue
                f = Path(dirpath) / fn
                mod = file_to_module(top, root, f)
                module_file[mod] = f
                try:
                    tree = ast.parse(f.read_text(), filename=str(f))
                except (SyntaxError, OSError, UnicodeDecodeError):
                    continue
                for node in tree.body:
                    if isinstance(node, ast.Import):
                        for a in node.names:
                            top_dep = a.name.split(".")[0]
                            if top_dep in IN_REPO_TOP and top_dep != top_dep_of(mod):
                                edges[mod].add(a.name)
                            elif top_dep in IN_REPO_TOP:
                                edges[mod].add(a.name)
                    elif isinstance(node, ast.ImportFrom):
                        if node.level and node.level > 0:
                            # relative import — resolve against package
                            pkg_parts = mod.split(".")
                            if pkg_parts[-1]:  # module, not package
                                pkg_parts = pkg_parts[:-1]
                            base = pkg_parts[: len(pkg_parts) - (node.level - 1)] if node.level > 1 else pkg_parts
                            dep = ".".join(base + ([node.module] if node.module else []))
                            edges[mod].add(dep)
                        elif node.module:
                            top_dep = node.module.split(".")[0]
                            if top_dep in IN_REPO_TOP:
                                edges[mod].add(node.module)
    # Normalize edges: collapse to known module set. Map foo.bar.baz to nearest
    # ancestor that exists in module_file (so we count it as an edge to a real
    # module rather than an imported attribute).
    norm: dict[str, set[str]] = defaultdict(set)
    for src, dsts in edges.items():
        for d in dsts:
            parts = d.split(".")
            while parts and ".".join(parts) not in module_file:
                parts.pop()
            if parts:
                tgt = ".".join(parts)
                if tgt != src:
                    norm[src].add(tgt)
    return module_file, norm


def top_dep_of(m: str) -> str:
    return m.split(".")[0]


def sccs(graph: dict[str, set[str]]) -> list[list[str]]:
    """Tarjan's algorithm."""
    index_counter = [0]
    stack: list[str] = []
    lowlinks: dict[str, int] = {}
    index: dict[str, int] = {}
    on_stack: dict[str, bool] = {}
    result: list[list[str]] = []
    nodes = set(graph) | {t for ts in graph.values() for t in ts}

    sys.setrecursionlimit(20000)

    def strong(v: str) -> None:
        index[v] = index_counter[0]
        lowlinks[v] = index_counter[0]
        index_counter[0] += 1
        stack.append(v)
        on_stack[v] = True
        for w in graph.get(v, ()):
            if w not in index:
                strong(w)
                lowlinks[v] = min(lowlinks[v], lowlinks[w])
            elif on_stack.get(w):
                lowlinks[v] = min(lowlinks[v], index[w])
        if lowlinks[v] == index[v]:
            comp = []
            while True:
                w = stack.pop()
                on_stack[w] = False
                comp.append(w)
                if w == v:
                    break
            result.append(comp)

    for v in nodes:
        if v not in index:
            strong(v)
    return result


def main() -> int:
    module_file, graph = collect()
    comps = sccs(graph)
    bad = [c for c in comps if len(c) > 1]
    print(f"Modules: {len(module_file)}  Edges: {sum(len(v) for v in graph.values())}")
    print(f"SCCs with size > 1: {len(bad)}")
    for c in sorted(bad, key=lambda x: -len(x)):
        print(f"\n-- cycle of {len(c)} modules --")
        for m in sorted(c):
            f = module_file.get(m)
            print(f"  {m}  ({f.relative_to(REPO) if f else '?'})")
        # Show edges within the cycle
        cs = set(c)
        for m in sorted(c):
            internal = sorted(graph.get(m, set()) & cs)
            if internal:
                print(f"  {m} -> {internal}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
