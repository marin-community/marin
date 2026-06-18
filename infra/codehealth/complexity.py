#!/usr/bin/env -S uv run --script
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///

"""Static-complexity *leads* for the lint review's complexity lane.

Pure `ast` walk — no third-party deps. Computes per-function cyclomatic
complexity, max nesting depth and length; per-class method count and length;
and per-module length and direct-member count. `compute_leads` formats a
compact, bounded "COMPLEXITY LEADS" block over a set of changed files.

These are LEADS, not findings: they tell the complexity-lane agent where to
look, exactly as a profiler points at hot functions. The numeric floors below
only bound what we *show* — they are not thresholds the agent must flag, and the
agent stays free to flag a unit the leads missed or to ignore a flagged one that
is genuinely cohesive.
"""
from __future__ import annotations

import ast
import sys
from dataclasses import dataclass

# Floors that bound what we surface (NOT rule thresholds — see module docstring).
FUNCTION_LOC_FLOOR = 60
FUNCTION_CYCLOMATIC_FLOOR = 12
FUNCTION_NESTING_FLOOR = 4
CLASS_METHODS_FLOOR = 15
CLASS_LOC_FLOOR = 300
MODULE_LOC_FLOOR = 600
MODULE_MEMBERS_FLOOR = 20

MAX_LEADS_PER_FILE = 12

# A nested def/class is a separate unit; complexity/nesting walks stop at it.
_BOUNDARY = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
# Statements that introduce a nesting level.
_NESTING = (ast.If, ast.For, ast.AsyncFor, ast.While, ast.With, ast.AsyncWith, ast.Try)
# Nodes that add a cyclomatic decision point (weight 1 unless overridden below).
_DECISION = (ast.If, ast.IfExp, ast.For, ast.AsyncFor, ast.While, ast.ExceptHandler)


@dataclass(frozen=True)
class FunctionMetric:
    qualname: str
    lineno: int
    loc: int
    cyclomatic: int
    max_nesting: int


@dataclass(frozen=True)
class ClassMetric:
    name: str
    lineno: int
    loc: int
    methods: int


@dataclass(frozen=True)
class ModuleMetric:
    loc: int
    members: int


@dataclass(frozen=True)
class FileMetrics:
    path: str
    module: ModuleMetric
    functions: list[FunctionMetric]
    classes: list[ClassMetric]


def _own_nodes(node: ast.AST):
    """Yield descendants of node's body, stopping at nested def/class boundaries."""
    for child in ast.iter_child_nodes(node):
        if isinstance(child, _BOUNDARY):
            continue
        yield child
        yield from _own_nodes(child)


def _cyclomatic(fn: ast.AST) -> int:
    complexity = 1
    for n in _own_nodes(fn):
        if isinstance(n, ast.BoolOp):
            complexity += len(n.values) - 1
        elif isinstance(n, ast.comprehension):
            complexity += 1 + len(n.ifs)
        elif isinstance(n, ast.match_case):
            complexity += 1
        elif isinstance(n, _DECISION):
            complexity += 1
    return complexity


def _max_nesting(fn: ast.AST) -> int:
    deepest = 0

    def walk(node: ast.AST, depth: int) -> None:
        nonlocal deepest
        for child in ast.iter_child_nodes(node):
            if isinstance(child, _BOUNDARY):
                continue
            if isinstance(child, _NESTING):
                deepest = max(deepest, depth + 1)
                walk(child, depth + 1)
            else:
                walk(child, depth)

    walk(fn, 0)
    return deepest


def _loc(node: ast.AST) -> int:
    end = getattr(node, "end_lineno", None) or node.lineno
    return end - node.lineno + 1


def _qualname(stack: list[str], name: str) -> str:
    return ".".join([*stack, name])


def analyze_source(path: str, source: str) -> FileMetrics | None:
    """Parse `source` and return its metrics, or None if it does not parse."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        # Leads are best-effort over arbitrary changed files; an unparseable
        # file simply contributes no leads.
        return None

    functions: list[FunctionMetric] = []
    classes: list[ClassMetric] = []

    def visit(node: ast.AST, stack: list[str]) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                functions.append(
                    FunctionMetric(
                        qualname=_qualname(stack, child.name),
                        lineno=child.lineno,
                        loc=_loc(child),
                        cyclomatic=_cyclomatic(child),
                        max_nesting=_max_nesting(child),
                    )
                )
                visit(child, [*stack, child.name])
            elif isinstance(child, ast.ClassDef):
                methods = sum(isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef)) for m in child.body)
                classes.append(
                    ClassMetric(name=_qualname(stack, child.name), lineno=child.lineno, loc=_loc(child), methods=methods)
                )
                visit(child, [*stack, child.name])

    visit(tree, [])

    members = sum(
        isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Assign, ast.AnnAssign))
        for s in tree.body
    )
    module = ModuleMetric(loc=len(source.splitlines()), members=members)
    return FileMetrics(path=path, module=module, functions=functions, classes=classes)


def _notable_functions(fm: FileMetrics) -> list[FunctionMetric]:
    return [
        f
        for f in fm.functions
        if f.loc >= FUNCTION_LOC_FLOOR
        or f.cyclomatic >= FUNCTION_CYCLOMATIC_FLOOR
        or f.max_nesting >= FUNCTION_NESTING_FLOOR
    ]


def _notable_classes(fm: FileMetrics) -> list[ClassMetric]:
    return [c for c in fm.classes if c.methods >= CLASS_METHODS_FLOOR or c.loc >= CLASS_LOC_FLOOR]


def _module_is_notable(fm: FileMetrics) -> bool:
    return fm.module.loc >= MODULE_LOC_FLOOR or fm.module.members >= MODULE_MEMBERS_FLOOR


def _function_salience(f: FunctionMetric) -> float:
    return f.cyclomatic + f.loc / 20 + f.max_nesting * 2


def format_leads(metrics: list[FileMetrics]) -> str:
    """Render the bounded COMPLEXITY LEADS block, or '' if nothing is notable."""
    blocks: list[str] = []
    for fm in sorted(metrics, key=lambda m: m.path):
        lines: list[str] = []
        if _module_is_notable(fm):
            lines.append(f"  module: {fm.module.loc} loc, {fm.module.members} members")
        for c in sorted(_notable_classes(fm), key=lambda c: -c.loc):
            lines.append(f"  class {c.name} (L{c.lineno}): {c.methods} methods, {c.loc} loc")
        for f in sorted(_notable_functions(fm), key=_function_salience, reverse=True):
            lines.append(
                f"  fn {f.qualname} (L{f.lineno}): {f.loc} loc, cyclomatic {f.cyclomatic}, nesting {f.max_nesting}"
            )
        if not lines:
            continue
        blocks.append(fm.path + "\n" + "\n".join(lines[:MAX_LEADS_PER_FILE]))

    if not blocks:
        return ""
    header = (
        "COMPLEXITY LEADS (advisory hotspots for changed files — NOT findings, NOT thresholds; "
        "confirm each against the rule's intent before flagging, and feel free to ignore a "
        "cohesive unit or flag one not listed here):"
    )
    return header + "\n" + "\n".join(blocks)


def compute_leads(read_file, rel_paths: list[str]) -> str:
    """I/O wrapper: read each path via `read_file(path) -> str` and format leads.

    `read_file` is injected so the pure analysis stays testable; callers pass a
    function that reads the working-tree contents of a repo-relative path.
    """
    metrics: list[FileMetrics] = []
    for path in rel_paths:
        if not path.endswith(".py"):
            continue
        source = read_file(path)
        if source is None:
            continue
        fm = analyze_source(path, source)
        if fm is not None:
            metrics.append(fm)
    return format_leads(metrics)


if __name__ == "__main__":
    paths = sys.argv[1:]

    def _read(p: str) -> str | None:
        try:
            with open(p, encoding="utf-8") as fh:
                return fh.read()
        except OSError:
            return None

    out = compute_leads(_read, paths)
    print(out or "(no notable complexity leads)")
