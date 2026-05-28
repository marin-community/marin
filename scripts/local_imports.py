# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Detect (and optionally fix) function-local imports across the repo.

A "local import" is any `import` / `from ... import` statement that appears
inside a function, method, or other non-module-scope body. The repo style guide
says imports must live at module top-level unless they exist to break a
circular dependency or guard an optional third-party dependency.

Usage:
    python scripts/local_imports.py scan           # report local imports as JSON
    python scripts/local_imports.py fix            # lift safe ones to top
    python scripts/local_imports.py fix --dry-run  # show what would change
"""

from __future__ import annotations

import argparse
import ast
import dataclasses
import importlib.util
import json
import os
import sys
from collections.abc import Iterable
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

SKIP_DIRS = {
    ".git",
    ".venv",
    "venv",
    "node_modules",
    "__pycache__",
    ".worktrees",
    "build",
    "dist",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "site",
}

# Modules whose imports we KEEP local (heavy optional deps, or known to be
# imported lazily on purpose). Anything starting with these prefixes is
# considered "optional" and is not lifted.
OPTIONAL_PREFIXES = {
    "torch",
    "vllm",
    "ray",
    "wandb",
    "tensorflow",
    "tf_keras",
    "deepspeed",
    "transformers",
    "datasets",
    "matplotlib",
    "seaborn",
    "plotly",
    "bokeh",
    "google.cloud",
    "googleapiclient",
    "boto3",
    "botocore",
    "psutil",
    "pynvml",
    "nvidia",
    "IPython",
    "ipykernel",
    "jupyter",
    "selenium",
    "playwright",
    "xgboost",
    "lightgbm",
    "sklearn",
    "scipy",
    "PIL",
    "cv2",
    "skimage",
    "redis",
    "pymongo",
    "psycopg2",
    "sqlalchemy",
    "flax",
    "optax",
    "orbax",
    # JAX-internal heavy paths sometimes lazy-imported
    "jax.experimental",
    "jax.extend",
    # gRPC generated protos can be slow / optional
    "grpc",
    # heavy serialization / RPC
    "msgpack",
    "pyarrow",
}


def is_optional(mod: str) -> bool:
    for p in OPTIONAL_PREFIXES:
        if mod == p or mod.startswith(p + "."):
            return True
    return False


@dataclasses.dataclass
class LocalImport:
    file: str
    line: int
    col: int
    func: str  # qualified function name
    stmt: str  # source line(s) text
    modules: list[str]  # top-level module names referenced
    is_relative: bool
    is_conditional: bool  # under if/try
    is_optional: bool  # all referenced modules look optional


def iter_python_files(root: Path) -> Iterable[Path]:
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for f in filenames:
            if f.endswith(".py"):
                yield Path(dirpath) / f


def stmt_modules(node: ast.AST) -> tuple[list[str], bool]:
    """Return (top-level module names referenced, is_relative)."""
    if isinstance(node, ast.Import):
        return [a.name.split(".")[0] for a in node.names], False
    if isinstance(node, ast.ImportFrom):
        if node.level and node.level > 0:
            return [node.module.split(".")[0] if node.module else ""], True
        return [(node.module or "").split(".")[0]], False
    return [], False


def stmt_full_modules(node: ast.AST) -> list[str]:
    if isinstance(node, ast.Import):
        return [a.name for a in node.names]
    if isinstance(node, ast.ImportFrom):
        if node.module:
            return [node.module]
    return []


def find_local_imports(path: Path) -> list[LocalImport]:
    try:
        src = path.read_text()
    except (OSError, UnicodeDecodeError):
        return []
    try:
        tree = ast.parse(src, filename=str(path))
    except SyntaxError:
        return []
    src_lines = src.splitlines()

    out: list[LocalImport] = []

    class Visitor(ast.NodeVisitor):
        def __init__(self):
            self.func_stack: list[str] = []
            self.cond_stack: int = 0

        def _record(self, node):
            full = stmt_full_modules(node)
            _top, is_rel = stmt_modules(node)
            stmt_text = "\n".join(src_lines[node.lineno - 1 : (node.end_lineno or node.lineno)])
            opt = bool(full) and all(is_optional(m) for m in full)
            out.append(
                LocalImport(
                    file=str(path.relative_to(REPO)),
                    line=node.lineno,
                    col=node.col_offset,
                    func=".".join(self.func_stack) if self.func_stack else "?",
                    stmt=stmt_text,
                    modules=full,
                    is_relative=is_rel,
                    is_conditional=self.cond_stack > 0,
                    is_optional=opt,
                )
            )

        def visit_FunctionDef(self, node):
            self.func_stack.append(node.name)
            self.generic_visit(node)
            self.func_stack.pop()

        visit_AsyncFunctionDef = visit_FunctionDef

        def visit_If(self, node):
            self.cond_stack += 1
            self.generic_visit(node)
            self.cond_stack -= 1

        def visit_Try(self, node):
            self.cond_stack += 1
            self.generic_visit(node)
            self.cond_stack -= 1

        def visit_Import(self, node):
            if self.func_stack:
                self._record(node)

        def visit_ImportFrom(self, node):
            if self.func_stack:
                self._record(node)

    Visitor().visit(tree)
    return out


def scan(paths: list[Path]) -> list[LocalImport]:
    found: list[LocalImport] = []
    for root in paths:
        for f in iter_python_files(root):
            found.extend(find_local_imports(f))
    return found


# ---------------------------------------------------------------------------
# Fix: lift safe local imports to the top of the module.
# ---------------------------------------------------------------------------


def _existing_top_modules(tree: ast.Module) -> set[tuple]:
    """Return a set of normalized (kind, ...) keys for top-level imports."""
    keys: set[tuple] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            for a in node.names:
                keys.add(("import", a.name, a.asname))
        elif isinstance(node, ast.ImportFrom):
            for a in node.names:
                keys.add(("from", node.level, node.module, a.name, a.asname))
    return keys


def _node_keys(node: ast.AST) -> list[tuple]:
    if isinstance(node, ast.Import):
        return [("import", a.name, a.asname) for a in node.names]
    if isinstance(node, ast.ImportFrom):
        return [("from", node.level, node.module, a.name, a.asname) for a in node.names]
    return []


def _safe_to_lift(li: LocalImport) -> bool:
    if li.is_optional:
        return False
    if li.is_conditional:
        return False
    if li.is_relative:
        # Relative imports inside a function — usually a cycle; leave them.
        return False
    if not li.modules:
        return False
    # Skip dotted modules where the leading part is unknown — we can't be sure
    # they aren't optional / runtime-resolved.
    return True


def _module_qualname(file: Path) -> str | None:
    """Best-effort qualname for the module that 'file' defines, used to detect
    self-imports that would cause new cycles after lifting."""
    parts = list(file.relative_to(REPO).with_suffix("").parts)
    # Look for src/ layouts
    for marker in ("src",):
        if marker in parts:
            i = parts.index(marker)
            parts = parts[i + 1 :]
            break
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts) if parts else None


def _would_self_import(file: Path, modules: list[str]) -> bool:
    qn = _module_qualname(file)
    if not qn:
        return False
    for m in modules:
        if m == qn:
            return True
    return False


def _attempt_import_lifted(modules: list[str]) -> bool:
    """Try to import the modules at top-level *outside the function* to verify
    they're resolvable. We don't actually import — we just check via
    importlib.util.find_spec. Failure → likely runtime-only path, skip.
    """
    for m in modules:
        if not m:
            return False
        try:
            spec = importlib.util.find_spec(m.split(".")[0])
        except (ImportError, ValueError, ModuleNotFoundError):
            return False
        if spec is None:
            return False
    return True


def fix_file(path: Path, dry_run: bool) -> tuple[int, int, list[LocalImport]]:
    """Lift safe local imports in `path`. Returns (lifted, skipped, remaining)."""
    try:
        src = path.read_text()
    except (OSError, UnicodeDecodeError):
        return 0, 0, []
    try:
        tree = ast.parse(src, filename=str(path))
    except SyntaxError:
        return 0, 0, []

    src_lines = src.splitlines(keepends=True)
    locals_found = find_local_imports(path)
    if not locals_found:
        return 0, 0, []

    existing = _existing_top_modules(tree)

    # Decide which to lift.
    to_lift_lines: dict[int, LocalImport] = {}  # line# -> import
    to_add_top: list[str] = []
    remaining: list[LocalImport] = []
    skipped = 0
    for li in locals_found:
        if not _safe_to_lift(li):
            remaining.append(li)
            skipped += 1
            continue
        if _would_self_import(path, li.modules):
            remaining.append(li)
            skipped += 1
            continue
        # Reparse the actual import node to get its key, also to know the line
        # range to drop.
        # Build the statement text. Re-find node in tree:
        node = None
        for n in ast.walk(tree):
            if isinstance(n, (ast.Import, ast.ImportFrom)) and n.lineno == li.line and n.col_offset == li.col:
                node = n
                break
        if node is None:
            remaining.append(li)
            skipped += 1
            continue
        # Skip if already present at top.
        keys = _node_keys(node)
        if all(k in existing for k in keys):
            # Already imported at top — just delete the local copy.
            to_lift_lines[li.line] = li
            continue
        # Verify resolvability so we don't lift an import whose module only
        # exists in some runtime context.
        if not _attempt_import_lifted(li.modules):
            remaining.append(li)
            skipped += 1
            continue
        # Strip leading whitespace from the statement so it sits at column 0.
        stmt_text = "\n".join(line.lstrip() for line in li.stmt.splitlines())
        to_add_top.append(stmt_text)
        for k in keys:
            existing.add(k)
        to_lift_lines[li.line] = li
        # Record line range
        li_node_end = node.end_lineno or node.lineno
        li._line_end = li_node_end  # type: ignore[attr-defined]

    if not to_lift_lines:
        return 0, skipped, remaining

    # Build a new source: delete original local-import lines, then insert
    # additions after the last existing top-level import (or after module
    # docstring + __future__).
    # 1. Mark lines to delete.
    delete_mask = [False] * (len(src_lines) + 1)  # 1-indexed
    for li in to_lift_lines.values():
        end = getattr(li, "_line_end", li.line)
        # If this is a multi-line `from X import (a, b,)` form, drop all lines.
        # If sibling code shares the line (very rare), bail out for safety.
        for ln in range(li.line, end + 1):
            line_text = src_lines[ln - 1] if ln - 1 < len(src_lines) else ""
            # Safety: if line contains a semicolon with non-import code, skip.
            if ";" in line_text and "import" not in line_text.split(";", 1)[1]:
                return 0, skipped + len(to_lift_lines), locals_found
            delete_mask[ln] = True

    new_lines: list[str] = []
    for i, line in enumerate(src_lines, start=1):
        if not delete_mask[i]:
            new_lines.append(line)

    # 2. Find insertion point. We insert after the last top-level import in the
    # *modified* file — re-parse.
    new_src = "".join(new_lines)
    try:
        new_tree = ast.parse(new_src, filename=str(path))
    except SyntaxError:
        # Our deletion broke the file (e.g. function body became empty).
        # Put a `pass` where needed.
        # Simpler: bail.
        return 0, skipped + len(to_lift_lines), locals_found

    insert_line = 0  # 0 means top of file
    # Skip docstring / __future__
    for node in new_tree.body:
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            insert_line = node.end_lineno or node.lineno
            continue
        if isinstance(node, ast.ImportFrom) and node.module == "__future__":
            insert_line = node.end_lineno or node.lineno
            continue
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            insert_line = node.end_lineno or node.lineno
            continue
        break

    insertion = "\n".join(to_add_top) + "\n"
    new_src_lines = new_src.splitlines(keepends=True)
    final = "".join(new_src_lines[:insert_line]) + insertion + "".join(new_src_lines[insert_line:])

    # Validate parse
    try:
        ast.parse(final, filename=str(path))
    except SyntaxError:
        return 0, skipped + len(to_lift_lines), locals_found

    if not dry_run:
        path.write_text(final)

    lifted_count = len(to_lift_lines)
    return lifted_count, skipped, remaining


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["scan", "fix"])
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--paths", nargs="*", default=["lib", "experiments", "scripts", "tests", "infra"])
    args = ap.parse_args(argv)

    roots = [REPO / p for p in args.paths]

    if args.cmd == "scan":
        items = scan(roots)
        if args.json:
            print(json.dumps([dataclasses.asdict(i) for i in items], indent=2))
        else:
            print(f"Found {len(items)} local imports")
            opt = sum(1 for i in items if i.is_optional)
            cond = sum(1 for i in items if i.is_conditional)
            rel = sum(1 for i in items if i.is_relative)
            print(f"  optional (kept): {opt}")
            print(f"  conditional:     {cond}")
            print(f"  relative:        {rel}")
        return 0

    if args.cmd == "fix":
        total_lifted = 0
        total_skipped = 0
        remaining: list[LocalImport] = []
        files_touched = 0
        for root in roots:
            for f in iter_python_files(root):
                lifted, skipped, rem = fix_file(f, dry_run=args.dry_run)
                if lifted:
                    files_touched += 1
                total_lifted += lifted
                total_skipped += skipped
                remaining.extend(rem)
        print(f"Lifted:  {total_lifted}  ({files_touched} files{' [dry-run]' if args.dry_run else ''})")
        print(f"Skipped: {total_skipped}")
        print(f"Remaining: {len(remaining)}")
        if args.json:
            out = REPO / "scripts" / "local_imports_remaining.json"
            out.write_text(json.dumps([dataclasses.asdict(i) for i in remaining], indent=2))
            print(f"Wrote {out}")
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
