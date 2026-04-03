#!/usr/bin/env -S uv run
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""
Generate keep-globs for high-value storage artifacts from experiment source.

This script does static analysis only:
- parses local Python files in `experiments/`
- follows local `experiments.*` imports from executor entry modules
- extracts candidate artifact paths from ExecutorStep outputs and pinned InputNames

It never opens objects in GCS.
"""

import argparse
import ast
import csv
import os
import re
from collections import deque
from dataclasses import dataclass
from pathlib import Path

DEFAULT_FOCUS_KEYWORDS = [
    "nemotron",
    "marin-8b",
    "marin_8b",
    "marin-32b",
    "32b",
    "tootsie-32b",
]

TYPO_ALIASES = {
    "raw/nemotro-cc": "raw/nemotron-cc",
}

SPECIAL_PATH_REWRITES = {
    "tokenized/*_marin_tokenizer": [
        "tokenized/smoltalk2_*_marin_tokenizer",
        "tokenized/nemotron_v2_*_marin_tokenizer",
    ],
    "tokenized/*_marin_tokenizer*": [
        "tokenized/smoltalk2_*_marin_tokenizer*",
        "tokenized/nemotron_v2_*_marin_tokenizer*",
    ],
}

STEP_BUILDERS = {
    "ExecutorStep": "",
    "default_train": "checkpoints",
    "default_sft": "checkpoints",
    "default_dpo": "checkpoints",
    "default_tokenize": "tokenized",
    "default_download": "",
}


@dataclass(frozen=True)
class PathRef:
    path: str


@dataclass(frozen=True)
class ArtifactCandidate:
    path: str
    source_kind: str
    module: str
    line: int
    reason: str


@dataclass
class ModuleInfo:
    path: Path
    tree: ast.Module
    imports: set[str]
    is_executor_entry: bool


def _module_name(root_dir: Path, file_path: Path) -> str:
    rel = file_path.relative_to(root_dir)
    parts = list(rel.parts)
    parts[-1] = parts[-1][:-3]  # strip ".py"
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join([root_dir.name, *parts]).rstrip(".")


def _build_module_index(root_dir: Path) -> dict[str, Path]:
    module_index: dict[str, Path] = {}
    for path in root_dir.rglob("*.py"):
        module_index[_module_name(root_dir, path)] = path
    return module_index


def _nearest_module(name: str, module_index: dict[str, Path]) -> str | None:
    parts = name.split(".")
    for i in range(len(parts), 0, -1):
        candidate = ".".join(parts[:i])
        if candidate in module_index:
            return candidate
    return None


def _resolve_from_base(module_name: str, node: ast.ImportFrom) -> str:
    if node.level == 0:
        return node.module or ""

    # Relative import resolution.
    package_parts = module_name.split(".")[:-1]
    up = node.level - 1
    if up > len(package_parts):
        base_parts: list[str] = []
    else:
        base_parts = package_parts[: len(package_parts) - up]

    if node.module:
        base_parts.extend(node.module.split("."))
    return ".".join(part for part in base_parts if part)


def _collect_imports(module_name: str, tree: ast.Module, module_index: dict[str, Path]) -> set[str]:
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if not alias.name.startswith("experiments"):
                    continue
                nearest = _nearest_module(alias.name, module_index)
                if nearest is not None:
                    imports.add(nearest)
        elif isinstance(node, ast.ImportFrom):
            base = _resolve_from_base(module_name, node)
            if not base.startswith("experiments"):
                continue

            nearest_base = _nearest_module(base, module_index)
            if nearest_base is not None:
                imports.add(nearest_base)

            for alias in node.names:
                if alias.name == "*":
                    continue
                candidate = f"{base}.{alias.name}" if base else alias.name
                nearest = _nearest_module(candidate, module_index)
                if nearest is not None:
                    imports.add(nearest)
    return imports


def _is_executor_entry(tree: ast.Module) -> bool:
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            fn_name = _call_name(node.func)
            if fn_name == "executor_main":
                return True
    return False


def _parse_modules(root_dir: Path, module_index: dict[str, Path]) -> dict[str, ModuleInfo]:
    modules: dict[str, ModuleInfo] = {}
    for module_name, module_path in module_index.items():
        try:
            tree = ast.parse(module_path.read_text(), filename=str(module_path))
        except SyntaxError:
            continue
        imports = _collect_imports(module_name, tree, module_index)
        modules[module_name] = ModuleInfo(
            path=module_path,
            tree=tree,
            imports=imports,
            is_executor_entry=_is_executor_entry(tree),
        )
    return modules


def _reachable_modules(modules: dict[str, ModuleInfo]) -> set[str]:
    roots = [name for name, info in modules.items() if info.is_executor_entry]
    if not roots:
        return set(modules.keys())

    reachable: set[str] = set()
    queue = deque(roots)
    while queue:
        current = queue.popleft()
        if current in reachable:
            continue
        reachable.add(current)
        for dep in modules[current].imports:
            if dep not in reachable and dep in modules:
                queue.append(dep)
    return reachable


def _call_name(fn: ast.expr) -> str | None:
    if isinstance(fn, ast.Name):
        return fn.id
    if isinstance(fn, ast.Attribute):
        return fn.attr
    return None


def _kwarg(call: ast.Call, name: str) -> ast.expr | None:
    for kw in call.keywords:
        if kw.arg == name:
            return kw.value
    return None


def _eval_literal(node: ast.expr, env: dict[str, object]) -> object | None:
    if isinstance(node, ast.Constant):
        if isinstance(node.value, str | int):
            return node.value
        return None
    if isinstance(node, ast.Name):
        return env.get(node.id)
    if isinstance(node, ast.JoinedStr):
        chunks: list[str] = []
        for value in node.values:
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                chunks.append(value.value)
            elif isinstance(value, ast.FormattedValue):
                inner = _eval_literal(value.value, env)
                chunks.append(str(inner) if isinstance(inner, str | int) else "*")
            else:
                chunks.append("*")
        return "".join(chunks)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left = _eval_literal(node.left, env)
        right = _eval_literal(node.right, env)
        if isinstance(left, str) and isinstance(right, str):
            return left + right
        return None
    if isinstance(node, ast.Call):
        fn = node.func
        if isinstance(fn, ast.Attribute) and fn.attr == "join":
            value = fn.value
            if isinstance(value, ast.Attribute) and value.attr == "path" and isinstance(value.value, ast.Name):
                if value.value.id == "os":
                    parts: list[str] = []
                    for arg in node.args:
                        evaluated = _eval_literal(arg, env)
                        if isinstance(evaluated, str | int):
                            parts.append(str(evaluated))
                        else:
                            parts.append("*")
                    return os.path.join(*parts)
    if isinstance(node, ast.Subscript):
        base = _eval_literal(node.value, env)
        key = _eval_literal(node.slice, env)
        if isinstance(base, dict) and isinstance(key, str):
            return base.get(key)
    if isinstance(node, ast.Dict):
        out: dict[str, object] = {}
        for key_node, value_node in zip(node.keys, node.values, strict=False):
            if key_node is None:
                return None
            key = _eval_literal(key_node, env)
            value = _eval_literal(value_node, env)
            if not isinstance(key, str):
                return None
            out[key] = value
        return out
    return None


def _eval_string(node: ast.expr, env: dict[str, object]) -> str | None:
    value = _eval_literal(node, env)
    if isinstance(value, str):
        return value
    if isinstance(value, int):
        return str(value)
    return None


def _join_path(base: str, suffix: str | None) -> str:
    if not suffix:
        return base
    return os.path.join(base, suffix)


def _eval_path_ref(node: ast.expr, env: dict[str, object]) -> PathRef | dict[str, PathRef] | None:
    if isinstance(node, ast.Name):
        value = env.get(node.id)
        if isinstance(value, PathRef | dict):
            return value
        return None

    if isinstance(node, ast.Subscript):
        base = _eval_path_ref(node.value, env)
        key = _eval_literal(node.slice, env)
        if isinstance(base, dict) and isinstance(key, str):
            return base.get(key)
        return None

    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
        left = _eval_path_ref(node.left, env)
        right = _eval_string(node.right, env)
        if isinstance(left, PathRef) and right is not None:
            return PathRef(_join_path(left.path, right))
        return None

    if not isinstance(node, ast.Call):
        return None

    fn_name = _call_name(node.func)
    if fn_name in STEP_BUILDERS:
        name_expr = _kwarg(node, "name")
        if name_expr is None and node.args:
            name_expr = node.args[0]
        name = _eval_string(name_expr, env) if name_expr is not None else None
        if name is None:
            return None

        prefix = STEP_BUILDERS[fn_name]
        if prefix:
            path = _join_path(prefix, name)
        else:
            path = name

        override_expr = _kwarg(node, "override_output_path")
        if override_expr is not None:
            override_path = _eval_string(override_expr, env)
            if override_path is not None:
                path = override_path
        return PathRef(path)

    if fn_name == "output_path_of":
        if not node.args:
            return None
        base = _eval_path_ref(node.args[0], env)
        if not isinstance(base, PathRef):
            return None
        suffix_expr = node.args[1] if len(node.args) > 1 else None
        suffix = _eval_string(suffix_expr, env) if suffix_expr is not None else None
        return PathRef(_join_path(base.path, suffix))

    if isinstance(node.func, ast.Attribute):
        attr = node.func.attr
        owner_ref = _eval_path_ref(node.func.value, env)
        if attr == "hardcoded":
            if isinstance(node.func.value, ast.Name) and node.func.value.id == "InputName" and node.args:
                path = _eval_string(node.args[0], env)
                if path is not None:
                    return PathRef(path)
        if attr == "cd" and isinstance(owner_ref, PathRef) and node.args:
            suffix = _eval_string(node.args[0], env)
            if suffix is not None:
                return PathRef(_join_path(owner_ref.path, suffix))
        if attr == "with_output_path" and isinstance(owner_ref, PathRef) and node.args:
            override_path = _eval_string(node.args[0], env)
            if override_path is not None:
                return PathRef(override_path)
        if attr in {"nonblocking", "as_input_name"} and isinstance(owner_ref, PathRef):
            return owner_ref

    return None


def _assign_target(target: ast.expr, value: object, env: dict[str, object]) -> None:
    if isinstance(target, ast.Name):
        env[target.id] = value
        return
    if isinstance(target, ast.Tuple | ast.List) and isinstance(value, tuple | list):
        for sub_target, sub_value in zip(target.elts, value, strict=False):
            _assign_target(sub_target, sub_value, env)


def _eval_iterable(node: ast.expr, env: dict[str, object]) -> list[object] | None:
    if isinstance(node, ast.Name):
        value = env.get(node.id)
        if isinstance(value, dict):
            return list(value.keys())
        if isinstance(value, list | tuple):
            return list(value)
        return None

    if isinstance(node, ast.List | ast.Tuple):
        values: list[object] = []
        for elt in node.elts:
            value = _eval_literal(elt, env)
            if value is None:
                return None
            values.append(value)
        return values

    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
        owner = _eval_literal(node.func.value, env)
        if isinstance(owner, dict):
            if node.func.attr == "keys":
                return list(owner.keys())
            if node.func.attr == "values":
                return list(owner.values())
            if node.func.attr == "items":
                return list(owner.items())
        if isinstance(owner, list | tuple):
            if node.func.attr == "__iter__":
                return list(owner)

    return None


def _classify_path(path: str) -> str | None:
    p = path.lower()
    if "checkpoints" in p or "/hf/step-" in p or "hf/step-" in p:
        return "checkpoint"
    if p.startswith("raw/") or "/raw/" in p or "tokenized" in p or "dataset" in p:
        return "dataset"
    return None


def _normalize_to_glob(path: str, bucket_glob: str) -> str:
    if path.startswith(("gs://", "mirror://", "s3://")):
        resolved = path
    else:
        resolved = f"{bucket_glob.rstrip('/')}/{path.lstrip('/')}"
    if resolved.endswith("*"):
        return resolved
    if resolved.endswith("/"):
        return resolved + "*"
    return resolved + "*"


def _expand_alias_paths(path: str) -> list[str]:
    paths = [path]
    for typo, corrected in TYPO_ALIASES.items():
        if typo in path:
            paths.append(path.replace(typo, corrected))
    rewritten: list[str] = []
    for candidate in paths:
        if candidate in SPECIAL_PATH_REWRITES:
            rewritten.extend(SPECIAL_PATH_REWRITES[candidate])
        else:
            rewritten.append(candidate)
    return rewritten


def _priority(path: str, kind: str, source_kind: str, focus_keywords: list[str]) -> int:
    score = 0
    if kind == "checkpoint":
        score += 80
    elif kind == "dataset":
        score += 70
    if source_kind == "pinned_input_name":
        score += 30
    elif source_kind == "executor_output":
        score += 20

    lower = path.lower()
    for keyword in focus_keywords:
        if keyword.lower() in lower:
            score += 40
    return score


def _is_overbroad_checkpoint_glob(path_glob: str) -> bool:
    marker = "/checkpoints/"
    if marker not in path_glob:
        return False

    tail = path_glob.split(marker, 1)[1]
    if not tail:
        return True

    first_segment = tail.split("/", 1)[0]
    if not first_segment:
        return True

    # If the first segment after checkpoints is purely wildcard/punctuation,
    # this is too broad for a keep-list purge pass.
    literal = re.sub(r"[*?\[\]{}]", "", first_segment).strip("._-")
    return literal == "" or first_segment.startswith("*")


def _maybe_add_candidate(
    candidates: list[ArtifactCandidate],
    ref: PathRef | None,
    source_kind: str,
    module: str,
    line: int,
    reason: str,
) -> None:
    if ref is None:
        return
    kind = _classify_path(ref.path)
    if kind is None:
        return
    candidates.append(
        ArtifactCandidate(
            path=ref.path,
            source_kind=source_kind,
            module=module,
            line=line,
            reason=reason,
        )
    )


def _scan_for_candidates(
    module_name: str,
    tree: ast.Module,
    initial_env: dict[str, object],
) -> list[ArtifactCandidate]:
    candidates: list[ArtifactCandidate] = []
    env = dict(initial_env)

    def scan_expr(node: ast.expr, scope: dict[str, object]) -> None:
        if isinstance(node, ast.Call):
            fn_name = _call_name(node.func)
            if fn_name in STEP_BUILDERS:
                ref = _eval_path_ref(node, scope)
                _maybe_add_candidate(
                    candidates,
                    ref if isinstance(ref, PathRef) else None,
                    "executor_output",
                    module_name,
                    node.lineno,
                    f"{fn_name} output",
                )
            elif fn_name == "output_path_of":
                ref = _eval_path_ref(node, scope)
                _maybe_add_candidate(
                    candidates,
                    ref if isinstance(ref, PathRef) else None,
                    "executor_output",
                    module_name,
                    node.lineno,
                    "output_path_of reference",
                )
            elif isinstance(node.func, ast.Attribute) and node.func.attr == "hardcoded":
                ref = _eval_path_ref(node, scope)
                _maybe_add_candidate(
                    candidates,
                    ref if isinstance(ref, PathRef) else None,
                    "pinned_input_name",
                    module_name,
                    node.lineno,
                    "InputName.hardcoded",
                )
            elif isinstance(node.func, ast.Attribute) and node.func.attr in {"cd", "with_output_path"}:
                ref = _eval_path_ref(node, scope)
                _maybe_add_candidate(
                    candidates,
                    ref if isinstance(ref, PathRef) else None,
                    "executor_output",
                    module_name,
                    node.lineno,
                    f".{node.func.attr}(...) path",
                )

        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.expr):
                scan_expr(child, scope)

    def scan_stmt(stmt: ast.stmt, scope: dict[str, object]) -> None:
        if isinstance(stmt, ast.Assign):
            scan_expr(stmt.value, scope)
            value_ref = _eval_path_ref(stmt.value, scope)
            value_lit = _eval_literal(stmt.value, scope) if value_ref is None else value_ref
            for target in stmt.targets:
                _assign_target(target, value_lit, scope)
            return

        if isinstance(stmt, ast.AnnAssign):
            if stmt.value is not None:
                scan_expr(stmt.value, scope)
                value_ref = _eval_path_ref(stmt.value, scope)
                value_lit = _eval_literal(stmt.value, scope) if value_ref is None else value_ref
                _assign_target(stmt.target, value_lit, scope)
            return

        if isinstance(stmt, ast.For):
            scan_expr(stmt.iter, scope)
            iterable = _eval_iterable(stmt.iter, scope)
            if iterable is None:
                body_scope = dict(scope)
                for body_stmt in stmt.body:
                    scan_stmt(body_stmt, body_scope)
            else:
                for item in iterable:
                    body_scope = dict(scope)
                    _assign_target(stmt.target, item, body_scope)
                    for body_stmt in stmt.body:
                        scan_stmt(body_stmt, body_scope)
            for else_stmt in stmt.orelse:
                scan_stmt(else_stmt, dict(scope))
            return

        if isinstance(stmt, ast.If):
            scan_expr(stmt.test, scope)
            then_scope = dict(scope)
            for body_stmt in stmt.body:
                scan_stmt(body_stmt, then_scope)
            else_scope = dict(scope)
            for else_stmt in stmt.orelse:
                scan_stmt(else_stmt, else_scope)
            return

        if isinstance(stmt, ast.FunctionDef | ast.AsyncFunctionDef):
            fn_scope = dict(scope)
            for body_stmt in stmt.body:
                scan_stmt(body_stmt, fn_scope)
            return

        if isinstance(stmt, ast.ClassDef):
            class_scope = dict(scope)
            for body_stmt in stmt.body:
                scan_stmt(body_stmt, class_scope)
            return

        for child in ast.iter_child_nodes(stmt):
            if isinstance(child, ast.stmt):
                scan_stmt(child, scope)
            elif isinstance(child, ast.expr):
                scan_expr(child, scope)

    for statement in tree.body:
        scan_stmt(statement, env)
    return candidates


def _extract_candidates_for_module(module_name: str, info: ModuleInfo) -> list[ArtifactCandidate]:
    return _scan_for_candidates(module_name, info.tree, initial_env={})


def generate_csv(
    experiments_dir: Path,
    output_csv: Path,
    bucket_glob: str,
    focus_keywords: list[str],
    min_priority: int,
) -> tuple[int, int]:
    module_index = _build_module_index(experiments_dir)
    modules = _parse_modules(experiments_dir, module_index)
    reachable = _reachable_modules(modules)

    all_candidates: list[ArtifactCandidate] = []
    for module_name in sorted(reachable):
        info = modules[module_name]
        all_candidates.extend(_extract_candidates_for_module(module_name, info))

    dedup: dict[str, dict[str, str | int]] = {}
    for candidate in all_candidates:
        kind = _classify_path(candidate.path)
        if kind is None:
            continue
        for path_variant in _expand_alias_paths(candidate.path):
            path_glob = _normalize_to_glob(path_variant, bucket_glob)
            score = _priority(path_glob, kind, candidate.source_kind, focus_keywords)
            if score < min_priority:
                continue
            if kind == "checkpoint" and _is_overbroad_checkpoint_glob(path_glob):
                continue

            existing = dedup.get(path_glob)
            if existing is None or score > int(existing["priority"]):
                dedup[path_glob] = {
                    "path_glob": path_glob,
                    "artifact_kind": kind,
                    "priority": score,
                    "source_kind": candidate.source_kind,
                    "reason": candidate.reason,
                    "module": candidate.module,
                    "line": candidate.line,
                }

    rows = sorted(dedup.values(), key=lambda row: (-int(row["priority"]), str(row["path_glob"])))
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "path_glob",
                "artifact_kind",
                "priority",
                "source_kind",
                "reason",
                "module",
                "line",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    return len(reachable), len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate high-value keep-globs from experiments source (no GCS reads)."
    )
    parser.add_argument(
        "--experiments-dir",
        type=Path,
        default=Path("experiments"),
        help="Directory to crawl for experiment modules.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("scripts/storage/high_value_keep_globs.csv"),
        help="Output CSV path.",
    )
    parser.add_argument(
        "--bucket-glob",
        default="gs://marin-*",
        help="Prefix used for relative artifact paths.",
    )
    parser.add_argument(
        "--focus-keyword",
        action="append",
        default=[],
        help="Keyword to prioritize. Repeat flag to add multiple.",
    )
    parser.add_argument(
        "--min-priority",
        type=int,
        default=0,
        help="Minimum priority score required to keep a row.",
    )
    args = parser.parse_args()

    focus_keywords = args.focus_keyword if args.focus_keyword else DEFAULT_FOCUS_KEYWORDS
    reachable_count, row_count = generate_csv(
        experiments_dir=args.experiments_dir,
        output_csv=args.output_csv,
        bucket_glob=args.bucket_glob,
        focus_keywords=focus_keywords,
        min_priority=args.min_priority,
    )
    print(f"Analyzed {reachable_count} experiment modules.")
    print(f"Wrote {row_count} keep-glob rows to {args.output_csv}.")


if __name__ == "__main__":
    main()
