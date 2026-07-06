# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Select which tests to run based on changed files.

Computes a precise test matrix from the diff between HEAD and a base ref using
top-level import analysis: only imports at module scope (not inside def/class bodies
or TYPE_CHECKING blocks) propagate test impact, matching the codebase's "no lazy
imports" convention.

Usage:
    python infra/ci/select_tests.py --base-ref <SHA>
    python infra/ci/select_tests.py --force-run-all
"""

import argparse
import ast
import json
import subprocess
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

# Ordered list of workspace member short names.
SCOPES: tuple[str, ...] = (
    "rigging",
    "haliax",
    "iris",
    "fray",
    "levanter",
    "zephyr",
    "marin",
)

# Files whose change triggers running every package's full test suite.
BROAD_TRIGGERS: frozenset[str] = frozenset(
    {
        "uv.lock",
        "pyproject.toml",
        "infra/ci/select_tests.py",
        ".github/workflows/unified-unit.yaml",
    }
)

# uv package names and pytest paths for each workspace scope.
UV_PACKAGE: dict[str, str] = {
    "rigging": "marin-rigging",
    "haliax": "marin-haliax",
    "iris": "marin-iris",
    "fray": "marin-fray",
    "levanter": "marin-levanter",
    "zephyr": "marin-zephyr",
    "marin": "marin-core",
}

# levanter's torch_test extra is intentionally omitted: torch/TPU legs stay in levanter-unit.yaml.
UV_EXTRAS: dict[str, list[str]] = {
    "marin": ["cpu", "dedup"],
}

TEST_DIR: dict[str, str] = {
    **{scope: f"lib/{scope}/tests" for scope in UV_PACKAGE if scope != "marin"},
    "marin": "tests",
}


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def top_level_imports(path: Path, module_name: str | None = None) -> set[str]:
    """Dotted module names from top-level import statements only.

    Skips imports inside def/class bodies and if TYPE_CHECKING blocks.
    Returns empty set on parse errors.
    """
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"), filename=str(path))
    except SyntaxError:
        return set()
    result: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                result.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level > 0:
                if module_name is not None:
                    # Determine parent package
                    if path.name == "__init__.py":
                        package = module_name
                    else:
                        package = module_name.rsplit(".", 1)[0] if "." in module_name else ""

                    parts = package.split(".") if package else []
                    up = node.level - 1
                    if up <= len(parts):
                        base_parts = parts[:-up] if up > 0 else parts
                        if node.module:
                            result.add(".".join(base_parts + node.module.split(".")))
                        else:
                            for alias in node.names:
                                result.add(".".join([*base_parts, alias.name]))
                else:
                    if node.module:
                        result.add(node.module)
            elif node.module:
                result.add(node.module)
    return result


def path_to_module(path: Path, scope: str, repo_root: Path) -> str | None:
    """Map a source .py file to its dotted module name, or None if not applicable.

    lib/levanter/src/levanter/store/cache.py -> levanter.store.cache
    """
    src_root = repo_root / f"lib/{scope}/src"
    try:
        rel = path.relative_to(src_root)
    except ValueError:
        return None
    parts = list(rel.with_suffix("").parts)
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts) if parts else None


def imports_touch_affected(imports: set[str], affected: set[str]) -> bool:
    """True if any imported name overlaps with an affected module.

    Three cases:
    - exact match: imp == aff
    - imp is a parent package: aff.startswith(imp + ".")  e.g. imp="levanter.store", aff="levanter.store.cache"
    - imp is a child of aff: imp.startswith(aff + ".")  e.g. imp="levanter.store.cache", aff="levanter.store"
      (Python executes __init__.py of every parent package when loading a child, so a change
      to levanter/store/__init__.py affects anything that imports levanter.store.cache)
    """
    for imp in imports:
        for aff in affected:
            if aff == imp or aff.startswith(imp + ".") or imp.startswith(aff + "."):
                return True
    return False


def downstream_modules(seeds: set[str], reverse: dict[str, set[str]]) -> set[str]:
    """BFS: all modules that transitively depend on any seed module."""
    visited = set(seeds)
    queue = list(seeds)
    while queue:
        mod = queue.pop()
        for dep in reverse.get(mod, ()):
            if dep not in visited:
                visited.add(dep)
                queue.append(dep)
    return visited


# ---------------------------------------------------------------------------
# Workspace graph
# ---------------------------------------------------------------------------


def build_reverse_deps(repo_root: Path) -> dict[str, set[str]]:
    """Build a top-level-import reverse-dependency map across all workspace source files.

    reverse[M] = {modules that import M at the top level of their source files}
    """
    reverse: dict[str, set[str]] = defaultdict(set)
    for scope in SCOPES:
        src = repo_root / f"lib/{scope}/src"
        if not src.exists():
            continue
        for py in src.rglob("*.py"):
            mod = path_to_module(py, scope, repo_root)
            if mod is None:
                continue
            for imp in top_level_imports(py, module_name=mod):
                reverse[imp].add(mod)
    return dict(reverse)


def all_test_files(scope: str, repo_root: Path) -> list[Path]:
    """All .py files in a scope's test directory, excluding conftest.py."""
    test_dir = repo_root / "tests" if scope == "marin" else repo_root / f"lib/{scope}/tests"
    if not test_dir.exists():
        return []
    return sorted(p for p in test_dir.rglob("*.py") if p.name != "conftest.py")


# ---------------------------------------------------------------------------
# Diff analysis
# ---------------------------------------------------------------------------


def git_changed_files(base_ref: str, repo_root: Path) -> list[str]:
    """Files changed between base_ref and HEAD (repo-root-relative POSIX paths)."""
    result = subprocess.run(
        ["git", "diff", "--name-only", f"{base_ref}...HEAD"],
        capture_output=True,
        text=True,
        cwd=repo_root,
        check=True,
    )
    return [line for line in result.stdout.splitlines() if line.strip()]


@dataclass(frozen=True)
class ClassifyResult:
    """Classification of repo-root-relative changed file paths."""

    broad: bool
    """True if any broad trigger was found (run everything)."""
    src_modules: set[str]
    """Dotted module names of changed source files."""
    direct_tests: dict[str, list[str]]
    """{scope: [repo-root-relative test file paths]}."""
    forced: set[str]
    """Scopes that must run their full test suite."""


def classify(
    changed_files: list[str],
    repo_root: Path,
) -> ClassifyResult:
    """Classify repo-root-relative changed file paths."""
    broad = False
    src_modules: set[str] = set()
    direct_tests: dict[str, list[str]] = defaultdict(list)
    forced: set[str] = set()

    for filepath in changed_files:
        p = Path(filepath)

        if filepath in BROAD_TRIGGERS:
            broad = True
            continue

        # Scope-specific paths
        for scope in SCOPES:
            src_prefix = f"lib/{scope}/src/"
            test_prefix = "tests/" if scope == "marin" else f"lib/{scope}/tests/"

            if filepath.startswith(src_prefix) and filepath.endswith(".py"):
                mod = path_to_module(repo_root / filepath, scope, repo_root)
                if mod:
                    src_modules.add(mod)
                break

            if filepath.startswith(test_prefix):
                if p.name == "conftest.py":
                    forced.add(scope)
                elif p.suffix == ".py":
                    direct_tests[scope].append(filepath)
                else:
                    # Non-Python test asset (snapshot, fixture, data file): run the full
                    # scope so the tests that own this file are not missed.
                    forced.add(scope)
                break

            # Per-package root files that can change test behavior
            if filepath in (f"lib/{scope}/conftest.py", f"lib/{scope}/pyproject.toml"):
                forced.add(scope)
                break

    return ClassifyResult(
        broad=broad,
        src_modules=src_modules,
        direct_tests=dict(direct_tests),
        forced=forced,
    )


# ---------------------------------------------------------------------------
# Test selection
# ---------------------------------------------------------------------------


def matrix_leg(scope: str, tests: list[str]) -> dict[str, str]:
    """Build one unified-unit matrix leg with uv/pytest arguments."""
    return {
        "package": UV_PACKAGE[scope],
        "extras": " ".join(f"--extra {extra}" for extra in UV_EXTRAS.get(scope, [])),
        "test_paths": " ".join(tests) if tests else TEST_DIR[scope],
    }


def compute_matrix(
    src_modules: set[str],
    direct_tests: dict[str, list[str]],
    forced_scopes: set[str],
    repo_root: Path,
) -> list[dict[str, str]]:
    """Compute the test matrix.

    Returns a list of matrix legs. Each leg has package (uv name), extras, and
    test_paths. An empty tests list means run the full suite directory.
    """
    if not (src_modules or direct_tests or forced_scopes):
        return []

    reverse = build_reverse_deps(repo_root)
    affected = downstream_modules(src_modules, reverse) if src_modules else set()

    forced_set = set(forced_scopes)
    matrix: list[dict[str, str]] = []

    for scope in SCOPES:
        if scope in forced_set:
            matrix.append(matrix_leg(scope, []))
            continue

        selected: list[str] = []

        for t in direct_tests.get(scope, []):
            if t not in selected:
                selected.append(t)

        if affected:
            for test_file in all_test_files(scope, repo_root):
                rel = str(test_file.relative_to(repo_root))
                if rel in selected:
                    continue
                if imports_touch_affected(top_level_imports(test_file), affected):
                    selected.append(rel)

        if selected:
            matrix.append(matrix_leg(scope, sorted(selected)))

    return matrix


def full_matrix() -> list[dict[str, str]]:
    """Matrix for --force-run-all: every scope with the full suite directory."""
    return [matrix_leg(scope, []) for scope in SCOPES]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Select tests to run from a diff against a base ref.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--base-ref", metavar="SHA", help="Git SHA or ref to diff HEAD against")
    mode.add_argument(
        "--force-run-all",
        action="store_true",
        help="Bypass the analyzer; run every package's full suite",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).parent.parent.parent

    if args.force_run_all:
        result = {"run_all": True, "reason": "force-run-all", "matrix": full_matrix()}
    else:
        changed = git_changed_files(args.base_ref, repo_root)
        classification = classify(changed, repo_root)
        if classification.broad:
            result = {"run_all": True, "reason": "broad-trigger", "matrix": full_matrix()}
        else:
            matrix = compute_matrix(
                classification.src_modules,
                classification.direct_tests,
                classification.forced,
                repo_root,
            )
            result = {"run_all": False, "reason": "diff-driven", "matrix": matrix}

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
