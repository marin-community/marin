# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Select which tests to run based on changed files.

Builds a module-level import graph over the workspace, walks it backwards from the
changed modules, and emits the test files that transitively import them.

Two rules shape the graph:

- Only imports at module scope propagate. The codebase forbids lazy imports, so an
  import inside a function body is assumed not to affect what a test exercises.
- ``import a.b`` depends on ``a`` as well as ``a.b``, because Python executes
  ``a/__init__.py`` on the way in. A package whose ``__init__`` re-exports its
  submodules therefore ties every importer to all of them.

Test helper modules under a test tree participate in the graph too, so a test that
reaches source code only through a shared helper is still selected.

Usage:
    python infra/ci/select_tests.py --base-ref <SHA>                   # pull request
    python infra/ci/select_tests.py --base-ref <SHA> --run-all-tests   # push to main
    python infra/ci/select_tests.py --run-all-tests                    # manual run
"""

import argparse
import ast
import json
import subprocess
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

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


@dataclass(frozen=True)
class SourceRoot:
    """A top-level package and the directory that must be importable for it to resolve."""

    package_dir: str
    """Repo-relative directory holding the package, e.g. ``lib/levanter/src/levanter``."""
    import_root: str
    """Repo-relative directory on ``sys.path``, e.g. ``lib/levanter/src``."""


SOURCE_ROOTS: tuple[SourceRoot, ...] = (
    *(SourceRoot(f"lib/{scope}/src/{scope}", f"lib/{scope}/src") for scope in SCOPES),
    SourceRoot("experiments", "."),
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

UV_EXTRAS: dict[str, list[str]] = {
    "marin": ["cpu", "dedup"],
}

TEST_DIR: dict[str, str] = {
    **{scope: f"lib/{scope}/tests" for scope in UV_PACKAGE if scope != "marin"},
    "marin": "tests",
}

# Suites that cannot be import-selected: each drives a whole subsystem (accelerator
# kernels, a browser-driven smoke test) rather than a set of importable modules, so
# path prefixes gate them. A locked dependency change moves the accelerator runtime
# out from under all of them.
DEPENDENCY_MANIFESTS: tuple[str, ...] = ("uv.lock", "pyproject.toml")
EXTRA_SUITE_TRIGGERS: dict[str, tuple[str, ...]] = {
    "levanter-torch": ("lib/levanter/", "lib/haliax/", *DEPENDENCY_MANIFESTS),
    "levanter-tpu": ("lib/levanter/", "lib/haliax/", *DEPENDENCY_MANIFESTS),
    "iris-e2e-smoke": ("lib/iris/", *DEPENDENCY_MANIFESTS),
}


# ---------------------------------------------------------------------------
# Import parsing
# ---------------------------------------------------------------------------


def path_to_module(path: Path, import_root: Path) -> str | None:
    """Dotted module name for a .py file, or None if it is outside ``import_root``.

    lib/levanter/src/levanter/store/cache.py -> levanter.store.cache
    """
    try:
        rel = path.relative_to(import_root)
    except ValueError:
        return None
    parts = list(rel.with_suffix("").parts)
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts) if parts else None


def ancestors(dotted: str) -> list[str]:
    """Every dotted prefix of a module name, shortest first: a.b.c -> [a, a.b, a.b.c]."""
    parts = dotted.split(".")
    return [".".join(parts[: i + 1]) for i in range(len(parts))]


def _absolute_base(node: ast.ImportFrom, module_name: str, is_package: bool) -> str:
    """Absolute dotted prefix named by a ``from ... import`` statement."""
    if node.level == 0:
        return node.module or ""
    package = module_name if is_package else module_name.rsplit(".", 1)[0] if "." in module_name else ""
    parts = package.split(".") if package else []
    up = node.level - 1
    if up > len(parts):
        return ""
    base_parts = parts[: len(parts) - up]
    return ".".join(base_parts + (node.module.split(".") if node.module else []))


def imported_names(path: Path, module_name: str) -> set[str]:
    """Absolute dotted names referenced by top-level import statements.

    ``from a.b import c`` yields both ``a.b`` and the candidate ``a.b.c``: the caller
    decides which of those is a real module. A file that will not parse would silently
    drop its edges from the graph, under-selecting tests, so the SyntaxError propagates.
    """
    tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"), filename=str(path))

    names: set[str] = set()
    is_package = path.name == "__init__.py"
    for node in tree.body:
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = _absolute_base(node, module_name, is_package)
            if not base:
                continue
            names.add(base)
            names.update(f"{base}.{alias.name}" for alias in node.names)
    return names


def resolve(names: set[str], known: set[str]) -> set[str]:
    """Known modules whose execution the given import names trigger, ancestors included."""
    return {ancestor for name in names for ancestor in ancestors(name) if ancestor in known}


# ---------------------------------------------------------------------------
# Workspace graph
# ---------------------------------------------------------------------------


def workspace_modules(repo_root: Path) -> dict[str, Path]:
    """Every importable workspace source module, dotted name -> file."""
    modules: dict[str, Path] = {}
    for source_root in SOURCE_ROOTS:
        package = repo_root / source_root.package_dir
        if not package.exists():
            continue
        import_root = repo_root / source_root.import_root
        for py in package.rglob("*.py"):
            module = path_to_module(py, import_root)
            if module:
                modules[module] = py
    return modules


def build_importers(modules: dict[str, Path]) -> dict[str, set[str]]:
    """importers[M] = modules whose top-level imports execute M."""
    known = set(modules)
    importers: dict[str, set[str]] = defaultdict(set)
    for module, py in modules.items():
        for dependency in resolve(imported_names(py, module), known):
            if dependency != module:
                importers[dependency].add(module)
    return dict(importers)


def affected_modules(seeds: set[str], importers: dict[str, set[str]]) -> set[str]:
    """BFS: the seeds plus every module that transitively imports one."""
    visited = set(seeds)
    queue = list(seeds)
    while queue:
        module = queue.pop()
        for importer in importers.get(module, ()):
            if importer not in visited:
                visited.add(importer)
                queue.append(importer)
    return visited


# ---------------------------------------------------------------------------
# Test trees
# ---------------------------------------------------------------------------


def is_test_module(filename: str) -> bool:
    """Whether pytest would collect this file by name (default ``python_files`` convention).

    Only such files may be passed to pytest explicitly: an explicit path is imported even
    when it does not match the collection convention, so handing pytest a helper module
    (workload script, stub, generator) crashes the run if the helper's imports are not
    installed in the lane's environment.
    """
    return (filename.startswith("test_") or filename.endswith("_test.py")) and filename.endswith(".py")


def _test_tree(scope: str, repo_root: Path) -> dict[str, Path]:
    """Every .py under a scope's test directory, keyed by the name it imports itself as.

    Test trees are imported as the ``tests`` package rooted at the test directory's parent,
    which is what both relative (``from .conftest import x``) and absolute
    (``from tests.cluster.conftest import x``) intra-tree imports resolve against.
    """
    test_dir = repo_root / TEST_DIR[scope]
    if not test_dir.exists():
        return {}
    import_root = repo_root / PurePosixPath(TEST_DIR[scope]).parent
    tree: dict[str, Path] = {}
    for py in test_dir.rglob("*.py"):
        module = path_to_module(py, import_root)
        if module:
            tree[module] = py
    return tree


def _tree_dependencies(
    module: str,
    tree: dict[str, Path],
    known: set[str],
    cache: dict[str, set[str]],
    visiting: set[str],
) -> set[str]:
    """Workspace modules a test-tree module depends on, following intra-tree helpers."""
    if module in cache:
        return cache[module]
    if module in visiting:
        return set()  # import cycle between helpers
    visiting.add(module)

    names = imported_names(tree[module], module)
    dependencies = resolve(names, known)
    for name in names:
        for ancestor in ancestors(name):
            if ancestor in tree and ancestor != module:
                dependencies |= _tree_dependencies(ancestor, tree, known, cache, visiting)

    visiting.discard(module)
    cache[module] = dependencies
    return dependencies


def dependencies_by_test_file(scope: str, repo_root: Path, known: set[str]) -> dict[str, set[str]]:
    """Collectable test file (repo-relative) -> workspace modules it transitively imports."""
    tree = _test_tree(scope, repo_root)
    cache: dict[str, set[str]] = {}
    return {
        str(py.relative_to(repo_root)): _tree_dependencies(module, tree, known, cache, set())
        for module, py in tree.items()
        if is_test_module(py.name)
    }


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


def classify(changed_files: list[str], repo_root: Path) -> ClassifyResult:
    """Classify repo-root-relative changed file paths."""
    broad = False
    src_modules: set[str] = set()
    direct_tests: dict[str, list[str]] = defaultdict(list)
    forced: set[str] = set()

    for filepath in changed_files:
        if filepath in BROAD_TRIGGERS:
            broad = True
            continue

        source_root = next(
            (root for root in SOURCE_ROOTS if filepath.startswith(f"{root.package_dir}/")),
            None,
        )
        if source_root is not None:
            if filepath.endswith(".py"):
                module = path_to_module(repo_root / filepath, repo_root / source_root.import_root)
                if module:
                    src_modules.add(module)
            continue

        for scope in SCOPES:
            if filepath.startswith(f"{TEST_DIR[scope]}/"):
                # conftest.py, helper modules (stubs, workload scripts, generators), and
                # non-Python assets (snapshots, fixtures, data files) can all change test
                # behavior without being directly collectable: run the full scope so the
                # tests that own this file are not missed.
                if not is_test_module(PurePosixPath(filepath).name):
                    forced.add(scope)
                elif (repo_root / filepath).exists():
                    # A test deleted by this diff still shows up in git's output; passing
                    # it to pytest would abort the run before a single test executes.
                    direct_tests[scope].append(filepath)
                break

            if filepath in (f"lib/{scope}/conftest.py", f"lib/{scope}/pyproject.toml"):
                forced.add(scope)
                break

    return ClassifyResult(
        broad=broad,
        src_modules=src_modules,
        direct_tests=dict(direct_tests),
        forced=forced,
    )


def extra_suites(changed_files: list[str]) -> list[str]:
    """Out-of-band suites to run for this diff."""
    return sorted(
        suite
        for suite, prefixes in EXTRA_SUITE_TRIGGERS.items()
        if any(filepath.startswith(prefix) for prefix in prefixes for filepath in changed_files)
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

    modules = workspace_modules(repo_root)
    known = set(modules)
    affected = affected_modules(src_modules, build_importers(modules)) if src_modules else set()

    matrix: list[dict[str, str]] = []
    for scope in SCOPES:
        if scope in forced_scopes:
            matrix.append(matrix_leg(scope, []))
            continue

        selected = list(direct_tests.get(scope, []))
        if affected:
            for test_file, dependencies in dependencies_by_test_file(scope, repo_root, known).items():
                if test_file not in selected and dependencies & affected:
                    selected.append(test_file)

        if selected:
            matrix.append(matrix_leg(scope, sorted(selected)))

    return matrix


def full_matrix() -> list[dict[str, str]]:
    """Every scope, each running its full suite directory."""
    return [matrix_leg(scope, []) for scope in SCOPES]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Select tests to run from a diff against a base ref.")
    parser.add_argument("--base-ref", metavar="SHA", help="Git SHA or ref to diff HEAD against")
    parser.add_argument(
        "--run-all-tests",
        action="store_true",
        help="Run every package's full suite regardless of the diff",
    )
    args = parser.parse_args()
    if not (args.base_ref or args.run_all_tests):
        parser.error("pass --base-ref, --run-all-tests, or both")

    repo_root = Path(__file__).parent.parent.parent

    # Without a base ref there is nothing to gate the out-of-band suites on, so run them all.
    if args.base_ref is None:
        result = {"reason": "run-all-tests", "matrix": full_matrix(), "suites": sorted(EXTRA_SUITE_TRIGGERS)}
        print(json.dumps(result, indent=2))
        return

    changed = git_changed_files(args.base_ref, repo_root)
    classification = classify(changed, repo_root)
    suites = extra_suites(changed)

    if args.run_all_tests:
        reason, matrix = "run-all-tests", full_matrix()
    elif classification.broad:
        reason, matrix = "broad-trigger", full_matrix()
    else:
        reason = "diff-driven"
        matrix = compute_matrix(
            classification.src_modules,
            classification.direct_tests,
            classification.forced,
            repo_root,
        )

    print(json.dumps({"reason": reason, "matrix": matrix, "suites": suites}, indent=2))


if __name__ == "__main__":
    main()
