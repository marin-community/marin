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
    python infra/ci/select_tests.py --base-ref <SHA>  # pull request or push
    python infra/ci/select_tests.py --run-all-tests   # scheduled or manual run
"""

import argparse
import ast
import json
import subprocess
from collections import defaultdict
from dataclasses import asdict, dataclass
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
    "dupekit",
    "finelog",
    "finestore",
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
    SourceRoot("infra/ci", "."),
    SourceRoot("infra/evaldash/src", "."),
)

# Dependency and native-build changes can affect every local test environment.
LOCAL_BROAD_TRIGGERS: frozenset[str] = frozenset({"uv.lock", "pyproject.toml", "scripts/rust_mode.py"})

# Selector and workflow changes run the complete CI matrix to validate the
# orchestration itself. Locally, their import-dependent tests are sufficient;
# the exhaustive matrix still runs after the branch is pushed.
CI_BROAD_TRIGGERS: frozenset[str] = frozenset({"infra/ci/select_tests.py", ".github/workflows/unified-unit.yaml"})

BROAD_TRIGGERS = LOCAL_BROAD_TRIGGERS | CI_BROAD_TRIGGERS

# uv package names and pytest paths for each workspace scope.
UV_PACKAGE: dict[str, str] = {
    "rigging": "marin-rigging",
    "haliax": "marin-haliax",
    "iris": "marin-iris",
    "fray": "marin-fray",
    "levanter": "marin-levanter",
    "zephyr": "marin-zephyr",
    "marin": "marin-core",
    "dupekit": "marin-dupekit",
    "finelog": "marin-finelog",
    "finestore": "marin-finestore",
}

UV_EXTRAS: dict[str, list[str]] = {
    "marin": ["cpu", "dedup"],
}

PYTHON_VERSION = "3.12"
PYTEST_ARGS: tuple[str, ...] = (
    "--durations=5",
    "-n",
    "auto",
    "--dist=worksteal",
    "--tb=short",
)

RUN_ALL_REASON = "run-all-tests"
BROAD_TRIGGER_REASON = "broad-trigger"
DIFF_DRIVEN_REASON = "diff-driven"

TEST_DIR: dict[str, str] = {
    **{scope: f"lib/{scope}/tests" for scope in UV_PACKAGE if scope != "marin"},
    "marin": "tests",
}

# Levanter's suite is the only unit leg that runs long enough to be worth spreading over
# extra runners; every other scope finishes well under the workflow's per-leg budget. A
# sharded scope splits its selected files across this many parallel matrix legs.
SHARD_COUNT: dict[str, int] = {"levanter": 4}

# A shard carries fixed environment-setup overhead, so stop adding runners once each would
# hold fewer than this many files: a small selection runs faster in one leg than spread thin.
MIN_FILES_PER_SHARD = 15

# Native (maturin) packages, keyed by their owning scope. A change under a crate's
# rust/ tree is invisible to the Python import graph, so classify force-selects the
# owning scope and builds it from source. Its own tests cover the extension;
# downstream consumers are selected only by their Python-level changes and run
# against the prebuilt wheel.
NATIVE_CRATE_DIR: dict[str, str] = {
    "dupekit": "lib/dupekit/rust",
    "finelog": "lib/finelog/rust",
    "iris": "lib/iris/rust",
}

# The matrix `setup` tag that unified-unit.yaml maps to the Rust source-build
# steps (toolchain + cargo cache + scripts/rust_mode.py dev).
RUST_SETUP_TAG = "rust"
# A native source build (finelog links the datafusion/arrow tree) exceeds the
# default per-leg budget; source-build legs carry this timeout instead.
SOURCE_BUILD_TIMEOUT = 30
DEFAULT_LEG_TIMEOUT = 15

# Suites that cannot be import-selected because they drive a non-Python subsystem.
# Levanter's accelerator lanes use the ordinary import-selected Levanter files below;
# only the browser smoke remains a directory-triggered suite.
DEPENDENCY_MANIFESTS: tuple[str, ...] = ("uv.lock", "pyproject.toml")
EXTRA_SUITE_TRIGGERS: dict[str, tuple[str, ...]] = {
    "iris-e2e-smoke": ("lib/iris/", *DEPENDENCY_MANIFESTS),
}

LEVANTER_ACCELERATOR_TRIGGERS: tuple[str, ...] = (
    "lib/levanter/",
    "lib/haliax/",
    "infra/ci/select_tests.py",
    ".github/workflows/unified-unit.yaml",
    *DEPENDENCY_MANIFESTS,
)

# These files are intentionally absent from the TPU command today. Keep the selection
# rule next to the selector so an affected-file TPU run does not start only to collect
# zero runnable tests.
TPU_IGNORED_TEST_PATHS: frozenset[str] = frozenset(
    {
        "lib/levanter/tests/test_audio.py",
        "lib/levanter/tests/test_new_cache.py",
        "lib/levanter/tests/test_hf_checkpoints.py",
        "lib/levanter/tests/test_hf_gpt2_serialize.py",
        "lib/levanter/tests/test_gdn_layer.py",
    }
)


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


def build_importers(modules: dict[str, Path], emptied: frozenset[str] = frozenset()) -> dict[str, set[str]]:
    """importers[M] = modules whose top-level imports execute M.

    ``emptied`` names modules whose own imports are ignored, as if their body were
    reduced to a docstring. The import-graph analyzer uses it to simulate deleting a
    re-export hub; production selection passes the default empty set.
    """
    known = set(modules)
    importers: dict[str, set[str]] = defaultdict(set)
    for module, py in modules.items():
        if module in emptied:
            continue
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


def has_static_test_items(path: Path) -> bool:
    tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"), filename=str(path))
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test_"):
            return True
        if isinstance(node, ast.ClassDef) and node.name.startswith("Test"):
            if any(
                isinstance(method, (ast.FunctionDef, ast.AsyncFunctionDef)) and method.name.startswith("test_")
                for method in node.body
            ):
                return True
    return False


def _test_tree(scope: str, repo_root: Path) -> dict[str, Path]:
    """Every .py under a scope's test directory, keyed by its repo-root module name.

    Package test trees need distinct internal names so dependency analysis does
    not conflate same-named helpers in different packages. Relative imports
    resolve against the same canonical name.
    """
    test_dir = repo_root / TEST_DIR[scope]
    if not test_dir.exists():
        return {}
    tree: dict[str, Path] = {}
    for py in test_dir.rglob("*.py"):
        module = path_to_module(py, repo_root)
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
    # --no-renames: a file moved out of a native crate's rust/ tree must surface as a
    # delete of its old path, or its scope would miss the source-build trigger; with
    # rename detection git reports only the destination.
    result = subprocess.run(
        ["git", "diff", "--name-only", "--no-renames", f"{base_ref}...HEAD"],
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
    native_changed: set[str]
    """Scopes whose native crate (lib/<scope>/rust) changed — need a source build."""


def classify(
    changed_files: list[str],
    repo_root: Path,
    broad_triggers: frozenset[str] = BROAD_TRIGGERS,
) -> ClassifyResult:
    """Classify repo-root-relative changed file paths."""
    broad = False
    src_modules: set[str] = set()
    direct_tests: dict[str, list[str]] = defaultdict(list)
    forced: set[str] = set()
    native_changed: set[str] = set()

    for filepath in changed_files:
        if filepath in broad_triggers:
            broad = True
            continue

        # A native crate's rust/ tree is not on any import root, so this branch
        # runs before source-root handling. The change is invisible to the Python
        # import graph, so force-select the owning scope and mark it for a source
        # build; its own tests exercise the extension.
        native_scope = next(
            (scope for scope, crate_dir in NATIVE_CRATE_DIR.items() if filepath.startswith(f"{crate_dir}/")),
            None,
        )
        if native_scope is not None:
            forced.add(native_scope)
            native_changed.add(native_scope)
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
                elif (repo_root / filepath).exists() and has_static_test_items(repo_root / filepath):
                    direct_tests[scope].append(filepath)
                elif (repo_root / filepath).exists():
                    # Helpers named test_*.py satisfy pytest's file convention but do not
                    # own test items. Their importers are not recoverable from a direct
                    # file selection, so run the scope that consumes the helper.
                    forced.add(scope)
                break

            if filepath in (f"lib/{scope}/conftest.py", f"lib/{scope}/pyproject.toml"):
                forced.add(scope)
                break

    return ClassifyResult(
        broad=broad,
        src_modules=src_modules,
        direct_tests=dict(direct_tests),
        forced=forced,
        native_changed=native_changed,
    )


def extra_suites(changed_files: list[str]) -> list[str]:
    """Out-of-band suites to run for this diff."""
    return sorted(
        suite
        for suite, prefixes in EXTRA_SUITE_TRIGGERS.items()
        if any(filepath.startswith(prefix) for prefix in prefixes for filepath in changed_files)
    )


def _node_has_torch_marker(node: ast.AST) -> bool:
    return any(
        (isinstance(child, ast.Name) and child.id == "skip_if_no_torch")
        or (isinstance(child, ast.Attribute) and child.attr == "torch")
        for child in ast.walk(node)
    )


def torch_membership_for_test_file(path: Path) -> tuple[bool, bool]:
    """Return whether a test file contains torch and non-torch tests.

    The Levanter helper ``skip_if_no_torch`` applies ``pytest.mark.torch``. The
    selector cannot import test modules because its job intentionally installs no test
    dependencies, so inspect decorators and module-level ``pytestmark`` assignments.
    Unknown/dynamically generated test shapes conservatively enter both lanes.
    """
    tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"), filename=str(path))
    module_is_torch = any(
        isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "pytestmark" for target in node.targets)
        and _node_has_torch_marker(node.value)
        for node in tree.body
    )

    has_torch = module_is_torch
    has_non_torch = False
    found_test = False
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test_"):
            found_test = True
            is_torch = module_is_torch or any(_node_has_torch_marker(decorator) for decorator in node.decorator_list)
            has_torch |= is_torch
            has_non_torch |= not is_torch
            continue

        if isinstance(node, ast.ClassDef) and node.name.startswith("Test"):
            class_is_torch = module_is_torch or any(
                _node_has_torch_marker(decorator) for decorator in node.decorator_list
            )
            for method in node.body:
                if isinstance(method, (ast.FunctionDef, ast.AsyncFunctionDef)) and method.name.startswith("test_"):
                    found_test = True
                    is_torch = class_is_torch or any(
                        _node_has_torch_marker(decorator) for decorator in method.decorator_list
                    )
                    has_torch |= is_torch
                    has_non_torch |= not is_torch

    if not found_test:
        return True, True
    return has_torch, has_non_torch


# ---------------------------------------------------------------------------
# Test selection
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MatrixLeg:
    """Configuration for one unified-unit pytest job."""

    label: str
    package: str
    python: str
    extras: str
    pytest_args: str
    test_paths: str
    setup: str
    timeout: int


def matrix_leg(
    scope: str,
    tests: list[str],
    shard: tuple[int, int] | None = None,
    *,
    source_build: bool = False,
) -> MatrixLeg:
    """Build one unified-unit matrix leg with uv/pytest arguments.

    ``shard`` is a ``(index, total)`` pair when the scope's suite is split across several
    runners; it rides in the label so each shard surfaces as its own workflow job.
    ``source_build`` flags a leg whose native extension must be built from source (the
    workflow reads the ``setup`` tag).
    """
    label = scope if shard is None else f"{scope} {shard[0]}/{shard[1]}"
    return MatrixLeg(
        label=label,
        package=UV_PACKAGE[scope],
        python=PYTHON_VERSION,
        extras=" ".join(f"--extra {extra}" for extra in UV_EXTRAS.get(scope, [])),
        pytest_args=" ".join(PYTEST_ARGS),
        test_paths=" ".join(tests) if tests else TEST_DIR[scope],
        setup=RUST_SETUP_TAG if source_build else "",
        timeout=SOURCE_BUILD_TIMEOUT if source_build else DEFAULT_LEG_TIMEOUT,
    )


def all_test_files(scope: str, repo_root: Path) -> list[str]:
    """Every collectable test file in a scope's suite, repo-relative and sorted."""
    return sorted(
        str(py.relative_to(repo_root)) for py in _test_tree(scope, repo_root).values() if is_test_module(py.name)
    )


def shard_files(tests: list[str], count: int) -> list[list[str]]:
    """Split ``tests`` into ``count`` contiguous, size-balanced chunks."""
    base, extra = divmod(len(tests), count)
    chunks: list[list[str]] = []
    start = 0
    for index in range(count):
        size = base + (1 if index < extra else 0)
        chunks.append(tests[start : start + size])
        start += size
    return chunks


def scope_legs(
    scope: str,
    tests: list[str] | None,
    repo_root: Path,
    *,
    source_build: bool = False,
) -> list[MatrixLeg]:
    """Matrix legs for one scope: one leg, or several when the scope is sharded.

    ``tests is None`` runs the full suite. A sharded scope expands that to its file list so
    full and diff-driven runs spread across the same runners; below MIN_FILES_PER_SHARD it
    stays a single leg. ``source_build`` flags the legs whose native extension must be built
    from source.
    """
    cap = SHARD_COUNT.get(scope, 1)
    files = tests if tests is not None else (all_test_files(scope, repo_root) if cap > 1 else None)
    if cap <= 1 or files is None or len(files) <= MIN_FILES_PER_SHARD:
        return [matrix_leg(scope, files or [], source_build=source_build)]

    # Floor, not ceil: pick the largest shard count that still leaves every shard at least
    # MIN_FILES_PER_SHARD files, so a medium selection is not split into runners so small that
    # setup overhead dominates (16 files stays one leg, not two 8-file legs).
    count = min(cap, len(files) // MIN_FILES_PER_SHARD)
    if count <= 1:
        return [matrix_leg(scope, sorted(files), source_build=source_build)]
    chunks = shard_files(sorted(files), count)
    return [
        matrix_leg(scope, chunk, shard=(index + 1, count), source_build=source_build)
        for index, chunk in enumerate(chunks)
    ]


def compute_matrix(
    src_modules: set[str],
    direct_tests: dict[str, list[str]],
    forced_scopes: set[str],
    source_build_scopes: set[str],
    repo_root: Path,
) -> list[MatrixLeg]:
    """Compute the test matrix.

    Returns a list of matrix legs. Each leg has a label, package (uv name), Python
    version, uv extras, pytest arguments, test paths, and a source-build
    ``setup``/``timeout``. An empty tests list means run the full suite directory;
    a scope may fan out into several sharded legs.
    ``source_build_scopes`` are the scopes whose legs must build the native extension from
    source.
    """
    if not (src_modules or direct_tests or forced_scopes):
        return []

    modules = workspace_modules(repo_root)
    known = set(modules)
    affected = affected_modules(src_modules, build_importers(modules)) if src_modules else set()

    matrix: list[MatrixLeg] = []
    for scope in SCOPES:
        source_build = scope in source_build_scopes
        if scope in forced_scopes:
            matrix.extend(scope_legs(scope, None, repo_root, source_build=source_build))
            continue

        selected = list(direct_tests.get(scope, []))
        if affected:
            for test_file, dependencies in dependencies_by_test_file(scope, repo_root, known).items():
                if test_file not in selected and dependencies & affected:
                    selected.append(test_file)

        if selected:
            matrix.extend(scope_legs(scope, sorted(selected), repo_root, source_build=source_build))

    return matrix


def full_matrix(repo_root: Path, source_build_scopes: set[str]) -> list[MatrixLeg]:
    """Every scope, each running its full suite (sharded where configured)."""
    legs: list[MatrixLeg] = []
    for scope in SCOPES:
        legs.extend(scope_legs(scope, None, repo_root, source_build=scope in source_build_scopes))
    return legs


def selected_scope_test_paths(matrix: list[MatrixLeg], scope: str) -> list[str]:
    """Return the unique pytest paths selected for one scope across all matrix shards."""
    package = UV_PACKAGE[scope]
    return sorted({path for leg in matrix if leg.package == package for path in leg.test_paths.split()})


def accelerator_suite_test_paths(
    changed_files: list[str],
    matrix: list[MatrixLeg],
    repo_root: Path,
    *,
    force: bool = False,
) -> dict[str, list[str]]:
    """Return affected Levanter tests split by accelerator lane."""
    is_triggered = force or any(
        filepath.startswith(prefix) for prefix in LEVANTER_ACCELERATOR_TRIGGERS for filepath in changed_files
    )
    if not is_triggered:
        return {}

    selected = selected_scope_test_paths(matrix, "levanter")
    if not selected:
        return {}

    torch_paths: list[str] = []
    tpu_paths: list[str] = []
    for test_path in selected:
        if test_path == TEST_DIR["levanter"]:
            torch_paths.append(test_path)
            tpu_paths.append(test_path)
            continue

        path = repo_root / test_path
        has_torch, has_non_torch = torch_membership_for_test_file(path)
        if has_torch:
            torch_paths.append(test_path)
        if has_non_torch and test_path not in TPU_IGNORED_TEST_PATHS:
            tpu_paths.append(test_path)

    suites: dict[str, list[str]] = {}
    if torch_paths:
        suites["levanter-torch"] = torch_paths
    if tpu_paths:
        suites["levanter-tpu"] = tpu_paths
    return suites


@dataclass(frozen=True)
class SelectionResult:
    """Selected CI matrix legs and out-of-band suites for a set of changed files."""

    reason: str
    matrix: list[MatrixLeg]
    suites: list[str]
    suite_test_paths: dict[str, list[str]]


def _select_changed_tests(
    changed_files: list[str],
    repo_root: Path,
    broad_triggers: frozenset[str],
    *,
    run_all_tests: bool = False,
) -> SelectionResult:
    classification = classify(changed_files, repo_root, broad_triggers)
    source_build_scopes = set(classification.native_changed)

    if run_all_tests:
        reason, matrix = RUN_ALL_REASON, full_matrix(repo_root, source_build_scopes)
    elif classification.broad:
        reason, matrix = BROAD_TRIGGER_REASON, full_matrix(repo_root, source_build_scopes)
    else:
        reason = DIFF_DRIVEN_REASON
        matrix = compute_matrix(
            classification.src_modules,
            classification.direct_tests,
            classification.forced,
            source_build_scopes,
            repo_root,
        )

    suite_test_paths = accelerator_suite_test_paths(changed_files, matrix, repo_root)
    suites = sorted((*extra_suites(changed_files), *suite_test_paths))
    return SelectionResult(
        reason=reason,
        matrix=matrix,
        suites=suites,
        suite_test_paths=suite_test_paths,
    )


def select_changed_tests(
    changed_files: list[str],
    repo_root: Path,
    *,
    run_all_tests: bool = False,
) -> SelectionResult:
    """Return the CI test plan for repo-relative changed paths."""
    return _select_changed_tests(
        changed_files,
        repo_root,
        BROAD_TRIGGERS,
        run_all_tests=run_all_tests,
    )


def select_local_tests(
    changed_files: list[str],
    repo_root: Path,
    *,
    run_all_tests: bool = False,
) -> SelectionResult:
    """Return affected local tests without expanding CI-only orchestration changes."""
    return _select_changed_tests(
        changed_files,
        repo_root,
        LOCAL_BROAD_TRIGGERS,
        run_all_tests=run_all_tests,
    )


def selection_payload(selection: SelectionResult) -> dict[str, object]:
    """Return a JSON-serializable GitHub Actions matrix payload."""
    return {
        "reason": selection.reason,
        "matrix": [asdict(leg) for leg in selection.matrix],
        "suites": selection.suites,
        "suite_test_paths": selection.suite_test_paths,
    }


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

    # Without a base ref there is no diff to inspect, so conservatively build every native
    # extension from source and run the out-of-band suites too.
    if args.base_ref is None:
        matrix = full_matrix(repo_root, set(NATIVE_CRATE_DIR))
        suite_test_paths = accelerator_suite_test_paths([], matrix, repo_root, force=True)
        selection = SelectionResult(
            reason=RUN_ALL_REASON,
            matrix=matrix,
            suites=sorted((*EXTRA_SUITE_TRIGGERS, *suite_test_paths)),
            suite_test_paths=suite_test_paths,
        )
        print(json.dumps(selection_payload(selection), indent=2))
        return

    changed = git_changed_files(args.base_ref, repo_root)
    selection = select_changed_tests(changed, repo_root, run_all_tests=args.run_all_tests)
    print(json.dumps(selection_payload(selection), indent=2))


if __name__ == "__main__":
    main()
