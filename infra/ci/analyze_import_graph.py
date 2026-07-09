# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Find import knots: modules whose change over-selects the test suite.

The CI selector (``infra/ci/select_tests.py``) walks a module-level import graph
backwards from a diff to the tests that transitively import it. Some source modules
sit on so many test files' dependency paths that touching them selects nearly the
whole suite. This script surfaces those knots so they can be untied.

Two shapes of knot show up:

- **Re-export hubs.** A package ``__init__`` that imports its own submodules ties
  every importer of *any* submodule to *all* of them, because ``import pkg.sub`` runs
  ``pkg/__init__.py``. A one-line change to one submodule then selects every test that
  touches the package. ``--hubs`` lists these and the reduction from emptying them.
- **Load-bearing modules.** A widely imported ``types.py``/``config.py`` genuinely sits
  under most tests. No cleanup narrows that; it is reported so it is not mistaken for a
  hub.

Metric: a module's *blast radius* is the number of test files the selector would run if
that module were the only changed file. Summing blast radius over every source module
gives a single ``total selected test files`` figure comparable before and after a change
to the graph.

Usage:
    python infra/ci/analyze_import_graph.py                 # top knots + hub summary
    python infra/ci/analyze_import_graph.py --hubs          # simulate emptying each hub
    python infra/ci/analyze_import_graph.py --file lib/levanter/src/levanter/utils/mesh.py
    python infra/ci/analyze_import_graph.py --json          # machine-readable
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from infra.ci.select_tests import (
    SCOPES,
    SOURCE_ROOTS,
    TEST_DIR,
    affected_modules,
    build_importers,
    dependencies_by_test_file,
    imported_names,
    path_to_module,
    resolve,
    workspace_modules,
)


def all_dependencies_by_test_file(repo_root: Path, known: set[str]) -> dict[str, set[str]]:
    """{repo-relative test file: workspace modules it transitively imports}, all scopes."""
    deps: dict[str, set[str]] = {}
    for scope in SCOPES:
        deps.update(dependencies_by_test_file(scope, repo_root, known))
    return deps


def dependent_tests_by_module(test_deps: dict[str, set[str]]) -> dict[str, set[str]]:
    """Invert test_deps: {module: test files that directly depend on it}."""
    index: dict[str, set[str]] = defaultdict(set)
    for test_file, modules in test_deps.items():
        for module in modules:
            index[module].add(test_file)
    return index


def blast_radius(module: str, importers: dict[str, set[str]], direct: dict[str, set[str]]) -> set[str]:
    """Test files the selector runs when ``module`` is the only changed file."""
    selected: set[str] = set()
    for affected in affected_modules({module}, importers):
        selected |= direct.get(affected, set())
    return selected


def total_selected(source_modules: set[str], importers: dict[str, set[str]], direct: dict[str, set[str]]) -> int:
    """Sum of blast radius over every source module: one comparable size for the whole graph."""
    return sum(len(blast_radius(module, importers, direct)) for module in source_modules)


def reexport_hubs(modules: dict[str, Path]) -> dict[str, set[str]]:
    """{package init module: the sibling submodules it re-exports}.

    A hub is an ``__init__`` that imports modules under its own package, e.g. ``levanter``
    importing ``levanter.trainer``. ``path_to_module`` maps an ``__init__.py`` to its package
    name, so these keys are the bare package names.
    """
    known = set(modules)
    hubs: dict[str, set[str]] = {}
    for module, py in modules.items():
        if py.name != "__init__.py":
            continue
        dependencies = resolve(imported_names(py, module), known)
        siblings = {dep for dep in dependencies if dep != module and dep.startswith(f"{module}.")}
        if siblings:
            hubs[module] = siblings
    return hubs


def scope_of(test_file: str) -> str:
    """Which workspace scope a repo-relative test file belongs to."""
    return next((scope for scope in SCOPES if test_file.startswith(f"{TEST_DIR[scope]}/")), "?")


def rank_by_blast_radius(
    source_modules: set[str],
    importers: dict[str, set[str]],
    direct: dict[str, set[str]],
) -> list[tuple[str, int]]:
    """Source modules ordered by how many test files their change selects, largest first."""
    scored = [(module, len(blast_radius(module, importers, direct))) for module in source_modules]
    scored.sort(key=lambda item: (-item[1], item[0]))
    return scored


def analyze(repo_root: Path) -> dict:
    """Build the graph and compute knots, hubs, and the emptied-hub simulation."""
    modules = workspace_modules(repo_root)
    known = set(modules)
    test_deps = all_dependencies_by_test_file(repo_root, known)
    direct = dependent_tests_by_module(test_deps)

    importers = build_importers(modules)
    ranked = rank_by_blast_radius(known, importers, direct)

    hubs = reexport_hubs(modules)
    baseline = total_selected(known, importers, direct)

    # Emptying every hub at once measures the full available reduction; per-hub numbers
    # below attribute it, and do not sum to the total because hubs share downstream tests.
    all_emptied = build_importers(modules, emptied=frozenset(hubs))
    swept_total = total_selected(known, importers=all_emptied, direct=direct)

    hub_report = []
    for hub, siblings in sorted(hubs.items()):
        emptied = build_importers(modules, emptied=frozenset({hub}))
        after = total_selected(known, emptied, direct)
        hub_report.append(
            {
                "hub": hub,
                "reexported_submodules": len(siblings),
                "blast_radius": len(blast_radius(hub, importers, direct)),
                "total_after_emptying": after,
                "reduction": baseline - after,
            }
        )
    hub_report.sort(key=lambda item: -item["reduction"])

    return {
        "total_test_files": len(test_deps),
        "baseline_total_selected": baseline,
        "swept_total_selected": swept_total,
        "reduction_from_sweep": baseline - swept_total,
        "hubs": hub_report,
        "top_modules": [{"module": module, "blast_radius": radius} for module, radius in ranked[:25]],
    }


def file_report(repo_root: Path, files: list[str]) -> list[dict]:
    """Blast radius of specific changed files, with a per-scope breakdown of selected tests."""
    modules = workspace_modules(repo_root)
    known = set(modules)
    test_deps = all_dependencies_by_test_file(repo_root, known)
    direct = dependent_tests_by_module(test_deps)
    importers = build_importers(modules)

    reports = []
    for filepath in files:
        source_root = next(
            (root for root in _import_roots(repo_root) if (repo_root / filepath).is_relative_to(root)),
            None,
        )
        module = path_to_module(repo_root / filepath, source_root) if source_root else None
        if module is None or module not in known:
            reports.append({"file": filepath, "error": "not a workspace source module"})
            continue
        selected = blast_radius(module, importers, direct)
        by_scope: dict[str, int] = defaultdict(int)
        for test_file in selected:
            by_scope[scope_of(test_file)] += 1
        reports.append(
            {
                "file": filepath,
                "module": module,
                "selected_test_files": len(selected),
                "by_scope": dict(sorted(by_scope.items())),
            }
        )
    return reports


def _import_roots(repo_root: Path) -> list[Path]:
    return [repo_root / root.import_root for root in SOURCE_ROOTS]


def _print_report(report: dict) -> None:
    baseline = report["baseline_total_selected"]
    swept = report["swept_total_selected"]
    reduction = report["reduction_from_sweep"]
    pct = (100 * reduction / baseline) if baseline else 0.0
    print(f"test files in graph: {report['total_test_files']}")
    print(f"total selected (single-file-change, summed over every source module): {baseline}")
    print(f"  after emptying every re-export hub: {swept}  (-{reduction}, -{pct:.1f}%)\n")

    print("re-export hubs, by reduction from emptying:")
    print(f"  {'hub':<28} {'re-exports':>10} {'blast':>7} {'reduction':>10}")
    for entry in report["hubs"]:
        print(
            f"  {entry['hub']:<28} {entry['reexported_submodules']:>10} "
            f"{entry['blast_radius']:>7} {entry['reduction']:>10}"
        )

    print("\ntop modules by blast radius (test files selected when this file alone changes):")
    for entry in report["top_modules"]:
        print(f"  {entry['blast_radius']:>5}  {entry['module']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--file", action="append", default=[], help="Report blast radius for a specific changed file")
    parser.add_argument("--hubs", action="store_true", help="Only the re-export hub simulation")
    parser.add_argument("--json", action="store_true", help="Emit JSON")
    args = parser.parse_args()

    repo_root = Path(__file__).parent.parent.parent

    if args.file:
        reports = file_report(repo_root, args.file)
        if args.json:
            print(json.dumps(reports, indent=2))
        else:
            for entry in reports:
                if "error" in entry:
                    print(f"{entry['file']}: {entry['error']}")
                else:
                    breakdown = ", ".join(f"{scope}={count}" for scope, count in entry["by_scope"].items())
                    print(
                        f"{entry['file']} ({entry['module']}): {entry['selected_test_files']} test files  [{breakdown}]"
                    )
        return

    report = analyze(repo_root)
    if args.hubs:
        report = {
            key: report[key]
            for key in ("baseline_total_selected", "swept_total_selected", "reduction_from_sweep", "hubs")
        }
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        _print_report(report if not args.hubs else {**report, "total_test_files": 0, "top_modules": []})


if __name__ == "__main__":
    main()
