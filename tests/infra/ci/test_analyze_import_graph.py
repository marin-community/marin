# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the import-knot analyzer (infra/ci/analyze_import_graph.py)."""

import textwrap
from pathlib import Path

from infra.ci.analyze_import_graph import (
    all_dependencies_by_test_file,
    analyze,
    blast_radius,
    dependent_tests_by_module,
    file_report,
    reexport_hubs,
)
from infra.ci.select_tests import build_importers, workspace_modules


def write(repo_root: Path, relative: str, body: str = "") -> Path:
    path = repo_root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(body))
    return path


def _hub_workspace(repo_root: Path) -> None:
    """A package whose __init__ re-exports two submodules, with a test per submodule.

    ``import levanter.a`` runs ``levanter/__init__.py``, which imports ``levanter.b`` too,
    so touching ``b`` currently selects the test that only imports ``a``.
    """
    write(repo_root, "lib/levanter/src/levanter/__init__.py", "from levanter.a import A\nfrom levanter.b import B\n")
    write(repo_root, "lib/levanter/src/levanter/a.py", "A = 1\n")
    write(repo_root, "lib/levanter/src/levanter/b.py", "B = 2\n")
    write(repo_root, "lib/levanter/tests/test_a.py", "from levanter.a import A\n")
    write(repo_root, "lib/levanter/tests/test_b.py", "from levanter.b import B\n")


def test_reexport_hubs_detects_init_that_imports_siblings(tmp_path: Path) -> None:
    _hub_workspace(tmp_path)

    hubs = reexport_hubs(workspace_modules(tmp_path))

    assert hubs == {"levanter": {"levanter.a", "levanter.b"}}


def test_docstring_only_init_is_not_a_hub(tmp_path: Path) -> None:
    write(tmp_path, "lib/iris/src/iris/__init__.py", '"""iris."""\n')
    write(tmp_path, "lib/iris/src/iris/core.py", "X = 1\n")

    assert reexport_hubs(workspace_modules(tmp_path)) == {}


def test_hub_ties_a_submodule_change_to_every_sibling_test(tmp_path: Path) -> None:
    _hub_workspace(tmp_path)
    modules = workspace_modules(tmp_path)
    direct = dependent_tests_by_module(all_dependencies_by_test_file(tmp_path, set(modules)))

    # Through the hub, a change to b selects test_a as well as test_b.
    full = blast_radius("levanter.b", build_importers(modules), direct)
    assert full == {"lib/levanter/tests/test_a.py", "lib/levanter/tests/test_b.py"}

    # Emptying the hub isolates them: a change to b selects only test_b.
    emptied = blast_radius("levanter.b", build_importers(modules, emptied=frozenset({"levanter"})), direct)
    assert emptied == {"lib/levanter/tests/test_b.py"}


def test_analyze_reports_the_reduction_from_emptying_hubs(tmp_path: Path) -> None:
    _hub_workspace(tmp_path)

    report = analyze(tmp_path)

    assert report["reduction_from_sweep"] > 0
    assert report["swept_total_selected"] < report["baseline_total_selected"]
    levanter_hub = next(entry for entry in report["hubs"] if entry["hub"] == "levanter")
    assert levanter_hub["reexported_submodules"] == 2
    assert levanter_hub["reduction"] > 0


def test_file_report_gives_blast_radius_by_scope(tmp_path: Path) -> None:
    _hub_workspace(tmp_path)

    [report] = file_report(tmp_path, ["lib/levanter/src/levanter/b.py"])

    assert report["module"] == "levanter.b"
    assert report["selected_test_files"] == 2
    assert report["by_scope"] == {"levanter": 2}
