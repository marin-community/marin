# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the import-driven test selector (infra/ci/select_tests.py)."""

import textwrap
from pathlib import Path

import pytest

from infra.ci.select_tests import (
    SCOPES,
    UV_PACKAGE,
    MatrixLeg,
    classify,
    matrix_leg,
    select_changed_tests,
    select_local_tests,
)


def select_matrix(changed_files: list[str], repo_root: Path) -> list[MatrixLeg]:
    """Return the selector's diff-driven matrix without invoking git."""
    return select_changed_tests(changed_files, repo_root).matrix


def write(repo_root: Path, relative: str, body: str = "") -> Path:
    path = repo_root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(body))
    return path


def leg_paths(matrix: list[MatrixLeg], scope: str) -> list[str]:
    leg = next(entry for entry in matrix if entry.package == UV_PACKAGE[scope])
    return leg.test_paths.split()


def scopes_in(matrix: list[MatrixLeg]) -> set[str]:
    packages = {entry.package for entry in matrix}
    return {scope for scope in SCOPES if UV_PACKAGE[scope] in packages}


def _workspace(repo_root: Path) -> None:
    """A workspace exercising each edge the selector has to walk."""
    write(repo_root, "lib/rigging/src/rigging/__init__.py")
    write(repo_root, "lib/rigging/src/rigging/timing.py", "TIMEOUT = 1\n")
    write(repo_root, "lib/rigging/src/rigging/other.py", "OTHER = 2\n")
    write(repo_root, "lib/rigging/tests/test_timing.py", "from rigging import timing\n")
    write(repo_root, "lib/rigging/tests/test_other.py", "import rigging.other\n")

    # iris.controller depends on rigging.timing, so a rigging change reaches iris tests.
    write(repo_root, "lib/iris/src/iris/__init__.py")
    write(repo_root, "lib/iris/src/iris/controller.py", "import rigging.timing\n")
    write(repo_root, "lib/iris/tests/test_controller.py", "from iris import controller\n")

    # zephyr.writers imports rigging lazily; a rigging change must not select it.
    write(repo_root, "lib/zephyr/src/zephyr/__init__.py")
    write(
        repo_root,
        "lib/zephyr/src/zephyr/writers.py",
        """\
        def write():
            from rigging import timing  # lazy import
        """,
    )
    write(repo_root, "lib/zephyr/tests/test_writers.py", "from zephyr import writers\n")


def test_top_level_import_reaches_transitive_dependents(tmp_path: Path) -> None:
    _workspace(tmp_path)

    matrix = select_matrix(["lib/rigging/src/rigging/timing.py"], tmp_path)

    assert leg_paths(matrix, "rigging") == ["lib/rigging/tests/test_timing.py"]
    assert leg_paths(matrix, "iris") == ["lib/iris/tests/test_controller.py"]
    assert "zephyr" not in scopes_in(matrix), "a lazy import must not propagate"


def test_selection_is_empty_when_nothing_depends_on_the_change(tmp_path: Path) -> None:
    _workspace(tmp_path)
    write(tmp_path, "lib/rigging/src/rigging/unused.py", "X = 1\n")

    assert select_matrix(["lib/rigging/src/rigging/unused.py"], tmp_path) == []
    assert select_matrix([], tmp_path) == []


def test_package_init_reexport_ties_importers_to_every_submodule(tmp_path: Path) -> None:
    """`import haliax` runs haliax/__init__.py, so a re-exported submodule reaches its importers."""
    write(tmp_path, "lib/haliax/src/haliax/__init__.py", "from haliax.core import dot\n")
    write(tmp_path, "lib/haliax/src/haliax/core.py", "def dot():\n    pass\n")
    write(tmp_path, "lib/haliax/tests/test_axis.py", "import haliax\n")

    matrix = select_matrix(["lib/haliax/src/haliax/core.py"], tmp_path)

    assert leg_paths(matrix, "haliax") == ["lib/haliax/tests/test_axis.py"]


def test_submodule_import_does_not_select_unrelated_siblings(tmp_path: Path) -> None:
    """With a docstring-only __init__, sibling modules stay independent."""
    write(tmp_path, "lib/iris/src/iris/__init__.py", '"""iris."""\n')
    write(tmp_path, "lib/iris/src/iris/scheduler.py", "SCHED = 1\n")
    write(tmp_path, "lib/iris/src/iris/worker.py", "WORKER = 2\n")
    write(tmp_path, "lib/iris/tests/test_scheduler.py", "from iris.scheduler import SCHED\n")
    write(tmp_path, "lib/iris/tests/test_worker.py", "from iris.worker import WORKER\n")

    matrix = select_matrix(["lib/iris/src/iris/scheduler.py"], tmp_path)

    assert leg_paths(matrix, "iris") == ["lib/iris/tests/test_scheduler.py"]


def test_experiments_changes_select_dependent_marin_tests(tmp_path: Path) -> None:
    write(tmp_path, "experiments/__init__.py")
    write(tmp_path, "experiments/tokenizer_sweep.py", "def sweep():\n    pass\n")
    write(tmp_path, "tests/test_tokenizer_sweep.py", "from experiments.tokenizer_sweep import sweep\n")
    write(tmp_path, "tests/test_unrelated.py", "def test_x():\n    pass\n")

    matrix = select_matrix(["experiments/tokenizer_sweep.py"], tmp_path)

    assert leg_paths(matrix, "marin") == ["tests/test_tokenizer_sweep.py"]


@pytest.mark.parametrize(
    "changed_file",
    ["lib/iris/src/iris/client.py", "lib/ducky/src/ducky/server.py"],
)
def test_iris_and_ducky_changes_select_dependent_ducky_test(tmp_path: Path, changed_file: str) -> None:
    write(tmp_path, "lib/iris/src/iris/__init__.py")
    write(tmp_path, "lib/iris/src/iris/client.py", "class IrisClient: ...\n")
    write(tmp_path, "lib/ducky/src/ducky/__init__.py")
    write(tmp_path, "lib/ducky/src/ducky/server.py", "from iris.client import IrisClient\n")
    write(tmp_path, "lib/ducky/tests/test_server.py", "from ducky.server import IrisClient\n")

    matrix = select_matrix([changed_file], tmp_path)

    assert leg_paths(matrix, "ducky") == ["lib/ducky/tests/test_server.py"]


def test_test_helper_module_propagates_source_changes(tmp_path: Path) -> None:
    """A test reaching source only through a shared helper is still selected."""
    write(tmp_path, "lib/iris/src/iris/__init__.py")
    write(tmp_path, "lib/iris/src/iris/scheduler.py", "SCHED = 1\n")
    write(tmp_path, "lib/iris/tests/support.py", "from iris.scheduler import SCHED\n")
    write(tmp_path, "lib/iris/tests/test_via_helper.py", "from lib.iris.tests.support import SCHED\n")
    write(tmp_path, "lib/iris/tests/test_relative_helper.py", "from .support import SCHED\n")
    write(tmp_path, "lib/iris/tests/test_direct.py", "def test_x():\n    pass\n")

    matrix = select_matrix(["lib/iris/src/iris/scheduler.py"], tmp_path)

    assert leg_paths(matrix, "iris") == [
        "lib/iris/tests/test_relative_helper.py",
        "lib/iris/tests/test_via_helper.py",
    ]


def test_changed_test_module_runs_directly(tmp_path: Path) -> None:
    _workspace(tmp_path)
    write(tmp_path, "lib/iris/tests/test_new.py", "def test_x():\n    pass\n")

    assert select_matrix(["lib/iris/tests/test_new.py"], tmp_path) == [
        matrix_leg("iris", ["lib/iris/tests/test_new.py"])
    ]


def test_deleted_test_module_is_not_handed_to_pytest(tmp_path: Path) -> None:
    """git reports deleted paths; pytest aborts the whole run on a missing path."""
    _workspace(tmp_path)

    assert select_matrix(["lib/iris/tests/test_removed.py"], tmp_path) == []


def test_changed_helper_module_forces_full_scope(tmp_path: Path) -> None:
    """A changed helper under tests/ runs the full scope, even when named test_*.py."""
    write(tmp_path, "lib/iris/tests/test_utils.py", "def helper():\n    pass\n")
    result = classify(
        ["lib/iris/tests/e2e/gang_jax_smoke_workload.py", "lib/iris/tests/test_utils.py"],
        tmp_path,
    )

    assert result.forced == {"iris"}
    assert result.direct_tests == {}


def test_local_selection_targets_ci_tool_dependents(tmp_path: Path) -> None:
    write(tmp_path, "infra/ci/__init__.py")
    write(tmp_path, "infra/ci/select_tests.py", "def select():\n    pass\n")
    write(tmp_path, "infra/ci/analyze_import_graph.py", "from infra.ci.select_tests import select\n")
    write(tmp_path, "tests/infra/ci/test_analyze_import_graph.py", "from infra.ci.analyze_import_graph import select\n")
    write(tmp_path, "tests/infra/ci/test_select_tests.py", "from infra.ci.select_tests import select\n")

    selection = select_local_tests(
        ["infra/ci/select_tests.py", ".github/workflows/unified-unit.yaml"],
        tmp_path,
    )

    assert selection.reason == "diff-driven"
    assert leg_paths(selection.matrix, "marin") == [
        "tests/infra/ci/test_analyze_import_graph.py",
        "tests/infra/ci/test_select_tests.py",
    ]


def test_source_files_map_to_dotted_modules(tmp_path: Path) -> None:
    write(tmp_path, "lib/levanter/src/levanter/store/cache.py")
    assert classify(["lib/levanter/src/levanter/store/cache.py"], tmp_path).src_modules == {"levanter.store.cache"}

    write(tmp_path, "experiments/grug/moe/model.py")
    assert classify(["experiments/grug/moe/model.py"], tmp_path).src_modules == {"experiments.grug.moe.model"}


def test_evaldash_source_maps_to_dotted_module(tmp_path: Path) -> None:
    write(tmp_path, "infra/evaldash/src/metrics.py")
    assert classify(["infra/evaldash/src/metrics.py"], tmp_path).src_modules == {"infra.evaldash.src.metrics"}


def test_broad_trigger_does_not_source_build(tmp_path: Path) -> None:
    """A uv.lock bump reruns the full matrix but keeps every leg on the prebuilt wheel."""
    matrix = select_matrix(["uv.lock"], tmp_path)
    assert matrix, "broad trigger emits the full matrix"
    assert all(leg.setup == "" for leg in matrix)
