# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the import-driven test selector (infra/ci/select_tests.py)."""

import subprocess
import textwrap
from collections.abc import Callable
from pathlib import Path

import pytest

from infra.ci.select_tests import (
    SCOPES,
    UV_PACKAGE,
    MatrixLeg,
    SelectionResult,
    classify,
    matrix_leg,
    select_all_tests,
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


@pytest.mark.parametrize("select_tests", [select_changed_tests, select_local_tests])
@pytest.mark.parametrize(
    "changed_file",
    ["experiments/moe/test_optimizer.py", "experiments/moe/optimizer.py", "lib/levanter/src/levanter/optim.py"],
)
def test_experiment_tests_run_for_test_and_dependency_changes(
    tmp_path: Path, select_tests: Callable[[list[str], Path], SelectionResult], changed_file: str
) -> None:
    write(tmp_path, "lib/levanter/src/levanter/optim.py", "RATE = 1\n")
    write(tmp_path, "experiments/moe/optimizer.py", "from levanter.optim import RATE\n")
    write(
        tmp_path,
        "experiments/moe/test_optimizer.py",
        "from experiments.moe.optimizer import RATE\n\ndef test_rate():\n    assert RATE == 1\n",
    )
    write(tmp_path, "experiments/moe/test_unrelated.py", "def test_other():\n    assert True\n")
    write(tmp_path, "tests/test_unrelated.py", "def test_other():\n    assert True\n")

    selection = select_tests([changed_file], tmp_path)

    assert leg_paths(selection.matrix, "marin") == ["experiments/moe/test_optimizer.py"]


@pytest.mark.parametrize(
    "changed_files, run_all_tests",
    [
        ([], True),
        (["pyproject.toml"], False),
        (["experiments/moe/conftest.py"], False),
        (["experiments/moe/fixtures/weights.json"], False),
    ],
)
def test_full_marin_suite_includes_experiment_tests(
    tmp_path: Path, changed_files: list[str], run_all_tests: bool
) -> None:
    write(tmp_path, "tests/test_root.py", "def test_root():\n    assert True\n")
    write(tmp_path, "experiments/moe/test_optimizer.py", "def test_optimizer():\n    assert True\n")

    selection = select_changed_tests(changed_files, tmp_path, run_all_tests=run_all_tests)

    assert leg_paths(selection.matrix, "marin") == ["tests", "experiments"]


def test_deleted_experiment_source_runs_full_marin_suite(tmp_path: Path) -> None:
    write(tmp_path, "experiments/moe/test_optimizer.py", "from experiments.moe.optimizer import RATE\n")

    matrix = select_matrix(["experiments/moe/optimizer.py"], tmp_path)

    assert leg_paths(matrix, "marin") == ["tests", "experiments"]


def test_deleted_experiment_test_is_not_handed_to_pytest(tmp_path: Path) -> None:
    write(tmp_path, "experiments/moe/test_other.py", "def test_other():\n    assert True\n")

    assert select_matrix(["experiments/moe/test_removed.py"], tmp_path) == []


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


@pytest.mark.parametrize(
    "changed_path,suites",
    [
        (
            "lib/taskcompendium/src/taskcompendium/lowering.py",
            ["rolloutengine-unit", "taskcompendium-unit", "taskforge-unit"],
        ),
        ("lib/rolloutengine/src/rolloutengine/engine.py", ["rolloutengine-unit", "taskforge-unit"]),
        ("lib/shellbox/src/shellbox/machine.py", ["rolloutengine-unit", "taskforge-unit"]),
        (
            "lib/verifyit/src/verifyit/grading.py",
            ["rolloutengine-unit", "taskcompendium-unit", "taskforge-unit"],
        ),
        ("lib/rigging/src/rigging/message.py", ["rolloutengine-unit", "taskcompendium-unit", "taskforge-unit"]),
        ("lib/taskforge/src/taskforge/llm/client.py", ["taskforge-unit"]),
    ],
)
def test_execution_dependencies_select_isolated_suites(tmp_path: Path, changed_path: str, suites: list[str]) -> None:
    selection = select_changed_tests([changed_path], tmp_path)

    assert selection.matrix == []
    assert selection.suites == suites

    full_selection = select_changed_tests([], tmp_path, run_all_tests=True)
    assert set(suites) <= set(full_selection.suites)


@pytest.mark.parametrize("changed_file", ["pyproject.toml", "uv.lock"])
def test_shared_dependency_change_selects_taskcompendium_harbor_suite(tmp_path: Path, changed_file: str) -> None:
    selection = select_changed_tests([changed_file], tmp_path)

    assert "taskcompendium-unit" in selection.suites


def test_verifier_change_selects_library_and_dependent_marin_tests(tmp_path: Path) -> None:
    write(tmp_path, "lib/verifyit/src/verifyit/__init__.py")
    write(tmp_path, "lib/verifyit/src/verifyit/grade.py", "def grade(): ...\n")
    write(tmp_path, "lib/verifyit/tests/test_grade.py", "from verifyit.grade import grade\n")
    write(tmp_path, "tests/test_verifier.py", "from verifyit.grade import grade\n")

    matrix = select_matrix(["lib/verifyit/src/verifyit/grade.py"], tmp_path)

    assert leg_paths(matrix, "verifyit") == ["lib/verifyit/tests/test_grade.py"]
    assert leg_paths(matrix, "marin") == ["tests/test_verifier.py"]
    verifier_leg = next(leg for leg in matrix if leg.package == "verifyit")
    assert verifier_leg.extras == "--extra all"


@pytest.mark.parametrize("run_all_tests", [False, True])
def test_verifier_manifest_and_full_runs_select_entire_library(tmp_path: Path, run_all_tests: bool) -> None:
    write(tmp_path, "lib/verifyit/tests/test_grade.py", "def test_grade(): ...\n")

    selection = select_changed_tests(
        [] if run_all_tests else ["lib/verifyit/pyproject.toml"], tmp_path, run_all_tests=run_all_tests
    )

    assert leg_paths(selection.matrix, "verifyit") == ["lib/verifyit/tests"]


def _verifier_members(verifier_source: str) -> str:
    if verifier_source == "workspace":
        return '"lib/levanter", "lib/haliax", "lib/verifyit"'
    return '"lib/levanter", "lib/haliax"'


def _tpu_lock(
    *,
    jax_version: str = "0.11.1",
    shared_version: str = "1",
    leaf_version: str = "1",
    verifier_source: str = "workspace",
    tpu_marker: str = "",
    jax_source: str = 'registry = "https://pypi.org/simple"',
) -> str:
    source = (
        'editable = "lib/verifyit"'
        if verifier_source == "workspace"
        else ('git = "https://github.com/marin-community/verifyit?rev=abc123"')
    )
    marker = f', marker = "{tpu_marker}"' if tpu_marker else ""
    return f"""\
    version = 1
    requires-python = ">=3.12"
    [manifest]
    members = [{_verifier_members(verifier_source)}]
    [[package]]
    name = "marin-root"
    version = "0.1.0"
    source = {{ editable = "." }}
    dependencies = [{{ name = "verifyit" }}]
    [[package]]
    name = "verifyit"
    version = "0.1.0"
    source = {{ {source} }}
    [[package]]
    name = "marin-levanter"
    version = "0.2.0"
    source = {{ editable = "lib/levanter" }}
    dependencies = [{{ name = "marin-haliax" }}, {{ name = "shared" }}]
    [package.optional-dependencies]
    tpu = [{{ name = "jax", version = "{jax_version}"{marker} }}]
    [package.dev-dependencies]
    test = [{{ name = "pytest" }}]
    [[package]]
    name = "marin-haliax"
    source = {{ editable = "lib/haliax" }}
    dependencies = [{{ name = "shared" }}]
    [[package]]
    name = "shared"
    version = "{shared_version}"
    source = {{ registry = "https://pypi.org/simple" }}
    dependencies = [{{ name = "leaf" }}]
    [[package]]
    name = "leaf"
    version = "{leaf_version}"
    source = {{ registry = "https://pypi.org/simple" }}
    [[package]]
    name = "jax"
    version = "{jax_version}"
    source = {{ {jax_source} }}
    [[package]]
    name = "pytest"
    version = "8.0.0"
    source = {{ registry = "https://pypi.org/simple" }}
    """


def _tpu_manifest(verifier_source: str = "workspace") -> str:
    source = (
        "{ workspace = true }"
        if verifier_source == "workspace"
        else ('{ git = "https://github.com/marin-community/verifyit", rev = "abc123" }')
    )
    return f"""\
    [project]
    name = "marin-root"
    requires-python = ">=3.12"
    dependencies = ["verifyit"]
    [tool.uv.workspace]
    members = [{_verifier_members(verifier_source)}]
    [tool.uv.sources]
    verifyit = {source}
    """


def _commit_base_tpu_workspace(tmp_path: Path) -> str:
    write(tmp_path, "uv.lock", _tpu_lock())
    write(tmp_path, "pyproject.toml", _tpu_manifest())
    write(tmp_path, "lib/levanter/tests/test_model.py", "def test_model():\n    assert True\n")
    write(tmp_path, "lib/levanter/tests/test_torch.py", "@pytest.mark.torch\ndef test_torch():\n    assert True\n")
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "-c", "user.name=Test", "-c", "user.email=test@example.com", "commit", "-qm", "base"],
        cwd=tmp_path,
        check=True,
    )
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=tmp_path, text=True).strip()


def test_verifier_git_source_keeps_cpu_coverage_without_tpu(tmp_path: Path) -> None:
    base = _commit_base_tpu_workspace(tmp_path)
    write(tmp_path, "uv.lock", _tpu_lock(verifier_source="git"))
    write(tmp_path, "pyproject.toml", _tpu_manifest(verifier_source="git"))

    selection = select_changed_tests(["uv.lock", "pyproject.toml"], tmp_path, base_ref=base)

    assert selection.reason == "broad-trigger"
    assert "marin-levanter" in {leg.package for leg in selection.matrix}
    assert "levanter-tpu" not in selection.suites
    assert selection.suite_test_paths["levanter-torch"] == ["lib/levanter/tests/test_torch.py"]


@pytest.mark.parametrize(
    "lock_change",
    [
        {"jax_version": "0.11.2"},
        {"shared_version": "2"},
        {"leaf_version": "2"},
        {"tpu_marker": "python_version >= '3.12'"},
        {"jax_source": 'git = "https://github.com/jax-ml/jax?rev=abc123"'},
    ],
)
def test_reachable_dependency_changes_select_tpu(tmp_path: Path, lock_change: dict[str, str]) -> None:
    base = _commit_base_tpu_workspace(tmp_path)
    write(tmp_path, "uv.lock", _tpu_lock(**lock_change))

    selection = select_changed_tests(["uv.lock"], tmp_path, base_ref=base)

    assert selection.suite_test_paths["levanter-tpu"] == ["lib/levanter/tests/test_model.py"]


@pytest.mark.parametrize("missing", [True, False])
def test_unavailable_or_invalid_dependency_graph_selects_tpu(tmp_path: Path, missing: bool) -> None:
    base = _commit_base_tpu_workspace(tmp_path)
    if missing:
        (tmp_path / "uv.lock").unlink()
    else:
        write(tmp_path, "uv.lock", "[[package]\n")

    selection = select_changed_tests(["uv.lock"], tmp_path, base_ref=base)

    assert "levanter-tpu" in selection.suites


def test_scheduled_full_suite_still_selects_tpu(tmp_path: Path) -> None:
    write(tmp_path, "lib/levanter/tests/test_model.py", "def test_model():\n    assert True\n")

    selection = select_all_tests(tmp_path)

    assert selection.reason == "run-all-tests"
    assert selection.suite_test_paths["levanter-tpu"] == ["lib/levanter/tests/test_model.py"]


@pytest.mark.parametrize(
    "path",
    [
        "lib/levanter/src/levanter/model.py",
        "lib/haliax/src/haliax/core.py",
        "infra/ci/select_tests.py",
        ".github/workflows/unified-unit.yaml",
    ],
)
def test_source_and_ci_changes_still_select_tpu(tmp_path: Path, path: str) -> None:
    write(tmp_path, "lib/levanter/tests/test_model.py", "def test_model():\n    assert True\n")
    selection = select_changed_tests([path], tmp_path, run_all_tests=True)

    assert "levanter-tpu" in selection.suites


def test_source_files_map_to_dotted_modules(tmp_path: Path) -> None:
    write(tmp_path, "lib/levanter/src/levanter/store/cache.py")
    assert classify(["lib/levanter/src/levanter/store/cache.py"], tmp_path).src_modules == {"levanter.store.cache"}

    write(tmp_path, "experiments/grug/moe/model.py")
    assert classify(["experiments/grug/moe/model.py"], tmp_path).src_modules == {"experiments.grug.moe.model"}


def test_evaldash_source_maps_to_dotted_module(tmp_path: Path) -> None:
    write(tmp_path, "infra/marina/apps/evaldash/metrics.py")
    assert classify(["infra/marina/apps/evaldash/metrics.py"], tmp_path).src_modules == {"evaldash.metrics"}


def test_deploy_change_selects_deploy_test(tmp_path: Path) -> None:
    write(tmp_path, "infra/deploy/src/marin_deploy/__init__.py")
    write(tmp_path, "infra/deploy/src/marin_deploy/cli.py", "def cli(): ...\n")
    write(
        tmp_path,
        "infra/deploy/tests/test_cli.py",
        "from marin_deploy.cli import cli\n\ndef test_cli(): ...\n",
    )

    matrix = select_matrix(["infra/deploy/src/marin_deploy/cli.py"], tmp_path)

    assert leg_paths(matrix, "deploy") == ["infra/deploy/tests/test_cli.py"]


def test_broad_trigger_does_not_source_build(tmp_path: Path) -> None:
    """A uv.lock bump reruns the full matrix but keeps every leg on the prebuilt wheel."""
    matrix = select_matrix(["uv.lock"], tmp_path)
    assert matrix, "broad trigger emits the full matrix"
    assert all(leg.setup == "" for leg in matrix)
