# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the import-driven test selector (infra/ci/select_tests.py)."""

import textwrap
from pathlib import Path

from infra.ci.select_tests import (
    SCOPES,
    UV_PACKAGE,
    classify,
    compute_matrix,
    dependencies_by_test_file,
    extra_suites,
    full_matrix,
    is_test_module,
    matrix_leg,
)


def select_matrix(changed_files: list[str], repo_root: Path) -> list[dict[str, str]]:
    """Mirror the diff-driven branch of select_tests.main without git."""
    classification = classify(changed_files, repo_root)
    if classification.broad:
        return full_matrix()
    return compute_matrix(
        classification.src_modules,
        classification.direct_tests,
        classification.forced,
        repo_root,
    )


def write(repo_root: Path, relative: str, body: str = "") -> Path:
    path = repo_root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(body))
    return path


def leg_paths(matrix: list[dict[str, str]], scope: str) -> list[str]:
    leg = next(entry for entry in matrix if entry["package"] == UV_PACKAGE[scope])
    return leg["test_paths"].split()


def scopes_in(matrix: list[dict[str, str]]) -> set[str]:
    packages = {entry["package"] for entry in matrix}
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


def test_test_helper_module_propagates_source_changes(tmp_path: Path) -> None:
    """A test reaching source only through a shared helper is still selected."""
    write(tmp_path, "lib/iris/src/iris/__init__.py")
    write(tmp_path, "lib/iris/src/iris/scheduler.py", "SCHED = 1\n")
    write(tmp_path, "lib/iris/tests/support.py", "from iris.scheduler import SCHED\n")
    write(tmp_path, "lib/iris/tests/test_via_helper.py", "from tests.support import SCHED\n")
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
    """A changed non-collectable .py under tests/ runs the full scope, not the file itself."""
    result = classify(
        ["lib/iris/tests/e2e/gang_jax_smoke_workload.py", "lib/iris/tests/cluster/test_types.py"],
        tmp_path,
    )

    assert result.forced == {"iris"}
    assert result.direct_tests == {}, "the changed test module does not exist on disk"


def test_conftest_and_package_metadata_force_full_scope(tmp_path: Path) -> None:
    assert "iris" in classify(["lib/iris/conftest.py"], tmp_path).forced
    assert "iris" in classify(["lib/iris/tests/conftest.py"], tmp_path).forced
    assert "iris" in classify(["lib/iris/pyproject.toml"], tmp_path).forced
    assert "marin" in classify(["tests/snapshots/expected/simple.md"], tmp_path).forced


def test_classify_broad_triggers(tmp_path: Path) -> None:
    for path in ("uv.lock", "pyproject.toml", "infra/ci/select_tests.py", ".github/workflows/unified-unit.yaml"):
        assert classify([path], tmp_path).broad, path

    ignored = classify(["docs/index.md", "lib/iris/docs/coreweave.md"], tmp_path)
    assert not ignored.broad
    assert not ignored.src_modules
    assert not ignored.direct_tests
    assert not ignored.forced


def test_source_files_map_to_dotted_modules(tmp_path: Path) -> None:
    write(tmp_path, "lib/levanter/src/levanter/store/cache.py")
    assert classify(["lib/levanter/src/levanter/store/cache.py"], tmp_path).src_modules == {"levanter.store.cache"}

    write(tmp_path, "experiments/grug/moe/model.py")
    assert classify(["experiments/grug/moe/model.py"], tmp_path).src_modules == {"experiments.grug.moe.model"}


def test_extra_suites_follow_the_owning_package_directory() -> None:
    """Accelerator and browser suites drive whole subsystems, so directory membership gates them."""
    assert extra_suites(["lib/iris/dashboard/src/App.vue"]) == ["iris-e2e-smoke"]
    assert extra_suites(["lib/haliax/src/haliax/core.py"]) == ["levanter-torch", "levanter-tpu"]
    assert extra_suites(["lib/levanter/tests/test_attention.py"]) == ["levanter-torch", "levanter-tpu"]
    assert extra_suites(["lib/zephyr/src/zephyr/writers.py"]) == []
    assert extra_suites(["docs/index.md"]) == []
    assert extra_suites(["uv.lock"]) == ["iris-e2e-smoke", "levanter-torch", "levanter-tpu"]


def test_selector_changes_do_not_wake_the_accelerator_suites() -> None:
    """A broad trigger reruns every unit test, but the TPU runner is serialized and scarce:
    only a dependency or in-package change can move what those suites exercise."""
    assert classify(["infra/ci/select_tests.py"], Path("/unused")).broad
    assert extra_suites(["infra/ci/select_tests.py", ".github/workflows/unified-unit.yaml"]) == []


def test_only_pytest_collectable_modules_are_selectable(tmp_path: Path) -> None:
    """Helper modules under tests/ must never be handed to pytest explicitly -- an explicit
    path is imported even when it does not match the collection convention, crashing the
    lane when the helper's deps are absent."""
    write(tmp_path, "lib/iris/src/iris/__init__.py")
    write(tmp_path, "lib/iris/src/iris/scheduler.py", "SCHED = 1\n")
    for name in ("test_client.py", "actor_test.py", "gang_jax_smoke_workload.py", "conftest.py"):
        write(tmp_path, f"lib/iris/tests/{name}", "from iris.scheduler import SCHED\n")

    selected = dependencies_by_test_file("iris", tmp_path, {"iris", "iris.scheduler"})

    assert sorted(selected) == ["lib/iris/tests/actor_test.py", "lib/iris/tests/test_client.py"]


def test_is_test_module_matches_pytest_defaults() -> None:
    assert is_test_module("test_client.py")
    assert is_test_module("gpt2_test.py")
    assert not is_test_module("conftest.py")
    assert not is_test_module("openai_stub.py")
    assert not is_test_module("test_data.json")


def test_broad_trigger_runs_every_scope() -> None:
    assert matrix_leg("marin", []) == {
        "package": "marin-core",
        "extras": "--extra cpu --extra dedup",
        "test_paths": "tests",
    }
    assert select_matrix(["uv.lock"], Path("/unused")) == full_matrix()
