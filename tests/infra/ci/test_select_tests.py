# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the import-driven test selector (infra/ci/select_tests.py)."""

import textwrap
from pathlib import Path

import pytest

from infra.ci.select_tests import (
    MIN_FILES_PER_SHARD,
    SCOPES,
    SHARD_COUNT,
    UV_PACKAGE,
    accelerator_suite_test_paths,
    classify,
    dependencies_by_test_file,
    extra_suites,
    full_matrix,
    is_test_module,
    matrix_leg,
    scope_legs,
    select_changed_tests,
    shard_files,
    torch_membership_for_test_file,
)


def select_matrix(changed_files: list[str], repo_root: Path) -> list[dict[str, str | int]]:
    """Return the selector's diff-driven matrix without invoking git."""
    return select_changed_tests(changed_files, repo_root).matrix


def select_accelerator_paths(changed_files: list[str], repo_root: Path) -> dict[str, list[str]]:
    matrix = select_matrix(changed_files, repo_root)
    return accelerator_suite_test_paths(changed_files, matrix, repo_root)


def write(repo_root: Path, relative: str, body: str = "") -> Path:
    path = repo_root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(body))
    return path


def leg_paths(matrix: list[dict[str, str | int]], scope: str) -> list[str]:
    leg = next(entry for entry in matrix if entry["package"] == UV_PACKAGE[scope])
    return str(leg["test_paths"]).split()


def scopes_in(matrix: list[dict[str, str | int]]) -> set[str]:
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
    """A changed helper under tests/ runs the full scope, even when named test_*.py."""
    write(tmp_path, "lib/iris/tests/test_utils.py", "def helper():\n    pass\n")
    result = classify(
        ["lib/iris/tests/e2e/gang_jax_smoke_workload.py", "lib/iris/tests/test_utils.py"],
        tmp_path,
    )

    assert result.forced == {"iris"}
    assert result.direct_tests == {}


def test_conftest_and_package_metadata_force_full_scope(tmp_path: Path) -> None:
    assert "iris" in classify(["lib/iris/conftest.py"], tmp_path).forced
    assert "iris" in classify(["lib/iris/tests/conftest.py"], tmp_path).forced
    assert "iris" in classify(["lib/iris/pyproject.toml"], tmp_path).forced
    assert "marin" in classify(["tests/snapshots/expected/simple.md"], tmp_path).forced


def test_classify_broad_triggers(tmp_path: Path) -> None:
    for path in (
        "uv.lock",
        "pyproject.toml",
        "infra/ci/run_tests.py",
        "infra/ci/select_tests.py",
        ".github/workflows/unified-unit.yaml",
    ):
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


def test_evaldash_source_maps_to_dotted_module(tmp_path: Path) -> None:
    write(tmp_path, "infra/evaldash/src/metrics.py")
    assert classify(["infra/evaldash/src/metrics.py"], tmp_path).src_modules == {"infra.evaldash.src.metrics"}


def _write_evaldash_workspace(repo_root: Path) -> None:
    """An evaldash package and the tests that import it."""
    write(repo_root, "infra/evaldash/src/metrics.py", "def build_matrix():\n    pass\n")
    write(repo_root, "infra/evaldash/src/samples.py", "def fetch_samples():\n    pass\n")
    write(repo_root, "infra/evaldash/src/fixtures.py", "def build_fixtures():\n    pass\n")
    write(
        repo_root,
        "infra/evaldash/src/server.py",
        "from infra.evaldash.src import samples\nfrom infra.evaldash.src.metrics import build_matrix\n",
    )
    write(
        repo_root, "tests/evaluation/test_evaldash_metrics.py", "from infra.evaldash.src.metrics import build_matrix\n"
    )
    write(
        repo_root,
        "tests/evaluation/test_evaldash_local_store.py",
        "from infra.evaldash.src import fixtures, server\n",
    )


@pytest.mark.parametrize(
    ("changed_file", "selected_test"),
    [
        ("infra/evaldash/src/metrics.py", "tests/evaluation/test_evaldash_metrics.py"),
        ("infra/evaldash/src/server.py", "tests/evaluation/test_evaldash_local_store.py"),
    ],
    ids=["metrics", "server"],
)
def test_evaldash_change_selects_importing_test(tmp_path: Path, changed_file: str, selected_test: str) -> None:
    _write_evaldash_workspace(tmp_path)

    matrix = select_matrix([changed_file], tmp_path)

    assert selected_test in leg_paths(matrix, "marin")


def test_extra_suites_follow_the_owning_package_directory() -> None:
    """Only suites without importable test dependencies use directory triggers."""
    assert extra_suites(["lib/iris/dashboard/src/App.vue"]) == ["iris-e2e-smoke"]
    assert extra_suites(["lib/haliax/src/haliax/core.py"]) == []
    assert extra_suites(["lib/levanter/tests/test_attention.py"]) == []
    assert extra_suites(["lib/zephyr/src/zephyr/writers.py"]) == []
    assert extra_suites(["docs/index.md"]) == []
    assert extra_suites(["uv.lock"]) == ["iris-e2e-smoke"]


def _levanter_accelerator_workspace(repo_root: Path) -> None:
    write(repo_root, "lib/haliax/src/haliax/__init__.py", '"""haliax."""\n')
    write(repo_root, "lib/haliax/src/haliax/core.py", "X = 1\n")
    write(repo_root, "lib/levanter/src/levanter/__init__.py", '"""levanter."""\n')
    write(repo_root, "lib/levanter/src/levanter/model.py", "from haliax.core import X\n")
    write(repo_root, "lib/levanter/src/levanter/unrelated.py", "Y = 2\n")
    write(
        repo_root,
        "lib/levanter/tests/test_model.py",
        """\
        from levanter.model import X
        from test_utils import skip_if_no_torch

        def test_cpu():
            assert X == 1

        @skip_if_no_torch
        def test_torch():
            assert X == 1
        """,
    )
    write(
        repo_root,
        "lib/levanter/tests/test_unrelated.py",
        "from levanter.unrelated import Y\n\ndef test_unrelated():\n    assert Y == 2\n",
    )


@pytest.mark.parametrize(
    "changed_file",
    ["lib/levanter/src/levanter/model.py", "lib/haliax/src/haliax/core.py"],
)
def test_accelerator_suites_reuse_import_selected_levanter_files(tmp_path: Path, changed_file: str) -> None:
    _levanter_accelerator_workspace(tmp_path)

    paths = select_accelerator_paths([changed_file], tmp_path)

    expected = ["lib/levanter/tests/test_model.py"]
    assert paths == {"levanter-torch": expected, "levanter-tpu": expected}


def test_accelerator_suites_split_torch_and_non_torch_files(tmp_path: Path) -> None:
    write(tmp_path, "lib/levanter/src/levanter/__init__.py")
    write(tmp_path, "lib/levanter/src/levanter/core.py", "X = 1\n")
    write(
        tmp_path,
        "lib/levanter/tests/test_torch_only.py",
        """\
        from levanter.core import X
        from test_utils import skip_if_no_torch

        @skip_if_no_torch
        def test_torch():
            assert X == 1
        """,
    )
    write(
        tmp_path,
        "lib/levanter/tests/test_cpu_only.py",
        """\
        from levanter.core import X

        def test_cpu():
            assert X == 1
        """,
    )

    paths = select_accelerator_paths(["lib/levanter/src/levanter/core.py"], tmp_path)

    assert paths == {
        "levanter-torch": ["lib/levanter/tests/test_torch_only.py"],
        "levanter-tpu": ["lib/levanter/tests/test_cpu_only.py"],
    }


def test_tpu_ignored_file_does_not_start_an_accelerator_suite(tmp_path: Path) -> None:
    write(tmp_path, "lib/levanter/tests/test_new_cache.py", "def test_cache():\n    pass\n")

    assert select_accelerator_paths(["lib/levanter/tests/test_new_cache.py"], tmp_path) == {}


def test_docs_only_levanter_change_does_not_start_accelerator_suites(tmp_path: Path) -> None:
    _levanter_accelerator_workspace(tmp_path)

    assert select_accelerator_paths(["lib/levanter/docs/index.md"], tmp_path) == {}


def test_torch_membership_handles_module_and_class_markers(tmp_path: Path) -> None:
    module_marked = write(
        tmp_path,
        "test_module.py",
        """\
        import pytest
        pytestmark = pytest.mark.torch

        def test_one():
            pass
        """,
    )
    mixed = write(
        tmp_path,
        "test_mixed.py",
        """\
        import pytest

        class TestParity:
            @pytest.mark.torch
            def test_torch(self):
                pass

            def test_cpu(self):
                pass
        """,
    )

    assert torch_membership_for_test_file(module_marked) == (True, False)
    assert torch_membership_for_test_file(mixed) == (True, True)


def test_selector_changes_are_broad_without_waking_the_browser_suite() -> None:
    assert classify(["infra/ci/select_tests.py"], Path("/unused")).broad
    assert extra_suites(["infra/ci/select_tests.py", ".github/workflows/unified-unit.yaml"]) == []


@pytest.mark.parametrize("changed_file", ["infra/ci/select_tests.py", ".github/workflows/unified-unit.yaml"])
def test_selector_and_workflow_changes_run_full_accelerator_suites(tmp_path: Path, changed_file: str) -> None:
    _levanter_accelerator_workspace(tmp_path)

    paths = select_accelerator_paths([changed_file], tmp_path)

    assert paths == {
        "levanter-torch": ["lib/levanter/tests/test_model.py"],
        "levanter-tpu": [
            "lib/levanter/tests/test_model.py",
            "lib/levanter/tests/test_unrelated.py",
        ],
    }


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


def test_shard_files_splits_into_balanced_contiguous_chunks() -> None:
    assert shard_files(list(range(10)), 4) == [[0, 1, 2], [3, 4, 5], [6, 7], [8, 9]]
    assert shard_files(list(range(4)), 4) == [[0], [1], [2], [3]]
    # No file is dropped or duplicated, and order is preserved.
    files = [f"t{i}" for i in range(23)]
    chunks = shard_files(files, 4)
    assert [f for chunk in chunks for f in chunk] == files


def _levanter_suite(repo_root: Path, count: int) -> list[str]:
    write(repo_root, "lib/levanter/src/levanter/__init__.py", '"""levanter."""\n')
    write(repo_root, "lib/levanter/src/levanter/core.py", "X = 1\n")
    files = []
    for i in range(count):
        path = f"lib/levanter/tests/test_mod_{i:03d}.py"
        write(repo_root, path, "from levanter.core import X\n")
        files.append(path)
    return sorted(files)


def test_scope_legs_shards_a_large_levanter_selection(tmp_path: Path) -> None:
    files = _levanter_suite(tmp_path, 60)

    legs = scope_legs("levanter", files, tmp_path)

    assert len(legs) == SHARD_COUNT["levanter"]
    assert [leg["label"] for leg in legs] == ["levanter 1/4", "levanter 2/4", "levanter 3/4", "levanter 4/4"]
    # Every selected file runs exactly once across the shards.
    covered = [path for leg in legs for path in leg["test_paths"].split()]
    assert sorted(covered) == files
    assert len(covered) == len(set(covered))


def test_scope_legs_keeps_a_small_selection_in_one_leg(tmp_path: Path) -> None:
    files = _levanter_suite(tmp_path, MIN_FILES_PER_SHARD)

    legs = scope_legs("levanter", files, tmp_path)

    assert len(legs) == 1
    assert legs[0]["label"] == "levanter"


def test_scope_legs_never_shards_below_the_minimum(tmp_path: Path) -> None:
    """A medium selection stays one leg rather than splitting into sub-minimum runners."""
    # Just over the threshold and just under two full shards both stay a single leg...
    for count in (MIN_FILES_PER_SHARD + 1, 2 * MIN_FILES_PER_SHARD - 1):
        legs = scope_legs("levanter", _levanter_suite(tmp_path, count), tmp_path)
        assert [leg["label"] for leg in legs] == ["levanter"], count

    # ...and two full shards' worth is the first size that fans out, each at/above the minimum.
    legs = scope_legs("levanter", _levanter_suite(tmp_path, 2 * MIN_FILES_PER_SHARD), tmp_path)
    assert len(legs) == 2
    assert all(len(leg["test_paths"].split()) >= MIN_FILES_PER_SHARD for leg in legs)


def test_scope_legs_shards_the_full_levanter_suite(tmp_path: Path) -> None:
    """A full-suite (tests=None) sharded scope expands its directory to the file list."""
    files = _levanter_suite(tmp_path, 60)

    legs = scope_legs("levanter", None, tmp_path)

    assert len(legs) == SHARD_COUNT["levanter"]
    covered = sorted(path for leg in legs for path in leg["test_paths"].split())
    assert covered == files


def test_scope_legs_does_not_shard_other_scopes(tmp_path: Path) -> None:
    write(tmp_path, "lib/iris/src/iris/__init__.py", '"""iris."""\n')
    write(tmp_path, "lib/iris/src/iris/core.py", "X = 1\n")
    files = [f"lib/iris/tests/test_{i:03d}.py" for i in range(60)]
    for path in files:
        write(tmp_path, path, "from iris.core import X\n")

    legs = scope_legs("iris", sorted(files), tmp_path)

    assert len(legs) == 1
    assert legs[0]["label"] == "iris"
    assert legs[0]["test_paths"].split() == sorted(files)


def test_broad_trigger_runs_every_scope() -> None:
    assert matrix_leg("marin", []) == {
        "label": "marin",
        "package": "marin-core",
        "python": "3.12",
        "extras": "--extra cpu --extra dedup",
        "pytest_args": "--durations=5 -n auto --dist=worksteal --tb=short",
        "test_paths": "tests",
        "setup": "",
        "timeout": 15,
    }
    assert select_matrix(["uv.lock"], Path("/unused")) == full_matrix(Path("/unused"), set())


def _leg(matrix: list[dict[str, str | int]], label: str) -> dict[str, str | int]:
    return next(entry for entry in matrix if entry["label"] == label)


def test_native_rust_change_forces_the_owning_scope(tmp_path: Path) -> None:
    """A rust/ change is invisible to the import graph, so it force-selects its owning scope
    and marks it for a source build."""
    result = classify(["lib/dupekit/rust/src/lib.rs"], tmp_path)
    assert result.forced == {"dupekit"}
    assert result.native_changed == {"dupekit"}
    # A Cargo.lock under the crate counts as a native change too.
    assert classify(["lib/finelog/rust/Cargo.lock"], tmp_path).native_changed == {"finelog"}


def test_native_rust_only_change_runs_just_the_owning_scope(tmp_path: Path) -> None:
    """A rust-only change runs the owning scope from source; its own tests cover the native,
    and consumers are not pulled in."""
    matrix = select_matrix(["lib/dupekit/rust/src/lib.rs"], tmp_path)
    assert scopes_in(matrix) == {"dupekit"}
    assert _leg(matrix, "dupekit")["setup"] == "rust"
    assert _leg(matrix, "dupekit")["timeout"] == 30


def test_native_change_source_builds_only_the_owning_scope(tmp_path: Path) -> None:
    """A finelog rust change source-builds only the finelog leg; a co-changed consumer (iris)
    is selected by its own Python change and runs against the prebuilt wheel."""
    write(tmp_path, "lib/iris/src/iris/__init__.py")
    write(tmp_path, "lib/iris/src/iris/log.py", "X = 1\n")
    write(tmp_path, "lib/iris/tests/test_log.py", "from iris.log import X\n")

    matrix = select_matrix(["lib/finelog/rust/pyext/src/lib.rs", "lib/iris/src/iris/log.py"], tmp_path)

    assert _leg(matrix, "finelog")["setup"] == "rust"
    assert _leg(matrix, "iris")["setup"] == ""


def test_broad_trigger_does_not_source_build(tmp_path: Path) -> None:
    """A uv.lock bump reruns the full matrix but keeps every leg on the prebuilt wheel."""
    matrix = select_matrix(["uv.lock"], tmp_path)
    assert matrix, "broad trigger emits the full matrix"
    assert all(leg["setup"] == "" for leg in matrix)
