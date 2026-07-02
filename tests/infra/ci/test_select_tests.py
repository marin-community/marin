# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
import textwrap
from pathlib import Path

from infra.ci.select_tests import (
    SCOPES,
    classify,
    compute_matrix,
    downstream_modules,
    full_matrix,
    imports_touch_affected,
    path_to_module,
    top_level_imports,
)

# ---------------------------------------------------------------------------
# top_level_imports
# ---------------------------------------------------------------------------


def test_top_level_imports_regular_import(tmp_path):
    f = tmp_path / "mod.py"
    f.write_text("import os\nimport sys\n")
    assert top_level_imports(f) == {"os", "sys"}


def test_top_level_imports_from_import(tmp_path):
    f = tmp_path / "mod.py"
    f.write_text("from pathlib import Path\nfrom levanter.store import cache\n")
    assert top_level_imports(f) == {"pathlib", "levanter.store"}


def test_top_level_imports_skips_function_body(tmp_path):
    f = tmp_path / "mod.py"
    f.write_text(
        textwrap.dedent(
            """\
        import os
        def foo():
            import sys
            from levanter import store
        """
        )
    )
    assert top_level_imports(f) == {"os"}


def test_top_level_imports_skips_type_checking(tmp_path):
    f = tmp_path / "mod.py"
    f.write_text(
        textwrap.dedent(
            """\
        from __future__ import annotations
        from typing import TYPE_CHECKING
        if TYPE_CHECKING:
            from levanter.store import cache
        """
        )
    )
    assert "levanter.store" not in top_level_imports(f)


def test_top_level_imports_skips_class_body(tmp_path):
    f = tmp_path / "mod.py"
    f.write_text(
        textwrap.dedent(
            """\
        import os
        class Foo:
            import bar
        """
        )
    )
    assert top_level_imports(f) == {"os"}


def test_top_level_imports_syntax_error_returns_empty(tmp_path):
    f = tmp_path / "mod.py"
    f.write_text("def (broken:\n")
    assert top_level_imports(f) == set()


def test_top_level_imports_relative_imports(tmp_path):
    # Case 1: normal module, level 1 relative import
    f1 = tmp_path / "core.py"
    f1.write_text("from .axis import Axis\n")
    assert top_level_imports(f1, module_name="haliax.core") == {"haliax.axis"}

    # Case 2: __init__.py, level 1 relative import with module None
    f2 = tmp_path / "__init__.py"
    f2.write_text("from . import axis\n")
    assert top_level_imports(f2, module_name="haliax") == {"haliax.axis"}

    # Case 3: nested module, level 2 relative import
    f3 = tmp_path / "einsum.py"
    f3.write_text("from ..core import NamedArray\n")
    assert top_level_imports(f3, module_name="haliax._src.einsum") == {"haliax.core"}

    # Case 4: __init__.py, level 1 relative import of a sub-module
    f4 = tmp_path / "__init__.py"
    f4.write_text("from ._src.dot import dot\n")
    assert top_level_imports(f4, module_name="haliax") == {"haliax._src.dot"}

    # Fallback when module_name is None
    assert top_level_imports(f1, module_name=None) == {"axis"}


# ---------------------------------------------------------------------------
# path_to_module
# ---------------------------------------------------------------------------


def test_path_to_module_basic(tmp_path):
    path = tmp_path / "lib/levanter/src/levanter/store/cache.py"
    path.parent.mkdir(parents=True)
    path.touch()
    assert path_to_module(path, "levanter", tmp_path) == "levanter.store.cache"


def test_path_to_module_init_file(tmp_path):
    path = tmp_path / "lib/iris/src/iris/__init__.py"
    path.parent.mkdir(parents=True)
    path.touch()
    assert path_to_module(path, "iris", tmp_path) == "iris"


def test_path_to_module_nested(tmp_path):
    path = tmp_path / "lib/marin/src/marin/training/trainer.py"
    path.parent.mkdir(parents=True)
    path.touch()
    assert path_to_module(path, "marin", tmp_path) == "marin.training.trainer"


def test_path_to_module_wrong_scope_returns_none(tmp_path):
    path = tmp_path / "lib/iris/src/iris/controller.py"
    path.parent.mkdir(parents=True)
    path.touch()
    assert path_to_module(path, "levanter", tmp_path) is None


# ---------------------------------------------------------------------------
# imports_touch_affected
# ---------------------------------------------------------------------------


def test_imports_touch_affected_exact_match():
    assert imports_touch_affected({"levanter.store.cache"}, {"levanter.store.cache"})


def test_imports_touch_affected_parent_package():
    # "from levanter.store import cache" gives imp="levanter.store"; affected="levanter.store.cache"
    assert imports_touch_affected({"levanter.store"}, {"levanter.store.cache"})


def test_imports_touch_affected_top_level_package():
    assert imports_touch_affected({"levanter"}, {"levanter.store.cache"})


def test_imports_touch_affected_no_match():
    assert not imports_touch_affected({"zephyr.writers"}, {"levanter.store.cache"})


def test_imports_touch_affected_empty_imports():
    assert not imports_touch_affected(set(), {"levanter.store.cache"})


def test_imports_touch_affected_child_of_affected():
    # __init__.py change: affected="levanter.store"; test imports "levanter.store.cache"
    # Python runs levanter/store/__init__.py before loading cache, so test must be selected.
    assert imports_touch_affected({"levanter.store.cache"}, {"levanter.store"})


# ---------------------------------------------------------------------------
# downstream_modules
# ---------------------------------------------------------------------------


def test_downstream_modules_direct():
    reverse = {"a": {"b"}}
    assert downstream_modules({"a"}, reverse) >= {"a", "b"}


def test_downstream_modules_transitive():
    reverse = {"a": {"b"}, "b": {"c"}}
    assert downstream_modules({"a"}, reverse) == {"a", "b", "c"}


def test_downstream_modules_no_downstream():
    result = downstream_modules({"z"}, {"x": {"y"}})
    assert result == {"z"}


# ---------------------------------------------------------------------------
# classify
# ---------------------------------------------------------------------------


def test_classify_uv_lock_is_broad_trigger(tmp_path):
    broad, _, _, _ = classify(["uv.lock"], tmp_path)
    assert broad


def test_classify_root_pyproject_is_broad_trigger(tmp_path):
    broad, _, _, _ = classify(["pyproject.toml"], tmp_path)
    assert broad


def test_classify_self_is_broad_trigger(tmp_path):
    broad, _, _, _ = classify(["infra/ci/select_tests.py"], tmp_path)
    assert broad


def test_classify_unified_unit_workflow_is_broad_trigger(tmp_path):
    broad, _, _, _ = classify([".github/workflows/unified-unit.yaml"], tmp_path)
    assert broad


def test_classify_other_workflow_is_ignored(tmp_path):
    broad, modules, direct, forced = classify([".github/workflows/marin-unit.yaml"], tmp_path)
    assert not broad
    assert not modules
    assert not direct
    assert not forced


def test_classify_source_file_extracts_module(tmp_path):
    src = tmp_path / "lib/levanter/src/levanter/store/cache.py"
    src.parent.mkdir(parents=True)
    src.touch()
    _, modules, _, _ = classify(["lib/levanter/src/levanter/store/cache.py"], tmp_path)
    assert "levanter.store.cache" in modules


def test_classify_direct_test_file(tmp_path):
    _, _, direct, _ = classify(["lib/iris/tests/test_cluster.py"], tmp_path)
    assert "lib/iris/tests/test_cluster.py" in direct.get("iris", [])


def test_classify_non_python_test_asset_forces_scope(tmp_path):
    # A snapshot or fixture file under tests/ must force the full scope, not be
    # passed as a pytest target (which would cause a collection error).
    broad, _, direct, forced = classify(["tests/snapshots/expected/simple.md"], tmp_path)
    assert not broad
    assert "marin" in forced
    assert "simple.md" not in str(direct)


def test_classify_marin_direct_test(tmp_path):
    _, _, direct, _ = classify(["tests/test_something.py"], tmp_path)
    assert "tests/test_something.py" in direct.get("marin", [])


def test_classify_scope_conftest_forces_scope(tmp_path):
    _, _, _, forced = classify(["lib/iris/conftest.py"], tmp_path)
    assert "iris" in forced


def test_classify_tests_conftest_forces_scope(tmp_path):
    _, _, _, forced = classify(["lib/iris/tests/conftest.py"], tmp_path)
    assert "iris" in forced


def test_classify_empty_diff(tmp_path):
    broad, modules, direct, forced = classify([], tmp_path)
    assert not broad
    assert not modules
    assert not direct
    assert not forced


# ---------------------------------------------------------------------------
# full_matrix
# ---------------------------------------------------------------------------


def test_full_matrix_contains_all_scopes():
    matrix = full_matrix()
    packages = {e["package"] for e in matrix}
    assert packages == set(SCOPES)


# ---------------------------------------------------------------------------
# compute_matrix integration
# ---------------------------------------------------------------------------


def _make_workspace(tmp_path: Path) -> None:
    """Create a minimal fake workspace for integration tests."""
    # rigging: timing.py (source), test_timing.py (imports it), test_other.py (doesn't)
    (tmp_path / "lib/rigging/src/rigging").mkdir(parents=True)
    (tmp_path / "lib/rigging/src/rigging/timing.py").write_text("# timing\n")
    (tmp_path / "lib/rigging/src/rigging/other.py").write_text("# other\n")

    (tmp_path / "lib/rigging/tests").mkdir(parents=True)
    (tmp_path / "lib/rigging/tests/test_timing.py").write_text("from rigging import timing\n")
    (tmp_path / "lib/rigging/tests/test_other.py").write_text("import rigging.other\n")

    # iris: controller.py imports rigging.timing
    (tmp_path / "lib/iris/src/iris").mkdir(parents=True)
    (tmp_path / "lib/iris/src/iris/controller.py").write_text("import rigging.timing\n")
    (tmp_path / "lib/iris/tests").mkdir(parents=True)
    (tmp_path / "lib/iris/tests/test_controller.py").write_text("from iris import controller\n")


def test_compute_matrix_empty_returns_empty(tmp_path):
    assert compute_matrix(set(), {}, set(), tmp_path) == []


def test_compute_matrix_direct_test(tmp_path):
    direct = {"iris": ["lib/iris/tests/test_foo.py"]}
    matrix = compute_matrix(set(), direct, set(), tmp_path)
    assert len(matrix) == 1
    assert matrix[0]["package"] == "iris"
    assert "lib/iris/tests/test_foo.py" in matrix[0]["tests"]


def test_compute_matrix_forced_scope(tmp_path):
    matrix = compute_matrix(set(), {}, {"levanter"}, tmp_path)
    assert len(matrix) == 1
    assert matrix[0] == {"package": "levanter", "tests": []}


def test_compute_matrix_import_driven(tmp_path):
    _make_workspace(tmp_path)
    # rigging.timing changed: test_timing.py imports it (selected); test_other.py doesn't (skipped)
    matrix = compute_matrix({"rigging.timing"}, {}, set(), tmp_path)
    rigging = next((e for e in matrix if e["package"] == "rigging"), None)
    assert rigging is not None
    assert "lib/rigging/tests/test_timing.py" in rigging["tests"]
    assert "lib/rigging/tests/test_other.py" not in rigging["tests"]


def test_compute_matrix_transitive_cross_scope(tmp_path):
    _make_workspace(tmp_path)
    # rigging.timing changed; iris.controller imports rigging.timing;
    # test_controller.py imports iris.controller -> selected
    matrix = compute_matrix({"rigging.timing"}, {}, set(), tmp_path)
    iris = next((e for e in matrix if e["package"] == "iris"), None)
    assert iris is not None
    assert "lib/iris/tests/test_controller.py" in iris["tests"]


def test_compute_matrix_lazy_import_not_propagated(tmp_path):
    # zephyr.writers has a lazy import of rigging.timing inside a function
    (tmp_path / "lib/rigging/src/rigging").mkdir(parents=True)
    (tmp_path / "lib/rigging/src/rigging/timing.py").write_text("# timing\n")
    (tmp_path / "lib/zephyr/src/zephyr").mkdir(parents=True)
    (tmp_path / "lib/zephyr/src/zephyr/writers.py").write_text(
        textwrap.dedent(
            """\
        def write():
            from rigging import timing  # lazy import
        """
        )
    )
    (tmp_path / "lib/zephyr/tests").mkdir(parents=True)
    (tmp_path / "lib/zephyr/tests/test_writers.py").write_text("from zephyr import writers\n")
    matrix = compute_matrix({"rigging.timing"}, {}, set(), tmp_path)
    zephyr = next((e for e in matrix if e["package"] == "zephyr"), None)
    assert zephyr is None, "lazy import should not propagate to zephyr tests"
