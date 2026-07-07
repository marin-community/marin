# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
import textwrap
from pathlib import Path

from infra.ci.select_tests import (
    SCOPES,
    UV_PACKAGE,
    classify,
    compute_matrix,
    full_matrix,
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


def _make_workspace(repo_root: Path) -> None:
    """Minimal multi-scope workspace for import-graph selection."""
    (repo_root / "lib/rigging/src/rigging").mkdir(parents=True)
    (repo_root / "lib/rigging/src/rigging/timing.py").write_text("# timing\n")
    (repo_root / "lib/rigging/src/rigging/other.py").write_text("# other\n")
    (repo_root / "lib/rigging/tests").mkdir(parents=True)
    (repo_root / "lib/rigging/tests/test_timing.py").write_text("from rigging import timing\n")
    (repo_root / "lib/rigging/tests/test_other.py").write_text("import rigging.other\n")

    (repo_root / "lib/iris/src/iris").mkdir(parents=True)
    (repo_root / "lib/iris/src/iris/controller.py").write_text("import rigging.timing\n")
    (repo_root / "lib/iris/tests").mkdir(parents=True)
    (repo_root / "lib/iris/tests/test_controller.py").write_text("from iris import controller\n")

    (repo_root / "lib/zephyr/src/zephyr").mkdir(parents=True)
    (repo_root / "lib/zephyr/src/zephyr/writers.py").write_text(
        textwrap.dedent(
            """\
            def write():
                from rigging import timing  # lazy import
            """
        )
    )
    (repo_root / "lib/zephyr/tests").mkdir(parents=True)
    (repo_root / "lib/zephyr/tests/test_writers.py").write_text("from zephyr import writers\n")


def _leg(package: str) -> dict[str, str] | None:
    return next((entry for entry in full_matrix() if entry["package"] == package), None)


def test_classify_changed_files(tmp_path: Path) -> None:
    for path in (
        "uv.lock",
        "pyproject.toml",
        "infra/ci/select_tests.py",
        ".github/workflows/unified-unit.yaml",
    ):
        assert classify([path], tmp_path).broad, path

    ignored = classify([".github/workflows/marin-unit.yaml"], tmp_path)
    assert not ignored.broad
    assert not ignored.src_modules
    assert not ignored.direct_tests
    assert not ignored.forced

    (tmp_path / "lib/levanter/src/levanter/store").mkdir(parents=True)
    (tmp_path / "lib/levanter/src/levanter/store/cache.py").touch()
    src = classify(["lib/levanter/src/levanter/store/cache.py"], tmp_path)
    assert "levanter.store.cache" in src.src_modules

    direct = classify(["lib/iris/tests/test_cluster.py"], tmp_path)
    assert direct.direct_tests["iris"] == ["lib/iris/tests/test_cluster.py"]

    marin_test = classify(["tests/test_something.py"], tmp_path)
    assert marin_test.direct_tests["marin"] == ["tests/test_something.py"]

    snapshot = classify(["tests/snapshots/expected/simple.md"], tmp_path)
    assert "marin" in snapshot.forced
    assert not snapshot.direct_tests

    assert "iris" in classify(["lib/iris/conftest.py"], tmp_path).forced
    assert "iris" in classify(["lib/iris/tests/conftest.py"], tmp_path).forced

    empty = classify([], tmp_path)
    assert not empty.broad
    assert not empty.src_modules
    assert not empty.direct_tests
    assert not empty.forced


def test_select_matrix_from_workspace(tmp_path: Path) -> None:
    _make_workspace(tmp_path)

    assert select_matrix([], tmp_path) == []

    matrix = select_matrix(["lib/rigging/src/rigging/timing.py"], tmp_path)
    rigging = next(entry for entry in matrix if entry["package"] == UV_PACKAGE["rigging"])
    iris = next(entry for entry in matrix if entry["package"] == UV_PACKAGE["iris"])
    assert "lib/rigging/tests/test_timing.py" in rigging["test_paths"]
    assert "lib/rigging/tests/test_other.py" not in rigging["test_paths"]
    assert "lib/iris/tests/test_controller.py" in iris["test_paths"]
    assert not any(entry["package"] == UV_PACKAGE["zephyr"] for entry in matrix)

    direct = select_matrix(["lib/iris/tests/test_foo.py"], tmp_path)
    assert direct == [matrix_leg("iris", ["lib/iris/tests/test_foo.py"])]

    forced = select_matrix(["lib/levanter/tests/conftest.py"], tmp_path)
    assert forced == [matrix_leg("levanter", [])]


def test_broad_trigger_and_full_matrix() -> None:
    matrix = full_matrix()
    assert {entry["package"] for entry in matrix} == {UV_PACKAGE[scope] for scope in SCOPES}
    assert _leg(UV_PACKAGE["marin"]) == {
        "package": "marin-core",
        "extras": "--extra cpu --extra dedup",
        "test_paths": "tests",
    }

    broad = select_matrix(["uv.lock"], Path("/unused"))
    assert broad == full_matrix()
