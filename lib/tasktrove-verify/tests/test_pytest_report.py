# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import sys
from pathlib import Path

import pytest
from tasktrove_verify.modes import pytest_report
from tasktrove_verify.spec import PytestSpec

REAL_TESTS = """
from calc import add


def test_add_positive():
    assert add(2, 3) == 5


def test_add_negative():
    assert add(-2, -3) == -5


def test_skipped():
    import pytest

    pytest.skip("not applicable here")
"""

TAMPERED_TESTS = """
def test_add_positive():
    assert True


def test_add_negative():
    assert True
"""

FIXED = "def add(a, b):\n    return a + b\n"
BROKEN = "def add(a, b):\n    return abs(a) + abs(b)\n"

PASSING = "tests/test_calc.py::test_add_positive"
REGRESSION = "tests/test_calc.py::test_add_negative"


def _project(tmp_path: Path, implementation: str, tests: str = REAL_TESTS) -> Path:
    workspace = tmp_path / "workspace"
    (workspace / "tests").mkdir(parents=True)
    (workspace / "calc.py").write_text(implementation)
    (workspace / "tests" / "test_calc.py").write_text(tests)
    return workspace


def _spec(**overrides) -> PytestSpec:
    return PytestSpec(**{"python": sys.executable, "timeout": 120.0, **overrides})


def test_pytest_required_ids_all_pass_scores_one(tmp_path):
    workspace = _project(tmp_path, FIXED)
    reward = pytest_report.grade(_spec(must_pass=(REGRESSION,), must_not_break=(PASSING,)), tmp_path, workspace)
    assert reward.reward == 1.0
    assert reward.detail["passed"] == 2


def test_pytest_required_id_failing_scores_zero_and_names_it(tmp_path):
    workspace = _project(tmp_path, BROKEN)
    reward = pytest_report.grade(_spec(must_pass=(REGRESSION,), must_not_break=(PASSING,)), tmp_path, workspace)
    assert reward.reward == 0.0
    assert reward.detail["first_failure"] == REGRESSION


def test_pytest_id_absent_from_report_counts_as_failed(tmp_path):
    workspace = _project(tmp_path, FIXED)
    missing = "tests/test_calc.py::test_never_written"
    reward = pytest_report.grade(_spec(must_pass=(missing,)), tmp_path, workspace)
    assert reward.reward == 0.0
    assert reward.detail["first_failure"] == missing


def test_pytest_without_id_lists_requires_whole_suite_to_pass(tmp_path):
    assert pytest_report.grade(_spec(), tmp_path, _project(tmp_path, FIXED)).reward == 1.0


def test_pytest_without_id_lists_fails_on_any_failure(tmp_path):
    reward = pytest_report.grade(_spec(), tmp_path, _project(tmp_path, BROKEN))
    assert reward.reward == 0.0
    assert reward.detail["first_failure"] == REGRESSION


def test_pytest_empty_workspace_scores_zero_with_no_tests(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    reward = pytest_report.grade(_spec(), tmp_path, workspace)
    assert reward.reward == 0.0
    assert reward.detail["reason"] == "no_tests"


def test_pytest_paths_limit_the_run(tmp_path):
    workspace = _project(tmp_path, FIXED)
    (workspace / "tests" / "test_other.py").write_text("def test_other():\n    assert False\n")
    assert pytest_report.grade(_spec(paths=("tests/test_calc.py",)), tmp_path, workspace).reward == 1.0
    assert pytest_report.grade(_spec(), tmp_path, workspace).reward == 0.0


def test_pytest_restore_undoes_agent_edits_to_the_tests(tmp_path):
    workspace = _project(tmp_path, BROKEN, tests=TAMPERED_TESTS)
    tests_dir = tmp_path / "tests_dir"
    (tests_dir / "tests").mkdir(parents=True)
    (tests_dir / "tests" / "test_calc.py").write_text(REAL_TESTS)
    spec = _spec(must_pass=(REGRESSION,), restore=("tests/test_calc.py",))
    assert pytest_report.grade(spec, tests_dir, workspace).reward == 0.0
    assert (workspace / "tests" / "test_calc.py").read_text() == REAL_TESTS


def test_pytest_skipped_test_does_not_block_a_clean_suite(tmp_path):
    workspace = _project(tmp_path, FIXED)
    reward = pytest_report.grade(_spec(), tmp_path, workspace)
    assert reward.detail["total"] == 2


def test_pytest_timeout_scores_zero_with_reason(tmp_path):
    workspace = _project(tmp_path, FIXED)
    (workspace / "tests" / "test_slow.py").write_text("import time\n\n\ndef test_slow():\n    time.sleep(30)\n")
    reward = pytest_report.grade(_spec(timeout=1.0), tmp_path, workspace)
    assert reward.reward == 0.0
    assert reward.detail["reason"] == "timeout"


def test_pytest_missing_json_report_plugin_is_an_infra_error(tmp_path):
    workspace = _project(tmp_path, FIXED)
    stub = tmp_path / "python-without-plugin"
    stub.write_text('#!/bin/sh\necho "error: unrecognized arguments: --json-report" >&2\nexit 4\n')
    stub.chmod(0o755)
    with pytest.raises(RuntimeError, match="json report"):
        pytest_report.grade(_spec(python=str(stub)), tmp_path, workspace)


def test_setup_runs_in_the_workspace_before_the_tests(tmp_path):
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    workspace = tmp_path / "app"
    workspace.mkdir()
    (workspace / "test_marker.py").write_text(
        "import pathlib\n\ndef test_marker():\n    assert pathlib.Path('made-by-setup').read_text() == 'tests-dir\\n'\n"
    )
    spec = PytestSpec(paths=("test_marker.py",), setup='echo tests-dir > made-by-setup; test -d "$TASKTROVE_TESTS_DIR"')
    assert pytest_report.grade(spec, tests_dir, workspace).reward == 1.0
    failing = PytestSpec(paths=("test_marker.py",), setup="exit 3")
    reward = pytest_report.grade(failing, tests_dir, workspace)
    assert reward.reward == 0.0 and reward.detail["reason"] == "setup_failed"
