# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import sys
from pathlib import Path

import pytest
from tasktrove_verify.grade import InvalidTask, Status, run
from tasktrove_verify.modes import grade_pytest
from tasktrove_verify.spec import PytestSpec, render_spec
from tasktrove_verify.spec import TestIdMatching as IdMatching

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
    reward = grade_pytest.grade(_spec(must_pass=(REGRESSION,), must_not_break=(PASSING,)), tmp_path, workspace)
    assert reward.reward == 1.0
    assert reward.detail["passed"] == 2


def test_pytest_required_id_failing_scores_zero_and_names_it(tmp_path):
    workspace = _project(tmp_path, BROKEN)
    reward = grade_pytest.grade(_spec(must_pass=(REGRESSION,), must_not_break=(PASSING,)), tmp_path, workspace)
    assert reward.reward == 0.0
    assert reward.detail["first_failure"] == REGRESSION


def test_pytest_ids_are_rebased_when_an_ini_moves_the_rootdir(tmp_path):
    """A ``tests/pytest.ini`` makes pytest report ``test_calc.py::...`` instead of
    ``tests/test_calc.py::...``; the spec's workspace-relative ids must still match."""
    workspace = _project(tmp_path, FIXED)
    (workspace / "tests" / "pytest.ini").write_text("[pytest]\n")
    reward = grade_pytest.grade(_spec(must_pass=(REGRESSION,), must_not_break=(PASSING,)), tmp_path, workspace)
    assert reward.reward == 1.0
    assert reward.detail["passed"] == 2


def test_pytest_failure_detail_carries_the_output_tail(tmp_path):
    workspace = _project(tmp_path, BROKEN)
    reward = grade_pytest.grade(_spec(must_pass=(REGRESSION,)), tmp_path, workspace)
    assert reward.reward == 0.0
    assert "test_add_negative" in reward.detail["output"]
    passing = grade_pytest.grade(_spec(must_pass=(PASSING,)), tmp_path, workspace)
    assert passing.reward == 1.0 and "output" not in passing.detail


def test_pytest_id_absent_from_report_counts_as_failed(tmp_path):
    workspace = _project(tmp_path, FIXED)
    missing = "tests/test_calc.py::test_never_written"
    reward = grade_pytest.grade(_spec(must_pass=(missing,)), tmp_path, workspace)
    assert reward.reward == 0.0
    assert reward.detail["first_failure"] == missing


def test_pytest_without_id_lists_requires_whole_suite_to_pass(tmp_path):
    assert grade_pytest.grade(_spec(), tmp_path, _project(tmp_path, FIXED)).reward == 1.0


def test_pytest_without_id_lists_fails_on_any_failure(tmp_path):
    reward = grade_pytest.grade(_spec(), tmp_path, _project(tmp_path, BROKEN))
    assert reward.reward == 0.0
    assert reward.detail["first_failure"] == REGRESSION


def test_pytest_empty_workspace_scores_zero_with_no_tests(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    reward = grade_pytest.grade(_spec(), tmp_path, workspace)
    assert reward.reward == 0.0
    assert reward.detail["reason"] == "no_tests"


def test_pytest_paths_limit_the_run(tmp_path):
    workspace = _project(tmp_path, FIXED)
    (workspace / "tests" / "test_other.py").write_text("def test_other():\n    assert False\n")
    assert grade_pytest.grade(_spec(paths=("tests/test_calc.py",)), tmp_path, workspace).reward == 1.0
    assert grade_pytest.grade(_spec(), tmp_path, workspace).reward == 0.0


def test_pytest_restore_undoes_agent_edits_to_the_tests(tmp_path):
    workspace = _project(tmp_path, BROKEN, tests=TAMPERED_TESTS)
    tests_dir = tmp_path / "tests_dir"
    (tests_dir / "tests").mkdir(parents=True)
    (tests_dir / "tests" / "test_calc.py").write_text(REAL_TESTS)
    spec = _spec(must_pass=(REGRESSION,), restore=("tests/test_calc.py",))
    assert grade_pytest.grade(spec, tests_dir, workspace).reward == 0.0
    assert (workspace / "tests" / "test_calc.py").read_text() == REAL_TESTS


def test_pytest_skipped_test_does_not_block_a_clean_suite(tmp_path):
    workspace = _project(tmp_path, FIXED)
    reward = grade_pytest.grade(_spec(), tmp_path, workspace)
    assert (reward.reward, reward.detail["total"]) == (1.0, 2)


def test_pytest_timeout_scores_zero_with_reason(tmp_path):
    workspace = _project(tmp_path, FIXED)
    (workspace / "tests" / "test_slow.py").write_text("import time\n\n\ndef test_slow():\n    time.sleep(30)\n")
    reward = grade_pytest.grade(_spec(timeout=1.0), tmp_path, workspace)
    assert reward.reward == 0.0
    assert reward.detail["reason"] == "timeout"


def test_pytest_missing_json_report_plugin_is_an_infra_error(tmp_path):
    workspace = _project(tmp_path, FIXED)
    stub = tmp_path / "python-without-plugin"
    stub.write_text('#!/bin/sh\necho "error: unrecognized arguments: --json-report" >&2\nexit 4\n')
    stub.chmod(0o755)
    with pytest.raises(RuntimeError, match="json report"):
        grade_pytest.grade(_spec(python=str(stub)), tmp_path, workspace)


def test_setup_runs_in_the_workspace_before_the_tests(tmp_path):
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    workspace = tmp_path / "app"
    workspace.mkdir()
    (workspace / "test_marker.py").write_text(
        "import pathlib\n\ndef test_marker():\n    assert pathlib.Path('made-by-setup').read_text() == 'tests-dir\\n'\n"
    )
    spec = PytestSpec(paths=("test_marker.py",), setup='echo tests-dir > made-by-setup; test -d "$TASKTROVE_TESTS_DIR"')
    assert grade_pytest.grade(spec, tests_dir, workspace).reward == 1.0
    failing = PytestSpec(paths=("test_marker.py",), setup="exit 3")
    reward = grade_pytest.grade(failing, tests_dir, workspace)
    assert reward.reward == 0.0 and reward.detail["reason"] == "setup_failed"


def test_task_owned_setup_failure_can_be_unscored(tmp_path):
    workspace = _project(tmp_path, FIXED)
    spec = _spec(paths=("tests/test_calc.py",), setup="exit 3", setup_failure_is_infra=True)
    with pytest.raises(RuntimeError, match="pytest setup exited 3"):
        grade_pytest.grade(spec, tmp_path, workspace)
    timed_out = _spec(paths=("tests/test_calc.py",), setup="sleep 2", timeout=0.1, setup_failure_is_infra=True)
    with pytest.raises(RuntimeError, match="pytest setup timed out"):
        grade_pytest.grade(timed_out, tmp_path, workspace)


def test_pytest_report_repeated_pass_does_not_erase_required_failure(tmp_path):
    workspace = _project(tmp_path, BROKEN)
    # Retry/reporting plugins can emit several observations of one node ID.
    (workspace / "conftest.py").write_text(
        "def pytest_json_modifyreport(json_report):\n"
        '    failed = next(test for test in json_report["tests"] if test["outcome"] == "failed")\n'
        '    json_report["tests"].append({**failed, "outcome": "passed"})\n'
        '    json_report["summary"]["total"] += 1\n'
        '    json_report["summary"]["passed"] += 1\n'
    )
    reward = grade_pytest.grade(_spec(must_not_break=(REGRESSION,)), tmp_path, workspace)
    assert (reward.reward, reward.detail["first_failure"]) == (0.0, REGRESSION)
    assert grade_pytest.grade(_spec(must_pass=(PASSING,)), tmp_path, workspace).reward == 1.0


def test_pytest_failclosed_interrupted_runner_cannot_report_success(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "test_candidate.py").write_text("def test_required():\n assert True\n")
    (workspace / "conftest.py").write_text("def pytest_sessionfinish(session,exitstatus):\n session.exitstatus=2\n")
    tests = tmp_path / "tests"
    tests.mkdir()
    (tests / "verifier.toml").write_text('mode="pytest"\nmust_pass=["test_candidate.py::test_required"]\n')
    reward = run(tests / "verifier.toml", workspace)
    assert reward.status == Status.INFRA_ERROR
    assert reward.reward == 0


@pytest.mark.parametrize("batch_size", [0, 1])
def test_pytest_failclosed_collection_error_cannot_be_hidden_by_passing_required_test(tmp_path, batch_size):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "test_candidate.py").write_text("def test_required():\n assert True\n")
    (workspace / "test_bad.py").write_text("def malformed(\n")
    tests = tmp_path / "tests"
    tests.mkdir()
    (tests / "verifier.toml").write_text(
        'mode="pytest"\nargs=["--continue-on-collection-errors"]\nmust_pass=["test_candidate.py::test_required"]\n'
        f'paths=["test_candidate.py", "test_bad.py"]\nbatch_size={batch_size}\n'
    )
    reward = run(tests / "verifier.toml", workspace)
    assert reward.status == Status.INFRA_ERROR
    assert reward.reward == 0


def test_pytest_failclosed_summary_cannot_hide_a_missing_failed_record(tmp_path):
    workspace = _project(tmp_path, BROKEN)
    (workspace / "conftest.py").write_text(
        "def pytest_json_modifyreport(json_report):\n"
        ' json_report["tests"] = [test for test in json_report["tests"] if test["outcome"] == "passed"]\n'
    )
    tests = tmp_path / "tests"
    tests.mkdir()
    (tests / "verifier.toml").write_text('mode="pytest"\nmust_pass=["tests/test_calc.py::test_add_positive"]\n')
    reward = run(tests / "verifier.toml", workspace)
    assert reward.status == Status.INFRA_ERROR
    assert reward.reward == 0


def test_batches_require_late_protected_success(tmp_path):
    workspace = _project(tmp_path, BROKEN)
    spec = _spec(paths=(PASSING, REGRESSION), batch_size=1, must_pass=(PASSING,), must_not_break=(REGRESSION,))
    failed = grade_pytest.grade(spec, tmp_path, workspace)
    assert failed.reward == 0.0
    assert failed.detail["first_failure"] == REGRESSION
    (workspace / "calc.py").write_text(FIXED)
    assert grade_pytest.grade(spec, tmp_path, workspace).reward == 1.0


def test_duplicate_test_failure_survives_later_passing_batch(tmp_path):
    tests = """from pathlib import Path

def test_repeat():
    marker = Path("already_ran")
    existed = marker.exists()
    marker.touch()
    assert existed
"""
    workspace = _project(tmp_path, FIXED, tests)
    test_id = "tests/test_calc.py::test_repeat"
    verdict = grade_pytest.grade(
        _spec(paths=(test_id, test_id), batch_size=1, must_pass=(test_id,)), tmp_path, workspace
    )
    assert verdict.reward == 0.0
    assert verdict.detail["first_failure"] == test_id


def test_setup_and_batches_share_one_deadline(tmp_path):
    tests = """import time
from pathlib import Path

def test_first():
    Path("first_finished").touch()

def test_finish():
    time.sleep(3)
    Path("finished").touch()
"""
    workspace = _project(tmp_path, FIXED, tests)
    spec = _spec(
        setup=f"{sys.executable} -c 'import time; time.sleep(1)'",
        paths=("tests/test_calc.py::test_first", "tests/test_calc.py::test_finish"),
        batch_size=1,
        timeout=3.0,
    )
    verdict = grade_pytest.grade(spec, tmp_path, workspace)
    assert verdict.reward == 0.0
    assert verdict.detail["reason"] == "timeout"
    assert (workspace / "first_finished").exists()
    assert not (workspace / "finished").exists()


@pytest.mark.parametrize(
    "required,expected",
    [("test_case[alphabet", 1.0), ("test_case[alph", 0.0), ("test_case", 0.0), ("test_case[💩]", 1.0)],
)
def test_unique_partial_ids_do_not_hide_skipped_ambiguity(tmp_path, required, expected):
    tests = """import pytest

@pytest.mark.parametrize("value", [pytest.param(1, marks=pytest.mark.skip), 2, 3], ids=["alpha", "alphabet", "💩"])
def test_case(value):
    assert value > 0
"""
    workspace = _project(tmp_path, FIXED, tests)
    spec = _spec(must_pass=("tests/test_calc.py::" + required,), id_matching=IdMatching.UNIQUE_PREFIX)
    verdict = grade_pytest.grade(spec, tmp_path, workspace)
    assert verdict.reward == expected


def test_passing_batch_cannot_hide_later_skipped_same_test(tmp_path):
    tests = """from pathlib import Path
import pytest

def test_repeat():
    marker = Path("already_ran")
    if marker.exists():
        pytest.skip("no longer executed")
    marker.touch()
"""
    workspace = _project(tmp_path, FIXED, tests)
    test_id = "tests/test_calc.py::test_repeat"
    verdict = grade_pytest.grade(
        _spec(paths=(test_id, test_id), batch_size=1, must_pass=(test_id,)), tmp_path, workspace
    )
    assert verdict.reward == 0.0
    assert verdict.detail["first_failure"] == test_id


@pytest.mark.parametrize(
    "required",
    [
        ("test_case[x", "test_case[xy"),
        ("test_case[xyz]", "test_case[xy"),
    ],
)
def test_distinct_required_ids_cannot_share_one_reported_case(tmp_path, required):
    workspace = _project(
        tmp_path,
        FIXED,
        'import pytest\n@pytest.mark.parametrize("value", [1], ids=["xyz"])\n'
        "def test_case(value):\n    assert value == 1\n",
    )
    spec = _spec(
        must_pass=tuple("tests/test_calc.py::" + value for value in required),
        id_matching=IdMatching.UNIQUE_PREFIX,
    )
    assert grade_pytest.grade(spec, tmp_path, workspace).reward == 0.0


@pytest.mark.parametrize("timeout", [float("nan"), float("inf"), -1.0, 0.0, True, 10**400])
def test_invalid_timeout_rejects_before_restore_or_setup(tmp_path, timeout):
    workspace = _project(tmp_path, FIXED)
    trusted = tmp_path / "trusted"
    (trusted / "tests").mkdir(parents=True)
    (trusted / "tests/test_calc.py").write_text(TAMPERED_TESTS)
    spec = _spec(timeout=timeout, restore=("tests/test_calc.py",), setup="touch setup-ran")
    with pytest.raises(InvalidTask):
        grade_pytest.grade(spec, trusted, workspace)
    assert (workspace / "tests/test_calc.py").read_text() == REAL_TESTS
    assert not (workspace / "setup-ran").exists()
    config = trusted / "verifier.toml"
    config.write_text(render_spec(spec))
    result = run(config, workspace)
    assert result.status is Status.INVALID_TASK
    assert result.reward == 0.0
    assert (workspace / "tests/test_calc.py").read_text() == REAL_TESTS
    assert not (workspace / "setup-ran").exists()
