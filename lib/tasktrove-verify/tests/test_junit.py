# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

from tasktrove_verify.modes import grade_junit
from tasktrove_verify.spec import JunitSpec

MAVEN_REPORT = """<?xml version="1.0" encoding="UTF-8"?>
<testsuite name="com.example.CalcTest" tests="4" failures="1" errors="0" skipped="1">
  <testcase classname="com.example.CalcTest" name="addsTwo" time="0.01"/>
  <testcase classname="com.example.CalcTest" name="addsNegatives" time="0.01">
    <failure message="expected:&lt;-5&gt; but was:&lt;5&gt;" type="java.lang.AssertionError"/>
  </testcase>
  <testcase classname="com.example.CalcTest" name="dividesByZero" time="0.01">
    <error message="boom" type="java.lang.RuntimeException"/>
  </testcase>
  <testcase classname="com.example.CalcTest" name="pending" time="0.0">
    <skipped/>
  </testcase>
</testsuite>
"""

CLEAN_REPORT = """<?xml version="1.0" encoding="UTF-8"?>
<testsuite name="com.example.CalcTest" tests="1" failures="0">
  <testcase classname="com.example.CalcTest" name="addsTwo" time="0.01"/>
</testsuite>
"""

ADDS_TWO = "com.example.CalcTest.addsTwo"
ADDS_NEGATIVES = "com.example.CalcTest.addsNegatives"


def _workspace(tmp_path: Path, report: str, name: str = "target/surefire-reports/TEST-com.example.CalcTest.xml") -> Path:
    workspace = tmp_path / "workspace"
    path = workspace / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report)
    return workspace


def test_junit_required_ids_pass_scores_one(tmp_path):
    workspace = _workspace(tmp_path, MAVEN_REPORT)
    spec = JunitSpec(command="true", must_pass=(ADDS_TWO,))
    reward = grade_junit.grade(spec, tmp_path, workspace)
    assert reward.reward == 1.0
    assert reward.detail["passed"] == 1


def test_junit_failure_element_fails_the_required_id(tmp_path):
    workspace = _workspace(tmp_path, MAVEN_REPORT)
    spec = JunitSpec(command="true", must_pass=(ADDS_TWO,), must_not_break=(ADDS_NEGATIVES,))
    reward = grade_junit.grade(spec, tmp_path, workspace)
    assert reward.reward == 0.0
    assert reward.detail["first_failure"] == ADDS_NEGATIVES


def test_junit_missing_required_id_fails(tmp_path):
    workspace = _workspace(tmp_path, CLEAN_REPORT)
    missing = "com.example.CalcTest.neverRan"
    reward = grade_junit.grade(JunitSpec(command="true", must_pass=(missing,)), tmp_path, workspace)
    assert reward.reward == 0.0
    assert reward.detail["first_failure"] == missing


def test_junit_without_id_lists_requires_every_case_to_pass(tmp_path):
    assert grade_junit.grade(JunitSpec(command="true"), tmp_path, _workspace(tmp_path, MAVEN_REPORT)).reward == 0.0
    assert grade_junit.grade(JunitSpec(command="true"), tmp_path, _workspace(tmp_path, CLEAN_REPORT)).reward == 1.0


def test_junit_skipped_case_is_neither_passed_nor_failed(tmp_path):
    workspace = _workspace(tmp_path, CLEAN_REPORT)
    (workspace / "target/surefire-reports/TEST-com.example.SkipTest.xml").write_text(
        '<testsuite name="s"><testcase classname="com.example.SkipTest" name="pending"><skipped/></testcase></testsuite>'
    )
    reward = grade_junit.grade(JunitSpec(command="true"), tmp_path, workspace)
    assert (reward.reward, reward.detail["total"]) == (1.0, 1)


def test_junit_no_report_files_scores_zero_with_reason(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    reward = grade_junit.grade(JunitSpec(command="true"), tmp_path, workspace)
    assert reward.reward == 0.0
    assert reward.detail["reason"] == "no_report"


def test_junit_restored_report_replaces_the_workspace_copy(tmp_path):
    workspace = _workspace(tmp_path, MAVEN_REPORT, name="results.xml")
    tests_dir = tmp_path / "tests_dir"
    tests_dir.mkdir()
    (tests_dir / "results.xml").write_text(CLEAN_REPORT)
    spec = JunitSpec(command="true", report="results.xml", restore=("results.xml",))
    assert grade_junit.grade(spec, tests_dir, workspace).reward == 1.0


def test_junit_command_timeout_scores_zero_with_reason(tmp_path):
    workspace = _workspace(tmp_path, CLEAN_REPORT)
    reward = grade_junit.grade(JunitSpec(command="sleep 30", timeout=0.5), tmp_path, workspace)
    assert reward.reward == 0.0
    assert reward.detail["reason"] == "timeout"
