# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Mode junit: run a JVM or gtest build command and grade the JUnit XML it leaves behind.

``spec.report`` is a glob relative to the workspace (maven's ``**/TEST-*.xml``, gradle's
``**/test-results/**/*.xml``, gtest's ``--gtest_output=xml`` file). A test id is the ``classname``
attribute joined to the ``name`` attribute with a dot, so a maven case
``<testcase classname="com.example.FooTest" name="addsTwo"/>`` has the id
``com.example.FooTest.addsTwo`` and a gtest case ``<testcase classname="FooSuite" name="AddsTwo"/>``
has the id ``FooSuite.AddsTwo``.
"""

from pathlib import Path
from xml.etree import ElementTree as ET

from verifyit.execution.command import run_command
from verifyit.file_ops.restore import restore
from verifyit.grade import InvalidTask, Reward, scored
from verifyit.modes.run import STDERR_TAIL, check_ids, run_setup, split_command, workdir
from verifyit.spec import JunitSpec

FAILURE_TAGS = ("failure", "error")
SKIP_TAG = "skipped"


def grade(spec: JunitSpec, tests_dir: Path, workspace: Path) -> Reward:
    directory = workdir(spec, workspace)
    restore(spec.restore, tests_dir, directory)
    if spec.setup:
        setup = run_setup(spec.setup, tests_dir, directory, spec.timeout)
        if setup.timed_out or setup.returncode != 0:
            return scored(0.0, reason="setup_failed", stderr=setup.stderr[-STDERR_TAIL:], passed=0, total=0)
    report_pattern = Path(spec.report)
    if report_pattern.is_absolute() or ".." in report_pattern.parts:
        raise InvalidTask("JUnit report glob must stay within the workspace")
    previous_reports = [report for report in directory.glob(spec.report) if report.is_file()]
    if any(not report.resolve().is_relative_to(directory.resolve()) for report in previous_reports):
        raise InvalidTask("JUnit report glob resolves outside the workspace")
    for report in previous_reports:
        report.unlink()
    result = run_command(split_command(spec.command), directory, spec.timeout)
    if result.timed_out:
        return scored(0.0, reason="timeout", passed=0, total=0)
    if result.returncode not in (0, 1):
        raise RuntimeError(f"JUnit producer failed with exit {result.returncode}: {result.stderr[-STDERR_TAIL:]}")
    reports = sorted(directory.glob(spec.report))
    if not reports:
        return scored(0.0, reason="no_report", passed=0, total=0, exit_code=result.returncode)
    if any(not report.resolve().is_relative_to(directory.resolve()) for report in reports):
        raise RuntimeError("JUnit producer report resolves outside the workspace")
    outcomes: dict[str, bool] = {}
    for report in reports:
        for test_id, passed in _outcomes(report).items():
            outcomes[test_id] = outcomes.get(test_id, True) and passed
    if result.returncode == 1 and outcomes and all(outcomes.values()):
        raise RuntimeError("JUnit producer failed without reporting a failing test")
    return check_ids(outcomes, spec.must_pass, spec.must_not_break, exit_code=result.returncode, reports=len(reports))


def _outcomes(report: Path) -> dict[str, bool]:
    """Test id to pass/fail for one report file. Skipped cases stay out of the map."""
    outcomes = {}
    root = ET.parse(report).getroot()
    for suite in root.iter():
        if suite.tag not in {"testsuite", "testsuites"}:
            continue
        cases = list(suite.iter("testcase"))
        observed = {
            "tests": len(cases),
            "failures": sum(case.find("failure") is not None for case in cases),
            "errors": sum(case.find("error") is not None for case in cases),
            "skipped": sum(case.find(SKIP_TAG) is not None for case in cases),
        }
        for field, count in observed.items():
            declared = suite.get(field)
            if declared is not None and int(declared) != count:
                raise RuntimeError(f"incomplete JUnit report: declared {field}={declared}, observed {count}")
    for case in root.iter("testcase"):
        name = case.get("name")
        if name is None:
            continue
        classname = case.get("classname", "")
        test_id = f"{classname}.{name}" if classname else name
        if case.find(SKIP_TAG) is not None:
            continue
        passed = all(case.find(tag) is None for tag in FAILURE_TAGS)
        outcomes[test_id] = outcomes.get(test_id, True) and passed
    return outcomes
