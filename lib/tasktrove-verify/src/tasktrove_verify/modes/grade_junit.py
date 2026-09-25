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

from tasktrove_verify.grade import Reward, scored
from tasktrove_verify.modes.run import STDERR_TAIL, check_ids, restore, run_command, run_setup, split_command, workdir
from tasktrove_verify.spec import JunitSpec

FAILURE_TAGS = ("failure", "error")
SKIP_TAG = "skipped"


def grade(spec: JunitSpec, tests_dir: Path, workspace: Path) -> Reward:
    directory = workdir(spec, workspace)
    restore(spec.restore, tests_dir, directory)
    if spec.setup:
        setup = run_setup(spec.setup, tests_dir, directory, spec.timeout)
        if setup.timed_out or setup.returncode != 0:
            return scored(0.0, reason="setup_failed", stderr=setup.stderr[-STDERR_TAIL:], passed=0, total=0)
    result = run_command(split_command(spec.command), directory, spec.timeout)
    if result.timed_out:
        return scored(0.0, reason="timeout", passed=0, total=0)
    reports = sorted(directory.glob(spec.report))
    if not reports:
        return scored(0.0, reason="no_report", passed=0, total=0, exit_code=result.returncode)
    outcomes: dict[str, bool] = {}
    for report in reports:
        outcomes.update(_outcomes(report))
    return check_ids(outcomes, spec.must_pass, spec.must_not_break, exit_code=result.returncode, reports=len(reports))


def _outcomes(report: Path) -> dict[str, bool]:
    """Test id to pass/fail for one report file. Skipped cases stay out of the map."""
    outcomes = {}
    for case in ET.parse(report).getroot().iter("testcase"):
        name = case.get("name")
        if name is None:
            continue
        classname = case.get("classname", "")
        test_id = f"{classname}.{name}" if classname else name
        if case.find(SKIP_TAG) is not None:
            continue
        outcomes[test_id] = all(case.find(tag) is None for tag in FAILURE_TAGS)
    return outcomes
