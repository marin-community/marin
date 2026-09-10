# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Mode pytest: run pytest with ``pytest-json-report`` and grade the node ids in the report.

This is the SWE-bench shape: ``must_pass`` is FAIL_TO_PASS (the bug the agent had to fix) and
``must_not_break`` is PASS_TO_PASS (what it must not regress). Node ids look like
``tests/test_x.py::test_y``. A missing pytest or json-report plugin is a defect in the task image,
not a failed attempt, so it raises instead of scoring zero.
"""

import json
import tempfile
from pathlib import Path

from tasktrove_verify.modes.run import STDERR_TAIL, check_ids, restore, run_command, run_setup, workdir
from tasktrove_verify.reward import Reward, scored
from tasktrove_verify.spec import PytestSpec

REPORT_NAME = "report.json"
PASS_OUTCOMES = frozenset({"passed", "xpassed"})
FAIL_OUTCOMES = frozenset({"failed", "error"})


def grade(spec: PytestSpec, tests_dir: Path, workspace: Path) -> Reward:
    directory = workdir(spec, workspace)
    restore(spec.restore, tests_dir, directory)
    if spec.setup:
        setup = run_setup(spec.setup, tests_dir, directory, spec.timeout)
        if setup.timed_out or setup.returncode != 0:
            return scored(0.0, reason="setup_failed", stderr=setup.stderr[-STDERR_TAIL:], passed=0, total=0)
    with tempfile.TemporaryDirectory(prefix="tasktrove-pytest-") as scratch:
        report_path = Path(scratch) / REPORT_NAME
        argv = [
            spec.python,
            "-m",
            "pytest",
            "--json-report",
            f"--json-report-file={report_path}",
            "-p",
            "no:cacheprovider",
            "-o",
            "addopts=",
            *spec.args,
            *spec.paths,
        ]
        result = run_command(argv, directory, spec.timeout)
        if result.timed_out:
            return scored(0.0, reason="timeout", passed=0, total=0)
        if not report_path.is_file():
            output = _tail(result.stderr or result.stdout)
            raise RuntimeError(f"pytest wrote no json report (exit {result.returncode}): {output}")
        report = json.loads(report_path.read_text())
    return check_ids(_outcomes(report), spec.must_pass, spec.must_not_break, exit_code=result.returncode)


def _outcomes(report: dict) -> dict[str, bool]:
    """Node id to pass/fail. Skipped and xfailed tests are neither and stay out of the map."""
    outcomes = {}
    for test in report.get("tests", []):
        outcome = test.get("outcome")
        if outcome in PASS_OUTCOMES:
            outcomes[test["nodeid"]] = True
        elif outcome in FAIL_OUTCOMES:
            outcomes[test["nodeid"]] = False
    return outcomes


def _tail(text: str, limit: int = 500) -> str:
    return text.strip()[-limit:]
