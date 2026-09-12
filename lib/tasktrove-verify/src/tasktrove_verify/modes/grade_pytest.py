# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Mode pytest: run pytest with ``pytest-json-report`` and grade the node ids in the report.

This is the SWE-bench shape: ``must_pass`` is FAIL_TO_PASS (the bug the agent had to fix) and
``must_not_break`` is PASS_TO_PASS (what it must not regress). Node ids look like
``tests/test_x.py::test_y``. A missing pytest or json-report plugin is a defect in the task image,
not a failed attempt, so it raises instead of scoring zero.
"""

import json
import os
import tempfile
from dataclasses import replace
from pathlib import Path

from tasktrove_verify.grade import Reward, scored
from tasktrove_verify.modes.run import STDERR_TAIL, check_ids, restore, run_command, run_setup, workdir
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
    outcomes = _outcomes(report, directory)
    reward = check_ids(outcomes, spec.must_pass, spec.must_not_break, exit_code=result.returncode)
    if reward.reward < 1.0:
        output = _tail(result.stdout + result.stderr, STDERR_TAIL)
        reward = replace(reward, detail={**reward.detail, "output": output})
    return reward


def _outcomes(report: dict, workspace: Path) -> dict[str, bool]:
    """Node id to pass/fail. Skipped and xfailed tests are neither and stay out of the map.

    pytest writes node ids relative to its rootdir, which a ``tests/pytest.ini`` moves below the
    workspace (``unit/test_x.py::test_y`` for ``tests/unit/test_x.py``); the spec's ids are relative
    to the workspace, so the ids are rebased before they are compared.
    """
    root = Path(report.get("root", workspace))
    outcomes = {}
    for test in report.get("tests", []):
        outcome = test.get("outcome")
        if outcome in PASS_OUTCOMES:
            outcomes[_rebase(test["nodeid"], root, workspace)] = True
        elif outcome in FAIL_OUTCOMES:
            outcomes[_rebase(test["nodeid"], root, workspace)] = False
    return outcomes


def _rebase(nodeid: str, root: Path, workspace: Path) -> str:
    file, separator, rest = nodeid.partition("::")
    return os.path.relpath(root / file, workspace) + separator + rest


def _tail(text: str, limit: int = 500) -> str:
    return text.strip()[-limit:]
