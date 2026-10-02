# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Mode gotest: run ``go test -json`` and grade the event stream.

``go test -json`` writes one JSON object per line. Test identities join the package import
path and test name, such as ``example.com/m/pkg.TestAdd``. Package terminal events establish
completion; pending test/package events or package failures without corresponding test failures
cannot produce positive credit. A build failure with no test outcomes scores zero.
"""

import json
from pathlib import Path

from tasktrove_verify.execution.command import run_command
from tasktrove_verify.file_ops.restore import restore
from tasktrove_verify.grade import Reward, scored
from tasktrove_verify.modes.run import STDERR_TAIL, check_ids, run_setup, workdir
from tasktrove_verify.spec import GotestSpec

GO = "go"
ACTION_OUTCOMES = {"pass": True, "fail": False}


def grade(spec: GotestSpec, tests_dir: Path, workspace: Path) -> Reward:
    directory = workdir(spec, workspace)
    restore(spec.restore, tests_dir, directory)
    if spec.setup:
        setup = run_setup(spec.setup, tests_dir, directory, spec.timeout)
        if setup.timed_out or setup.returncode != 0:
            return scored(0.0, reason="setup_failed", stderr=setup.stderr[-STDERR_TAIL:], passed=0, total=0)
    result = run_command([GO, "test", "-json", *spec.args, *spec.packages], directory, spec.timeout)
    if result.timed_out:
        return scored(0.0, reason="timeout", passed=0, total=0)
    if result.returncode not in (0, 1):
        raise RuntimeError(f"go test producer failed with exit {result.returncode}: {result.stderr[-STDERR_TAIL:]}")
    outcomes, complete = _parse_run(result.stdout)
    if not complete and outcomes:
        raise RuntimeError("go test stream has incomplete or inconsistent test/package events")
    if result.returncode == 1 and outcomes and all(outcomes.values()):
        raise RuntimeError("go test producer failed without reporting a failing test")
    return check_ids(outcomes, spec.must_pass, spec.must_not_break, exit_code=result.returncode)


def parse_events(stream: str) -> dict[str, bool]:
    """Test id to pass/fail from a ``go test -json`` stream. Skipped tests stay out of the map."""
    return _parse_run(stream)[0]


def _parse_run(stream: str) -> tuple[dict[str, bool], bool]:
    outcomes: dict[str, bool] = {}
    packages: set[str] = set()
    finished_packages: set[str] = set()
    failed_packages: set[str] = set()
    failed_test_packages: set[str] = set()
    running: set[tuple[str, str]] = set()
    for line in stream.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        event = json.loads(line)
        package = event.get("Package", "")
        action = event.get("Action")
        test = event.get("Test")
        packages.add(package)
        if test is None and action in {"pass", "fail", "skip"}:
            finished_packages.add(package)
            if action == "fail":
                failed_packages.add(package)
        if test is not None:
            if action == "run":
                running.add((package, test))
            elif action in {"pass", "fail", "skip"}:
                running.discard((package, test))
        if test is not None and action == "fail":
            failed_test_packages.add(package)
        passed = ACTION_OUTCOMES.get(event.get("Action"))
        if test is None or passed is None:
            continue
        test_id = f"{event.get('Package', '')}.{test}"
        outcomes[test_id] = outcomes.get(test_id, True) and passed
    return outcomes, packages <= finished_packages and not running and failed_packages <= failed_test_packages
