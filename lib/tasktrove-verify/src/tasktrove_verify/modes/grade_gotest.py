# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Mode gotest: run ``go test -json`` and grade the event stream.

``go test -json`` writes one JSON object per line. Events that carry a ``Test`` field report a
single test or subtest; the id is the package import path joined to the test name with a dot, as in
``example.com/m/pkg.TestAdd`` or ``example.com/m/pkg.TestAdd/negative``. Events without a ``Test``
field are package-level and are ignored: a build failure produces no test events at all, which
grades as a run with no tests.
"""

import json
from pathlib import Path

from tasktrove_verify.modes.run import STDERR_TAIL, check_ids, restore, run_command, run_setup, workdir
from tasktrove_verify.grade import Reward, scored
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
    return check_ids(parse_events(result.stdout), spec.must_pass, spec.must_not_break, exit_code=result.returncode)


def parse_events(stream: str) -> dict[str, bool]:
    """Test id to pass/fail from a ``go test -json`` stream. Skipped tests stay out of the map."""
    outcomes: dict[str, bool] = {}
    for line in stream.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        event = json.loads(line)
        test = event.get("Test")
        passed = ACTION_OUTCOMES.get(event.get("Action"))
        if test is None or passed is None:
            continue
        outcomes[f"{event.get('Package', '')}.{test}"] = passed
    return outcomes
