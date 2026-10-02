# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Mode-specific execution setup and required-test grading helpers."""

import shlex
from collections.abc import Mapping, Sequence
from pathlib import Path

from tasktrove_verify.execution.command import Completed, run_command
from tasktrove_verify.grade import InvalidTask, Reward, scored
from tasktrove_verify.spec import DEFAULT_WORKSPACE, GotestSpec, JunitSpec, PytestSpec, StdioSpec

STDERR_TAIL = 2000
"""Characters of stderr kept in a reward detail."""


def workdir(spec: StdioSpec | PytestSpec | JunitSpec | GotestSpec, workspace: Path) -> Path:
    """The directory a command runs in.

    A spec that left ``workspace`` at the container default defers to the caller's workspace, so a
    local gate can grade a task in a temporary directory.
    """
    declared = Path(spec.workspace)
    if spec.workspace == DEFAULT_WORKSPACE and workspace != declared:
        return workspace
    return declared


def run_setup(command: str, tests_dir: Path, workspace: Path, timeout: float) -> Completed:
    """Run a spec's ``setup`` shell command in the workspace with the tests directory in its environment."""
    env = {"TASKTROVE_TESTS_DIR": str(tests_dir), "TASKTROVE_WORKSPACE": str(workspace)}
    return run_command(["bash", "-lc", command], workspace, timeout, env=env)


def split_command(command: str) -> list[str]:
    """Parse an executable command into argv without shell interpretation."""
    argv = shlex.split(command)
    if not argv:
        raise InvalidTask("command is empty")
    return argv


def check_ids(
    outcomes: Mapping[str, bool],
    must_pass: Sequence[str],
    must_not_break: Sequence[str],
    **detail: object,
) -> Reward:
    """Score a run of a test framework.

    ``outcomes`` maps a test id to whether it passed and omits tests that neither passed nor failed,
    such as skipped ones. When ``must_pass`` or ``must_not_break`` names ids, every one of them must
    be present and passing; an id the run never reported counts as failed. When both lists are empty
    the whole suite must pass and at least one test must have run.
    """
    required = [*must_pass, *must_not_break]
    if required:
        failures = [test_id for test_id in required if not outcomes.get(test_id, False)]
        return _reward(len(required) - len(failures), len(required), failures, detail)
    if not outcomes:
        return scored(0.0, reason="no_tests", passed=0, total=0, **detail)
    failures = [test_id for test_id, passed in outcomes.items() if not passed]
    return _reward(len(outcomes) - len(failures), len(outcomes), failures, detail)


def _reward(passed: int, total: int, failures: Sequence[str], detail: dict) -> Reward:
    if failures:
        return scored(0.0, passed=passed, total=total, first_failure=failures[0], **detail)
    return scored(1.0, passed=passed, total=total, **detail)
