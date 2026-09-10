# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Subprocess, restore, and test-id helpers shared by the execution modes.

Execution modes run a real toolchain inside the task image, so they only use the standard library.
A command runs in its own process group and is killed as a group on timeout, which stops the
grandchildren a build tool leaves behind.
"""

import os
import shlex
import shutil
import signal
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from tasktrove_verify.reward import InvalidTask, Reward, scored
from tasktrove_verify.spec import DEFAULT_WORKSPACE, GotestSpec, JunitSpec, PytestSpec, StdioSpec

KILL_GRACE = 5.0


@dataclass(frozen=True)
class Completed:
    """One finished command. ``stdout``/``stderr`` are decoded with replacement, never raising."""

    returncode: int
    stdout: str
    stderr: str
    timed_out: bool


def workdir(spec: StdioSpec | PytestSpec | JunitSpec | GotestSpec, workspace: Path) -> Path:
    """The directory a command runs in.

    A spec that left ``workspace`` at the container default defers to the caller's workspace, so a
    local gate can grade a task in a temporary directory.
    """
    declared = Path(spec.workspace)
    if spec.workspace == DEFAULT_WORKSPACE and workspace != declared:
        return workspace
    return declared


def run_command(
    argv: Sequence[str],
    cwd: Path,
    timeout: float,
    stdin_text: str | None = None,
    env: Mapping[str, str] | None = None,
) -> Completed:
    """Run ``argv`` in ``cwd``, capturing output. A timeout kills the whole process group.

    Raises ``FileNotFoundError`` when the program does not exist; callers decide whether that is the
    agent's fault or the image's.
    """
    proc = subprocess.Popen(
        list(argv),
        cwd=str(cwd),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        errors="replace",
        start_new_session=True,
        env={**os.environ, **env} if env else None,
    )
    with proc:
        try:
            stdout, stderr = proc.communicate(stdin_text or "", timeout=timeout)
            return Completed(proc.returncode, stdout, stderr, timed_out=False)
        except subprocess.TimeoutExpired:
            _kill_group(proc)
            try:
                stdout, stderr = proc.communicate(timeout=KILL_GRACE)
            except subprocess.TimeoutExpired:
                stdout, stderr = "", ""
            return Completed(proc.returncode, stdout, stderr, timed_out=True)


def _kill_group(proc: subprocess.Popen) -> None:
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        proc.kill()


def run_setup(command: str, tests_dir: Path, workspace: Path, timeout: float) -> Completed:
    """Run a spec's ``setup`` shell command in the workspace with the tests directory in its environment."""
    env = {"TASKTROVE_TESTS_DIR": str(tests_dir), "TASKTROVE_WORKSPACE": str(workspace)}
    return run_command(["bash", "-lc", command], workspace, timeout, env=env)


def split_command(command: str) -> list[str]:
    """Split a spec's command string into argv. Commands are programs, not shell scripts."""
    argv = shlex.split(command)
    if not argv:
        raise InvalidTask("command is empty")
    return argv


def restore(entries: Sequence[str], tests_dir: Path, workspace: Path) -> None:
    """Copy each entry from ``tests_dir`` over the workspace, undoing agent edits to the tests."""
    for entry in entries:
        source = tests_dir / entry
        destination = workspace / entry
        if source.is_dir():
            shutil.copytree(source, destination, dirs_exist_ok=True)
        elif source.is_file():
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
        else:
            raise InvalidTask(f"restore entry {entry!r} is not in the tests directory")


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
