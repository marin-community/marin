# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Command execution and process-group cleanup."""

import os
import signal
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

KILL_GRACE = 5.0


@dataclass(frozen=True)
class Completed:
    """One finished command. ``stdout``/``stderr`` are decoded with replacement, never raising."""

    returncode: int
    stdout: str
    stderr: str
    timed_out: bool


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
        finally:
            _kill_group(proc)


def _kill_group(proc: subprocess.Popen) -> None:
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        proc.kill()
