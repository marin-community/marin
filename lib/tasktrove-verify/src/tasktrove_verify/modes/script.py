# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Mode script: run the task's own grading script and read back the reward it reports.

This is the fallback for converters whose original ``test.sh`` logic fits no other mode. The script
runs in the agent's workspace with ``TASKTROVE_TESTS_DIR``, ``TASKTROVE_WORKSPACE`` and
``TASKTROVE_LOGS_DIR`` exported, and reports its reward through one of three channels, checked in
this order: ``$TASKTROVE_LOGS_DIR/reward.json`` holding ``{"reward": <float>, ...}``,
``$TASKTROVE_LOGS_DIR/reward.txt`` holding a bare float, or a float on the last non-empty line of
stdout. A script that exits without reporting a reward is an infrastructure failure, not a zero.
"""

import json
import logging
import os
import tempfile
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from tasktrove_verify.modes.extract import last_line
from tasktrove_verify.modes.run import STDERR_TAIL, run_command
from tasktrove_verify.reward import REWARD_JSON, REWARD_TXT, InvalidTask, Reward, scored
from tasktrove_verify.spec import DEFAULT_WORKSPACE, ScriptSpec, Spec

SHELL = "bash"
PYTHON = "python3"

logger = logging.getLogger(__name__)


class Channel(StrEnum):
    REWARD_JSON = "reward.json"
    REWARD_TXT = "reward.txt"
    STDOUT = "stdout"


@dataclass(frozen=True)
class Completion:
    """One script run. ``exit_code`` is ``None`` when the script was killed at the timeout."""

    exit_code: int | None
    stdout: str
    stderr: str


@dataclass(frozen=True)
class Reported:
    value: float
    channel: Channel


def grade(spec: Spec, tests_dir: Path, workspace: Path) -> Reward:
    assert isinstance(spec, ScriptSpec)
    script = tests_dir / spec.path
    if not script.is_file():
        raise InvalidTask(f"script {spec.path!r} is missing from the tests directory")

    cwd = workspace if spec.workspace == DEFAULT_WORKSPACE else Path(spec.workspace)
    interpreter = SHELL if script.suffix == ".sh" else PYTHON
    command = [interpreter, str(script), *spec.args]

    with tempfile.TemporaryDirectory(prefix="tasktrove-script-") as logs:
        logs_dir = Path(logs)
        env = {
            **os.environ,
            "TASKTROVE_TESTS_DIR": str(tests_dir),
            "TASKTROVE_WORKSPACE": str(cwd),
            "TASKTROVE_LOGS_DIR": str(logs_dir),
        }
        completion = _run(command, cwd, env, spec.timeout)
        reported = _reported_reward(logs_dir, completion.stdout)

    detail: dict = {"exit_code": completion.exit_code, "stderr": completion.stderr[-STDERR_TAIL:]}
    if completion.exit_code is None:
        return scored(0.0, reason="timeout", timeout=spec.timeout, **detail)
    if reported is None:
        raise RuntimeError(
            f"script {spec.path!r} exited {completion.exit_code} without reporting a reward; "
            f"stderr tail: {completion.stderr[-STDERR_TAIL:]!r}"
        )
    detail["channel"] = reported.channel.value
    if not 0.0 <= reported.value <= 1.0:
        return scored(0.0, reason="reward_out_of_range", reported=reported.value, **detail)
    return scored(reported.value, **detail)


def _run(command: list[str], cwd: Path, env: dict[str, str], timeout: float) -> Completion:
    completed = run_command(command, cwd, timeout, env=env)
    if completed.timed_out:
        logger.warning("script %s exceeded %.1fs; killed its process group", command[1], timeout)
        return Completion(None, completed.stdout, completed.stderr)
    return Completion(completed.returncode, completed.stdout, completed.stderr)


def _reported_reward(logs_dir: Path, stdout: str) -> Reported | None:
    value = _json_reward(logs_dir / REWARD_JSON)
    if value is not None:
        return Reported(value, Channel.REWARD_JSON)
    value = _float((logs_dir / REWARD_TXT).read_text(errors="replace")) if (logs_dir / REWARD_TXT).is_file() else None
    if value is not None:
        return Reported(value, Channel.REWARD_TXT)
    value = _float(last_line(stdout))
    return Reported(value, Channel.STDOUT) if value is not None else None


def _json_reward(path: Path) -> float | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(errors="replace"))
    except ValueError:
        logger.warning("%s is not valid JSON", path)
        return None
    return _float(payload.get("reward")) if isinstance(payload, dict) else None


def _float(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, str | int | float):
        return None
    try:
        return float(value)
    except ValueError:
        return None
