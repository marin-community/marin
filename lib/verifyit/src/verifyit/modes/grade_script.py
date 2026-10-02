# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Mode script: run the task's own grading script and read back the reward it reports.

This is the fallback for converters whose original ``test.sh`` logic fits no other mode. The script
runs in the agent's workspace with ``VERIFYIT_TESTS_DIR``, ``VERIFYIT_WORKSPACE`` and
``VERIFYIT_LOGS_DIR`` exported, and reports its reward through one of three channels, checked in
this order: ``$VERIFYIT_LOGS_DIR/reward.json`` holding the finite numeric ``spec.reward_key``,
``$VERIFYIT_LOGS_DIR/reward.txt`` holding a bare float, or a float on the last non-empty line of
stdout. Named keys require reward.json. Numeric auxiliary metrics are retained in verdict detail.
Malformed authoritative files and scripts reporting no reward produce infrastructure failures.
Nonzero producers never score positively. Optional verdict_file declares an authoritative
private-log JSON verdict with status, reward and detail; detail.script is reserved for execution
diagnostics. Structured producers must finish successfully, including before their timeout.
"""

import json
import logging
import math
import os
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from verifyit.execution.command import run_command
from verifyit.execution.worker import call_bounded
from verifyit.file_ops.read import read_regular_bytes
from verifyit.grade import (
    REWARD_JSON,
    REWARD_TXT,
    InvalidTask,
    Reward,
    Status,
    _validated_reward,
    infra_error,
    invalid_task,
    scored,
)
from verifyit.json_objects import unique_object
from verifyit.modes.extract import last_line
from verifyit.modes.run import STDERR_TAIL
from verifyit.spec import DEFAULT_REWARD_KEY, DEFAULT_WORKSPACE, ScriptSpec, Spec

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
    metrics: dict[str, float] | None = None


def grade_script_callable(function: Callable[..., Reward], *args: object, timeout: float, **kwargs: object) -> Reward:
    """Run a trusted importable Python grader under the Script verdict contract.

    The callable is client-owned grading code, never candidate-provided code.
    Existing bounded execution owns transport, deadlines and process cleanup.
    """
    try:
        return _validated_reward(call_bounded(function, *args, timeout=timeout, **kwargs))
    except InvalidTask as error:
        return invalid_task(str(error))
    except Exception as error:
        return infra_error(f"{type(error).__name__}: {error}")


def grade(spec: Spec, tests_dir: Path, workspace: Path) -> Reward:
    assert isinstance(spec, ScriptSpec)
    if spec.verdict_file is not None:
        if not isinstance(spec.verdict_file, str):
            raise InvalidTask("verdict_file must be a relative file path string")
        declared = Path(spec.verdict_file)
        if not spec.verdict_file or declared.is_absolute() or ".." in declared.parts or declared == Path("."):
            raise InvalidTask("verdict_file must name a file within the private script logs directory")
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
            "VERIFYIT_TESTS_DIR": str(tests_dir),
            "VERIFYIT_WORKSPACE": str(cwd),
            "VERIFYIT_LOGS_DIR": str(logs_dir),
        }
        completion = _run(command, cwd, env, spec.timeout)
        if spec.verdict_file is not None:
            return _structured_verdict(logs_dir, spec.verdict_file, completion)
        if completion.exit_code is not None and completion.exit_code != 0:
            return Reward(
                0.0,
                Status.INFRA_ERROR,
                {
                    "error": "script producer failed",
                    "exit_code": completion.exit_code,
                    "stderr": completion.stderr[-STDERR_TAIL:],
                },
            )
        reported = _reported_reward(logs_dir, completion.stdout, spec.reward_key)

    detail: dict = {"exit_code": completion.exit_code, "stderr": completion.stderr[-STDERR_TAIL:]}
    if completion.exit_code is None:
        return scored(0.0, reason="timeout", timeout=spec.timeout, **detail)
    if reported is None:
        raise RuntimeError(
            f"script {spec.path!r} exited {completion.exit_code} without reporting a reward; "
            f"stderr tail: {completion.stderr[-STDERR_TAIL:]!r}"
        )
    detail["channel"] = reported.channel.value
    if reported.metrics is not None:
        detail["metrics"] = reported.metrics
    if not 0.0 <= reported.value <= 1.0:
        return scored(0.0, reason="reward_out_of_range", reported=reported.value, **detail)
    return scored(reported.value, **detail)


def _run(command: list[str], cwd: Path, env: dict[str, str], timeout: float) -> Completion:
    completed = run_command(command, cwd, timeout, env=env)
    if completed.timed_out:
        logger.warning("script %s exceeded %.1fs; killed its process group", command[1], timeout)
        return Completion(None, completed.stdout, completed.stderr)
    return Completion(completed.returncode, completed.stdout, completed.stderr)


def _reported_reward(logs_dir: Path, stdout: str, reward_key: str) -> Reported | None:
    json_path = logs_dir / REWARD_JSON
    if json_path.exists() or json_path.is_symlink():
        return _json_reward(json_path, reward_key)
    if reward_key != DEFAULT_REWARD_KEY:
        raise RuntimeError(f"named reward {reward_key!r} requires reward.json")
    text_path = logs_dir / REWARD_TXT
    if text_path.exists() or text_path.is_symlink():
        value = _float(read_regular_bytes(text_path).decode(errors="replace"))
        if value is None or not math.isfinite(value):
            raise RuntimeError(f"{text_path} does not contain a finite numeric reward")
        return Reported(value, Channel.REWARD_TXT)
    value = _float(last_line(stdout))
    if value is not None and not math.isfinite(value):
        raise RuntimeError("script stdout does not contain a finite numeric reward")
    return Reported(value, Channel.STDOUT) if value is not None else None


def parse_json_reward(text: str, reward_key: str) -> tuple[float, dict]:
    """Validate a reward object while leaving auxiliary metric selection to the caller."""
    payload = json.loads(text, object_pairs_hook=unique_object)
    if not isinstance(payload, dict) or reward_key not in payload:
        raise ValueError(f"reward object lacks selected key {reward_key!r}")
    json.dumps(payload, allow_nan=False)
    return parse_reward_number(payload[reward_key]), payload


def parse_reward_number(value: object) -> float:
    """Read a finite numeric reward; the grading contract decides its allowed range."""
    result = _float(value)
    if result is None or not math.isfinite(result):
        raise ValueError("reward must be finite and numeric")
    return result


def _json_reward(path: Path, reward_key: str) -> Reported:
    try:
        value, payload = parse_json_reward(read_regular_bytes(path).decode(errors="replace"), reward_key)
    except ValueError as error:
        raise RuntimeError(f"{path} is not valid JSON reward: {error}") from error
    metrics = {}
    for key, raw_value in payload.items():
        metric = _float(raw_value)
        if metric is not None:
            if not math.isfinite(metric):
                raise RuntimeError(f"{path} contains nonfinite metric {key!r}")
            metrics[key] = metric
    return Reported(value, Channel.REWARD_JSON, metrics)


def _float(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, str | int | float):
        return None
    try:
        return float(value)
    except (ValueError, OverflowError):
        return None


def _structured_verdict(logs: Path, filename: str, completion: Completion) -> Reward:
    execution = {"exit_code": completion.exit_code, "stderr": completion.stderr[-STDERR_TAIL:]}
    if completion.exit_code != 0:
        return Reward(
            0.0,
            Status.INFRA_ERROR,
            {"error": "declared verdict producer did not complete successfully", "script": execution},
        )
    path = logs / filename
    if not path.resolve().is_relative_to(logs.resolve()):
        raise RuntimeError("declared script verdict escapes its private logs directory")
    try:
        payload = json.loads(
            read_regular_bytes(path).decode(), parse_constant=_reject_json_constant, object_pairs_hook=unique_object
        )
    except (OSError, ValueError) as error:
        return Reward(
            0.0,
            Status.INFRA_ERROR,
            {"error": f"declared script verdict is missing or malformed: {error}", "script": execution},
        )
    if not isinstance(payload, dict):
        raise RuntimeError("declared script verdict must be a JSON object")
    try:
        json.dumps(payload, allow_nan=False)
    except ValueError as error:
        raise RuntimeError("declared script verdict contains nonfinite JSON data") from error
    try:
        status = Status(payload["status"])
        value = payload["reward"]
        detail = payload["detail"]
    except (KeyError, ValueError, TypeError) as error:
        raise RuntimeError("declared script verdict requires status, reward and detail") from error
    verdict = _validated_reward(Reward(value, status, detail))
    return Reward(float(verdict.reward), verdict.status, {**verdict.detail, "script": execution})


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"nonfinite JSON constant {value}")
