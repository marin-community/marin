# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Grade a TaskTrove task and write its Harbor reward."""

import argparse
import importlib
import json
import logging
import sys
import traceback
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

from tasktrove_verify.spec import (
    DEFAULT_WORKSPACE,
    RUBRIC_REFERENCE,
    ExactSpec,
    JudgeSpec,
    MathSpec,
    McqSpec,
    Mode,
    NumericSpec,
    Spec,
    mode_of,
    parse_spec,
)

DEFAULT_LOGS_DIR = "/logs/verifier"
REWARD_JSON = "reward.json"
REWARD_TXT = "reward.txt"
VERDICT_JSON = "verdict.json"

logger = logging.getLogger("tasktrove_verify")


class Status(StrEnum):
    SCORED = "scored"
    INVALID_TASK = "invalid_task"
    INFRA_ERROR = "infra_error"


class InvalidTask(Exception):
    """The task is malformed: a reference is missing or its grading contract is invalid."""


@dataclass(frozen=True)
class Reward:
    reward: float
    status: Status
    detail: dict = field(default_factory=dict)


def scored(reward: float, **detail: object) -> Reward:
    return Reward(float(reward), Status.SCORED, dict(detail))


def invalid_task(message: str) -> Reward:
    return Reward(0.0, Status.INVALID_TASK, {"error": message})


def infra_error(message: str) -> Reward:
    return Reward(0.0, Status.INFRA_ERROR, {"error": message})


def write_reward(logs_dir: Path, reward: Reward) -> None:
    """Write a verdict and, for a scored grade, Harbor's reward files."""
    logs_dir.mkdir(parents=True, exist_ok=True)
    for name in (REWARD_JSON, REWARD_TXT):
        (logs_dir / name).unlink(missing_ok=True)
    verdict = {"reward": reward.reward, "status": reward.status.value, "detail": reward.detail}
    (logs_dir / VERDICT_JSON).write_text(json.dumps(verdict) + "\n")
    if reward.status != Status.SCORED:
        return
    (logs_dir / REWARD_JSON).write_text(json.dumps({"reward": reward.reward}) + "\n")
    (logs_dir / REWARD_TXT).write_text(f"{reward.reward}\n")


def local_output_path(output: str, workspace: Path) -> Path:
    """Re-root an output under the container's default workspace for local grading."""
    path = Path(output)
    prefix = Path(DEFAULT_WORKSPACE).parts
    if path.is_absolute() and path.parts[: len(prefix)] == prefix:
        return workspace.joinpath(*path.parts[len(prefix) :])
    return path


def read_output(spec: Spec, workspace: Path) -> str | None:
    """Read a non-empty candidate from an output-file spec."""
    output = local_output_path(spec.output, workspace)  # type: ignore[union-attr]
    if not output.is_file():
        return None
    text = output.read_text(errors="replace")
    return text if text.strip() else None


def positive_candidate(spec: Spec) -> str | None:
    """Return a candidate that must score one, when the mode has a safe probe."""
    if isinstance(spec, McqSpec):
        return f"Answer: {spec.expected}"
    if isinstance(spec, MathSpec | NumericSpec):
        return f"\\boxed{{{spec.expected}}}"
    if isinstance(spec, ExactSpec):
        return "\n".join(spec.expected)
    if isinstance(spec, JudgeSpec):
        gated = spec.rubric == RUBRIC_REFERENCE and spec.exact_gate and spec.references and not spec.constraints
        return spec.references[0] if gated else None
    return None


def negative_candidate(spec: Spec) -> str | None:
    """Return a candidate that must score zero, when a safe perturbation exists."""
    if isinstance(spec, McqSpec):
        other = "B" if spec.expected.upper() != "B" else "A"
        return f"Answer: {other}"
    if isinstance(spec, NumericSpec):
        return f"\\boxed{{{spec.expected + 1.0}}}"
    if isinstance(spec, ExactSpec) and len(spec.expected) > 1 and spec.ordered:
        return "\n".join(reversed(spec.expected))
    return None


# Each mode module takes its own spec type; the dispatch key guarantees the match.
Grader = Callable[[Any, Path, Path], Reward]

# Mode modules are imported on first use: several depend on an extra (math-verify, jsonschema,
# reasoning-gym, openai) that only the images needing that mode install. A missing extra
# surfaces as an ImportError from the grader, which the CLI records as infra_error.
MODE_MODULES: dict[Mode, str] = {
    Mode.MCQ: "grade_mcq",
    Mode.MATH: "grade_math",
    Mode.NUMERIC: "grade_math",
    Mode.EXACT: "grade_exact",
    Mode.JSON_SCHEMA: "grade_json_schema",
    Mode.XML_ELEMENTS: "grade_xml",
    Mode.CSV_COLUMNS: "grade_csv",
    Mode.IFEVAL: "grade_ifeval",
    Mode.REASONING_GYM: "grade_reasoning_gym",
    Mode.STDIO: "grade_stdio",
    Mode.PYTEST: "grade_pytest",
    Mode.JUNIT: "grade_junit",
    Mode.GOTEST: "grade_gotest",
    Mode.JUDGE: "grade_judge",
    Mode.SCRIPT: "grade_script",
}
GRADERS: dict[Mode, Grader] = {}


def grader_for(mode: Mode) -> Grader:
    if mode not in GRADERS:
        GRADERS[mode] = importlib.import_module(f"tasktrove_verify.modes.{MODE_MODULES[mode]}").grade
    return GRADERS[mode]


def grade(spec: Spec, tests_dir: Path, workspace: Path) -> Reward:
    """Grade one task. ``tests_dir`` holds verifier.toml and its data; ``workspace`` is the agent's tree.

    Modes that read an output file use ``spec.output``; execution modes use ``spec.workspace``.
    ``workspace`` here is the fallback for specs that leave those at their defaults but run
    somewhere else, such as a local gate in a temporary directory.
    """
    try:
        return grader_for(mode_of(spec))(spec, tests_dir, workspace)
    except InvalidTask as error:
        return invalid_task(str(error))


def run(spec_path: Path, workspace: Path) -> Reward:
    try:
        spec = parse_spec(spec_path.read_text())
    except (OSError, ValueError, KeyError) as error:
        return invalid_task(f"cannot read verifier spec {spec_path}: {error}")
    try:
        return grade(spec, tests_dir=spec_path.parent, workspace=workspace)
    except Exception as error:
        logger.error("grader crashed: %s", traceback.format_exc())
        return infra_error(f"{type(error).__name__}: {error}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("spec", type=Path, help="path to verifier.toml")
    parser.add_argument("--logs-dir", type=Path, default=Path(DEFAULT_LOGS_DIR))
    parser.add_argument("--workspace", type=Path, default=Path(DEFAULT_WORKSPACE))
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, stream=sys.stderr, format="%(levelname)s %(message)s")
    reward = run(args.spec, args.workspace)
    write_reward(args.logs_dir, reward)
    logger.info("reward=%s status=%s", reward.reward, reward.status.value)
    return 0
