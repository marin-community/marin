# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Grade a TaskTrove task and write its Harbor reward."""

import argparse
import importlib
import json
import logging
import math
import sys
import traceback
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from fractions import Fraction
from pathlib import Path
from typing import Any

from harbor_config.errors import ErrorCategory

from verifyit.file_ops.read import read_text
from verifyit.numeric import MAX_NUMERIC_DIGITS, numeric_literal
from verifyit.spec import (
    DEFAULT_WORKSPACE,
    RUBRIC_REFERENCE,
    EmptyOutputPolicy,
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

logger = logging.getLogger("verifyit")


class Status(StrEnum):
    SCORED = "scored"
    INVALID_TASK = "invalid_task"
    INFRA_ERROR = "infra_error"


class Aggregation(StrEnum):
    ALL = "all"
    MEAN = "mean"
    MAX = "max"
    MIN = "min"
    PRODUCT = "product"


class InvalidTask(Exception):
    """The task is malformed: a reference is missing or its grading contract is invalid."""


@dataclass(frozen=True)
class Reward:
    reward: float
    status: Status
    detail: dict = field(default_factory=dict)


def scored(reward: float, **detail: object) -> Reward:
    verdict = Reward(reward, Status.SCORED, dict(detail))
    error = _reward_error(verdict)
    if error is not None:
        raise RuntimeError(error)
    return Reward(float(reward), Status.SCORED, dict(detail))


def _reward_error(verdict: object) -> str | None:
    if not isinstance(verdict, Reward):
        return "grader did not return a Reward"
    if not isinstance(verdict.status, Status):
        return "grader returned an invalid verdict status"
    value = verdict.reward
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not 0.0 <= value <= 1.0
        or not math.isfinite(value)
    ):
        return "grader reward must be a finite number in [0, 1]"
    if verdict.status != Status.SCORED and value != 0.0:
        return "unscored verdict must have zero reward"
    if not isinstance(verdict.detail, dict):
        return "grader verdict detail must be an object"
    try:
        json.dumps(verdict.detail, allow_nan=False)
    except (TypeError, ValueError, OverflowError, RecursionError):
        return "grader verdict detail is not valid JSON"
    return None


def _validated_reward(verdict: object) -> Reward:
    error = _reward_error(verdict)
    if error is not None:
        return infra_error(error)
    assert isinstance(verdict, Reward)
    return verdict


def aggregate_rewards(
    verdicts: Sequence[Reward], *, expected_total: int, policy: Aggregation, round_digits: int | None = None
) -> Reward:
    """Combine component grades without dropping missing or unscored components.

    Missing components earn zero. An invalid task or infrastructure error discards
    all credit; infrastructure errors take precedence when both occur. MIN keeps
    fractional credit only when every required component permits it. Optional
    decimal rounding is applied after aggregation, with ties-to-even scaling.
    """
    if type(expected_total) is not int or expected_total <= 0:
        return invalid_task("expected_total must be a positive integer")
    if not isinstance(policy, Aggregation):
        return invalid_task("unknown reward aggregation policy")
    if round_digits is not None and (type(round_digits) is not int or not 0 <= round_digits <= 6):
        return invalid_task("aggregation rounding must be an integer from zero to six or None")
    if len(verdicts) > expected_total:
        return invalid_task("more component grades than expected")
    validated = [_validated_reward(verdict) for verdict in verdicts]
    for status in (Status.INFRA_ERROR, Status.INVALID_TASK):
        for index, verdict in enumerate(validated):
            if verdict.status == status:
                return Reward(0.0, status, {"component": index, "cause": verdict.detail, "total": expected_total})
    passed = sum(verdict.reward == 1.0 for verdict in validated)
    if policy == Aggregation.ALL:
        reward = float(passed == expected_total)
    elif policy == Aggregation.MAX:
        reward = max((verdict.reward for verdict in validated), default=0.0)
    elif policy == Aggregation.MIN:
        reward = min((verdict.reward for verdict in validated), default=0.0) if len(validated) == expected_total else 0.0
    elif policy == Aggregation.PRODUCT:
        reward = math.prod(verdict.reward for verdict in validated) if len(validated) == expected_total else 0.0
    else:
        reward = sum(verdict.reward for verdict in validated) / expected_total
    if round_digits is not None:
        scale = 10**round_digits
        reward = round(reward * scale) / scale
    return scored(reward, passed=passed, total=expected_total, missing=expected_total - len(validated))


def aggregate_first_fit(verdicts: Sequence[Sequence[Reward]], *, expected_total: int) -> Reward:
    """Require a first-fit one-to-one assignment of binary primitive grades.

    Rows and columns retain caller-declared order. Each row consumes its first
    still-unused passing column; this is deliberately not maximum matching.
    Validate every edge before selection, including edges that selection would
    not visit. Missing or extra candidate dimensions score zero, ragged matrices
    are infrastructure failures, and an empty trusted assignment is invalid.
    """
    if type(expected_total) is not int or expected_total <= 0:
        return invalid_task("expected_total must be a positive integer")
    columns = len(verdicts[0]) if verdicts else 0
    if any(len(row) != columns for row in verdicts):
        return infra_error("assignment grades must form a rectangular matrix")
    edges = [verdict for row in verdicts for verdict in row]
    if edges:
        admission = aggregate_rewards(edges, expected_total=len(edges), policy=Aggregation.ALL)
        if admission.status != Status.SCORED:
            return admission
        if any(verdict.reward not in (0.0, 1.0) for verdict in edges):
            return infra_error("assignment requires binary primitive grades")
    if len(verdicts) != expected_total or columns != expected_total:
        return scored(0.0, reason="assignment_cardinality", rows=len(verdicts), columns=columns, total=expected_total)
    used: set[int] = set()
    for row in verdicts:
        match = next((index for index, verdict in enumerate(row) if index not in used and verdict.reward == 1.0), None)
        if match is None:
            return scored(0.0, matched=len(used), total=expected_total)
        used.add(match)
    return scored(1.0, matched=len(used), total=expected_total)


def invalid_task(message: str) -> Reward:
    return Reward(0.0, Status.INVALID_TASK, {"error": message})


def finalize_preparation_failure(
    *, status: Status, category: ErrorCategory, error_type: str, message: str, stage: str
) -> Reward:
    """Terminate failed preparation at minimum reward without grading partial data.

    Task validity is separate from the imported framework error category.
    PASSTHROUGH requires a completed grade; preparation has none to preserve.
    """
    source_status = status
    if status != Status.INVALID_TASK and (status != Status.SCORED or category != ErrorCategory.AGENT):
        status = Status.INFRA_ERROR
    return _validated_reward(
        Reward(
            0.0,
            status,
            {
                "error": message,
                "error_type": error_type,
                "category": category,
                "stage": stage,
                "source_status": source_status,
                "finalization_policy": "failed_preparation_v1",
            },
        )
    )


def numeric_tolerance(spec: NumericSpec) -> Fraction:
    """Validate exact private literals and return the effective tolerance."""
    try:
        expected = numeric_literal(spec.expected)
        absolute = numeric_literal(spec.tolerance_abs)
        relative = numeric_literal(spec.tolerance_rel)
    except ValueError as error:
        raise InvalidTask(f"invalid numeric contract: {error}") from error
    if absolute < 0 or relative < 0:
        raise InvalidTask("numeric tolerances must be nonnegative")
    return max(absolute, relative * abs(expected))


def infra_error(message: str) -> Reward:
    return Reward(0.0, Status.INFRA_ERROR, {"error": message})


def write_reward(logs_dir: Path, reward: Reward) -> None:
    """Write a verdict and, for a scored grade, Harbor's reward files."""
    logs_dir.mkdir(parents=True, exist_ok=True)
    for name in (REWARD_JSON, REWARD_TXT, VERDICT_JSON):
        (logs_dir / name).unlink(missing_ok=True)
    reward = _validated_reward(reward)
    verdict = {"reward": reward.reward, "status": reward.status.value, "detail": reward.detail}
    try:
        serialized = json.dumps(verdict, allow_nan=False)
    except (TypeError, ValueError, OverflowError, RecursionError):
        reward = infra_error("grader verdict detail is not valid JSON")
        serialized = json.dumps({"reward": reward.reward, "status": reward.status.value, "detail": reward.detail})
    (logs_dir / VERDICT_JSON).write_text(serialized + "\n")
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


def empty_output_policy(spec: Spec) -> EmptyOutputPolicy:
    """Validate the explicit policy shared by output-file and direct text grading."""
    policy = getattr(spec, "empty_output", None)
    if not isinstance(policy, EmptyOutputPolicy):
        raise InvalidTask("empty_output must be 'zero' or 'grade'")
    return policy


def read_output(spec: Spec, workspace: Path) -> str | None:
    """Read a present answer under its task policy; an absent file is never an answer."""
    policy = empty_output_policy(spec)
    output = local_output_path(spec.output, workspace)  # type: ignore[union-attr]
    if not output.is_file():
        return None
    text = read_text(output, errors="replace")
    return text if text.strip() or policy is EmptyOutputPolicy.GRADE else None


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
        try:
            tolerance = numeric_tolerance(spec)
        except InvalidTask:
            return None
        expected = numeric_literal(spec.expected)
        candidate = expected + max(2 * tolerance, 1)
        limit = 10**MAX_NUMERIC_DIGITS
        if abs(candidate.numerator) >= limit or candidate.denominator >= limit:
            return None
        return f"\\boxed{{{candidate}}}"
    if isinstance(spec, ExactSpec) and len(spec.expected) > 1 and spec.ordered:
        return "\n".join(reversed(spec.expected))
    return None


# Each mode module takes its own spec type; the dispatch key guarantees the match.
Grader = Callable[[Any, Path, Path], Reward]

# Mode modules are imported on first use: several depend on an extra (math-verify, jsonschema,
# reasoning-gym, openai) that only the images needing that mode install. A missing extra
# surfaces as an ImportError from the grader, which the CLI records as infra_error.
MODE_MODULES: dict[Mode, str] = {
    Mode.STRUCTURED_EXACT: "grade_structured_exact",
    Mode.PREDICTED_ACTION: "grade_predicted_action",
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
        GRADERS[mode] = importlib.import_module(f"verifyit.modes.{MODE_MODULES[mode]}").grade
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
        spec = parse_spec(read_text(spec_path))
    except (OSError, ValueError, KeyError) as error:
        return invalid_task(f"cannot read verifier spec {spec_path}: {error}")
    try:
        return _validated_reward(grade(spec, tests_dir=spec_path.parent, workspace=workspace))
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
