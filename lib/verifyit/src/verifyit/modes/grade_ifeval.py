# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Mode ifeval: the answer file must satisfy every IFEval constraint the spec lists.

The reward is all-or-nothing, and the detail records each constraint's verdict. An unknown
constraint raises ``InvalidTask`` before the candidate is read.
"""

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from verifyit.grade import (
    Aggregation,
    InvalidTask,
    Reward,
    aggregate_rewards,
    empty_output_policy,
    infra_error,
    read_output,
    scored,
)
from verifyit.modes.ifeval import CONSTRAINTS, Check
from verifyit.spec import Constraint, EmptyOutputPolicy, IfevalSpec


def resolve_checks(
    constraints: tuple[Constraint, ...], registry: Mapping[str, Check] | None = None
) -> list[tuple[Constraint, Check]]:
    if not constraints:
        raise InvalidTask("ifeval spec lists no constraints")
    checks = CONSTRAINTS
    if registry is not None:
        if set(registry) & set(CONSTRAINTS) or any(not callable(check) for check in registry.values()):
            raise InvalidTask("custom ifeval checks must be callable and use distinct names")
        checks = {**CONSTRAINTS, **registry}
    unknown = sorted({c.name for c in constraints} - set(checks))
    if unknown:
        raise InvalidTask(f"unknown ifeval constraints: {unknown}")
    return [(c, checks[c.name]) for c in constraints]


@dataclass(frozen=True)
class ConstraintVerdict:
    name: str
    passed: bool
    detail: str
    error: Exception | None = None


def _evaluated_checks(checks: list[tuple[Constraint, Check]], text: str) -> Iterator[ConstraintVerdict]:
    for constraint, check in checks:
        try:
            passed, detail = check(text, constraint.params)
            yield ConstraintVerdict(constraint.name, passed, detail)
        except Exception as error:
            yield ConstraintVerdict(constraint.name, False, f"{type(error).__name__}: {error}", error)


def grade(spec: IfevalSpec, tests_dir: Path, workspace: Path) -> Reward:
    checks = resolve_checks(spec.constraints)
    text = read_output(spec, workspace)
    if text is None:
        return scored(0.0, reason="no_output")

    return _grade_checks(checks, text)


def grade_ifeval_candidate(spec: IfevalSpec, candidate: str, *, registry: Mapping[str, Check] | None = None) -> Reward:
    """Grade prepared text with optional additional trusted, process-local checks."""
    checks = resolve_checks(spec.constraints, registry)
    policy = empty_output_policy(spec)
    if not isinstance(candidate, str):
        raise InvalidTask("ifeval candidate must be text")
    if not candidate.strip() and policy is EmptyOutputPolicy.ZERO:
        return scored(0.0, reason="no_output")
    return _grade_checks(checks, candidate)


def grade_ifeval_chat_candidate(constraints: tuple[Constraint, ...], text: str) -> Reward:
    """Keep direct-chat parameter failures separate from candidate constraint failures."""
    results = []
    for verdict in _evaluated_checks(resolve_checks(constraints), text):
        if isinstance(verdict.error, (KeyError, TypeError, ValueError)):
            return infra_error(f"Invalid constraint parameters: {verdict.error}")
        if verdict.error is not None:
            raise verdict.error
        results.append(verdict.passed)
    return scored(float(all(results)))


def _grade_checks(checks: list[tuple[Constraint, Check]], text: str) -> Reward:
    results = []
    for verdict in _evaluated_checks(checks, text):
        if type(verdict.passed) is not bool or not isinstance(verdict.detail, str):
            raise RuntimeError("ifeval check returned an invalid result")
        results.append({"name": verdict.name, "passed": verdict.passed, "detail": verdict.detail})
    failed = [result["name"] for result in results if not result["passed"]]
    return scored(0.0 if failed else 1.0, constraints=results, failed=failed)


def grade_instruction_observations(observations: Sequence[tuple[dict[str, Any] | IfevalSpec, Any]]) -> Reward:
    """Require every prepared Schema or IFEval observation to pass."""
    # Schema dependencies are optional; ordinary IFEval needs only the standard library.
    from verifyit.modes.grade_json_schema import grade_json_schema_candidate  # noqa: PLC0415

    verdicts = []
    for constraint, candidate in observations:
        if isinstance(constraint, IfevalSpec):
            verdicts.append(grade_ifeval_candidate(constraint, candidate))
        elif isinstance(constraint, dict):
            verdicts.append(grade_json_schema_candidate(constraint, candidate))
        else:
            raise InvalidTask("Instruction observation requires Schema or IFEval inputs")
    return aggregate_rewards(verdicts, expected_total=len(observations), policy=Aggregation.ALL)
