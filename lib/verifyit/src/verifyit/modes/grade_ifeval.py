# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Mode ifeval: the answer file must satisfy every IFEval constraint the spec lists.

The reward is all-or-nothing, and the detail records each constraint's verdict. An unknown
constraint raises ``InvalidTask`` before the candidate is read.
"""

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from verifyit.grade import InvalidTask, Reward, infra_error, read_output, scored
from verifyit.modes.ifeval import CONSTRAINTS, Check
from verifyit.spec import Constraint, IfevalSpec


def resolve_checks(constraints: tuple[Constraint, ...]) -> list[tuple[Constraint, Check]]:
    if not constraints:
        raise InvalidTask("ifeval spec lists no constraints")
    unknown = sorted({c.name for c in constraints} - set(CONSTRAINTS))
    if unknown:
        raise InvalidTask(f"unknown ifeval constraints: {unknown}")
    return [(c, CONSTRAINTS[c.name]) for c in constraints]


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


def grade_ifeval_candidate(constraints: tuple[Constraint, ...], text: str) -> Reward:
    """Score a chat answer with the source's all-or-nothing constraint contract."""
    checks = resolve_checks(constraints)
    results = []
    for verdict in _evaluated_checks(checks, text):
        if isinstance(verdict.error, (KeyError, TypeError, ValueError)):
            return infra_error(f"Invalid constraint parameters: {verdict.error}")
        if verdict.error is not None:
            raise verdict.error
        results.append(verdict.passed)
    return scored(float(all(results)))


def grade(spec: IfevalSpec, tests_dir: Path, workspace: Path) -> Reward:
    checks = resolve_checks(spec.constraints)
    text = read_output(spec, workspace)
    if text is None:
        return scored(0.0, reason="no_output")

    results = [
        {"name": verdict.name, "passed": verdict.passed, "detail": verdict.detail}
        for verdict in _evaluated_checks(checks, text)
    ]
    failed = [result["name"] for result in results if not result["passed"]]
    return scored(0.0 if failed else 1.0, constraints=results, failed=failed)
