# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Grade text against the pinned IFEval instruction constraints."""

from dataclasses import dataclass
from typing import Self

from pydantic import JsonValue, model_validator
from tasktrove_verify.modes.grade_ifeval import resolve_checks
from tasktrove_verify.spec import Constraint

from taskcompendium.grading import GradeResult, Outcome, Verifier
from taskcompendium.models import VerifierKind, VerifierSpec
from taskcompendium.submission import GradingAttempt, Submission, TextSubmission


@dataclass(frozen=True)
class IFEvalConstraint:
    """One validated source IFEval constraint and its private parameters."""

    name: str
    params: dict[str, JsonValue]


class IFEvalVerifier(Verifier):
    """Apply every pinned IFEval check to an extracted text answer."""

    constraints: tuple[IFEvalConstraint, ...]

    @model_validator(mode="after")
    def validate_constraints(self) -> Self:
        resolve_checks(tuple(Constraint(item.name, item.params) for item in self.constraints))
        return self

    async def grade(self, submission: Submission, *, attempt: GradingAttempt) -> GradeResult:
        if not isinstance(submission, TextSubmission):
            raise TypeError("IFEval verifier requires a text submission")

        checks = resolve_checks(tuple(Constraint(item.name, item.params) for item in self.constraints))
        for constraint, check in checks:
            try:
                passed, _ = check(submission.value, constraint.params)
            except Exception as error:
                return GradeResult(Outcome.INFRA_ERROR, None, f"IFEval checker crashed: {type(error).__name__}")
            if not passed:
                return GradeResult(Outcome.GRADED, 0.0)
        return GradeResult(Outcome.GRADED, 1.0)


def ifeval_answer(constraints: tuple[IFEvalConstraint, ...]) -> VerifierSpec:
    """Construct a generic IFEval verifier from a validated source contract."""
    verifier = IFEvalVerifier(constraints=constraints)
    return VerifierSpec(kind=VerifierKind.IFEVAL, parameters_json=verifier.model_dump_json())
