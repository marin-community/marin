# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Grade mathematical answers with the shared symbolic comparison contract."""

from typing import Self

from pydantic import model_validator
from tasktrove_verify.grade import InvalidTask
from tasktrove_verify.modes.grade_math import grade_math_candidate
from tasktrove_verify.spec import MathSpec, MathType

from taskcompendium.grading import GradeResult, Outcome, Verifier
from taskcompendium.models import VerifierKind, VerifierSpec
from taskcompendium.submission import GradingAttempt, Submission, TextSubmission


class MathematicalAnswerVerifier(Verifier):
    """Compare extracted answer text with a private mathematical expression."""

    expected: str
    math_type: MathType

    @model_validator(mode="after")
    def validate_contract(self) -> Self:
        try:
            grade_math_candidate(MathSpec(expected=self.expected, math_type=self.math_type), None)
        except InvalidTask as error:
            raise ValueError(f"Invalid mathematical verifier contract: {error}") from error
        return self

    async def grade(self, submission: Submission, *, attempt: GradingAttempt) -> GradeResult:
        if not isinstance(submission, TextSubmission):
            raise TypeError("Mathematical verifier requires a text submission")
        result = grade_math_candidate(MathSpec(expected=self.expected, math_type=self.math_type), submission.value)
        return GradeResult(Outcome.GRADED, result.reward)


def mathematical_answer(expected: str, math_type: MathType) -> VerifierSpec:
    """Construct a symbolic verifier preserving the original expression and comparison type."""
    verifier = MathematicalAnswerVerifier(expected=expected, math_type=math_type)
    return VerifierSpec(kind=VerifierKind.MATHEMATICAL_ANSWER, parameters_json=verifier.model_dump_json())
