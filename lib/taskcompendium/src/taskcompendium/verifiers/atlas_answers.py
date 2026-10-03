# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Typed cleanup math scoring and the TaskTrove abstention exact gate."""

from typing import Literal

from verifyit.modes.grade_judge import grade_abstention_candidate
from verifyit.spec import MathSpec, MathType

from taskcompendium.grading import GradeResult
from taskcompendium.verifiers.base import GradingAttempt, Verifier, grade_extracted, grade_result
from taskcompendium.verifiers.reasoning import grade_answer
from taskcompendium.verifiers.reference_answers import ReferenceAnswersVerifier


class MathAnswerVerifier(Verifier):
    """Reuse the cleanup comparator without claiming original scorer parity."""

    expected: str
    math_type: Literal["scalar", "equation", "interval", "set", "tuple", "list"]

    def grade(self, attempt: GradingAttempt) -> GradeResult:
        return grade_extracted(
            attempt,
            lambda candidate: grade_answer(
                MathSpec(expected=self.expected, math_type=MathType(self.math_type)), candidate
            ),
        )


class AbstentionAnswersVerifier(Verifier):
    """Reject abstention before the explicitly unbound semantic fallback."""

    reference: ReferenceAnswersVerifier
    abstention_token: str | None

    def grade(self, attempt: GradingAttempt) -> GradeResult:
        return grade_extracted(
            attempt,
            lambda candidate: grade_result(
                grade_abstention_candidate(self.reference.references, self.abstention_token, candidate)
            ),
        )
