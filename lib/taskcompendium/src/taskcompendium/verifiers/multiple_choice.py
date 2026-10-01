# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Grade multiple-choice answers delivered through TaskCompendium conventions."""

from typing import Self

from pydantic import model_validator
from verifyit.grade import InvalidTask
from verifyit.modes.grade_mcq import grade_mcq_candidate
from verifyit.spec import McqSpec

from taskcompendium.grading import GradeResult, GradingAttempt, Outcome, Verifier
from taskcompendium.models import VerifierKind, VerifierSpec
from taskcompendium.submission import extract_answer


class MultipleChoiceVerifier(Verifier):
    """Grade one option letter after submission extraction."""

    expected: str
    options: int

    @model_validator(mode="after")
    def validate_contract(self) -> Self:
        try:
            grade_mcq_candidate(McqSpec(expected=self.expected, options=self.options), self.expected)
        except InvalidTask as error:
            raise ValueError(f"Invalid MCQ verifier contract: {error}") from error
        return self

    def grade(self, attempt: GradingAttempt) -> GradeResult:
        try:
            candidate = extract_answer(attempt.conversation[-1], attempt.convention).strip()
        except (ValueError, TypeError) as error:
            return GradeResult(Outcome.EXTRACTION_ERROR, None, str(error))
        if len(candidate) != 1 or not "A" <= candidate.upper() <= "Z":
            return GradeResult(Outcome.EXTRACTION_ERROR, None, "MCQA response requires one option letter")
        result = grade_mcq_candidate(McqSpec(expected=self.expected, options=self.options), candidate)
        return GradeResult(Outcome.GRADED, result.reward)


def multiple_choice_answer(expected: str, options: int) -> VerifierSpec:
    """Construct a verifier for one of the first ``options`` letters."""
    verifier = MultipleChoiceVerifier(expected=expected.strip().upper(), options=options)
    return VerifierSpec(kind=VerifierKind.MCQ_ANSWER, parameters_json=verifier.model_dump_json())
