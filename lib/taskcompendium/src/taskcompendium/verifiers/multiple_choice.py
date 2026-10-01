# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Grade multiple-choice answers delivered through TaskCompendium conventions."""

from typing import Self

from pydantic import model_validator
from tasktrove_verify.grade import InvalidTask
from tasktrove_verify.modes.grade_mcq import grade_mcq_candidate
from tasktrove_verify.spec import McqSpec

from taskcompendium.grading import GradeResult, GradingAttempt, Outcome, Verifier
from taskcompendium.models import VerifierKind, VerifierSpec
from taskcompendium.submission import Submission, TextSubmission


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

    async def grade(self, submission: Submission, *, attempt: GradingAttempt) -> GradeResult:
        if not isinstance(submission, TextSubmission):
            raise TypeError("Multiple-choice verifier requires a text submission")
        candidate = submission.value.strip()
        if len(candidate) != 1 or not "A" <= candidate.upper() <= "Z":
            return GradeResult(Outcome.SUBMISSION_FAILURE, 0.0, "MCQA response requires one option letter")
        result = grade_mcq_candidate(McqSpec(expected=self.expected, options=self.options), candidate)
        return GradeResult(Outcome.GRADED, result.reward)


def multiple_choice_answer(expected: str, options: int) -> VerifierSpec:
    """Construct a verifier for one of the first ``options`` letters."""
    verifier = MultipleChoiceVerifier(expected=expected.strip().upper(), options=options)
    return VerifierSpec(kind=VerifierKind.MCQ_ANSWER, parameters_json=verifier.model_dump_json())
