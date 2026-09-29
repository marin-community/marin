# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Adapt TaskTrove's pinned verifier contract to TaskCompendium submissions."""

from typing import Self

from pydantic import model_validator
from tasktrove_verify.grade import InvalidTask
from tasktrove_verify.modes.grade_mcq import grade_mcq_candidate
from tasktrove_verify.spec import McqSpec

from taskcompendium.grading import GradeResult, GradingAttempt, Outcome, Verifier
from taskcompendium.models import VerifierKind, VerifierSpec
from taskcompendium.submission import extract_answer


class TaskTroveVerifier(Verifier):
    """Use TaskTrove's MCQ scorer after TaskCompendium submission extraction.

    This direct-answer slice supports the MCQ mode. Executable modes need
    verifier-only resources and an isolated runtime before they can be added.
    """

    expected: str
    options: int

    @model_validator(mode="after")
    def validate_contract(self) -> Self:
        try:
            grade_mcq_candidate(McqSpec(expected=self.expected, options=self.options), self.expected)
        except InvalidTask as error:
            raise ValueError(f"Invalid TaskTrove MCQ contract: {error}") from error
        return self

    def grade(self, attempt: GradingAttempt) -> GradeResult:
        try:
            candidate = extract_answer(attempt.response, attempt.convention).strip()
        except (ValueError, TypeError) as error:
            return GradeResult(Outcome.EXTRACTION_ERROR, None, str(error))
        if len(candidate) != 1 or not "A" <= candidate.upper() <= "Z":
            return GradeResult(Outcome.EXTRACTION_ERROR, None, "MCQA response requires one option letter")
        result = grade_mcq_candidate(McqSpec(expected=self.expected, options=self.options), candidate)
        return GradeResult(Outcome.GRADED, result.reward)


def tasktrove_verifier(contract: McqSpec) -> VerifierSpec:
    """Construct a TaskTrove verifier descriptor from a parsed MCQ contract."""
    verifier = TaskTroveVerifier(expected=contract.expected.strip().upper(), options=contract.options)
    return VerifierSpec(kind=VerifierKind.TASKTROVE, parameters_json=verifier.model_dump_json())
