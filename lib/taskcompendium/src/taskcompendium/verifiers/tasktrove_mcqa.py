# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""TaskTrove MCQA verifier handler for normalized option letters."""

from typing import Self

from pydantic import model_validator

from taskcompendium.grading import GradeResult, GradingAttempt, Outcome, Verifier
from taskcompendium.models import VerifierKind, VerifierSpec
from taskcompendium.submission import extract_answer


class TaskTroveMcqaVerifier(Verifier):
    """Original TaskTrove MCQA answer and option count."""

    expected: str
    options: int

    @model_validator(mode="after")
    def validate_answer(self) -> Self:
        if (
            not 1 <= self.options <= 26
            or len(self.expected) != 1
            or not "A" <= self.expected < chr(ord("A") + self.options)
        ):
            raise ValueError("MCQA verifier requires an option letter and option count")
        return self

    def grade(self, attempt: GradingAttempt) -> GradeResult:
        try:
            candidate = extract_answer(attempt.response, attempt.convention).strip().upper()
        except (ValueError, TypeError) as error:
            return GradeResult(Outcome.EXTRACTION_ERROR, None, str(error))
        if len(candidate) != 1 or not "A" <= candidate <= "Z":
            return GradeResult(Outcome.EXTRACTION_ERROR, None, "MCQA response requires one option letter")
        return GradeResult(Outcome.GRADED, float(candidate == self.expected))


def tasktrove_mcqa_verifier(expected: str, options: int) -> VerifierSpec:
    verifier = TaskTroveMcqaVerifier(expected=expected.strip().upper(), options=options)
    return VerifierSpec(kind=VerifierKind.TASKTROVE_MCQA, parameters_json=verifier.model_dump_json())
