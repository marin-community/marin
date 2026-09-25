# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""TaskTrove MCQA verifier handler for normalized option letters."""

from typing import Self

from pydantic import BaseModel, ConfigDict, model_validator

from taskcompendium.grading import GradeResult, GradingAttempt, Outcome, VerifierHandler
from taskcompendium.models import VerifierSpec
from taskcompendium.rendering import extract_answer

TASKTROVE_MCQA_KIND = "tasktrove_mcqa"


class TaskTroveMcqaPayload(BaseModel):
    """Original TaskTrove MCQA answer and option count."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

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


def tasktrove_mcqa(expected: str, options: int) -> VerifierSpec:
    """Construct a private TaskTrove MCQA verifier descriptor."""
    payload = TaskTroveMcqaPayload(expected=expected.strip().upper(), options=options)
    return VerifierSpec(kind=TASKTROVE_MCQA_KIND, parameters=payload.model_dump(mode="json"))


def _grade_tasktrove_mcqa(payload: TaskTroveMcqaPayload, attempt: GradingAttempt) -> GradeResult:
    try:
        candidate = extract_answer(attempt.response, attempt.rendering).strip().upper()
    except (ValueError, TypeError) as error:
        return GradeResult(Outcome.EXTRACTION_ERROR, None, str(error))
    if len(candidate) != 1 or not "A" <= candidate <= "Z":
        return GradeResult(Outcome.EXTRACTION_ERROR, None, "MCQA response requires one option letter")
    return GradeResult(Outcome.GRADED, float(candidate == payload.expected))


def tasktrove_mcqa_handler() -> VerifierHandler:
    return VerifierHandler(TaskTroveMcqaPayload, _grade_tasktrove_mcqa)
