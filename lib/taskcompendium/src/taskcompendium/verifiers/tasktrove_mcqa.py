# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""TaskTrove MCQA verifier handler backed by the source checker."""

import tempfile
from pathlib import Path

import msgspec
from tasktrove_verify.grade import Status, grade
from tasktrove_verify.spec import McqSpec

from taskcompendium.grading import GradeResult, GradingAttempt, Outcome, VerifierHandler
from taskcompendium.models import VerifierSpec
from taskcompendium.rendering import extract_answer

TASKTROVE_MCQA_KIND = "tasktrove_mcqa"


class TaskTroveMcqaPayload(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    """Original TaskTrove MCQA contract needed by its source verifier."""

    expected: str
    options: int

    def __post_init__(self) -> None:
        if (
            not 1 <= self.options <= 26
            or len(self.expected) != 1
            or not "A" <= self.expected < chr(ord("A") + self.options)
        ):
            raise ValueError("MCQA verifier requires an option letter and option count")


def tasktrove_mcqa(expected: str, options: int) -> VerifierSpec:
    """Construct a private TaskTrove MCQA verifier descriptor."""
    payload = TaskTroveMcqaPayload(expected.strip().upper(), options)
    return VerifierSpec(TASKTROVE_MCQA_KIND, msgspec.to_builtins(payload))


def _grade_tasktrove_mcqa(payload: TaskTroveMcqaPayload, attempt: GradingAttempt) -> GradeResult:
    try:
        candidate = extract_answer(attempt.response, attempt.rendering).strip().upper()
    except (ValueError, TypeError) as error:
        return GradeResult(Outcome.EXTRACTION_ERROR, None, str(error))
    if len(candidate) != 1 or not "A" <= candidate <= "Z":
        return GradeResult(Outcome.EXTRACTION_ERROR, None, "MCQA response requires one option letter")
    with tempfile.TemporaryDirectory(prefix="taskcompendium-answer-") as temporary:
        output = Path(temporary) / "answer.txt"
        output.write_text(f"Answer: {candidate}")
        contract = McqSpec(expected=payload.expected, options=payload.options, output=str(output))
        result = grade(contract, output.parent, output.parent)
    if result.status != Status.SCORED:
        return GradeResult(Outcome.INFRA_ERROR, None, str(result.detail))
    return GradeResult(Outcome.GRADED, result.reward)


def tasktrove_mcqa_handler() -> VerifierHandler:
    return VerifierHandler(TaskTroveMcqaPayload, _grade_tasktrove_mcqa)
