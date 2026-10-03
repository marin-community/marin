# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""TaskCompendium verifiers that need source-specific or runtime evidence."""

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum

from pydantic import BaseModel, ConfigDict
from verifyit.grade import Reward, Status

from taskcompendium.grading import GradeResult, Outcome
from taskcompendium.models import ConversationEvent
from taskcompendium.submission import Submission, extract_answer


class VerifierKind(StrEnum):
    CAPTURE_OUTPUT = "capture_output"
    CALENDAR_STATE = "calendar_state"
    IFEVAL = "ifeval"
    JSON_SCHEMA = "json_schema"
    STRUCTURED_FIELDS = "structured_fields"
    TASKTROVE_EXECUTABLE = "tasktrove_executable"
    REASONING_GYM = "reasoning_gym"
    PUZZLE_ANSWER = "puzzle_answer"
    SCHEDULE_ANSWER = "schedule_answer"
    REFERENCE_ANSWERS = "reference_answers"
    RUBRIC_JUDGE = "rubric_judge"
    REPOSITORY_PATCH = "repository_patch"
    SOURCE_CONTRACT = "source_contract"
    PREFERENCE_EVIDENCE = "preference_evidence"
    MATH_ANSWER = "math_answer"
    ABSTENTION_ANSWERS = "abstention_answers"
    ARC_GRID = "arc_grid"
    ARC_TRANSFORM = "arc_transform"
    INDIRECT_INJECTION = "indirect_injection"


@dataclass(frozen=True)
class GradingAttempt:
    """Submission and optional runtime evidence available to a custom verifier."""

    convention: Submission
    conversation: tuple[ConversationEvent, ...]
    environment: object


class Verifier(BaseModel, ABC):
    """Validated configuration for one source-specific or runtime grader."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    @abstractmethod
    def grade(self, attempt: GradingAttempt) -> GradeResult:
        """Grade the submission with this verifier's configuration."""


def grade_result(result: Reward) -> GradeResult:
    """Translate a standalone scorer result into the task grading contract."""
    if result.status != Status.SCORED:
        return GradeResult(Outcome.INFRA_ERROR, None, result.detail.get("error"))
    return GradeResult(Outcome.GRADED, result.reward, result.detail.get("error"))


def grade_extracted(attempt: GradingAttempt, score: Callable[[str], GradeResult]) -> GradeResult:
    """Extract a submission, preserving extraction errors separately from scorer failures."""
    try:
        candidate = extract_answer(attempt.conversation[-1], attempt.convention)
    except (ValueError, TypeError) as error:
        return GradeResult(Outcome.EXTRACTION_ERROR, None, str(error))
    return score(candidate)
