# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Structured TaskCompendium grading outcomes."""

from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import Any

from rigging.task_errors import (
    REWARD_FILE_EMPTY_ERROR,
    REWARD_FILE_NOT_FOUND_ERROR,
    VERIFIER_OUTPUT_PARSE_ERROR,
    VERIFIER_RUNTIME_ERROR,
    VERIFIER_TIMEOUT_ERROR,
    ErrorIdentity,
)

GRADING_DURATION_SECONDS = "grading_duration_seconds"
GRADING_FAILURE = "grading_failure"


class Outcome(StrEnum):
    GRADED = "graded"
    SUBMISSION_FAILURE = "submission_failure"
    INVALID_TASK = "invalid_task"
    INFRA_ERROR = "infra_error"
    UNAVAILABLE = "unavailable"


class GradingFailure(StrEnum):
    TIMEOUT = "timeout"
    MISSING_REWARD = "missing_reward"
    EMPTY_REWARD = "empty_reward"
    INVALID_REWARD = "invalid_reward"
    EXECUTION = "execution"
    GRADER_EXIT = "grader_exit"
    GRADER_TIMEOUT = "grader_timeout"


@dataclass(frozen=True)
class GradeResult:
    status: Outcome
    reward: float | None
    error: str | None = None
    detail: dict | None = None
    passed: bool | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)
    failure: GradingFailure | None = None
    score_min: float = 0.0
    score_max: float = 1.0
    verifier_setup_seconds: float = 0.0


_GRADING_ERROR_TYPES = MappingProxyType(
    {
        GradingFailure.TIMEOUT: VERIFIER_TIMEOUT_ERROR,
        GradingFailure.MISSING_REWARD: REWARD_FILE_NOT_FOUND_ERROR,
        GradingFailure.EMPTY_REWARD: REWARD_FILE_EMPTY_ERROR,
        GradingFailure.INVALID_REWARD: VERIFIER_OUTPUT_PARSE_ERROR,
        GradingFailure.EXECUTION: VERIFIER_RUNTIME_ERROR,
    }
)


def grading_failure_identity(grade: GradeResult) -> ErrorIdentity | None:
    """Synthesize an identity for an INFRA_ERROR grade."""
    if grade.status is not Outcome.INFRA_ERROR:
        return None
    exception_type = VERIFIER_RUNTIME_ERROR if grade.failure is None else _GRADING_ERROR_TYPES[grade.failure]
    return ErrorIdentity(exception_type=exception_type)
