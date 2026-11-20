# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Structured TaskCompendium grading outcomes."""

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from rigging.task_errors import ErrorIdentity


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
    error_identity: ErrorIdentity | None = None


_GRADING_ERROR_TYPES = {
    GradingFailure.TIMEOUT: "VerifierTimeoutError",
    GradingFailure.MISSING_REWARD: "RewardFileNotFoundError",
    GradingFailure.EMPTY_REWARD: "RewardFileEmptyError",
    GradingFailure.INVALID_REWARD: "VerifierOutputParseError",
    GradingFailure.EXECUTION: "VerifierRuntimeError",
}


def grading_failure_identity(grade: GradeResult) -> ErrorIdentity | None:
    """Resolve a verifier outcome to the shared task error identity."""
    if grade.error_identity is not None:
        return grade.error_identity
    if grade.status is not Outcome.INFRA_ERROR:
        return None
    return ErrorIdentity(exception_type=_GRADING_ERROR_TYPES.get(grade.failure, "VerifierRuntimeError"))
