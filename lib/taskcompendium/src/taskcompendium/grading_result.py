# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Structured TaskCompendium grading outcomes."""

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class Outcome(StrEnum):
    GRADED = "graded"
    SUBMISSION_FAILURE = "submission_failure"
    INVALID_TASK = "invalid_task"
    INFRA_ERROR = "infra_error"
    UNAVAILABLE = "unavailable"


class GradingFailure(StrEnum):
    """Why a grading ended as ``Outcome.INFRA_ERROR``; ``STAGING`` identifies verifier preparation failures."""

    TIMEOUT = "timeout"
    MISSING_REWARD = "missing_reward"
    EMPTY_REWARD = "empty_reward"
    INVALID_REWARD = "invalid_reward"
    EXECUTION = "execution"
    STAGING = "staging"


class GradingPhase(StrEnum):
    STAGING = "staging"
    GRADER = "grader"


@dataclass
class GradingProgress:
    """Current phase used to classify an expired verifier deadline."""

    phase: GradingPhase = GradingPhase.GRADER


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
