# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Structured TaskCompendium grading outcomes."""

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class Outcome(StrEnum):
    GRADED = "graded"
    EXTRACTION_ERROR = "extraction_error"
    INVALID_TASK = "invalid_task"
    INFRA_ERROR = "infra_error"
    UNAVAILABLE = "unavailable"
    SKIPPED = "skipped"


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
    passed: bool | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)
    failure: GradingFailure | None = None
    score_min: float = 0.0
    score_max: float = 1.0
    rewards: dict[str, float] = field(default_factory=dict)
    detail: dict | None = None
