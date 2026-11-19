# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Structured TaskCompendium grading outcomes."""

import json
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


def verdict_bytes(grade: GradeResult) -> bytes:
    """Render the production structured verdict without scoring infrastructure failures."""
    status = {
        Outcome.GRADED: "scored",
        Outcome.SUBMISSION_FAILURE: "scored",
        Outcome.INVALID_TASK: "invalid_task",
        Outcome.INFRA_ERROR: "infra_error",
        Outcome.UNAVAILABLE: "infra_error",
    }[grade.status]
    detail = dict(grade.detail or {})
    if grade.diagnostics:
        detail["diagnostics"] = grade.diagnostics
    if grade.error is not None:
        detail["error"] = grade.error
    if grade.failure is not None:
        detail["failure"] = grade.failure.value
    return json.dumps(
        {"status": status, "reward": grade.reward if status == "scored" else 0.0, "detail": detail}, allow_nan=False
    ).encode()
