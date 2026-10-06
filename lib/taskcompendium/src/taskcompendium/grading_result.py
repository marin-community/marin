# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Structured TaskCompendium grading outcomes."""

from dataclasses import dataclass
from enum import StrEnum


class Outcome(StrEnum):
    GRADED = "graded"
    SUBMISSION_FAILURE = "submission_failure"
    INVALID_TASK = "invalid_task"
    INFRA_ERROR = "infra_error"


@dataclass(frozen=True)
class GradeResult:
    status: Outcome
    reward: float | None
    error: str | None = None
    detail: dict | None = None
