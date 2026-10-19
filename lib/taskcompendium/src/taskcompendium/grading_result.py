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
    """Why a grading ended as ``Outcome.INFRA_ERROR``.

    ``STAGING`` is the verifier machine's preparation failing before the grader's command ran:
    the staged archive, its unpacking, the grading workspace or the reward files. The other kinds
    describe commands the task's author chose, the grader's own command, its environment's setup
    commands and its collect commands, and what the grader wrote.
    """

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
    """Where a sandbox grading stands, for a caller whose own deadline may expire before the grader runs.

    ``grade_in_sandbox`` sets ``STAGING`` while it starts and prepares the verifier machine and
    ``GRADER`` once the grader's command starts; a caller that captures the task's output files
    before grading sets ``STAGING`` around that capture too. A caller that times the whole grading
    out reads the phase to report the preparation, not the grader, as what ran out of time.
    """

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
