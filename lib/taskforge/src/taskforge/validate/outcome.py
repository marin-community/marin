# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Typed trial outcomes: a trial either produced a grade or failed for a classified ``Cause``.

``Graded`` holds a rollout whose grade is a judgment of the submission: ``GRADED``, or
``SUBMISSION_FAILURE`` (the model ended without a valid submission, which scores zero). An agent that runs
out of ``TaskExecution.agent_timeout`` is a budget stop like running out of turns, so it is
``Graded`` too (``classify.trial_outcome``). Everything else is ``Ungraded``: task setup,
infrastructure, grader and contract failures that say nothing about the submission. Statistics use graded outcomes
only; ungraded ones make the evidence incomplete.
"""

from dataclasses import dataclass
from enum import StrEnum

from rolloutengine.contracts import RolloutData
from taskcompendium.grading_result import GradeResult
from taskcompendium.grading_result import Outcome as GradeStatus


class TrialKind(StrEnum):
    SOLVER = "solver"
    ADVERSARY = "adversary"
    CONTROL = "control"


class Cause(StrEnum):
    """Why a trial has no grade. ``classify`` is the only function that produces one."""

    MACHINE_START = "machine_start"
    """Machine creation or file install raised."""
    MACHINE_START_TIMEOUT = "machine_start_timeout"
    MACHINE_UNSUPPORTED = "machine_unsupported"
    """The factories cannot provide what the task asks for (``task_refusals`` or ``UnsupportedMachineSpec``)."""
    TASK_SETUP = "task_setup"
    """The task's own setup failed: a setup command exited non-zero or timed out, a healthcheck never
    passed, or the environment lacks the capabilities the task requires. A task defect, not flakiness."""
    SESSION_PREPARE = "session_prepare"
    ATTEMPT_TIMEOUT = "attempt_timeout"
    """``TaskExecution.attempt_timeout`` expired."""
    AGENT_TIMEOUT = "agent_timeout"
    """``TaskExecution.agent_timeout`` expired. The rollout keeps the grade RolloutEngine gave the
    state the agent left, but the trial is not counted as graded."""
    CLEANUP = "cleanup"
    """Removing a stage's private grader files failed before the next stage could run."""
    MODEL_UNAVAILABLE = "model_unavailable"
    """``GlmClient`` spent its attempts or its infrastructure hold."""
    MODEL_REJECTED = "model_rejected"
    """The server rejected a request in a way a retry cannot fix."""
    TOOL_EXECUTION = "tool_execution"
    """A shell tool step raised (the machine failed, not the command)."""
    GENERATION_LIMIT = "generation_limit"
    """The first prompt left no generation budget, so nothing was graded."""
    TOKEN_CONTRACT = "token_contract"
    """The model transport broke the exact-token contract."""
    GRADER_RAISED = "grader_raised"
    GRADER_TIMEOUT = "grader_timeout"
    GRADER_MISSING_REWARD = "grader_missing_reward"
    GRADER_EMPTY_REWARD = "grader_empty_reward"
    GRADER_INVALID_REWARD = "grader_invalid_reward"
    GRADER_EXECUTION = "grader_execution"
    GRADER_INFRA = "grader_infra"
    """An infrastructure grade with no ``GradingFailure``."""
    INVALID_TASK = "invalid_task"
    """The grader reported the task itself invalid (TaskCompendium ``Outcome.INVALID_TASK``)."""
    VERIFIER_SKIPPED = "verifier_skipped"
    NO_GRADE = "no_grade"
    """The engine returned an unavailable grade for another reason."""
    UNCLASSIFIED = "unclassified"
    """No rule matched. Counted like every other cause, never dropped."""


RETRYABLE = frozenset(
    {
        Cause.MACHINE_START,
        Cause.MACHINE_START_TIMEOUT,
        Cause.SESSION_PREPARE,
        Cause.ATTEMPT_TIMEOUT,
        Cause.CLEANUP,
        Cause.MODEL_UNAVAILABLE,
        Cause.TOOL_EXECUTION,
        Cause.GRADER_RAISED,
        Cause.GRADER_TIMEOUT,
    }
)
"""Causes a fresh attempt can fix. The rest are properties of the task, the grader or the model."""

GRADED_STATUSES = frozenset({GradeStatus.GRADED, GradeStatus.SUBMISSION_FAILURE})


@dataclass(frozen=True)
class Graded:
    rollout: RolloutData

    def __post_init__(self) -> None:
        if self.rollout.grade.status not in GRADED_STATUSES:
            raise ValueError(f"A {self.rollout.grade.status} rollout is not graded")

    @property
    def grade(self) -> GradeResult:
        return self.rollout.grade

    @property
    def reward(self) -> float:
        """The reward; a graded result always has one, and a submission failure scores zero."""
        return 0.0 if self.grade.reward is None else self.grade.reward


@dataclass(frozen=True)
class Ungraded:
    """A trial without a grade. ``rollout`` is the last completed record, when the engine kept one."""

    cause: Cause
    detail: str
    rollout: RolloutData | None

    @property
    def retryable(self) -> bool:
        return self.cause in RETRYABLE


type Outcome = Graded | Ungraded
