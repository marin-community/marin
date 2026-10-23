# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Typed outcomes distinguish usable grades from incomplete evaluation.

GRADED and SUBMISSION_FAILURE carry completed evaluations; submission failures
score zero. A candidate-code failure is a graded zero with a separate diagnostic
flag. Timeouts, generation limits, setup and grading failures remain Ungraded.
Statistics use graded outcomes only; incomplete outcomes prevent a final verdict.
"""

from dataclasses import dataclass
from enum import StrEnum

from rolloutengine.models import TOTAL_TURN_TIMEOUT_STOP_REASON, Rollout
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
    MACHINE_TERMINATED = "machine_terminated"
    """The machine ended under the engine: killed, expired, preempted or lost with its host
    (shellbox ``MachineTerminated``)."""
    MACHINE_UNSUPPORTED = "machine_unsupported"
    """A factory is missing or rejects the task's machine, before the trial (``task_refusals``) or while
    it runs (``UnsupportedMachineSpec``)."""
    SUBMISSION_UNSUPPORTED = "submission_unsupported"
    """The task's answer format cannot carry its answer for its grader (TaskCompendium
    ``submission_compatibility``)."""
    TASK_SETUP = "task_setup"
    """The task's own setup failed: a setup command exited non-zero or timed out, or the environment
    lacks the capabilities the task requires. A task defect, not flakiness."""
    SESSION_PREPARE = "session_prepare"
    """The harness failed, or returned, before its first model turn (RolloutEngine's ``PREPARE``)."""
    ATTEMPT_TIMEOUT = "attempt_timeout"
    """``RolloutLimits.attempt_timeout`` expired, during machine start, the turns or grading. The engine does
    not grade such an attempt, also when the harness ignored the deadline's cancellation and ended after it."""
    AGENT_TIMEOUT = "agent_timeout"
    """``RolloutLimits.total_turn_timeout`` expired before the first response, so there was nothing
    to grade. A deadline after that is a ``Graded`` trial with ``timed_out`` set."""
    MODEL_UNAVAILABLE = "model_unavailable"
    """The model client spent its attempts or its infrastructure hold."""
    MODEL_TIMEOUT = "model_timeout"
    """One model call outlived ``RolloutLimits.model_turn_timeout``."""
    MODEL_REJECTED = "model_rejected"
    """The server rejected a request in a way a retry cannot fix."""
    TOOL_EXECUTION = "tool_execution"
    """A shell tool step raised (the machine failed, not the command), or outlived
    ``RolloutLimits.tool_turn_timeout``."""
    BUDGET_EXHAUSTED = "budget_exhausted"
    """An explicitly configured agent turn budget ended evaluation before completion."""
    GENERATION_LIMIT = "generation_limit"
    """A physical context or output boundary interrupted generation."""
    GRADER_RAISED = "grader_raised"
    GRADER_TIMEOUT = "grader_timeout"
    GRADER_MISSING_REWARD = "grader_missing_reward"
    GRADER_EMPTY_REWARD = "grader_empty_reward"
    GRADER_INVALID_REWARD = "grader_invalid_reward"
    GRADER_REWARD_RANGE = "grader_reward_range"
    """The grader returned a reward outside its declared score range: a grader defect, and the trial
    has no usable grade."""
    GRADER_EXECUTION = "grader_execution"
    GRADER_INFRA = "grader_infra"
    """An infrastructure grade with no ``GradingFailure``."""
    OUTPUT_CAPTURE = "output_capture"
    """The engine could not capture the task's declared output paths before grading: over the capture
    budget (1,024 files, 16 MiB), a file over the per-file limit, or a symlink among them. The agent
    wrote them, so the trial counts against the task and is not retried."""
    INVALID_TASK = "invalid_task"
    """The grader reported the task itself invalid (TaskCompendium ``Outcome.INVALID_TASK``)."""
    VERIFIER_SKIPPED = "verifier_skipped"
    """The task has no runnable grader (TaskCompendium ``NoGrader``), so its grade is unavailable."""
    NO_GRADE = "no_grade"
    """The engine returned an unavailable grade for another reason."""
    UNCLASSIFIED = "unclassified"
    """No rule matched. Counted like every other cause, never dropped."""


RETRYABLE = frozenset(
    {
        Cause.MACHINE_START,
        Cause.MACHINE_START_TIMEOUT,
        Cause.MACHINE_TERMINATED,
        Cause.SESSION_PREPARE,
        Cause.ATTEMPT_TIMEOUT,
        Cause.MODEL_UNAVAILABLE,
        Cause.MODEL_TIMEOUT,
        Cause.TOOL_EXECUTION,
        Cause.GRADER_RAISED,
        Cause.GRADER_TIMEOUT,
    }
)
"""Causes a fresh attempt can fix. The rest are properties of the task, the grader or the model."""

TASK_DEFECTS = frozenset(
    {
        Cause.TASK_SETUP,
        Cause.SUBMISSION_UNSUPPORTED,
        Cause.GRADER_MISSING_REWARD,
        Cause.GRADER_EMPTY_REWARD,
        Cause.GRADER_INVALID_REWARD,
        Cause.GRADER_REWARD_RANGE,
        Cause.INVALID_TASK,
        Cause.VERIFIER_SKIPPED,
    }
)
"""Ungraded causes that establish an authored defect. All other causes require a hold, not a task verdict."""

GRADED_STATUSES = frozenset({GradeStatus.GRADED, GradeStatus.SUBMISSION_FAILURE})


@dataclass(frozen=True)
class Graded:
    rollout: Rollout
    grade: GradeResult
    candidate_code_error: bool
    """A verifyit pytest grade of 0 because the candidate's own code failed to import or be collected
    (``classify.candidate_code_error``): a failed attempt, kept distinct so the band and the summary can show it."""

    def __post_init__(self) -> None:
        if self.grade.status not in GRADED_STATUSES:
            raise ValueError(f"A {self.grade.status} rollout is not graded")
        if self.candidate_code_error and self.reward != 0.0:
            raise ValueError(f"A candidate code error scores 0, not {self.reward}")

    @property
    def timed_out(self) -> bool:
        """Whether the total-turn deadline ended the trial; the grade is of the state the agent left."""
        return self.rollout.stop_reason == TOTAL_TURN_TIMEOUT_STOP_REASON

    @property
    def reward(self) -> float:
        """The reward; a graded result always has one, and a submission failure scores zero."""
        return 0.0 if self.grade.reward is None else self.grade.reward


@dataclass(frozen=True)
class Ungraded:
    """A trial without a usable grade.

    ``rollout`` is the record the engine kept, and ``grade`` the grade it gave: an unusable one, the grade of
    the completed turns of an interrupted attempt, or an unavailable grade when it stopped before grading.
    Both are ``None`` when the engine kept no record.
    """

    cause: Cause
    detail: str
    rollout: Rollout | None
    grade: GradeResult | None

    @property
    def retryable(self) -> bool:
        return self.cause in RETRYABLE


type Outcome = Graded | Ungraded
