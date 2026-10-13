# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Typed trial outcomes: a trial either produced a grade or failed for a classified ``Cause``.

``Graded`` holds a rollout whose grade is a judgment of the submission: ``GRADED``, or
``SUBMISSION_FAILURE`` (the model ended without a valid submission, which scores zero). An agent that runs
out of ``TaskSessionSpec.total_turn_timeout`` is a budget stop like running out of turns, so it is
``Graded`` too (``classify.trial_outcome``), and so is a verifyit pytest grade of 0 because the candidate's own code
did not load (``Graded.candidate_code_error``): a failed attempt, flagged so the band and the summary can show it.
Everything else is ``Ungraded``: task setup, infrastructure, grader and contract failures that say nothing about the
submission. Statistics use graded outcomes only; ungraded ones make the evidence incomplete.
"""

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from rolloutengine.contracts import TOTAL_TURN_TIMEOUT_STOP_REASON, RolloutData
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
    """The factories cannot provide what the task asks for (``task_refusals`` or ``UnsupportedMachineSpec``)."""
    SUBMISSION_UNSUPPORTED = "submission_unsupported"
    """The task's answer format cannot carry its answer for its grader (TaskCompendium
    ``submission_compatibility``)."""
    TASK_SETUP = "task_setup"
    """The task's own setup failed: a setup command exited non-zero or timed out, or the environment
    lacks the capabilities the task requires. A task defect, not flakiness."""
    SESSION_PREPARE = "session_prepare"
    ATTEMPT_TIMEOUT = "attempt_timeout"
    """``TaskSessionSpec.attempt_timeout`` expired."""
    AGENT_TIMEOUT = "agent_timeout"
    """``TaskSessionSpec.total_turn_timeout`` expired before the first response, so there was nothing
    to grade. A deadline after that is a ``Graded`` trial with ``timed_out`` set."""
    MODEL_UNAVAILABLE = "model_unavailable"
    """The model client spent its attempts or its infrastructure hold."""
    MODEL_TIMEOUT = "model_timeout"
    """One model call outlived ``TaskSessionSpec.model_turn_timeout``."""
    MODEL_REJECTED = "model_rejected"
    """The server rejected a request in a way a retry cannot fix."""
    TOOL_EXECUTION = "tool_execution"
    """A shell tool step raised (the machine failed, not the command)."""
    GENERATION_LIMIT = "generation_limit"
    """The first prompt left no generation budget, so nothing was graded."""
    TOKEN_CONTRACT = "token_contract"
    """The model transport broke the exact-token contract. Not in ``RETRYABLE``: a sampled trial has
    its own retry budget (``TrialPlan.token_contract_retries``); a scripted control is deterministic
    and has none."""
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

GRADED_STATUSES = frozenset({GradeStatus.GRADED, GradeStatus.SUBMISSION_FAILURE})


@dataclass(frozen=True)
class Graded:
    rollout: RolloutData
    candidate_code_error: bool
    """A verifyit pytest grade of 0 because the candidate's own code failed to import or be collected
    (``classify.candidate_code_error``): a failed attempt, kept distinct so the band and the summary can show it."""

    def __post_init__(self) -> None:
        if self.rollout.grade.status not in GRADED_STATUSES:
            raise ValueError(f"A {self.rollout.grade.status} rollout is not graded")
        if self.candidate_code_error and self.reward != 0.0:
            raise ValueError(f"A candidate code error scores 0, not {self.reward}")

    @property
    def grade(self) -> GradeResult:
        return self.rollout.grade

    @property
    def timed_out(self) -> bool:
        """Whether the total-turn deadline ended the trial; the grade is of the state the agent left."""
        return self.rollout.stop_reason == TOTAL_TURN_TIMEOUT_STOP_REASON

    @property
    def reward(self) -> float:
        """The reward; a graded result always has one, and a submission failure scores zero."""
        return 0.0 if self.grade.reward is None else self.grade.reward


@dataclass(frozen=True)
class ContractBreak:
    """Where a sampled rollout broke the exact-token contract, with the conversation up to the break.

    The engine keeps no rollout for a contract break, so ``messages`` (the chat messages the model
    was asked to continue) are the partial rollout's only record.
    """

    turn: int
    prefix_length: int
    messages: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class Ungraded:
    """A trial without a grade. ``rollout`` is the last completed record, when the engine kept one;
    ``contract_break`` is set for a ``TOKEN_CONTRACT`` trial whose model reported where it broke."""

    cause: Cause
    detail: str
    rollout: RolloutData | None
    contract_break: ContractBreak | None

    @property
    def retryable(self) -> bool:
        return self.cause in RETRYABLE


type Outcome = Graded | Ungraded
