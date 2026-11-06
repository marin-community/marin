# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# Source: Taskforge integration 9807c5cfb3, stage60_review/{decision,rules}.py and stage50_validate/outcome.py.

"""Completed validation evidence and the decisions made from it."""

from copy import deepcopy
from dataclasses import dataclass, replace
from enum import StrEnum

from taskcompendium.grading_result import GradeResult

from forge.model.controls import Candidate, RewardBand


class Cause(StrEnum):
    """Why a trial has no grade. Incomplete evaluation cannot support acceptance."""

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
    SEARCH_INCOMPLETE = "search_incomplete"
    """The finder did not complete; preserve its evidence for resumption."""
    GENERATION_LIMIT = "generation_limit"
    """A physical context or output boundary interrupted generation."""
    GRADER_RAISED = "grader_raised"
    GRADER_TIMEOUT = "grader_timeout"
    GRADER_MISSING_REWARD = "grader_missing_reward"
    GRADER_EMPTY_REWARD = "grader_empty_reward"
    GRADER_INVALID_REWARD = "grader_invalid_reward"
    GRADER_INVALID_CHECKS = "grader_invalid_checks"
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


class BandOutcome(StrEnum):
    IN_BAND = "in_band"
    TOO_EASY = "too_easy"
    TOO_HARD = "too_hard"


class RejectKind(StrEnum):
    TASK = "task"
    BUDGET = "budget"
    HOST = "host"


@dataclass(frozen=True, slots=True)
class Accept:
    band: BandOutcome


@dataclass(frozen=True, slots=True)
class Reject:
    kind: RejectKind
    reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Modify:
    findings: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Retry:
    cause: Cause
    count: int


@dataclass(frozen=True, slots=True)
class Incomplete:
    reasons: tuple[str, ...]


type Decision = Accept | Reject | Modify | Retry | Incomplete


@dataclass(frozen=True, slots=True)
class BandRules:
    """Inclusive solve-rate band, copied from Taskforge's review rules."""

    low: float
    high: float
    accepted: frozenset[BandOutcome]

    def __post_init__(self) -> None:
        if not 0 <= self.low <= self.high <= 1:
            raise ValueError("the solve-rate band must satisfy 0 <= low <= high <= 1")
        if self.accepted - {BandOutcome.IN_BAND}:
            raise ValueError("only completed in-band evidence can be accepted")

    def place(self, solve_rate: float) -> BandOutcome:
        if solve_rate < self.low:
            return BandOutcome.TOO_HARD
        if solve_rate > self.high:
            return BandOutcome.TOO_EASY
        return BandOutcome.IN_BAND


@dataclass(frozen=True, slots=True)
class TrialResult:
    trial: int
    grade: GradeResult | None
    cause: Cause | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "grade", deepcopy(self.grade))


@dataclass(frozen=True, slots=True)
class ControlResult:
    index: int
    grade: GradeResult | None
    met: bool
    cause: Cause | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "grade", deepcopy(self.grade))


@dataclass(frozen=True, slots=True)
class AdversaryFinding:
    claim: str
    candidate: Candidate
    expected: RewardBand
    grade: GradeResult | None
    met: bool
    cause: Cause | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "candidate", replace(self.candidate))
        object.__setattr__(self, "grade", deepcopy(self.grade))


@dataclass(frozen=True, slots=True)
class AdversaryResult:
    name: str
    completed: bool
    findings: tuple[AdversaryFinding, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "findings", tuple(replace(finding) for finding in self.findings))


@dataclass(frozen=True, slots=True)
class ValidationResult:
    task_digest: str
    solver: tuple[TrialResult, ...]
    controls: tuple[ControlResult, ...]
    adversary: tuple[AdversaryResult, ...]
    lowering_error: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "solver", tuple(replace(result) for result in self.solver))
        object.__setattr__(self, "controls", tuple(replace(result) for result in self.controls))
        object.__setattr__(self, "adversary", tuple(replace(result) for result in self.adversary))


@dataclass(frozen=True, slots=True)
class Budget:
    max_modify_rounds: int
    max_validation_retries: int
