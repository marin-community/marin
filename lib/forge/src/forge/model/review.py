# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# Source: Taskforge integration 9807c5cfb3, stage60_review/{decision,rules}.py and stage50_validate/outcome.py.

"""Completed validation evidence and the decisions made from it."""

import math
from dataclasses import dataclass
from enum import StrEnum
from typing import Annotated, Literal

from pydantic import Field, TypeAdapter
from taskcompendium.grading_result import GradeResult

from forge.model.controls import Candidate, RewardBand


class Cause(StrEnum):
    """Why a trial has no grade. Incomplete evaluation cannot support acceptance."""

    AGENT_TIMEOUT = "agent_timeout"
    BUDGET_EXHAUSTED = "budget_exhausted"
    GENERATION_LIMIT = "generation_limit"
    GRADER_TIMEOUT = "grader_timeout"
    GRADER_MISSING_REWARD = "grader_missing_reward"
    GRADER_EMPTY_REWARD = "grader_empty_reward"
    GRADER_INVALID_REWARD = "grader_invalid_reward"
    GRADER_INVALID_CHECKS = "grader_invalid_checks"
    GRADER_REWARD_RANGE = "grader_reward_range"
    GRADER_EXECUTION = "grader_execution"
    GRADER_INFRA = "grader_infra"
    INVALID_TASK = "invalid_task"
    VERIFIER_SKIPPED = "verifier_skipped"
    NO_GRADE = "no_grade"


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
    reasons: tuple[str, ...] = ()
    revert: bool = False
    kind: Literal["accept"] = "accept"


@dataclass(frozen=True, slots=True)
class Reject:
    category: RejectKind
    reasons: tuple[str, ...]
    kind: Literal["reject"] = "reject"


@dataclass(frozen=True, slots=True)
class Modify:
    findings: tuple[str, ...]
    placement: BandOutcome | None = None
    kind: Literal["modify"] = "modify"


@dataclass(frozen=True, slots=True)
class Retry:
    cause: Cause
    count: int
    reasons: tuple[str, ...] = ()
    kind: Literal["retry"] = "retry"


@dataclass(frozen=True, slots=True)
class Incomplete:
    reasons: tuple[str, ...]
    kind: Literal["incomplete"] = "incomplete"


@dataclass(frozen=True, slots=True)
class Abandon:
    reasons: Annotated[tuple[str, ...], Field(min_length=1)]
    kind: Literal["abandon"] = "abandon"

    def __post_init__(self) -> None:
        if not self.reasons or any(not reason.strip() for reason in self.reasons):
            raise ValueError("Abandon requires non-empty reasons")


type Decision = Annotated[Accept | Reject | Modify | Retry | Incomplete | Abandon, Field(discriminator="kind")]


@dataclass(frozen=True, slots=True)
class BandRules:
    """Inclusive solve-rate band, copied from Taskforge's review rules."""

    low: float
    high: float
    accepted: frozenset[BandOutcome]

    def __post_init__(self) -> None:
        if not 0 <= self.low <= self.high <= 1:
            raise ValueError("the solve-rate band must satisfy 0 <= low <= high <= 1")
        if self.accepted - {BandOutcome.IN_BAND, BandOutcome.TOO_EASY}:
            raise ValueError("too-hard tasks cannot be accepted")

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


@dataclass(frozen=True, slots=True)
class ControlResult:
    index: int
    grade: GradeResult | None
    met: bool
    cause: Cause | None = None


class Defect(StrEnum):
    """How the grader mis-scores a candidate, relative to the reward the candidate deserves."""

    FALSE_ACCEPT = "false_accept"
    FALSE_REJECT = "false_reject"
    GRADER_CRASH = "grader_crash"


class ObservedStatus(StrEnum):
    GRADED = "graded"
    SUBMISSION_FAILURE = "submission_failure"
    GRADER_ERROR = "grader_error"


# Rewards a grader prints and the model copies back agree to this tolerance.
REWARD_TOLERANCE = 1e-6


@dataclass(frozen=True, slots=True)
class Observation:
    """What a grader did to a candidate: a reward exactly when graded."""

    status: ObservedStatus
    reward: float | None = None

    def __post_init__(self) -> None:
        if (self.status is ObservedStatus.GRADED) != (self.reward is not None):
            raise ValueError("an observed reward is given exactly when the status is graded")
        if self.reward is not None and not math.isfinite(self.reward):
            raise ValueError(f"observed reward {self.reward} is not finite")

    def matches(self, other: "Observation") -> bool:
        if self.reward is None or other.reward is None:
            return self.status is other.status
        return abs(self.reward - other.reward) <= REWARD_TOLERANCE


@dataclass(frozen=True, slots=True)
class DefectClaim:
    """What the grader DID to a candidate, kept apart from what the candidate DESERVED (#11183)."""

    defect: Defect
    observed: Observation
    deserved_reward: float | None

    def shown_by(self, observation: Observation) -> bool:
        """Whether ``observation`` is wrong in the claimed direction; a correct grade never qualifies."""
        if self.defect is Defect.GRADER_CRASH:
            return observation.status is ObservedStatus.GRADER_ERROR
        deserved = self.deserved_reward
        if deserved is None or observation.status is ObservedStatus.GRADER_ERROR:
            return False
        if observation.status is ObservedStatus.SUBMISSION_FAILURE:
            return self.defect is Defect.FALSE_REJECT and deserved > REWARD_TOLERANCE
        assert observation.reward is not None
        if self.defect is Defect.FALSE_ACCEPT:
            return observation.reward > deserved + REWARD_TOLERANCE
        return observation.reward < deserved - REWARD_TOLERANCE


@dataclass(frozen=True, slots=True)
class AdversaryFinding:
    """A replayed claim. ``expected`` is the legacy observed-grade band; findings with ``defect`` replay by direction.

    ``revision`` is the report round that produced the finding: 0 is the first report, later rounds answer replay
    feedback. Every replayed finding of every round is kept, reproduced or not.
    """

    claim: str
    candidate: Candidate
    expected: RewardBand | None
    grade: GradeResult | None
    met: bool
    cause: Cause | None = None
    defect: DefectClaim | None = None
    revision: int = 0


@dataclass(frozen=True, slots=True)
class AdversaryResult:
    name: str
    completed: bool
    findings: tuple[AdversaryFinding, ...]
    reasons: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ValidationResult:
    task_digest: str
    solver: tuple[TrialResult, ...]
    controls: tuple[ControlResult, ...]
    adversary: tuple[AdversaryResult, ...]
    lowering_error: str | None = None
    lowering_cause: Cause | None = None


@dataclass(frozen=True, slots=True)
class Budget:
    max_modify_rounds: int
    max_validation_retries: int


VALIDATION_RESULT = TypeAdapter(ValidationResult)
