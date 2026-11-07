# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Authored candidates and their expected rewards; reward checks copied from Taskforge."""

import math
from dataclasses import dataclass
from enum import StrEnum

from taskcompendium.grading_result import GradeResult, Outcome
from taskcompendium.models import TaskResource


class ControlKind(StrEnum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    MALFORMED = "malformed"


EXPECTED_STATUSES = frozenset({Outcome.GRADED, Outcome.SUBMISSION_FAILURE})


@dataclass(frozen=True, slots=True)
class RewardBand:
    """The grade a control must receive.

    A graded expectation with only an upper bound, or with a lower bound of 0.0 that
    every reward meets, is a no-credit expectation: a submission failure, which
    scores zero, meets it as well as a graded reward within the bound.
    """

    status: Outcome
    reward_min: float | None = None
    reward_max: float | None = None

    def __post_init__(self) -> None:
        if self.status not in EXPECTED_STATUSES:
            raise ValueError(f"A control expects a graded result or a submission failure, not {self.status}")
        bounds = (self.reward_min, self.reward_max)
        if self.status == Outcome.SUBMISSION_FAILURE and bounds != (None, None):
            raise ValueError("A submission failure should not have reward bounds")
        for bound in bounds:
            if bound is not None and not (math.isfinite(bound) and 0 <= bound <= 1):
                raise ValueError(f"Reward bound {bound} is outside [0, 1]")
        if self.reward_min is not None and self.reward_max is not None and self.reward_min > self.reward_max:
            raise ValueError("Reward bounds are reversed")

    @property
    def no_credit(self) -> bool:
        """Whether the expectation asks only that the submission earn at most ``reward_max``."""
        return self.status == Outcome.GRADED and self.reward_min in (None, 0.0) and self.reward_max is not None

    def met_by(self, grade: GradeResult) -> bool:
        """Whether ``grade`` meets this expectation."""
        if self.no_credit and grade.status == Outcome.SUBMISSION_FAILURE:
            return True
        if grade.status != self.status:
            return False
        if self.status != Outcome.GRADED:
            return True
        assert grade.reward is not None
        if self.reward_min is not None and grade.reward < self.reward_min:
            return False
        return self.reward_max is None or grade.reward <= self.reward_max


@dataclass(frozen=True, slots=True)
class Candidate:
    reply: str
    files: tuple[TaskResource, ...] = ()


@dataclass(frozen=True, slots=True)
class Control:
    candidate: Candidate
    expected: RewardBand
    kind: ControlKind
