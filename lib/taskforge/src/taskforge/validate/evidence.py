# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The trial evidence for one item: outcomes by trial kind, and whether every trial was graded."""

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass

from taskforge.validate.outcome import Cause, Graded, Outcome, TrialKind, Ungraded


@dataclass(frozen=True)
class Complete:
    """Every trial has a grade."""


@dataclass(frozen=True)
class Incomplete:
    """Some trials have no grade; ``causes`` counts them by cause, ``UNCLASSIFIED`` included."""

    causes: Counter[Cause]


@dataclass(frozen=True)
class RewardStats:
    """Statistics over graded outcomes only.

    ``solved`` counts grades that pass: ``GradeResult.passed`` when the grader reports it, else a
    reward at the grader's ``score_max``.
    """

    graded: int
    mean_reward: float | None
    solved: int


@dataclass(frozen=True)
class Evidence:
    outcomes: Mapping[TrialKind, tuple[Outcome, ...]]

    @property
    def status(self) -> Complete | Incomplete:
        causes = Counter(
            outcome.cause for kind in self.outcomes.values() for outcome in kind if isinstance(outcome, Ungraded)
        )
        return Incomplete(causes) if causes else Complete()

    def reward_stats(self, kind: TrialKind) -> RewardStats:
        graded = [outcome for outcome in self.outcomes.get(kind, ()) if isinstance(outcome, Graded)]
        if not graded:
            return RewardStats(graded=0, mean_reward=None, solved=0)
        solved = sum(
            (outcome.reward >= outcome.grade.score_max) if outcome.grade.passed is None else outcome.grade.passed
            for outcome in graded
        )
        return RewardStats(
            graded=len(graded), mean_reward=sum(outcome.reward for outcome in graded) / len(graded), solved=solved
        )
