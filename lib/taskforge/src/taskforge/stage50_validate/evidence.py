# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The trial evidence for one workpiece: outcomes by trial kind, and whether every trial was graded."""

import math
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass

from taskcompendium.grading_result import GradeResult

from taskforge.stage50_validate.outcome import GRADED_STATUSES, Cause, Graded, Outcome, TrialKind, Ungraded


def passed(grade: GradeResult) -> bool:
    """The grader's pass flag, or else a reward at the scale's maximum, read within float rounding.

    A grader that sums partial credits can land at 0.9999999999999999 for a full solve.
    """
    if grade.status not in GRADED_STATUSES:
        return False
    if grade.passed is not None:
        return grade.passed
    reward, score_max = grade.reward or 0.0, grade.score_max
    return reward >= score_max or math.isclose(reward, score_max, rel_tol=1e-9, abs_tol=1e-9)


@dataclass(frozen=True)
class RewardStats:
    """Statistics over graded outcomes only.

    ``solved`` counts grades that pass: ``GradeResult.passed`` when the grader reports it, else a
    reward at the grader's ``score_max``. ``timed_out`` counts the graded trials the total-turn
    deadline ended, and ``candidate_code_errors`` those whose candidate code did not load
    (``Graded.candidate_code_error``); both are included in ``graded`` and ``mean_reward``, and a
    candidate code error is never ``solved``.
    """

    graded: int
    mean_reward: float | None
    solved: int
    timed_out: int
    candidate_code_errors: int


@dataclass(frozen=True)
class Evidence:
    """The final outcome of each trial by kind."""

    outcomes: Mapping[TrialKind, tuple[Outcome, ...]]

    @property
    def ungraded(self) -> Counter[Cause]:
        return Counter(
            outcome.cause for kind in self.outcomes.values() for outcome in kind if isinstance(outcome, Ungraded)
        )

    def reward_stats(self, kind: TrialKind) -> RewardStats:
        """Reward, solve and failure counts over the graded outcomes of ``kind``."""
        graded = [outcome for outcome in self.outcomes.get(kind, ()) if isinstance(outcome, Graded)]
        if not graded:
            return RewardStats(graded=0, mean_reward=None, solved=0, timed_out=0, candidate_code_errors=0)
        solved = sum(passed(outcome.grade) for outcome in graded)
        return RewardStats(
            graded=len(graded),
            mean_reward=sum(outcome.reward for outcome in graded) / len(graded),
            solved=solved,
            timed_out=sum(outcome.timed_out for outcome in graded),
            candidate_code_errors=sum(outcome.candidate_code_error for outcome in graded),
        )
