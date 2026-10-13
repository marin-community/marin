# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The trial evidence for one item: outcomes by trial kind, and whether every trial was graded."""

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass

from taskforge.stage50_validate.outcome import Cause, Graded, Outcome, TrialKind, Ungraded


@dataclass(frozen=True)
class Complete:
    """Every trial has a grade."""


@dataclass(frozen=True)
class Incomplete:
    """Some trials have no grade; ``causes`` counts them by cause, ``UNCLASSIFIED`` included."""

    causes: Counter[Cause]


@dataclass(frozen=True)
class RewardStats:
    """Statistics over graded outcomes only, and the attempts a broken token contract discarded.

    ``solved`` counts grades that pass: ``GradeResult.passed`` when the grader reports it, else a
    reward at the grader's ``score_max``. ``timed_out`` counts the graded trials the total-turn
    deadline ended, and ``candidate_code_errors`` those whose candidate code did not load
    (``Graded.candidate_code_error``); both are included in ``graded`` and ``mean_reward``, and a
    candidate code error is never ``solved``. ``contract_retries`` counts the attempts that broke the
    exact-token contract (``Cause.TOKEN_CONTRACT``); they are not trials, so no other field counts them.
    """

    graded: int
    mean_reward: float | None
    solved: int
    timed_out: int
    candidate_code_errors: int
    contract_retries: int


@dataclass(frozen=True)
class Evidence:
    """The final outcome of each trial by kind, and how many attempts of each kind broke the token contract.

    ``contract_retries`` comes from the attempt files (``trials.contract_retries``), since a trial's
    final outcome does not keep the attempts before it.
    """

    outcomes: Mapping[TrialKind, tuple[Outcome, ...]]
    contract_retries: Mapping[TrialKind, int]

    @property
    def status(self) -> Complete | Incomplete:
        causes = Counter(
            outcome.cause for kind in self.outcomes.values() for outcome in kind if isinstance(outcome, Ungraded)
        )
        return Incomplete(causes) if causes else Complete()

    def reward_stats(self, kind: TrialKind) -> RewardStats:
        graded = [outcome for outcome in self.outcomes.get(kind, ()) if isinstance(outcome, Graded)]
        retries = self.contract_retries.get(kind, 0)
        if not graded:
            return RewardStats(
                graded=0, mean_reward=None, solved=0, timed_out=0, candidate_code_errors=0, contract_retries=retries
            )
        solved = sum(
            (outcome.reward >= outcome.grade.score_max) if outcome.grade.passed is None else outcome.grade.passed
            for outcome in graded
        )
        return RewardStats(
            graded=len(graded),
            mean_reward=sum(outcome.reward for outcome in graded) / len(graded),
            solved=solved,
            timed_out=sum(outcome.timed_out for outcome in graded),
            candidate_code_errors=sum(outcome.candidate_code_error for outcome in graded),
            contract_retries=retries,
        )
