# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from collections import Counter
from dataclasses import dataclass

import pytest

from taskforge.review.decision import Reject, RejectKind, Retry
from taskforge.review.rules import BandChoice, BandRules, decide
from taskforge.validate.calibration import CalibrationBand, CalibrationSummary, summarize
from taskforge.validate.evidence import Incomplete, RewardStats
from taskforge.validate.outcome import Cause, Outcome, Ungraded

RULES = BandRules(too_easy=BandChoice.ACCEPT, too_hard=BandChoice.REJECT)
BAND = CalibrationBand(min_solve_rate=0.125, max_solve_rate=0.875)


def incomplete(causes: dict[Cause, int]) -> CalibrationSummary:
    return CalibrationSummary(
        task_digest="task1",
        policy_digest="pol",
        band=BAND,
        status=Incomplete(Counter(causes)),
        k=4,
        solver=RewardStats(graded=1, mean_reward=1.0, solved=1, timed_out=0),
        solve_rate=1.0,
        findings=(),
    )


@pytest.mark.parametrize(
    ("causes", "decision"),
    [
        ({Cause.MODEL_REJECTED: 2, Cause.MODEL_UNAVAILABLE: 1}, Retry(cause=Cause.MODEL_UNAVAILABLE, count=1)),
        ({Cause.MODEL_REJECTED: 3}, Reject(kind=RejectKind.TASK, reasons=("model_rejected: 3 trials ungraded",))),
    ],
)
def test_an_incomplete_round_retries_only_a_cause_a_re_run_can_change(causes, decision):
    assert decide(incomplete(causes), RULES) == decision


@dataclass(frozen=True)
class Round:
    task_digest: str
    solver: tuple[Outcome, ...]


@dataclass(frozen=True)
class Policy:
    k: int
    band: CalibrationBand
    digest: str


def test_a_round_the_grading_machine_failed_retries_rather_than_rejecting_the_task():
    failed = Ungraded(Cause.GRADER_INFRA, "Sandbox grading requires a machine factory and specification", None)

    summary = summarize(Round("task1", (failed,) * 4), Policy(k=4, band=BAND, digest="pol"))

    assert summary.findings == ()
    assert decide(summary, RULES) == Retry(cause=Cause.GRADER_INFRA, count=4)
