# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from collections import Counter

import pytest

from taskforge.review.decision import Reject, RejectKind, Retry
from taskforge.review.rules import BandChoice, BandRules, decide
from taskforge.validate.calibration import CalibrationBand, CalibrationSummary
from taskforge.validate.evidence import Incomplete, RewardStats
from taskforge.validate.outcome import Cause

RULES = BandRules(too_easy=BandChoice.ACCEPT, too_hard=BandChoice.REJECT)


def incomplete(causes: dict[Cause, int]) -> CalibrationSummary:
    return CalibrationSummary(
        task_digest="task1",
        policy_digest="pol",
        band=CalibrationBand(min_solve_rate=0.125, max_solve_rate=0.875),
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
