# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest

from experiments.data_belay.judge import GROUP_METRICS, LadderStep, Verdict, judge, ladder_step

CONTROL = [1.000, 1.002, 0.998, 1.001]
LIMITS = {group: 0.01 for group in GROUP_METRICS}


def _arms(changes: dict[str, float]) -> tuple[dict, dict]:
    control = {group: CONTROL for group in GROUP_METRICS}
    treatment = {group: [v * (1 + changes.get(group, 0.0)) for v in CONTROL[:3]] for group in GROUP_METRICS}
    return control, treatment


@pytest.mark.parametrize(
    ("changes", "expected"),
    [
        ({"math": -0.05}, Verdict.HELPS),
        # A gain in one group does not hide a harm beyond the limit in another.
        ({"math": -0.05, "paloma": 0.03}, Verdict.HURTS),
        ({}, Verdict.NO_GAIN),
        # A harm inside the limit keeps a gain a gain.
        ({"code": -0.04, "paloma": 0.004}, Verdict.HELPS),
    ],
)
def test_verdict(changes, expected):
    control, treatment = _arms(changes)
    verdict, deltas = judge(control, treatment, LIMITS)
    assert verdict is expected
    for delta in deltas:
        assert delta.low <= changes.get(delta.group, 0.0) + 1e-9 <= delta.high + 2e-9


def test_a_gain_with_a_possible_harm_is_a_trade_off():
    control = {group: CONTROL for group in GROUP_METRICS}
    treatment = {group: [v * 0.95 for v in CONTROL[:3]] for group in GROUP_METRICS}
    # Too noisy to rule out a harm beyond the limit, and not wholly above it.
    treatment["paloma"] = [0.97, 1.03, 1.01]
    verdict, _ = judge(control, treatment, LIMITS)
    assert verdict is Verdict.TRADE_OFF


H, T, N, X = Verdict.HELPS, Verdict.TRADE_OFF, Verdict.NO_GAIN, Verdict.HURTS


@pytest.mark.parametrize(
    ("verdicts", "expected"),
    [
        # Every candidate gets a second rung, whatever the first says.
        ((X,), LadderStep.NEXT_RUNG),
        ((N, X), LadderStep.STOP),
        # One no is not enough to stop: a d512 gain that d768 does not hold goes to d1024.
        ((T, X), LadderStep.NEXT_RUNG),
        ((X, T), LadderStep.NEXT_RUNG),
        ((T, X, N), LadderStep.STOP),
        ((N, T, H), LadderStep.YES),
        ((T, T, X), LadderStep.MIXED),
        ((X, T, N), LadderStep.MIXED),
    ],
)
def test_ladder_stops_only_after_two_consecutive_nos(verdicts, expected):
    assert ladder_step(verdicts, rungs=3) is expected
