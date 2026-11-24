# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest

from experiments.data_belay.judge import (
    GROUP_METRICS,
    Change,
    GroupCall,
    GroupDelta,
    LadderStep,
    RungResult,
    Verdict,
    group_values,
    harm_limits,
    judge,
    ladder_step,
    verdict,
)

CONTROL = [1.000, 1.002, 0.998, 1.001]
REFERENCES = [1.000, 1.003, 0.997]
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
def test_verdict_without_dilution_references(changes, expected):
    control, treatment = _arms(changes)
    result = judge(control, treatment, LIMITS, None)
    assert result.verdict is expected
    for delta in result.groups:
        assert delta.gain.low <= changes.get(delta.group, 0.0) + 1e-9 <= delta.gain.high + 2e-9
        assert delta.harm == delta.gain


def test_a_gain_under_the_minimum_is_not_a_gain_even_when_its_interval_clears_0():
    control, treatment = _arms({"math": -0.006})
    result = judge(control, treatment, LIMITS, None)
    math = next(d for d in result.groups if d.group == "math")
    assert math.gain.high < 0
    assert result.verdict is Verdict.NO_GAIN


def test_a_cost_that_the_dilution_references_share_is_not_a_harm():
    # Giving up the share of the base mix costs Paloma 3% on its own, and the treatment costs it no more than that.
    control, treatment = _arms({"math": -0.05, "paloma": 0.03})
    dilution = {group: [v * (1.03 if group == "paloma" else 1.0) for v in REFERENCES] for group in GROUP_METRICS}
    assert judge(control, treatment, LIMITS, dilution).verdict is Verdict.HELPS


def test_beating_only_the_dilution_references_is_not_a_gain():
    control, treatment = _arms({})
    dilution = {group: [v * 1.03 for v in REFERENCES] for group in GROUP_METRICS}
    assert judge(control, treatment, LIMITS, dilution).verdict is Verdict.NO_GAIN


def test_a_gain_with_a_possible_harm_is_a_trade_off():
    control = {group: CONTROL for group in GROUP_METRICS}
    treatment = {group: [v * 0.95 for v in CONTROL[:3]] for group in GROUP_METRICS}
    # Too noisy to rule out a harm beyond the limit, and not wholly above it.
    treatment["paloma"] = [0.97, 1.03, 1.01]
    assert judge(control, treatment, LIMITS, None).verdict is Verdict.TRADE_OFF


def test_one_cost_is_a_harm_where_the_control_seeds_agree_and_not_where_they_vary():
    # Paloma's control seeds vary 0.17%, so its limit is the 1% floor. Code's vary 2.5%, a limit of 6.2%, which the
    # interval of a 2% cost clears.
    control = {group: CONTROL for group in GROUP_METRICS} | {"code": [1.00, 1.03, 0.97, 1.01]}
    treatment = {group: [v * 1.02 for v in CONTROL[:3]] for group in GROUP_METRICS}
    calls = {d.group: d.call for d in judge(control, treatment, harm_limits(control), None).groups}
    assert calls["paloma"] is GroupCall.WORSE
    assert calls["code"] is GroupCall.NEUTRAL


def test_knowledge_weighs_choice_loss_and_answer_bpb_by_their_change_not_their_size():
    # Choice loss is about 20 bits per question and TriviaQA's answer BPB about 2.5, in a control run.
    control = {"eval/eval_policy/knowledge/trivia_qa/bpb": 2.5, "choice/mmlu_pro": 25.0, "choice/truthful_qa": 20.0}
    run = {key: 2.0 for keys in GROUP_METRICS.values() for key in keys} | control
    choice_gain = group_values(run | {"choice/mmlu_pro": 25.0 * 0.97}, control)
    bpb_gain = group_values(run | {"eval/eval_policy/knowledge/trivia_qa/bpb": 2.5 * 0.97}, control)
    assert choice_gain["knowledge"] == pytest.approx(bpb_gain["knowledge"]) == pytest.approx(1 - 0.03 / 3)
    # The groups that average BPB alone are not scaled.
    assert choice_gain["paloma"] == 2.0


def _rung(harm_offset: float = 0.0, **deltas: float) -> RungResult:
    """A rung whose groups moved by ``deltas`` against the control (the rest by +0.2%), each ±0.5%, and by
    ``harm_offset`` more against the dilution references."""
    groups = tuple(
        GroupDelta(group, Change(d, d - 0.005, d + 0.005), Change(h, h - 0.005, h + 0.005), 0.01)
        for group in GROUP_METRICS
        for d in [deltas.get(group, 0.002)]
        for h in [d + harm_offset]
    )
    return RungResult(verdict(groups), groups)


H = _rung(math=-0.05)
T = _rung(math=-0.05, paloma=0.012)
N = _rung()
X = _rung(paloma=0.03)
# 'no gain', with code below 0 against the control, inside its interval.
N_CODE = _rung(code=-0.003)


@pytest.mark.parametrize(
    ("results", "expected"),
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
        # Two nos with a group below 0 at the second go on while a rung is left, and stop at a third no.
        ((N, N_CODE), LadderStep.NEXT_RUNG),
        ((N, N_CODE, N), LadderStep.STOP),
        # A yes after them still needs the rung below to say yes too.
        ((N, N_CODE, T), LadderStep.MIXED),
        # A better group in a rung that hurts is below 0 too.
        ((N, _rung(math=-0.05, code=0.03)), LadderStep.NEXT_RUNG),
        # Below 0 against the dilution references alone is not a gain: every group costs 2% against the control.
        ((N, _rung(harm_offset=-0.025, **dict.fromkeys(GROUP_METRICS, 0.02))), LadderStep.STOP),
    ],
)
def test_ladder_steps(results, expected):
    assert ladder_step(results, rungs=3) is expected


def test_a_group_below_0_does_not_climb_past_the_top_rung():
    assert ladder_step((N, N_CODE), rungs=2) is LadderStep.STOP
