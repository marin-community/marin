# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest
from verifyit.grade import (
    Component,
    ComponentRole,
    Reward,
    Status,
    aggregate_weighted,
    gates_passed,
    infra_error,
    invalid_task,
    scored,
)

GATE = Component("compiles", ComponentRole.GATE)
CORRECT = Component("correct", ComponentRole.CRITERION, weight=3.0)
STYLE = Component("style", ComponentRole.CRITERION, weight=1.0)
VERBOSE = Component("verbose", ComponentRole.PENALTY, weight=2.0)
RUBRIC = [GATE, CORRECT, STYLE, VERBOSE]


def test_weights_and_penalties_combine_over_criterion_weight():
    verdicts = {"compiles": scored(1), "correct": scored(1), "style": scored(0.5), "verbose": scored(0.25)}
    result = aggregate_weighted(RUBRIC, verdicts)
    assert result.status == Status.SCORED
    # (3 * 1 + 1 * 0.5 - 2 * 0.25) / 4
    assert result.reward == pytest.approx(0.75)
    assert result.detail["positive_sum"] == pytest.approx(3.5)
    assert result.detail["penalty_sum"] == pytest.approx(0.5)
    assert result.detail["denominator"] == pytest.approx(4.0)
    assert result.detail["failed_gates"] == []
    assert result.detail["rewards"] == {"compiles": 1.0, "correct": 1.0, "style": 0.5, "verbose": 0.25}


def test_penalties_cannot_drive_reward_below_zero():
    verdicts = {"compiles": scored(1), "correct": scored(0.5), "style": scored(0), "verbose": scored(1)}
    result = aggregate_weighted(RUBRIC, verdicts)
    assert (result.reward, result.status) == (0.0, Status.SCORED)


def test_failed_gate_zeroes_full_credit_and_skips_ungraded_components():
    full = aggregate_weighted(RUBRIC, {"compiles": scored(0.99), "correct": scored(1), "style": scored(1)})
    assert (full.reward, full.status) == (0.0, Status.SCORED)
    assert full.detail["failed_gates"] == ["compiles"]
    skipped = aggregate_weighted(RUBRIC, {"compiles": scored(0)})
    assert (skipped.reward, skipped.status) == (0.0, Status.SCORED)
    assert skipped.detail["missing"] == ["correct", "style", "verbose"]


def test_missing_components_score_zero():
    result = aggregate_weighted(RUBRIC, {"compiles": scored(1), "style": scored(1)})
    assert result.reward == pytest.approx(0.25)
    assert result.detail["missing"] == ["correct", "verbose"]
    assert result.detail["rewards"] == {"compiles": 1.0, "correct": 0.0, "style": 1.0, "verbose": 0.0}
    missing_gate = aggregate_weighted(RUBRIC, {"correct": scored(1), "style": scored(1)})
    assert missing_gate.reward == 0.0
    assert missing_gate.detail["failed_gates"] == ["compiles"]


def test_gate_weight_does_not_affect_reward():
    rubric = [Component("compiles", ComponentRole.GATE, weight=float("nan")), STYLE]
    result = aggregate_weighted(rubric, {"compiles": scored(1), "style": scored(0.5)})
    assert (result.reward, result.status) == (0.5, Status.SCORED)


def test_rounding_applies_after_aggregation():
    rubric = [Component("a", ComponentRole.CRITERION), Component("b", ComponentRole.CRITERION, weight=2)]
    result = aggregate_weighted(rubric, {"a": scored(1), "b": scored(0)}, round_digits=2)
    assert result.reward == 0.33


@pytest.mark.parametrize("failure", [invalid_task("bad reference"), infra_error("judge down")])
def test_unscored_component_discards_credit_even_after_failed_gate(failure):
    result = aggregate_weighted(RUBRIC, {"compiles": scored(0), "correct": scored(1), "verbose": failure})
    assert (result.reward, result.status) == (0.0, failure.status)
    assert result.detail["component"] == "verbose"


def test_infrastructure_error_takes_precedence_over_invalid_task():
    verdicts = {"compiles": invalid_task("bad"), "correct": scored(1), "style": infra_error("down")}
    result = aggregate_weighted(RUBRIC, verdicts)
    assert result.status == Status.INFRA_ERROR
    assert result.detail["component"] == "style"


@pytest.mark.parametrize("value", [float("nan"), 2.0, True])
def test_malformed_component_grade_is_an_infrastructure_error(value):
    result = aggregate_weighted(RUBRIC, {"compiles": scored(1), "correct": Reward(value, Status.SCORED)})
    assert (result.reward, result.status) == (0.0, Status.INFRA_ERROR)


@pytest.mark.parametrize("weight", [0, -1.0, float("inf"), float("nan"), True])
@pytest.mark.parametrize("role", [ComponentRole.CRITERION, ComponentRole.PENALTY])
def test_invalid_weight_is_an_invalid_task(weight, role):
    rubric = [STYLE, Component("x", role, weight=weight)]
    result = aggregate_weighted(rubric, {"style": scored(1), "x": scored(0)})
    assert (result.reward, result.status) == (0.0, Status.INVALID_TASK)


@pytest.mark.parametrize(
    "rubric,verdicts",
    [
        ([STYLE, Component("style", ComponentRole.PENALTY)], {"style": scored(1)}),
        ([GATE, VERBOSE], {"compiles": scored(1), "verbose": scored(0)}),
        ([GATE, STYLE], {"compiles": scored(1), "style": scored(1), "extra": scored(1)}),
    ],
    ids=["duplicate_name", "no_criterion", "undeclared_grade"],
)
def test_malformed_rubric_is_an_invalid_task(rubric, verdicts):
    result = aggregate_weighted(rubric, verdicts)
    assert (result.reward, result.status) == (0.0, Status.INVALID_TASK)


@pytest.mark.parametrize(
    "gate,expected",
    [
        (scored(1), True),
        (scored(0.5), False),
        (infra_error("down"), False),
        (Reward(float("nan"), Status.SCORED), False),
        (None, False),
    ],
)
def test_gates_passed_requires_every_gate_at_full_credit(gate, expected):
    verdicts = {} if gate is None else {"compiles": gate}
    assert gates_passed(RUBRIC, verdicts) is expected


def test_gates_passed_ignores_ungraded_criteria():
    assert gates_passed(RUBRIC, {"compiles": scored(1)})


@pytest.mark.parametrize(
    "rubric",
    [[GATE, Component("compiles", ComponentRole.CRITERION)], [GATE, VERBOSE]],
    ids=["duplicate_name", "no_criterion"],
)
def test_gates_passed_rejects_malformed_rubric(rubric):
    assert not gates_passed(rubric, {"compiles": scored(1)})
