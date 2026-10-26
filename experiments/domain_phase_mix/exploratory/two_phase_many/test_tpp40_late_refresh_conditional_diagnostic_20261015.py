# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Pure equation, derivative, feasibility, and lower-bound behavior tests."""

from dataclasses import replace

import numpy as np
import pytest
from scipy.optimize import linprog

from experiments.domain_phase_mix.exploratory.two_phase_many import tpp40_late_refresh_20261015 as law
from experiments.domain_phase_mix.exploratory.two_phase_many import (
    tpp40_late_refresh_conditional_diagnostic_20261015 as diagnostic,
)


def toy_head(power: float, strength: float) -> diagnostic.TaskHead:
    return diagnostic.TaskHead(
        rate=0.25,
        power=power,
        threshold=3.0,
        strength=strength,
        intercept=0.2,
        benefit=np.array([0.8, 1.1]),
        harm=np.array([0.03, 0.07]),
        floor=0.6,
    )


def frozen_logits(head: diagnostic.TaskHead, total: np.ndarray, late: np.ndarray, beta: float) -> np.ndarray:
    matrix, _ = law.design_and_offset(
        np.broadcast_to(total, late.shape) - late,
        late,
        1 - beta,
        {"rate": head.rate, "power": head.power, "threshold": head.threshold},
        "late_refresh_hazard",
        {"lambda": head.strength},
    )
    return head.intercept + matrix @ np.r_[head.benefit, head.harm]


@pytest.mark.parametrize("power", [0.3, 0.7, 1.0])
def test_analytic_gradient_matches_frozen_law_finite_difference(power: float) -> None:
    total, late, beta = np.array([10.0, 1.0]), np.array([1.1, 0.29]), 0.2
    head = toy_head(power, 0.5)
    point = diagnostic.task_point(head, total, late, beta)
    epsilon = 1e-5
    perturbation = epsilon * np.eye(2)
    plus = frozen_logits(head, total, late + perturbation, beta)
    minus = frozen_logits(head, total, late - perturbation, beta)
    finite_difference = (plus - minus) / (2 * epsilon)
    np.testing.assert_allclose(point.logit_gradient, finite_difference, rtol=5e-8, atol=2e-10)
    np.testing.assert_allclose(point.logit, frozen_logits(head, total, late[None], beta)[0], atol=3e-15, rtol=0)
    loss_difference = (np.exp(plus) - np.exp(minus)) / (2 * epsilon)
    np.testing.assert_allclose(np.exp(point.logit) * point.logit_gradient, loss_difference, rtol=5e-8, atol=2e-10)


def test_capacitated_linear_dual_matches_independent_linear_program() -> None:
    gradient = np.array([-3.0, -3.0, -1.0, 0.7])
    cost = np.array([0.5, 0.5, 0.2, 0.9])
    upper = np.array([0.4, 0.8, 1.1, 0.5])
    budget = 0.7
    result = diagnostic.linear_minimum(gradient, upper, cost, budget)
    reference = linprog(gradient, A_eq=cost[None], b_eq=[budget], bounds=list(zip(np.zeros(4), upper, strict=True)))
    assert reference.success
    assert result.dual_value == pytest.approx(reference.fun, abs=2e-14)
    assert result.primal_value == pytest.approx(reference.fun, abs=2e-14)
    assert abs(result.budget_residual) < 2e-15
    assert abs(result.primal_minus_dual) < 2e-14
    # Equal marginal slopes consume coordinate zero first, without changing the bound.
    tied = diagnostic.linear_minimum(gradient[:2], upper[:2], cost[:2], 0.15)
    np.testing.assert_allclose(tied.allocation, [0.3, 0.0], atol=1e-15, rtol=0)


def test_weighted_task_bounds_are_below_dense_feasible_fiber() -> None:
    heads = (toy_head(0.3, 0.5), replace(toy_head(1.0, 0.9), rate=1.3, benefit=np.array([1.2, 0.6])))
    weights = np.array([0.3, 0.7])
    total, cost, beta = np.array([10.0, 1.0]), np.array([0.05, 0.5]), 0.2
    reference = np.array([1.0, 0.3])
    result = diagnostic.conditional_diagnostic(
        heads, weights, total, reference, cost, beta, 1.0, interior_fraction=1e-6, feasibility_tolerance=1e-10
    )
    first = np.linspace(0, 4, 2001)
    late = np.column_stack([first, (beta - cost[0] * first) / cost[1]])
    losses = np.column_stack(
        [head.floor + np.exp(np.clip(frozen_logits(head, total, late, beta), -30, 30)) for head in heads]
    )
    weighted = result["weighted"]
    assert result["bounds_available"]
    assert weighted["unclipped_support_is_native_lower_bound"]
    assert weighted["native_combined_lower_bound"] <= np.min(losses @ weights) + 2e-14
    for index, task in enumerate(result["task_results"]):
        assert task["native_loss_lower_bound"] <= np.min(losses[:, index]) + 2e-14
    gradient = np.asarray(weighted["unclipped_reference_gradient"])
    steps = np.array([1.0, -cost[0] / cost[1]]) * 1e-5
    observed = []
    for point in (reference + steps, reference - steps):
        observed.append(
            sum(
                weight * (head.floor + np.exp(frozen_logits(head, total, point[None], beta)[0]))
                for head, weight in zip(heads, weights, strict=True)
            )
        )
    assert (observed[0] - observed[1]) / 2 == pytest.approx(float(gradient @ steps), abs=2e-14)


def test_singular_dropout_interiorization_and_zero_coordinate_semantics() -> None:
    total, late, cost, beta = np.array([1.0, 1.0, 0.0]), np.array([0.0, 0.4, 0.0]), np.full(3, 0.5), 0.2
    head = replace(toy_head(0.7, 0.5), benefit=np.array([1.0, 0.0, 5.0]), harm=np.zeros(3))
    point = diagnostic.task_point(head, total, late, beta)
    assert point.singular_indices == (0,)
    assert point.logit_gradient[0] == -np.inf
    np.testing.assert_array_equal(point.logit_gradient[1:], [0.0, 0.0])
    result = diagnostic.conditional_diagnostic(
        (head,), np.ones(1), total, late, cost, beta, 1, interior_fraction=1e-6, feasibility_tolerance=1e-10
    )
    assert result["interiorization_applied"]
    reference = np.asarray(result["reference_late"])
    assert reference[0] > 0
    assert reference[2] == 0
    assert float(cost @ reference) == pytest.approx(beta, abs=1e-15)
    assert np.isfinite(result["task_results"][0]["reference"]["logit_gradient"]).all()
    assert result["weighted"]["native_remaining_improvement_upper_estimate"] >= 0
    null = diagnostic.task_point(replace(head, strength=0), total, late, beta)
    np.testing.assert_array_equal(null.logit_gradient, np.zeros(3))
    linear_power = replace(head, power=1.0)
    finite_zero = diagnostic.task_point(linear_power, total, late, beta)
    step = np.array([1e-7, 0, 0])
    difference = (diagnostic.task_point(linear_power, total, late + step, beta).logit - finite_zero.logit) / step[0]
    assert finite_zero.logit_gradient[0] == pytest.approx(difference, rel=3e-7)


def test_native_null_has_zero_conditional_gap() -> None:
    total, cost, late = np.array([10.0, 1.0]), np.array([0.05, 0.5]), np.array([0.0, 0.4])
    result = diagnostic.conditional_diagnostic(
        (toy_head(0.7, 0.0),),
        np.ones(1),
        total,
        late,
        cost,
        0.2,
        1,
        interior_fraction=1e-6,
        feasibility_tolerance=1e-10,
    )
    assert not result["interiorization_applied"]
    assert abs(result["weighted"]["native_remaining_improvement_upper_estimate"]) < 2e-15


@pytest.mark.parametrize("intercept", [31.0, -35.0])
def test_clipping_keeps_native_and_unclipped_support_distinct(intercept: float) -> None:
    head = replace(toy_head(0.7, 0.5), intercept=intercept)
    total, cost, beta = np.array([10.0, 1.0]), np.array([0.05, 0.5]), 0.2
    result = diagnostic.conditional_diagnostic(
        (head,),
        np.ones(1),
        total,
        np.array([1.0, 0.3]),
        cost,
        beta,
        1,
        interior_fraction=1e-6,
        feasibility_tolerance=1e-10,
    )
    weighted = result["weighted"]
    if intercept > 0:
        assert not weighted["upper_cap_excluded_over_full_box"]
        assert not weighted["unclipped_support_is_native_lower_bound"]
        assert weighted["native_combined_lower_bound"] == weighted["native_taskwise_lower_bound"]
    else:
        assert weighted["unclipped_support_is_native_lower_bound"]
        assert weighted["native_reference_value"] > weighted["unclipped_reference_value"]
    first = np.linspace(0, 4, 1001)
    late = np.column_stack([first, (beta - cost[0] * first) / cost[1]])
    native = head.floor + np.exp(np.clip(frozen_logits(head, total, late, beta), -30, 30))
    assert weighted["native_combined_lower_bound"] <= np.min(native) * (1 + 3e-15)


def test_exposure_cost_derivation_and_infeasible_policy_gets_no_gap_claim() -> None:
    alpha = 0.8
    scales = np.array([20.0, 2.0])
    costs = diagnostic.normalized_exposure_cost(alpha * scales, (1 - alpha) * scales, alpha)
    phase0, phase1 = np.array([0.5, 0.5]), np.array([0.2, 0.8])
    total = scales * (alpha * phase0 + (1 - alpha) * phase1)
    late = scales * (1 - alpha) * phase1
    assert float(costs @ total) == pytest.approx(1, abs=1e-15)
    assert float(costs @ late) == pytest.approx(1 - alpha, abs=1e-15)
    valid = diagnostic.conditional_diagnostic(
        (toy_head(0.7, 0.5),),
        np.ones(1),
        total,
        late,
        costs,
        1 - alpha,
        1,
        interior_fraction=1e-6,
        feasibility_tolerance=1e-10,
    )
    assert valid["bounds_available"]
    invalid = diagnostic.conditional_diagnostic(
        (toy_head(0.7, 0.5),),
        np.ones(1),
        total,
        late + np.array([1e-5, 0]),
        costs,
        1 - alpha,
        1,
        interior_fraction=1e-6,
        feasibility_tolerance=1e-10,
    )
    assert not invalid["bounds_available"]
    assert not invalid["feasible_at_tolerance"]
    assert "weighted" not in invalid
