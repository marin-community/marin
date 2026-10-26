# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
import json
from dataclasses import replace

import numpy as np
import pytest

from experiments.domain_phase_mix.exploratory.two_phase_many import run_tpp40_readout_isolation_20261015 as model
from experiments.domain_phase_mix.exploratory.two_phase_many import tpp40_causal_acquisition_retention_20261015 as state
from experiments.domain_phase_mix.exploratory.two_phase_many.test_tpp40_causal_acquisition_retention_20261015 import (
    ode_mastery,
)

CONTROL = state.Quadrature(1e-12, 512)


@pytest.mark.parametrize("mu,timing", [(0.0, "early_heavy"), (2.1, "tied")])
def test_toy_cell_receipt_records_physical_budget_and_native_departure_checks(mu: float, timing: str) -> None:
    case = model.Case("toy", "toy", "mixed", "global_pool", 0.73, 0.37, timing, mu, model.phase_first(0.37, timing))
    record = json.loads(json.dumps(model.evaluate_case(case, CONTROL), allow_nan=False))
    np.testing.assert_allclose(record["phase_token_totals"], np.array([0.8, 0.2]) * 0.73, atol=1e-15, rtol=0)
    assert record["phase_budget_max_abs_error"] < 1e-15
    assert record["aggregate_first_abs_error"] < 1e-15
    assert record["weight_box_max_violation"] == record["weight_simplex_max_abs_error"] == 0
    assert record["native_branch_departure_max_abs_error"] == 0
    assert record["numerical_checks_passed"]


def test_registered_geometry_preserves_aggregate_and_each_phase_budget_without_clipping() -> None:
    for case in model.cases():
        weights, fractions, scales, _ = model.covariates(case)
        pools = np.asarray(model.CONTRACT["pool_inventories"][case.inventory])
        epochs = state.policy_epochs(weights, fractions, scales)
        np.testing.assert_allclose(epochs @ pools, fractions * case.token_total, rtol=2e-15, atol=1e-15)
        assert fractions @ weights[:, 0] == pytest.approx(case.aggregate_first, abs=2e-15)
        assert (weights >= 0).all() and (weights <= 1).all()
        if case.timing == "early_heavy":
            assert weights[0, 0] >= weights[1, 0]
        elif case.timing == "late_heavy":
            assert weights[0, 0] <= weights[1, 0]


@pytest.mark.parametrize("scales", [np.zeros(2), np.array([3.0, 0.0]), np.array([3.0, 0.8])])
def test_exact_ties_and_null_preserve_native_features_and_zero_head_gain(scales: np.ndarray) -> None:
    for mu, weights in ((0.0, np.array([[0.2, 0.8], [0.9, 0.1]])), (3.0, np.array([[0.4, 0.6], [0.4, 0.6]]))):
        law = state.Law(1.0, 0.7, mu, ("first", "second"), np.array([0.5, 0.5]))
        result = state.evaluate_policy(weights, np.array([0.8, 0.2]), scales, law, CONTROL)
        features = model.readouts(result, law)
        np.testing.assert_array_equal(features["state_contrast"], np.zeros(2))
        np.testing.assert_array_equal(features["hazard_benefit"], features["native_benefit"])
        np.testing.assert_array_equal(features["additive_benefit"], features["native_benefit"])
        head = model.head_values(features, np.array([1.0, 3.0]))
        assert head["hazard_gain"] == head["additive_gain"] == 0.0


@pytest.mark.parametrize(
    "scales,masses", [(np.array([3.0, 3.0]), np.array([0.5, 0.5])), (np.array([3.0, 0.3]), np.array([1 / 11, 10 / 11]))]
)
def test_both_readouts_match_independent_state_ode(scales: np.ndarray, masses: np.ndarray) -> None:
    weights, fractions = np.array([[0.2, 0.8], [0.9, 0.1]]), np.array([0.8, 0.2])
    law = state.Law(1.0, 0.7, 2.0, ("first", "second"), masses)
    phases = state.policy_epochs(weights, fractions, scales)
    result = state.evaluate_policy(weights, fractions, scales, law, CONTROL)
    actual = model.readouts(result, law)
    path = ode_mastery(phases, law)
    tied = ode_mastery(phases.sum(axis=0)[None], law)
    hazard = (phases.sum(axis=0) * law.rate) ** law.power
    native = -np.expm1(-hazard)
    expected_hazard = 1 - np.exp(-hazard) * (1 - path) / (1 - tied)
    np.testing.assert_allclose(actual["additive_benefit"], native + path - tied, atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(actual["hazard_benefit"], expected_hazard, atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(actual["state_contrast"], path - tied, atol=1e-10, rtol=1e-10)


def test_finite_state_readout_preserves_overshoot_and_negative_phase_effects() -> None:
    fractions, scales = np.full(2, 0.5), np.array([8.0, 1.0])
    law = state.Law(1.0, 1.0, 3.0, ("first", "second"), np.array([0.5, 0.5]))
    late_receiver = state.evaluate_policy(np.array([[0.0, 1.0], [1.0, 0.0]]), fractions, scales, law, CONTROL)
    early_receiver = state.evaluate_policy(np.array([[1.0, 0.0], [0.0, 1.0]]), fractions, scales, law, CONTROL)
    late, early = model.readouts(late_receiver, law), model.readouts(early_receiver, law)
    assert late["additive_benefit"][0] > 1
    assert late["hazard_benefit"][0] <= 1
    assert early["state_contrast"][0] < 0 < late["state_contrast"][0]
    for features in (late, early):
        np.testing.assert_allclose(
            features["hazard_departure"], features["attenuation"] * features["additive_departure"], atol=0, rtol=0
        )
        assert model.head_values(features, np.array([1.0, 0.0]))["additive_gain"] * features["state_contrast"][0] > 0


def test_first_order_readouts_share_the_analytic_zero_strength_derivative() -> None:
    law = state.Law(1.0, 1.0, 0.0, ("first", "second"), np.array([0.5, 0.5]))
    weights, fractions, scales = np.array([[5 / 44, 39 / 44], [0.0, 1.0]]), np.array([0.8, 0.2]), np.full(2, 2.2)
    expected = np.exp(-np.array([0.2, 2.0])) * np.array([-0.008285011824549, 0.067580580407280])
    errors, second_order = [], []
    for mu in (1e-3, 1e-4):
        changed_law = replace(law, mu=mu)
        features = model.readouts(state.evaluate_policy(weights, fractions, scales, changed_law, CONTROL), changed_law)
        errors.append(max(np.max(abs(features[f"{name}_departure"] / mu - expected)) for name in ("hazard", "additive")))
        second_order.append(np.max(abs(features["additive_departure"] - features["hazard_departure"])) / mu**2)
    assert errors[1] < 0.12 * errors[0]
    assert errors[1] < 2e-6
    assert second_order[1] == pytest.approx(second_order[0], rel=0.01)


def test_saturated_native_readout_keeps_subtraction_hidden_state_and_loss_contrasts() -> None:
    law = state.Law(1.0, 1.0, 3.0, ("first", "second"), np.full(2, 0.5))
    epochs = np.array([60.0, 0.0])
    path = state.State(epochs, np.array([np.log(0.6), -np.inf]), np.array([np.log(0.4), -np.inf]), 0, 0, 0.0)
    tied = state.State(epochs, np.array([np.log(0.3), -np.inf]), np.array([np.log(0.7), -np.inf]), 0, 0, 0.0)
    effective = path.survival_hazard(law) + tied.deficit(law)
    result = state.PolicyResult(-np.expm1(-effective), effective, path, tied, False)
    features = model.readouts(result, law)
    assert features["hazard_benefit"][0] == features["native_benefit"][0] == 1
    assert features["hazard_departure"][0] > 0
    assert features["state_contrast"][0] == pytest.approx(0.3, abs=2e-16)
    head = model.head_values(features, np.array([1.0, 3.0]))
    assert head["hazard_direct_gain"] == 0.0 < head["hazard_gain"]
