# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "scipy", "threadpoolctl", "pytest"]
# ///
"""Pure numerical behavior checks; no research covariates or fitted heads."""

import json
from collections.abc import Callable

import numpy as np
import pytest

from experiments.domain_phase_mix.exploratory.two_phase_many import tpp40_two_bucket_optimization_20261015 as search
from experiments.domain_phase_mix.exploratory.two_phase_many.tpp40_tangent_optimization_20261015 import PolicyGeometry


def callbacks(
    objective: Callable[[float, float], float],
    *,
    alpha: float,
    scales: np.ndarray,
    replay: Callable[[float, float], float] | None = None,
) -> search.Predictors:
    def search_value(early: np.ndarray, late: np.ndarray) -> float:
        return objective(float(early[1] / (alpha * scales[1])), float(late[1] / ((1 - alpha) * scales[1])))

    def replay_value(early: np.ndarray, late: np.ndarray) -> float:
        function = objective if replay is None else replay
        return function(float(early[1] / (alpha * scales[1])), float(late[1] / ((1 - alpha) * scales[1])))

    return search.Predictors(
        search=search_value,
        replay=replay_value,
        identity={"purpose": "invented analytic objective"},
        search_numerics={"arithmetic": "float64"},
        replay_numerics={"arithmetic": "independent callback"},
    )


def test_corner_optimum_and_physical_accounting_replay_without_projection():
    alpha, scales = 0.8, np.array([2.0, 7.0])
    result = search.optimize_two_bucket(
        callbacks(lambda x, y: x**2 + (y - 1) ** 2, alpha=alpha, scales=scales),
        alpha=alpha,
        scales=scales,
        restriction=search.Restriction.TEMPORAL,
        observed_policies=np.array([[[1, 0], [0, 1]]]),
    )
    assert result["best"]["coordinates"] == [0.0, 1.0]
    assert result["best"]["predicted"] == 0.0
    assert result["best_static"]["predicted"] == pytest.approx(0.5, abs=1e-12)
    assert result["predicted_gain_same_aggregate"] == pytest.approx(0.68, abs=1e-14)
    assert result["nearest_policy_sqrt_distance"] == 0.0
    assert not result["numerical_audit_unresolved"]
    assert result["calls"]["search"] <= 8929
    assert result["calls"]["accounting"] <= 1
    for record in result["traces"]:
        first, second = np.array(record["phase0"]), np.array(record["phase1"])
        assert first.sum() == second.sum() == 1
        np.testing.assert_array_equal(record["early_epochs"], alpha * scales * first)
        np.testing.assert_array_equal(record["late_epochs"], (1 - alpha) * scales * second)
    assert result["replay"]["maximum_absolute_change"] == 0.0
    assert result["replay"]["selected_reporting_replays"]["best"]["replay_value"] == 0.0
    assert json.loads(json.dumps(result, allow_nan=False))["best"]["coordinates"] == [0.0, 1.0]


def test_interior_basin_with_a_second_local_minimum_is_recovered():
    def objective(x: float, y: float) -> float:
        return min((x - 0.23) ** 2 + (y - 0.71) ** 2, 0.02 + (x - 0.8) ** 2 + (y - 0.2) ** 2)

    result = search.optimize_two_bucket(
        callbacks(objective, alpha=0.5, scales=np.array([2.0, 2.0])),
        alpha=0.5,
        scales=np.array([2.0, 2.0]),
        restriction=search.Restriction.TEMPORAL,
    )
    np.testing.assert_allclose(result["best"]["coordinates"], [0.23, 0.71], atol=2e-6, rtol=0)
    assert result["best"]["predicted"] < 1e-12
    assert result["best"]["predicted"] <= result["best_static"]["predicted"]
    assert len(set(map(tuple, result["powell_starts"]))) == 12
    assert not result["replay"]["unresolved"]


def test_single_accounting_point_can_replace_the_incumbent_without_another_search():
    def narrow_tied_dip(x: float, y: float) -> float:
        # A missed narrow diagonal well exercises the approved accounting step.
        if abs(x - y) < 1e-14 and abs(x - 0.2) < 1e-14:
            return -1.0
        return x**2 + (y - 1) ** 2

    result = search.optimize_two_bucket(
        callbacks(narrow_tied_dip, alpha=0.8, scales=np.array([2.0, 7.0])),
        alpha=0.8,
        scales=np.array([2.0, 7.0]),
        restriction=search.Restriction.TEMPORAL,
    )
    assert result["original_search_best"]["coordinates"] == [0.0, 1.0]
    assert result["accounting_tied_replaced_incumbent"]
    assert result["best"]["predicted"] == -1.0
    assert result["best"]["tied"]
    assert result["predicted_gain_same_aggregate"] == result["predicted_gain_vs_static"] == 0.0
    assert result["calls"]["accounting"] == 1
    assert result["replay"]["selected_reporting_replays"]["aggregate_tied"]["replay_value"] == -1.0


def test_failed_temporal_evaluations_preserve_exact_tied_incumbent_and_failures():
    def only_tied(x: float, y: float) -> float:
        if x != y:
            raise ArithmeticError("Invented integration failure away from the diagonal")
        return (x - 0.321) ** 2

    scales = np.array([2.0, 2.0])
    predictors = callbacks(only_tied, alpha=0.5, scales=scales)
    result = search.optimize_two_bucket(predictors, alpha=0.5, scales=scales, restriction=search.Restriction.TEMPORAL)
    assert result["best"]["tied"]
    assert result["best"]["predicted"] == result["best_static"]["predicted"]
    assert result["best"]["coordinates"][0] == pytest.approx(0.321, abs=1e-8)
    assert result["predicted_gain_vs_static"] == result["phase_tv"] == 0.0
    assert result["failed_local_solves"] > 0
    assert result["numerical_audit_unresolved"]
    assert any(record["status"] == "finite" and record.get("solver_success") is False for record in result["traces"])
    assert all(result["traces"][index]["predicted"] is None for index in result["failed_evaluation_trials"])
    evaluator = search.Evaluator(predictors, 0.5, scales)
    outside = evaluator.evaluate((-np.finfo(float).eps, 0.0), "unprojected_probe")
    assert outside["coordinates"][0] < 0
    assert outside["status"] == "infeasible_coordinates"
    assert outside["predicted"] is None


@pytest.mark.parametrize("failed_replay", [False, True])
def test_tighter_replay_flags_instability_without_reranking(failed_replay: bool):
    def objective(x: float, y: float) -> float:
        return (x - 0.3) ** 2 + (y - 0.3) ** 2

    def altered_replay(x: float, y: float) -> float:
        if failed_replay and x == 0:
            raise FloatingPointError("Invented stricter integration failure")
        return -objective(x, y)

    result = search.optimize_two_bucket(
        callbacks(objective, alpha=0.5, scales=np.array([2.0, 2.0]), replay=altered_replay),
        alpha=0.5,
        scales=np.array([2.0, 2.0]),
        restriction=search.Restriction.AGGREGATE,
    )
    assert result["best"]["coordinates"][0] == pytest.approx(0.3, abs=1e-8)
    assert result["replay"]["strict_pairwise_ordering_reversals"] > 0
    assert result["replay"]["maximum_absolute_change"] > 1e-8
    assert result["numerical_audit_unresolved"]
    assert not result["replay"]["selection_reranked"]
    assert any(record["status"] != "finite" for record in result["replay"]["records"]) == failed_replay


def test_registered_aggregate_returns_identical_static_and_free_policy():
    result = search.optimize_two_bucket(
        callbacks(lambda x, y: (0.8 * x + 0.2 * y - 0.4) ** 2, alpha=0.8, scales=np.array([3.0, 4.0])),
        alpha=0.8,
        scales=np.array([3.0, 4.0]),
        restriction=search.Restriction.AGGREGATE,
    )
    assert result["best"] == result["best_static"]
    assert result["best"]["coordinates"][0] == pytest.approx(0.4, abs=1e-8)
    assert result["phase_tv"] == result["predicted_gain_same_aggregate"] == result["predicted_gain_vs_static"] == 0
    assert all(record["tied"] for record in result["traces"])
    assert not result["numerical_audit_unresolved"]


class CornerTruth:
    def predict(self, early: np.ndarray, late: np.ndarray) -> np.ndarray:
        return early[:, 1] ** 2 + (late[:, 1] - 1) ** 2


def test_signed_truth_regret_preserves_a_missed_truth_incumbent():
    selected = {
        "best": {"phase0": [1.0, 0.0], "phase1": [0.0, 1.0], "predicted": 0.01},
        "best_static": {"phase0": [0.5, 0.5], "phase1": [0.5, 0.5], "predicted": 0.5},
        "aggregate": [0.5, 0.5],
        "predicted_gain_same_aggregate": 0.49,
    }
    geometry = PolicyGeometry(
        phase0=np.array([[1.0, 0.0]]),
        phase1=np.array([[0.0, 1.0]]),
        scales=np.array([2.0, 2.0]),
        pool_tokens=np.ones(2),
        alpha=0.5,
    )
    result = search.SIGNED_TRUTH_ACCOUNTING(
        selected, CornerTruth(), geometry, {"best": {"predicted": 0.1}, "best_static": {"predicted": 0.5}}
    )
    assert result["regret_vs_truth_numerical_incumbent"] == -0.1
    assert result["regret_vs_truth_static_incumbent"] == -0.5
    assert result["true_gain_same_aggregate"] == 0.5
    assert result["phase_gain_overprediction"] == pytest.approx(-0.01, abs=1e-14)
    assert result["selected_optimism"] == -0.01
