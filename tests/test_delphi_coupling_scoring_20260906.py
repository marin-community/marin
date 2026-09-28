# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from experiments.domain_phase_mix.exploratory.two_phase_many import score_delphi_coupling_20260906 as scoring


def test_weighted_error_decomposition_separates_cancelled_task_errors():
    errors = np.array([[2.0, 0.0], [0.0, 2.0]])
    result = scoring.error_decomposition(errors, np.array([0.25, 0.75]))
    assert result["weighted_component_mse"] == 2.0
    assert result["aggregate_mse"] == 1.25
    assert result["task_deviation_mse"] == 0.75
    np.testing.assert_allclose(result["mean_component_rmse"], np.sqrt(2), atol=1e-15, rtol=0)


def test_float32_evaluator_weights_preserve_the_exact_aggregate_error():
    weights = np.array([1 / 3] * 3, dtype=np.float32).astype(float)
    errors = np.array([[1.0, 2.0, 3.0], [3.0, 1.0, 2.0]])
    result = scoring.error_decomposition(errors, weights)
    np.testing.assert_allclose(result["aggregate_mse"], (6 * weights[0]) ** 2, atol=1e-15, rtol=0)
    np.testing.assert_allclose(
        result["weighted_component_mse"],
        result["aggregate_mse_component_contribution"] + result["task_deviation_mse"],
        atol=1e-15,
        rtol=0,
    )


def test_interaction_removal_reconstructs_anchored_bucket_main_effects():
    shard = {
        "test": np.array([0]),
        "train": np.array([1]),
        "anchor": np.array([0.5, 0.5]),
        "feature_scale": np.array([[0.1], [0.2]]),
        "projection": np.eye(2),
        "parameters": np.array([[np.log(2), 0.2, -0.6]]),
        "outcome_scale": np.array([3.0]),
    }
    data = {"weights": np.array([[0.6, 0.4], [0.5, 0.5]]), "inventory": np.array([2.0, 8.0])}
    bank = {"weights": np.array([[0.5, 0.5]])}
    result = scoring.removed_atomic(shard, data, bank, scoring.coupling.Basis.SHARES)
    expected = np.array([[6 * (np.exp(0.2) + np.exp(0.3) - 1)], [6.0], [6.0]])
    np.testing.assert_allclose(result, expected, atol=1e-13, rtol=0)
