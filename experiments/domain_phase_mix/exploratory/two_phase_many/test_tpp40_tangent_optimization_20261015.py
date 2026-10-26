# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Independent convex controls for the raw policy diagnostic."""

import numpy as np

from experiments.domain_phase_mix.exploratory.two_phase_many import tpp40_tangent_optimization_20261015 as optimization


class QuadraticTruth:
    def predict(self, early: np.ndarray, late: np.ndarray) -> np.ndarray:
        return 0.5 + np.square(early / 0.8 - [0.8, 0.2]).sum(axis=1) + np.square(late / 0.2 - [0.2, 0.8]).sum(axis=1)


def test_raw_search_finds_convex_minimum_and_keeps_the_static_incumbent():
    geometry = optimization.PolicyGeometry(
        np.array([[0.5, 0.5], [0.2, 0.8], [0.8, 0.2]]),
        np.array([[0.5, 0.5], [0.8, 0.2], [0.2, 0.8]]),
        np.ones(2),
        np.ones(2),
        0.8,
    )
    truth = QuadraticTruth()
    result = optimization.optimize_fit(truth, geometry, seed=20261015, phase_blind=False)
    np.testing.assert_allclose(result["best"]["predicted"], 0.5, atol=1e-8, rtol=0)
    np.testing.assert_allclose(result["best_static"]["predicted"], 0.86, atol=1e-8, rtol=0)
    np.testing.assert_allclose(result["best"]["phase0"], [0.8, 0.2], atol=1e-5, rtol=0)
    np.testing.assert_allclose(result["best"]["phase1"], [0.2, 0.8], atol=1e-5, rtol=0)
    evaluated = optimization.evaluate_selected_truth(result, truth, geometry, result)
    np.testing.assert_allclose(evaluated["phase_gain_overprediction"], 0.0, atol=1e-12, rtol=0)
    np.testing.assert_allclose(evaluated["selected_optimism"], 0.0, atol=1e-12, rtol=0)
    np.testing.assert_allclose(evaluated["regret_vs_truth_numerical_incumbent"], 0.0, atol=1e-12, rtol=0)
    assert all(record["feasible"] for record in result["traces"])
