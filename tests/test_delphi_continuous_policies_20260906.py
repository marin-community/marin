# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from experiments.domain_phase_mix.exploratory.two_phase_many import optimize_delphi_matched_policies_20260906 as policy


@pytest.mark.parametrize(
    "weights,upper,expected",
    [([0.9, 0.05, 0.05], [0.4, 0.8, 0.8], [0.4, 0.3, 0.3]), ([-0.5, 0.5, 1], [1, 1, 1], [0, 0.25, 0.75])],
)
def test_start_projection_matches_unique_quadratic_solution(weights, upper, expected):
    actual = policy.project_start(np.array(weights), np.array(upper), np.full(3, 1 / 3))
    np.testing.assert_allclose(actual, expected, atol=1e-10)


def test_capped_olmix_policy_reaches_known_convex_optimum():
    surrogate = policy.OlmixSurrogate("synthetic", np.ones(1), np.zeros(1), np.array([[0, 1, 2]]))
    natural = np.full(3, 1 / 3)
    for start in (natural, np.array([0.1, 0.1, 0.8])):
        weights, result = policy.optimize_start(surrogate.predict, start, natural, np.array([0.4, 0.8, 0.8]), 0)
        np.testing.assert_allclose(weights, [0.4, 0.6, 0], atol=1e-7)
        assert result["success"]
        assert result["objective"] == pytest.approx(1 + np.exp(0.6), abs=1e-8)


def test_kl_regularized_policy_uses_capped_information_projection():
    natural = np.array([0.6, 0.3, 0.1])
    weights, result = policy.optimize_start(
        lambda query: np.ones(len(query)), np.array([0.3, 0.4, 0.3]), natural, np.array([0.4, 0.6, 0.6]), 0.02
    )
    np.testing.assert_allclose(weights, [0.4, 0.45, 0.15], atol=2e-6)
    assert result["success"]
    assert result["objective"] < result["start_objective"]


def test_support_distances_distinguish_vertices_from_convex_hull():
    panel = np.array([[1, 0, 0], [0, 1, 0]])
    outside = policy.support_distances(panel, np.array([0.1, 0.2, 0.7]))
    inside = policy.support_distances(panel, np.array([0.2, 0.8, 0]))
    assert outside["nearest_panel_tv"] == pytest.approx(0.8)
    assert outside["hull_distance_tv"] == pytest.approx(0.7)
    assert inside["hull_distance_tv"] == pytest.approx(0)
    assert inside["nearest_panel_tv"] == pytest.approx(0.2)
