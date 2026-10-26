# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Behavioral checks for the conditional response-space diagnostic."""

import json

import numpy as np
import pytest
from scipy.optimize import nnls

from experiments.domain_phase_mix.exploratory.two_phase_many import tpp40_response_profile_20261015 as model


@pytest.mark.parametrize("floor", [None, 0.6])
@pytest.mark.parametrize("target", ["interior", "floor_boundary", "scale_boundary"])
def test_envelope_gradient_matches_directional_difference(floor, target):
    rng = np.random.default_rng(917)
    matrix = rng.uniform(-1, 1, (45, 3))
    amplitudes = np.array([0.3, 0.7, 0.5])
    phi = np.exp(matrix @ amplitudes - np.max(matrix @ amplitudes))
    if target == "interior":
        observed = 0.8 + 2.2 * phi + 0.02 * np.sin(np.arange(len(phi)))
    elif target == "floor_boundary":
        observed = -0.3 + 1.7 * phi
    else:
        observed = 0.5 - 0.2 * phi
    result = model.evaluate(matrix, observed, amplitudes, floor)
    direction = np.array([0.1, -0.05, 0.13])
    epsilon = 2e-5
    left = model.evaluate(matrix, observed, amplitudes - epsilon * direction, floor)["scaled_objective"]
    right = model.evaluate(matrix, observed, amplitudes + epsilon * direction, floor)["scaled_objective"]
    numeric = (right - left) / (2 * epsilon)
    analytic = float(np.asarray(result["gradient"]) @ direction)
    assert abs(analytic - numeric) <= 3e-5 * max(1, abs(analytic))
    if floor is None:
        expected = {"interior": "interior", "floor_boundary": "floor_zero", "scale_boundary": "scale_zero"}[target]
        assert result["face"] == expected


def test_profile_boundaries_match_independent_nnls():
    phi = np.linspace(0.1, 1.0, 23)
    matrix = np.column_stack((np.ones(23), phi))
    for observed in (0.3 + 2 * phi, -0.3 + 2 * phi, 1 - 0.5 * phi, -np.ones(23)):
        expected, _ = nnls(matrix, observed)
        result = model.linear_profile(phi, observed, None)
        np.testing.assert_allclose(result.prediction, matrix @ expected, rtol=0, atol=2e-15)


def test_constant_learned_profile_zero_gradient_does_not_certify_optimum():
    matrix = np.linspace(-1, 1, 21)[:, None]
    observed = 1 + np.exp(0.2 * matrix[:, 0])
    constant = model.evaluate(matrix, observed, np.zeros(1), None)
    departure = model.evaluate(matrix, observed, np.array([0.01]), None)
    assert constant["gradient"] == [0.0]
    assert constant["nonregular_learned_floor"]
    assert not constant["first_order_tolerance_met"]
    assert departure["scaled_objective"] < constant["scaled_objective"]
    known = model.evaluate(matrix, observed, np.zeros(1), 0.6)
    assert not known["nonregular_learned_floor"]
    assert known["gradient"] != [0.0]


def test_tiny_positive_profile_variance_is_preserved():
    phi = 1 + np.linspace(-1, 1, 21) * 1e-12
    observed = 0.3 + 2 * phi
    result = model.linear_profile(phi, observed, None)
    assert not result.constant
    assert result.centered_variance > 0
    np.testing.assert_allclose(result.prediction, observed, rtol=0, atol=2e-15)


@pytest.mark.parametrize("floor", [None, 0.6])
def test_five_training_local_starts_recover_clean_function_and_replay(floor):
    rng = np.random.default_rng(36)
    training = rng.uniform(-1, 1, (80, 3))
    query = rng.uniform(-0.8, 0.8, (25, 3))
    coefficients, intercept = np.array([0.4, 0.7, 0.3]), -0.2
    observed = 0.6 + np.exp(intercept + training @ coefficients)
    starts = model.gamma_starts(training, observed)
    fit = model.fit_response(training, observed, floor, starts)
    reloaded = json.loads(json.dumps(fit, allow_nan=False))
    predicted = model.predict(query, reloaded["selected"])
    np.testing.assert_allclose(predicted, 0.6 + np.exp(intercept + query @ coefficients), atol=1e-8, rtol=0)
    assert len(fit["solves"]) == 5
    assert [item["start"] for item in fit["solves"]] == starts
    assert fit["selected"]["scaled_objective"] <= min(
        model.evaluate(training, observed, np.asarray(item["amplitudes"]), floor)["scaled_objective"] for item in starts
    )


def test_normalized_replay_and_gradient_ignore_large_common_logit():
    matrix = np.full((4, 1), 1e16)
    amplitude = np.ones(1)
    fitted = model.evaluate(matrix, np.full(4, 3.0), amplitude, 0.6)
    np.testing.assert_allclose(model.predict(matrix, fitted), 3.0, rtol=0, atol=5e-16)
    nonconstant_response = np.array([1.1, 2.2, 3.3, 4.4])
    centered = model.evaluate(np.zeros_like(matrix), nonconstant_response, amplitude, 0.6)
    shifted = model.evaluate(matrix, nonconstant_response, amplitude, 0.6)
    np.testing.assert_array_equal(centered["gradient"], shifted["gradient"])
    np.testing.assert_array_equal(shifted["gradient"], [0.0])
    np.testing.assert_allclose(model.predict(matrix, shifted), np.mean(nonconstant_response), rtol=0, atol=5e-16)


def test_near_constant_learned_profile_is_retained_as_unresolved():
    matrix = np.linspace(-1, 1, 21)[:, None] * 1e-16
    observed = np.full(21, 2.0)
    starts = [{"gamma": value, "amplitudes": [1.0]} for value in model.GAMMAS]
    fitted = model.fit_response(matrix, observed, None, starts)
    assert fitted["selected"]["near_constant_phi"]
    assert fitted["unresolved"]
    np.testing.assert_allclose(model.predict(matrix, fitted["selected"]), observed, rtol=0, atol=5e-16)
