# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parent))

from experiments.domain_phase_mix.exploratory.two_phase_many import round4_mean_head_20260921 as mean_head


@pytest.fixture
def regression_data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(20260924)
    aggregate = rng.uniform(-1, 1, (37, 2))
    phase = rng.uniform(-0.5, 0.5, (37, 1))
    response = 0.4 + np.exp(-0.3 + aggregate @ [-0.25, 0.4] + phase[:, 0] * -0.6)
    response += rng.normal(0, 0.008, len(response))
    return aggregate, phase, response


@pytest.mark.parametrize("ridge", [0.0, 0.03, 1.0])
def test_mean_fit_matches_independent_numerical_derivative_solver(regression_data, ridge) -> None:
    aggregate, phase, response = regression_data
    spec = mean_head.mariner.FloorSpec(np.nan, 0.0, 2.5)
    aggregate_penalty, phase_penalty = np.array([0.7, 2.0]), np.array([1.3])
    actual = mean_head.fit_mean_head(aggregate, phase, response, ridge, spec, aggregate_penalty, phase_penalty)
    deficit = np.maximum(response - actual.floor, mean_head.mariner.DEFICIT_FLOOR)
    matrix = np.column_stack([np.ones(len(response)), aggregate, phase])
    penalty = np.r_[0.0, aggregate_penalty, phase_penalty]

    def objective(vector: np.ndarray) -> float:
        eta = matrix @ vector
        return float(np.sum(np.exp(eta) - deficit * eta) / deficit.mean() + ridge * np.sum((penalty * vector) ** 2) / 2)

    # SLSQP uses numerical derivatives and raw coordinates, independent of the
    # production Newton Hessian, scaling, line search, and BVLS subproblems.
    reference = minimize(
        objective,
        np.r_[np.log(deficit.mean()), np.zeros(3)],
        method="SLSQP",
        bounds=[(None, None), (0, None), (0, None), (None, None)],
        options={"ftol": 1e-12, "maxiter": 1000},
    )
    expected = actual.floor + np.exp(matrix @ reference.x)
    np.testing.assert_allclose(actual.predict(aggregate, phase), expected, rtol=2e-6, atol=2e-7)
    fitted = np.r_[actual.intercept, actual.aggregate_coefficients, actual.phase_coefficients]
    assert objective(fitted) <= objective(reference.x) + 1e-9
    assert actual.certified_kkt_residual <= mean_head.KKT_TOLERANCE
    assert actual.aggregate_coefficients[0] == 0


def test_mean_objective_derivatives_match_central_differences() -> None:
    matrix = np.array([[1.0, 0.2, -0.4], [1, -0.7, 0.9], [1, 1.2, 0.3], [1, 0.1, 0.5]])
    vector, deficit, penalty = np.array([-0.3, 0.2, -0.4]), np.array([0.2, 1.1, 0.7, 2.0]), np.array([0, 1.4, 0.6])
    _, gradient, hessian = mean_head.mean_objective(vector, matrix, deficit, penalty, 0.3)
    step = 1e-5
    numeric_gradient, numeric_hessian = [], []
    for coordinate in np.eye(3):
        upper = mean_head.mean_objective(vector + step * coordinate, matrix, deficit, penalty, 0.3)
        lower = mean_head.mean_objective(vector - step * coordinate, matrix, deficit, penalty, 0.3)
        numeric_gradient.append((upper[0] - lower[0]) / (2 * step))
        numeric_hessian.append((upper[1] - lower[1]) / (2 * step))
    np.testing.assert_allclose(gradient, numeric_gradient, rtol=1e-8, atol=1e-9)
    np.testing.assert_allclose(hessian, np.array(numeric_hessian).T, rtol=1e-8, atol=1e-9)
    assert np.linalg.eigvalsh(hessian).min() > 0


@pytest.mark.parametrize("gamma", [1.0, 2.5, 6.0])
def test_mean_intercept_only_recovers_arithmetic_mean_with_native_floor(gamma) -> None:
    response = np.array([0.4, 0.8, 1.2, 3.0])
    empty = np.empty((4, 0))
    head = mean_head.fit_mean_head(empty, empty, response, 0.1, mean_head.mariner.FloorSpec(np.nan, 0, gamma), [], [])
    expected = head.floor + np.mean(np.maximum(response - head.floor, mean_head.mariner.DEFICIT_FLOOR))
    np.testing.assert_allclose(head.predict(empty, empty), expected, rtol=1e-10, atol=1e-11)
    assert head.floor < 0 if gamma == 6 else head.floor <= response.min()


def test_signed_mean_head_matches_two_group_means() -> None:
    phase = np.array([[-1.0], [-1.0], [-1.0], [1.0], [1.0], [1.0]])
    response = np.array([1.0, 1.2, 1.4, 0.5, 0.8, 0.9])
    empty = np.empty((len(response), 0))
    head = mean_head.fit_mean_head(empty, phase, response, 0, mean_head.mariner.FloorSpec(np.nan, 0, 3), [], [1])
    expected = np.repeat([response[:3].mean(), response[3:].mean()], 3)
    np.testing.assert_allclose(head.predict(empty, phase), expected, rtol=1e-9, atol=1e-10)
    assert head.phase_coefficients[0] < 0


def test_mean_head_preserves_response_and_feature_units(regression_data) -> None:
    aggregate, phase, response = regression_data
    spec = mean_head.mariner.FloorSpec(np.nan, 0, 3)
    original = mean_head.fit_mean_head(aggregate, phase, response, 0.1, spec, [0.7, 2.0], [1.3])
    aggregate_scale, phase_scale, response_scale = np.array([0.01, 12]), -3.0, 4.7
    changed = mean_head.fit_mean_head(
        aggregate * aggregate_scale,
        phase * phase_scale,
        response * response_scale,
        0.1,
        spec,
        np.array([0.7, 2.0]) * aggregate_scale,
        np.array([1.3]) * abs(phase_scale),
    )
    np.testing.assert_allclose(
        changed.predict(aggregate * aggregate_scale, phase * phase_scale),
        response_scale * original.predict(aggregate, phase),
        rtol=2e-9,
        atol=2e-10,
    )


def test_mean_objective_does_not_clip_large_positive_linear_predictors() -> None:
    matrix = np.ones((2, 1))
    low = mean_head.mean_objective(np.array([31.0]), matrix, np.ones(2), np.zeros(1), 0)[0]
    high = mean_head.mean_objective(np.array([32.0]), matrix, np.ones(2), np.zeros(1), 0)[0]
    assert high > 2.7 * low
