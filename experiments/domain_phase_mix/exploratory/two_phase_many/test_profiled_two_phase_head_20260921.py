# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.optimize import lsq_linear

sys.path.insert(0, str(Path(__file__).resolve().parent))

from experiments.domain_phase_mix.exploratory.two_phase_many import profiled_two_phase_head_20260921 as profile


@pytest.fixture
def regression_data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(20260921)
    matrix = rng.uniform(-1, 1, (31, 4))
    response = 0.6 + np.exp(-0.2 + matrix @ np.array([0.45, 0.0, 0.12, 0.35]))
    response += rng.normal(0, 0.005, len(response))
    return matrix, response, np.array([2.0, 2.4, 2.1, 2.2])


@pytest.mark.parametrize("ridge,rho", [(0.0, 0.0), (0.01, 0.75), (0.1, -1.0)])
def test_profile_fit_matches_independent_bounded_full_system(regression_data, ridge, rho) -> None:
    matrix, response, spans = regression_data
    floor = rho * np.min(response)
    deficit = response - floor
    # The reference retains a free intercept column in the bounded solve; it
    # does not use the production solver's weighted centering or NNLS routine.
    full_matrix = deficit[:, None] * np.column_stack([np.ones(len(response)), matrix])
    penalty = np.column_stack([np.zeros(matrix.shape[1]), np.sqrt(ridge) * deficit.mean() * np.diag(spans)])
    reference = lsq_linear(
        np.vstack([full_matrix, penalty]),
        np.concatenate([deficit * np.log(deficit), np.zeros(matrix.shape[1])]),
        bounds=(np.r_[-np.inf, np.zeros(matrix.shape[1])], np.full(matrix.shape[1] + 1, np.inf)),
        method="bvls",
        tol=1e-12,
    )
    actual = profile.fit_profile_head(matrix, response, ridge, rho, spans)
    expected = floor + np.exp(reference.x[0] + matrix @ reference.x[1:])
    np.testing.assert_allclose(actual.predict(matrix), expected, rtol=2e-11, atol=2e-12)


def test_profile_fit_preserves_predictions_when_feature_units_change(regression_data) -> None:
    matrix, response, spans = regression_data
    factors = np.array([0.01, 10, 0.2, 5])
    original = profile.fit_profile_head(matrix, response, 0.01, 0.5, spans)
    rescaled = profile.fit_profile_head(matrix * factors, response, 0.01, 0.5, spans * factors)
    query = np.vstack([matrix, np.full(4, -2), np.full(4, 2)])
    np.testing.assert_allclose(original.predict(query), rescaled.predict(query * factors), rtol=2e-11, atol=2e-12)


def test_profile_fit_preserves_response_unit_scaling(regression_data) -> None:
    matrix, response, spans = regression_data
    original = profile.fit_profile_head(matrix, response, 0.01, -1.0, spans)
    factor = 3.7
    rescaled = profile.fit_profile_head(matrix, factor * response, 0.01, -1.0, spans)
    np.testing.assert_allclose(rescaled.predict(matrix), factor * original.predict(matrix), rtol=2e-11, atol=2e-12)


def test_profile_fit_with_zero_span_constant_column_preserves_predictions(regression_data) -> None:
    matrix, response, spans = regression_data
    original = profile.fit_profile_head(matrix, response, 0.1, 0.25, spans)
    extended = profile.fit_profile_head(
        np.column_stack([matrix, np.ones(len(matrix))]), response, 0.1, 0.25, np.r_[spans, 0]
    )
    np.testing.assert_allclose(
        extended.predict(np.column_stack([matrix, np.ones(len(matrix))])),
        original.predict(matrix),
        rtol=2e-11,
        atol=2e-12,
    )


def test_profile_cv_heldout_labels_change_errors_but_not_fitted_floor(regression_data) -> None:
    matrix, response, spans = regression_data
    train = np.arange(21)
    held = np.arange(21, 31)
    changed = response.copy()
    changed[held] = 0.1 * response[held]
    head = profile.fit_profile_head(matrix[train], response[train], 0.01, 0.75, spans)
    expected = float(np.sqrt(np.mean((head.predict(matrix[held]) - changed[held]) ** 2)))
    actual = profile.profile_cv_rmse(matrix, changed, 0.01, 0.75, spans, [(train, held)])
    assert actual == pytest.approx(expected, rel=2e-13, abs=2e-14)


def test_native_head_can_represent_negative_floor_without_a_profile_cap() -> None:
    head = profile.mariner.Head(-0.3, np.array([0.4, 0.2]), -1.25)
    query = np.array([[-1, 0.5], [0.2, 0.0], [3, 1]])
    np.testing.assert_allclose(head.predict(query), -1.25 + np.exp(-0.3 + query @ np.array([0.4, 0.2])), atol=1e-14)


def test_profile_fit_rejects_zero_span_for_varying_column(regression_data) -> None:
    matrix, response, spans = regression_data
    spans[0] = 0
    with pytest.raises(ValueError, match="constant feature"):
        profile.fit_profile_head(matrix, response, 0.1, 0.25, spans)
