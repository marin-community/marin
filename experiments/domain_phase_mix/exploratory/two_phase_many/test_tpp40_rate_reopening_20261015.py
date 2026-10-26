# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Equation-level checks before registered synthetic label generation or fits."""

import numpy as np
import pytest

from experiments.domain_phase_mix.exploratory.two_phase_many import tpp40_rate_reopening_20261015 as candidate


@pytest.mark.parametrize("kappa,nu", [(0.0, 0.0), (0.5, 1.0), (8.0, 2.0), (128.0, 0.5)])
def test_tied_and_zero_exposure_preserve_native_columns(kappa: float, nu: float) -> None:
    alpha = 0.8
    total = np.array([[0.0, 0.0], [0.01, 1.0], [3.0, 20.0], [1e4, 1e8]])
    shape = {"rate": 0.25, "power": 0.7, "threshold": 3.0}
    actual = candidate.design(alpha * total, (1 - alpha) * total, alpha, shape, {"kappa": kappa, "nu": nu})
    np.testing.assert_array_equal(actual, candidate.mariner.design_matrix(total, shape))


def test_rate_harm_extremes_and_tied_derivative_show_directional_penalty() -> None:
    alpha, nu, exposure = 0.8, 1.0, 4.0
    shape = {"rate": 0.25, "power": 0.7, "threshold": 3.0}
    early = np.array([[0.0], [exposure]])
    actual = candidate.design(early, exposure - early, alpha, shape, {"kappa": 0.0, "nu": nu})[:, 1]
    origin, whole = candidate.mariner.harm(np.array([0.0, exposure]), shape["threshold"])
    np.testing.assert_allclose(
        actual, origin + (whole - origin) * np.array([(1 - alpha) ** -nu, alpha**-nu]), atol=1e-14, rtol=1e-13
    )
    epsilon = 1e-5
    x = np.array([[alpha - epsilon], [alpha + epsilon]])
    harms = candidate.design(x * exposure, (1 - x) * exposure, alpha, shape, {"kappa": 0.0, "nu": nu})[:, 1]
    slope = (harms[1] - harms[0]) / (2 * epsilon)
    prefix = candidate.mariner.harm(np.array(alpha * exposure), shape["threshold"])
    expected = nu * ((prefix - origin) - alpha * (whole - origin)) / (alpha * (1 - alpha))
    np.testing.assert_allclose(slope, expected, atol=1e-10, rtol=1e-8)
    assert slope < 0  # Convex native harm can initially decrease with early-heavy allocation.


def test_combined_law_preserves_historical_function_and_stays_finite_at_expanded_strength() -> None:
    early = np.array([[0.0, 2.0], [0.1, 0.0], [40.0, 1e6], [0.3, 2.5]])
    late = np.array([[1.0, 0.0], [0.0, 0.1], [1e6, 40.0], [0.7, 0.2]])
    shape = {"rate": 0.5, "power": 0.7, "threshold": 3.0}
    old, _ = candidate.historical.design_and_offset(
        early, late, 0.8, shape, "bounded_benefit_rate", {"kappa": 8.0, "nu": 1.0}
    )
    new = candidate.design(early, late, 0.8, shape, {"kappa": 8.0, "nu": 1.0})
    np.testing.assert_allclose(new, old, atol=2e-15, rtol=2e-15)
    expanded = candidate.design(early, late, 0.8, shape, {"kappa": 128.0, "nu": 1.0})
    assert np.isfinite(expanded).all()
    assert np.all((expanded[:, :2] >= -1) & (expanded[:, :2] <= 0))
    np.testing.assert_array_equal(expanded[:, 2:], old[:, 2:])
