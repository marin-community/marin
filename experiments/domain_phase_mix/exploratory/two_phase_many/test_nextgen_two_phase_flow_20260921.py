# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
from scipy.linalg import expm
from scipy.special import exprel

from experiments.domain_phase_mix.exploratory.two_phase_many.nextgen_two_phase_flow_20260921 import (
    flow_readout,
    phase_benefit,
)


def _policies() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    rng = np.random.default_rng(20260921)
    first = rng.dirichlet(np.ones(4), 13)
    second = rng.dirichlet(np.ones(4), 13)
    scales = np.array([0.003, 0.5, 8, 150])
    alpha = 0.8
    return alpha * first * scales, (1 - alpha) * second * scales, first, second, alpha


def _matrix_reference(
    early: np.ndarray,
    late: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
    alpha: float,
    rate: float,
    power: float,
    forgetting: float,
    consolidation: float,
) -> np.ndarray:
    result = np.empty_like(early)
    for index in np.ndindex(early.shape):
        total = early[index] + late[index]
        clock = (rate * total) ** power
        state = np.array([1.0, 0.0, 0.0])
        for exposure, weight, duration in (
            (early[index], first[index], alpha),
            (late[index], second[index], 1 - alpha),
        ):
            acquisition = clock * exposure / total / duration if total else 0.0
            loss = forgetting * (1 - weight)
            generator = np.array(
                [
                    [-acquisition, loss, 0],
                    [acquisition, -loss - consolidation, 0],
                    [0, consolidation, 0],
                ]
            )
            state = expm(duration * generator) @ state
        result[index] = state[1:].sum()
    return result


@pytest.mark.parametrize(
    "rate,power,forgetting,consolidation",
    [
        (0.3, 0.5, 0, 0),
        (0.3, 0.5, 2, 0),
        (1, 1, 0, 2),
        (1, 1.5, 0.5, 2),
        (3, 2, 8, 0.5),
        (1e-8, 1, 1e-8, 1e-8),
    ],
)
def test_flow_matches_three_state_matrix_exponential(rate, power, forgetting, consolidation) -> None:
    args = (*_policies(), rate, power, forgetting, consolidation)
    actual = flow_readout(*args)
    expected = _matrix_reference(*args)
    np.testing.assert_allclose(actual, expected, rtol=3e-11, atol=3e-13)
    assert actual.min() >= 0
    assert actual.max() <= 1 + 3e-13


@pytest.mark.parametrize("mode", ["additive", "ratio"])
def test_tied_feature_equals_mariner_benefit(mode) -> None:
    _, _, first, _, alpha = _policies()
    scales = np.array([0.003, 0.5, 8, 150])
    total = first * scales
    actual = phase_benefit(alpha * total, (1 - alpha) * total, first, first, alpha, 0.3, 1.5, 2, 0.5, mode)
    np.testing.assert_allclose(actual, -np.expm1(-((0.3 * total) ** 1.5)), rtol=3e-13, atol=3e-15)


@pytest.mark.parametrize("consolidation", [0, 0.5, 2, 8])
def test_no_forgetting_is_exact_exposure_only_readout(consolidation) -> None:
    early, late, first, second, alpha = _policies()
    expected = -np.expm1(-((0.3 * (early + late)) ** 1.5))
    actual = flow_readout(early, late, first, second, alpha, 0.3, 1.5, 0, consolidation)
    np.testing.assert_allclose(actual, expected, rtol=3e-12, atol=3e-14)
    corrected = phase_benefit(early, late, first, second, alpha, 0.3, 1.5, 0, consolidation)
    np.testing.assert_allclose(corrected, expected, rtol=3e-12, atol=3e-14)


def test_without_consolidation_matches_scalar_acquisition_forgetting() -> None:
    early, late, first, second, alpha = _policies()
    clock = (0.3 * (early + late)) ** 1.5
    expected = np.zeros_like(early)
    for exposure, weights, duration in ((early, first, alpha), (late, second, 1 - alpha)):
        acquisition = clock * exposure / (early + late) / duration
        decay = acquisition + 2 * (1 - weights)
        expected = expected * np.exp(-decay * duration) + acquisition * duration * exprel(-decay * duration)
    actual = flow_readout(early, late, first, second, alpha, 0.3, 1.5, 2, 0)
    np.testing.assert_allclose(actual, expected, rtol=3e-12, atol=3e-14)


def test_repeated_eigenvalue_and_nearby_rates_match_matrix_exponential() -> None:
    # With no forgetting and acquisition=consolidation, the transient
    # generator has a repeated eigenvalue and cannot use a spectral gap divide.
    alpha = 0.8
    total = np.array([[0.0, 1 - 1e-10, 1.0, 1 + 1e-10, 1e-100]])
    first = second = np.full_like(total, 0.2)
    args = (alpha * total, (1 - alpha) * total, first, second, alpha, 1.0, 1.0, 0.0, 1.0)
    np.testing.assert_allclose(flow_readout(*args), _matrix_reference(*args), rtol=3e-12, atol=0)


def test_permuted_buckets_and_zero_exposure_preserve_physical_readout() -> None:
    early, late, first, second, alpha = _policies()
    early[:, 0] = late[:, 0] = 0
    expected = flow_readout(early, late, first, second, alpha, 0.3, 1.5, 2, 0.5)
    order = [3, 1, 0, 2]
    actual = flow_readout(early[:, order], late[:, order], first[:, order], second[:, order], alpha, 0.3, 1.5, 2, 0.5)
    np.testing.assert_array_equal(actual, expected[:, order])
    np.testing.assert_array_equal(expected[:, 0], np.zeros(len(expected)))


def test_ratio_feature_has_finite_unclipped_low_dose_limit() -> None:
    first = np.array([[0.9, 0.1]])
    second = np.array([[0.1, 0.9]])
    alpha = 0.8
    scaled = []
    for dose in (1e-100, 1e-200):
        early = alpha * dose * first
        late = (1 - alpha) * dose * second
        actual = phase_benefit(early, late, first, second, alpha, 1, 1, 2, 0.5, "ratio")
        scaled.append(actual / dose)
    np.testing.assert_allclose(scaled[0], scaled[1], rtol=3e-12, atol=0)
    assert np.isfinite(scaled).all()
    assert (scaled[0] > 0).all()


def test_consolidation_protects_retention_against_labile_forgetting() -> None:
    args = (*_policies(), 0.3, 1.5, 2)
    scalar = flow_readout(*args, 0)
    protected = flow_readout(*args, 2)
    assert (protected >= scalar - 3e-14).all()
    assert np.max(protected - scalar) > 0.01
