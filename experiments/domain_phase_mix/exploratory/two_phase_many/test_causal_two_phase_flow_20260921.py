# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
from scipy.integrate import solve_ivp
from scipy.linalg import expm

from experiments.domain_phase_mix.exploratory.two_phase_many.causal_two_phase_flow_20260921 import (
    advance_causal_state,
    causal_flow_readout,
    causal_phase_benefit,
)


def _policies() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    rng = np.random.default_rng(20260921)
    first = rng.dirichlet(np.ones(4), 7)
    second = rng.dirichlet(np.ones(4), 7)
    first[0] = [0, 0.2, 0.3, 0.5]
    second[1] = [0.1, 0, 0.4, 0.5]
    scales = np.array([0.003, 0.5, 8, 150])
    alpha = 0.8
    return alpha * first * scales, (1 - alpha) * second * scales, first, second, alpha


def _reference_phase(
    state: np.ndarray,
    previous: float,
    added: float,
    weight: float,
    duration: float,
    rate: float,
    power: float,
    forgetting: float,
    consolidation: float,
) -> np.ndarray:
    # Independent physical-time parametrization t=duration*u**(1/power),
    # integrated with an implicit solver and all three conservation equations.
    # Production instead uses linear cumulative-clock coordinates and two states.
    def derivative(progress, current):
        elapsed = progress ** (1 / power)
        time_derivative = duration / power * progress ** (1 / power - 1)
        exposure = previous + added * elapsed
        if previous == 0:
            clock_derivative = (rate * added) ** power
        else:
            clock_derivative = rate**power * exposure ** (power - 1) * added * progress ** (1 / power - 1)
        unlearned, labile, _ = current
        acquired = clock_derivative * unlearned
        lost = forgetting * (1 - weight) * time_derivative * labile
        protected = consolidation * time_derivative * labile
        return [-acquired + lost, acquired - lost - protected, protected]

    solution = solve_ivp(derivative, (0, 1), state, method="Radau", rtol=1e-11, atol=1e-13)
    assert solution.success
    return solution.y[:, -1]


@pytest.mark.parametrize("rate,power,forgetting,consolidation", [(0.5, 0.3, 64, 16), (0.5, 0.7, 4, 1), (4, 1, 1, 0)])
def test_causal_flow_matches_independent_physical_clock_integration(rate, power, forgetting, consolidation) -> None:
    early, late, first, second, alpha = _policies()
    actual = causal_flow_readout(early, late, first, second, alpha, rate, power, forgetting, consolidation)
    expected = np.empty_like(actual)
    for index in np.ndindex(actual.shape):
        state = _reference_phase(
            np.array([1.0, 0.0, 0.0]), 0, early[index], first[index], alpha, rate, power, forgetting, consolidation
        )
        state = _reference_phase(
            state, early[index], late[index], second[index], 1 - alpha, rate, power, forgetting, consolidation
        )
        expected[index] = state[1:].sum()
    np.testing.assert_allclose(actual, expected, rtol=2e-8, atol=2e-10)
    assert actual.min() >= 0
    assert actual.max() <= 1 + 2e-10


def test_unit_power_matches_constant_generator_matrix_exponential() -> None:
    early, late, first, second, alpha = _policies()
    actual = causal_flow_readout(early, late, first, second, alpha, 0.5, 1, 4, 1)
    expected = np.empty_like(actual)
    for index in np.ndindex(actual.shape):
        state = np.array([1.0, 0.0, 0.0])
        for exposure, weight, duration in ((early[index], first[index], alpha), (late[index], second[index], 1 - alpha)):
            acquisition = 0.5 * exposure / duration
            loss = 4 * (1 - weight)
            generator = np.array([[-acquisition, loss, 0], [acquisition, -loss - 1, 0], [0, 1, 0]])
            state = expm(duration * generator) @ state
        expected[index] = state[1:].sum()
    np.testing.assert_allclose(actual, expected, rtol=2e-8, atol=2e-10)


@pytest.mark.parametrize("mode", ["additive", "ratio"])
def test_tied_centering_and_no_forgetting_preserve_mariner_law(mode) -> None:
    early, late, first, second, alpha = _policies()
    total = early + late
    aggregate = alpha * first + (1 - alpha) * second
    expected = -np.expm1(-((0.3 * total) ** 0.7))
    tied = causal_phase_benefit(alpha * total, (1 - alpha) * total, aggregate, aggregate, alpha, 0.3, 0.7, 4, 1, mode)
    no_forgetting = causal_phase_benefit(early, late, first, second, alpha, 0.3, 0.7, 0, 16, mode)
    np.testing.assert_allclose(tied, expected, rtol=2e-12, atol=2e-14)
    np.testing.assert_allclose(no_forgetting, expected, rtol=2e-12, atol=2e-14)


def test_prefix_state_survives_arbitrary_schedule_subdivision() -> None:
    early, late, first, second, alpha = _policies()
    state = np.zeros((*early.shape, 3))
    state[..., 0] = 1
    prefix = advance_causal_state(state, np.zeros_like(early), early, first, alpha, 0.5, 0.3, 16, 4)
    partitioned = state.copy()
    previous = np.zeros_like(early)
    for fraction in [0.013, 0.187, 0.6, 0.2]:
        partitioned = advance_causal_state(
            partitioned, previous, fraction * early, first, fraction * alpha, 0.5, 0.3, 16, 4
        )
        previous += fraction * early
    np.testing.assert_allclose(partitioned, prefix, rtol=3e-8, atol=3e-10)
    whole = advance_causal_state(prefix, early, late, second, 1 - alpha, 0.5, 0.3, 16, 4)
    resumed = advance_causal_state(partitioned, early, late, second, 1 - alpha, 0.5, 0.3, 16, 4)
    np.testing.assert_allclose(resumed, whole, rtol=3e-8, atol=3e-10)
    np.testing.assert_allclose(whole.sum(axis=-1), 1, rtol=0, atol=3e-15)
    assert whole.min() >= -3e-10
    # Different continuations branch from the same already-materialized prefix.
    reverse = advance_causal_state(prefix, early, late[:, ::-1], second[:, ::-1], 1 - alpha, 0.5, 0.3, 16, 4)
    assert np.max(np.abs(reverse - whole)) > 0.01


def test_without_late_exposure_loss_and_consolidation_have_analytic_solution() -> None:
    early, _, first, second, alpha = _policies()
    state = np.zeros((*early.shape, 3))
    state[..., 0] = 1
    prefix = advance_causal_state(state, np.zeros_like(early), early, first, alpha, 0.5, 0.7, 4, 1)
    result = advance_causal_state(prefix, early, np.zeros_like(early), second, 1 - alpha, 0.5, 0.7, 4, 1)
    total_decay = 4 * (1 - second) + 1
    lost_labile = prefix[..., 1] * (-np.expm1(-total_decay * (1 - alpha)))
    np.testing.assert_allclose(result[..., 1], prefix[..., 1] - lost_labile, rtol=1e-14, atol=1e-16)
    np.testing.assert_allclose(result[..., 2], prefix[..., 2] + lost_labile / total_decay, rtol=1e-14, atol=1e-16)


@pytest.mark.parametrize("power", [0.3, 1.0])
def test_ratio_remains_stable_at_tiny_and_zero_exposure(power) -> None:
    first = np.array([[0.9, 0.1, 0]])
    second = np.array([[0.1, 0.9, 0]])
    values = []
    for dose in [1e-100, 1e-200]:
        result = causal_phase_benefit(
            0.8 * dose * first, 0.2 * dose * second, first, second, 0.8, 1, power, 4, 1, "ratio"
        )
        values.append(result / dose**power)
        assert np.isfinite(result).all()
        assert result[0, 2] == 0
        assert (result[0, :2] > 0).all()
    np.testing.assert_allclose(values[0], values[1], rtol=3e-8, atol=3e-10)


def test_independent_bucket_permutation_preserves_retention() -> None:
    early, late, first, second, alpha = _policies()
    expected = causal_flow_readout(early, late, first, second, alpha, 0.3, 0.7, 4, 1)
    order = [3, 1, 0, 2]
    actual = causal_flow_readout(
        early[:, order], late[:, order], first[:, order], second[:, order], alpha, 0.3, 0.7, 4, 1
    )
    np.testing.assert_allclose(actual, expected[:, order], rtol=2e-10, atol=2e-12)


@pytest.mark.parametrize("forgetting", [0.0, 4.0])
def test_saturated_acquisition_preserves_positive_unlearned_tail(forgetting) -> None:
    added = np.asarray([[20.912861229854922, 0.5]])
    weights = np.asarray([[1.0, 0.1]])
    state = np.zeros((*added.shape, 3))
    state[..., 0] = 1
    result = advance_causal_state(state, np.zeros_like(added), added, weights, 0.8, 4, 1, forgetting, 0)
    expected_tail = np.exp(-4 * added[0, 0])
    np.testing.assert_allclose(result[0, 0, 0], expected_tail, rtol=2e-14, atol=0)
    assert result[0, 0, 0] > 0
    continued = advance_causal_state(result, added, 0.1 * added, weights, 0.2, 4, 1, forgetting, 0)
    np.testing.assert_allclose(continued[0, 0, 0], np.exp(-4.4 * added[0, 0]), rtol=2e-14, atol=0)
