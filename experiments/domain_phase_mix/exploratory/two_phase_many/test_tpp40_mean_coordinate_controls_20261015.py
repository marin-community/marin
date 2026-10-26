# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Pure equations and feasibility examples; no empirical data or fitted heads."""

from decimal import Decimal, localcontext

import numpy as np
import pytest
from scipy.optimize import brentq

from experiments.domain_phase_mix.exploratory.two_phase_many import tpp40_mean_coordinate_controls_20261015 as law

FAMILIES = ("hazard_coordinate_mean", "benefit_coordinate_mean")


@pytest.mark.parametrize("model", FAMILIES)
@pytest.mark.parametrize("rate,power", [(0.05, 0.3), (0.25, 0.7), (4.0, 1.0)])
def test_null_tied_zero_and_physical_harm_match_native(model: str, rate: float, power: float) -> None:
    alpha = 3040 / 3814
    total = np.array([[0.0, 0.0], [1e-20, 4e-20], [0.01, 1.0], [3.0, 20.0], [1e4, 1e6]])
    shape = {"rate": rate, "power": power, "threshold": 3.0}
    early, late = alpha * total, (1 - alpha) * total
    native = law.mariner.design_matrix(early + late, shape)
    for strength in law.STRENGTHS:
        matrix, offset = law.design_and_offset(early, late, alpha, shape, model, {"lambda": strength})
        np.testing.assert_array_equal(matrix, native)
        np.testing.assert_array_equal(offset, np.zeros(len(total)))
    early, late = total * np.array([0.2, 1.0]), total * np.array([0.8, 0.0])
    native = law.mariner.design_matrix(early + late, shape)
    for strength in (0.0, 0.5, 1.0):
        matrix, _ = law.design_and_offset(early, late, alpha, shape, model, {"lambda": strength})
        np.testing.assert_array_equal(matrix[:, 2:], native[:, 2:])
        np.testing.assert_array_equal(matrix[0], native[0])
        assert np.all((matrix[:, :2] >= -1) & (matrix[:, :2] <= 0))
        if strength == 0:
            np.testing.assert_array_equal(matrix, native)


@pytest.mark.parametrize("model", FAMILIES)
def test_decimal_reference_preserves_small_doses_and_refresh(model: str) -> None:
    rows = [(0.0, 0.0), (1e-20, 0.0), (1e-12, 1e-25), (0.1, 0.2), (20.0, 0.0), (0.0, 40.0), (200.0, 1e-24)]
    early, late = (np.array([row[i] for row in rows])[:, None] for i in (0, 1))
    for strength in (0.1, 0.5, 1.0):
        expected = []
        with localcontext() as context:
            context.prec = 100
            rate, power, beta, weight = map(Decimal, ("0.25", "0.7", "0.2", str(strength)))
            for first, last in rows:
                total = Decimal(str(first)) + Decimal(str(last))
                normalized_late = Decimal(str(last)) / beta
                native_hazard = (power * (rate * total).ln()).exp() if total else Decimal(0)
                late_hazard = (power * (rate * normalized_late).ln()).exp() if normalized_late else Decimal(0)
                if model == "hazard_coordinate_mean":
                    value = 1 - (-((1 - weight) * native_hazard + weight * late_hazard)).exp()
                else:
                    value = (1 - weight) * (1 - (-native_hazard).exp()) + weight * (1 - (-late_hazard).exp())
                expected.append(float(value))
        actual = law.mean_benefit(early, late, 0.8, 0.25, 0.7, model, strength).ravel()
        np.testing.assert_allclose(actual, expected, rtol=3e-14, atol=0)
        assert actual[-1] > 0


@pytest.mark.parametrize("model", FAMILIES)
def test_tiny_phase_and_tiny_asymmetry_are_not_spurious_ties(model: str) -> None:
    alpha = np.nextafter(1.0, 0.0)
    early, late = np.ones((1, 1)), np.zeros((1, 1))
    assert law.mean_benefit(early, late, alpha, 0.25, 0.7, model, 1.0)[0, 0] == 0
    tied = law.mean_benefit(alpha * early, (1 - alpha) * early, alpha, 0.25, 0.7, model, 0.5)
    np.testing.assert_array_equal(tied, law.mariner.benefit(early, 0.25, 0.7))
    early, late = np.array([[1e-20]]), np.array([[0.0]])
    actual = law.mean_benefit(early, late, 0.8, 0.25, 0.7, model, 1.0)
    assert actual[0, 0] == 0
    assert law.mariner.benefit(early, 0.25, 0.7)[0, 0] > 0


@pytest.mark.parametrize("power", [0.3, 0.7, 1.0])
def test_endpoints_low_dose_ordering_and_high_dose_limits(power: float) -> None:
    beta, rate = 0.2, 0.25
    total = np.array([[1e-20], [0.1], [8.0], [1e5]])
    zero = np.zeros_like(total)
    native_hazard = (rate * total) ** power
    for strength in law.STRENGTHS:
        hazard_early = law.mean_benefit(total, zero, 0.8, rate, power, FAMILIES[0], strength)
        benefit_early = law.mean_benefit(total, zero, 0.8, rate, power, FAMILIES[1], strength)
        np.testing.assert_allclose(hazard_early, -np.expm1(-(1 - strength) * native_hazard), rtol=1e-14, atol=0)
        np.testing.assert_allclose(
            benefit_early, (1 - strength) * law.mariner.benefit(total, rate, power), rtol=1e-14, atol=0
        )
        for model in FAMILIES:
            late_only = law.mean_benefit(zero, total, 0.8, rate, power, model, 1.0)
            np.testing.assert_allclose(late_only, law.mariner.benefit(total / beta, rate, power), rtol=1e-14, atol=0)
    enormous = np.array([[1e10]])
    assert law.mean_benefit(enormous, zero[:1], 0.8, rate, power, FAMILIES[0], 0.5)[0, 0] == 1
    assert law.mean_benefit(enormous, zero[:1], 0.8, rate, power, FAMILIES[1], 0.5)[0, 0] == 0.5
    fractions = np.linspace(0, 1, 31)[None, :]
    early, late = total * (1 - fractions), total * fractions
    for strength in (0.1, 0.5, 0.9):
        benefit_mean = law.mean_benefit(early, late, 0.8, rate, power, FAMILIES[1], strength)
        hazard_mean = law.mean_benefit(early, late, 0.8, rate, power, FAMILIES[0], strength)
        clock = law.mariner.benefit((1 - strength) * total + strength * late / beta, rate, power)
        assert np.max(benefit_mean - hazard_mean) <= 3e-15
        assert np.max(hazard_mean - clock) <= 3e-15
        np.testing.assert_allclose(benefit_mean[0], hazard_mean[0], rtol=3e-6, atol=0)
        if power == 1:
            np.testing.assert_allclose(hazard_mean, clock, rtol=3e-14, atol=1e-15)
        else:
            assert np.max(clock - hazard_mean) > 1e-3
        assert np.max(hazard_mean - benefit_mean) > 1e-3


@pytest.mark.parametrize("model", FAMILIES)
@pytest.mark.parametrize("power", [0.3, 0.7, 1.0])
def test_common_tied_derivative_and_late_zero_cusp(model: str, power: float) -> None:
    alpha, beta, rate, strength = 0.8, 0.2, 0.25, 0.5
    total = np.array([[0.1, 1.0, 10.0]])
    step = total * 1e-5
    minus = law.mean_benefit(alpha * total + step, beta * total - step, alpha, rate, power, model, strength)
    plus = law.mean_benefit(alpha * total - step, beta * total + step, alpha, rate, power, model, strength)
    minus_twice = law.mean_benefit(
        alpha * total + 2 * step, beta * total - 2 * step, alpha, rate, power, model, strength
    )
    plus_twice = law.mean_benefit(alpha * total - 2 * step, beta * total + 2 * step, alpha, rate, power, model, strength)
    derivative = (minus_twice - 8 * minus + 8 * plus - plus_twice) / (12 * step)
    expected = strength / beta * np.exp(-((rate * total) ** power)) * power * rate**power * total ** (power - 1)
    np.testing.assert_allclose(derivative, expected, rtol=2e-9, atol=1e-11)
    if power < 1:
        late = np.array([[1e-6], [1e-9]])
        absent = law.mean_benefit(np.array([[4.0]]), np.zeros((1, 1)), alpha, rate, power, model, strength)[0, 0]
        slopes = (law.mean_benefit(4 - late, late, alpha, rate, power, model, strength)[:, 0] - absent) / late[:, 0]
        assert slopes[1] > 5 * slopes[0]


@pytest.mark.parametrize("model", FAMILIES)
@pytest.mark.parametrize("power", [0.3, 0.7, 1.0])
def test_unclipped_loss_is_convex_on_feasible_fixed_aggregate_fiber(model: str, power: float) -> None:
    alpha, beta = 0.8, 0.2
    total, inventory = np.array([10.0, 1.0]), np.array([20.0, 2.0])
    position = np.linspace(0, 1, 101)
    late_policy = np.column_stack([position, 1 - position])
    late = beta * inventory * late_policy
    early = total - late
    assert np.min(early) >= 0
    np.testing.assert_allclose((early / (alpha * inventory)).sum(axis=1), 1, atol=2e-15)
    shape = {"rate": 0.25, "power": power, "threshold": 3.0}
    coefficients = np.array([0.8, 1.1, 0.03, 0.07])
    for strength in (0.0, 0.5, 1.0):
        matrix, _ = law.design_and_offset(early, late, alpha, shape, model, {"lambda": strength})
        eta = 0.2 + matrix @ coefficients
        assert np.max(abs(eta)) < 30
        loss = 0.6 + np.exp(eta)
        assert np.min(np.diff(loss, n=2)) >= -2e-14
        assert np.max(loss - ((1 - position) * loss[0] + position * loss[-1])) <= 2e-14
        if strength == 0:
            np.testing.assert_allclose(loss, np.full_like(loss, loss[0]), rtol=0, atol=1e-15)


@pytest.mark.parametrize("model", FAMILIES)
def test_common_strength_static_policy_dominates_benefit_only_heads(model: str) -> None:
    alpha, beta, strength = 0.8, 0.2, 0.5
    first, last = np.array([0.1, 0.6, 0.3]), np.array([0.7, 0.1, 0.2])
    inventory = np.array([12.0, 3.0, 20.0])
    aggregate = alpha * first + beta * last
    static = (1 - strength) * aggregate + strength * last
    assert np.min(static) >= 0
    assert static.sum() == pytest.approx(1, abs=2e-15)
    early, late = (alpha * inventory * first)[None, :], (beta * inventory * last)[None, :]
    for rate, power, amplitudes in [(0.25, 0.7, np.array([1.0, 0.5, 0.2])), (0.5, 0.3, np.array([0.2, 0.8, 0.4]))]:
        phase = law.mean_benefit(early, late, alpha, rate, power, model, strength)[0]
        dominating = law.mariner.benefit(inventory * static, rate, power)
        assert np.max(phase - dominating) <= 2e-15
        target = float(phase @ amplitudes)
        endpoint_values = [
            float(law.mariner.benefit(inventory * item, rate, power) @ amplitudes) for item in (aggregate, last)
        ]
        endpoint = (aggregate, last)[int(np.argmin(endpoint_values))]

        def residual(
            fraction: float,
            endpoint: np.ndarray,
            rate: float,
            power: float,
            amplitudes: np.ndarray,
            target: float,
        ) -> float:
            policy = (1 - fraction) * endpoint + fraction * static
            return float(law.mariner.benefit(inventory * policy, rate, power) @ amplitudes) - target

        # This scalar equation demonstrates the continuity argument; no head is estimated.
        arguments = (endpoint, rate, power, amplitudes, target)
        match = brentq(residual, 0, 1, args=arguments, xtol=1e-14)
        assert abs(residual(match, *arguments)) < 2e-14
