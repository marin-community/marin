# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Equation contracts and counterexamples, using no empirical policy/outcome data."""

from decimal import Decimal, localcontext

import numpy as np
import pytest

from experiments.domain_phase_mix.exploratory.two_phase_many import tpp40_late_refresh_20261015 as law


def benefit(early: np.ndarray, late: np.ndarray, alpha: float, rate: float, power: float, strength: float) -> np.ndarray:
    return -np.expm1(-law.effective_hazard(early, late, alpha, rate, power, strength))


@pytest.mark.parametrize("rate,power", [(0.05, 0.3), (0.25, 0.7), (4.0, 1.0)])
def test_null_and_physical_harm_preserve_native_design(rate: float, power: float) -> None:
    early = np.array([[0, 0], [0, 2], [3, 0], [1e-18, 2e-18], [1e4, 1e6]], dtype=float)
    late = np.array([[0, 0], [1, 0], [0, 4], [3e-18, 0], [1e6, 1e4]], dtype=float)
    shape = {"rate": rate, "power": power, "threshold": 3.0}
    native = law.mariner.design_matrix(early + late, shape)
    null, offset = law.design_and_offset(early, late, 0.8, shape, "late_refresh_hazard", {"lambda": 0})
    np.testing.assert_array_equal(null, native)
    np.testing.assert_array_equal(offset, np.zeros(len(early)))
    for strength in (0.5, 1.0):
        matrix, _ = law.design_and_offset(early, late, 0.8, shape, "late_refresh_hazard", {"lambda": strength})
        np.testing.assert_array_equal(matrix[:, 2:], native[:, 2:])
        assert np.all((matrix[:, :2] >= -1) & (matrix[:, :2] <= 0))
        np.testing.assert_array_equal(matrix[0], native[0])


@pytest.mark.parametrize("alpha", [0.8, 3040 / 3814, 21856 / 27336])
@pytest.mark.parametrize("strength", [0.1, 0.5, 1.0])
def test_tied_design_is_exact_at_zero_tiny_and_saturated_doses(alpha: float, strength: float) -> None:
    total = np.array([[0, 0], [1e-18, 4e-18], [0.01, 1], [3, 20], [1e4, 1e6]])
    early, late = alpha * total, (1 - alpha) * total
    shape = {"rate": 0.25, "power": 0.7, "threshold": 3.0}
    matrix, _ = law.design_and_offset(early, late, alpha, shape, "late_refresh_hazard", {"lambda": strength})
    np.testing.assert_array_equal(matrix, law.mariner.design_matrix(early + late, shape))


def test_decimal_reference_including_tiny_positive_refresh() -> None:
    rows = [(1e-18, 0.0), (1e-12, 1e-22), (0.1, 0.2), (20.0, 0.0), (0.0, 40.0), (200.0, 1e-24)]
    early, late = (np.array([item[index] for item in rows])[:, None] for index in (0, 1))
    for strength in (0.5, 1.0):
        expected = []
        with localcontext() as context:
            context.prec = 90
            rate, power, beta, weight = map(Decimal, ("0.25", "0.7", "0.2", str(strength)))
            for e0, e1 in rows:
                total, last = Decimal(str(e0)) + Decimal(str(e1)), Decimal(str(e1))
                hazard = (power * (rate * total).ln()).exp()
                reference_hazard = (power * (rate * beta * total).ln()).exp()
                final_hazard = (power * (rate * last).ln()).exp() if last else Decimal(0)
                ratio = (1 - (-final_hazard).exp()) / (1 - (-reference_hazard).exp())
                value = 1 - (-(hazard * ((1 - weight) + weight * ratio))).exp()
                expected.append(float(value))
        actual = benefit(early, late, 0.8, 0.25, 0.7, strength).ravel()
        np.testing.assert_allclose(actual, expected, atol=2e-14, rtol=2e-13)
        if strength == 1:
            assert actual[0] == actual[3] == 0
            assert actual[-1] > 0  # Cancellation in 1 + lambda*(R-1) would incorrectly produce zero.


def test_vanishing_phase_duration_does_not_turn_dropout_into_a_tied_policy() -> None:
    alpha = np.nextafter(1.0, 0.0)
    early, late = np.array([[1.0]]), np.array([[0.0]])
    native = (0.25 * early) ** 0.7
    for strength in (0.5, 1.0):
        actual = law.effective_hazard(early, late, alpha, 0.25, 0.7, strength)
        np.testing.assert_array_equal(actual, native * (1 - strength))
        tied = law.effective_hazard(alpha * early, (1 - alpha) * early, alpha, 0.25, 0.7, strength)
        np.testing.assert_array_equal(tied, native)


@pytest.mark.parametrize("power", [0.3, 0.7, 1.0])
def test_dose_bound_and_monotonicity_in_each_physical_exposure(power: float) -> None:
    doses = np.r_[0.0, np.logspace(-18, 6, 100)]
    early, late = np.meshgrid(doses, doses)
    early, late = early.ravel()[:, None], late.ravel()[:, None]
    total, beta, rate = early + late, 0.2, 0.25
    native_hazard = (rate * total) ** power
    for strength in (0.0, 0.5, 1.0):
        hazard = law.effective_hazard(early, late, 0.8, rate, power, strength)
        assert np.all(hazard >= native_hazard * (1 - strength) - 1e-14)
        bound = native_hazard * (1 - strength + strength * beta**-power)
        assert np.all(hazard <= bound + 1e-12 * np.maximum(1, bound))
        effective = hazard ** (1 / power) / rate
        assert np.all(effective <= total / beta + 1e-12 * np.maximum(1, total / beta))
        values = (-np.expm1(-hazard)).reshape(len(doses), len(doses))
        assert np.min(np.diff(values, axis=0)) >= -2e-14
        assert np.min(np.diff(values, axis=1)) >= -2e-14


def test_low_dose_limit_is_late_clock_but_finite_dose_is_not_a_static_mixture() -> None:
    rate, power, beta = 0.25, 0.7, 0.2
    total = np.array([[1e-12], [2e-12]])
    late = total * np.array([[0.1], [0.3]])
    early = total - late
    hazard = law.effective_hazard(early, late, 0.8, rate, power, 1.0)
    np.testing.assert_allclose(hazard, (rate * late / beta) ** power, rtol=5e-9, atol=0)
    total = np.array([[20.0, 0.2]])
    late = total * np.array([[0.1, 0.3]])
    pools = 0.5 / total
    phase0 = (total - late) * pools / 0.8
    phase1 = late * pools / beta
    np.testing.assert_allclose(phase0.sum(), 1, atol=1e-15)
    np.testing.assert_allclose(phase1.sum(), 1, atol=1e-15)
    effective = law.effective_hazard(total - late, late, 0.8, 1.0, 1.0, 1.0)
    mass = float(np.sum(effective * pools))
    assert mass == pytest.approx(1.1829977923123187, abs=2e-14)
    assert mass > 1  # Per-bucket effective-dose bounds do not produce one feasible static policy.


def test_late_dropout_cusp_and_nonuniform_high_dose_limit() -> None:
    total, rate, power = 4.0, 0.25, 0.7
    late = np.array([[1e-7], [1e-10]])
    absent = benefit(np.array([[total]]), np.zeros((1, 1)), 0.8, rate, power, 0.5)[0, 0]
    slopes = (benefit(total - late, late, 0.8, rate, power, 0.5)[:, 0] - absent) / late[:, 0]
    assert slopes[1] > 5 * slopes[0]
    exposure = np.array([[1e4]])
    for fraction in (0.1, 0.2, 1.0):
        hazard = law.effective_hazard((1 - fraction) * exposure, fraction * exposure, 0.8, 1.0, 1.0, 1.0)
        np.testing.assert_array_equal(hazard, exposure)
    zero = np.zeros_like(exposure)
    assert benefit(exposure, zero, 0.8, 1.0, 1.0, 1.0)[0, 0] == 0
    assert benefit(exposure, zero, 0.8, 1.0, 1.0, 0.9)[0, 0] == 1
    # Native b(E) rounds to one, but finite hazard still permits a late-dropout effect.
    assert law.mariner.benefit(np.array([[10.0]]), 4.0, 1.0)[0, 0] == 1
    value = benefit(np.array([[10.0]]), np.zeros((1, 1)), 0.8, 4.0, 1.0, 0.9)[0, 0]
    assert value == pytest.approx(-np.expm1(-4.0), abs=2e-15)


@pytest.mark.parametrize("power", [0.3, 0.7, 1.0])
def test_unclipped_loss_is_convex_along_a_feasible_fixed_aggregate_phase_simplex(power: float) -> None:
    alpha, beta = 0.8, 0.2
    total, pools = np.array([10.0, 1.0]), np.array([0.05, 0.5])
    allocations = np.linspace(0, 1, 101)
    phase1 = np.column_stack([allocations, 1 - allocations])
    late = beta * phase1 / pools
    early = total - late
    np.testing.assert_allclose((early * pools / alpha).sum(axis=1), 1, atol=2e-15)
    assert np.min(early) >= 0
    shape = {"rate": 0.25, "power": power, "threshold": 3.0}
    coefficients = np.array([0.8, 1.1, 0.03, 0.07])
    for strength in (0.0, 0.5, 1.0):
        matrix, _ = law.design_and_offset(early, late, alpha, shape, "late_refresh_hazard", {"lambda": strength})
        linear = 0.2 + matrix @ coefficients
        assert np.max(abs(linear)) < 30
        loss = 0.6 + np.exp(linear)
        if strength == 0:
            np.testing.assert_allclose(loss, np.full_like(loss, loss[0]), atol=1e-15, rtol=0)
        assert np.min(np.diff(loss, n=2)) >= -2e-14
        chord = (1 - allocations) * loss[0] + allocations * loss[-1]
        assert np.max(loss - chord) <= 2e-14
