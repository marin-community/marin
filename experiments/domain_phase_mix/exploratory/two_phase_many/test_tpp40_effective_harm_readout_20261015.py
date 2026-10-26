# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Pure readout invariants and counterexamples; no empirical rows or fits."""

from decimal import Decimal, localcontext

import numpy as np
import pytest

from experiments.domain_phase_mix.exploratory.two_phase_many import tpp40_effective_harm_readout_20261015 as law


@pytest.mark.parametrize("rate,power", [(0.05, 0.3), (0.25, 0.7), (4.0, 1.0)])
def test_null_is_native_and_both_channels_use_effective_exposure(rate: float, power: float) -> None:
    early = np.array([[0, 0], [1e-18, 2e-18], [0, 2], [3, 0], [200, 20]], dtype=float)
    late = np.array([[0, 0], [3e-18, 0], [1, 0], [0, 4], [1e-24, 5]], dtype=float)
    shape = {"rate": rate, "power": power, "threshold": 3.0}
    native = law.mariner.design_matrix(early + late, shape)
    null, offset = law.design_and_offset(early, late, 0.8, shape, "late_refresh_effective_harm", {"lambda": 0})
    np.testing.assert_array_equal(null, native)
    np.testing.assert_array_equal(offset, np.zeros(len(early)))
    for strength in (0.5, 1.0):
        matrix, _ = law.design_and_offset(early, late, 0.8, shape, "late_refresh_effective_harm", {"lambda": strength})
        physical, _ = law.refresh.design_and_offset(early, late, 0.8, shape, "late_refresh_hazard", {"lambda": strength})
        np.testing.assert_allclose(matrix[:, :2], physical[:, :2], atol=2e-15, rtol=2e-14)
        assert np.max(abs(matrix[:, 2:] - physical[:, 2:])) > 0.001
        np.testing.assert_array_equal(matrix[0], native[0])


@pytest.mark.parametrize("alpha", [0.8, 3040 / 3814, 1e-12, np.nextafter(1.0, 0.0)])
def test_tied_readout_preserves_exact_exposure_and_native_harm(alpha: float) -> None:
    total = np.array([[0, 0], [1e-18, 4e-18], [0.01, 1], [3, 20], [1e4, 1e6]])
    early, late = alpha * total, (1 - alpha) * total
    shape = {"rate": 0.25, "power": 0.7, "threshold": 3.0}
    for strength in (0.1, 0.5, 1.0):
        exposure = law.effective_exposure(early, late, alpha, 0.25, 0.7, strength)
        np.testing.assert_array_equal(exposure, early + late)
        matrix, _ = law.design_and_offset(early, late, alpha, shape, "late_refresh_effective_harm", {"lambda": strength})
        np.testing.assert_array_equal(matrix, law.mariner.design_matrix(early + late, shape))


def test_unit_refresh_makes_all_early_exposure_invisible_to_both_channels() -> None:
    early = np.array([[0], [1e-18], [0.1], [10], [1e6]])
    late = np.zeros_like(early)
    shape = {"rate": 0.25, "power": 0.7, "threshold": 3.0}
    for alpha in (0.8, np.nextafter(1.0, 0.0)):
        exposure = law.effective_exposure(early, late, alpha, 0.25, 0.7, 1.0)
        np.testing.assert_array_equal(exposure, np.zeros_like(early))
        matrix, _ = law.design_and_offset(early, late, alpha, shape, "late_refresh_effective_harm", {"lambda": 1})
        np.testing.assert_array_equal(matrix[:, 0], np.zeros(len(early)))
        expected_harm = float(np.logaddexp(-3.0, 0.0) ** 2)
        np.testing.assert_array_equal(matrix[:, 1], np.full(len(early), expected_harm))
        physical, _ = law.refresh.design_and_offset(early, late, alpha, shape, "late_refresh_hazard", {"lambda": 1})
        assert physical[-1, 1] > physical[0, 1] + 100
        retained = law.effective_exposure(early, late, alpha, 0.25, 0.7, 0.5)
        np.testing.assert_allclose(retained, early * 0.5 ** (1 / 0.7), atol=0, rtol=3e-16)


def test_decimal_reference_preserves_tiny_positive_refresh_and_zero_dose() -> None:
    rows = [(0.0, 0.0), (1e-18, 0.0), (1e-12, 1e-22), (0.1, 0.2), (200.0, 1e-24)]
    early, late = (np.array([row[index] for row in rows])[:, None] for index in (0, 1))
    for strength in (0.5, 1.0):
        expected = []
        with localcontext() as context:
            context.prec = 100
            rate, power, beta, weight = map(Decimal, ("0.25", "0.7", "0.2", str(strength)))
            for first, last in rows:
                total, final = Decimal(str(first)) + Decimal(str(last)), Decimal(str(last))
                if not total:
                    expected.append(0.0)
                    continue
                hazard = (power * (rate * total).ln()).exp()
                reference = (power * (rate * beta * total).ln()).exp()
                late_hazard = (power * (rate * final).ln()).exp() if final else Decimal(0)
                ratio = (1 - (-late_hazard).exp()) / (1 - (-reference).exp())
                effective_hazard = hazard * ((1 - weight) + weight * ratio)
                exposure = ((effective_hazard.ln() / power).exp() / rate) if effective_hazard else Decimal(0)
                expected.append(float(exposure))
        actual = law.effective_exposure(early, late, 0.8, 0.25, 0.7, strength).ravel()
        np.testing.assert_allclose(actual, expected, atol=0, rtol=3e-14)
        assert actual[-1] > 0


def test_effective_harm_admits_negative_even_cost_on_a_feasible_code_web_fiber() -> None:
    total = np.array([[10.0, 1.0]])
    late = np.array([[1.0, 0.3], [2.0, 0.2], [3.0, 0.1]])
    early, pools, alpha = total - late, np.array([0.05, 0.5]), 0.8
    phase0 = np.array([[0.5625, 0.4375], [0.5, 0.5], [0.4375, 0.5625]])
    phase1 = np.array([[0.25, 0.75], [0.5, 0.5], [0.75, 0.25]])
    np.testing.assert_allclose(early * pools, alpha * phase0, atol=2e-16, rtol=0)
    np.testing.assert_allclose(late * pools, (1 - alpha) * phase1, atol=3e-16, rtol=0)
    np.testing.assert_allclose(phase0.sum(axis=1), 1, atol=2e-16, rtol=0)
    np.testing.assert_allclose(phase1.sum(axis=1), 1, atol=3e-16, rtol=0)
    np.testing.assert_allclose(phase0[0], [0.5625, 0.4375], atol=1e-15, rtol=0)
    np.testing.assert_allclose(phase1[0], [0.25, 0.75], atol=1e-15, rtol=0)
    shape = {"rate": 1.0, "power": 1.0, "threshold": 1.0}
    matrix, _ = law.design_and_offset(early, late, alpha, shape, "late_refresh_effective_harm", {"lambda": 1})
    physical, _ = law.refresh.design_and_offset(early, late, alpha, shape, "late_refresh_hazard", {"lambda": 1})
    coefficients = np.array([0.01, 0, 0.1, 0])
    loss = 0.6 + np.exp(matrix @ coefficients)
    np.testing.assert_allclose(loss, [1.804604427098588, 1.8866332662944276, 1.91660837411384], atol=2e-14, rtol=0)
    even_cost = 0.5 * (loss[0] + loss[2]) - loss[1]
    assert even_cost == pytest.approx(-0.026026865688213663, abs=2e-14)
    physical_loss = 0.6 + np.exp(physical @ coefficients)
    assert 0.5 * (physical_loss[0] + physical_loss[2]) - physical_loss[1] > 0


def test_exposure_bound_and_small_dose_late_clock_limit() -> None:
    alpha, beta, rate, power = 0.8, 0.2, 0.25, 0.7
    total = np.array([[1e-12], [2e-12]])
    late = total * np.array([[0.1], [0.3]])
    exposure = law.effective_exposure(total - late, late, alpha, rate, power, 1.0)
    np.testing.assert_allclose(exposure, late / beta, atol=0, rtol=8e-9)
    total = np.array([[0.1, 20.0, 1e6]])
    for fraction in (0.0, 0.1, 0.2, 1.0):
        late = fraction * total
        exposure = law.effective_exposure(total - late, late, alpha, rate, power, 1.0)
        assert np.min(exposure) >= 0
        assert np.all(exposure <= total / beta + 1e-12 * np.maximum(1, total / beta))
        if fraction > 0:
            assert exposure[0, -1] == total[0, -1]
