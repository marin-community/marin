# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Pure-law checks on invented policies only; no empirical geometry or outcomes."""

from dataclasses import replace

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from experiments.domain_phase_mix.exploratory.two_phase_many import tpp40_causal_acquisition_retention_20261015 as model

CONTROL = model.Quadrature(1e-11, 512)
FRACTIONS = np.array([0.8, 0.2])
WEIGHTS = np.array([[0.2, 0.8], [0.8, 0.2]])
SCALES = np.array([10.0, 2.0])


def law(power: float, mu: float) -> model.Law:
    return model.Law(0.25, power, mu, ("first", "second"), np.array([0.5, 0.5]))


def ode_mastery(phases: np.ndarray, parameters: model.Law) -> np.ndarray:
    """Independent moderate-dose state ODE, not integrating-factor quadrature."""
    start, mastery = np.zeros(2), np.zeros(2)
    for dose in phases:

        def derivative(unit: float, state: np.ndarray, start: np.ndarray = start, dose: np.ndarray = dose) -> np.ndarray:
            physical_time = unit ** (1 / parameters.power)
            epochs = start + physical_time * dose
            hazard = (parameters.rate * epochs) ** parameters.power
            velocity = np.zeros(2)
            for bucket in range(2):
                if dose[bucket] == 0:
                    continue
                if start[bucket] == 0:
                    velocity[bucket] = (parameters.rate * dose[bucket]) ** parameters.power
                elif unit > 0:
                    velocity[bucket] = (
                        parameters.rate**parameters.power
                        * dose[bucket]
                        * unit ** (1 / parameters.power - 1)
                        * epochs[bucket] ** (parameters.power - 1)
                    )
                elif parameters.power == 1:
                    velocity[bucket] = parameters.rate * dose[bucket]
            competitor = (np.exp(-hazard) * velocity * parameters.masses)[::-1]
            return (1 - state) * velocity - parameters.mu * state * competitor

        result = solve_ivp(derivative, (0, 1), mastery, method="DOP853", rtol=2e-12, atol=2e-14)
        assert result.success
        mastery = result.y[:, -1]
        start += dose
    return mastery


@pytest.mark.parametrize("power", [0.3, 0.7, 1.0])
def test_null_tied_and_zero_dose_native_restriction(power: float) -> None:
    for scales in (np.zeros(2), np.array([1e-20, 2e-20]), SCALES, np.array([1e5, 1e6])):
        for mu, weights in ((0.0, WEIGHTS), (2.0, np.tile([0.3, 0.7], (2, 1)))):
            parameters = law(power, mu)
            actual = model.evaluate_policy(weights, FRACTIONS, scales, parameters, CONTROL)
            total = model.policy_epochs(weights, FRACTIONS, scales).sum(axis=0)
            expected = -np.expm1(-((parameters.rate * total) ** power))
            np.testing.assert_array_equal(actual.benefit, expected)
            assert actual.exact_native_branch


@pytest.mark.parametrize("power", [0.3, 0.7, 1.0])
@pytest.mark.parametrize("mu", [0.4, 4.0])
def test_positive_quadrature_agrees_with_independent_ode_and_order_bounds(power: float, mu: float) -> None:
    parameters = law(power, mu)
    phases = model.policy_epochs(WEIGHTS, FRACTIONS, SCALES)
    actual = model.evaluate_policy(WEIGHTS, FRACTIONS, SCALES, parameters, CONTROL)
    own = ode_mastery(phases, parameters)
    tied = ode_mastery(phases.sum(axis=0)[None], parameters)
    total_hazard = (parameters.rate * phases.sum(axis=0)) ** power
    expected_hazard = total_hazard - np.log1p(-own) + np.log1p(-tied)
    np.testing.assert_allclose(actual.effective_hazard, expected_hazard, rtol=2e-10, atol=2e-11)
    np.testing.assert_allclose(np.exp(actual.path.log_mastery), own, rtol=2e-10, atol=2e-11)
    attenuation = np.exp(-actual.tied.deficit(parameters))
    centered_mastery = -np.expm1(-total_hazard) + attenuation * (own - tied)
    np.testing.assert_allclose(actual.benefit, centered_mastery, rtol=0, atol=2e-11)
    assert np.all(attenuation >= 0) and np.all(attenuation <= 1)
    coarse = model.policy_ordering_bounds(WEIGHTS, FRACTIONS, SCALES, parameters, 32)
    fine = model.policy_ordering_bounds(WEIGHTS, FRACTIONS, SCALES, parameters, 128)
    assert np.all(actual.benefit >= fine.lower - 2e-12)
    assert np.all(actual.benefit <= fine.upper + 2e-12)
    assert np.max(fine.upper - fine.lower) < 0.3 * np.max(coarse.upper - coarse.lower)
    assert np.all(actual.effective_hazard >= 0)
    assert np.all(actual.effective_hazard <= 2 * total_hazard + 2e-12)


@pytest.mark.parametrize("power", [0.3, 0.7, 1.0])
def test_pure_blocks_have_exact_order_sensitive_transitions(power: float) -> None:
    parameters = law(power, 3.0)
    phases = np.array([[4.0, 0.0], [0.0, 2.0]])
    first = model.integrate(phases, parameters, CONTROL)
    reversed_path = model.integrate(phases[::-1], parameters, CONTROL)
    hazards = (parameters.rate * phases.sum(axis=0)) ** power
    native = -np.expm1(-hazards)
    expected = native.copy()
    expected[0] *= np.exp(-parameters.mu * parameters.masses[1] * native[1])
    reverse_expected = native.copy()
    reverse_expected[1] *= np.exp(-parameters.mu * parameters.masses[0] * native[0])
    np.testing.assert_allclose(np.exp(first.log_mastery), expected, atol=2e-15, rtol=2e-14)
    np.testing.assert_allclose(np.exp(reversed_path.log_mastery), reverse_expected, atol=2e-15, rtol=2e-14)
    assert first.evaluations == reversed_path.evaluations == 0
    assert first.survival_hazard(parameters)[0] < reversed_path.survival_hazard(parameters)[0]


def test_log_survival_retains_tiny_interference_after_saturated_acquisition() -> None:
    parameters = model.Law(1.0, 1.0, 1e-250, ("first", "second"), np.array([0.5, 0.5]))
    state = model.integrate(np.array([[1000.0, 0.0], [0.0, 1.0]]), parameters, CONTROL)
    log_k = np.log(parameters.mu) + np.log(0.5) + np.log(-np.expm1(-1.0))
    # q=e^-1000+(1-e^-1000)*(1-e^-k); log(q) is log(k) to much better than double precision.
    assert state.survival_hazard(parameters)[0] == pytest.approx(-log_k, abs=2e-12)
    assert state.deficit(parameters)[0] == pytest.approx(1000 + log_k, abs=2e-12)
    assert np.isfinite(state.survival_hazard(parameters)).all()


def test_prefix_causality_and_phase_splitting_preserve_the_path() -> None:
    parameters = law(0.3, 2.0)
    phases = model.policy_epochs(WEIGHTS, FRACTIONS, SCALES)
    prefix = model.integrate(phases[:1], parameters, CONTROL)
    direct = model.integrate(phases, parameters, CONTROL)
    resumed = model.advance(prefix, phases[1], parameters, CONTROL)
    changed_future = model.advance(prefix, phases[1, ::-1], parameters, CONTROL)
    np.testing.assert_array_equal(prefix.epochs, phases[0])
    np.testing.assert_array_equal(resumed.log_mastery, direct.log_mastery)
    assert np.max(abs(changed_future.log_mastery - resumed.log_mastery)) > 1e-3
    pieces = np.vstack([np.tile(phases[0] / 4, (4, 1)), np.tile(phases[1] / 3, (3, 1))])
    divided = model.integrate(pieces, parameters, CONTROL)
    np.testing.assert_allclose(
        divided.survival_hazard(parameters), direct.survival_hazard(parameters), atol=2e-11, rtol=0
    )
    np.testing.assert_allclose(divided.deficit(parameters), direct.deficit(parameters), atol=2e-11, rtol=0)


def test_near_ties_are_continuous_without_a_finite_tied_band() -> None:
    parameters = law(0.7, 2.0)
    tied_weights = np.tile([0.3, 0.7], (2, 1))
    tied = model.evaluate_policy(tied_weights, FRACTIONS, SCALES, parameters, CONTROL)
    errors = []
    for displacement in (1e-3, 1e-4, 1e-5):
        weights = tied_weights.copy()
        weights[0] += np.array([displacement, -displacement])
        weights[1] -= 4 * np.array([displacement, -displacement])
        changed = model.evaluate_policy(weights, FRACTIONS, SCALES, parameters, CONTROL)
        assert not changed.exact_native_branch
        errors.append(np.max(abs(changed.benefit - tied.benefit)))
    assert errors[2] < 0.02 * errors[0]
    assert errors[2] > 1e-9


def test_zero_mu_derivative_matches_closed_form_capacity_example_and_finite_difference() -> None:
    parameters = model.Law(1.0, 1.0, 0.0, ("first", "second"), np.array([0.5, 0.5]))
    weights = np.array([[5 / 44, 39 / 44], [0.0, 1.0]])
    scales = np.full(2, 2.2)
    response = model.zero_mu_response(weights, FRACTIONS, scales, parameters, CONTROL)
    expected = np.exp(-np.array([0.2, 2.0])) * np.array([-0.008285011824549, 0.067580580407280])
    np.testing.assert_allclose(response.derivative, expected, atol=2e-14, rtol=2e-12)
    amplitude = np.exp([0.2, 2.0])
    assert amplitude @ response.derivative == pytest.approx(0.0592955685827313, abs=2e-13)
    baseline = model.evaluate_policy(weights, FRACTIONS, scales, parameters, CONTROL)
    errors = []
    for mu in (1e-3, 1e-4):
        shifted = model.evaluate_policy(weights, FRACTIONS, scales, replace(parameters, mu=mu), CONTROL)
        finite_difference = (shifted.benefit - baseline.benefit) / mu
        errors.append(np.max(abs(finite_difference - response.derivative)))
    assert errors[1] < 0.12 * errors[0]
    assert errors[1] < 2e-6


def test_fixed_mass_bucket_refinement_preserves_interference() -> None:
    original = law(0.7, 2.0)
    epochs = np.array([[0.3, 2.0], [1.2, 0.2]])
    first = model.integrate(epochs, original, CONTROL)
    split = model.Law(0.25, 0.7, 2.0, ("first", "second", "second"), np.array([0.5, 0.2, 0.3]))
    refined = model.integrate(epochs[:, [0, 1, 1]], split, CONTROL)
    np.testing.assert_allclose(refined.survival_hazard(split), first.survival_hazard(original)[[0, 1, 1]], atol=2e-12)
    np.testing.assert_allclose(refined.deficit(split), first.deficit(original)[[0, 1, 1]], atol=2e-12)


@pytest.mark.parametrize("mu,expected_even", [(1.0, 0.0007810756378577), (10.0, -0.0006745384933903)])
def test_opposite_fixed_aggregate_contrasts_allow_both_even_loss_signs(mu: float, expected_even: float) -> None:
    parameters = model.Law(1.0, 1.0, mu, ("first", "second"), np.array([0.5, 0.5]))
    scales = np.full(2, 2.0)
    policies = np.array([[[0.58, 0.42], [0.18, 0.82]], [[0.5, 0.5], [0.5, 0.5]], [[0.42, 0.58], [0.82, 0.18]]])
    benefits, ode_benefits = [], []
    tied_mastery = ode_mastery(np.array([[1.0, 1.0]]), parameters)
    for weights in policies:
        # Opposite contrasts at alpha=.8 preserve the aggregate; swapping phases would not.
        phases = model.policy_epochs(weights, FRACTIONS, scales)
        np.testing.assert_allclose(phases.sum(axis=0), np.ones(2), rtol=0, atol=2e-16)
        result = model.evaluate_policy(weights, FRACTIONS, scales, parameters, CONTROL)
        benefits.append(result.benefit)
        mastery = ode_mastery(phases, parameters)
        ode_benefits.append(1 - np.exp(-1) * (1 - mastery) / (1 - tied_mastery))
    np.testing.assert_allclose(benefits, ode_benefits, rtol=0, atol=2e-11)
    # Both amplitudes are nonnegative; any fixed physical-harm contribution is constant here.
    losses = 0.6 + np.exp(-np.array(benefits).sum(axis=1))
    even = (losses[0] + losses[2]) / 2 - losses[1]
    assert even == pytest.approx(expected_even, abs=2e-12)


def test_segmented_zero_mu_integral_avoids_negative_remaining_dose_from_cancellation() -> None:
    parameters = law(0.3, 0.0)
    pieces = np.tile([0.3, 0.7], (7, 1)) / 7
    # This physically nonnegative suffix is negative under total-prefix-current subtraction.
    naive_remainder = pieces.sum(axis=0) - pieces[:-1].sum(axis=0) - pieces[-1]
    assert np.min(naive_remainder) < 0
    with_zeros = np.insert(pieces, [0, 3, 7], 0.0, axis=0)
    segmented = model.zero_mu_integral(with_zeros, parameters, CONTROL)
    whole = model.zero_mu_integral(np.array([[0.3, 0.7]]), parameters, CONTROL)
    np.testing.assert_allclose(np.exp(segmented.value), np.exp(whole.value), rtol=2e-12, atol=2e-14)
    assert np.isfinite(segmented.value).all()


def test_quadrature_refines_and_fails_closed_when_accuracy_budget_is_insufficient() -> None:
    parameters = law(0.3, 20.0)
    weights = np.array([[0.001, 0.999], [0.9, 0.1]])
    scales = np.array([100.0, 0.02])
    with pytest.raises(ArithmeticError):
        model.evaluate_policy(weights, FRACTIONS, scales, parameters, model.Quadrature(1e-12, 1))
    ordinary = model.evaluate_policy(weights, FRACTIONS, scales, parameters, model.Quadrature(1e-8, 512))
    precise = model.evaluate_policy(weights, FRACTIONS, scales, parameters, CONTROL)
    np.testing.assert_allclose(ordinary.benefit, precise.benefit, rtol=0, atol=2e-10)
    assert precise.path.panels > 2
    assert precise.path.maximum_relative_error_estimate <= CONTROL.relative_tolerance


@pytest.mark.parametrize("corrupt", [np.nan, np.inf])
def test_nonfinite_log_integrand_cannot_be_reported_as_zero_error(corrupt: float) -> None:
    with pytest.raises(FloatingPointError, match="Log integrand"):
        model.positive_log_quadrature(lambda unit: np.full((len(unit), 1), corrupt), CONTROL)
    zero = model.positive_log_quadrature(lambda unit: np.full((len(unit), 1), -np.inf), CONTROL)
    np.testing.assert_array_equal(zero.value, [-np.inf])
    assert zero.relative_error == 0


def test_corrupt_prefix_cannot_bypass_probability_conservation() -> None:
    parameters = law(0.7, 1.0)
    valid = model.integrate(np.array([[1.0, 1.0]]), parameters, CONTROL)
    for field in ("log_mastery", "log_excess_survival"):
        for corrupt in (np.nan, np.inf):
            state = replace(valid, **{field: np.array([corrupt, -1.0])})
            with pytest.raises(FloatingPointError, match="Log state"):
                model.advance(state, np.zeros(2), parameters, CONTROL)
    # A finite but physically impossible state must also fail at conservation.
    invalid_probability = replace(valid, log_mastery=np.full(2, 1000.0))
    with np.errstate(over="ignore"), pytest.raises(ArithmeticError, match="conserve probability"):
        model.advance(invalid_probability, np.zeros(2), parameters, CONTROL)
