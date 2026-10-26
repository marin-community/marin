# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Invented pure qualification cases, distinct from the registered TP044 table."""

from dataclasses import replace

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from experiments.domain_phase_mix.exploratory.two_phase_many import tpp40_competitive_hazard_clock_20261015 as model

CONTROL = model.Quadrature(1e-11, 512)
PHASES = np.array([[0.7, 4.0], [1.2, 0.3]])


def parameters(power: float = 0.7, mu: float = 2.0) -> model.Law:
    return model.Law(0.4, power, mu, ("receiver", "donor"), np.array([0.35, 0.65]))


def independent_ode(phases: np.ndarray, law: model.Law) -> np.ndarray:
    """Solve physical-time dynamics after the exact t=u**(1/p) origin transform."""
    start = np.zeros(len(law.groups))
    mastery = np.zeros(len(law.groups))
    for dose in phases:

        def derivative(unit: float, state: np.ndarray, start: np.ndarray = start, dose: np.ndarray = dose) -> np.ndarray:
            time = unit ** (1 / law.power)
            epochs = start + time * dose
            velocity = np.zeros(len(law.groups))
            for bucket in range(len(law.groups)):
                if dose[bucket] == 0:
                    continue
                if start[bucket] == 0:
                    velocity[bucket] = (law.rate * dose[bucket]) ** law.power
                elif unit > 0 or law.power == 1:
                    velocity[bucket] = (
                        law.rate**law.power
                        * dose[bucket]
                        * unit ** (1 / law.power - 1)
                        * epochs[bucket] ** (law.power - 1)
                    )
            competition = np.array(
                [
                    sum(law.masses[j] * velocity[j] for j in range(len(law.groups)) if group != law.groups[j])
                    for group in law.groups
                ]
            )
            return (1 - state) * velocity - law.mu * state * competition

        result = solve_ivp(derivative, (0, 1), mastery, method="DOP853", rtol=2e-12, atol=2e-14)
        assert result.success
        mastery = result.y[:, -1]
        start = start + dose
    return mastery


@pytest.mark.parametrize("power", [0.3, 0.7, 1.0])
def test_analytic_tied_state_and_independent_physical_time_ode(power: float) -> None:
    law = parameters(power)
    total = PHASES.sum(axis=0)
    hazard = (law.rate * total) ** power
    decay = hazard + law.mu * (law.masses * hazard)[::-1]
    expected = hazard / decay * -np.expm1(-decay)
    tied = model.integrate(total[None], law, CONTROL)
    np.testing.assert_allclose(np.exp(tied.log_mastery), expected, rtol=2e-11, atol=2e-13)
    path = model.integrate(PHASES, law, CONTROL)
    oracle = independent_ode(PHASES, law)
    np.testing.assert_allclose(np.exp(path.log_mastery), oracle, rtol=2e-10, atol=2e-11)
    np.testing.assert_allclose(path.survival_hazard(law), -np.log1p(-oracle), rtol=0, atol=2e-11)
    assert np.all(np.exp(path.log_mastery) <= -np.expm1(-hazard) + 2e-12)


def test_zero_mu_and_exact_ties_preserve_native_and_zero_departure() -> None:
    weights = np.array([[0.23, 0.77], [0.81, 0.19]])
    fractions = np.array([0.65, 0.35])
    for law, policy in ((parameters(mu=0), weights), (parameters(), np.tile([0.23, 0.77], (2, 1)))):
        for scales in (np.zeros(2), np.array([1e-19, 2e-19]), np.array([7.0, 2.0])):
            actual = model.evaluate_policy(policy, fractions, scales, law, CONTROL)
            total = (fractions[:, None] * policy * scales).sum(axis=0)
            expected = -np.expm1(-((law.rate * total) ** law.power))
            np.testing.assert_array_equal(actual.benefit, expected)
            assert actual.path is actual.tied and actual.exact_native_branch


def test_p1_global_pool_clock_is_exact_ng005_coefficient_rescaling() -> None:
    pools, total = np.array([2.0, 7.0]), 5.5
    weights, fractions = np.array([[0.23, 0.77], [0.81, 0.19]]), np.array([0.65, 0.35])
    law = model.Law(0.4, 1.0, 2.3, ("receiver", "donor"), pools / pools.sum())
    phases = fractions[:, None] * weights * total / pools
    mu_old = law.mu * law.rate * total / pools.sum()
    expected = np.zeros(2)
    for fraction, weight in zip(fractions, weights, strict=True):
        acquisition = law.rate * total / pools * weight
        decay = acquisition + mu_old * (1 - weight)
        equilibrium = acquisition / decay
        expected = equilibrium + (expected - equilibrium) * np.exp(-fraction * decay)
    actual = model.integrate(phases, law, CONTROL)
    np.testing.assert_allclose(np.exp(actual.log_mastery), expected, atol=2e-13, rtol=2e-12)


def test_prefix_resume_split_and_unchanged_group_bucket_refinement() -> None:
    law = parameters(0.3)
    prefix = model.integrate(PHASES[:1], law, CONTROL)
    actual = model.integrate(PHASES, law, CONTROL)
    resumed = model.advance(prefix, PHASES[1], law, CONTROL)
    np.testing.assert_array_equal(resumed.log_mastery, actual.log_mastery)
    pieces = np.vstack([np.tile(PHASES[0] / 4, (4, 1)), np.zeros((1, 2)), np.tile(PHASES[1] / 3, (3, 1))])
    divided = model.integrate(pieces, law, CONTROL)
    np.testing.assert_allclose(divided.survival_hazard(law), actual.survival_hazard(law), atol=2e-11, rtol=0)
    np.testing.assert_allclose(divided.deficit(law), actual.deficit(law), atol=2e-11, rtol=0)
    refined_law = replace(law, groups=("receiver", "donor", "donor"), masses=np.array([0.35, 0.2, 0.45]))
    refined = model.integrate(PHASES[:, [0, 1, 1]], refined_law, CONTROL)
    np.testing.assert_allclose(refined.survival_hazard(refined_law), actual.survival_hazard(law)[[0, 1, 1]], atol=2e-12)
    np.testing.assert_allclose(refined.deficit(refined_law), actual.deficit(law)[[0, 1, 1]], atol=2e-12)


def test_saturated_mastery_retains_tiny_interference_in_log_survival() -> None:
    law = model.Law(1.0, 1.0, 1e-250, ("receiver", "donor"), np.array([0.35, 0.65]))
    actual = model.integrate(np.array([[1000.0, 0.0], [0.0, 2.1]]), law, CONTROL)
    log_k = np.log(law.mu) + np.log(0.65 * 2.1)
    assert actual.survival_hazard(law)[0] == pytest.approx(-log_k, abs=2e-12)
    assert actual.deficit(law)[0] == pytest.approx(1000 + log_k, abs=2e-12)


def test_new_clock_keeps_historical_fail_closed_state_guards() -> None:
    law = parameters()
    actual = model.integrate(PHASES, law, CONTROL)
    corrupt = replace(actual, log_mastery=np.array([np.nan, -1.0]))
    with pytest.raises(FloatingPointError, match="Log state"):
        model.advance(corrupt, np.zeros(2), law, CONTROL)
    steep = np.array([[0.001, 60.0], [70.0, 0.001]])
    with pytest.raises(ArithmeticError):
        model.integrate(steep, parameters(0.3, 20.0), model.Quadrature(1e-12, 1))
