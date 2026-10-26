# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "scipy"]
# ///
"""Pure donor-hazard clock control for TP044; no fits, data I/O or kinetic grid.

The state obeys dm=(1-m)dH-mu*m*dJ, with J_i=sum outside-group v_j*H_j.
Only the interference clock differs from frozen TP034. Positive integrating-
factor quadrature retains separate mastery and excess survival in log space.
At p=1 with global-pool masses and singleton groups, this is exactly NG2-005
at a fixed budget after rescaling mu. Causal prefix composition is not new.
"""

from __future__ import annotations

import numpy as np

from experiments.domain_phase_mix.exploratory.two_phase_many.tpp40_causal_acquisition_retention_20261015 import (
    Law,
    PolicyResult,
    Quadrature,
    State,
    empty_state,
    hazard_increment,
    log_acquired,
    native_hazard,
    outside_logsum,
    phase_coordinates,
    policy_epochs,
    positive_log_quadrature,
    validate_state,
)


def advance(state: State, increment: np.ndarray, law: Law, control: Quadrature) -> State:
    """Propagate a causal prefix through one constant physical allocation segment."""
    validate_state(state)
    increment = np.asarray(increment, dtype=float)
    if increment.shape != state.epochs.shape or increment.shape != law.masses.shape:
        raise ValueError("State, increment and fixed bucket axis differ")
    if np.any(increment < 0) or not np.isfinite(increment).all():
        raise ValueError("Epoch increments must be finite and nonnegative")
    start = state.epochs
    final = start + increment
    initial_hazard = native_hazard(start, law)
    delta_hazard = hazard_increment(start, increment, law)
    with np.errstate(divide="ignore"):
        log_delta_j = outside_logsum(np.log(delta_hazard), law)
    log_k = log_delta_j + np.log(law.mu) if law.mu else np.full_like(log_delta_j, -np.inf)
    kinetic = np.exp(log_k)
    decay = delta_hazard + kinetic
    if not np.isfinite(decay).all():
        raise FloatingPointError("Combined acquisition/interference clock overflowed")
    log_m, log_d = state.log_mastery.copy(), state.log_excess_survival.copy()
    acquisition_only = ~np.isfinite(log_k)
    forgetting_only = (delta_hazard == 0) & ~acquisition_only
    active = ~(acquisition_only | forgetting_only)
    log_q = np.logaddexp(-initial_hazard, log_d)
    log_m[acquisition_only] = np.logaddexp(
        log_m[acquisition_only], log_q[acquisition_only] + log_acquired(delta_hazard[acquisition_only])
    )
    log_d[acquisition_only] -= delta_hazard[acquisition_only]
    log_one_minus_decay = log_k.copy()
    representable = log_k > -36
    log_one_minus_decay[representable] = log_acquired(kinetic[representable])
    log_d[forgetting_only] = np.logaddexp(
        log_d[forgetting_only], log_m[forgetting_only] + log_one_minus_decay[forgetting_only]
    )
    log_m[forgetting_only] -= kinetic[forgetting_only]
    evaluations, panels, estimate = 0, 0, 0.0
    if active.any():

        def integrand(unit: np.ndarray) -> np.ndarray:
            epochs, hazard, log_dh, time = phase_coordinates(unit, start, increment, law)
            remaining = (1 - time) * increment
            delta = hazard_increment(epochs, remaining, law)
            with np.errstate(divide="ignore"):
                log_remaining_j = outside_logsum(np.log(delta), law)
            attenuation = delta + np.exp(np.log(law.mu) + log_remaining_j)
            log_dj = outside_logsum(log_dh, law)
            mastery = (log_dh - attenuation)[:, active]
            excess = (np.log(law.mu) + log_dj + log_acquired(hazard) - attenuation)[:, active]
            return np.hstack([mastery, excess])

        integral = positive_log_quadrature(integrand, control)
        count = int(active.sum())
        log_m[active] = np.logaddexp(log_m[active] - decay[active], integral.value[:count])
        log_d[active] = np.logaddexp(log_d[active] - decay[active], integral.value[count:])
        evaluations, panels, estimate = integral.evaluations, integral.panels, integral.relative_error
    result = State(
        final,
        log_m,
        log_d,
        state.evaluations + evaluations,
        state.panels + panels,
        max(state.maximum_relative_error_estimate, estimate),
    )
    conservation = np.exp(log_m) + np.exp(np.logaddexp(-native_hazard(final, law), log_d))
    if not np.isfinite(conservation).all() or (
        np.max(abs(conservation - 1)) > 20 * control.relative_tolerance + 256 * np.finfo(float).eps
    ):
        raise ArithmeticError("Independent mastery/survival integrals do not conserve probability")
    return result


def integrate(phase_epochs: np.ndarray, law: Law, control: Quadrature) -> State:
    """Integrate a whole path, retaining prefix state rather than final-normalized rates."""
    phase_epochs = np.asarray(phase_epochs, dtype=float)
    if phase_epochs.ndim != 2 or phase_epochs.shape[1] != len(law.groups):
        raise ValueError("Expected a phase-by-bucket epoch matrix")
    state = empty_state(len(law.groups))
    for increment in phase_epochs:
        state = advance(state, increment, law, control)
    return state


def evaluate_policy(
    weights: np.ndarray, fractions: np.ndarray, scales: np.ndarray, law: Law, control: Quadrature
) -> PolicyResult:
    """Return the terminal counterfactual readout; it is not a prefix state."""
    epochs = policy_epochs(weights, fractions, scales)
    total = epochs.sum(axis=0)
    tied = integrate(total[None], law, control)
    exact_tied = np.array_equal(weights, np.broadcast_to(weights[0], weights.shape))
    if law.mu == 0 or exact_tied:
        hazard = native_hazard(total, law)
        return PolicyResult(-np.expm1(-hazard), hazard, tied, tied, True)
    path = integrate(epochs, law, control)
    hazard = path.survival_hazard(law) + tied.deficit(law)
    return PolicyResult(-np.expm1(-hazard), hazard, path, tied, False)
