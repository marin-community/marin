# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "scipy"]
# ///
"""Causal conserved acquisition, labile loss, and consolidation features.

The acquisition clock is xi=(rate*cumulative_epochs)**power. Unlike the
endpoint-attributed clock, a prefix uses only exposure already delivered.
Physical-time forgetting and consolidation preserve U+L+C=1. This module
contains no outcome fitter; tied centering requires constant exposure geometry.
"""

from typing import Literal

import numpy as np
from scipy.integrate import solve_ivp
from scipy.special import exprel

RELATIVE_TOLERANCE = 2e-10
ABSOLUTE_TOLERANCE = 2e-12


def advance_causal_state(
    state: np.ndarray,
    previous_epochs: np.ndarray,
    added_epochs: np.ndarray,
    weights: np.ndarray,
    duration: float,
    rate: float,
    power: float,
    forgetting: float,
    consolidation: float,
) -> np.ndarray:
    """Advance (U,L,C) over a constant-weight, constant-delivery-rate segment.

    The last state axis contains the three occupancies. A segment depends only
    on its incoming state, prior exposure, and the exposure delivered within it.
    Kinetic rates are per unit of the original total training progress; dividing
    a segment must not renormalize them to the shorter segment's duration.
    """
    state = np.asarray(state, dtype=float)
    previous, added, weights = (np.asarray(value, dtype=float) for value in (previous_epochs, added_epochs, weights))
    if previous.shape != added.shape or weights.shape != added.shape or state.shape != (*added.shape, 3):
        raise ValueError("State must end in (U,L,C) and share the exposure/weight dimensions")
    if not all(np.isfinite(value).all() for value in (state, previous, added, weights)):
        raise ValueError("States, exposures, and weights must be finite")
    if np.any(previous < 0) or np.any(added < 0) or np.any((weights < 0) | (weights > 1)):
        raise ValueError("Exposures must be nonnegative and weights must be in [0,1]")
    scalars = np.asarray([duration, rate, power, forgetting, consolidation])
    if not np.isfinite(scalars).all() or duration < 0 or rate <= 0 or not 0 < power <= 1:
        raise ValueError("Require duration >= 0, rate > 0, and 0 < power <= 1")
    if min(forgetting, consolidation) < 0:
        raise ValueError("Forgetting and consolidation must be nonnegative")
    if np.any(state < -1e-10) or np.max(np.abs(state.sum(axis=-1) - 1)) > 1e-9:
        raise ValueError("Incoming occupancy must be nonnegative and sum to one")
    if duration == 0:
        if np.any(added > 0):
            raise ValueError("Positive exposure requires positive duration")
        return state.copy()

    loss = forgetting * (1 - weights)
    total = previous + added
    clock_end = (rate * total) ** power
    if not np.isfinite(clock_end).all():
        raise ValueError("Acquisition clock overflowed")
    delivered_share = np.divide(added, total, out=np.zeros_like(added), where=total > 0)
    positive_prefix = (previous > 0) & (added > 0)
    clock_fraction = np.zeros_like(added)
    clock_fraction[added > 0] = 1
    log_prefix_share = np.zeros_like(added)
    small_addition = positive_prefix & (delivered_share < 0.5)
    large_addition = positive_prefix & ~small_addition
    log_prefix_share[small_addition] = np.log1p(-delivered_share[small_addition])
    log_prefix_share[large_addition] = np.log(previous[large_addition]) - np.log(total[large_addition])
    clock_fraction[positive_prefix] = -np.expm1(power * log_prefix_share[positive_prefix])
    clock_increment = clock_end * clock_fraction
    acquisition_only = (loss == 0) & (consolidation == 0)
    acquired = state[..., 0] * (-np.expm1(-clock_increment))
    exact_unlearned = state[..., 0] * np.exp(-clock_increment)
    if acquisition_only.all():
        return np.stack((exact_unlearned, state[..., 1] + acquired, state[..., 2]), axis=-1)
    if not np.any(clock_increment):
        decay = (loss + consolidation) * duration
        result = state.copy()
        result[..., 1] = state[..., 1] * np.exp(-decay)
        result[..., 2] += state[..., 1] * consolidation * duration * exprel(-decay)
        result[..., 0] = 1 - result[..., 1] - result[..., 2]
        return result

    # Integrate in normalized acquisition-clock progress z. The time Jacobian
    # is bounded at a zero-dose origin for the native MARINER powers <= 1.
    clock_start = 1 - clock_fraction
    time_scale = np.divide(
        duration * clock_fraction,
        power * delivered_share,
        out=np.full_like(added, duration),
        where=delivered_share > 0,
    )
    scale = np.maximum(state[..., 1] + state[..., 2], -np.expm1(-clock_end))
    scale = np.where(scale > 0, scale, 1.0)
    acquisition_source = clock_increment / scale
    initial = np.stack((state[..., 1] / scale, state[..., 2] / scale))

    def derivative(progress: float, flattened: np.ndarray) -> np.ndarray:
        labile, protected = flattened.reshape(initial.shape)
        elapsed_derivative = time_scale * (clock_start + progress * clock_fraction) ** (1 / power - 1)
        retained_derivative = (
            acquisition_source
            - clock_increment * (labile + protected)
            - (loss + consolidation) * elapsed_derivative * labile
        )
        protected_derivative = consolidation * elapsed_derivative * labile
        return np.stack((retained_derivative, protected_derivative)).ravel()

    solution = solve_ivp(
        derivative,
        (0.0, 1.0),
        initial.ravel(),
        method="DOP853",
        rtol=RELATIVE_TOLERANCE,
        atol=ABSOLUTE_TOLERANCE,
        t_eval=[1.0],
    )
    if not solution.success:
        raise RuntimeError(f"Causal flow integration failed: {solution.message}")
    labile, protected = solution.y[:, -1].reshape(initial.shape) * scale
    unlearned = 1 - labile - protected
    # A saturated acquisition-only bucket may retain U far below solver atol.
    # Preserve that analytic positive tail instead of subtracting L from one.
    unlearned = np.where(acquisition_only, exact_unlearned, unlearned)
    labile = np.where(acquisition_only, state[..., 1] + acquired, labile)
    protected = np.where(acquisition_only, state[..., 2], protected)
    return np.stack((unlearned, labile, protected), axis=-1)


def causal_flow_readout(
    early_epochs: np.ndarray,
    late_epochs: np.ndarray,
    phase0: np.ndarray,
    phase1: np.ndarray,
    alpha: float,
    rate: float,
    power: float,
    forgetting: float,
    consolidation: float,
) -> np.ndarray:
    """Return the causal retained L+C probability for every row and bucket."""
    arrays = tuple(np.asarray(value, dtype=float) for value in (early_epochs, late_epochs, phase0, phase1))
    early, late, first, second = arrays
    if early.ndim != 2 or any(value.shape != early.shape for value in arrays):
        raise ValueError("Exposure and phase arrays must share a two-dimensional shape")
    if not 0 < alpha < 1:
        raise ValueError("Alpha must lie strictly between zero and one")
    state = np.zeros((*early.shape, 3))
    state[..., 0] = 1
    state = advance_causal_state(
        state, np.zeros_like(early), early, first, alpha, rate, power, forgetting, consolidation
    )
    state = advance_causal_state(state, early, late, second, 1 - alpha, rate, power, forgetting, consolidation)
    if forgetting == 0:
        # This exact conservation identity also keeps the no-forgetting tied
        # restriction independent of the adaptive integration mesh.
        return -np.expm1(-((rate * (early + late)) ** power))
    return state[..., 1] + state[..., 2]


def causal_phase_benefit(
    early_epochs: np.ndarray,
    late_epochs: np.ndarray,
    phase0: np.ndarray,
    phase1: np.ndarray,
    alpha: float,
    rate: float,
    power: float,
    forgetting: float,
    consolidation: float,
    mode: Literal["additive", "ratio"] = "additive",
) -> np.ndarray:
    """Return an exactly tied-centered MARINER benefit with causal retention.

    Actual and tied flows use the same cumulative exposure clock. The ratio
    uses the continuous zero-dose limit, with no epsilon denominator or cap.
    """
    actual = causal_flow_readout(
        early_epochs, late_epochs, phase0, phase1, alpha, rate, power, forgetting, consolidation
    )
    total = np.asarray(early_epochs, dtype=float) + np.asarray(late_epochs, dtype=float)
    aggregate = alpha * np.asarray(phase0) + (1 - alpha) * np.asarray(phase1)
    tied = causal_flow_readout(
        alpha * total, (1 - alpha) * total, aggregate, aggregate, alpha, rate, power, forgetting, consolidation
    )
    benefit = -np.expm1(-((rate * total) ** power))
    if mode == "additive":
        return benefit + actual - tied
    if mode != "ratio":
        raise ValueError(f"Unknown phase benefit mode: {mode}")
    positive = benefit > 0
    if np.any(positive & (tied <= 0)):
        raise ValueError("Positive-dose tied retention underflowed; ratio cannot be evaluated")
    return benefit * np.divide(actual, tied, out=np.zeros_like(actual), where=positive)
