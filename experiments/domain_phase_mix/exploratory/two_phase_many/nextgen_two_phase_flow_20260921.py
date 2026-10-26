# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "scipy"]
# ///
"""Conserved acquisition, labile-memory loss, and consolidation features.

Each bucket has probabilities U, L, C with U -> L acquisition, L -> U
forgetting, and L -> C consolidation. This module contains no outcome fitter.
Tied centering assumes constant exposure geometry across the two phases.
"""

import math
from typing import Literal

import numpy as np
from scipy.special import exprel, gammainc


def _decay_moment(decay: np.ndarray, duration: float, power: int) -> np.ndarray:
    """Integrate time**power times exp(-decay*time) without small-rate loss."""
    x = decay * duration
    result = np.empty_like(x)
    small = x < 0.01
    term = np.ones_like(x[small])
    series = term / (power + 1)
    for index in range(1, 10):
        term = term * (-x[small]) / index
        series = series + term / (power + index + 1)
    result[small] = duration ** (power + 1) * series
    result[~small] = (
        duration ** (power + 1) * math.factorial(power) * gammainc(power + 1, x[~small]) / x[~small] ** (power + 1)
    )
    return result


def _advance(
    unlearned: np.ndarray,
    labile: np.ndarray,
    consolidated: np.ndarray,
    acquisition: np.ndarray,
    forgetting: np.ndarray,
    consolidation: float,
    duration: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # The transient matrix on (U,L) is [[-a,f],[a,-f-k]]. Its two
    # nonpositive eigenvalues admit a stable product-based slow root.
    difference = acquisition - forgetting - consolidation
    delta = np.hypot(difference, 2 * np.sqrt(acquisition * forgetting))
    total_rate = acquisition + forgetting + consolidation
    denominator = total_rate + delta
    slow = np.divide(
        -2 * acquisition * consolidation,
        denominator,
        out=np.zeros_like(acquisition),
        where=denominator > 0,
    )
    fast = -0.5 * denominator
    fast_exp = np.exp(fast * duration)
    divided_exp = np.exp(slow * duration) * duration * exprel(-delta * duration)

    # Positive formulas for the diagonal entries of M-fast*I avoid
    # subtracting almost equal rates when one transition is very weak.
    high = 0.5 * (delta + np.abs(difference))
    low = np.divide(
        acquisition * forgetting,
        high,
        out=np.zeros_like(acquisition),
        where=high > 0,
    )
    upper_diagonal = np.where(difference >= 0, low, high)
    lower_diagonal = np.where(difference >= 0, high, low)

    fast_integral = duration * exprel(fast * duration)
    divided_integral = np.empty_like(delta)
    close = delta * duration < 1e-3
    separated = ~close
    divided_integral[separated] = (duration * exprel(slow[separated] * duration) - fast_integral[separated]) / delta[
        separated
    ]
    # This is the divided difference of the integrated exponential. Near
    # repeated eigenvalues, its symmetric series avoids cancellation.
    midpoint_decay = -0.5 * (slow[close] + fast[close])
    divided_integral[close] = (
        _decay_moment(midpoint_decay, duration, 1)
        + delta[close] ** 2 * _decay_moment(midpoint_decay, duration, 3) / 24
        + delta[close] ** 4 * _decay_moment(midpoint_decay, duration, 5) / 1920
    )

    new_unlearned = (fast_exp + upper_diagonal * divided_exp) * unlearned + (forgetting * divided_exp * labile)
    new_labile = acquisition * divided_exp * unlearned + (fast_exp + lower_diagonal * divided_exp) * labile
    integrated_labile = (
        acquisition * divided_integral * unlearned + (fast_integral + lower_diagonal * divided_integral) * labile
    )
    new_consolidated = consolidated + consolidation * integrated_labile
    return new_unlearned, new_labile, new_consolidated


def _inputs(
    early_epochs: np.ndarray,
    late_epochs: np.ndarray,
    phase0: np.ndarray,
    phase1: np.ndarray,
    alpha: float,
    rate: float,
    power: float,
    forgetting: float,
    consolidation: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    arrays = tuple(np.asarray(value, dtype=float) for value in (early_epochs, late_epochs, phase0, phase1))
    early, late, first, second = arrays
    if early.ndim != 2 or any(value.shape != early.shape for value in arrays):
        raise ValueError("Exposure and phase arrays must have the same two-dimensional shape")
    if any(not np.isfinite(value).all() for value in arrays):
        raise ValueError("Exposure and phase arrays must be finite")
    if np.any(early < 0) or np.any(late < 0):
        raise ValueError("Exposures must be nonnegative")
    if np.any((first < 0) | (first > 1)) or np.any((second < 0) | (second > 1)):
        raise ValueError("Phase weights must be in [0,1]")
    scalars = np.asarray([alpha, rate, power, forgetting, consolidation])
    if not np.isfinite(scalars).all() or not 0 < alpha < 1 or min(rate, power) <= 0:
        raise ValueError("Rates must be finite, with positive rate/power and alpha in (0,1)")
    if min(forgetting, consolidation) < 0:
        raise ValueError("Forgetting and consolidation must be nonnegative")
    clock = (rate * (early + late)) ** power
    if not np.isfinite(clock).all():
        raise ValueError("Acquisition clock overflowed")
    return early, late, first, second, clock


def flow_readout(
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
    """Return retained L+C probability for each row and bucket.

    The total acquisition clock is (rate * total_epochs)**power and is
    attributed between phases in proportion to their physical epochs.
    Forgetting acts on L at forgetting * (1 - phase_weight); C is protected.
    Both kinetic rates are per unit of total training progress.
    """
    early, late, first, second, clock = _inputs(
        early_epochs, late_epochs, phase0, phase1, alpha, rate, power, forgetting, consolidation
    )
    total_epochs = early + late
    early_share = np.divide(early, total_epochs, out=np.zeros_like(early), where=total_epochs > 0)
    state = (np.ones_like(early), np.zeros_like(early), np.zeros_like(early))
    state = _advance(*state, clock * early_share / alpha, forgetting * (1 - first), consolidation, alpha)
    state = _advance(
        *state,
        clock * (1 - early_share) / (1 - alpha),
        forgetting * (1 - second),
        consolidation,
        1 - alpha,
    )
    return state[1] + state[2]


def phase_benefit(
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
    """Return a MARINER benefit feature with an exactly tied-centered flow.

    The additive feature is bounded but is not itself an occupancy probability.
    The ratio feature uses its continuous zero-dose limit without clipping.
    Both require constant exposure geometry: tied mixtures must split each
    bucket's total epochs in the same physical alpha proportion.
    """
    actual = flow_readout(early_epochs, late_epochs, phase0, phase1, alpha, rate, power, forgetting, consolidation)
    total = np.asarray(early_epochs, dtype=float) + np.asarray(late_epochs, dtype=float)
    aggregate = alpha * np.asarray(phase0) + (1 - alpha) * np.asarray(phase1)
    tied = flow_readout(
        alpha * total,
        (1 - alpha) * total,
        aggregate,
        aggregate,
        alpha,
        rate,
        power,
        forgetting,
        consolidation,
    )
    benefit = -np.expm1(-((rate * total) ** power))
    if mode == "additive":
        return benefit + actual - tied
    if mode != "ratio":
        raise ValueError(f"Unknown phase benefit mode: {mode}")
    active = benefit > 0
    if np.any(active & (tied <= 0)):
        raise ValueError("Positive-dose tied retention underflowed; ratio cannot be evaluated")
    ratio = np.divide(actual, tied, out=np.zeros_like(actual), where=active)
    return benefit * ratio
