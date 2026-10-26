# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "scipy"]
# ///
"""Policy-local evaluation of the frozen causal acquisition and memory law.

Each adaptive integration receives one policy row, with fixed bucket width.
Changing other query rows therefore cannot change its trajectory or numerical
error control. The underlying differential equation and tolerances are unchanged.
Tied centering uses its analytic identity under constant exposure geometry.
"""

from typing import Literal

import causal_two_phase_flow_20260921 as previous
import numpy as np


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
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    arrays = tuple(np.asarray(value, dtype=float) for value in (early_epochs, late_epochs, phase0, phase1))
    early, late, first, second = arrays
    if early.ndim != 2 or any(value.shape != early.shape for value in arrays):
        raise ValueError("Exposure and phase arrays must share a two-dimensional shape")
    if not all(np.isfinite(value).all() for value in arrays):
        raise ValueError("Exposures and weights must be finite")
    if np.any(early < 0) or np.any(late < 0):
        raise ValueError("Exposures must be nonnegative")
    if np.any((first < 0) | (first > 1)) or np.any((second < 0) | (second > 1)):
        raise ValueError("Weights must lie in [0,1]")
    scalars = np.asarray([alpha, rate, power, forgetting, consolidation])
    if not np.isfinite(scalars).all() or not 0 < alpha < 1 or rate <= 0 or not 0 < power <= 1:
        raise ValueError("Require alpha in (0,1), rate > 0, and power in (0,1]")
    if min(forgetting, consolidation) < 0:
        raise ValueError("Forgetting and consolidation must be nonnegative")
    if not np.isfinite((rate * (early + late)) ** power).all():
        raise ValueError("Acquisition clock overflowed")
    return early, late, first, second


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
    """Advance the conserved state, integrating each policy row independently.

    Bucket coordinates form the last exposure axis. Kinetic rates remain per
    unit of original training progress when a segment is split or resumed.
    """
    state = np.asarray(state, dtype=float)
    previous_epochs, added_epochs, weights = (
        np.asarray(value, dtype=float) for value in (previous_epochs, added_epochs, weights)
    )
    if added_epochs.ndim < 1 or state.shape != (*added_epochs.shape, 3):
        raise ValueError("State must share exposure dimensions and end in (U,L,C)")
    if previous_epochs.shape != added_epochs.shape or weights.shape != added_epochs.shape:
        raise ValueError("Exposure and weight dimensions must match")
    width = added_epochs.shape[-1]
    state_rows = state.reshape(-1, width, 3)
    previous_rows = previous_epochs.reshape(-1, width)
    added_rows = added_epochs.reshape(-1, width)
    weight_rows = weights.reshape(-1, width)
    result = np.empty_like(state_rows)
    for index in range(len(state_rows)):
        result[index : index + 1] = previous.advance_causal_state(
            state_rows[index : index + 1],
            previous_rows[index : index + 1],
            added_rows[index : index + 1],
            weight_rows[index : index + 1],
            duration,
            rate,
            power,
            forgetting,
            consolidation,
        )
    return result.reshape(state.shape)


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
    """Return L+C with no dependence on the other policy rows in the batch."""
    early, late, first, second = _inputs(
        early_epochs, late_epochs, phase0, phase1, alpha, rate, power, forgetting, consolidation
    )
    if forgetting == 0:
        return -np.expm1(-((rate * (early + late)) ** power))
    state = np.zeros((*early.shape, 3))
    state[..., 0] = 1
    state = advance_causal_state(
        state, np.zeros_like(early), early, first, alpha, rate, power, forgetting, consolidation
    )
    state = advance_causal_state(state, early, late, second, 1 - alpha, rate, power, forgetting, consolidation)
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
    """Return MARINER benefit with the unchanged causal phase correction.

    Constant exposure geometry is required: equal phase weights must deliver
    early_epochs=alpha*total_epochs. On that exact policy diagonal the two paths
    coincide, so their correction is evaluated as zero algebraically.
    """
    early, late, first, second = _inputs(
        early_epochs, late_epochs, phase0, phase1, alpha, rate, power, forgetting, consolidation
    )
    if mode not in ("additive", "ratio"):
        raise ValueError(f"Unknown phase benefit mode: {mode}")
    total = early + late
    benefit = -np.expm1(-((rate * total) ** power))
    if not np.isfinite(benefit).all():
        raise ValueError("Acquisition clock overflowed")
    diagonal = first == second
    active_rows = np.flatnonzero(~np.all(diagonal, axis=1))
    if forgetting == 0 or not len(active_rows):
        return benefit
    aggregate = alpha * first[active_rows] + (1 - alpha) * second[active_rows]
    actual = causal_flow_readout(
        early[active_rows],
        late[active_rows],
        first[active_rows],
        second[active_rows],
        alpha,
        rate,
        power,
        forgetting,
        consolidation,
    )
    tied = causal_flow_readout(
        alpha * total[active_rows],
        (1 - alpha) * total[active_rows],
        aggregate,
        aggregate,
        alpha,
        rate,
        power,
        forgetting,
        consolidation,
    )
    selected_benefit = benefit[active_rows]
    if mode == "additive":
        corrected = selected_benefit + actual - tied
    else:
        positive = selected_benefit > 0
        if np.any(positive & (tied <= 0)):
            raise ValueError("Positive-dose tied retention underflowed; ratio cannot be evaluated")
        corrected = selected_benefit * np.divide(actual, tied, out=np.zeros_like(actual), where=positive)
    result = benefit.copy()
    result[active_rows] = np.where(diagonal[active_rows], selected_benefit, corrected)
    return result
