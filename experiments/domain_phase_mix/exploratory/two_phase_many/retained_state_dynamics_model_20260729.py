# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Retained power law with an explicit acquisition-and-forgetting state transition.

The retained-power-law surrogate discounts early state through a function of the phase contrast.
That fits the StarCoder surface well, but it is not a dynamical law: splitting one constant policy
into more intervals changes the implied retention. This variant changes only that mechanism.

For bucket ``i`` during phase ``p``, normalized training time evolves a latent retained state as

    dS_i / dt = q_p w_i - lambda (1 - w_i) S_i.

The first term acquires bucket-specific state. The second loses existing state in proportion to
training on other buckets, so phase order matters through a genuine interaction: late non-bucket
tokens erase early bucket state. The exact piecewise-constant transition is used rather than a
discretization.

``q_0 = 1`` and ``q_1`` is the late-phase utility multiplier. At ``lambda = 0`` the state reduces to
the additive phase-weighted dose ``beta_0 w_0 + q_1 beta_1 w_1``. With ``q_1 = 1``, a tied policy has
the same final state under every partition of training time, because the transition is a semigroup.
Those are the two important limiting cases.

Everything downstream is held fixed: power-law benefit, epoch-based repetition damage,
within-window concentration, hierarchical amplitudes, robust head, and nested shape selection.
Consequently, any difference from retained power law is attributable to the state transition.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np

from experiments.domain_phase_mix.exploratory.two_phase_many import (
    retained_power_law_model_20260728 as base,
)

FORGETTING_RATES = (0.0, 1.0, 2.5, 5.0, 10.0)
TRANSITION_EPSILON = 1e-12

Geometry = base.Geometry


@dataclass(frozen=True)
class Shape:
    """Parameters that enter the response nonlinearly."""

    benefit_exponent: float
    benefit_offset: float
    damage_exponent: float
    damage_threshold: float
    forgetting_rate: float
    late_multiplier: float
    ordering_channel: bool


def shape_grid() -> tuple[Shape, ...]:
    """The retained-power-law grid with forgetting replacing contrast retention."""
    return tuple(
        Shape(*values)
        for values in product(
            base.BENEFIT_EXPONENTS,
            base.BENEFIT_OFFSETS,
            base.DAMAGE_EXPONENTS,
            base.DAMAGE_THRESHOLDS,
            FORGETTING_RATES,
            base.LATE_MULTIPLIERS,
            base.ORDERING_CHANNELS,
        )
    )


def _transition(
    state: np.ndarray,
    weights: np.ndarray,
    duration: float,
    acquisition_multiplier: float,
    forgetting_rate: float,
) -> np.ndarray:
    """Apply the exact constant-input transition over one phase."""
    hazard = forgetting_rate * (1.0 - weights)
    survival = np.exp(-hazard * duration)
    integrated_survival = np.full_like(hazard, duration)
    np.divide(
        -np.expm1(-hazard * duration),
        hazard,
        out=integrated_survival,
        where=hazard > TRANSITION_EPSILON,
    )
    acquired = acquisition_multiplier * weights * integrated_survival
    return survival * state + acquired


def retained_state(
    weights: np.ndarray,
    geometry: Geometry,
    forgetting_rate: float,
    late_multiplier: float,
) -> np.ndarray:
    """Final latent state after the two exact phase transitions."""
    phase_0, phase_1 = weights[:, 0, :], weights[:, 1, :]
    state = np.zeros_like(phase_0)
    state = _transition(state, phase_0, geometry.phase_0_fraction, 1.0, forgetting_rate)
    return _transition(state, phase_1, geometry.phase_1_fraction, late_multiplier, forgetting_rate)


def design_matrix(weights: np.ndarray, geometry: Geometry, shape: Shape) -> np.ndarray:
    """Build the same response blocks as retained power law from the dynamical state."""
    state = retained_state(weights, geometry, shape.forgetting_rate, shape.late_multiplier)
    benefit = (state + shape.benefit_offset) ** (-shape.benefit_exponent)
    excess = np.maximum(base.total_epochs(weights, geometry) - shape.damage_threshold, 0.0)
    return np.column_stack(
        [
            base._hierarchical_block(benefit, geometry),
            base._hierarchical_block(excess**shape.damage_exponent, geometry),
            base._signed(base.concentration_gap(weights, geometry)),
        ]
        + ([base.marginal_phase_block(weights, geometry, shape)] if shape.ordering_channel else [])
    )


def penalty_multipliers(geometry: Geometry, shape: Shape) -> np.ndarray:
    """Use the retained-power-law hierarchy and phase shrinkage unchanged."""
    families = len(np.unique(geometry.families))
    excess = len(geometry.excess_domains)
    block = np.concatenate([np.zeros(families), np.ones(excess)])
    phase = np.concatenate([np.ones(4 * families), np.zeros(2)]) if shape.ordering_channel else np.empty(0)
    return np.concatenate([block, block, np.zeros(2), phase])


@dataclass(frozen=True)
class Fitted:
    """A selected shape and robustly fitted linear response head."""

    shape: Shape
    ridge: float
    intercept: float
    coefficients: np.ndarray
    geometry: Geometry

    def predict(self, weights: np.ndarray) -> np.ndarray:
        return self.intercept + design_matrix(weights, self.geometry, self.shape) @ self.coefficients


def fit(
    weights: np.ndarray,
    target: np.ndarray,
    geometry: Geometry,
    folds: tuple[tuple[np.ndarray, np.ndarray], ...],
) -> Fitted:
    """Select the transition and response shape out of fold, then refit the robust head."""
    best_score, best_shape, best_ridge = np.inf, None, 0.0
    for shape in shape_grid():
        design = design_matrix(weights, geometry, shape)
        if not np.all(np.isfinite(design)):
            continue
        multipliers = penalty_multipliers(geometry, shape)
        for ridge in base.RIDGE_GRID:
            errors = []
            for train, test in folds:
                intercept, coefficients = base.solve_head(design[train], target[train], ridge, multipliers)
                errors.append(intercept + design[test] @ coefficients - target[test])
            score = float(np.sqrt(np.mean(np.concatenate(errors) ** 2)))
            if score < best_score:
                best_score, best_shape, best_ridge = score, shape, ridge
    assert best_shape is not None, "empty state-dynamics shape grid"

    design = design_matrix(weights, geometry, best_shape)
    intercept, coefficients = base.solve_head(
        design,
        target,
        best_ridge,
        penalty_multipliers(geometry, best_shape),
    )
    return Fitted(
        shape=best_shape,
        ridge=best_ridge,
        intercept=intercept,
        coefficients=coefficients,
        geometry=geometry,
    )
