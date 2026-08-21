# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Aggregate response with a gradient-tied phase-control correction.

Let ``a = beta0 * w0 + beta1 * w1`` be the aggregate mixture and
``delta = w1 - w0`` the phase contrast. The model is

    L(w0, w1) = A(a) + xi * grad A(a)^T delta + gamma * I_beta(a, delta),

where ``A`` is a hierarchical power-law benefit plus repetition-damage model.
The phase-order term is not an independently fitted response surface: its
coefficients are exactly the coefficients of ``A`` differentiated with respect
to the aggregate. ``xi`` is a dimensionless phase-control strength. The
dimensionless phase-information cost

    I_beta(a, delta) = 0.5 * beta0 * beta1 * sum_i delta_i^2 / a_i

is the Fisher-quadratic term of weighted Jensen-Shannon divergence. Unlike the
full divergence under unequal phase lengths, it is exactly even under
fixed-aggregate contrast reversal.

This construction has three useful invariants:

* At a tied policy, ``delta = 0`` and ``I_beta = 0``, so the model reduces
  exactly to the independently fittable aggregate model ``A``.
* At an interior stationary point of ``A`` on the simplex, the tangent
  projection of ``grad A`` vanishes, so the first-order phase-order term is
  zero for every feasible contrast.
* Away from that stationary point, the same aggregate gradient identifies
  which domains the late phase should increase or decrease. No per-domain
  phase coefficients are introduced.

The response is deliberately a local control model rather than a reweighted
cumulative-dose model. Replacing the first-order term with ``A(a + xi*delta)``
would put the model back in the phase-weighted-dose null class, which cannot
represent a strict two-phase advantage over the complete tied-policy class.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np
from scipy.optimize import lsq_linear

from experiments.domain_phase_mix.exploratory.two_phase_many import retained_power_law_model_20260728 as retained

BENEFIT_EXPONENTS = retained.BENEFIT_EXPONENTS
BENEFIT_OFFSETS = retained.BENEFIT_OFFSETS
DAMAGE_EXPONENTS = retained.DAMAGE_EXPONENTS
DAMAGE_THRESHOLDS = retained.DAMAGE_THRESHOLDS
# A dimensionless geometric grid around the natural unit coupling. Zero is the
# phase-free nested ablation. Negative values are excluded: the mechanism says
# that the late phase steers the final state along the aggregate descent
# direction, not against it.
PHASE_LEVERAGES = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0)
RIDGE_GRID = retained.RIDGE_GRID

Geometry = retained.Geometry


@dataclass(frozen=True)
class Shape:
    """Parameters that enter the response nonlinearly."""

    benefit_exponent: float
    benefit_offset: float
    damage_exponent: float
    damage_threshold: float
    phase_leverage: float


def shape_grid() -> tuple[Shape, ...]:
    return tuple(
        Shape(*values)
        for values in product(
            BENEFIT_EXPONENTS,
            BENEFIT_OFFSETS,
            DAMAGE_EXPONENTS,
            DAMAGE_THRESHOLDS,
            PHASE_LEVERAGES,
        )
    )


@dataclass(frozen=True)
class Fitted:
    shape: Shape
    ridge: float
    intercept: float
    coefficients: np.ndarray
    geometry: Geometry

    def predict(self, weights: np.ndarray) -> np.ndarray:
        return self.intercept + design_matrix(weights, self.geometry, self.shape) @ self.coefficients

    @property
    def phase_information_cost(self) -> float:
        return float(self.coefficients[-1])


@dataclass(frozen=True)
class TwoStageFitted:
    """Aggregate response identified on tied rows, then two phase scalars."""

    shape: Shape
    ridge: float
    intercept: float
    aggregate_coefficients: np.ndarray
    phase_leverage: float
    phase_information_cost: float
    geometry: Geometry

    def predict(self, weights: np.ndarray) -> np.ndarray:
        aggregate = aggregate_design_matrix(weights, self.geometry, self.shape)
        directional = directional_design_matrix(weights, self.geometry, self.shape)
        baseline = self.intercept + aggregate @ self.aggregate_coefficients
        ordering = directional @ self.aggregate_coefficients
        information = phase_information_cost(weights, self.geometry)
        return baseline + self.phase_leverage * ordering + self.phase_information_cost * information


def aggregate_mixture(weights: np.ndarray, geometry: Geometry) -> np.ndarray:
    beta0 = geometry.phase_0_fraction
    return beta0 * weights[:, 0, :] + (1.0 - beta0) * weights[:, 1, :]


def epoch_scale(geometry: Geometry) -> np.ndarray:
    """Epochs per unit aggregate weight.

    The experiment materializer guarantees ``c0 / beta0 = c1 / beta1``:
    total physical epochs therefore depend on the aggregate mixture alone.
    """
    beta0 = geometry.phase_0_fraction
    beta1 = geometry.phase_1_fraction
    early = geometry.c0 / beta0
    late = geometry.c1 / beta1
    relative_error = np.max(np.abs(early - late) / np.maximum(np.abs(early), 1e-12))
    assert relative_error < 1e-5, f"phase epoch scales disagree by {relative_error:.3e}"
    return 0.5 * (early + late)


def phase_information_cost(weights: np.ndarray, geometry: Geometry) -> np.ndarray:
    """Fisher-quadratic phase information, exactly even at fixed aggregate."""
    beta0 = geometry.phase_0_fraction
    beta1 = geometry.phase_1_fraction
    aggregate = aggregate_mixture(weights, geometry)
    contrast = weights[:, 1, :] - weights[:, 0, :]
    normalized = np.zeros_like(aggregate)
    np.divide(contrast**2, aggregate, out=normalized, where=aggregate > 0.0)
    return 0.5 * beta0 * beta1 * normalized.sum(axis=1)


def aggregate_and_directional_derivative(
    weights: np.ndarray,
    geometry: Geometry,
    shape: Shape,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Benefit and damage bases together with their derivatives along phase contrast."""
    aggregate = aggregate_mixture(weights, geometry)
    contrast = weights[:, 1, :] - weights[:, 0, :]

    benefit_scale = aggregate + shape.benefit_offset
    benefit = benefit_scale ** (-shape.benefit_exponent)
    benefit_derivative = -shape.benefit_exponent * benefit_scale ** (-(shape.benefit_exponent + 1.0)) * contrast

    scale = epoch_scale(geometry)
    epochs = aggregate * scale
    excess = np.maximum(epochs - shape.damage_threshold, 0.0)
    damage = excess**shape.damage_exponent
    active = epochs > shape.damage_threshold
    damage_derivative = (
        shape.damage_exponent * scale * excess ** max(shape.damage_exponent - 1.0, 0.0) * contrast * active
    )
    return benefit, benefit_derivative, damage, damage_derivative


def design_matrix(weights: np.ndarray, geometry: Geometry, shape: Shape) -> np.ndarray:
    """Linear-amplitude design with aggregate and phase coefficients tied."""
    aggregate = aggregate_design_matrix(weights, geometry, shape)
    directional = directional_design_matrix(weights, geometry, shape)
    controlled = aggregate + shape.phase_leverage * directional
    return np.column_stack([controlled, phase_information_cost(weights, geometry)])


def aggregate_design_matrix(weights: np.ndarray, geometry: Geometry, shape: Shape) -> np.ndarray:
    """The phase-tied aggregate response block."""
    benefit, _benefit_derivative, damage, _damage_derivative = aggregate_and_directional_derivative(
        weights, geometry, shape
    )
    return np.column_stack(
        [
            retained._hierarchical_block(benefit, geometry),
            retained._hierarchical_block(damage, geometry),
        ]
    )


def directional_design_matrix(weights: np.ndarray, geometry: Geometry, shape: Shape) -> np.ndarray:
    """Exact directional derivative of the aggregate design along phase contrast."""
    _benefit, benefit_derivative, _damage, damage_derivative = aggregate_and_directional_derivative(
        weights, geometry, shape
    )
    return np.column_stack(
        [
            retained._hierarchical_block(benefit_derivative, geometry),
            retained._hierarchical_block(damage_derivative, geometry),
        ]
    )


def penalty_multipliers(geometry: Geometry) -> np.ndarray:
    """Shrink bucket departures toward their family; leave pooled mechanisms free."""
    families = len(np.unique(geometry.families))
    excess = len(geometry.excess_domains)
    block = np.concatenate([np.zeros(families), np.ones(excess)])
    return np.concatenate([block, block, np.zeros(1)])


def aggregate_penalty_multipliers(geometry: Geometry) -> np.ndarray:
    return penalty_multipliers(geometry)[:-1]


def _row_indices(selection: np.ndarray) -> np.ndarray:
    selection = np.asarray(selection)
    return np.flatnonzero(selection) if selection.dtype == bool else selection.astype(int)


def _fit_nonnegative_phase_head(design: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Robust two-coefficient solve without an intercept."""
    scale = np.maximum(np.abs(design).max(axis=0), retained.COLUMN_SCALE_FLOOR)
    normalized = design / scale
    weights = np.ones(len(target))
    coefficients = np.zeros(design.shape[1])
    for _ in range(retained.HUBER_ITERATIONS):
        root = np.sqrt(weights)
        solved = lsq_linear(
            normalized * root[:, None],
            target * root,
            bounds=(np.zeros(design.shape[1]), np.full(design.shape[1], np.inf)),
            method="trf",
            tol=1e-10,
            max_iter=200,
        )
        updated = solved.x
        residual = normalized @ updated - target
        spread = retained.MAD_TO_SIGMA * float(np.median(np.abs(residual - np.median(residual))))
        shift = float(np.max(np.abs(normalized @ (updated - coefficients))))
        coefficients = updated
        if spread <= 0.0 or shift < retained.HUBER_TOLERANCE * spread:
            break
        cut = retained.HUBER_SCALE * spread
        weights = np.minimum(1.0, cut / np.maximum(np.abs(residual), 1e-12))
    return coefficients / scale


def fit_two_stage(
    weights: np.ndarray,
    target: np.ndarray,
    geometry: Geometry,
    folds: tuple[tuple[np.ndarray, np.ndarray], ...],
) -> TwoStageFitted:
    """Identify aggregate shape from tied rows, then fit phase control on residuals.

    The aggregate response cannot be rewritten by the more numerous asymmetric
    policies. Phase rows only estimate ``xi`` and ``gamma`` after the aggregate
    coefficients are frozen.
    """
    tied = np.max(np.abs(weights[:, 0, :] - weights[:, 1, :]), axis=1) < 1e-9
    tied_rows = np.flatnonzero(tied)
    assert len(tied_rows) >= 10, f"two-stage fit needs tied aggregate evidence, found {len(tied_rows)} rows"

    best_score, best_shape, best_ridge = np.inf, None, 0.0
    multipliers = aggregate_penalty_multipliers(geometry)
    for shape in shape_grid():
        if shape.phase_leverage != 0.0:
            continue
        design = aggregate_design_matrix(weights, geometry, shape)
        for ridge in RIDGE_GRID:
            errors = []
            for train, test in folds:
                train_tied = np.intersect1d(_row_indices(train), tied_rows)
                test_tied = np.intersect1d(_row_indices(test), tied_rows)
                if len(train_tied) < 5 or not len(test_tied):
                    continue
                intercept, coefficients = retained.solve_head(design[train_tied], target[train_tied], ridge, multipliers)
                errors.append(intercept + design[test_tied] @ coefficients - target[test_tied])
            if not errors:
                continue
            score = float(np.sqrt(np.mean(np.concatenate(errors) ** 2)))
            if score < best_score:
                best_score, best_shape, best_ridge = score, shape, ridge
    assert best_shape is not None, "no fold contained enough tied rows to select the aggregate response"

    aggregate = aggregate_design_matrix(weights, geometry, best_shape)
    intercept, coefficients = retained.solve_head(
        aggregate[tied_rows],
        target[tied_rows],
        best_ridge,
        multipliers,
    )
    baseline = intercept + aggregate @ coefficients
    ordering = directional_design_matrix(weights, geometry, best_shape) @ coefficients
    information = phase_information_cost(weights, geometry)
    phase_coefficients = _fit_nonnegative_phase_head(
        np.column_stack([ordering, information]),
        target - baseline,
    )
    return TwoStageFitted(
        shape=best_shape,
        ridge=best_ridge,
        intercept=intercept,
        aggregate_coefficients=coefficients,
        phase_leverage=float(phase_coefficients[0]),
        phase_information_cost=float(phase_coefficients[1]),
        geometry=geometry,
    )


def fit(
    weights: np.ndarray,
    target: np.ndarray,
    geometry: Geometry,
    folds: tuple[tuple[np.ndarray, np.ndarray], ...],
) -> Fitted:
    """Select nonlinear shape and ridge out of fold, then refit all rows."""
    best_score, best_shape, best_ridge = np.inf, None, 0.0
    multipliers = penalty_multipliers(geometry)
    for shape in shape_grid():
        design = design_matrix(weights, geometry, shape)
        if not np.all(np.isfinite(design)):
            continue
        for ridge in RIDGE_GRID:
            errors = []
            for train, test in folds:
                intercept, coefficients = retained.solve_head(design[train], target[train], ridge, multipliers)
                errors.append(intercept + design[test] @ coefficients - target[test])
            score = float(np.sqrt(np.mean(np.concatenate(errors) ** 2)))
            if score < best_score:
                best_score, best_shape, best_ridge = score, shape, ridge
    assert best_shape is not None, "empty shape grid"

    design = design_matrix(weights, geometry, best_shape)
    intercept, coefficients = retained.solve_head(design, target, best_ridge, multipliers)
    return Fitted(
        shape=best_shape,
        ridge=best_ridge,
        intercept=intercept,
        coefficients=coefficients,
        geometry=geometry,
    )
