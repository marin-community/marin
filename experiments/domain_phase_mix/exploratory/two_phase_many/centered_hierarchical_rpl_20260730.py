# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Retained power law with exchangeable within-family partial pooling.

This candidate changes only the estimator used for the two bucket-resolved
response blocks in retained power law (RPL). RPL represents a bucket amplitude
as a nonnegative family floor plus a nonnegative bucket excess. After profiling
the redundant family floor, the resulting prior shrinks every member toward
the smallest amplitude in its family. Here every bucket has one directly
identified nonnegative amplitude and the fitted effects are shrunk toward
their family mean:

    penalty(theta) = ridge * sum_f sum_{i in f} (theta_i - mean_f(theta))**2.

The amplitudes above are the physical coefficients multiplying the unscaled
benefit and repetition-damage features. Family means are unpenalized. The
explicit phase-order columns retain RPL's max-normalized ridge coordinates.
Retained-state dynamics, power-law benefit, repetition damage, concentration,
marginal phase columns, robust loss, shape grid, and coefficient signs are
otherwise unchanged.

With singleton families the centering operator is zero, so both the design and
penalty reduce exactly to RPL. This makes StarCoder WSD80 a numerical-invariance
test rather than another opportunity to select this estimator.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, replace

import numpy as np

from experiments.domain_phase_mix.exploratory.two_phase_many import retained_power_law_model_20260728 as rpl

Geometry = rpl.Geometry
Shape = rpl.Shape


def family_centering_operator(geometry: Geometry) -> np.ndarray:
    """Return the block-diagonal complete-graph Laplacian square root.

    Each family block is ``I - 11' / n``. It is symmetric and idempotent, so
    its squared norm is the desired within-family sum of squared deviations.
    """
    families = geometry.families
    operator = np.zeros((len(families), len(families)), dtype=float)
    for family in np.unique(families):
        members = np.flatnonzero(families == family)
        operator[np.ix_(members, members)] = np.eye(len(members)) - np.ones((len(members), len(members))) / len(members)
    return operator


def response_blocks(
    weights: np.ndarray,
    geometry: Geometry,
    shape: Shape,
) -> tuple[np.ndarray, np.ndarray]:
    """Return direct per-bucket benefit and repetition-damage features."""
    retained = rpl.retained_share(weights, geometry, shape.retention, shape.late_multiplier)
    benefit = (retained + shape.benefit_offset) ** (-shape.benefit_exponent)
    excess = np.maximum(rpl.total_epochs(weights, geometry) - shape.damage_threshold, 0.0)
    return benefit, excess**shape.damage_exponent


def design_matrix(weights: np.ndarray, geometry: Geometry, shape: Shape) -> np.ndarray:
    """Columns whose coefficients enter the centered-hierarchy model linearly."""
    benefit, damage = response_blocks(weights, geometry, shape)
    blocks = [
        benefit,
        damage,
        rpl._signed(rpl.concentration_gap(weights, geometry)),
    ]
    if shape.ordering_channel:
        blocks.append(rpl.marginal_phase_block(weights, geometry, shape))
    return np.column_stack(blocks)


def penalty_operator(geometry: Geometry, shape: Shape) -> np.ndarray:
    """Penalty expressed in the candidate's mixed physical coordinates.

    The first two blocks are physical response amplitudes and receive the
    family-centering penalty. Concentration and the final signed asymmetry pair
    remain unpenalized, matching RPL. RPL penalizes the four signed
    family-ordering blocks in normalized coefficient coordinates; those
    identity blocks are retained without modification.
    """
    domains = len(geometry.c0)
    families = len(np.unique(geometry.families))
    columns = 2 * domains + 2 + (4 * families + 2 if shape.ordering_channel else 0)
    operator = np.zeros((columns, columns), dtype=float)
    centering = family_centering_operator(geometry)
    operator[:domains, :domains] = centering
    operator[domains : 2 * domains, domains : 2 * domains] = centering
    if shape.ordering_channel:
        phase_start = 2 * domains + 2
        phase_stop = phase_start + 4 * families
        operator[phase_start:phase_stop, phase_start:phase_stop] = np.eye(4 * families)
    return operator


def penalty_in_normalized_coordinates(
    operator: np.ndarray,
    scale: np.ndarray,
    geometry: Geometry,
) -> np.ndarray:
    """Map the mixed-coordinate penalty onto the solver coefficients.

    The solver uses ``u = scale * theta`` because its data matrix is
    ``design / scale``. The first two response blocks are preregistered in
    physical amplitudes ``theta``, so their operator columns are divided by
    ``scale``. Explicit phase-order columns intentionally keep RPL's normalized
    coefficient ridge and therefore are not transformed.
    """
    if operator.shape != (len(scale), len(scale)):
        raise ValueError("penalty operator must have one row and column per design column")
    domains = len(geometry.c0)
    transformed = operator.copy()
    transformed[:, : 2 * domains] /= scale[: 2 * domains][None, :]
    return transformed


def solve_head(
    design: np.ndarray,
    target: np.ndarray,
    ridge: float,
    operator: np.ndarray,
    geometry: Geometry,
) -> tuple[float, np.ndarray]:
    """Fit RPL's robust nonnegative head with a general quadratic penalty."""
    scale = np.maximum(np.abs(design).max(axis=0), rpl.COLUMN_SCALE_FLOOR)
    augmented = np.column_stack([np.ones(len(target)), design / scale])
    response = target
    if ridge > 0.0:
        normalized_operator = penalty_in_normalized_coordinates(operator, scale, geometry)
        # Keep RPL's all-zero intercept row so singleton-family panels produce
        # the exact same augmented system, not only the same objective.
        penalty = np.zeros((operator.shape[0] + 1, operator.shape[1] + 1))
        penalty[1:, 1:] = np.sqrt(ridge) * normalized_operator
        augmented = np.vstack([augmented, penalty])
        response = np.concatenate([target, np.zeros(penalty.shape[0])])

    coefficients = rpl._bounded_solve(augmented, response, design.shape[1], None)
    if rpl.HUBER_SCALE is not None:
        for _ in range(rpl.HUBER_ITERATIONS):
            residual = augmented[: len(target)] @ coefficients - target
            spread = rpl.MAD_TO_SIGMA * float(np.median(np.abs(residual - np.median(residual))))
            if spread <= 0.0:
                break
            cut = rpl.HUBER_SCALE * spread
            row_weights = np.minimum(1.0, cut / np.maximum(np.abs(residual), 1e-12))
            updated = rpl._bounded_solve(augmented, response, design.shape[1], row_weights)
            shift = float(np.max(np.abs(augmented[: len(target)] @ (updated - coefficients))))
            coefficients = updated
            if shift < rpl.HUBER_TOLERANCE * spread:
                break
    return float(coefficients[0]), coefficients[1:] / scale


@dataclass(frozen=True)
class Fitted:
    """A selected RPL shape with a centered hierarchical linear head."""

    shape: Shape
    ridge: float
    intercept: float
    coefficients: np.ndarray
    geometry: Geometry

    def predict(self, weights: np.ndarray) -> np.ndarray:
        return self.intercept + design_matrix(weights, self.geometry, self.shape) @ self.coefficients

    @property
    def concentration(self) -> float:
        start = 2 * len(self.geometry.c0)
        return float(self.coefficients[start] - self.coefficients[start + 1])


def _shape_score(
    weights: np.ndarray,
    target: np.ndarray,
    geometry: Geometry,
    folds: tuple[tuple[np.ndarray, np.ndarray], ...],
    shape: Shape,
) -> tuple[float, Shape, float]:
    design = design_matrix(weights, geometry, shape)
    if not np.all(np.isfinite(design)):
        return np.inf, shape, rpl.RIDGE_GRID[0]
    operator = penalty_operator(geometry, shape)
    best_score = np.inf
    best_ridge = rpl.RIDGE_GRID[0]
    for ridge in rpl.RIDGE_GRID:
        errors = []
        for train, test in folds:
            intercept, coefficients = solve_head(design[train], target[train], ridge, operator, geometry)
            errors.append(intercept + design[test] @ coefficients - target[test])
        score = float(np.sqrt(np.mean(np.concatenate(errors) ** 2)))
        if score < best_score:
            best_score = score
            best_ridge = ridge
    return best_score, shape, best_ridge


def _shape_batch_scores(
    weights: np.ndarray,
    target: np.ndarray,
    geometry: Geometry,
    folds: tuple[tuple[np.ndarray, np.ndarray], ...],
    shapes: tuple[Shape, ...],
) -> list[tuple[float, Shape, float]]:
    return [_shape_score(weights, target, geometry, folds, shape) for shape in shapes]


def _best_shape_and_ridge(
    scores: list[tuple[float, Shape, float]],
    shapes: tuple[Shape, ...],
) -> tuple[Shape, float]:
    """Select by score with the canonical grid order as the tie-breaker."""
    score_by_shape = {shape: (score, ridge) for score, shape, ridge in scores}
    best_shape = min(shapes, key=lambda shape: score_by_shape[shape][0])
    _, best_ridge = score_by_shape[best_shape]
    return best_shape, best_ridge


def fit(
    weights: np.ndarray,
    target: np.ndarray,
    geometry: Geometry,
    folds: tuple[tuple[np.ndarray, np.ndarray], ...],
    workers: int = 1,
) -> Fitted:
    """Select shape and ridge by grouped OOF error, then refit all rows."""
    if workers < 1:
        raise ValueError("workers must be positive")

    shapes = rpl.shape_grid()
    if workers == 1:
        scores = [_shape_score(weights, target, geometry, folds, shape) for shape in shapes]
    else:
        worker_count = min(workers, len(shapes))
        batch_count = min(len(shapes), worker_count * 4)
        batch_size = (len(shapes) + batch_count - 1) // batch_count
        batches = tuple(shapes[start : start + batch_size] for start in range(0, len(shapes), batch_size))
        scores = []
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            futures = [
                executor.submit(_shape_batch_scores, weights, target, geometry, folds, batch) for batch in batches
            ]
            for future in as_completed(futures):
                scores.extend(future.result())

    best_shape, best_ridge = _best_shape_and_ridge(scores, shapes)
    design = design_matrix(weights, geometry, best_shape)
    intercept, coefficients = solve_head(
        design,
        target,
        best_ridge,
        penalty_operator(geometry, best_shape),
        geometry,
    )
    return Fitted(
        shape=best_shape,
        ridge=best_ridge,
        intercept=intercept,
        coefficients=coefficients,
        geometry=geometry,
    )


def without_phase_terms(model: Fitted) -> Fitted:
    """Return the algebraic phase-free restriction of a fitted model."""
    coefficients = model.coefficients.copy()
    coefficients[2 * len(model.geometry.c0) :] = 0.0
    return replace(
        model,
        shape=replace(model.shape, retention=0.0, late_multiplier=1.0),
        coefficients=coefficients,
    )
