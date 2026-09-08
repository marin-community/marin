# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Retained power law with diminishing returns on pooled family evidence.

This candidate changes one response-law choice in retained power law (RPL).
RPL transforms each bucket's retained state and then sums transformed values
within a family. Here, retained state is summed within a predeclared semantic
family before the power-law response:

    B_f = (sum_{i in f} s_i + |f| * e0) ** -p.

Bucket residual columns and repetition damage remain bucket-specific. The
existing approximate ordering block is deliberately unchanged so this is a
single-axis comparison of pre- versus post-nonlinearity family pooling.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass

import numpy as np

from experiments.domain_phase_mix.exploratory.two_phase_many import retained_power_law_model_20260728 as rpl

Geometry = rpl.Geometry
Shape = rpl.Shape
penalty_multipliers = rpl.penalty_multipliers


def family_counts(geometry: Geometry) -> np.ndarray:
    """Number of buckets in each predeclared family."""
    families = geometry.families
    return np.bincount(families, minlength=int(families.max()) + 1)


def pooled_family_benefit(retained_state: np.ndarray, geometry: Geometry, shape: Shape) -> np.ndarray:
    """Power-law response to total retained evidence in each family."""
    family_state = rpl._family_totals(retained_state, geometry)
    offset = family_counts(geometry) * shape.benefit_offset
    return (family_state + offset) ** (-shape.benefit_exponent)


def mean_family_benefit(retained_state: np.ndarray, geometry: Geometry, shape: Shape) -> np.ndarray:
    """Equivalent average-state form, before its absorbable family scale."""
    counts = family_counts(geometry)
    family_mean = rpl._family_totals(retained_state, geometry) / counts
    return (family_mean + shape.benefit_offset) ** (-shape.benefit_exponent)


def benefit_block(weights: np.ndarray, geometry: Geometry, shape: Shape) -> np.ndarray:
    """Family response plus shrunk bucket departures, matching RPL's column count."""
    retained_state = rpl.retained_share(weights, geometry, shape.retention, shape.late_multiplier)
    family = pooled_family_benefit(retained_state, geometry, shape)
    excess = geometry.excess_domains
    if not len(excess):
        return family
    bucket = (retained_state + shape.benefit_offset) ** (-shape.benefit_exponent)
    return np.column_stack([family, bucket[:, excess]])


def design_matrix(weights: np.ndarray, geometry: Geometry, shape: Shape) -> np.ndarray:
    """Fixed response features for one nonlinear shape."""
    excess = np.maximum(rpl.total_epochs(weights, geometry) - shape.damage_threshold, 0.0)
    blocks = [
        benefit_block(weights, geometry, shape),
        rpl._hierarchical_block(excess**shape.damage_exponent, geometry),
        rpl._signed(rpl.concentration_gap(weights, geometry)),
    ]
    if shape.ordering_channel:
        blocks.append(rpl.marginal_phase_block(weights, geometry, shape))
    return np.column_stack(blocks)


@dataclass(frozen=True)
class Fitted:
    """A selected nonlinear shape and its fitted linear response head."""

    shape: Shape
    ridge: float
    intercept: float
    coefficients: np.ndarray
    geometry: Geometry

    def predict(self, weights: np.ndarray) -> np.ndarray:
        return self.intercept + design_matrix(weights, self.geometry, self.shape) @ self.coefficients

    @property
    def concentration(self) -> float:
        families = len(np.unique(self.geometry.families))
        start = 2 * (families + len(self.geometry.excess_domains))
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

    multipliers = rpl.penalty_multipliers(geometry, shape)
    best_score = np.inf
    best_ridge = rpl.RIDGE_GRID[0]
    for ridge in rpl.RIDGE_GRID:
        errors = []
        for train, test in folds:
            intercept, coefficients = rpl.solve_head(design[train], target[train], ridge, multipliers)
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

    _, best_shape, best_ridge = min(scores, key=lambda result: result[0])
    design = design_matrix(weights, geometry, best_shape)
    intercept, coefficients = rpl.solve_head(
        design,
        target,
        best_ridge,
        rpl.penalty_multipliers(geometry, best_shape),
    )
    return Fitted(
        shape=best_shape,
        ridge=best_ridge,
        intercept=intercept,
        coefficients=coefficients,
        geometry=geometry,
    )
