# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "cvxpy",
#   "numpy",
#   "pandas",
#   "plotly",
#   "scikit-learn",
#   "scipy",
#   "tabulate",
# ]
# ///

"""Test whether the calibrated dense WSD80 interiors admit a convex surrogate.

This is a shape-falsification audit, not a new headline surrogate. It combines
two complementary tests:

1. Exact midpoint Jensen inequalities with coordinate-specific heteroskedastic
   uncertainty. These are model-free necessary conditions for convexity.
2. Spatial CV for exact PSD and unconstrained quadratics, plus nested spatial
   CV for matched conditional-control, ridge-spline, and nonparametric convex
   regression diagnostics.

Response fitting uses the shared 125-coordinate design in each of 28
horizon-by-replay cells, with calibration-derived inverse-variance weights.
The larger single-seed shape screen excludes the noisy boundary coordinates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

import cvxpy as cp
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.optimize import lsq_linear
from scipy.spatial import Delaunay
from scipy.stats import norm, spearmanr, t
from sklearn.cluster import KMeans

SCRIPT_DIR = Path(__file__).resolve().parent
REFERENCE_OUTPUTS = SCRIPT_DIR / "reference_outputs"
CALIBRATION_DIR = REFERENCE_OUTPUTS / "starcoder_wsd80_dense_support_calibration_results_20260813"
DEFAULT_COVERAGE = CALIBRATION_DIR / "coverage_with_calibration_weights.csv"
DEFAULT_EVIDENCE = CALIBRATION_DIR / "surface_evidence_comparison.csv"
DEFAULT_CALIBRATION_SEEDS = CALIBRATION_DIR / "calibration_aligned_seed_observations.csv"
DEFAULT_OUTPUT_DIR = REFERENCE_OUTPUTS / "starcoder_wsd80_convex_surrogate_viability_20260815"

INTERIOR_MARGIN = 0.025
OUTER_FOLDS = 4
INNER_FOLDS = 3
OUTER_SEED = 20260815
INNER_SEED = 20260816
RIDGE_GRID = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0)
CONVEX_REGRESSION_RIDGE_GRID = (1e-2, 1e-1, 1.0, 10.0, 100.0)
SPLINE_DIRECTIONS = 12
SPLINE_KNOT_QUANTILES = (0.2, 0.4, 0.6, 0.8)
AGGREGATE_CONTROL_DEGREE = 3
OPTIMUM_GRID = 241
HOLM_ALPHA = 0.05
VARIANCE_INFLATION_GRID = (1.0, 4.0, 9.0, 17.7, 25.0, 100.0)

SUPPORT_ORDER = ("full", "m0125", "m025", "m050", "m100", "m200", "m400")
SUPPORT_LABELS = {
    "full": "full pool",
    "m0125": "0.125x",
    "m025": "0.25x",
    "m050": "0.5x",
    "m100": "1x",
    "m200": "2x",
    "m400": "4x",
}
MODEL_ORDER = (
    "convex_quadratic",
    "unconstrained_quadratic",
    "conditional_convex_cubic",
    "conditional_unconstrained_cubic",
    "convex_ridge_spline",
    "unconstrained_ridge_spline",
    "nonparametric_convex_regression",
)
MODEL_LABELS = {
    "convex_quadratic": "Exact PSD quadratic",
    "unconstrained_quadratic": "Matched unconstrained quadratic",
    "conditional_convex_cubic": "Aggregate-conditioned convex cubic",
    "conditional_unconstrained_cubic": "Matched conditional cubic",
    "convex_ridge_spline": "Convex ridge-spline",
    "unconstrained_ridge_spline": "Matched unconstrained ridge-spline",
    "nonparametric_convex_regression": "Nonparametric convex regression",
}


@dataclass(frozen=True)
class Basis:
    """Coordinate-only basis specification."""

    kind: str
    directions: np.ndarray
    knots: np.ndarray | None
    nonnegative: bool


@dataclass(frozen=True)
class SurfaceModel:
    """One fitted shape-constrained or unconstrained surface."""

    basis: Basis
    alpha: float
    response_center: float
    response_scale: float
    free_scale: np.ndarray
    constrained_scale: np.ndarray
    coefficients: np.ndarray

    def predict(self, coordinates: np.ndarray) -> np.ndarray:
        free, constrained = design_matrix(coordinates, self.basis)
        normalized = (free / self.free_scale) @ self.coefficients[: free.shape[1]]
        normalized += (constrained / self.constrained_scale) @ self.coefficients[free.shape[1] :]
        return self.response_center + self.response_scale * normalized


@dataclass(frozen=True)
class QuadraticSurfaceModel:
    """A direct quadratic with an optional exact PSD Hessian constraint."""

    coordinate_center: np.ndarray
    coordinate_scale: np.ndarray
    response_center: float
    response_scale: float
    intercept: float
    linear: np.ndarray
    hessian: np.ndarray

    def predict(self, coordinates: np.ndarray) -> np.ndarray:
        normalized = (np.asarray(coordinates, dtype=float) - self.coordinate_center) / self.coordinate_scale
        quadratic = 0.5 * np.einsum("ni,ij,nj->n", normalized, self.hessian, normalized)
        prediction = self.intercept + normalized @ self.linear + quadratic
        return self.response_center + self.response_scale * prediction


@dataclass(frozen=True)
class ConvexRegressionModel:
    """A max-affine convex regression fit used only as a shape-class diagnostic."""

    coordinate_center: np.ndarray
    coordinate_scale: np.ndarray
    response_center: float
    response_scale: float
    training_coordinates: np.ndarray
    fitted_values: np.ndarray
    gradients: np.ndarray
    solver: str

    def predict(self, coordinates: np.ndarray) -> np.ndarray:
        normalized = (np.asarray(coordinates, dtype=float) - self.coordinate_center) / self.coordinate_scale
        intercepts = self.fitted_values - np.sum(self.gradients * self.training_coordinates, axis=1)
        prediction = np.max(intercepts[:, None] + self.gradients @ normalized.T, axis=0)
        return self.response_center + self.response_scale * prediction


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coverage", type=Path, default=DEFAULT_COVERAGE)
    parser.add_argument("--evidence", type=Path, default=DEFAULT_EVIDENCE)
    parser.add_argument("--calibration-seeds", type=Path, default=DEFAULT_CALIBRATION_SEEDS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def interior_rows(frame: pd.DataFrame) -> pd.Series:
    p0 = frame["phase_0_starcoder"]
    p1 = frame["phase_1_starcoder"]
    return p0.gt(INTERIOR_MARGIN) & p0.lt(1.0 - INTERIOR_MARGIN) & p1.gt(INTERIOR_MARGIN) & p1.lt(1.0 - INTERIOR_MARGIN)


def make_bases(coordinates: np.ndarray) -> dict[str, Basis]:
    spline_angles = np.linspace(0.0, 2.0 * np.pi, SPLINE_DIRECTIONS, endpoint=False)
    spline_directions = np.column_stack((np.cos(spline_angles), np.sin(spline_angles)))
    projections = coordinates @ spline_directions.T
    knots = np.quantile(projections, SPLINE_KNOT_QUANTILES, axis=0).T
    return {
        "conditional_convex_cubic": Basis(
            kind="aggregate_control",
            directions=np.empty((0, 2)),
            knots=None,
            nonnegative=True,
        ),
        "conditional_unconstrained_cubic": Basis(
            kind="aggregate_control",
            directions=np.empty((0, 2)),
            knots=None,
            nonnegative=False,
        ),
        "convex_ridge_spline": Basis(
            kind="ridge_spline",
            directions=spline_directions,
            knots=knots,
            nonnegative=True,
        ),
        "unconstrained_ridge_spline": Basis(
            kind="ridge_spline",
            directions=spline_directions,
            knots=knots,
            nonnegative=False,
        ),
    }


def fit_quadratic_surface(
    coordinates: np.ndarray,
    response: np.ndarray,
    weights: np.ndarray,
    *,
    convex: bool,
) -> QuadraticSurfaceModel:
    weights = np.asarray(weights, dtype=float)
    weights = weights / np.mean(weights)
    coordinate_center = np.average(coordinates, axis=0, weights=weights)
    coordinate_scale = np.sqrt(np.average((coordinates - coordinate_center) ** 2, axis=0, weights=weights))
    coordinate_scale = np.maximum(coordinate_scale, 1e-6)
    normalized = (coordinates - coordinate_center) / coordinate_scale
    response_center = float(np.average(response, weights=weights))
    response_scale = float(np.sqrt(np.average((response - response_center) ** 2, weights=weights)))
    response_scale = max(response_scale, 1e-6)
    target = (response - response_center) / response_scale
    root_weight = np.sqrt(weights)

    if convex:
        intercept = cp.Variable()
        linear = cp.Variable(2)
        hessian = cp.Variable((2, 2), symmetric=True)
        quadratic = 0.5 * cp.sum(cp.multiply(normalized @ hessian, normalized), axis=1)
        residual = cp.multiply(root_weight, intercept + normalized @ linear + quadratic - target)
        problem = cp.Problem(cp.Minimize(cp.sum_squares(residual)), [hessian >> 0])
        problem.solve(solver="CLARABEL")
        if problem.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
            raise RuntimeError(f"PSD quadratic solve failed: {problem.status}")
        fitted_intercept = float(intercept.value)
        fitted_linear = np.asarray(linear.value, dtype=float)
        fitted_hessian = np.asarray(hessian.value, dtype=float)
    else:
        design = np.column_stack(
            (
                np.ones(len(normalized)),
                normalized[:, 0],
                normalized[:, 1],
                0.5 * normalized[:, 0] ** 2,
                normalized[:, 0] * normalized[:, 1],
                0.5 * normalized[:, 1] ** 2,
            )
        )
        coefficients = np.linalg.lstsq(design * root_weight[:, None], target * root_weight, rcond=None)[0]
        fitted_intercept = float(coefficients[0])
        fitted_linear = coefficients[1:3]
        fitted_hessian = np.asarray(
            [[coefficients[3], coefficients[4]], [coefficients[4], coefficients[5]]],
            dtype=float,
        )

    return QuadraticSurfaceModel(
        coordinate_center=coordinate_center,
        coordinate_scale=coordinate_scale,
        response_center=response_center,
        response_scale=response_scale,
        intercept=fitted_intercept,
        linear=fitted_linear,
        hessian=fitted_hessian,
    )


def fit_convex_regression(
    coordinates: np.ndarray,
    response: np.ndarray,
    weights: np.ndarray,
    alpha: float,
) -> ConvexRegressionModel:
    """Fit least-squares convex regression with ridge-regularized subgradients."""
    weights = np.asarray(weights, dtype=float)
    weights = weights / np.mean(weights)
    coordinate_center = np.average(coordinates, axis=0, weights=weights)
    coordinate_scale = np.sqrt(np.average((coordinates - coordinate_center) ** 2, axis=0, weights=weights))
    coordinate_scale = np.maximum(coordinate_scale, 1e-6)
    normalized = (coordinates - coordinate_center) / coordinate_scale
    response_center = float(np.average(response, weights=weights))
    response_scale = float(np.sqrt(np.average((response - response_center) ** 2, weights=weights)))
    response_scale = max(response_scale, 1e-6)
    target = (response - response_center) / response_scale

    point_count = len(coordinates)
    fitted_values = cp.Variable(point_count)
    gradients = cp.Variable((point_count, 2))
    tangent_intercepts = fitted_values - cp.sum(cp.multiply(gradients, normalized), axis=1)
    tangent_values = cp.reshape(tangent_intercepts, (point_count, 1), order="C") + gradients @ normalized.T
    fitted_row = cp.reshape(fitted_values, (1, point_count), order="C")
    objective = cp.sum(cp.multiply(weights, cp.square(fitted_values - target)))
    objective += alpha * cp.sum_squares(gradients)
    problem = cp.Problem(cp.Minimize(objective), [fitted_row >= tangent_values])
    solver = "CLARABEL"
    try:
        problem.solve(solver=solver)
    except cp.error.SolverError:
        solver = "SCS"
        problem.solve(solver=solver, eps=1e-6, max_iters=100_000)
    if problem.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE} and solver != "SCS":
        solver = "SCS"
        problem.solve(solver=solver, eps=1e-6, max_iters=100_000)
    if problem.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
        raise RuntimeError(f"Convex regression solve failed: {problem.status}")
    fitted_array = np.asarray(fitted_values.value, dtype=float)
    gradient_array = np.asarray(gradients.value, dtype=float)
    intercept_array = fitted_array - np.sum(gradient_array * normalized, axis=1)
    minimum_slack = np.min(fitted_array[None, :] - (intercept_array[:, None] + gradient_array @ normalized.T))
    if minimum_slack < -1e-4:
        raise RuntimeError(f"Convex regression violated a shape constraint by {-minimum_slack:.3e}")
    return ConvexRegressionModel(
        coordinate_center=coordinate_center,
        coordinate_scale=coordinate_scale,
        response_center=response_center,
        response_scale=response_scale,
        training_coordinates=normalized,
        fitted_values=fitted_array,
        gradients=gradient_array,
        solver=solver,
    )


def design_matrix(coordinates: np.ndarray, basis: Basis) -> tuple[np.ndarray, np.ndarray]:
    coordinates = np.asarray(coordinates, dtype=float)
    if basis.kind == "aggregate_control":
        aggregate = 0.8 * coordinates[:, 0] + 0.2 * coordinates[:, 1]
        contrast = coordinates[:, 0] - coordinates[:, 1]
        bernstein = np.column_stack(
            [
                math.comb(AGGREGATE_CONTROL_DEGREE, index)
                * aggregate**index
                * (1.0 - aggregate) ** (AGGREGATE_CONTROL_DEGREE - index)
                for index in range(AGGREGATE_CONTROL_DEGREE + 1)
            ]
        )
        free = np.column_stack((bernstein, contrast[:, None] * bernstein))
        constrained = 0.5 * contrast[:, None] ** 2 * bernstein
        return free, constrained

    free = np.column_stack((np.ones(len(coordinates)), coordinates))
    projections = coordinates @ basis.directions.T
    if basis.kind == "quadratic":
        constrained = projections**2
    elif basis.kind == "ridge_spline":
        assert basis.knots is not None
        constrained = np.maximum(projections[:, :, None] - basis.knots[None, :, :], 0.0) ** 2
        constrained = constrained.reshape(len(coordinates), -1)
    else:
        raise ValueError(f"Unknown basis kind: {basis.kind}")
    return free, constrained


def fit_surface(
    coordinates: np.ndarray,
    response: np.ndarray,
    weights: np.ndarray,
    basis: Basis,
    alpha: float,
) -> SurfaceModel:
    weights = np.asarray(weights, dtype=float)
    weights = weights / np.mean(weights)
    response_center = float(np.average(response, weights=weights))
    response_scale = float(np.sqrt(np.average((response - response_center) ** 2, weights=weights)))
    response_scale = max(response_scale, 1e-6)
    normalized_response = (response - response_center) / response_scale

    free, constrained = design_matrix(coordinates, basis)
    free_scale = np.ones(free.shape[1])
    if basis.kind == "aggregate_control":
        aggregate_terms = AGGREGATE_CONTROL_DEGREE + 1
        free_scale[aggregate_terms:] = np.sqrt(np.average(free[:, aggregate_terms:] ** 2, axis=0, weights=weights))
        free_scale = np.maximum(free_scale, 1e-8)
    normalized_free = free / free_scale
    constrained_scale = np.sqrt(np.average(constrained**2, axis=0, weights=weights))
    constrained_scale = np.maximum(constrained_scale, 1e-8)
    normalized_constrained = constrained / constrained_scale
    design = np.column_stack((normalized_free, normalized_constrained))

    root_weight = np.sqrt(weights)
    weighted_design = design * root_weight[:, None]
    weighted_response = normalized_response * root_weight
    if basis.kind == "aggregate_control":
        aggregate_terms = AGGREGATE_CONTROL_DEGREE + 1
        ridge_indices = np.arange(aggregate_terms, design.shape[1])
    else:
        ridge_indices = np.arange(free.shape[1], design.shape[1])
    ridge = np.zeros((len(ridge_indices), design.shape[1]))
    ridge[np.arange(len(ridge_indices)), ridge_indices] = np.sqrt(alpha)
    augmented_design = np.vstack((weighted_design, ridge))
    augmented_response = np.concatenate((weighted_response, np.zeros(len(ridge))))

    if basis.nonnegative:
        lower = np.concatenate((np.full(free.shape[1], -np.inf), np.zeros(normalized_constrained.shape[1])))
        upper = np.full(design.shape[1], np.inf)
        solved = lsq_linear(
            augmented_design,
            augmented_response,
            bounds=(lower, upper),
            method="trf",
            lsmr_tol="auto",
            max_iter=5000,
        )
        if not solved.success:
            raise RuntimeError(f"Shape-constrained solve failed: {solved.message}")
        coefficients = solved.x
    else:
        coefficients = np.linalg.lstsq(augmented_design, augmented_response, rcond=None)[0]

    return SurfaceModel(
        basis=basis,
        alpha=alpha,
        response_center=response_center,
        response_scale=response_scale,
        free_scale=free_scale,
        constrained_scale=constrained_scale,
        coefficients=coefficients,
    )


def spatial_folds(coordinates: np.ndarray, n_splits: int, seed: int) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    labels = KMeans(n_clusters=n_splits, n_init=20, random_state=seed).fit_predict(coordinates)
    indices = np.arange(len(coordinates))
    return tuple((indices[labels != label], indices[labels == label]) for label in np.unique(labels))


def weighted_rmse(residual: np.ndarray, weights: np.ndarray) -> float:
    return float(np.sqrt(np.average(residual**2, weights=weights)))


def select_alpha(
    coordinates: np.ndarray,
    response: np.ndarray,
    weights: np.ndarray,
    basis: Basis,
    folds: tuple[tuple[np.ndarray, np.ndarray], ...],
) -> tuple[float, tuple[tuple[float, float], ...]]:
    scores: list[tuple[float, float]] = []
    for alpha in RIDGE_GRID:
        residuals: list[np.ndarray] = []
        held_weights: list[np.ndarray] = []
        for train, test in folds:
            model = fit_surface(coordinates[train], response[train], weights[train], basis, alpha)
            residuals.append(model.predict(coordinates[test]) - response[test])
            held_weights.append(weights[test])
        score = weighted_rmse(np.concatenate(residuals), np.concatenate(held_weights))
        scores.append((alpha, score))
    selected = min(scores, key=lambda item: (item[1], -item[0]))[0]
    return selected, tuple(scores)


def nested_predictions(
    coordinates: np.ndarray,
    response: np.ndarray,
    weights: np.ndarray,
    basis: Basis,
) -> tuple[np.ndarray, tuple[float, ...], float, tuple[tuple[float, float], ...]]:
    outer = spatial_folds(coordinates, OUTER_FOLDS, OUTER_SEED)
    predictions = np.full(len(response), np.nan)
    selected_alphas: list[float] = []
    for fold_id, (train, test) in enumerate(outer):
        local_folds = spatial_folds(coordinates[train], INNER_FOLDS, INNER_SEED + fold_id)
        inner = tuple((train[local_train], train[local_test]) for local_train, local_test in local_folds)
        alpha, _ = select_alpha(coordinates, response, weights, basis, inner)
        selected_alphas.append(alpha)
        model = fit_surface(coordinates[train], response[train], weights[train], basis, alpha)
        predictions[test] = model.predict(coordinates[test])
    if not np.isfinite(predictions).all():
        raise RuntimeError("Nested spatial CV did not predict every interior coordinate")
    full_alpha, full_scores = select_alpha(coordinates, response, weights, basis, outer)
    return predictions, tuple(selected_alphas), full_alpha, full_scores


def quadratic_spatial_predictions(
    coordinates: np.ndarray,
    response: np.ndarray,
    weights: np.ndarray,
    *,
    convex: bool,
) -> tuple[np.ndarray, QuadraticSurfaceModel]:
    predictions = np.full(len(response), np.nan)
    for train, test in spatial_folds(coordinates, OUTER_FOLDS, OUTER_SEED):
        model = fit_quadratic_surface(
            coordinates[train],
            response[train],
            weights[train],
            convex=convex,
        )
        predictions[test] = model.predict(coordinates[test])
    if not np.isfinite(predictions).all():
        raise RuntimeError("Spatial CV did not predict every interior coordinate")
    return predictions, fit_quadratic_surface(coordinates, response, weights, convex=convex)


def select_convex_regression_alpha(
    coordinates: np.ndarray,
    response: np.ndarray,
    weights: np.ndarray,
    folds: tuple[tuple[np.ndarray, np.ndarray], ...],
) -> tuple[float, tuple[tuple[float, float], ...]]:
    scores: list[tuple[float, float]] = []
    for alpha in CONVEX_REGRESSION_RIDGE_GRID:
        residuals: list[np.ndarray] = []
        held_weights: list[np.ndarray] = []
        for train, test in folds:
            model = fit_convex_regression(coordinates[train], response[train], weights[train], alpha)
            residuals.append(model.predict(coordinates[test]) - response[test])
            held_weights.append(weights[test])
        score = weighted_rmse(np.concatenate(residuals), np.concatenate(held_weights))
        scores.append((alpha, score))
    selected = min(scores, key=lambda item: (item[1], -item[0]))[0]
    return selected, tuple(scores)


def convex_regression_spatial_predictions(
    coordinates: np.ndarray,
    response: np.ndarray,
    weights: np.ndarray,
) -> tuple[np.ndarray, tuple[float, ...], float, tuple[tuple[float, float], ...], ConvexRegressionModel]:
    outer = spatial_folds(coordinates, OUTER_FOLDS, OUTER_SEED)
    predictions = np.full(len(response), np.nan)
    selected_alphas: list[float] = []
    for fold_id, (train, test) in enumerate(outer):
        local_folds = spatial_folds(coordinates[train], INNER_FOLDS, INNER_SEED + fold_id)
        inner = tuple((train[local_train], train[local_test]) for local_train, local_test in local_folds)
        alpha, _ = select_convex_regression_alpha(coordinates, response, weights, inner)
        selected_alphas.append(alpha)
        model = fit_convex_regression(coordinates[train], response[train], weights[train], alpha)
        predictions[test] = model.predict(coordinates[test])
    if not np.isfinite(predictions).all():
        raise RuntimeError("Nested convex regression did not predict every coordinate")
    full_alpha, full_scores = select_convex_regression_alpha(coordinates, response, weights, outer)
    full_model = fit_convex_regression(coordinates, response, weights, full_alpha)
    return predictions, tuple(selected_alphas), full_alpha, full_scores, full_model


def optimum_diagnostics(
    model: SurfaceModel | QuadraticSurfaceModel | ConvexRegressionModel,
    coordinates: np.ndarray,
    response: np.ndarray,
    evidence: pd.Series,
) -> dict[str, float]:
    axis = np.linspace(0.0, 1.0, OPTIMUM_GRID)
    p0, p1 = np.meshgrid(axis, axis, indexing="ij")
    candidates = np.column_stack((p0.ravel(), p1.ravel()))
    hull = Delaunay(coordinates)
    inside = hull.find_simplex(candidates) >= 0
    usable = candidates[inside]
    prediction = model.predict(usable)
    best = int(np.argmin(prediction))
    optimum = usable[best]

    tied = np.column_stack((axis, axis))
    tied_inside = hull.find_simplex(tied) >= 0
    tied = tied[tied_inside]
    tied_prediction = model.predict(tied)
    tied_best = int(np.argmin(tied_prediction))
    nearest = float(np.min(np.linalg.norm(coordinates - optimum[None, :], axis=1)))
    observed_grid_prediction = model.predict(coordinates)
    observed_grid_rank = np.argsort(observed_grid_prediction)
    observed_grid_best = int(np.argmin(response))
    observed_grid_top_five = observed_grid_rank[: min(5, len(observed_grid_rank))]
    fresh_tied = np.asarray([evidence["tied_p0"], evidence["tied_p1"]], dtype=float)
    fresh_untied = np.asarray([evidence["untied_p0"], evidence["untied_p1"]], dtype=float)
    fresh_pair_prediction = model.predict(np.stack((fresh_tied, fresh_untied)))
    return {
        "predicted_tied_p": float(tied[tied_best, 0]),
        "predicted_tied_bpb": float(tied_prediction[tied_best]),
        "predicted_untied_p0": float(optimum[0]),
        "predicted_untied_p1": float(optimum[1]),
        "predicted_untied_bpb": float(prediction[best]),
        "predicted_global_two_phase_gain_bpb": float(tied_prediction[tied_best] - prediction[best]),
        "predicted_fresh_selected_pair_gain_bpb": float(fresh_pair_prediction[0] - fresh_pair_prediction[1]),
        "optimum_nearest_design_l2": nearest,
        "optimum_distance_to_fresh_selected_l2": float(np.linalg.norm(optimum - fresh_untied)),
        "full_fit_observed_grid_regret_at_1": float(response[observed_grid_rank[0]] - response[observed_grid_best]),
        "full_fit_observed_grid_regret_at_5": float(
            np.min(response[observed_grid_top_five]) - response[observed_grid_best]
        ),
    }


def midpoint_triples(coordinates: np.ndarray) -> list[tuple[int, int, int]]:
    lookup = {tuple(np.round(point, 10)): index for index, point in enumerate(coordinates)}
    triples: set[tuple[int, int, int]] = set()
    for left in range(len(coordinates)):
        for right in range(left + 1, len(coordinates)):
            middle = lookup.get(tuple(np.round(0.5 * (coordinates[left] + coordinates[right]), 10)))
            if middle is None or middle in (left, right):
                continue
            triples.add((left, middle, right))
    return sorted(triples)


def direction_label(delta: np.ndarray) -> str:
    scale = float(np.max(np.abs(delta)))
    if scale == 0.0:
        return "degenerate"
    tolerance = 1e-7 * scale
    if abs(delta[0] - delta[1]) <= tolerance:
        return "tied_diagonal"
    if abs(0.8 * delta[0] + 0.2 * delta[1]) <= tolerance:
        return "fixed_aggregate_fiber"
    if abs(delta[0]) <= tolerance:
        return "phase_1_axis"
    if abs(delta[1]) <= tolerance:
        return "phase_0_axis"
    return "mixed"


def holm_adjust(p_values: np.ndarray) -> np.ndarray:
    order = np.argsort(p_values)
    adjusted = np.empty_like(p_values)
    running = 0.0
    total = len(p_values)
    for rank, index in enumerate(order):
        running = max(running, (total - rank) * float(p_values[index]))
        adjusted[index] = min(running, 1.0)
    return adjusted


def jensen_rows(
    group: pd.DataFrame,
    coordinates: np.ndarray,
    triples: list[tuple[int, int, int]],
) -> list[dict[str, object]]:
    response = group["bpb"].to_numpy(dtype=float)
    sigma = group["predicted_sd_bpb"].to_numpy(dtype=float)
    observed_best = float(np.min(response))
    aliases = group["is_alias"].astype(bool).to_numpy()
    rows: list[dict[str, object]] = []
    for left, middle, right in triples:
        gap = float(response[middle] - 0.5 * (response[left] + response[right]))
        standard_error = float(np.sqrt(sigma[middle] ** 2 + 0.25 * sigma[left] ** 2 + 0.25 * sigma[right] ** 2))
        z_score = gap / standard_error
        segment_max = float(max(response[left], response[middle], response[right]))
        rows.append(
            {
                "cell_id": group["cell_id"].iloc[0],
                "support_id": group["support_id"].iloc[0],
                "triple_id": f"{left}:{middle}:{right}",
                "left_coordinate_id": group["coordinate_id"].iloc[left],
                "middle_coordinate_id": group["coordinate_id"].iloc[middle],
                "right_coordinate_id": group["coordinate_id"].iloc[right],
                "direction": direction_label(coordinates[right] - coordinates[left]),
                "contains_alias": bool(aliases[left] or aliases[middle] or aliases[right]),
                "segment_max_bpb": segment_max,
                "segment_max_excess_over_best_bpb": segment_max - observed_best,
                "jensen_gap_bpb": gap,
                "conservative_standard_error_bpb": standard_error,
                "z_score": z_score,
                "one_sided_p": float(norm.sf(z_score)),
            }
        )
    p_values = np.asarray([row["one_sided_p"] for row in rows], dtype=float)
    adjusted = holm_adjust(p_values)
    for row, value in zip(rows, adjusted, strict=True):
        row["holm_p"] = float(value)
        row["holm_rejects_convexity"] = bool(value < HOLM_ALPHA and row["jensen_gap_bpb"] > 0.0)
    return rows


def calibration_variance_components(calibration_seeds: pd.DataFrame) -> pd.DataFrame:
    """Estimate within-block noise after removing a shared seed offset."""
    records: list[dict[str, object]] = []
    for (cell_id, support_id), group in calibration_seeds.groupby(["cell_id", "support_id"], sort=False):
        pivot = group.pivot(index="coordinate_id", columns="data_seed", values="bpb")
        if pivot.isna().any().any():
            raise ValueError(f"Calibration seed block {(cell_id, support_id)} is not rectangular")
        values = pivot.to_numpy(dtype=float)
        residual = values - values.mean(axis=1, keepdims=True) - values.mean(axis=0, keepdims=True) + values.mean()
        degrees_of_freedom = (values.shape[0] - 1) * (values.shape[1] - 1)
        if degrees_of_freedom <= 0:
            raise ValueError(f"Calibration seed block {(cell_id, support_id)} has no residual degrees of freedom")
        idiosyncratic_variance = float(np.sum(residual**2) / degrees_of_freedom)
        seed_offsets = values.mean(axis=0) - values.mean()
        records.append(
            {
                "cell_id": cell_id,
                "support_id": support_id,
                "calibration_coordinates": values.shape[0],
                "calibration_seeds": values.shape[1],
                "variance_degrees_of_freedom": degrees_of_freedom,
                "idiosyncratic_variance_bpb2": idiosyncratic_variance,
                "idiosyncratic_sd_bpb": float(np.sqrt(idiosyncratic_variance)),
                "common_seed_offset_sd_bpb": float(np.std(seed_offsets, ddof=1)),
            }
        )
    return pd.DataFrame(records)


def apply_variance_component_inference(
    jensen: pd.DataFrame,
    variance_components: pd.DataFrame,
) -> pd.DataFrame:
    result = jensen.merge(
        variance_components,
        on=["cell_id", "support_id"],
        how="left",
        validate="many_to_one",
    )
    if result["idiosyncratic_variance_bpb2"].isna().any():
        raise ValueError("Missing calibration variance component for a Jensen contrast")
    result["paired_standard_error_bpb"] = np.sqrt(1.5 * result["idiosyncratic_variance_bpb2"])
    result["paired_t_score"] = result["jensen_gap_bpb"] / result["paired_standard_error_bpb"]
    result["paired_one_sided_p"] = t.sf(
        result["paired_t_score"],
        result["variance_degrees_of_freedom"],
    )
    result["paired_cell_holm_p"] = np.nan
    result["paired_cell_holm_rejects_convexity"] = False
    eligible = ~result["contains_alias"]
    for _, indices in result.loc[eligible].groupby(["cell_id", "support_id"], sort=False).groups.items():
        adjusted = holm_adjust(result.loc[indices, "paired_one_sided_p"].to_numpy(dtype=float))
        result.loc[indices, "paired_cell_holm_p"] = adjusted
        result.loc[indices, "paired_cell_holm_rejects_convexity"] = (adjusted < HOLM_ALPHA) & result.loc[
            indices, "jensen_gap_bpb"
        ].gt(0.0)
    global_indices = result.index[eligible]
    global_adjusted = holm_adjust(result.loc[global_indices, "paired_one_sided_p"].to_numpy(dtype=float))
    result["paired_global_holm_p"] = np.nan
    result["paired_global_holm_rejects_convexity"] = False
    result.loc[global_indices, "paired_global_holm_p"] = global_adjusted
    result.loc[global_indices, "paired_global_holm_rejects_convexity"] = (global_adjusted < HOLM_ALPHA) & result.loc[
        global_indices, "jensen_gap_bpb"
    ].gt(0.0)
    return result


def coordinate_influence(jensen: pd.DataFrame) -> pd.DataFrame:
    """Recompute the global Holm result after deleting each implicated coordinate."""
    primary = jensen.loc[~jensen["contains_alias"]]
    implicated = sorted(
        set(primary.loc[primary["paired_global_holm_rejects_convexity"], "left_coordinate_id"])
        | set(primary.loc[primary["paired_global_holm_rejects_convexity"], "middle_coordinate_id"])
        | set(primary.loc[primary["paired_global_holm_rejects_convexity"], "right_coordinate_id"])
    )
    records: list[dict[str, object]] = []
    for coordinate_id in [None, *implicated]:
        if coordinate_id is None:
            subset = primary
            label = "none"
        else:
            subset = primary.loc[
                primary[["left_coordinate_id", "middle_coordinate_id", "right_coordinate_id"]]
                .ne(coordinate_id)
                .all(axis=1)
            ]
            label = coordinate_id
        adjusted = holm_adjust(subset["paired_one_sided_p"].to_numpy(dtype=float))
        rejected = subset.loc[(adjusted < HOLM_ALPHA) & subset["jensen_gap_bpb"].gt(0.0)]
        records.append(
            {
                "removed_coordinate_id": label,
                "remaining_contrasts": len(subset),
                "global_holm_rejections": len(rejected),
                "rejecting_surfaces": rejected[["cell_id", "support_id"]].drop_duplicates().shape[0],
                "rejecting_exact_triples": rejected["triple_id"].nunique(),
            }
        )
    return pd.DataFrame(records)


def collinear_chords(coordinates: np.ndarray) -> list[tuple[int, int, int, float]]:
    chords: list[tuple[int, int, int, float]] = []
    for left in range(len(coordinates)):
        for right in range(left + 1, len(coordinates)):
            direction = coordinates[right] - coordinates[left]
            squared_length = float(direction @ direction)
            if squared_length <= 1e-15:
                continue
            for middle in range(len(coordinates)):
                if middle in (left, right):
                    continue
                fraction = float((coordinates[middle] - coordinates[left]) @ direction / squared_length)
                if not 1e-8 < fraction < 1.0 - 1e-8:
                    continue
                expected = coordinates[left] + fraction * direction
                if np.linalg.norm(coordinates[middle] - expected) <= 1e-8:
                    chords.append((left, middle, right, fraction))
    return chords


def aligned_seed_chord_tests(calibration_seeds: pd.DataFrame) -> pd.DataFrame:
    """Test convexity directly with paired seed contrasts at repeated coordinates."""
    records: list[dict[str, object]] = []
    for (cell_id, support_id), group in calibration_seeds.groupby(["cell_id", "support_id"], sort=False):
        coordinate_frame = (
            group[["coordinate_id", "phase_0_starcoder", "phase_1_starcoder"]]
            .drop_duplicates()
            .sort_values("coordinate_id")
            .reset_index(drop=True)
        )
        coordinates = coordinate_frame[["phase_0_starcoder", "phase_1_starcoder"]].to_numpy(dtype=float)
        pivot = group.pivot(index="data_seed", columns="coordinate_id", values="bpb")
        for left, middle, right, fraction in collinear_chords(coordinates):
            coordinate_ids = coordinate_frame["coordinate_id"].iloc[[left, middle, right]].astype(str).tolist()
            contrast = pivot[coordinate_ids[1]] - (
                (1.0 - fraction) * pivot[coordinate_ids[0]] + fraction * pivot[coordinate_ids[2]]
            )
            mean = float(contrast.mean())
            standard_deviation = float(contrast.std(ddof=1))
            standard_error = standard_deviation / np.sqrt(len(contrast))
            statistic = mean / standard_error if standard_error > 0.0 else np.inf
            records.append(
                {
                    "cell_id": cell_id,
                    "support_id": support_id,
                    "triple_id": ":".join(coordinate_ids),
                    "left_coordinate_id": coordinate_ids[0],
                    "middle_coordinate_id": coordinate_ids[1],
                    "right_coordinate_id": coordinate_ids[2],
                    "middle_fraction": fraction,
                    "direction": direction_label(coordinates[right] - coordinates[left]),
                    "seed_count": len(contrast),
                    "mean_jensen_gap_bpb": mean,
                    "jensen_gap_sd_bpb": standard_deviation,
                    "paired_t_score": statistic,
                    "paired_one_sided_p": float(t.sf(statistic, len(contrast) - 1)),
                }
            )
    result = pd.DataFrame(records)
    result["cell_holm_p"] = np.nan
    result["cell_holm_rejects_convexity"] = False
    for _, indices in result.groupby(["cell_id", "support_id"], sort=False).groups.items():
        adjusted = holm_adjust(result.loc[indices, "paired_one_sided_p"].to_numpy(dtype=float))
        result.loc[indices, "cell_holm_p"] = adjusted
        result.loc[indices, "cell_holm_rejects_convexity"] = (adjusted < HOLM_ALPHA) & result.loc[
            indices, "mean_jensen_gap_bpb"
        ].gt(0.0)
    global_adjusted = holm_adjust(result["paired_one_sided_p"].to_numpy(dtype=float))
    result["global_holm_p"] = global_adjusted
    result["global_holm_rejects_convexity"] = (global_adjusted < HOLM_ALPHA) & result["mean_jensen_gap_bpb"].gt(0.0)
    return result


def jensen_variance_sensitivity(jensen: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, float | int]] = []
    eligible = jensen.loc[~jensen["contains_alias"]]
    for variance_multiplier in VARIANCE_INFLATION_GRID:
        adjusted_groups: list[pd.DataFrame] = []
        for _, group in eligible.groupby(["cell_id", "support_id"], sort=False):
            inflated_p = t.sf(
                group["paired_t_score"].to_numpy(dtype=float) / np.sqrt(variance_multiplier),
                group["variance_degrees_of_freedom"].to_numpy(dtype=float),
            )
            adjusted = holm_adjust(inflated_p)
            adjusted_groups.append(
                group.assign(
                    sensitivity_holm_p=adjusted,
                    sensitivity_reject=(adjusted < HOLM_ALPHA) & group["jensen_gap_bpb"].gt(0.0),
                )
            )
        result = pd.concat(adjusted_groups, ignore_index=True)
        significant = result.loc[result["sensitivity_reject"]]
        global_p = t.sf(
            result["paired_t_score"].to_numpy(dtype=float) / np.sqrt(variance_multiplier),
            result["variance_degrees_of_freedom"].to_numpy(dtype=float),
        )
        global_adjusted = holm_adjust(global_p)
        global_significant = result.loc[(global_adjusted < HOLM_ALPHA) & result["jensen_gap_bpb"].gt(0.0)]
        records.append(
            {
                "variance_multiplier": variance_multiplier,
                "standard_error_multiplier": float(np.sqrt(variance_multiplier)),
                "holm_rejections": len(significant),
                "rejecting_surfaces": int(significant[["cell_id", "support_id"]].drop_duplicates().shape[0]),
                "triples_rejecting_in_at_least_two_surfaces": int(significant.groupby("triple_id").size().ge(2).sum()),
                "global_holm_rejections": len(global_significant),
                "global_holm_rejecting_surfaces": int(
                    global_significant[["cell_id", "support_id"]].drop_duplicates().shape[0]
                ),
            }
        )
    return pd.DataFrame(records)


def prediction_metrics(
    response: np.ndarray,
    prediction: np.ndarray,
    weights: np.ndarray,
    sigma: np.ndarray,
) -> dict[str, float]:
    residual = prediction - response
    ranked = np.argsort(prediction)
    observed_best = int(np.argmin(response))
    top_five = ranked[: min(5, len(ranked))]
    return {
        "spatial_cv_rmse": float(np.sqrt(np.mean(residual**2))),
        "weighted_spatial_cv_rmse": weighted_rmse(residual, weights),
        "spatial_cv_rmse_in_median_sd": float(np.sqrt(np.mean(residual**2)) / np.median(sigma)),
        "spatial_cv_spearman": float(spearmanr(prediction, response).statistic),
        "pooled_oof_regret_at_1": float(response[ranked[0]] - response[observed_best]),
        "pooled_oof_regret_at_5": float(np.min(response[top_five]) - response[observed_best]),
    }


def build_overview(cell_metrics: pd.DataFrame, jensen: pd.DataFrame, path: Path) -> None:
    convex = cell_metrics.loc[cell_metrics["model"].eq("convex_quadratic")].copy()
    unconstrained = cell_metrics.loc[cell_metrics["model"].eq("unconstrained_quadratic")].copy()
    matched = convex.merge(
        unconstrained,
        on=["cell_id", "support_id", "rung", "materialized_tokens_b"],
        suffixes=("_convex", "_unconstrained"),
        validate="one_to_one",
    )
    matched["rmse_ratio"] = (
        matched["interior_spatial_cv_rmse_convex"] / matched["interior_spatial_cv_rmse_unconstrained"]
    )
    rejects = (
        jensen.groupby(["cell_id", "support_id"], as_index=False)["paired_cell_holm_rejects_convexity"]
        .sum()
        .rename(columns={"paired_cell_holm_rejects_convexity": "holm_rejections"})
    )
    matched = matched.merge(rejects, on=["cell_id", "support_id"], how="left", validate="one_to_one")
    matched["holm_rejections"] = matched["holm_rejections"].fillna(0)

    cells = matched[["cell_id", "rung", "materialized_tokens_b"]].drop_duplicates().sort_values("rung")
    cell_order = cells["cell_id"].tolist()
    cell_labels = [f"D={value:.2f}B" for value in cells["materialized_tokens_b"]]
    ratio = matched.pivot(index="support_id", columns="cell_id", values="rmse_ratio").reindex(
        index=SUPPORT_ORDER, columns=cell_order
    )
    violation = matched.pivot(index="support_id", columns="cell_id", values="holm_rejections").reindex(
        index=SUPPORT_ORDER, columns=cell_order
    )

    figure = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Exploratory pooled-variance midpoint violations",
            "Blocked-CV cost of quadratic convexity",
            "Discovery-seed midpoint screen",
            "Convex predicted gain versus fresh selected-policy gain",
        ),
        horizontal_spacing=0.12,
        vertical_spacing=0.17,
    )
    figure.add_trace(
        go.Heatmap(
            z=violation.to_numpy(),
            x=cell_labels,
            y=[SUPPORT_LABELS[value] for value in SUPPORT_ORDER],
            coloraxis="coloraxis",
            text=violation.to_numpy().astype(int),
            texttemplate="%{text}",
            hovertemplate="%{x}<br>%{y}<br>Holm rejections=%{z:.0f}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    figure.add_trace(
        go.Heatmap(
            z=ratio.to_numpy(),
            x=cell_labels,
            y=[SUPPORT_LABELS[value] for value in SUPPORT_ORDER],
            zmid=1.0,
            coloraxis="coloraxis2",
            text=np.round(ratio.to_numpy(), 3),
            texttemplate="%{text:.3f}",
            hovertemplate="%{x}<br>%{y}<br>Convex / unconstrained RMSE=%{z:.4f}<extra></extra>",
        ),
        row=1,
        col=2,
    )
    support_colors = {
        "full": "#1A9850",
        "m0125": "#66BD63",
        "m025": "#A6D96A",
        "m050": "#FEE08B",
        "m100": "#FDAE61",
        "m200": "#F46D43",
        "m400": "#D73027",
    }
    for support_id in SUPPORT_ORDER:
        subset = jensen.loc[jensen["support_id"].eq(support_id)]
        figure.add_trace(
            go.Scatter(
                x=subset["segment_max_excess_over_best_bpb"],
                y=subset["jensen_gap_bpb"],
                mode="markers",
                name=SUPPORT_LABELS[support_id],
                legendgroup="support",
                marker={
                    "size": 6,
                    "opacity": 0.55,
                    "color": support_colors[support_id],
                    "line": {"width": 0},
                },
                customdata=np.column_stack(
                    (
                        subset["cell_id"],
                        subset["direction"],
                        subset["paired_global_holm_p"],
                    )
                ),
                hovertemplate=(
                    "%{customdata[0]}<br>%{customdata[1]}<br>max excess=%{x:.4f} BPB<br>"
                    "Jensen gap=%{y:+.5f} BPB<br>global Holm p=%{customdata[2]:.4g}<extra></extra>"
                ),
            ),
            row=2,
            col=1,
        )
    figure.add_hline(y=0.0, line={"color": "#17324D", "dash": "dash"}, row=2, col=1)
    figure.add_trace(
        go.Scatter(
            x=convex["fresh_selected_gain_bpb"],
            y=convex["predicted_fresh_selected_pair_gain_bpb"],
            mode="markers",
            marker={
                "size": 10,
                "color": convex["rung"],
                "colorscale": "RdYlGn_r",
                "line": {"color": "#17324D", "width": 0.8},
            },
            customdata=np.column_stack((convex["cell_id"], convex["support_id"])),
            hovertemplate=(
                "%{customdata[0]} / %{customdata[1]}<br>Fresh selected gain=%{x:+.6f}<br>"
                "Predicted selected-pair gain=%{y:+.6f}<extra></extra>"
            ),
            showlegend=False,
        ),
        row=2,
        col=2,
    )
    gain_limit = float(
        max(
            np.abs(convex["fresh_selected_gain_bpb"]).max(),
            np.abs(convex["predicted_fresh_selected_pair_gain_bpb"]).max(),
        )
    )
    figure.add_trace(
        go.Scatter(
            x=[-gain_limit, gain_limit],
            y=[-gain_limit, gain_limit],
            mode="lines",
            line={"color": "#17324D", "dash": "dash"},
            showlegend=False,
        ),
        row=2,
        col=2,
    )
    figure.update_layout(
        title={
            "text": (
                "Can the dense StarCoder WSD80 interiors support a convex surrogate?<br>"
                "<sup>Aligned-seed shape tests and matched-model spatial CV across 28 cells</sup>"
            ),
            "x": 0.5,
        },
        template="plotly_white",
        paper_bgcolor="#F7F3E8",
        plot_bgcolor="#FFFDF8",
        font={"family": "Avenir Next, sans-serif", "color": "#17324D"},
        height=980,
        width=1380,
        coloraxis={"colorscale": "RdYlGn_r", "colorbar": {"title": "violations", "x": 0.46, "y": 0.79, "len": 0.34}},
        coloraxis2={
            "colorscale": "RdYlGn_r",
            "cmid": 1.0,
            "colorbar": {"title": "RMSE ratio", "x": 1.01, "y": 0.79, "len": 0.34},
        },
        margin={"l": 95, "r": 120, "t": 125, "b": 80},
    )
    figure.update_xaxes(title_text="Largest segment loss above observed best (BPB)", row=2, col=1)
    figure.update_yaxes(title_text="Midpoint Jensen gap (BPB)", row=2, col=1)
    figure.update_xaxes(title_text="Fresh selected-policy gain (BPB)", row=2, col=2)
    figure.update_yaxes(title_text="Predicted gain at the same selected pair (BPB)", row=2, col=2)
    figure.write_html(path, include_plotlyjs=True, full_html=True, config={"displaylogo": False})


def selected_policy_gate(rows: pd.DataFrame) -> bool:
    if rows.empty:
        return False
    gain_compatible = rows["predicted_fresh_selected_pair_gain_bpb"].between(
        rows["fresh_selected_ci95_low"],
        rows["fresh_selected_ci95_high"],
    )
    return bool(rows["optimum_distance_to_fresh_selected_l2"].max() <= 0.05 and gain_compatible.all())


def write_report(
    cell_metrics: pd.DataFrame,
    jensen: pd.DataFrame,
    aligned_chords: pd.DataFrame,
    sensitivity: pd.DataFrame,
    influence: pd.DataFrame,
    path: Path,
) -> None:
    pivot = cell_metrics.pivot_table(
        index=["cell_id", "support_id"], columns="model", values="interior_spatial_cv_rmse"
    ).reset_index()
    pivot["quadratic_convex_to_unconstrained_ratio"] = pivot["convex_quadratic"] / pivot["unconstrained_quadratic"]
    pivot["spline_convex_to_unconstrained_ratio"] = pivot["convex_ridge_spline"] / pivot["unconstrained_ridge_spline"]
    ratios = pivot["quadratic_convex_to_unconstrained_ratio"]
    spline_ratios = pivot["spline_convex_to_unconstrained_ratio"]
    pivot["conditional_convex_to_unconstrained_ratio"] = (
        pivot["conditional_convex_cubic"] / pivot["conditional_unconstrained_cubic"]
    )
    conditional_ratios = pivot["conditional_convex_to_unconstrained_ratio"]
    cell_significant = jensen.loc[jensen["paired_cell_holm_rejects_convexity"]]
    global_significant = jensen.loc[jensen["paired_global_holm_rejects_convexity"]]
    direct_cell_significant = aligned_chords.loc[aligned_chords["cell_holm_rejects_convexity"]]
    direct_global_significant = aligned_chords.loc[aligned_chords["global_holm_rejects_convexity"]]
    predictive_gate = bool(ratios.median() <= 1.05 and ratios.quantile(0.9) <= 1.10)
    global_shape_gate = len(direct_global_significant) == 0
    convex_quadratic = cell_metrics.loc[cell_metrics["model"].eq("convex_quadratic")]
    confirmed_quadratic = convex_quadratic.loc[convex_quadratic["fresh_selected_holm_positive"].astype(bool)]
    confirmed_gain_error = (
        confirmed_quadratic["predicted_fresh_selected_pair_gain_bpb"] - confirmed_quadratic["fresh_selected_gain_bpb"]
    ).abs()
    selection_gate = selected_policy_gate(confirmed_quadratic)
    unconstrained_quadratic = cell_metrics.loc[cell_metrics["model"].eq("unconstrained_quadratic")]
    confirmed_unconstrained_quadratic = unconstrained_quadratic.loc[
        unconstrained_quadratic["fresh_selected_holm_positive"].astype(bool)
    ]
    unconstrained_selection_gate = selected_policy_gate(confirmed_unconstrained_quadratic)
    conditional = cell_metrics.loc[cell_metrics["model"].eq("conditional_convex_cubic")]
    confirmed_conditional = conditional.loc[conditional["fresh_selected_holm_positive"].astype(bool)]
    conditional_gain_error = (
        confirmed_conditional["predicted_fresh_selected_pair_gain_bpb"]
        - confirmed_conditional["fresh_selected_gain_bpb"]
    ).abs()
    conditional_selection_gate = selected_policy_gate(confirmed_conditional)
    conditional_unconstrained = cell_metrics.loc[cell_metrics["model"].eq("conditional_unconstrained_cubic")]
    confirmed_conditional_unconstrained = conditional_unconstrained.loc[
        conditional_unconstrained["fresh_selected_holm_positive"].astype(bool)
    ]
    conditional_unconstrained_selection_gate = selected_policy_gate(confirmed_conditional_unconstrained)
    convex_regression = cell_metrics.loc[cell_metrics["model"].eq("nonparametric_convex_regression")]
    confirmed_convex_regression = convex_regression.loc[convex_regression["fresh_selected_holm_positive"].astype(bool)]
    convex_regression_selection_gate = selected_policy_gate(confirmed_convex_regression)
    quadratic_rmse_sd = convex_quadratic["interior_spatial_cv_rmse_in_median_sd"]
    conditional_rmse_sd = conditional["interior_spatial_cv_rmse_in_median_sd"]
    convex_regression_rmse_sd = convex_regression["interior_spatial_cv_rmse_in_median_sd"]

    def alpha_spans(rows: pd.DataFrame) -> pd.Series:
        parsed = rows["outer_selected_alphas"].map(json.loads)
        return parsed.map(lambda values: max(values) / min(values))

    conditional_alpha_spans = alpha_spans(conditional)
    convex_regression_alpha_spans = alpha_spans(convex_regression)

    if len(direct_global_significant):
        shape_statement = "Direct aligned-seed chords reject global raw-BPB convexity"
    elif len(direct_cell_significant):
        shape_statement = (
            "Direct aligned-seed chords find localized within-cell nonconvexity, but not a familywise global rejection"
        )
    else:
        shape_statement = "Direct aligned-seed chords do not reject raw-BPB convexity"
    parity_statement = "passes" if predictive_gate else "fails"
    selection_statement = "passes" if selection_gate else "fails"
    conditional_statement = "passes" if conditional_selection_gate else "fails"
    verdict = (
        f"{shape_statement}. The exact PSD quadratic {parity_statement} the blocked-CV relative-parity screen and "
        f"{selection_statement} the fresh-seed selected-policy gate. The aggregate-conditioned convex model "
        f"{conditional_statement} that same gate, and nonparametric convex regression also fails it. The present "
        "evidence therefore does not support a deployable globally convex raw-BPB surrogate. It does leave open a "
        "narrower route: retain a flexible aggregate response and make only the phase-control subproblem convex "
        "conditional on the aggregate and frozen training state."
    )

    aggregate = cell_metrics.groupby("model", as_index=False).agg(
        median_full_spatial_cv_rmse=("spatial_cv_rmse", "median"),
        median_interior_spatial_cv_rmse=("interior_spatial_cv_rmse", "median"),
        p90_interior_spatial_cv_rmse=("interior_spatial_cv_rmse", lambda values: values.quantile(0.9)),
        median_interior_fit_rmse=("interior_fit_rmse", "median"),
        median_pooled_oof_regret_at_1=("pooled_oof_regret_at_1", "median"),
        median_pooled_oof_regret_at_5=("pooled_oof_regret_at_5", "median"),
        median_predicted_global_gain=("predicted_global_two_phase_gain_bpb", "median"),
    )
    aggregate["model"] = aggregate["model"].map(MODEL_LABELS)
    top_violations = (
        jensen.loc[~jensen["contains_alias"]]
        .sort_values("paired_t_score", ascending=False)
        .head(12)[
            [
                "cell_id",
                "support_id",
                "direction",
                "left_coordinate_id",
                "middle_coordinate_id",
                "right_coordinate_id",
                "segment_max_excess_over_best_bpb",
                "jensen_gap_bpb",
                "paired_t_score",
                "paired_global_holm_p",
                "paired_global_holm_rejects_convexity",
            ]
        ]
    )
    direct_top = aligned_chords.sort_values("paired_t_score", ascending=False).head(12)[
        [
            "cell_id",
            "support_id",
            "triple_id",
            "direction",
            "middle_fraction",
            "mean_jensen_gap_bpb",
            "paired_t_score",
            "cell_holm_p",
            "global_holm_p",
        ]
    ]
    confirmed = cell_metrics.loc[
        cell_metrics["fresh_selected_holm_positive"].astype(bool)
        & cell_metrics["model"].isin(
            (
                "convex_quadratic",
                "unconstrained_quadratic",
                "conditional_convex_cubic",
                "conditional_unconstrained_cubic",
                "convex_ridge_spline",
                "nonparametric_convex_regression",
            )
        ),
        [
            "cell_id",
            "support_id",
            "model",
            "fresh_selected_gain_bpb",
            "fresh_selected_ci95_low",
            "fresh_selected_ci95_high",
            "predicted_global_two_phase_gain_bpb",
            "predicted_fresh_selected_pair_gain_bpb",
            "predicted_untied_p0",
            "predicted_untied_p1",
            "optimum_distance_to_fresh_selected_l2",
            "pooled_oof_regret_at_1",
            "pooled_oof_regret_at_5",
        ],
    ]

    contrast_count = int(jensen.groupby(["cell_id", "support_id"]).size().median())
    rejecting_surface_count = global_significant[["cell_id", "support_id"]].drop_duplicates().shape[0]
    surface_count = jensen[["cell_id", "support_id"]].drop_duplicates().shape[0]
    four_replay_rejections = int(global_significant["support_id"].eq("m400").sum())
    two_replay_rejections = int(global_significant["support_id"].eq("m200").sum())
    lower_replay_rejections = len(global_significant) - four_replay_rejections - two_replay_rejections
    sensitivity_17 = sensitivity.loc[sensitivity["variance_multiplier"].eq(17.7)].iloc[0]
    longest_horizon = jensen.loc[jensen["cell_id"].str.startswith("r3_")]
    longest_horizon_rejections = int(longest_horizon["paired_global_holm_rejects_convexity"].sum())
    fiber = jensen.loc[(~jensen["contains_alias"]) & jensen["direction"].eq("fixed_aggregate_fiber")]
    fiber_nominal_rejections = int(((fiber["paired_one_sided_p"] < HOLM_ALPHA) & fiber["jensen_gap_bpb"].gt(0.0)).sum())
    fiber_global_rejections = int(fiber["paired_global_holm_rejects_convexity"].sum())
    influence_baseline = influence.loc[influence["removed_coordinate_id"].eq("none")]
    influential = pd.concat(
        (
            influence_baseline,
            influence.loc[~influence["removed_coordinate_id"].eq("none")]
            .sort_values(["global_holm_rejections", "rejecting_surfaces", "removed_coordinate_id"])
            .head(11),
        ),
        ignore_index=True,
    )
    direct_fiber = aligned_chords.loc[aligned_chords["direction"].eq("fixed_aggregate_fiber")]
    direct_fiber_positive = int(direct_fiber["mean_jensen_gap_bpb"].gt(0.0).sum())
    direct_fiber_geometries = direct_fiber["triple_id"].nunique()
    confirmed_table = (
        confirmed.to_markdown(index=False, floatfmt=".6f")
        if len(confirmed)
        else "No fresh Holm-positive cell was present."
    )
    report = (
        "# Convex-surrogate viability audit for dense StarCoder WSD80\n\n"
        "## Verdict\n\n"
        f"{verdict}\n\n"
        "This is a development-data shape audit. It does not establish that any particular convex mechanism "
        "is correct.\n\n"
        "## Interpretation rules\n\n"
        "- Model fits and optimization use all 125 coordinates per cell, including boundaries. Shape diagnostics "
        f"using single discovery observations are restricted to `{INTERIOR_MARGIN} < p0,p1 < "
        f"{1.0 - INTERIOR_MARGIN}`.\n"
        "- Primary shape evidence uses exact collinear chords for which all three coordinates have four aligned "
        "seeds. The chord contrast is formed within each seed, so the common seed offset cancels without a "
        "homoskedasticity assumption. Both within-cell and all-28-cell Holm adjustments are reported.\n"
        "- The larger discovery-seed midpoint screen uses a pooled idiosyncratic variance component. Because "
        "variance rises sharply with aggregate and calibration ends at aggregate 0.70, its high-aggregate p-values "
        "are exploratory rather than load-bearing.\n"
        "- Predictive parity: the convex quadratic must have median blocked-CV RMSE at most `1.05x` and "
        "90th-percentile ratio at most `1.10x` the identical unconstrained quadratic basis.\n"
        "- Fresh-seed selection: the predicted optimum must be within `0.05` L2 of each confirmed selected "
        "coordinate and its predicted selected-pair gain must fall inside that pair's fresh 95% CI.\n"
        "- No outcome-selected neighborhood is used to claim local convexity. The fresh-selected coordinates are "
        "discrete discovery minima confirmed with new seeds, not known continuous optima. Their repeated outcomes "
        "are fresh, but this audit and its thresholds are post-hoc development analyses, not preregistered "
        "confirmation.\n\n"
        "## Headline numbers\n\n"
        f"- Direct aligned-seed chord tests: `{len(aligned_chords)}`. Within-cell Holm retains "
        f"`{len(direct_cell_significant)}` violations; one Holm family over all cells retains "
        f"`{len(direct_global_significant)}`.\n"
        f"- Exploratory discovery-seed midpoint contrasts per cell: `{contrast_count}`. Pooled-variance per-cell "
        f"Holm retains `{len(cell_significant)}`; global Holm retains `{len(global_significant)}` in "
        f"`{rejecting_surface_count}` of `{surface_count}` cells.\n"
        f"- Repetition concentration: `{four_replay_rejections}` at 4x and `{two_replay_rejections}` at 2x; "
        f"`{lower_replay_rejections}` at lower replay.\n"
        f"- At the longest r3 horizon, which contains all three fresh-confirmed two-phase gains, global Holm "
        f"violations: `{longest_horizon_rejections}`. This is absence of detected violations, not evidence of "
        "local convexity.\n"
        "- Pooled variance-component sensitivity: at `17.7x` variance, "
        f"`{int(sensitivity_17['holm_rejections'])}` within-cell Holm violations remain in "
        f"`{int(sensitivity_17['rejecting_surfaces'])}` surfaces, with "
        f"`{int(sensitivity_17['triples_rejecting_in_at_least_two_surfaces'])}` repeated triple. Global Holm retains "
        f"`{int(sensitivity_17['global_holm_rejections'])}` violations.\n"
        "- Matched quadratic convex/unconstrained interior blocked-CV RMSE ratio: "
        f"median `{ratios.median():.4f}`, p90 `{ratios.quantile(0.9):.4f}`, maximum `{ratios.max():.4f}`.\n"
        "- Absolute exact-PSD quadratic interior blocked-CV RMSE: median "
        f"`{quadratic_rmse_sd.median():.2f}` modeled coordinate SDs, p90 "
        f"`{quadratic_rmse_sd.quantile(0.9):.2f}`. Relative parity is therefore not absolute adequacy.\n"
        "- Flexible spline ratio, reported only as a stability warning: "
        f"median `{spline_ratios.median():.4f}`, p90 `{spline_ratios.quantile(0.9):.4f}`, "
        f"maximum `{spline_ratios.max():.4f}`.\n"
        "- Aggregate-conditioned convex/unconstrained interior blocked-CV RMSE ratio: "
        f"median `{conditional_ratios.median():.4f}`, p90 `{conditional_ratios.quantile(0.9):.4f}`, "
        f"maximum `{conditional_ratios.max():.4f}`.\n"
        "- Absolute conditional-convex interior blocked-CV RMSE: median "
        f"`{conditional_rmse_sd.median():.2f}` modeled coordinate SDs, p90 "
        f"`{conditional_rmse_sd.quantile(0.9):.2f}`.\n"
        "- Nonparametric convex regression, an optimistic shape-class diagnostic rather than an admissible "
        "surrogate: median interior blocked-CV RMSE "
        f"`{convex_regression['interior_spatial_cv_rmse'].median():.6f}` BPB, or "
        f"`{convex_regression_rmse_sd.median():.2f}` modeled coordinate SDs.\n"
        "- Ridge selection is spatially unstable: conditional-convex outer folds select different penalties in "
        f"`{int(conditional_alpha_spans.gt(1.0).sum())}` of `{len(conditional_alpha_spans)}` cells with median "
        f"max/min span `{conditional_alpha_spans.median():.0f}x`; nonparametric convex regression does so in "
        f"`{int(convex_regression_alpha_spans.gt(1.0).sum())}` of `{len(convex_regression_alpha_spans)}` cells "
        f"with median span `{convex_regression_alpha_spans.median():.0f}x`.\n"
        f"- Fixed-aggregate midpoint contrasts: `{len(fiber)}`; nominal one-sided rejections "
        f"`{fiber_nominal_rejections}`; primary global-Holm rejections `{fiber_global_rejections}`.\n"
        f"- Direct repeated-seed fixed-aggregate chords: `{len(direct_fiber)}` across "
        f"`{direct_fiber_geometries}` coordinate geometry; positive mean gaps `{direct_fiber_positive}`. This "
        "coverage is too narrow to certify conditional convexity.\n"
        "- Convex-quadratic fresh-positive alignment: maximum optimum distance "
        f"`{confirmed_quadratic['optimum_distance_to_fresh_selected_l2'].max():.4f}`; maximum gain error "
        f"`{confirmed_gain_error.max():.6f}` BPB.\n"
        "- Aggregate-conditioned convex fresh-positive alignment: maximum optimum distance "
        f"`{confirmed_conditional['optimum_distance_to_fresh_selected_l2'].max():.4f}`; maximum gain error "
        f"`{conditional_gain_error.max():.6f}` BPB.\n"
        "- Pooled OOF regret is reported descriptively only because each fold uses a different fitted model.\n"
        f"- Familywise direct shape rejection: `{'yes' if not global_shape_gate else 'no'}`.\n"
        f"- Relative predictive-parity screen: `{'pass' if predictive_gate else 'fail'}`; absolute adequacy fails.\n"
        f"- PSD / unconstrained optimum-selection gates: `{'pass' if selection_gate else 'fail'}` / "
        f"`{'pass' if unconstrained_selection_gate else 'fail'}`.\n"
        f"- Conditional-convex / matched-unconstrained selection gates: "
        f"`{'pass' if conditional_selection_gate else 'fail'}` / "
        f"`{'pass' if conditional_unconstrained_selection_gate else 'fail'}`.\n"
        "- Nonparametric convex-regression selection gate: "
        f"`{'pass' if convex_regression_selection_gate else 'fail'}`.\n\n"
        "## Matched-model summary\n\n"
        f"{aggregate.to_markdown(index=False, floatfmt='.6f')}\n\n"
        "## Direct aligned-seed chord tests\n\n"
        "These tests do not extrapolate a variance model. Positive gaps violate convexity. The two strongest "
        "r0/m400 chords survive correction within that cell but not across all 180 chords.\n\n"
        f"{direct_top.to_markdown(index=False, floatfmt='.6f')}\n\n"
        "## Pooled Jensen variance sensitivity\n\n"
        "This is sensitivity for the larger exploratory midpoint screen. It scales the paired variance component "
        "over the same 1,448 non-alias contrasts used by that screen.\n\n"
        f"{sensitivity.to_markdown(index=False, floatfmt='.4f')}\n\n"
        "## Coordinate influence\n\n"
        "The violations are not broadly distributed. Each row removes every contrast touching one implicated "
        "coordinate and recomputes the single global Holm family.\n\n"
        f"{influential.to_markdown(index=False)}\n\n"
        "## Fresh-confirmed positive cells\n\n"
        f"{confirmed_table}\n\n"
        "## Largest model-free Jensen gaps\n\n"
        "Positive gaps violate convexity before uncertainty adjustment.\n\n"
        f"{top_violations.to_markdown(index=False, floatfmt='.6f')}\n\n"
        "## Scope\n\n"
        "The tested conditional model uses cubic Bernstein functions for `A(a)`, `B(a)`, and `C(a)`: "
        "`L(a, delta) = A(a) + B(a) delta + 1/2 C(a) delta^2`, with "
        "`a = 0.8 p0 + 0.2 p1`, `delta = p0 - p1`, and nonnegative Bernstein coefficients for `C`. "
        "It has 12 coefficients per surface; blocked CV selects one shared ridge penalty on the eight ordering "
        "and curvature coefficients. Its unconstrained match removes only `C(a) >= 0`. Their shared fresh-optimum "
        "failures show that this low-order state representation, rather than convexity alone, is inadequate.\n\n"
        "The nonparametric convex-regression diagnostic assigns one fitted value and one two-dimensional "
        "subgradient to every training coordinate, constrained by all pairwise supporting-hyperplane inequalities. "
        "Its max-affine predictor is globally convex, but sample-dependent and therefore not an admissible "
        "mechanistic surrogate. It distinguishes algebraic compatibility from spatial generalization: the "
        "in-sample fit can be much closer than nested blocked CV, yet fresh-seed optimum selection still fails.\n\n"
        "The constrained fits are probes, not promoted surrogates. A mechanistically plausible convexifiable "
        "phase-control model at fixed aggregate `a` is\n\n"
        "`V_a(delta) = c(a) - b(a)^T delta + 1/2 delta^T H(a) delta "
        "+ sum[p,i] lambda[p,i] [e[p,i] - tau[p,i]]_+^q[p,i]`,\n\n"
        "with `H(a)` PSD, `lambda >= 0`, and `q >= 1`. The linear term represents phase-specific target-gradient "
        "alignment; the PSD and hinge terms represent interference and repetition/overload costs. For fixed `a` "
        "and frozen state-dependent coefficients, optimizing `delta` over mixture feasibility constraints is convex "
        "when every exposure `e[p,i]` is affine in `delta`. Joint optimization over `a` and `delta` additionally "
        "requires a jointly convex response, not merely PSD curvature at one fixed aggregate; otherwise the "
        "defensible route is aggregate search followed by a conditional convex phase solve, or sequential convex "
        "trust regions.\n"
        "\nRaw-BPB convexity is stronger than optimization requires. If `L = g(V)` for a fixed strictly increasing link "
        "`g`, minimizing BPB is equivalent to minimizing the latent score `V`, whose convexity can differ from "
        "that of `L`. This audit did not freeze and test such links. An arbitrary learned monotone transform would "
        "be an inadmissible calibration layer; a mechanistically predeclared link such as log reducible BPB remains "
        "a separate convexification hypothesis. A monotone link preserves ordering and therefore cannot repair a "
        "latent model that ranks policies incorrectly.\n"
        "\nAffine reparameterization does not alter this verdict. Within one horizon-support cell, phase epochs are "
        "fixed positive multiples of `(p0, p1)`, and `(a, delta)` is an invertible affine transformation. Global "
        "convexity is invariant under either change of coordinates. A nonlinear retained state can change the "
        "function class, but transitions such as late-mixture-dependent forgetting are generally bilinear and break "
        "joint convexity; freezing the incoming state restores a conditional convex subproblem.\n"
    )
    path.write_text(report, encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    coverage = pd.read_csv(args.coverage)
    evidence = pd.read_csv(args.evidence).set_index(["cell_id", "support_id"])
    calibration_seeds = pd.read_csv(args.calibration_seeds)
    required = {
        "cell_id",
        "support_id",
        "coordinate_id",
        "phase_0_starcoder",
        "phase_1_starcoder",
        "bpb",
        "predicted_sd_bpb",
        "surface_weight",
        "rung",
        "materialized_tokens",
    }
    missing = required - set(coverage.columns)
    if missing:
        raise ValueError(f"Coverage table is missing columns: {sorted(missing)}")
    if coverage[["cell_id", "support_id", "coordinate_id"]].duplicated().any():
        raise ValueError("Coverage coordinates must be unique within each horizon-support cell")
    if coverage[["cell_id", "support_id", "phase_0_starcoder", "phase_1_starcoder"]].duplicated().any():
        raise ValueError("Coverage contains duplicate coordinate pairs within a horizon-support cell")

    shape_coverage = coverage.loc[interior_rows(coverage)].copy()
    grouped = list(coverage.groupby(["cell_id", "support_id"], sort=False))
    shape_grouped = {
        key: group.sort_values("coordinate_id").reset_index(drop=True)
        for key, group in shape_coverage.groupby(["cell_id", "support_id"], sort=False)
    }
    if len(grouped) != 28:
        raise ValueError(f"Expected 28 horizon-support cells, found {len(grouped)}")

    first = grouped[0][1].sort_values("coordinate_id").reset_index(drop=True)
    coordinates = first[["phase_0_starcoder", "phase_1_starcoder"]].to_numpy(dtype=float)
    coordinate_ids = first["coordinate_id"].astype(str).tolist()
    for key, group in grouped[1:]:
        ordered = group.sort_values("coordinate_id").reset_index(drop=True)
        if ordered["coordinate_id"].astype(str).tolist() != coordinate_ids:
            raise ValueError(f"Model-fit coordinate IDs differ in cell {key}")
        candidate = ordered[["phase_0_starcoder", "phase_1_starcoder"]].to_numpy(dtype=float)
        if not np.allclose(candidate, coordinates, atol=1e-12, rtol=0.0):
            raise ValueError(f"Model-fit coordinates differ in cell {key}")

    bases = make_bases(coordinates)
    first_shape = shape_grouped[next(iter(shape_grouped))]
    shape_coordinates = first_shape[["phase_0_starcoder", "phase_1_starcoder"]].to_numpy(dtype=float)
    shape_coordinate_ids = first_shape["coordinate_id"].astype(str).tolist()
    for key, group in shape_grouped.items():
        if group["coordinate_id"].astype(str).tolist() != shape_coordinate_ids:
            raise ValueError(f"Interior coordinate IDs differ in cell {key}")
        candidate = group[["phase_0_starcoder", "phase_1_starcoder"]].to_numpy(dtype=float)
        if not np.allclose(candidate, shape_coordinates, atol=1e-12, rtol=0.0):
            raise ValueError(f"Interior coordinates differ in cell {key}")
    triples = midpoint_triples(shape_coordinates)
    if not triples:
        raise ValueError("The interior design has no exact midpoint triples")

    metric_rows: list[dict[str, object]] = []
    prediction_rows: list[pd.DataFrame] = []
    jensen_records: list[dict[str, object]] = []
    selection_rows: list[dict[str, object]] = []

    for (cell_id, support_id), raw_group in grouped:
        group = raw_group.sort_values("coordinate_id").reset_index(drop=True)
        response = group["bpb"].to_numpy(dtype=float)
        sigma = group["predicted_sd_bpb"].to_numpy(dtype=float)
        weights = group["surface_weight"].to_numpy(dtype=float)
        cell_evidence = evidence.loc[(cell_id, support_id)]
        jensen_records.extend(jensen_rows(shape_grouped[(cell_id, support_id)], shape_coordinates, triples))

        for model_name in MODEL_ORDER:
            if model_name in {"convex_quadratic", "unconstrained_quadratic"}:
                predictions, full_model = quadratic_spatial_predictions(
                    coordinates,
                    response,
                    weights,
                    convex=model_name == "convex_quadratic",
                )
                fold_alphas: tuple[float, ...] = ()
                full_alpha = np.nan
                alpha_scores: tuple[tuple[float, float], ...] = ()
            elif model_name == "nonparametric_convex_regression":
                predictions, fold_alphas, full_alpha, alpha_scores, full_model = convex_regression_spatial_predictions(
                    coordinates, response, weights
                )
            else:
                basis = bases[model_name]
                predictions, fold_alphas, full_alpha, alpha_scores = nested_predictions(
                    coordinates,
                    response,
                    weights,
                    basis,
                )
                full_model = fit_surface(coordinates, response, weights, basis, full_alpha)
            row: dict[str, object] = {
                "cell_id": cell_id,
                "support_id": support_id,
                "rung": int(group["rung"].iloc[0]),
                "materialized_tokens_b": float(group["materialized_tokens"].iloc[0] / 1e9),
                "model": model_name,
                "full_selected_alpha": full_alpha,
                "outer_selected_alphas": json.dumps(fold_alphas),
                "full_fit_solver": full_model.solver if isinstance(full_model, ConvexRegressionModel) else "",
                "fresh_selected_gain_bpb": float(cell_evidence["fresh_selected_gain_bpb"]),
                "fresh_selected_ci95_low": float(cell_evidence["fresh_selected_ci95_low"]),
                "fresh_selected_ci95_high": float(cell_evidence["fresh_selected_ci95_high"]),
                "fresh_selected_holm_p": float(cell_evidence["fresh_selected_holm_p"]),
                "fresh_selected_holm_positive": bool(cell_evidence["fresh_selected_holm_positive"]),
            }
            row.update(prediction_metrics(response, predictions, weights, sigma))
            interior = interior_rows(group).to_numpy(dtype=bool)
            row.update(
                {
                    f"interior_{name}": value
                    for name, value in prediction_metrics(
                        response[interior],
                        predictions[interior],
                        weights[interior],
                        sigma[interior],
                    ).items()
                }
            )
            fitted = full_model.predict(coordinates)
            row["fit_rmse"] = float(np.sqrt(np.mean((fitted - response) ** 2)))
            row["interior_fit_rmse"] = float(np.sqrt(np.mean((fitted[interior] - response[interior]) ** 2)))
            row.update(optimum_diagnostics(full_model, coordinates, response, cell_evidence))
            metric_rows.append(row)
            prediction_rows.append(
                pd.DataFrame(
                    {
                        "cell_id": cell_id,
                        "support_id": support_id,
                        "model": model_name,
                        "coordinate_id": group["coordinate_id"],
                        "phase_0_starcoder": coordinates[:, 0],
                        "phase_1_starcoder": coordinates[:, 1],
                        "observed_bpb": response,
                        "predicted_bpb": predictions,
                        "predicted_sd_bpb": sigma,
                    }
                )
            )
            for alpha, score in alpha_scores:
                selection_rows.append(
                    {
                        "cell_id": cell_id,
                        "support_id": support_id,
                        "model": model_name,
                        "alpha": alpha,
                        "blocked_cv_weighted_rmse": score,
                        "selected_for_full_fit": alpha == full_alpha,
                    }
                )

    cell_metrics = pd.DataFrame(metric_rows)
    predictions = pd.concat(prediction_rows, ignore_index=True)
    jensen = pd.DataFrame(jensen_records)
    jensen["global_holm_p"] = holm_adjust(jensen["one_sided_p"].to_numpy(dtype=float))
    jensen["global_holm_rejects_convexity"] = jensen["global_holm_p"].lt(HOLM_ALPHA) & jensen["jensen_gap_bpb"].gt(0.0)
    variance_components = calibration_variance_components(calibration_seeds)
    jensen = apply_variance_component_inference(jensen, variance_components)
    aligned_chords = aligned_seed_chord_tests(calibration_seeds)
    influence = coordinate_influence(jensen)
    selection = pd.DataFrame(selection_rows)
    sensitivity = jensen_variance_sensitivity(jensen)

    cell_metrics.to_csv(args.output_dir / "cell_metrics.csv", index=False)
    predictions.to_csv(args.output_dir / "nested_spatial_oof_predictions.csv", index=False)
    jensen.to_csv(args.output_dir / "exact_midpoint_jensen_contrasts.csv", index=False)
    variance_components.to_csv(args.output_dir / "calibration_variance_components.csv", index=False)
    aligned_chords.to_csv(args.output_dir / "aligned_seed_chord_tests.csv", index=False)
    influence.to_csv(args.output_dir / "jensen_coordinate_influence.csv", index=False)
    sensitivity.to_csv(args.output_dir / "jensen_variance_sensitivity.csv", index=False)
    selection.to_csv(args.output_dir / "ridge_selection.csv", index=False)

    protocol = {
        "protocol": "wsd80-convex-surrogate-viability-v3",
        "sources": {
            str(args.coverage): file_sha256(args.coverage),
            str(args.evidence): file_sha256(args.evidence),
            str(args.calibration_seeds): file_sha256(args.calibration_seeds),
            str(Path(__file__).resolve()): file_sha256(Path(__file__).resolve()),
        },
        "interior_margin": INTERIOR_MARGIN,
        "model_fit_domain": "all 125 coordinates per cell",
        "discovery_midpoint_domain": "84 interior coordinates per cell",
        "outer_folds": OUTER_FOLDS,
        "inner_folds": INNER_FOLDS,
        "outer_seed": OUTER_SEED,
        "inner_seed": INNER_SEED,
        "ridge_grid": RIDGE_GRID,
        "convex_regression_ridge_grid": CONVEX_REGRESSION_RIDGE_GRID,
        "spline_directions": SPLINE_DIRECTIONS,
        "spline_knot_quantiles": SPLINE_KNOT_QUANTILES,
        "holm_alpha": HOLM_ALPHA,
        "variance_inflation_grid": VARIANCE_INFLATION_GRID,
        "interpretation": {
            "primary_shape_evidence": (
                "direct four-aligned-seed collinear chord contrasts; report within-cell and all-cell Holm families"
            ),
            "exploratory_shape_screen": (
                "pooled variance-component midpoint tests on discovery observations; heteroskedasticity-extrapolative"
            ),
            "relative_predictive_parity": (
                "median convex/unconstrained quadratic blocked-CV RMSE ratio <=1.05 and p90 <=1.10; "
                "not an absolute adequacy gate"
            ),
            "optimum_selection": (
                "all fresh-positive optimum distances <=0.05 and predicted gains inside fresh paired 95% CIs"
            ),
            "local_convexity": "not inferred; no outcome-selected neighborhood is used",
            "response_link_scope": "raw BPB only; no monotone latent-response link is fit or tested",
        },
    }
    (args.output_dir / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
    build_overview(cell_metrics, jensen, args.output_dir / "convex_surrogate_viability.html")
    write_report(cell_metrics, jensen, aligned_chords, sensitivity, influence, args.output_dir / "report.md")
    print(f"Wrote convex-surrogate viability audit to {args.output_dir}")


if __name__ == "__main__":
    main()
