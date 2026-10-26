# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pandas", "scipy", "scikit-learn", "threadpoolctl"]
# ///
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Fit MARINER heads with profiled floors and a linearized BPB objective.

For a fixed floor phi = rho * min(y), weights w = y - phi, d = mean(w),
and physical feature spans R, minimize

    sum((w * (log(w) - c - X beta))**2) + ridge * d**2 * ||R beta||**2

over a free intercept and nonnegative coefficients. This convex conditional
problem approximates response-space squared error; it is not nonlinear BPB
least squares. Floor and hyperparameter selection belong to the caller's
inner validation. No bound on the native Head's prediction floor is added.
"""

import math

import benchmark_two_phase_20260921 as benchmark
import numpy as np
from scipy.optimize import nnls

mariner = benchmark.mariner


def fit_profile_head(
    matrix: np.ndarray,
    response: np.ndarray,
    ridge: float,
    rho: float,
    feature_spans: np.ndarray,
) -> mariner.Head:
    """Fit a floor-profile candidate using only the supplied training rows.

    Args:
        matrix: Signed benefit/harm features, with no intercept column.
        response: Positive observed BPB values for these training rows.
        ridge: Nonnegative coefficient-penalty multiplier.
        rho: Finite fraction of the training minimum; must be below one.
        feature_spans: Nonnegative physical feature ranges, fixed independently
            of outcomes. A zero span requires a constant feature column.
    """
    matrix = np.asarray(matrix, dtype=float)
    response = np.asarray(response, dtype=float)
    feature_spans = np.asarray(feature_spans, dtype=float)
    if matrix.ndim != 2 or response.ndim != 1 or matrix.shape[0] != len(response) or not len(response):
        raise ValueError("matrix and response must have the same nonzero number of rows")
    if feature_spans.shape != (matrix.shape[1],):
        raise ValueError("physical feature spans must match the feature width")
    if not np.isfinite(matrix).all() or not np.isfinite(response).all() or np.any(response <= 0):
        raise ValueError("features must be finite and observed BPB must be finite and positive")
    if not np.isfinite(feature_spans).all() or np.any(feature_spans < 0):
        raise ValueError("physical feature spans must be finite and nonnegative")
    if not math.isfinite(ridge) or ridge < 0 or not math.isfinite(rho) or rho >= 1:
        raise ValueError("ridge must be finite and nonnegative, and rho must be finite and below one")
    zero_span = feature_spans == 0
    if np.any(matrix[:, zero_span] != matrix[0, zero_span]):
        raise ValueError("a zero physical span requires a constant feature column")

    floor = float(rho * np.min(response))
    weights = response - floor
    target = np.log(weights)
    squared_weights = weights**2
    if not np.isfinite(squared_weights).all() or not np.isfinite(target).all():
        raise FloatingPointError("floor profile produces nonfinite weights or log deficits")
    mean_weights = squared_weights / np.sum(squared_weights)
    design_mean = mean_weights @ matrix
    target_mean = float(mean_weights @ target)
    rows = weights[:, None] * (matrix - design_mean)
    rhs = weights * (target - target_mean)
    if ridge > 0:
        penalty = math.sqrt(ridge) * float(np.mean(weights)) * feature_spans
        rows = np.vstack([rows, np.diag(penalty)])
        rhs = np.concatenate([rhs, np.zeros(matrix.shape[1])])
    if matrix.shape[1]:
        coefficients, _ = nnls(rows, rhs, maxiter=mariner.NNLS_MAXITER_FACTOR * matrix.shape[1])
    else:
        coefficients = np.zeros(0)
    intercept = target_mean - float(design_mean @ coefficients)
    return mariner.Head(intercept, coefficients, floor)


def profile_cv_rmse(
    matrix: np.ndarray,
    response: np.ndarray,
    ridge: float,
    rho: float,
    feature_spans: np.ndarray,
    folds: mariner.Folds,
) -> float:
    """Score actual BPB predictions after recomputing the head within each fold."""
    matrix = np.asarray(matrix, dtype=float)
    response = np.asarray(response, dtype=float)
    error = 0.0
    count = 0
    for train, validation in folds:
        head = fit_profile_head(matrix[train], response[train], ridge, rho, feature_spans)
        prediction = head.predict(matrix[validation])
        if not np.isfinite(prediction).all():
            return float("inf")
        error += float(np.sum((prediction - response[validation]) ** 2))
        count += len(validation)
    if not count:
        raise ValueError("cross-validation requires at least one validation observation")
    return math.sqrt(error / count)
