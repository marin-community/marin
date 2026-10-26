# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pandas", "scipy", "scikit-learn", "threadpoolctl"]
# ///
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""MARINER log-deficit fitting with nonnegative aggregate and signed phase terms.

For the native training-local floor phi, fit a free intercept c and minimize

    ||log(max(y - phi, DEFICIT_FLOOR)) - c - A beta - Z theta||**2
    + ridge * (||R_aggregate beta||**2 + ||R_phase theta||**2)

subject only to beta >= 0. Penalty multipliers are explicit caller inputs;
this module neither estimates them from column variation nor selects a floor,
shape, phase representation, or ridge. A tied-zero Z preserves the MARINER
tied response law, including its numerical log clipping.
"""

import dataclasses
import math

import benchmark_two_phase_20260921 as benchmark
import numpy as np
from scipy.optimize import lsq_linear

mariner = benchmark.mariner


@dataclasses.dataclass(frozen=True)
class Head:
    intercept: float
    aggregate_coefficients: np.ndarray
    phase_coefficients: np.ndarray
    floor: float

    def predict(self, aggregate_matrix: np.ndarray, phase_matrix: np.ndarray) -> np.ndarray:
        linear = self.intercept + aggregate_matrix @ self.aggregate_coefficients + phase_matrix @ self.phase_coefficients
        return self.floor + np.exp(np.clip(linear, -mariner.LOG_CLIP, mariner.LOG_CLIP))


def fit_log_head(
    aggregate_matrix: np.ndarray,
    phase_matrix: np.ndarray,
    response: np.ndarray,
    ridge: float,
    spec: mariner.FloorSpec,
    aggregate_penalty: np.ndarray,
    phase_penalty: np.ndarray,
) -> Head:
    """Fit one conditional head using only the supplied training responses.

    Args:
        aggregate_matrix: Signed MARINER benefit/harm columns, without intercept.
        phase_matrix: Tied-zero phase columns whose coefficients have either sign.
        response: Finite, positive BPB observations for the same rows.
        ridge: Nonnegative penalty strength.
        spec: Native MARINER floor specification, evaluated on these rows.
        aggregate_penalty: Nonnegative per-column aggregate penalty multipliers.
        phase_penalty: Nonnegative per-column phase penalty multipliers.
    """
    aggregate_matrix = np.asarray(aggregate_matrix, dtype=float)
    phase_matrix = np.asarray(phase_matrix, dtype=float)
    response = np.asarray(response, dtype=float)
    aggregate_penalty = np.asarray(aggregate_penalty, dtype=float)
    phase_penalty = np.asarray(phase_penalty, dtype=float)
    if response.ndim != 1 or not len(response):
        raise ValueError("response must be a nonempty vector")
    if aggregate_matrix.ndim != 2 or phase_matrix.ndim != 2:
        raise ValueError("aggregate and phase matrices must be two-dimensional")
    if aggregate_matrix.shape[0] != len(response) or phase_matrix.shape[0] != len(response):
        raise ValueError("feature matrices and response must have the same number of rows")
    if aggregate_penalty.shape != (aggregate_matrix.shape[1],) or phase_penalty.shape != (phase_matrix.shape[1],):
        raise ValueError("penalty multipliers must match their feature widths")
    matrix = np.hstack([aggregate_matrix, phase_matrix])
    penalty = np.concatenate([aggregate_penalty, phase_penalty])
    if not np.isfinite(matrix).all() or not np.isfinite(response).all() or np.any(response <= 0):
        raise ValueError("features must be finite and observed BPB must be finite and positive")
    if not math.isfinite(ridge) or ridge < 0 or not np.isfinite(penalty).all() or np.any(penalty < 0):
        raise ValueError("ridge and penalty multipliers must be finite and nonnegative")

    floor = mariner.floor_value(response, spec)
    target = np.log(np.maximum(response - floor, mariner.DEFICIT_FLOOR))
    if not math.isfinite(floor) or not np.isfinite(target).all():
        raise FloatingPointError("native floor produces a nonfinite log-deficit target")
    matrix_mean = matrix.mean(axis=0)
    target_mean = float(target.mean())
    width = matrix.shape[1]
    if not width:
        return Head(target_mean, np.zeros(0), np.zeros(0), floor)

    rows = matrix - matrix_mean
    rhs = target - target_mean
    if ridge > 0:
        rows = np.vstack([rows, np.diag(math.sqrt(ridge) * penalty)])
        rhs = np.concatenate([rhs, np.zeros(width)])
    aggregate_width = aggregate_matrix.shape[1]
    lower = np.r_[np.zeros(aggregate_width), np.full(phase_matrix.shape[1], -np.inf)]
    # BVLS uses least squares within each active set, retaining the actual
    # column space when signed contrasts have algebraically dependent columns.
    result = lsq_linear(
        rows,
        rhs,
        bounds=(lower, np.full(width, np.inf)),
        method="bvls",
        tol=1e-12,
        max_iter=mariner.NNLS_MAXITER_FACTOR * width,
    )
    if not result.success:
        raise RuntimeError(f"Mixed-bound log head did not converge: {result.message}")
    coefficients = result.x
    intercept = target_mean - float(matrix_mean @ coefficients)
    return Head(intercept, coefficients[:aggregate_width], coefficients[aggregate_width:], floor)
