# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pandas", "scipy", "scikit-learn", "threadpoolctl"]
# ///
"""Convex positive-response mean fitting with the native MARINER floor/link.

For fixed native floor phi and d = max(y - phi, DEFICIT_FLOOR), minimize
sum(exp(eta) - d * eta) / mean(d) + ridge * ||R coefficients||**2 / 2.
The intercept is free; aggregate coefficients are nonnegative and phase
coefficients are signed. This is a mean loss, not a count-noise assumption.

Bounded Newton steps solve their convex quadratic subproblem by BVLS and use
Armijo descent on the unclipped objective. If finite-precision BVLS gives a
non-descending step, a diagonally scaled projected-gradient step provides a
feasible descent direction without weakening first-order certification.
A response-only constant makes the
reported objective a nonnegative exponential Bregman divergence. Centering and
column scaling are numerical coordinate changes: penalty multipliers transform
inversely, so the declared raw-coefficient penalty is exactly preserved.
"""

import dataclasses
import math

import numpy as np
import round3_log_head_20260921 as log_head
from scipy.optimize import lsq_linear
from scipy.special import logsumexp

mariner = log_head.mariner
KKT_TOLERANCE = 1e-10
MAX_ITERATIONS = 100
MAX_BACKTRACKS = 60
ARMIJO_FRACTION = 1e-4


@dataclasses.dataclass(frozen=True)
class Head:
    intercept: float
    aggregate_coefficients: np.ndarray
    phase_coefficients: np.ndarray
    floor: float
    objective: float
    iterations: int
    certified_kkt_residual: float

    def predict(self, aggregate_matrix: np.ndarray, phase_matrix: np.ndarray) -> np.ndarray:
        linear = self.intercept + aggregate_matrix @ self.aggregate_coefficients + phase_matrix @ self.phase_coefficients
        return self.floor + np.exp(np.clip(linear, -mariner.LOG_CLIP, mariner.LOG_CLIP))


def mean_value_gradient(
    vector: np.ndarray,
    matrix: np.ndarray,
    deficit: np.ndarray,
    penalty: np.ndarray,
    ridge: float,
) -> tuple[float, np.ndarray]:
    """Return the centered objective and analytic gradient.

    ``matrix`` includes the intercept column; its penalty entry must be zero.
    ``deficit`` is positive and already has the native numerical deficit floor.
    No exponential clipping, prediction cap, or n normalization is applied.
    """
    linear = matrix @ vector
    difference = linear - np.log(deficit)
    with np.errstate(over="ignore", invalid="ignore"):
        mean = np.exp(linear)
        # expm1 retains precision near the saturated fit. The subtracted
        # response-only constant has no effect on gradients or minimizers.
        data_loss = deficit * (np.expm1(difference) - difference)
        strength = ridge * penalty**2
        value = float(np.sum(data_loss) / deficit.mean() + 0.5 * np.sum(strength * vector**2))
        gradient = matrix.T @ ((mean - deficit) / deficit.mean()) + strength * vector
    return value, gradient


def mean_objective(
    vector: np.ndarray,
    matrix: np.ndarray,
    deficit: np.ndarray,
    penalty: np.ndarray,
    ridge: float,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Return the same mean loss with its analytic gradient and Hessian."""
    value, gradient = mean_value_gradient(vector, matrix, deficit, penalty, ridge)
    mean = np.exp(matrix @ vector)
    hessian = matrix.T @ ((mean / deficit.mean())[:, None] * matrix) + np.diag(ridge * penalty**2)
    return value, gradient, hessian


def profile_intercept(vector: np.ndarray, matrix: np.ndarray, deficit: np.ndarray) -> np.ndarray:
    """Minimize the free intercept exactly at fixed feature coefficients."""
    result = vector.copy()
    result[0] = math.log(float(deficit.sum())) - float(logsumexp(matrix[:, 1:] @ vector[1:]))
    return result


def projected_kkt(vector: np.ndarray, gradient: np.ndarray, aggregate_width: int, rows: int) -> float:
    """Maximum projected gradient per row in the solver's scaled coordinates."""
    projected = gradient.copy()
    constrained = slice(1, 1 + aggregate_width)
    # The projected-gradient mapping avoids declaring a numerically tiny
    # positive coefficient interior when its descent step reaches the bound.
    projected[constrained] = vector[constrained] - np.maximum(0, vector[constrained] - gradient[constrained])
    return float(np.max(np.abs(projected)) / rows)


def projected_descent(vector: np.ndarray, gradient: np.ndarray, curvature: np.ndarray, lower: np.ndarray) -> np.ndarray:
    """Return a feasible diagonally preconditioned projected-gradient step."""
    scale = np.where(curvature > 0, curvature, 1.0)
    return np.maximum(lower, vector - gradient / scale) - vector


def fit_mean_head(
    aggregate_matrix: np.ndarray,
    phase_matrix: np.ndarray,
    response: np.ndarray,
    ridge: float,
    spec: mariner.FloorSpec,
    aggregate_penalty: np.ndarray,
    phase_penalty: np.ndarray,
) -> Head:
    """Fit one conditional mean head and require first-order KKT certification."""
    aggregate_matrix = np.asarray(aggregate_matrix, dtype=float)
    phase_matrix = np.asarray(phase_matrix, dtype=float)
    response = np.asarray(response, dtype=float)
    aggregate_penalty = np.asarray(aggregate_penalty, dtype=float)
    phase_penalty = np.asarray(phase_penalty, dtype=float)
    # Native log fitting supplies a finite warm start and validates this shared
    # API's shapes, positive BPB, and nonnegative penalty multipliers.
    start = log_head.fit_log_head(
        aggregate_matrix, phase_matrix, response, ridge, spec, aggregate_penalty, phase_penalty
    )
    raw = np.hstack([aggregate_matrix, phase_matrix])
    center = raw.mean(axis=0)
    raw_penalty = np.concatenate([aggregate_penalty, phase_penalty])
    scale = np.sqrt(np.mean((raw - center) ** 2, axis=0) + ridge * raw_penalty**2 / len(response))
    scale = np.where(scale > 0, scale, 1.0)
    matrix = np.column_stack([np.ones(len(response)), (raw - center) / scale])
    penalty = np.r_[0.0, raw_penalty / scale]
    raw_start = np.r_[start.aggregate_coefficients, start.phase_coefficients]
    vector = np.r_[start.intercept + center @ raw_start, scale * raw_start]
    deficit = np.maximum(response - start.floor, mariner.DEFICIT_FLOOR)
    aggregate_width = aggregate_matrix.shape[1]
    lower = np.r_[-np.inf, np.zeros(aggregate_width), np.full(phase_matrix.shape[1], -np.inf)]
    upper = np.full(len(vector), np.inf)
    mean_deficit = float(deficit.mean())
    vector = profile_intercept(np.maximum(lower, vector), matrix, deficit)
    for iteration in range(MAX_ITERATIONS + 1):
        value, gradient = mean_value_gradient(vector, matrix, deficit, penalty, ridge)
        if not math.isfinite(value) or not np.isfinite(gradient).all():
            raise FloatingPointError("Mean-head objective is nonfinite at its current iterate")
        residual = projected_kkt(vector, gradient, aggregate_width, len(response))
        if residual <= KKT_TOLERANCE:
            coefficients = vector[1:] / scale
            intercept = float(vector[0] - center @ coefficients)
            return Head(
                intercept,
                coefficients[:aggregate_width],
                coefficients[aggregate_width:],
                start.floor,
                value,
                iteration,
                residual,
            )
        if iteration == MAX_ITERATIONS:
            break
        linear = matrix @ vector
        mean = np.exp(linear)
        weight = np.sqrt(mean / mean_deficit)
        if np.any(weight == 0):
            raise FloatingPointError("Mean-head Newton curvature underflowed")
        design = weight[:, None] * matrix
        rhs = (deficit - mean) / (mean_deficit * weight)
        if ridge > 0:
            design = np.vstack([design, np.diag(math.sqrt(ridge) * penalty)])
            rhs = np.r_[rhs, -math.sqrt(ridge) * penalty * vector]
        quadratic = lsq_linear(
            design,
            rhs,
            bounds=(lower - vector, upper),
            method="bvls",
            tol=1e-12,
            max_iter=mariner.NNLS_MAXITER_FACTOR * len(vector),
        )
        if not quadratic.success:
            raise RuntimeError(f"Mean-head Newton subproblem failed: {quadratic.message}")
        direction = quadratic.x
        directional_derivative = float(gradient @ direction)
        if directional_derivative >= 0:
            curvature = np.sum(design**2, axis=0)
            direction = projected_descent(vector, gradient, curvature, lower)
            directional_derivative = float(gradient @ direction)
            if directional_derivative >= 0:
                raise RuntimeError(f"Mean-head projected direction is not descending; KKT residual {residual:.3g}")
        step = 1.0
        for _ in range(MAX_BACKTRACKS):
            # Displacement-form BVLS can put a coefficient a few ulps below
            # its bound when added to the current iterate. Keep every actual
            # iterate feasible before testing descent or profiling intercept.
            proposal = profile_intercept(np.maximum(lower, vector + step * direction), matrix, deficit)
            displacement = proposal - vector
            linear_displacement = matrix @ displacement
            # Evaluate the objective change directly. Subtracting two full
            # losses loses Newton decreases near a stationary point and can
            # stall before the unchanged KKT tolerance is met.
            with np.errstate(over="ignore", invalid="ignore"):
                change = float(
                    gradient @ displacement
                    + np.sum(mean * (np.expm1(linear_displacement) - linear_displacement)) / mean_deficit
                    + 0.5 * ridge * np.sum((penalty * displacement) ** 2)
                )
            if math.isfinite(change) and change <= ARMIJO_FRACTION * step * directional_derivative:
                vector = proposal
                break
            step *= 0.5
        else:
            raise RuntimeError(f"Mean-head line search failed; KKT residual {residual:.3g}")
    raise RuntimeError(f"Mean-head KKT certification failed after {MAX_ITERATIONS} steps: {residual:.3g}")
