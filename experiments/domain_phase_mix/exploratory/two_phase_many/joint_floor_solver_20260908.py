# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Response-space controls for the frozen log-deficit surrogate's floor estimate.

For fixed shape features X and ridge lambda, minimize
  0.5 * (sum(((phi(gamma) + exp(c + X beta) - y) / s)**2) + lambda * ||beta||**2),
with beta >= 0 and gamma in [1, 6]. The fixed control pins gamma at the incumbent.
The scale s = mean(y - phi(incumbent_gamma)) is fixed for every start and gamma.
Thus, in raw BPB units the coefficient penalty is lambda * s**2 * ||beta||**2.
No validation responses enter this optimizer or its starts.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.optimize import least_squares

LOG_CLIP = 30.0
DEFICIT_FLOOR = 1e-9
GAMMA_BOUNDS = (1.0, 6.0)
NOISE_MARGIN_SDS = 3.0

ConditionalSolve = Callable[[np.ndarray, np.ndarray, float], tuple[float, np.ndarray]]


@dataclass(frozen=True)
class ResponseStart:
    """An explicit coefficient/gamma start, including a previous response fit."""

    intercept: float
    coefficients: np.ndarray
    gamma: float


@dataclass(frozen=True)
class StartDiagnostic:
    start_gamma: float
    final_gamma: float
    success: bool
    cost: float
    optimality: float
    nfev: int
    njev: int
    status: int
    message: str
    active_bounds: tuple[str, ...]


@dataclass(frozen=True)
class ResponseHeadFit:
    intercept: float
    coefficients: np.ndarray
    floor: float
    gamma: float
    deficit_scale: float
    success: bool
    cost: float
    optimality: float
    nfev: int
    active_bounds: tuple[str, ...]
    starts: tuple[StartDiagnostic, ...]

    def predict(self, matrix: np.ndarray) -> np.ndarray:
        return self.floor + np.exp(np.clip(self.intercept + matrix @ self.coefficients, -LOG_CLIP, LOG_CLIP))


def floor_parameters(response: np.ndarray, anchor: float) -> tuple[float, float]:
    """Resolve the anchor and gap exactly as the standalone floor_value does."""
    anchor = float(np.median(response)) if math.isnan(anchor) else float(anchor)
    gap = anchor - float(np.min(response))
    if gap <= 0.0:
        gap = max(float(np.std(response)), 1e-3 * abs(anchor), 1e-6)
    return anchor, gap


def response_residual_jacobian(
    parameters: np.ndarray,
    matrix: np.ndarray,
    response: np.ndarray,
    ridge: float,
    anchor: float,
    gap: float,
    noise_sd: float,
    deficit_scale: float,
    fixed_gamma: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return residuals and their analytic Jacobian, including the ridge rows.

    Parameter order is intercept, nonnegative coefficients, and gamma when free.
    At the noise-margin kink the gamma derivative uses the gamma-active side.
    At and beyond the numerical clipping boundaries the exponential derivative
    is zero. Neither kink has a unique classical derivative.
    """
    width = matrix.shape[1]
    coefficients = parameters[1 : width + 1]
    gamma = float(parameters[-1]) if fixed_gamma is None else fixed_gamma
    floor = anchor - max(gamma * gap, NOISE_MARGIN_SDS * noise_sd)
    linear = parameters[0] + matrix @ coefficients
    deficit = np.exp(np.clip(linear, -LOG_CLIP, LOG_CLIP))
    derivative = deficit * ((linear > -LOG_CLIP) & (linear < LOG_CLIP)) / deficit_scale
    residual = (floor + deficit - response) / deficit_scale
    jacobian = np.empty((len(response), len(parameters)), dtype=float)
    jacobian[:, 0] = derivative
    jacobian[:, 1 : width + 1] = derivative[:, None] * matrix
    if fixed_gamma is None:
        floor_derivative = -gap if gamma * gap >= NOISE_MARGIN_SDS * noise_sd else 0.0
        jacobian[:, -1] = floor_derivative / deficit_scale
    if ridge > 0.0:
        penalty = math.sqrt(ridge)
        residual = np.concatenate([residual, penalty * coefficients])
        ridge_jacobian = np.zeros((width, len(parameters)), dtype=float)
        ridge_jacobian[:, 1 : width + 1] = penalty * np.eye(width)
        jacobian = np.vstack([jacobian, ridge_jacobian])
    return residual, jacobian


def fit_response_head(
    matrix: np.ndarray,
    response: np.ndarray,
    ridge: float,
    anchor: float,
    noise_sd: float,
    incumbent_gamma: float,
    conditional_solve: ConditionalSolve,
    mode: Literal["fixed", "joint"] = "joint",
    gamma_starts: Sequence[float] | None = None,
    extra_starts: Sequence[ResponseStart] = (),
    max_nfev: int = 2000,
    tolerance: float = 1e-9,
) -> ResponseHeadFit:
    """Fit a response head from common conditional-NNLS and explicit starts.

    Both modes initialize coefficients from the same candidate floors. Only the
    joint mode optimizes gamma. Pass a fixed-control result as an extra start to
    the joint fit to retain a feasible fixed-control candidate; pass a previous
    result and an empty gamma_starts sequence for convergence polishing.

    The lowest-cost finite result is returned even when its termination reports
    failure. Callers must inspect success and the per-start diagnostics.
    """
    matrix = np.asarray(matrix, dtype=float)
    response = np.asarray(response, dtype=float)
    if matrix.ndim != 2 or response.ndim != 1 or matrix.shape[0] != len(response) or not len(response):
        raise ValueError("matrix and response must contain the same nonzero number of rows")
    if not np.isfinite(matrix).all() or not np.isfinite(response).all():
        raise ValueError("matrix and response must be finite")
    if mode not in ("fixed", "joint"):
        raise ValueError(f"unknown fitting mode: {mode}")
    if not GAMMA_BOUNDS[0] <= incumbent_gamma <= GAMMA_BOUNDS[1]:
        raise ValueError("incumbent_gamma must be in [1, 6]")
    if not math.isfinite(ridge) or ridge < 0.0 or not math.isfinite(noise_sd) or noise_sd < 0.0:
        raise ValueError("ridge and noise_sd must be finite and nonnegative")
    anchor, gap = floor_parameters(response, anchor)
    if not math.isfinite(anchor):
        raise ValueError("anchor must be finite or NaN for the training median")
    incumbent_floor = anchor - max(incumbent_gamma * gap, NOISE_MARGIN_SDS * noise_sd)
    deficit_scale = max(float(np.mean(response - incumbent_floor)), DEFICIT_FLOOR)
    candidates = (incumbent_gamma, 1.0, 1.5, 2.5, 6.0) if gamma_starts is None else gamma_starts
    starts = []
    for gamma in dict.fromkeys(float(value) for value in candidates):
        if not GAMMA_BOUNDS[0] <= gamma <= GAMMA_BOUNDS[1]:
            raise ValueError("every gamma start must be in [1, 6]")
        floor = anchor - max(gamma * gap, NOISE_MARGIN_SDS * noise_sd)
        target = np.log(np.maximum(response - floor, DEFICIT_FLOOR))
        intercept, coefficients = conditional_solve(matrix, target, ridge)
        starts.append(ResponseStart(float(intercept), np.asarray(coefficients, float), gamma))
    starts.extend(extra_starts)
    if not starts:
        raise ValueError("at least one conditional or explicit start is required")
    width = matrix.shape[1]
    lower = np.concatenate([[-np.inf], np.zeros(width)])
    upper = np.full(width + 1, np.inf)
    fixed_gamma = incumbent_gamma if mode == "fixed" else None
    if fixed_gamma is None:
        lower = np.concatenate([lower, [GAMMA_BOUNDS[0]]])
        upper = np.concatenate([upper, [GAMMA_BOUNDS[1]]])

    def residual_and_jacobian(parameters: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return response_residual_jacobian(
            parameters, matrix, response, ridge, anchor, gap, noise_sd, deficit_scale, fixed_gamma
        )

    results = []
    diagnostics = []
    for start in starts:
        if start.coefficients.shape != (width,):
            raise ValueError("start coefficients must match the feature width")
        if not GAMMA_BOUNDS[0] <= start.gamma <= GAMMA_BOUNDS[1]:
            raise ValueError("explicit start gamma must be in [1, 6]")
        initial = np.concatenate([[start.intercept], start.coefficients])
        if fixed_gamma is None:
            initial = np.concatenate([initial, [start.gamma]])
        result = least_squares(
            lambda parameters: residual_and_jacobian(parameters)[0],
            initial,
            jac=lambda parameters: residual_and_jacobian(parameters)[1],
            bounds=(lower, upper),
            method="trf",
            x_scale="jac",
            max_nfev=max_nfev,
            ftol=tolerance,
            xtol=tolerance,
            gtol=tolerance,
        )
        if not np.isfinite(result.cost):
            raise FloatingPointError("response-space solver returned a non-finite cost")
        gamma = float(result.x[-1]) if fixed_gamma is None else fixed_gamma
        active = [f"coefficient_{index}_lower" for index in np.flatnonzero(result.active_mask[1 : width + 1])]
        if fixed_gamma is None and result.active_mask[-1]:
            active.append("gamma_lower" if result.active_mask[-1] < 0 else "gamma_upper")
        diagnostic = StartDiagnostic(
            start_gamma=float(start.gamma),
            final_gamma=gamma,
            success=bool(result.success),
            cost=float(result.cost),
            optimality=float(result.optimality),
            nfev=int(result.nfev),
            njev=int(result.njev or 0),
            status=int(result.status),
            message=str(result.message),
            active_bounds=tuple(active),
        )
        results.append(result)
        diagnostics.append(diagnostic)
    winner = min(range(len(results)), key=lambda index: results[index].cost)
    result = results[winner]
    diagnostic = diagnostics[winner]
    return ResponseHeadFit(
        intercept=float(result.x[0]),
        coefficients=np.asarray(result.x[1 : width + 1], dtype=float),
        floor=anchor - max(diagnostic.final_gamma * gap, NOISE_MARGIN_SDS * noise_sd),
        gamma=diagnostic.final_gamma,
        deficit_scale=deficit_scale,
        success=diagnostic.success,
        cost=diagnostic.cost,
        optimality=diagnostic.optimality,
        nfev=diagnostic.nfev,
        active_bounds=diagnostic.active_bounds,
        starts=tuple(diagnostics),
    )
