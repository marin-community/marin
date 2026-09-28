# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "scipy"]
# ///
"""A tied-centered temporal correction to a fixed single-phase log-deficit head.

The caller supplies the frozen benefit/harm laws and their task amplitudes. The
two columns sum amplitude-weighted differences between phase-0 exposure and
phase-0 exposure of the physically aggregated tied schedule. Their coefficients
are free-sign departures in the log link, not physical acquisition/damage rates.
No intercept, floor, task amplitudes, or response shapes are fitted here.

Run this file with ``uv run`` for structural checks using synthetic curved laws.
Call ``run_structural_checks`` with the frozen laws to check their integration.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum

import numpy as np
from scipy.optimize import least_squares

LOG_CLIP = 30.0
PENALTY_GRID = (0.0, 0.01, 0.1, 1.0, 10.0, 100.0, math.inf)
START_SIGNAL_FRACTION = 0.1
ResponseLaw = Callable[[np.ndarray], np.ndarray]


class TemporalArm(StrEnum):
    DAMAGE_ONLY = "damage-only"
    BENEFIT_DAMAGE = "benefit+damage"


@dataclass(frozen=True)
class TemporalBasis:
    aggregate: np.ndarray
    total_exposure: np.ndarray
    phase0_exposure: np.ndarray
    tied_phase0_exposure: np.ndarray
    columns: np.ndarray


@dataclass(frozen=True)
class StartDiagnostic:
    name: str
    cost: float
    success: bool
    nfev: int
    optimality: float
    message: str


@dataclass(frozen=True)
class TemporalFit:
    """Fit diagnostics; nfev counts evaluations across all optimization starts."""

    theta: np.ndarray
    column_norms: np.ndarray
    active_columns: np.ndarray
    arm: TemporalArm
    penalty: float
    cost: float
    null_cost: float
    success: bool
    nfev: int
    selected_start: str
    starts: tuple[StartDiagnostic, ...]

    def to_json(self) -> dict[str, object]:
        return {
            "theta": self.theta.tolist(),
            "column_norms": self.column_norms.tolist(),
            "active_columns": self.active_columns.tolist(),
            "arm": str(self.arm),
            "penalty": "inf" if math.isinf(self.penalty) else self.penalty,
            "cost": self.cost,
            "null_cost": self.null_cost,
            "success": self.success,
            "nfev": self.nfev,
            "selected_start": self.selected_start,
            "starts": [
                {
                    "name": start.name,
                    "cost": start.cost,
                    "success": start.success,
                    "nfev": start.nfev,
                    "optimality": start.optimality,
                    "message": start.message,
                }
                for start in self.starts
            ],
        }


def temporal_basis(
    phase0_weights: np.ndarray,
    phase1_weights: np.ndarray,
    epoch_rate0: float | np.ndarray,
    epoch_rate1: float | np.ndarray,
    benefit_amplitudes: np.ndarray,
    harm_amplitudes: np.ndarray,
    benefit: ResponseLaw,
    harm: ResponseLaw,
) -> TemporalBasis:
    """Construct benefit/harm columns relative to each physical tied schedule.

    Args:
        phase0_weights: Rows by buckets of nonnegative simplex weights.
        phase1_weights: Corresponding phase-1 weights with the same shape.
        epoch_rate0: Epochs at unit share, broadcastable to the weight shape.
        epoch_rate1: Phase-1 epochs at unit share, with the same convention.
        benefit_amplitudes: Frozen nonnegative task amplitudes, one per bucket.
        harm_amplitudes: Frozen nonnegative task amplitudes, one per bucket.
        benefit: Frozen benefit law, preserving its exposure argument's shape.
        harm: Frozen harm law, preserving its exposure argument's shape.

    Returns:
        Exposures, the physical aggregate, and columns ordered benefit, harm.
        Identical phase weights produce bitwise-zero columns.
    """
    weights0 = np.asarray(phase0_weights, dtype=float)
    weights1 = np.asarray(phase1_weights, dtype=float)
    if weights0.ndim != 2 or weights0.shape != weights1.shape or not all(weights0.shape):
        raise ValueError("phase weights must have the same nonempty rows-by-buckets shape")
    rates0 = np.broadcast_to(np.asarray(epoch_rate0, dtype=float), weights0.shape)
    rates1 = np.broadcast_to(np.asarray(epoch_rate1, dtype=float), weights0.shape)
    amplitudes = (np.asarray(benefit_amplitudes, dtype=float), np.asarray(harm_amplitudes, dtype=float))
    if any(amplitude.shape != (weights0.shape[1],) for amplitude in amplitudes):
        raise ValueError("each amplitude vector must contain one value per bucket")
    for value in (weights0, weights1, rates0, rates1, *amplitudes):
        if not np.isfinite(value).all() or np.any(value < 0.0):
            raise ValueError("weights, epoch rates, and frozen amplitudes must be finite and nonnegative")
    if not np.allclose(weights0.sum(axis=1), 1.0, rtol=0.0, atol=1e-10) or not np.allclose(
        weights1.sum(axis=1), 1.0, rtol=0.0, atol=1e-10
    ):
        raise ValueError("phase weights must sum to one within each row")
    if np.any(rates0 + rates1 <= 0.0):
        raise ValueError("total epoch rate must be positive in every bucket")
    exposure0 = rates0 * weights0
    total = exposure0 + rates1 * weights1
    aggregate = total / (rates0 + rates1)
    # Avoid roundoff in the algebraic identity (c0*w + c1*w)/(c0+c1) = w.
    aggregate = np.where(weights0 == weights1, weights0, aggregate)
    tied_exposure0 = rates0 * aggregate
    columns = []
    for law, amplitude in zip((benefit, harm), amplitudes, strict=True):
        early = np.asarray(law(exposure0), dtype=float)
        tied = np.asarray(law(tied_exposure0), dtype=float)
        if early.shape != weights0.shape or tied.shape != weights0.shape:
            raise ValueError("response laws must preserve the exposure shape")
        if not np.isfinite(early).all() or not np.isfinite(tied).all():
            raise ValueError("response laws must produce finite values")
        columns.append((early - tied) @ amplitude)
    return TemporalBasis(aggregate, total, exposure0, tied_exposure0, np.column_stack(columns))


def predict_bpb_delta(q: np.ndarray, columns: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """Predict a BPB contrast as q*expm1(Z theta), clipping the correction at 30.

    q is the fixed aggregate schedule's positive deficit exp(eta_A). Clipping is
    an overflow guard on the correction; it is not a coefficient constraint.
    """
    return np.asarray(q, dtype=float) * np.expm1(np.clip(columns @ theta, -LOG_CLIP, LOG_CLIP))


def contrast_residual_jacobian(
    normalized_coefficients: np.ndarray,
    q: np.ndarray,
    normalized_columns: np.ndarray,
    observed_delta: np.ndarray,
    penalty: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return contrast and penalty residuals with their analytic Jacobian.

    Each column of q*normalized_columns has unit L2 norm on training rows;
    normalized_coefficients[j] equals theta[j] * ||q*Z[:, j]||. Thus the ridge
    residual sqrt(penalty)*x penalizes the linearized BPB contribution, without
    centering the contrast or fitting an intercept. At and beyond either clip
    boundary the exponential derivative is zero.
    """
    linear = normalized_columns @ normalized_coefficients
    clipped = np.clip(linear, -LOG_CLIP, LOG_CLIP)
    residual = q * np.expm1(clipped) - observed_delta
    derivative = q * np.exp(clipped) * ((linear > -LOG_CLIP) & (linear < LOG_CLIP))
    jacobian = derivative[:, None] * normalized_columns
    if penalty > 0.0:
        penalty_scale = math.sqrt(penalty)
        residual = np.concatenate((residual, penalty_scale * normalized_coefficients))
        jacobian = np.vstack((jacobian, penalty_scale * np.eye(len(normalized_coefficients))))
    return residual, jacobian


def fit_bpb_contrasts(
    q: np.ndarray,
    columns: np.ndarray,
    observed_delta: np.ndarray,
    penalty: float,
    arm: TemporalArm = TemporalArm.BENEFIT_DAMAGE,
    max_nfev: int = 2000,
    tolerance: float = 1e-10,
) -> TemporalFit:
    """Fit a free-sign temporal head with an explicit exact-null candidate.

    The objective is half the squared BPB contrast error plus
    penalty/2 * sum_j(theta[j]**2 * sum_r((q[r]*Z[r,j])**2)). Infinite penalty
    selects theta=0 exactly. Zero-norm columns and the excluded benefit column
    in the damage-only arm are exactly inactive. Three deterministic starts
    use zero and small positive/negative normalized coefficients, scaled only
    by the training response norm. The lowest-cost candidate is returned even
    if optimization fails; inspect success and starts before using the fit.
    """
    q = np.asarray(q, dtype=float)
    columns = np.asarray(columns, dtype=float)
    observed_delta = np.asarray(observed_delta, dtype=float)
    arm = TemporalArm(arm)
    if q.ndim != 1 or not len(q) or observed_delta.shape != q.shape or columns.shape != (len(q), 2):
        raise ValueError("q and observed_delta must be vectors; columns must have matching rows and two columns")
    if not all(np.isfinite(value).all() for value in (q, columns, observed_delta)) or np.any(q <= 0.0):
        raise ValueError("fit inputs must be finite, with strictly positive q")
    if math.isnan(penalty) or penalty < 0.0:
        raise ValueError("penalty must be nonnegative or positive infinity")
    if max_nfev <= 0 or not np.finfo(float).eps < tolerance < 1.0:
        raise ValueError("max_nfev must be positive and tolerance must lie between machine epsilon and one")
    column_norms = np.linalg.norm(q[:, None] * columns, axis=0)
    if not np.isfinite(column_norms).all():
        raise ValueError("linearized BPB column norms must be finite")
    active = column_norms > 0.0
    if arm == TemporalArm.DAMAGE_ONLY:
        active[0] = False
    theta = np.zeros(2)
    null_cost = 0.5 * float(observed_delta @ observed_delta)
    normalized_columns = columns[:, active] / column_norms[active]
    null_optimality = float(np.max(np.abs((q[:, None] * normalized_columns).T @ observed_delta), initial=0.0))
    fixed_null = math.isinf(penalty) or not np.any(active) or null_cost == 0.0
    diagnostics = [
        StartDiagnostic(
            "null",
            null_cost,
            fixed_null or null_optimality <= tolerance,
            0,
            0.0 if fixed_null else null_optimality,
            "Exact zero correction; no optimizer evaluation.",
        )
    ]
    best = diagnostics[0]
    if not fixed_null:
        width = int(active.sum())
        step = START_SIGNAL_FRACTION * float(np.linalg.norm(observed_delta)) / math.sqrt(width)
        starts = (("zero", np.zeros(width)), ("positive", np.full(width, step)), ("negative", np.full(width, -step)))

        def residual(parameters: np.ndarray) -> np.ndarray:
            return contrast_residual_jacobian(parameters, q, normalized_columns, observed_delta, penalty)[0]

        def jacobian(parameters: np.ndarray) -> np.ndarray:
            return contrast_residual_jacobian(parameters, q, normalized_columns, observed_delta, penalty)[1]

        for name, start in starts:
            result = least_squares(
                residual,
                start,
                jac=jacobian,
                max_nfev=max_nfev,
                ftol=tolerance,
                xtol=tolerance,
                gtol=tolerance,
            )
            diagnostic = StartDiagnostic(
                name,
                float(result.cost),
                bool(result.success),
                int(result.nfev),
                float(result.optimality),
                str(result.message),
            )
            diagnostics.append(diagnostic)
            if math.isfinite(diagnostic.cost) and diagnostic.cost < best.cost:
                best = diagnostic
                theta[active] = result.x / column_norms[active]
    return TemporalFit(
        theta,
        column_norms,
        active,
        arm,
        penalty,
        best.cost,
        null_cost,
        best.success,
        sum(diagnostic.nfev for diagnostic in diagnostics),
        best.name,
        tuple(diagnostics),
    )


def run_structural_checks(benefit: ResponseLaw, harm: ResponseLaw) -> dict[str, object]:
    """Check tied parity, harm-offset cancellation, gradients, and synthetic recovery.

    The supplied fixed laws must yield two independent columns on the synthetic
    design; recovery tests identifiability as well as the optimizer. These checks
    use no observed endpoint outcomes and neither alter nor refit the laws.
    """
    random = np.random.default_rng(20260907)
    weights0 = random.dirichlet(np.ones(4), size=80)
    weights1 = random.dirichlet(np.ones(4), size=80)
    rates0 = np.array([0.03, 0.7, 6.0, 40.0])
    rates1 = rates0 / 4.0
    benefit_amplitudes = np.array([0.4, 0.2, 0.7, 0.1])
    harm_amplitudes = np.array([0.3, 0.6, 0.1, 0.5])

    def basis(second: np.ndarray, damage: ResponseLaw) -> TemporalBasis:
        return temporal_basis(weights0, second, rates0, rates1, benefit_amplitudes, harm_amplitudes, benefit, damage)

    temporal = basis(weights1, harm)
    tied = basis(weights0, harm)
    q = np.linspace(0.2, 1.4, len(weights0))
    truth = np.array([-0.35, 0.18])
    assert np.array_equal(tied.columns, np.zeros_like(tied.columns)), "Tied schedules must have exact zero columns"
    assert np.array_equal(predict_bpb_delta(q, tied.columns, truth), np.zeros_like(q)), "Tied BPB parity failed"
    shifted = basis(weights1, lambda exposure: harm(exposure) + 3.25)
    offset_error = float(np.max(np.abs(shifted.columns - temporal.columns)))
    assert offset_error < 1e-12, "A constant harm offset changed temporal contrasts"

    norms = np.linalg.norm(q[:, None] * temporal.columns, axis=0)
    assert np.all(norms > 0), "Synthetic design has an inactive column"
    normalized = temporal.columns / norms
    parameters = truth * norms
    observed = predict_bpb_delta(q, temporal.columns, truth)
    residual, analytic = contrast_residual_jacobian(parameters, q, normalized, observed, 0.1)
    numerical = np.empty_like(analytic)
    step = 1e-6
    for column in range(len(parameters)):
        direction = np.zeros_like(parameters)
        direction[column] = step
        plus = contrast_residual_jacobian(parameters + direction, q, normalized, observed, 0.1)[0]
        minus = contrast_residual_jacobian(parameters - direction, q, normalized, observed, 0.1)[0]
        numerical[:, column] = (plus - minus) / (2.0 * step)
    jacobian_error = float(np.max(np.abs(analytic - numerical)))
    assert jacobian_error < 1e-8, "Analytic contrast/penalty Jacobian disagrees with finite differences"
    assert len(residual) == len(q) + len(truth)

    clipped_columns = np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0]])
    clipped_parameters = np.array([31.0, 0.2])
    clip_q = np.ones(3)
    clip_observed = np.zeros(3)
    _, clip_jacobian = contrast_residual_jacobian(clipped_parameters, clip_q, clipped_columns, clip_observed, 0.0)
    assert np.array_equal(clip_jacobian[:2], np.zeros((2, 2))), "Clipped derivatives must vanish"

    fitted = fit_bpb_contrasts(q, temporal.columns, observed, 0.0, tolerance=1e-12)
    recovery_error = float(np.max(np.abs(fitted.theta - truth)))
    assert fitted.success and recovery_error < 1e-7, "Noiseless temporal coefficients were not recovered"
    damage_truth = np.array([0.0, -0.23])
    damage_delta = predict_bpb_delta(q, temporal.columns, damage_truth)
    damage_fit = fit_bpb_contrasts(q, temporal.columns, damage_delta, 0.0, TemporalArm.DAMAGE_ONLY, tolerance=1e-12)
    assert damage_fit.success and np.max(np.abs(damage_fit.theta - damage_truth)) < 1e-7
    exact_null = fit_bpb_contrasts(q, temporal.columns, observed, math.inf)
    assert np.array_equal(exact_null.theta, np.zeros(2)) and exact_null.nfev == 0
    inactive = fit_bpb_contrasts(q, tied.columns, observed, 0.0)
    assert np.array_equal(inactive.theta, np.zeros(2)) and not np.any(inactive.active_columns)
    penalized = fit_bpb_contrasts(q, temporal.columns, observed, 1.0)
    direct_cost = 0.5 * float(np.sum((predict_bpb_delta(q, temporal.columns, penalized.theta) - observed) ** 2))
    direct_cost += 0.5 * float(np.sum((penalized.theta * penalized.column_norms) ** 2))
    assert math.isclose(penalized.cost, direct_cost, rel_tol=1e-12, abs_tol=1e-14)
    return {
        "passed": True,
        "tied_max_abs_delta": 0.0,
        "constant_harm_shift_max_abs_error": offset_error,
        "analytic_jacobian_max_abs_error": jacobian_error,
        "noiseless_theta_max_abs_error": recovery_error,
        "noiseless_fit": fitted.to_json(),
        "damage_only_theta": damage_fit.theta.tolist(),
        "exact_null_theta": exact_null.theta.tolist(),
        "penalized_objective_direct_abs_error": abs(penalized.cost - direct_cost),
    }


def main() -> None:
    checks = run_structural_checks(
        benefit=lambda exposure: exposure / (1.0 + exposure),
        harm=lambda exposure: (0.25 + np.log1p(exposure)) ** 2,
    )
    print(json.dumps({"laws": "synthetic curved laws", **checks}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
