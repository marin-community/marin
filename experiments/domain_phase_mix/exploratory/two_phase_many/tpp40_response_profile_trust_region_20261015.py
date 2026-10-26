# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "scipy", "pandas", "scikit-learn"]
# ///
"""TP040: bounded TRF control of the unchanged TP036 response profile."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy.optimize import least_squares

from experiments.domain_phase_mix.exploratory.two_phase_many import tpp40_response_profile_20261015 as prior

SOLVER_OPTIONS = {"method": "trf", "x_scale": "jac", "ftol": 1e-14, "xtol": 1e-14, "gtol": 5e-11}
EVALUATION_CAP = 5000
FACE_ROUNDOFF_MULTIPLIER = 64.0


class EvaluationBudgetExceeded(RuntimeError):
    """The registered cap includes every actual objective evaluation."""


def residual_jacobian(
    matrix: np.ndarray, observed: np.ndarray, record: dict[str, Any], floor: float | None
) -> tuple[np.ndarray, np.ndarray | None, dict[str, Any]]:
    """Differentiate the exact linear profile on a constant-rank active face.

    A nonregular point still supplies its finite residual and diagnostics. Its
    Jacobian is unavailable; callers must retain the point without certifying it.
    """
    amplitudes = np.asarray(record["amplitudes"])
    logits = matrix @ amplitudes
    phi = np.exp(logits - record["normalization"])
    residual = record["floor"] + record["scale"] * phi - observed
    normalized = residual / (prior.NOISE_SCALE * math.sqrt(len(observed)))
    columns = np.column_stack((np.ones(len(phi)), phi)) if floor is None else phi[:, None]
    coefficients = np.asarray([record["floor"], record["scale"]] if floor is None else [record["scale"]])
    active = coefficients > 0
    z = columns[:, active]
    dual = columns.T @ residual
    dual_tolerance = (
        FACE_ROUNDOFF_MULTIPLIER
        * np.finfo(float).eps
        * np.maximum(1.0, np.linalg.norm(columns, axis=0) * np.linalg.norm(residual))
    )
    ambiguous = bool(np.any(dual[~active] <= dual_tolerance[~active]))
    u, singular, vt = np.linalg.svd(z, full_matrices=False)
    rank_tolerance = np.finfo(float).eps * max(z.shape) * singular[0] if len(singular) else 0.0
    rank = int(np.sum(singular > rank_tolerance))
    reasons = []
    if rank != z.shape[1]:
        reasons.append("active_linear_rank_loss")
    if ambiguous:
        reasons.append("inactive_linear_face_not_strict")
    if record["nonregular_learned_floor"] or (floor is None and record["near_constant_phi"]):
        reasons.append("constant_or_near_constant_learned_profile")
    if record["training_exponential_underflow_count"]:
        reasons.append("training_exponential_underflow")
    diagnostics = {
        "jacobian_regular": not reasons,
        "jacobian_limitations": reasons,
        "active_linear_columns": np.flatnonzero(active).tolist(),
        "active_linear_rank": rank,
        "active_linear_singular_values": singular.tolist(),
        "inactive_dual": dual[~active].tolist(),
        "inactive_dual_tolerance": dual_tolerance[~active].tolist(),
        "zero_scale_closure": record["scale"] == 0,
        "minimum_amplitude_bound_distance": float(np.min(amplitudes)),
        "exact_zero_amplitudes": int(np.sum(amplitudes == 0)),
    }
    if reasons:
        return normalized, None, diagnostics
    dphi = phi[:, None] * (matrix - matrix[int(np.argmax(logits))])
    direct = record["scale"] * dphi
    # Only the final full linear column depends on amplitudes.
    differential_transpose_residual = np.zeros((z.shape[1], matrix.shape[1]))
    if active[-1]:
        differential_transpose_residual[-1] = dphi.T @ residual
    jacobian = direct - u @ (u.T @ direct)
    if len(singular):
        jacobian -= (u / singular) @ vt @ differential_transpose_residual
    jacobian /= prior.NOISE_SCALE * math.sqrt(len(observed))
    if not np.isfinite(normalized).all() or not np.isfinite(jacobian).all():
        raise ArithmeticError("Nonfinite profiled residual/Jacobian")
    diagnostics["envelope_gradient_max_discrepancy"] = float(
        np.max(abs(2 * jacobian.T @ normalized - np.asarray(record["gradient"])))
    )
    return normalized, jacobian, diagnostics


def fit_response(
    matrix: np.ndarray,
    observed: np.ndarray,
    floor: float | None,
    starts: list[dict[str, Any]],
    *,
    evaluation_cap: int = EVALUATION_CAP,
) -> dict[str, Any]:
    """Retain all finite trials under one bounded TRF solve per supplied start."""
    trials: list[dict[str, Any]] = []
    solves: list[dict[str, Any]] = []
    for start_index, start in enumerate(starts):
        first_trial = len(trials)
        actual_calls = 0
        cached_x: np.ndarray | None = None
        cached_residual: np.ndarray | None = None
        cached_jacobian: np.ndarray | None = None
        cached_diagnostics: dict[str, Any] = {}

        def point(
            amplitudes: np.ndarray, start_index: int = start_index
        ) -> tuple[np.ndarray, np.ndarray | None, dict[str, Any]]:
            nonlocal actual_calls, cached_x, cached_residual, cached_jacobian, cached_diagnostics
            if cached_x is not None and np.array_equal(amplitudes, cached_x):
                assert cached_residual is not None
                return cached_residual, cached_jacobian, cached_diagnostics
            if actual_calls >= evaluation_cap:
                raise EvaluationBudgetExceeded("Frozen actual-evaluation cap reached")
            actual_calls += 1
            try:
                value = prior.evaluate(matrix, observed, amplitudes, floor)
            except (ArithmeticError, ValueError, np.linalg.LinAlgError) as error:
                trials.append(
                    {
                        "trial_id": len(trials),
                        "start_index": start_index,
                        "status": "numerical_failure",
                        "amplitudes": amplitudes.tolist(),
                        "exception": type(error).__name__,
                        "message": str(error),
                    }
                )
                raise
            try:
                residual, jacobian, diagnostics = residual_jacobian(matrix, observed, value, floor)
            except (ArithmeticError, np.linalg.LinAlgError) as error:
                logits = matrix @ amplitudes
                residual = (value["floor"] + value["scale"] * np.exp(logits - value["normalization"]) - observed) / (
                    prior.NOISE_SCALE * math.sqrt(len(observed))
                )
                jacobian = None
                diagnostics = {
                    "jacobian_regular": False,
                    "jacobian_limitations": ["nonfinite_jacobian"],
                    "jacobian_exception": str(error),
                    "zero_scale_closure": value["scale"] == 0,
                }
            trials.append(
                {"trial_id": len(trials), "start_index": start_index, "status": "finite", **value, **diagnostics}
            )
            cached_x, cached_residual, cached_jacobian = amplitudes.copy(), residual, jacobian
            cached_diagnostics = diagnostics
            return residual, jacobian, diagnostics

        def fun(amplitudes: np.ndarray) -> np.ndarray:
            return point(amplitudes)[0]

        def jac(amplitudes: np.ndarray) -> np.ndarray:
            _, value, diagnostics = point(amplitudes)
            if value is None:
                raise ArithmeticError(f"Nonregular residual Jacobian: {diagnostics['jacobian_limitations']}")
            return value

        try:
            initial = np.asarray(start["amplitudes"], float)
            fun(initial)
            jac(initial)
            result = least_squares(
                fun,
                initial,
                jac=jac,
                bounds=(np.zeros(matrix.shape[1]), np.full(matrix.shape[1], np.inf)),
                max_nfev=evaluation_cap,
                **SOLVER_OPTIONS,
            )
            termination = {0: "evaluation_cap", 1: "gradient", 2: "ftol", 3: "xtol", 4: "ftol_and_xtol"}[result.status]
            solve = {
                "success": bool(result.success),
                "status": int(result.status),
                "message": str(result.message),
                "termination": termination,
                "nfev": int(result.nfev),
                "njev": int(result.njev),
                "x": result.x.tolist(),
                "cost": float(result.cost),
                "optimality": float(result.optimality),
                "active_mask": result.active_mask.tolist(),
            }
        except (EvaluationBudgetExceeded, ArithmeticError, ValueError, np.linalg.LinAlgError) as error:
            solve = {
                "success": False,
                "status": -1,
                "exception": type(error).__name__,
                "message": str(error),
                "termination": "evaluation_cap" if isinstance(error, EvaluationBudgetExceeded) else "numerical_failure",
            }
        finite = [index for index in range(first_trial, len(trials)) if trials[index]["status"] == "finite"]
        best = min(finite, key=lambda index: (trials[index]["scaled_objective"], index)) if finite else None
        solves.append(
            {
                "start_index": start_index,
                "start": start,
                "first_trial": first_trial,
                "last_trial": len(trials) - 1,
                "best_trial": best,
                "actual_evaluation_calls": actual_calls,
                **solve,
            }
        )
    finite = [index for index, trial in enumerate(trials) if trial["status"] == "finite"]
    winner = min(finite, key=lambda index: (trials[index]["scaled_objective"], index)) if finite else None
    selected = None if winner is None else trials[winner]
    successful = selected is not None and solves[selected["start_index"]]["success"]
    unresolved = selected is None or not successful or not selected["first_order_tolerance_met"]
    if selected is not None:
        unresolved = unresolved or not selected["jacobian_regular"] or selected["zero_scale_closure"]
    return {
        "status": "no_finite_candidate" if selected is None else "finite_descriptive_incumbent",
        "floor_mode": "learned_nonnegative" if floor is None else "known",
        "known_floor": floor,
        "noise_scale": prior.NOISE_SCALE,
        "objective": "mean_squared_bpb_divided_by_noise_scale_squared",
        "solver_options": dict(SOLVER_OPTIONS),
        "actual_evaluation_cap_per_start": evaluation_cap,
        "starts": starts,
        "trials": trials,
        "solves": solves,
        "selected_trial": winner,
        "selected": selected,
        "selected_from_successful_solve": successful,
        "unresolved": bool(unresolved),
        "global_optimum_certified": False,
    }
