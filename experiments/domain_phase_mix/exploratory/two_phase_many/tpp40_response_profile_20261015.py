# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "scipy", "pandas", "threadpoolctl"]
# ///
"""Conditional BPB profile diagnostics; no policy, shape or kinetic selection."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import minimize

from experiments.domain_phase_mix.exploratory.two_phase_many import tpp40_temporal_models_20261014 as temporal

GAMMAS = (1.0, 1.5, 2.5, 4.0, 6.0)
NOISE_SCALE = 0.003
SOLVER_OPTIONS = {"maxiter": 1000, "maxfun": 5000, "maxls": 40, "ftol": 1e-14, "gtol": 1e-10}


class EvaluationBudgetExceeded(ArithmeticError):
    """The frozen per-start evaluation cap was reached."""


@dataclass(frozen=True)
class Profile:
    floor: float
    scale: float
    prediction: np.ndarray
    face: str
    constant: bool
    near_constant: bool
    centered_variance: float
    singular_values: np.ndarray
    unconstrained_floor: float | None
    unconstrained_scale: float | None


def linear_profile(phi: np.ndarray, observed: np.ndarray, floor: float | None) -> Profile:
    """Solve the one-column or nonnegative two-column response profile exactly."""
    if not np.isfinite(phi).all() or not np.isfinite(observed).all() or np.any(phi < 0) or not np.any(phi > 0):
        raise ValueError("The profile requires finite responses and a nonnegative nonzero column")
    mean_phi, mean_y = float(np.mean(phi)), float(np.mean(observed))
    centered = phi - mean_phi
    variance = float(centered @ centered)
    constant = bool(np.ptp(phi) == 0)
    if variance == 0 and not constant:
        raise ArithmeticError("Positive profile spread underflowed; do not silently replace it with a constant")
    singular = np.linalg.svd(np.column_stack((np.ones(len(phi)), phi)), compute_uv=False)
    near = bool(singular[-1] <= np.finfo(float).eps * max(len(phi), 2) * singular[0])
    candidates: list[tuple[float, int, float, float, str]] = []
    free_floor, free_scale = None, None

    def retain(f: float, k: float, face: str, order: int) -> None:
        if not math.isfinite(f) or not math.isfinite(k) or f < 0 or k < 0:
            return
        error = observed - f - k * phi
        value = float(error @ error)
        if math.isfinite(value):
            candidates.append((value, order, f, k, face))

    norm = float(phi @ phi)
    if floor is not None:
        if floor < 0 or not math.isfinite(floor):
            raise ValueError("The supplied floor must be finite and nonnegative")
        scale = max(0.0, float(phi @ (observed - floor)) / norm)
        retain(floor, scale, "known_floor" if scale > 0 else "known_floor_scale_zero", 0)
    else:
        if not constant:
            free_scale = float(centered @ (observed - mean_y)) / variance
            free_floor = mean_y - free_scale * mean_phi
            retain(free_floor, free_scale, "interior", 0)
        retain(0.0, max(0.0, float(phi @ observed) / norm), "floor_zero", 1)
        retain(max(0.0, mean_y), 0.0, "scale_zero", 2)
        retain(0.0, 0.0, "both_zero", 3)
        if constant:
            # This is only a canonical representation, never a stationarity certificate.
            candidates = [item for item in candidates if item[4] == "scale_zero"]
    if not candidates:
        raise ArithmeticError("No finite conditional profile")
    _, _, selected_floor, selected_scale, face = min(candidates)
    return Profile(
        selected_floor,
        selected_scale,
        selected_floor + selected_scale * phi,
        face,
        constant,
        near,
        variance,
        singular,
        free_floor,
        free_scale,
    )


def evaluate(matrix: np.ndarray, observed: np.ndarray, amplitudes: np.ndarray, floor: float | None) -> dict[str, Any]:
    """Evaluate noise-scaled mean SSE and its regular envelope gradient."""
    if not np.isfinite(amplitudes).all() or np.any(amplitudes < 0):
        raise ValueError("Nonnegative finite amplitudes are required")
    logits = matrix @ amplitudes
    if not np.isfinite(logits).all():
        raise ArithmeticError("Nonfinite training logit")
    normalization = float(np.max(logits))
    phi = np.exp(logits - normalization)
    profile = linear_profile(phi, observed, floor)
    residual = observed - profile.prediction
    raw = float(np.mean(residual**2))
    centered_matrix = matrix - matrix[int(np.argmax(logits))]
    gradient = -2 * profile.scale / (len(observed) * NOISE_SCALE**2) * centered_matrix.T @ (phi * residual)
    if not np.isfinite(gradient).all() or not math.isfinite(raw):
        raise ArithmeticError("Nonfinite response profile or gradient")
    projected = np.where((amplitudes == 0) & (gradient > 0), 0, gradient)
    nonregular = floor is None and profile.constant
    return {
        "amplitudes": amplitudes.tolist(),
        "floor": profile.floor,
        "scale": profile.scale,
        "normalization": normalization,
        "intercept": None if profile.scale == 0 else math.log(profile.scale) - normalization,
        "raw_mean_squared_bpb": raw,
        "scaled_objective": raw / NOISE_SCALE**2,
        "gradient": gradient.tolist(),
        "projected_gradient_inf": float(np.max(abs(projected))),
        "face": profile.face,
        "constant_phi": profile.constant,
        "near_constant_phi": profile.near_constant,
        "nonregular_learned_floor": nonregular,
        "centered_phi_variance": profile.centered_variance,
        "profile_singular_values": profile.singular_values.tolist(),
        "unconstrained_floor": profile.unconstrained_floor,
        "unconstrained_scale": profile.unconstrained_scale,
        "training_exponential_underflow_count": int(np.sum(phi == 0)),
        "scale_orthogonality_residual": float(profile.scale * (phi @ residual)),
        "first_order_tolerance_met": bool(not nonregular and np.max(abs(projected)) <= SOLVER_OPTIONS["gtol"]),
    }


def predict(matrix: np.ndarray, record: dict[str, Any]) -> np.ndarray:
    """Evaluate a saved conditional head without a prediction cap."""
    if record["scale"] == 0:
        return np.full(len(matrix), record["floor"])
    # The normalized representation is authoritative; subtract before adding log(scale).
    linear = math.log(record["scale"]) + (matrix @ np.asarray(record["amplitudes"]) - record["normalization"])
    with np.errstate(over="raise", invalid="raise"):
        predicted = record["floor"] + np.exp(linear)
    if not np.isfinite(predicted).all():
        raise ArithmeticError("Nonfinite query response")
    return predicted


def gamma_starts(matrix: np.ndarray, observed: np.ndarray) -> list[dict[str, Any]]:
    """Use the five existing training-local native fits solely for initialization."""
    records = []
    for gamma in GAMMAS:
        head = temporal.fit_head_with_offset(matrix, np.zeros(len(observed)), observed, 0.0, gamma)
        records.append(
            {
                "gamma": gamma,
                "provisional_floor": head.floor,
                "provisional_intercept": head.intercept,
                "amplitudes": head.coefficients.tolist(),
                "deficit_clamp_count": int(np.sum(observed - head.floor < temporal.mariner.DEFICIT_FLOOR)),
            }
        )
    return records


def fit_response(
    matrix: np.ndarray, observed: np.ndarray, floor: float | None, starts: list[dict[str, Any]]
) -> dict[str, Any]:
    """Run exactly five prescribed starts, retaining every finite evaluated candidate."""
    if tuple(item["gamma"] for item in starts) != GAMMAS:
        raise ValueError("The exact five gamma-derived starts are required")
    trials: list[dict[str, Any]] = []
    solves = []
    for start_index, start in enumerate(starts):
        first_trial = len(trials)

        def objective(
            amplitudes: np.ndarray, first_trial: int = first_trial, start_index: int = start_index
        ) -> tuple[float, np.ndarray]:
            if len(trials) - first_trial >= SOLVER_OPTIONS["maxfun"]:
                raise EvaluationBudgetExceeded("Frozen per-start objective cap reached")
            try:
                value = evaluate(matrix, observed, amplitudes, floor)
            except (ArithmeticError, ValueError) as error:
                trials.append(
                    {
                        "trial_id": len(trials),
                        "start_index": start_index,
                        "amplitudes": amplitudes.tolist(),
                        "status": "numerical_failure",
                        "exception": type(error).__name__,
                        "message": str(error),
                    }
                )
                raise
            trials.append({"trial_id": len(trials), "start_index": start_index, "status": "finite", **value})
            return value["scaled_objective"], np.asarray(value["gradient"])

        initial = np.asarray(start["amplitudes"], float)
        try:
            objective(initial)
            if trials[-1]["nonregular_learned_floor"]:
                solve = {
                    "success": False,
                    "status": -2,
                    "message": "Constant-profile start is nonregular; retained only",
                    "nit": 0,
                    "nfev": 1,
                    "njev": 1,
                    "x": initial.tolist(),
                    "termination": "nonregular_start",
                }
            else:
                result = minimize(
                    objective,
                    initial,
                    method="L-BFGS-B",
                    jac=True,
                    bounds=[(0.0, None)] * matrix.shape[1],
                    options=SOLVER_OPTIONS,
                )
                message = str(result.message)
                solve = {
                    "success": bool(result.success),
                    "status": int(result.status),
                    "message": message,
                    "nit": int(result.nit),
                    "nfev": int(result.nfev),
                    "njev": int(result.njev),
                    "x": result.x.tolist(),
                    "fun": float(result.fun),
                    "termination": (
                        "gradient" if "GRADIENT" in message else "ftol" if "RELATIVE REDUCTION" in message else "other"
                    ),
                }
        except EvaluationBudgetExceeded as error:
            solve = {"success": False, "status": -3, "message": str(error), "termination": "evaluation_cap"}
        except (ArithmeticError, ValueError) as error:
            solve = {
                "success": False,
                "status": -1,
                "message": str(error),
                "exception": type(error).__name__,
                "termination": "numerical_failure",
            }
        indices = [index for index in range(first_trial, len(trials)) if trials[index]["status"] == "finite"]
        best = min(indices, key=lambda index: (trials[index]["scaled_objective"], index)) if indices else None
        solves.append(
            {
                "start_index": start_index,
                "start": start,
                "first_trial": first_trial,
                "last_trial": len(trials) - 1,
                "best_trial": best,
                "initial_evaluation_calls": 1,
                "recorded_objective_calls": len(trials) - first_trial,
                **solve,
            }
        )
    finite = [i for i, trial in enumerate(trials) if trial["status"] == "finite"]
    if not finite:
        return {
            "status": "no_finite_candidate",
            "floor_mode": "learned_nonnegative" if floor is None else "known",
            "trials": trials,
            "solves": solves,
            "selected_trial": None,
            "selected": None,
            "unresolved": True,
        }
    winner = min(finite, key=lambda index: (trials[index]["scaled_objective"], index))
    selected = trials[winner]
    selected_solve = solves[selected["start_index"]]
    unresolved = (
        not selected_solve["success"]
        or selected["nonregular_learned_floor"]
        or (floor is None and selected["near_constant_phi"])
        or selected["training_exponential_underflow_count"] > 0
        or not selected["first_order_tolerance_met"]
    )
    return {
        "status": "finite_descriptive_incumbent",
        "floor_mode": "learned_nonnegative" if floor is None else "known",
        "known_floor": floor,
        "noise_scale": NOISE_SCALE,
        "objective": "mean_squared_bpb_divided_by_noise_scale_squared",
        "solver_options": dict(SOLVER_OPTIONS),
        "starts": starts,
        "trials": trials,
        "solves": solves,
        "selected_trial": winner,
        "selected": selected,
        "selected_from_successful_solve": selected_solve["success"],
        "unresolved": bool(unresolved),
        "global_optimum_certified": False,
    }
