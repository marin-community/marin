# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pandas", "scipy", "threadpoolctl"]
# ///
"""Joint static-shape, temporal, ridge and floor selection on asymmetric data.

An explicit shape tuple defines J1; omitting it searches all native shapes (J2).
The production search visits the complete discrete grid, profiles the floor at
the winning shape/kinetics for every ridge, and rescans the complete permitted
shape/kinetics/ridge grid at the best refined floor multiplier. Every comparison
uses pooled inner-fold BPB error with training-local amplitudes and floors.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import minimize_scalar
from threadpoolctl import threadpool_limits

from experiments.domain_phase_mix.exploratory.two_phase_many import tpp40_mechanism_models_20261014 as mechanisms
from experiments.domain_phase_mix.exploratory.two_phase_many import tpp40_temporal_models_20261014 as temporal

mariner = temporal.mariner
MODEL_NAMES = ("aggregate_mariner", "bounded_benefit_odds", "contrast_magnitude")
GAMMA_GRID = (1.0, 1.5, 2.5, 4.0, 6.0)


@dataclass(frozen=True)
class TestGrid:
    """Explicit small candidate menus for behavior tests, never production fits."""

    parameters: tuple[dict[str, float], ...]
    ridges: tuple[float, ...]
    gammas: tuple[float, ...]

    __test__ = False


def parameter_grid(model: str) -> tuple[dict[str, float], ...]:
    if model not in MODEL_NAMES:
        raise ValueError(f"Unsupported joint model: {model}")
    return ({},) if model == "aggregate_mariner" else mechanisms.parameter_grid(model)


def design_and_offset(
    early: np.ndarray,
    late: np.ndarray,
    alpha: float,
    shape: dict[str, float],
    model: str,
    parameters: dict[str, float],
) -> tuple[np.ndarray, np.ndarray]:
    """Delegate to the frozen round-two law, including its exact tied restriction."""
    if model == "aggregate_mariner":
        return mariner.design_matrix(early + late, shape), np.zeros(len(early))
    return mechanisms.design_and_offset(early, late, alpha, shape, model, parameters)


def fit_fixed_head(
    matrix: np.ndarray,
    offset: np.ndarray,
    observed: np.ndarray,
    ridge: float,
    gamma: float | None,
    known_floor: float | None,
) -> mariner.Head:
    """Fit native amplitudes at a fixed training-local gamma or diagnostic floor."""
    if known_floor is None:
        assert gamma is not None
        return temporal.fit_head_with_offset(matrix, offset, observed, ridge, gamma)
    if not np.all(observed >= known_floor):
        raise ValueError("The diagnostic absolute floor must not exceed a training outcome")
    if not np.any(offset):
        return mariner.fit_head(matrix, observed, ridge, mariner.FloorSpec(known_floor, 0.0, 0.0))
    target = np.log(np.maximum(observed - known_floor, mariner.DEFICIT_FLOOR)) - offset
    intercept, coefficients = mariner.nonnegative_solve(matrix, target, ridge)
    return mariner.Head(float(intercept), coefficients, known_floor)


def candidate_oof(
    matrix: np.ndarray,
    offset: np.ndarray,
    observed: np.ndarray,
    labels: np.ndarray,
    ridge: float,
    gamma: float | None,
    *,
    known_floor: float | None = None,
) -> np.ndarray:
    """Predict each inner holdout using only that split's training outcomes."""
    predicted = np.empty(len(observed))
    for train, held in temporal.inner_splits(labels):
        head = fit_fixed_head(matrix[train], offset[train], observed[train], ridge, gamma, known_floor)
        predicted[held] = temporal.predict_head(head, matrix[held], offset[held])
    return predicted


@dataclass(frozen=True)
class JointFit:
    model: str
    alpha: float
    shape: dict[str, float]
    parameters: dict[str, float]
    ridge: float
    kappa: float | None
    known_floor: float | None
    inner_rmse: float
    head: mariner.Head
    selection_profile: tuple[dict[str, Any], ...]
    search_space: dict[str, Any]

    def predict(self, early: np.ndarray, late: np.ndarray) -> np.ndarray:
        matrix, offset = design_and_offset(early, late, self.alpha, self.shape, self.model, self.parameters)
        return temporal.predict_head(self.head, matrix, offset)

    def to_json(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "alpha": self.alpha,
            "shape": self.shape,
            "parameters": self.parameters,
            "ridge": self.ridge,
            "kappa": self.kappa,
            "known_floor": self.known_floor,
            "inner_rmse": self.inner_rmse,
            "head": {
                "intercept": self.head.intercept,
                "coefficients": self.head.coefficients.tolist(),
                "floor": self.head.floor,
            },
            "selection_profile": list(self.selection_profile),
            "search_space": self.search_space,
            "calibration_contract": (
                "known absolute floor; diagnostic only"
                if self.known_floor is not None
                else "asymmetric training median and minimum; training-local gamma floor"
            ),
            "selection_contract": (
                "joint discrete shape/kinetic/ridge search at known absolute floor; diagnostic only"
                if self.known_floor is not None
                else "joint discrete search, ridge-wise gamma profile, exhaustive conditional rescan"
            ),
        }

    @classmethod
    def from_json(cls, value: dict[str, Any]) -> JointFit:
        head = value["head"]
        return cls(
            model=value["model"],
            alpha=float(value["alpha"]),
            shape={key: float(number) for key, number in value["shape"].items()},
            parameters={key: float(number) for key, number in value["parameters"].items()},
            ridge=float(value["ridge"]),
            kappa=None if value["kappa"] is None else float(value["kappa"]),
            known_floor=None if value["known_floor"] is None else float(value["known_floor"]),
            inner_rmse=float(value["inner_rmse"]),
            head=mariner.Head(
                float(head["intercept"]), np.asarray(head["coefficients"], dtype=float), float(head["floor"])
            ),
            selection_profile=tuple(value["selection_profile"]),
            search_space=value["search_space"],
        )


def _fit(
    early: np.ndarray,
    late: np.ndarray,
    observed: np.ndarray,
    labels: np.ndarray,
    alpha: float,
    model: str,
    shapes: tuple[dict[str, float], ...] | None,
    test_grid: TestGrid | None,
    known_floor: float | None,
    matched_null_fit: JointFit | None,
) -> JointFit:
    if early.ndim != 2 or early.shape != late.shape or observed.shape != (len(early),) or labels.shape != observed.shape:
        raise ValueError("Training exposures, outcomes and inner labels must align")
    if not 0 < alpha < 1 or np.any(early < 0) or np.any(late < 0) or not np.isfinite(early + late).all():
        raise ValueError("Expected nonnegative finite exposures and an interior phase fraction")
    if not np.isfinite(observed).all() or temporal.tied_rows(early, late, alpha).any():
        raise ValueError("Only finite asymmetric training outcomes are admissible")
    temporal.inner_splits(labels)
    production_parameters = parameter_grid(model)
    shape_grid = tuple(dict(shape) for shape in (mariner.SHAPES if shapes is None else shapes))
    parameters = production_parameters if test_grid is None else test_grid.parameters
    ridges = mariner.RIDGE_GRID if test_grid is None else test_grid.ridges
    gammas = GAMMA_GRID if test_grid is None else test_grid.gammas
    if not shape_grid or not parameters or not ridges or not gammas:
        raise ValueError("All candidate menus must be nonempty")
    if any(parameters[0].values()):
        raise ValueError("The first kinetic setting must be the exact null")
    features: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
    for si, shape in enumerate(shape_grid):
        for pi, setting in enumerate(parameters):
            matrix, offset = design_and_offset(early, late, alpha, shape, model, setting)
            if model == "contrast_magnitude" and pi:
                matrix = features[si, 0][0]
            features[si, pi] = matrix, offset
    cache: dict[tuple[int, int, int, float | None], float] = {}
    profile: list[dict[str, Any]] = []
    refinements: list[dict[str, Any]] = []
    # A final occurrence index resolves duplicate evaluations only.
    best: tuple[float, int, int, int, float, int] | None = None

    def trial(si: int, pi: int, ri: int, gamma: float | None, stage: str, refinement_index: int | None = None) -> float:
        nonlocal best
        key = si, pi, ri, gamma
        cached = key in cache
        if not cached:
            matrix, offset = features[si, pi]
            predicted = candidate_oof(matrix, offset, observed, labels, ridges[ri], gamma, known_floor=known_floor)
            cache[key] = float(np.sqrt(np.mean(np.square(predicted - observed))))
        score = cache[key]
        index = len(profile)
        profile.append(
            {
                "stage": stage,
                "shape_index": si,
                "parameter_index": pi,
                "ridge_index": ri,
                "gamma": gamma,
                "inner_rmse": score if math.isfinite(score) else None,
                "cached": cached,
                "refinement_index": refinement_index,
            }
        )
        candidate = score, pi, si, ri, 0.0 if gamma is None else gamma, index
        if math.isfinite(score) and (best is None or candidate < best):
            best = candidate
        return score

    with threadpool_limits(limits=1):
        retained_null = None
        if matched_null_fit is not None:
            if matched_null_fit.model != "aggregate_mariner" or matched_null_fit.alpha != alpha:
                raise ValueError("The retained null must be the matching aggregate estimator")
            if matched_null_fit.known_floor != known_floor:
                raise ValueError("The retained null must use the same floor contract")
            si = shape_grid.index(matched_null_fit.shape)
            ri = ridges.index(matched_null_fit.ridge)
            null_score = trial(si, 0, ri, matched_null_fit.kappa, "retained_matched_null")
            retained_null = {
                "shape_index": si,
                "ridge_index": ri,
                "gamma": matched_null_fit.kappa,
                "rescored_inner_rmse": null_score,
                "supplied_inner_rmse": matched_null_fit.inner_rmse,
                "supplied_from_failed_refinement": matched_null_fit.search_space["selected_from_failed_refinement"],
            }
        for pi in range(len(parameters)):
            for si in range(len(shape_grid)):
                for ri in range(len(ridges)):
                    for gamma in (None,) if known_floor is not None else gammas:
                        trial(si, pi, ri, gamma, "known_floor" if known_floor is not None else "joint_grid")
        if best is None:
            raise ValueError("No finite joint candidate")
        if known_floor is None:
            _, pi, si, _, _, _ = best
            first_profile = len(profile)
            for ri in range(len(ridges)):
                refinement_index = len(refinements)
                first_probe = len(profile)

                def objective(log_gamma: float, ridge_index: int = ri, index: int = refinement_index) -> float:
                    score = trial(si, pi, ridge_index, math.exp(log_gamma), "gamma_refinement", index)
                    return score if math.isfinite(score) else mariner.INFINITE_CV_PENALTY

                result = minimize_scalar(
                    objective,
                    bounds=tuple(math.log(value) for value in mariner.KAPPA_BOUNDS),
                    method="bounded",
                    options={"maxiter": mariner.KAPPA_SEARCH_EVALUATIONS, "xatol": mariner.KAPPA_SEARCH_XATOL},
                )
                refinements.append(
                    {
                        "shape_index": si,
                        "parameter_index": pi,
                        "ridge_index": ri,
                        "success": bool(result.success),
                        "status": int(result.status),
                        "message": str(result.message),
                        "nfev": int(result.nfev),
                        "x": float(result.x),
                        "fun": float(result.fun),
                        "first_probe_index": first_probe,
                        "last_probe_index": len(profile) - 1,
                    }
                )
            refined = [item for item in profile[first_profile:] if item["inner_rmse"] is not None]
            if not refined:
                raise ValueError("No finite floor-refinement probe")
            refined_best = min(refined, key=lambda item: (item["inner_rmse"], item["ridge_index"], item["gamma"]))
            refined_gamma = refined_best["gamma"]
            for pi in range(len(parameters)):
                for si in range(len(shape_grid)):
                    for ri in range(len(ridges)):
                        trial(si, pi, ri, refined_gamma, "full_rescan", refined_best["refinement_index"])
        assert best is not None
        score, pi, si, ri, _, selected = best
        gamma = profile[selected]["gamma"]
        matrix, offset = features[si, pi]
        head = fit_fixed_head(matrix, offset, observed, ridges[ri], gamma, known_floor)
    return JointFit(
        model,
        alpha,
        shape_grid[si],
        dict(parameters[pi]),
        ridges[ri],
        gamma,
        known_floor,
        score,
        head,
        tuple(profile),
        {
            "shapes": list(shape_grid),
            "parameters": list(parameters),
            "ridges": list(ridges),
            "gammas": list(gammas) if known_floor is None else [],
            "test_grid_override": test_grid is not None,
            "selected_trial_index": selected,
            "unique_cv_evaluations": len(cache),
            "flat_profile_overwrite": False,
            "retained_matched_null": retained_null,
            "refinement_results": refinements,
            "selected_from_failed_refinement": (
                (
                    profile[selected]["refinement_index"] is not None
                    and not refinements[profile[selected]["refinement_index"]]["success"]
                )
                or (
                    profile[selected]["stage"] == "retained_matched_null"
                    and retained_null is not None
                    and retained_null["supplied_from_failed_refinement"]
                )
            ),
        },
    )


def fit_joint(
    early: np.ndarray,
    late: np.ndarray,
    observed: np.ndarray,
    labels: np.ndarray,
    alpha: float,
    model: str,
    *,
    shapes: tuple[dict[str, float], ...] | None = None,
    test_grid: TestGrid | None = None,
    matched_null_fit: JointFit | None = None,
) -> JointFit:
    """Fit J1 with explicit shape(s), or J2 with the complete native shape grid."""
    return _fit(early, late, observed, labels, alpha, model, shapes, test_grid, None, matched_null_fit)


def fit_known_floor(
    early: np.ndarray,
    late: np.ndarray,
    observed: np.ndarray,
    labels: np.ndarray,
    alpha: float,
    model: str,
    *,
    floor: float,
    shapes: tuple[dict[str, float], ...] | None = None,
    test_grid: TestGrid | None = None,
    matched_null_fit: JointFit | None = None,
) -> JointFit:
    """Diagnose shape/kinetic/ridge recovery with a supplied absolute floor."""
    return _fit(early, late, observed, labels, alpha, model, shapes, test_grid, floor, matched_null_fit)
