# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pandas", "scipy", "scikit-learn", "threadpoolctl"]
# ///
"""Select one absolute floor by grouped held-out BPB error.

Every conditional head retains the native log-deficit loss, raw ridge and
nonnegative BVLS coefficients. The floor is common across the supplied inner
folds and the caller's final refit. Admissibility uses each conditional fit's
training labels only. This fixed-domain, sampled-basin search does not certify
a global optimum, identify a physical floor, or span every native fitted floor.
"""

import dataclasses
import functools
import math
from typing import Any

import numpy as np
import round3_log_head_20260921 as log_head
from scipy.optimize import minimize_scalar

mariner = log_head.mariner
FLOOR_BOUNDS = (-8.0, 8.0)
FLOOR_GRID = tuple(sorted((*FLOOR_BOUNDS, *((index + 0.5) / 4 for index in range(-32, 32)))))
TRANSITION_BISECTIONS = 20
REFINEMENT_SUBINTERVALS = 4
REFINEMENT_XATOL = 1e-8
REFINEMENT_MAXITER = 64


@dataclasses.dataclass(frozen=True)
class FoldTrial:
    fold_index: int
    training_rows: int
    validation_rows: int
    training_minimum: float
    admissible: bool
    deficit_clamp_count: int
    validation_sse: float | None


@dataclasses.dataclass(frozen=True)
class FloorTrial:
    floor: float
    inner_rmse: float | None
    folds: tuple[FoldTrial, ...]
    stage: str
    stage_index: int | None
    cached: bool


@dataclasses.dataclass(frozen=True)
class FloorTransition:
    coarse_lower: float
    coarse_upper: float
    final_finite: float
    final_inadmissible: float
    bisections: int


@dataclasses.dataclass(frozen=True)
class FloorRefinement:
    bracket_index: int
    subinterval_index: int
    lower_floor: float
    upper_floor: float
    floor: float
    inner_rmse: float
    success: bool
    status: int
    message: str
    evaluations: int


@dataclasses.dataclass(frozen=True)
class FloorCVResult:
    floor: float
    inner_rmse: float
    trace: tuple[FloorTrial, ...]
    refinements: tuple[FloorRefinement, ...]
    transitions: tuple[FloorTransition, ...]
    duplicate_subintervals_skipped: int

    def to_json(self) -> dict[str, Any]:
        """Return search metadata; inadmissible scores serialize as null."""
        finite_scores = [trial.inner_rmse for trial in self.trace if trial.inner_rmse is not None]
        return {
            "floor": self.floor,
            "inner_rmse": self.inner_rmse,
            "floor_bounds": list(FLOOR_BOUNDS),
            "coarse_grid": list(FLOOR_GRID),
            "refinement_subintervals": REFINEMENT_SUBINTERVALS,
            "duplicate_subintervals_skipped": self.duplicate_subintervals_skipped,
            "bound_hits": [
                name for name, bound in zip(("lower", "upper"), FLOOR_BOUNDS, strict=True) if self.floor == bound
            ],
            "finite_rmse_spread": max(finite_scores) - min(finite_scores),
            "unique_evaluations": sum(not trial.cached for trial in self.trace),
            "conditional_solves": sum(
                sum(fold.validation_sse is not None for fold in trial.folds) for trial in self.trace if not trial.cached
            ),
            "trace": [dataclasses.asdict(trial) for trial in self.trace],
            "refinements": [dataclasses.asdict(refinement) for refinement in self.refinements],
            "transitions": [dataclasses.asdict(transition) for transition in self.transitions],
            "search_claim": (
                "Fixed absolute domain and finite sampled basins; no global or floor-identification guarantee"
            ),
        }


def training_arrays(matrix: np.ndarray, response: np.ndarray, ridge: float) -> tuple[np.ndarray, np.ndarray]:
    matrix, response = np.asarray(matrix, dtype=float), np.asarray(response, dtype=float)
    if matrix.ndim != 2 or response.ndim != 1 or not len(response) or len(matrix) != len(response):
        raise ValueError("Features and BPB response need the same nonzero row count")
    if not np.isfinite(matrix).all() or not np.isfinite(response).all() or np.any(response <= 0):
        raise ValueError("Features and BPB must be finite, with observed BPB positive")
    if not math.isfinite(ridge) or ridge < 0:
        raise ValueError("Raw ridge must be finite and nonnegative")
    return matrix, response


def fixed_floor_head(matrix: np.ndarray, response: np.ndarray, ridge: float, floor: float) -> log_head.Head:
    """Fit the unchanged native conditional log head at an admissible floor.

    The fixed floor may be negative. Equality with the training minimum retains
    the native deficit clamp. This conditional API does not impose search bounds.
    """
    matrix, response = training_arrays(matrix, response, ridge)
    if not math.isfinite(floor) or floor > float(np.min(response)):
        raise ValueError("Absolute floor must be finite and no greater than this fit's training minimum")
    return log_head.fit_log_head(
        matrix,
        np.empty((len(response), 0)),
        response,
        ridge,
        mariner.FloorSpec(floor, 0.0, 0.0),
        np.ones(matrix.shape[1]),
        np.empty(0),
    )


def validated_folds(folds: mariner.Folds, rows: int) -> mariner.Folds:
    if not folds:
        raise ValueError("Absolute floor selection requires held-out folds")
    normalized = []
    for training, validation in folds:
        pair = (np.asarray(training), np.asarray(validation))
        for indices in pair:
            if indices.ndim != 1 or not len(indices) or not np.issubdtype(indices.dtype, np.integer):
                raise ValueError("Each fold needs nonempty one-dimensional integer row indices")
            if np.min(indices) < 0 or np.max(indices) >= rows or len(np.unique(indices)) != len(indices):
                raise ValueError("Fold indices must be unique and inside the actual fitting subset")
        if np.intersect1d(*pair).size:
            raise ValueError("A fold's training and validation rows must be disjoint")
        normalized.append(pair)
    return tuple(normalized)


def search_floor_cv(matrix: np.ndarray, response: np.ndarray, ridge: float, folds: mariner.Folds) -> FloorCVResult:
    """Select a shared absolute floor using pooled inner-validation BPB RMSE.

    Args:
        matrix: Native aggregate or unchanged shared-boundary feature matrix.
        response: Actual outer-training BPB labels, including inner validation.
        ridge: Fixed raw coefficient ridge for this configuration.
        folds: Training/validation indices relative to the supplied response.
    """
    matrix, response = training_arrays(matrix, response, ridge)
    folds = validated_folds(folds, len(response))
    cache: dict[float, tuple[float, tuple[FoldTrial, ...]]] = {}
    trace: list[FloorTrial] = []

    def evaluate(floor: float, stage: str, stage_index: int | None) -> float:
        floor = float(floor)
        if not math.isfinite(floor) or not FLOOR_BOUNDS[0] <= floor <= FLOOR_BOUNDS[1]:
            raise ValueError("Absolute floor search left its fixed registered domain")
        cached = floor in cache
        if not cached:
            local = tuple(
                FoldTrial(
                    index,
                    len(training),
                    len(validation),
                    float(np.min(response[training])),
                    floor <= float(np.min(response[training])),
                    int(np.count_nonzero(response[training] - floor < mariner.DEFICIT_FLOOR)),
                    None,
                )
                for index, (training, validation) in enumerate(folds)
            )
            score = float("inf")
            if all(fold.admissible for fold in local):
                scored = []
                for fold, (training, validation) in zip(local, folds, strict=True):
                    head = fixed_floor_head(matrix[training], response[training], ridge, floor)
                    prediction = head.predict(matrix[validation], np.empty((len(validation), 0)))
                    squared_error = float(np.sum((prediction - response[validation]) ** 2))
                    if not math.isfinite(squared_error):
                        raise FloatingPointError("A conditional head produced nonfinite validation BPB error")
                    scored.append(dataclasses.replace(fold, validation_sse=squared_error))
                local = tuple(scored)
                score = math.sqrt(
                    sum(fold.validation_sse for fold in local if fold.validation_sse is not None)
                    / sum(fold.validation_rows for fold in local)
                )
            cache[floor] = score, local
        score, local = cache[floor]
        trace.append(FloorTrial(floor, score if math.isfinite(score) else None, local, stage, stage_index, cached))
        return score

    coarse = [evaluate(floor, "coarse", None) for floor in FLOOR_GRID]
    transitions = []
    transition_bounds: dict[tuple[float, float], float] = {}
    for index in range(len(FLOOR_GRID) - 1):
        if math.isfinite(coarse[index]) and not math.isfinite(coarse[index + 1]):
            lower, upper = FLOOR_GRID[index : index + 2]
            finite, invalid = lower, upper
            for _ in range(TRANSITION_BISECTIONS):
                middle = (finite + invalid) / 2
                if math.isfinite(evaluate(middle, "transition", len(transitions))):
                    finite = middle
                else:
                    invalid = middle
            transitions.append(FloorTransition(lower, upper, finite, invalid, TRANSITION_BISECTIONS))
            transition_bounds[lower, upper] = finite

    brackets = []
    for index in range(1, len(FLOOR_GRID) - 1):
        if math.isfinite(coarse[index]) and coarse[index] <= coarse[index - 1] and coarse[index] <= coarse[index + 1]:
            lower, upper = FLOOR_GRID[index - 1], FLOOR_GRID[index + 1]
            if not math.isfinite(coarse[index + 1]):
                upper = transition_bounds[FLOOR_GRID[index], upper]
            brackets.append((lower, upper))
    for left, right in ((0, 1), (len(FLOOR_GRID) - 2, len(FLOOR_GRID) - 1)):
        if math.isfinite(coarse[left]) and math.isfinite(coarse[right]):
            brackets.append((FLOOR_GRID[left], FLOOR_GRID[right]))
    brackets.extend((transition.coarse_lower, transition.final_finite) for transition in transitions)

    refinements = []
    seen_subintervals: set[tuple[float, float]] = set()
    duplicate_subintervals = 0
    for bracket_index, (lower, upper) in enumerate(dict.fromkeys(brackets)):
        if lower == upper:
            continue
        endpoints = np.linspace(lower, upper, REFINEMENT_SUBINTERVALS + 1)
        for endpoint in endpoints:
            evaluate(float(endpoint), "refinement_endpoint", bracket_index)
        for subinterval_index in range(REFINEMENT_SUBINTERVALS):
            start, stop = float(endpoints[subinterval_index]), float(endpoints[subinterval_index + 1])
            if (start, stop) in seen_subintervals:
                duplicate_subintervals += 1
                continue
            seen_subintervals.add((start, stop))
            result = minimize_scalar(
                functools.partial(evaluate, stage="refinement", stage_index=len(refinements)),
                bounds=(start, stop),
                method="bounded",
                options={"xatol": REFINEMENT_XATOL, "maxiter": REFINEMENT_MAXITER},
            )
            refinements.append(
                FloorRefinement(
                    bracket_index,
                    subinterval_index,
                    start,
                    stop,
                    float(result.x),
                    float(result.fun),
                    bool(result.success),
                    int(result.status),
                    str(result.message),
                    int(result.nfev),
                )
            )
    floor = min(cache, key=lambda value: (cache[value][0], value))
    score = cache[floor][0]
    if not math.isfinite(score):
        raise ValueError("No admissible finite absolute floor was evaluated")
    return FloorCVResult(floor, score, tuple(trace), tuple(refinements), tuple(transitions), duplicate_subintervals)
