# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pandas", "scipy", "threadpoolctl"]
# ///
"""Compact temporal extensions conditional on native MARINER aggregate selection.

Only the arrays passed to fitting are available to the estimator. Callers must
pass asymmetric training rows and outcome-independent inner-fold labels. Shape,
ridge and floor kappa use the standalone's complete native search. Extensions
reuse those selected hyperparameters, then refit amplitudes and training-local
floors for every kinetic candidate and inner partition. Thus the kinetic inner
score is a conditional selection criterion, not an independent performance
estimate; complete outer folds provide the latter.
"""

from __future__ import annotations

import itertools
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[4].parent / "mixture-selection"))

import mixture_selection as mariner

MODEL_NAMES = (
    "aggregate_mariner",
    "shared_clock",
    "benefit_clock",
    "benefit_persistence",
    "harm_persistence",
    "joint_persistence",
    "pool_order",
)
CLOCK_GRID = (1.0, 0.25, 0.5, 2.0, 4.0, 8.0, 16.0)
PERSISTENCE_GRID = (0.0, 0.5, 1.0)
ORDER_GRID = (0.0, -0.03, 0.03, -0.1, 0.1, -0.3, 0.3, -1.0, 1.0)
TIED_ROUNDOFF_MULTIPLIER = 64.0


def tied_rows(early: np.ndarray, late: np.ndarray, alpha: float) -> np.ndarray:
    """Recognize exact tied policies up to exposure arithmetic roundoff."""
    total = early + late
    tolerance = TIED_ROUNDOFF_MULTIPLIER * np.finfo(float).eps * np.maximum(1.0, total)
    return np.all(np.abs(early - alpha * total) <= tolerance, axis=1)


def effective_exposure(early: np.ndarray, late: np.ndarray, alpha: float, multiplier: float) -> np.ndarray:
    """Normalized shared clock, equal to the exposure of a static convex mixture."""
    total = early + late
    deviation = early - alpha * total
    deviation[tied_rows(early, late, alpha)] = 0.0
    denominator = alpha + multiplier * (1.0 - alpha)
    return np.maximum(total + (1.0 - multiplier) * deviation / denominator, 0.0)


def pool_order_feature(
    early: np.ndarray,
    late: np.ndarray,
    alpha: float,
    shape: dict[str, float],
    large_pool: np.ndarray,
) -> np.ndarray:
    """Centered acquisition-order wedge between fixed large- and small-pool axes.

    These label-free axes describe pool size, not an established prerequisite
    relation. Saturating acquisition uses the aggregate-selected MARINER shape.
    """
    total = early + late
    rate, power = shape["rate"], shape["power"]
    whole = mariner.benefit(total, rate, power)
    first = mariner.benefit(early, rate, power)
    tied_first = mariner.benefit(alpha * total, rate, power)
    early_delta = first - tied_first
    broad = whole[:, large_pool].mean(axis=1)
    specific = whole[:, ~large_pool].mean(axis=1)
    value = early_delta[:, large_pool].mean(axis=1) * specific
    value -= early_delta[:, ~large_pool].mean(axis=1) * broad
    value[tied_rows(early, late, alpha)] = 0.0
    return value


def design_and_offset(
    early: np.ndarray,
    late: np.ndarray,
    alpha: float,
    shape: dict[str, float],
    model: str,
    parameters: dict[str, float],
    large_pool: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return native-sign benefit/harm features and a possible log-scale offset."""
    total = early + late
    base = mariner.design_matrix(total, shape)
    offset = np.zeros(len(total))
    if model == "aggregate_mariner":
        return base, offset
    if model in ("shared_clock", "benefit_clock"):
        effective = effective_exposure(early, late, alpha, parameters["multiplier"])
        matrix = mariner.design_matrix(effective, shape)
        if model == "benefit_clock":
            matrix[:, total.shape[1] :] = base[:, total.shape[1] :]
    elif model in ("benefit_persistence", "harm_persistence", "joint_persistence"):
        reference = mariner.design_matrix(alpha * total, shape)
        actual_early = mariner.design_matrix(early, shape)
        correction = reference - actual_early
        correction[:, : total.shape[1]] *= parameters["benefit_loss"]
        correction[:, total.shape[1] :] *= parameters["harm_loss"]
        matrix = base + correction
    elif model == "pool_order":
        matrix = base.copy()
        offset = parameters["order_coefficient"] * pool_order_feature(early, late, alpha, shape, large_pool)
    else:
        raise ValueError(f"Unknown temporal model: {model}")
    # Delegating tied rows to the exact native evaluator avoids magnifying
    # arithmetic-scale cancellation through the exponential response link.
    tied = tied_rows(early, late, alpha)
    matrix[tied] = base[tied]
    offset[tied] = 0.0
    return matrix, offset


def parameter_grid(model: str) -> tuple[dict[str, float], ...]:
    """Enumerate fixed kinetics, with the zero-phase member first."""
    if model == "aggregate_mariner":
        return ({},)
    if model in ("shared_clock", "benefit_clock"):
        return tuple({"multiplier": value} for value in CLOCK_GRID)
    if model == "benefit_persistence":
        return tuple({"benefit_loss": value, "harm_loss": 0.0} for value in PERSISTENCE_GRID)
    if model == "harm_persistence":
        return tuple({"benefit_loss": 0.0, "harm_loss": value} for value in PERSISTENCE_GRID)
    if model == "joint_persistence":
        return tuple(
            {"benefit_loss": benefit, "harm_loss": harm}
            for benefit, harm in itertools.product(PERSISTENCE_GRID, repeat=2)
        )
    if model == "pool_order":
        return tuple({"order_coefficient": value} for value in ORDER_GRID)
    raise ValueError(f"Unknown temporal model: {model}")


def fit_head_with_offset(
    matrix: np.ndarray, offset: np.ndarray, observed: np.ndarray, ridge: float, kappa: float
) -> mariner.Head:
    """Fit native log-deficit coefficients after subtracting a specified offset."""
    spec = mariner.FloorSpec(float("nan"), 0.0, kappa)
    if not np.any(offset):
        return mariner.fit_head(matrix, observed, ridge, spec)
    floor = mariner.floor_value(observed, spec)
    target = np.log(np.maximum(observed - floor, mariner.DEFICIT_FLOOR)) - offset
    intercept, coefficients = mariner.nonnegative_solve(matrix, target, ridge)
    return mariner.Head(intercept, coefficients, floor)


def predict_head(head: mariner.Head, matrix: np.ndarray, offset: np.ndarray) -> np.ndarray:
    """Evaluate the native response link with the selected temporal offset."""
    if not np.any(offset):
        return head.predict(matrix)
    linear = head.intercept + matrix @ head.coefficients + offset
    return head.floor + np.exp(np.clip(linear, -mariner.LOG_CLIP, mariner.LOG_CLIP))


@dataclass(frozen=True)
class TemporalFit:
    model: str
    alpha: float
    large_pool: np.ndarray
    shape: dict[str, float]
    ridge: float
    kappa: float
    flat_profile: bool
    aggregate_inner_rmse: float
    conditional_inner_rmse: float
    parameters: dict[str, float]
    head: mariner.Head
    selection_profile: tuple[dict[str, Any], ...]

    def predict(self, early: np.ndarray, late: np.ndarray) -> np.ndarray:
        matrix, offset = design_and_offset(
            early, late, self.alpha, self.shape, self.model, self.parameters, self.large_pool
        )
        return predict_head(self.head, matrix, offset)

    def to_json(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "alpha": self.alpha,
            "large_pool": self.large_pool.tolist(),
            "shape": self.shape,
            "ridge": self.ridge,
            "kappa": self.kappa,
            "flat_profile": self.flat_profile,
            "aggregate_inner_rmse": self.aggregate_inner_rmse,
            "conditional_inner_rmse": self.conditional_inner_rmse,
            "parameters": self.parameters,
            "head": {
                "intercept": self.head.intercept,
                "coefficients": self.head.coefficients.tolist(),
                "floor": self.head.floor,
            },
            "selection_profile": list(self.selection_profile),
            "calibration_contract": "asymmetric training median; no tied anchor or repeat noise",
            "selection_contract": "native aggregate search then conditional kinetics; inner amplitudes and floors refit",
        }

    @classmethod
    def from_json(cls, value: dict[str, Any]) -> TemporalFit:
        head = value["head"]
        return cls(
            model=value["model"],
            alpha=float(value["alpha"]),
            large_pool=np.asarray(value["large_pool"], dtype=bool),
            shape={key: float(number) for key, number in value["shape"].items()},
            ridge=float(value["ridge"]),
            kappa=float(value["kappa"]),
            flat_profile=bool(value["flat_profile"]),
            aggregate_inner_rmse=float(value["aggregate_inner_rmse"]),
            conditional_inner_rmse=float(value["conditional_inner_rmse"]),
            parameters={key: float(number) for key, number in value["parameters"].items()},
            head=mariner.Head(float(head["intercept"]), np.asarray(head["coefficients"]), float(head["floor"])),
            selection_profile=tuple(value["selection_profile"]),
        )


def inner_splits(labels: np.ndarray) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    labels = np.asarray(labels)
    unique = np.unique(labels)
    if len(unique) < 2 or np.any(labels < 0):
        raise ValueError("At least two nonnegative inner folds are required")
    return tuple((np.flatnonzero(labels != fold), np.flatnonzero(labels == fold)) for fold in unique)


def validate_training(
    early: np.ndarray, late: np.ndarray, observed: np.ndarray, labels: np.ndarray, alpha: float, pool_sizes: np.ndarray
) -> np.ndarray:
    if early.ndim != 2 or early.shape != late.shape or early.shape[0] != len(observed) or len(labels) != len(observed):
        raise ValueError("Training exposures, outcomes and folds must align")
    if pool_sizes.shape != (early.shape[1],) or not np.isfinite(pool_sizes).all() or np.any(pool_sizes <= 0):
        raise ValueError("One finite positive pool size is required per bucket")
    if not 0 < alpha < 1 or not np.isfinite(early).all() or not np.isfinite(late).all():
        raise ValueError("Training exposures and phase fraction must be finite")
    if np.any(early < 0) or np.any(late < 0) or not np.isfinite(observed).all():
        raise ValueError("Training exposures must be nonnegative and outcomes finite")
    if tied_rows(early, late, alpha).any():
        raise ValueError("Tied outcomes are evaluation-only and cannot enter fitting")
    large = np.log(pool_sizes) >= np.median(np.log(pool_sizes))
    if not large.any() or large.all():
        raise ValueError("Pool-size order needs both a large and small pool group")
    return large


def fit_aggregate(
    early: np.ndarray,
    late: np.ndarray,
    observed: np.ndarray,
    labels: np.ndarray,
    alpha: float,
    pool_sizes: np.ndarray,
) -> TemporalFit:
    """Run the complete standalone MARINER search on asymmetric aggregate exposures."""
    large = validate_training(early, late, observed, labels, alpha, pool_sizes)
    splits = inner_splits(labels)
    exposures = early + late
    anchor = mariner.Anchor(float("nan"), 0.0)
    provisional = mariner.FloorSpec(float("nan"), 0.0, mariner.KAPPA_PROVISIONAL)
    with threadpool_limits(limits=1):
        shape, ridge, _ = mariner.select_shape_and_ridge(exposures, observed, provisional, splits)
        matrix = mariner.design_matrix(exposures, shape)
        kappa, flat, score = mariner.search_kappa(matrix, observed, ridge, anchor, splits)
        head = mariner.fit_head(matrix, observed, ridge, mariner.FloorSpec(float("nan"), 0.0, kappa))
    return TemporalFit("aggregate_mariner", alpha, large, shape, ridge, kappa, flat, score, score, {}, head, ())


def fit_extension(
    early: np.ndarray,
    late: np.ndarray,
    observed: np.ndarray,
    labels: np.ndarray,
    aggregate: TemporalFit,
    model: str,
) -> TemporalFit:
    """Select compact kinetics by conditional CV with freshly fitted native heads."""
    if model == "aggregate_mariner":
        return aggregate
    splits = inner_splits(labels)
    profile = []
    best = None
    with threadpool_limits(limits=1):
        for index, parameters in enumerate(parameter_grid(model)):
            matrix, offset = design_and_offset(
                early, late, aggregate.alpha, aggregate.shape, model, parameters, aggregate.large_pool
            )
            squared_error = 0.0
            count = 0
            for train, held in splits:
                head = fit_head_with_offset(
                    matrix[train], offset[train], observed[train], aggregate.ridge, aggregate.kappa
                )
                predicted = predict_head(head, matrix[held], offset[held])
                squared_error += float(np.sum(np.square(predicted - observed[held])))
                count += len(held)
            score = math.sqrt(squared_error / count)
            profile.append({"parameters": parameters, "conditional_inner_rmse": score})
            candidate = (score, index)
            if best is None or candidate < best:
                best = candidate
        assert best is not None
        if not math.isfinite(best[0]):
            raise ValueError(f"No finite conditional fit for {model}")
        parameters = parameter_grid(model)[best[1]]
        matrix, offset = design_and_offset(
            early, late, aggregate.alpha, aggregate.shape, model, parameters, aggregate.large_pool
        )
        head = fit_head_with_offset(matrix, offset, observed, aggregate.ridge, aggregate.kappa)
    return TemporalFit(
        model,
        aggregate.alpha,
        aggregate.large_pool,
        aggregate.shape,
        aggregate.ridge,
        aggregate.kappa,
        aggregate.flat_profile,
        aggregate.aggregate_inner_rmse,
        best[0],
        parameters,
        head,
        tuple(profile),
    )


def fit_bundle(
    early: np.ndarray,
    late: np.ndarray,
    observed: np.ndarray,
    labels: np.ndarray,
    alpha: float,
    pool_sizes: np.ndarray,
    models: tuple[str, ...] = MODEL_NAMES,
) -> dict[str, TemporalFit]:
    """Share one native aggregate search across the registered temporal controls."""
    aggregate = fit_aggregate(early, late, observed, labels, alpha, pool_sizes)
    return {name: fit_extension(early, late, observed, labels, aggregate, name) for name in models}
