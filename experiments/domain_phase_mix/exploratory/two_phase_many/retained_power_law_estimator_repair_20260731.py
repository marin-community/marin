# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Identifiable, fully penalized estimator for the retained-power-law surrogate.

This module preserves the retained-power-law latent state, transition, and
response basis. It changes only the linear estimator:

* signed phase features use one unconstrained coefficient rather than two
  collinear nonnegative coefficients;
* phase features are scaled by their training-fold root-mean-square magnitude;
* every phase-control coefficient receives ridge shrinkage; and
* nested selection first enforces the frozen five-percent RMSE gate, then uses
  asymmetric Regret@1 and lower-tail error.

The repair is therefore an estimator ablation of retained power law, not a new
training-dynamics mechanism.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, replace

import numpy as np
from scipy.optimize import lsq_linear
from scipy.stats import spearmanr

from experiments.domain_phase_mix.exploratory.two_phase_many import (
    retained_power_law_model_20260728 as parent,
)

CORE_RMSE_RATIO_LIMIT = 1.05
REGRET_AT_1_SLACK = 0.002
LOWER_TAIL_FRACTION = 0.15
LOWER_TAIL_MIN_COUNT = 5
PHASE_PENALTY_MULTIPLIER = 1.0


@dataclass(frozen=True)
class SelectionContext:
    """Row identities needed by phase-sensitive nested model selection."""

    tied: np.ndarray
    pair_tied: np.ndarray
    pair_asymmetric: np.ndarray

    def __post_init__(self) -> None:
        tied = np.asarray(self.tied, dtype=bool)
        pair_tied = np.asarray(self.pair_tied, dtype=int)
        pair_asymmetric = np.asarray(self.pair_asymmetric, dtype=int)
        if pair_tied.shape != pair_asymmetric.shape:
            raise ValueError("paired tied and asymmetric indices must have the same shape")
        if np.any(pair_tied < 0) or np.any(pair_asymmetric < 0):
            raise ValueError("pair indices must be nonnegative")
        object.__setattr__(self, "tied", tied)
        object.__setattr__(self, "pair_tied", pair_tied)
        object.__setattr__(self, "pair_asymmetric", pair_asymmetric)


@dataclass(frozen=True)
class CandidateMetrics:
    """Nested OOF diagnostics for one shape and ridge candidate."""

    shape_index: int
    ridge_index: int
    ridge: float
    all_rmse: float
    all_spearman: float
    asymmetric_rmse: float
    asymmetric_regret_at_1: float
    asymmetric_lower_tail_rmse: float
    pair_delta_rmse: float


@dataclass(frozen=True)
class SelectionSummary:
    """Frozen selection decision and the thresholds that admitted it."""

    candidate_count: int
    selected_shape_index: int
    selected_ridge_index: int
    selected_metrics: CandidateMetrics
    minimum_all_rmse: float
    maximum_eligible_all_rmse: float
    minimum_eligible_regret_at_1: float
    maximum_eligible_regret_at_1: float
    rmse_eligible_count: int
    regret_eligible_count: int


@dataclass(frozen=True)
class FeatureLayout:
    """Column partition for constrained aggregate and signed phase features."""

    aggregate_count: int
    phase_count: int

    @property
    def total_count(self) -> int:
        return self.aggregate_count + self.phase_count


@dataclass(frozen=True)
class Fitted:
    """Retained-power-law fit with an identifiable mixed-sign response head."""

    shape: parent.Shape
    ridge: float
    intercept: float
    aggregate_coefficients: np.ndarray
    phase_coefficients: np.ndarray
    phase_blind: bool
    geometry: parent.Geometry
    selection: SelectionSummary

    @property
    def coefficients(self) -> np.ndarray:
        return np.concatenate([self.aggregate_coefficients, self.phase_coefficients])

    def predict(self, weights: np.ndarray) -> np.ndarray:
        build_design = phase_blind_design_matrix if self.phase_blind else design_matrix
        design, layout = build_design(weights, self.geometry, self.shape)
        if layout.aggregate_count != len(self.aggregate_coefficients):
            raise ValueError("aggregate coefficient count does not match the fitted design")
        if layout.phase_count != len(self.phase_coefficients):
            raise ValueError("phase coefficient count does not match the fitted design")
        return self.intercept + design @ self.coefficients


def _collapsed_phase_block(
    weights: np.ndarray,
    geometry: parent.Geometry,
    shape: parent.Shape,
) -> np.ndarray:
    concentration = parent.concentration_gap(weights, geometry)[:, None]
    if not shape.ordering_channel:
        return concentration

    aggregate = geometry.phase_0_fraction * weights[:, 0, :] + geometry.phase_1_fraction * weights[:, 1, :]
    contrast = weights[:, 1, :] - weights[:, 0, :]
    scale = aggregate + shape.benefit_offset
    epochs = parent.total_epochs(weights, geometry)
    benefit_slope = scale ** (-(shape.benefit_exponent + 1.0))
    damage_slope = np.maximum(epochs - shape.damage_threshold, 0.0) ** max(shape.damage_exponent - 1.0, 0.0)
    ordering_benefit = parent._family_totals(benefit_slope * contrast, geometry)
    ordering_damage = parent._family_totals(damage_slope * contrast, geometry)
    asymmetry = (scale ** (-(shape.benefit_exponent + 2.0)) * contrast**2).sum(axis=1, keepdims=True)
    return np.column_stack([concentration, ordering_benefit, ordering_damage, asymmetry])


def design_matrix(
    weights: np.ndarray,
    geometry: parent.Geometry,
    shape: parent.Shape,
) -> tuple[np.ndarray, FeatureLayout]:
    """Build the unchanged RPL response span without signed-column duplication."""

    retained = parent.retained_share(weights, geometry, shape.retention, shape.late_multiplier)
    benefit = (retained + shape.benefit_offset) ** (-shape.benefit_exponent)
    excess = np.maximum(parent.total_epochs(weights, geometry) - shape.damage_threshold, 0.0)
    aggregate = np.column_stack(
        [
            parent._hierarchical_block(benefit, geometry),
            parent._hierarchical_block(excess**shape.damage_exponent, geometry),
        ]
    )
    phase = _collapsed_phase_block(weights, geometry, shape)
    return np.column_stack([aggregate, phase]), FeatureLayout(
        aggregate_count=aggregate.shape[1],
        phase_count=phase.shape[1],
    )


def phase_blind_shape_grid(geometry: parent.Geometry) -> tuple[parent.Shape, ...]:
    """Return aggregate-only shapes spanning every full-RPL tied response.

    At a tied policy, full RPL's retained share is ``k * w`` with
    ``k = alpha_0 + late_multiplier * alpha_1``. The benefit amplitude absorbs
    ``k**-a``, while the equivalent aggregate-only offset is ``offset / k``.
    Keeping those transformed offsets prevents the phase ablation from also
    deleting aggregate-response shapes.
    """

    offsets = sorted(
        {
            offset / (geometry.phase_0_fraction + late_multiplier * geometry.phase_1_fraction)
            for offset in parent.BENEFIT_OFFSETS
            for late_multiplier in parent.LATE_MULTIPLIERS
        }
    )
    return tuple(
        parent.Shape(
            benefit_exponent=benefit_exponent,
            benefit_offset=benefit_offset,
            damage_exponent=damage_exponent,
            damage_threshold=damage_threshold,
            retention=0.0,
            late_multiplier=1.0,
            ordering_channel=False,
        )
        for benefit_exponent in parent.BENEFIT_EXPONENTS
        for benefit_offset in offsets
        for damage_exponent in parent.DAMAGE_EXPONENTS
        for damage_threshold in parent.DAMAGE_THRESHOLDS
    )


def phase_blind_design_matrix(
    weights: np.ndarray,
    geometry: parent.Geometry,
    shape: parent.Shape,
) -> tuple[np.ndarray, FeatureLayout]:
    """Build the aggregate-only RPL response at the policy's physical exposure."""

    if shape.retention != 0.0 or shape.late_multiplier != 1.0 or shape.ordering_channel:
        raise ValueError("phase-blind RPL requires retention=0, late_multiplier=1, and no ordering channel")
    design, layout = design_matrix(weights, geometry, shape)
    aggregate = design[:, : layout.aggregate_count]
    return aggregate, FeatureLayout(aggregate_count=layout.aggregate_count, phase_count=0)


def penalty_multipliers(geometry: parent.Geometry, layout: FeatureLayout) -> np.ndarray:
    """Shrink bucket departures and every phase-control coefficient."""

    families = len(np.unique(geometry.families))
    excess = len(geometry.excess_domains)
    aggregate_block = np.concatenate([np.zeros(families), np.ones(excess)])
    aggregate = np.concatenate([aggregate_block, aggregate_block])
    if len(aggregate) != layout.aggregate_count:
        raise ValueError("aggregate penalty layout does not match the response design")
    phase = np.full(layout.phase_count, PHASE_PENALTY_MULTIPLIER)
    return np.concatenate([aggregate, phase])


def feature_names(
    geometry: parent.Geometry,
    shape: parent.Shape,
    *,
    include_phase: bool = True,
) -> tuple[str, ...]:
    """Stable names for coefficient and sign-stability diagnostics."""

    families = tuple(int(value) for value in np.unique(geometry.families))
    excess = tuple(int(value) for value in geometry.excess_domains)
    aggregate = (
        tuple(f"benefit_family_{family}" for family in families)
        + tuple(f"benefit_bucket_departure_{domain}" for domain in excess)
        + tuple(f"damage_family_{family}" for family in families)
        + tuple(f"damage_bucket_departure_{domain}" for domain in excess)
    )
    phase: tuple[str, ...] = ("phase_concentration",) if include_phase else ()
    if include_phase and shape.ordering_channel:
        phase += (
            tuple(f"phase_ordering_benefit_family_{family}" for family in families)
            + tuple(f"phase_ordering_damage_family_{family}" for family in families)
            + ("phase_asymmetry",)
        )
    return aggregate + phase


def _column_scale(design: np.ndarray, layout: FeatureLayout) -> np.ndarray:
    aggregate = np.maximum(
        np.max(np.abs(design[:, : layout.aggregate_count]), axis=0),
        parent.COLUMN_SCALE_FLOOR,
    )
    phase = np.maximum(
        np.sqrt(np.mean(design[:, layout.aggregate_count :] ** 2, axis=0)),
        parent.COLUMN_SCALE_FLOOR,
    )
    return np.concatenate([aggregate, phase])


def _coefficient_bounds(layout: FeatureLayout) -> tuple[np.ndarray, np.ndarray]:
    lower = np.concatenate(
        [
            [-np.inf],
            np.zeros(layout.aggregate_count),
            np.full(layout.phase_count, -np.inf),
        ]
    )
    upper = np.full(1 + layout.total_count, np.inf)
    return lower, upper


def _bounded_solve(
    augmented: np.ndarray,
    response: np.ndarray,
    bounds: tuple[np.ndarray, np.ndarray],
    data_rows: int,
    row_weights: np.ndarray | None,
) -> np.ndarray:
    if row_weights is not None:
        if len(row_weights) != data_rows:
            raise ValueError("one robust weight is required per data row")
        root = np.sqrt(row_weights)[:, None]
        augmented = augmented.copy()
        response = response.copy()
        augmented[:data_rows] *= root
        response[:data_rows] *= root[:, 0]
    solved = lsq_linear(
        augmented,
        response,
        bounds=bounds,
        method="trf",
        tol=1e-10,
        max_iter=500,
    )
    predicted = augmented[:data_rows] @ solved.x
    if not np.all(np.isfinite(predicted)):
        raise RuntimeError("mixed-sign bounded solve produced non-finite predictions")
    limit = parent.PREDICTION_SCALE_LIMIT * max(float(np.max(np.abs(response[:data_rows]))), 1e-12)
    if np.max(np.abs(predicted)) > limit:
        raise RuntimeError("mixed-sign bounded solve is not identified at this ridge")
    return np.asarray(solved.x, dtype=float)


def solve_head(
    design: np.ndarray,
    target: np.ndarray,
    ridge: float,
    multipliers: np.ndarray,
    layout: FeatureLayout,
    huber_scale: float | None = parent.HUBER_SCALE,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Fit nonnegative aggregate amplitudes and signed, penalized phase controls.

    Passing ``huber_scale=None`` performs one bounded least-squares solve. It is
    used only to shortlist nonlinear shapes before robust rescoring.
    """

    if design.shape[1] != layout.total_count:
        raise ValueError("feature layout does not match the design")
    if len(multipliers) != layout.total_count:
        raise ValueError("one penalty multiplier is required per design column")
    scale = _column_scale(design, layout)
    augmented = np.column_stack([np.ones(len(target)), design / scale])
    penalty = np.diag(np.concatenate([[0.0], np.sqrt(ridge * multipliers)]))
    augmented = np.vstack([augmented, penalty])
    response = np.concatenate([target, np.zeros(penalty.shape[0])])
    bounds = _coefficient_bounds(layout)

    coefficients = _bounded_solve(augmented, response, bounds, len(target), None)
    if huber_scale is not None:
        for _ in range(parent.HUBER_ITERATIONS):
            residual = augmented[: len(target)] @ coefficients - target
            spread = parent.MAD_TO_SIGMA * float(np.median(np.abs(residual - np.median(residual))))
            if spread <= 0.0:
                break
            cut = huber_scale * spread
            row_weights = np.minimum(1.0, cut / np.maximum(np.abs(residual), 1e-12))
            updated = _bounded_solve(augmented, response, bounds, len(target), row_weights)
            shift = float(np.max(np.abs(augmented[: len(target)] @ (updated - coefficients))))
            coefficients = updated
            if shift < parent.HUBER_TOLERANCE * spread:
                break

    unscaled = coefficients[1:] / scale
    return (
        float(coefficients[0]),
        unscaled[: layout.aggregate_count],
        unscaled[layout.aggregate_count :],
    )


def _safe_spearman(observed: np.ndarray, predicted: np.ndarray) -> float:
    if len(observed) < 2 or np.std(observed) <= 0.0 or np.std(predicted) <= 0.0:
        return float("nan")
    return float(spearmanr(observed, predicted).statistic)


def _rmse(observed: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.sqrt(np.mean((predicted - observed) ** 2)))


def _lower_tail_rmse(observed: np.ndarray, predicted: np.ndarray, eligible: np.ndarray) -> float:
    indices = np.flatnonzero(eligible)
    if not len(indices):
        return _rmse(observed, predicted)
    count = min(
        len(indices),
        max(LOWER_TAIL_MIN_COUNT, math.ceil(LOWER_TAIL_FRACTION * len(indices))),
    )
    selected = indices[np.argsort(predicted[indices])[:count]]
    return _rmse(observed[selected], predicted[selected])


def _regret_at_1(
    observed: np.ndarray,
    predicted: np.ndarray,
    folds: Sequence[tuple[np.ndarray, np.ndarray]],
    eligible: np.ndarray,
) -> float:
    regrets = []
    for _train, test in folds:
        candidates = test[eligible[test]]
        if not len(candidates):
            continue
        selected = int(candidates[np.argmin(predicted[candidates])])
        regrets.append(float(observed[selected] - np.min(observed[candidates])))
    return float(np.mean(regrets)) if regrets else 0.0


def _pair_delta_rmse(
    observed: np.ndarray,
    predicted: np.ndarray,
    context: SelectionContext,
) -> float:
    if not len(context.pair_tied):
        return 0.0
    observed_delta = observed[context.pair_asymmetric] - observed[context.pair_tied]
    predicted_delta = predicted[context.pair_asymmetric] - predicted[context.pair_tied]
    return _rmse(observed_delta, predicted_delta)


def _candidate_metrics(
    observed: np.ndarray,
    predicted: np.ndarray,
    folds: Sequence[tuple[np.ndarray, np.ndarray]],
    context: SelectionContext,
    shape_index: int,
    ridge_index: int,
    ridge: float,
) -> CandidateMetrics:
    asymmetric = ~context.tied
    eligible = asymmetric if np.any(asymmetric) else np.ones(len(observed), dtype=bool)
    return CandidateMetrics(
        shape_index=shape_index,
        ridge_index=ridge_index,
        ridge=ridge,
        all_rmse=_rmse(observed, predicted),
        all_spearman=_safe_spearman(observed, predicted),
        asymmetric_rmse=_rmse(observed[eligible], predicted[eligible]),
        asymmetric_regret_at_1=_regret_at_1(observed, predicted, folds, eligible),
        asymmetric_lower_tail_rmse=_lower_tail_rmse(observed, predicted, eligible),
        pair_delta_rmse=_pair_delta_rmse(observed, predicted, context),
    )


def _shape_scores(
    weights: np.ndarray,
    target: np.ndarray,
    geometry: parent.Geometry,
    folds: tuple[tuple[np.ndarray, np.ndarray], ...],
    context: SelectionContext,
    indexed_shapes: tuple[tuple[int, parent.Shape], ...],
    phase_blind: bool,
) -> list[CandidateMetrics]:
    rows: list[CandidateMetrics] = []
    build_design = phase_blind_design_matrix if phase_blind else design_matrix
    for shape_index, shape in indexed_shapes:
        design, layout = build_design(weights, geometry, shape)
        if not np.all(np.isfinite(design)):
            continue
        multipliers = penalty_multipliers(geometry, layout)
        for ridge_index, ridge in enumerate(parent.RIDGE_GRID):
            predicted = np.full(len(target), np.nan, dtype=float)
            for train, test in folds:
                intercept, aggregate, phase = solve_head(
                    design[train],
                    target[train],
                    ridge,
                    multipliers,
                    layout,
                )
                predicted[test] = intercept + design[test] @ np.concatenate([aggregate, phase])
            if not np.isfinite(predicted).all():
                raise RuntimeError("candidate produced incomplete nested OOF predictions")
            rows.append(
                _candidate_metrics(
                    target,
                    predicted,
                    folds,
                    context,
                    shape_index,
                    ridge_index,
                    ridge,
                )
            )
    return rows


def _select_candidate(scores: Sequence[CandidateMetrics]) -> tuple[CandidateMetrics, SelectionSummary]:
    if not scores:
        raise RuntimeError("no finite repaired-head candidates were scored")
    minimum_rmse = min(row.all_rmse for row in scores)
    maximum_rmse = CORE_RMSE_RATIO_LIMIT * minimum_rmse
    rmse_eligible = [row for row in scores if row.all_rmse <= maximum_rmse]
    minimum_regret = min(row.asymmetric_regret_at_1 for row in rmse_eligible)
    maximum_regret = minimum_regret + REGRET_AT_1_SLACK
    regret_eligible = [row for row in rmse_eligible if row.asymmetric_regret_at_1 <= maximum_regret]
    selected = min(
        regret_eligible,
        key=lambda row: (
            row.asymmetric_lower_tail_rmse,
            row.pair_delta_rmse,
            row.asymmetric_rmse,
            row.all_rmse,
            row.shape_index,
            row.ridge_index,
        ),
    )
    summary = SelectionSummary(
        candidate_count=len(scores),
        selected_shape_index=selected.shape_index,
        selected_ridge_index=selected.ridge_index,
        selected_metrics=selected,
        minimum_all_rmse=minimum_rmse,
        maximum_eligible_all_rmse=maximum_rmse,
        minimum_eligible_regret_at_1=minimum_regret,
        maximum_eligible_regret_at_1=maximum_regret,
        rmse_eligible_count=len(rmse_eligible),
        regret_eligible_count=len(regret_eligible),
    )
    return selected, summary


def fit(
    weights: np.ndarray,
    target: np.ndarray,
    geometry: parent.Geometry,
    folds: tuple[tuple[np.ndarray, np.ndarray], ...],
    context: SelectionContext,
    workers: int = 1,
    *,
    phase_blind: bool = False,
) -> Fitted:
    """Select the repaired RPL head by frozen phase-sensitive nested criteria."""

    if workers < 1:
        raise ValueError("workers must be positive")
    if len(context.tied) != len(target):
        raise ValueError("selection context must cover every fitted row")
    shapes = phase_blind_shape_grid(geometry) if phase_blind else parent.shape_grid()
    indexed = tuple(enumerate(shapes))
    if workers == 1:
        scores = _shape_scores(weights, target, geometry, folds, context, indexed, phase_blind)
    else:
        worker_count = min(workers, len(indexed))
        batch_count = min(len(indexed), worker_count * 4)
        batch_size = (len(indexed) + batch_count - 1) // batch_count
        batches = tuple(indexed[start : start + batch_size] for start in range(0, len(indexed), batch_size))
        scores = []
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            futures = [
                executor.submit(
                    _shape_scores,
                    weights,
                    target,
                    geometry,
                    folds,
                    context,
                    batch,
                    phase_blind,
                )
                for batch in batches
            ]
            for future in as_completed(futures):
                scores.extend(future.result())

    selected, summary = _select_candidate(scores)
    shape = shapes[selected.shape_index]
    build_design = phase_blind_design_matrix if phase_blind else design_matrix
    design, layout = build_design(weights, geometry, shape)
    intercept, aggregate, phase = solve_head(
        design,
        target,
        selected.ridge,
        penalty_multipliers(geometry, layout),
        layout,
    )
    return Fitted(
        shape=shape,
        ridge=selected.ridge,
        intercept=intercept,
        aggregate_coefficients=aggregate,
        phase_coefficients=phase,
        phase_blind=phase_blind,
        geometry=geometry,
        selection=summary,
    )


def without_phase_terms(model: Fitted) -> Fitted:
    """Return the algebraically phase-blind restriction of a repaired RPL fit."""

    scale = model.geometry.phase_0_fraction + model.shape.late_multiplier * model.geometry.phase_1_fraction
    block_width = len(np.unique(model.geometry.families)) + len(model.geometry.excess_domains)
    aggregate = model.aggregate_coefficients.copy()
    aggregate[:block_width] *= scale ** (-model.shape.benefit_exponent)
    return replace(
        model,
        shape=replace(
            model.shape,
            benefit_offset=model.shape.benefit_offset / scale,
            retention=0.0,
            late_multiplier=1.0,
            ordering_channel=False,
        ),
        aggregate_coefficients=aggregate,
        phase_coefficients=np.zeros(0),
        phase_blind=True,
    )
