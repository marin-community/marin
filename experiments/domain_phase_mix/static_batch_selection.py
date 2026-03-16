# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Static batch selectors for StarCoder subset studies and nextgen design."""

from __future__ import annotations

import math
from dataclasses import dataclass
from collections.abc import Sequence
from typing import Literal

import numpy as np
import pandas as pd
from scipy.stats import qmc

from experiments.domain_phase_mix.config import WeightConfig
from experiments.domain_phase_mix.experiment import MixtureExperiment
from experiments.domain_phase_mix.exploratory.dsre_ceq_tools import (
    DsreCeqArtifacts,
    fit_dsre_ceq_artifacts,
)
from experiments.domain_phase_mix.exploratory.general_scaling_models import DatasetSpec
from experiments.domain_phase_mix.nextgen.contracts import LoopConfig, RunRecord
from experiments.domain_phase_mix.nextgen.dataset_metadata import resolve_dataset_epoch_metadata

DEFAULT_INFO_RIDGE = 1e-3
FEATURE_EPS = 1e-8
KERNEL_JITTER = 1e-6
GenericSelectionMethod = Literal["feature_maximin", "feature_dpp", "feature_bayes_linear"]


@dataclass(frozen=True)
class SelectionResult:
    """Indices and diagnostics for one selected batch."""

    selected_indices: list[int]
    info_logdet: float
    diagnostics: dict[str, float]


@dataclass(frozen=True)
class ReplayMatch:
    """Offline replay mapping from proposed schedules to observed runs."""

    selected_indices: list[int]
    mean_distance: float
    max_distance: float


@dataclass(frozen=True)
class ScheduleFeatureBundle:
    """Feature-space view of a schedule pool."""

    raw_matrix: np.ndarray
    standardized_matrix: np.ndarray
    feature_names: tuple[str, ...]
    distance_matrix: np.ndarray
    center_distances: np.ndarray


def run_records_to_dataframe(runs: Sequence[RunRecord]) -> pd.DataFrame:
    """Convert run records into a flat dataframe with phase columns and metrics."""
    rows: list[dict[str, float | str | int | None]] = []
    for run in runs:
        row: dict[str, float | str | int | None] = {
            "wandb_run_id": run.wandb_run_id,
            "source_experiment": run.source_experiment,
            "run_id": run.local_run_id,
            "run_name": run.run_name,
            "status": run.status,
        }
        for phase_name, domain_weights in run.phase_weights.items():
            for domain_name, weight in domain_weights.items():
                row[f"{phase_name}_{domain_name}"] = float(weight)
        for metric_name, value in run.metrics.items():
            row[metric_name] = float(value)
        rows.append(row)
    return pd.DataFrame(rows)


def build_dataset_spec_from_frame(
    df: pd.DataFrame,
    *,
    objective_metric: str,
    name: str,
    loop: LoopConfig | None = None,
) -> DatasetSpec:
    """Construct a DatasetSpec from a run dataframe using StarCoder metadata when available."""
    if objective_metric not in df.columns:
        raise ValueError(f"Objective metric '{objective_metric}' missing from dataframe")

    model_df = df[df[objective_metric].notna()].copy()
    if model_df.empty:
        raise ValueError(f"No rows with non-null objective metric '{objective_metric}'")

    phase_names = sorted(
        {
            col.split("_", 2)[0] + "_" + col.split("_", 2)[1]
            for col in model_df.columns
            if col.startswith("phase_") and not col.endswith("_epochs")
        }
    )
    if not phase_names:
        raise ValueError("No phase columns found in dataframe")

    first_phase = phase_names[0]
    domain_names = [
        col.replace(f"{first_phase}_", "")
        for col in model_df.columns
        if col.startswith(f"{first_phase}_") and not col.endswith("_epochs")
    ]
    weights = np.zeros((len(model_df), len(phase_names), len(domain_names)), dtype=float)
    for phase_idx, phase_name in enumerate(phase_names):
        for domain_idx, domain_name in enumerate(domain_names):
            weights[:, phase_idx, domain_idx] = model_df[f"{phase_name}_{domain_name}"].to_numpy(dtype=float)

    epoch_multipliers, small_domains = resolve_dataset_epoch_metadata(
        loop=loop,
        phase_names=phase_names,
        domain_names=domain_names,
    )

    return DatasetSpec(
        weights=weights,
        y=model_df[objective_metric].to_numpy(dtype=float),
        epoch_multipliers=epoch_multipliers,
        domain_names=domain_names,
        phase_names=phase_names,
        small_domains=small_domains,
        name=name,
    )


def _broadcast_epoch_multipliers(epoch_multipliers: np.ndarray, n_phases: int, n_domains: int) -> np.ndarray:
    arr = np.asarray(epoch_multipliers, dtype=float)
    if arr.ndim == 1:
        if arr.shape != (n_domains,):
            raise ValueError(f"epoch_multipliers shape {arr.shape} != ({n_domains},)")
        return np.tile(arr[None, :], (n_phases, 1))
    if arr.ndim == 2:
        if arr.shape != (n_phases, n_domains):
            raise ValueError(f"epoch_multipliers shape {arr.shape} != ({n_phases}, {n_domains})")
        return arr
    raise ValueError(f"epoch_multipliers must be 1D or 2D, got {arr.ndim}D")


def _normalized_entropy(values: np.ndarray, axis: int) -> np.ndarray:
    total = np.sum(values, axis=axis, keepdims=True)
    probs = np.divide(values, np.maximum(total, FEATURE_EPS), out=np.zeros_like(values), where=total > FEATURE_EPS)
    safe_probs = np.where(probs > FEATURE_EPS, probs, 1.0)
    entropy = -np.sum(np.where(probs > FEATURE_EPS, probs * np.log(safe_probs), 0.0), axis=axis)
    n_choices = values.shape[axis]
    if n_choices <= 1:
        return np.zeros(values.shape[:axis] + values.shape[axis + 1 :], dtype=float)
    return entropy / math.log(n_choices)


def _pairwise_sqdist(matrix: np.ndarray) -> np.ndarray:
    sq_norm = np.sum(matrix**2, axis=1, keepdims=True)
    sqdist = sq_norm + sq_norm.T - 2.0 * (matrix @ matrix.T)
    return np.maximum(sqdist, 0.0)


def average_phase_tv_distance(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Average total-variation distance across phases."""
    return np.abs(left - right).sum(axis=(1, 2)) / (2.0 * left.shape[1])


def pairwise_distance_matrix(weights: np.ndarray) -> np.ndarray:
    """Compute the average phase TV distance matrix for a pool."""
    n_points = weights.shape[0]
    distances = np.zeros((n_points, n_points), dtype=float)
    for idx in range(n_points):
        distances[idx] = average_phase_tv_distance(weights, weights[idx : idx + 1])
    return distances


def build_schedule_feature_matrix(
    weights: np.ndarray,
    epoch_multipliers: np.ndarray,
    small_domains: list[int] | None = None,
) -> ScheduleFeatureBundle:
    """Construct a fixed generic feature map for schedule selection."""
    del small_domains

    weights = np.asarray(weights, dtype=float)
    if weights.ndim != 3:
        raise ValueError(f"weights must have shape (R, N, M), got {weights.shape}")

    n_runs, n_phases, n_domains = weights.shape
    multipliers = _broadcast_epoch_multipliers(epoch_multipliers, n_phases, n_domains)
    epochs = weights * multipliers[None, :, :]
    phase_weight_deltas = np.diff(weights, axis=1) if n_phases > 1 else np.zeros((n_runs, 0, n_domains), dtype=float)
    phase_epoch_deltas = np.diff(epochs, axis=1) if n_phases > 1 else np.zeros((n_runs, 0, n_domains), dtype=float)

    raw_features = [
        weights.reshape(n_runs, -1),
        epochs.reshape(n_runs, -1),
        epochs.sum(axis=1),
        phase_weight_deltas.reshape(n_runs, -1),
        phase_epoch_deltas.reshape(n_runs, -1),
        _normalized_entropy(weights.transpose(0, 2, 1), axis=2),
        _normalized_entropy(weights, axis=2),
        weights.max(axis=1) - weights.min(axis=1),
        np.mean((weights <= 0.05) | (weights >= 0.95), axis=(1, 2))[:, None],
    ]
    feature_names = (
        [f"weight_p{phase_idx}_d{domain_idx}" for phase_idx in range(n_phases) for domain_idx in range(n_domains)]
        + [f"epoch_p{phase_idx}_d{domain_idx}" for phase_idx in range(n_phases) for domain_idx in range(n_domains)]
        + [f"total_epoch_d{domain_idx}" for domain_idx in range(n_domains)]
        + [
            f"delta_weight_p{phase_idx}_d{domain_idx}"
            for phase_idx in range(max(n_phases - 1, 0))
            for domain_idx in range(n_domains)
        ]
        + [
            f"delta_epoch_p{phase_idx}_d{domain_idx}"
            for phase_idx in range(max(n_phases - 1, 0))
            for domain_idx in range(n_domains)
        ]
        + [f"phase_entropy_d{domain_idx}" for domain_idx in range(n_domains)]
        + [f"domain_entropy_p{phase_idx}" for phase_idx in range(n_phases)]
        + [f"phase_range_d{domain_idx}" for domain_idx in range(n_domains)]
        + ["boundary_fraction"]
    )

    raw_matrix = np.column_stack(raw_features)
    mean = raw_matrix.mean(axis=0)
    std = raw_matrix.std(axis=0)
    safe_std = np.where(std > FEATURE_EPS, std, 1.0)
    standardized = (raw_matrix - mean) / safe_std
    sqdist = _pairwise_sqdist(standardized)
    distance_matrix = np.sqrt(sqdist)
    center = standardized.mean(axis=0, keepdims=True)
    center_distances = np.sqrt(np.sum((standardized - center) ** 2, axis=1))

    return ScheduleFeatureBundle(
        raw_matrix=raw_matrix,
        standardized_matrix=standardized,
        feature_names=tuple(feature_names),
        distance_matrix=distance_matrix,
        center_distances=center_distances,
    )


def greedy_k_center_indices(weights: np.ndarray, k: int, seed: int = 0) -> list[int]:
    """Select a diverse subset by farthest-first traversal."""
    n_points = weights.shape[0]
    if k <= 0 or n_points == 0:
        return []
    if k >= n_points:
        return list(range(n_points))

    rng = np.random.default_rng(seed)
    distances = pairwise_distance_matrix(weights)
    current = int(rng.integers(n_points))
    selected = [current]
    min_dist = distances[current].copy()
    min_dist[current] = -np.inf

    while len(selected) < k:
        candidate = int(np.argmax(min_dist))
        selected.append(candidate)
        min_dist = np.minimum(min_dist, distances[candidate])
        min_dist[selected] = -np.inf

    return selected


def _logdet(matrix: np.ndarray) -> float:
    sign, logdet = np.linalg.slogdet(matrix)
    if sign <= 0:
        return float("-inf")
    return float(logdet)


def _tie_break_candidate(
    candidate_indices: np.ndarray,
    *,
    selected: Sequence[int],
    min_distance: np.ndarray | None,
    center_distances: np.ndarray | None,
) -> int:
    if len(candidate_indices) == 0:
        if min_distance is not None:
            candidate_indices = np.flatnonzero(np.isfinite(min_distance))
        elif center_distances is not None:
            candidate_indices = np.arange(len(center_distances), dtype=int)
            if selected:
                selected_mask = np.zeros(len(center_distances), dtype=bool)
                selected_mask[np.asarray(selected, dtype=int)] = True
                candidate_indices = candidate_indices[~selected_mask[candidate_indices]]
        if len(candidate_indices) == 0:
            raise ValueError("No candidate indices available for tie-breaking")

    if len(candidate_indices) == 1:
        return int(candidate_indices[0])

    if selected and min_distance is not None:
        scores = min_distance[candidate_indices]
        finite_mask = np.isfinite(scores)
        if finite_mask.any():
            best_score = float(np.max(scores[finite_mask]))
            candidate_indices = candidate_indices[finite_mask & np.isclose(scores, best_score, atol=1e-12, rtol=1e-10)]
            if len(candidate_indices) == 1:
                return int(candidate_indices[0])

    if center_distances is not None:
        scores = center_distances[candidate_indices]
        finite_mask = np.isfinite(scores)
        if finite_mask.any():
            best_score = float(np.max(scores[finite_mask]))
            candidate_indices = candidate_indices[finite_mask & np.isclose(scores, best_score, atol=1e-12, rtol=1e-10)]
            if len(candidate_indices) == 1:
                return int(candidate_indices[0])

    return int(np.min(candidate_indices))


def _stable_cholesky(matrix: np.ndarray, *, jitter: float) -> np.ndarray | None:
    """Return a Cholesky factor after diagonal inflation, or `None` if all attempts fail."""
    if matrix.size == 0:
        return np.zeros((0, 0), dtype=float)

    eye = np.eye(matrix.shape[0], dtype=float)
    inflation = 0.0
    for _ in range(8):
        try:
            return np.linalg.cholesky(matrix + inflation * eye)
        except np.linalg.LinAlgError:
            inflation = jitter if inflation == 0.0 else inflation * 10.0
    return None


def _choose_remaining_maximin(
    bundle: ScheduleFeatureBundle,
    *,
    selected: Sequence[int],
    min_distance: np.ndarray,
) -> int:
    remaining = np.flatnonzero(np.isfinite(min_distance))
    if len(remaining) == 0:
        raise ValueError("No remaining candidates available for feature-space fallback")

    scores = min_distance[remaining]
    finite_mask = np.isfinite(scores)
    if finite_mask.any():
        best_score = float(np.max(scores[finite_mask]))
        candidate_indices = remaining[finite_mask & np.isclose(scores, best_score, atol=1e-12, rtol=1e-10)]
    else:
        candidate_indices = remaining
    return _tie_break_candidate(
        candidate_indices,
        selected=selected,
        min_distance=min_distance,
        center_distances=bundle.center_distances,
    )


def _compute_feature_subset_diagnostics(
    bundle: ScheduleFeatureBundle,
    selected_indices: Sequence[int],
) -> dict[str, float]:
    if not selected_indices:
        return {}

    subset_dist = bundle.distance_matrix[np.ix_(selected_indices, selected_indices)]
    nonzero = subset_dist[np.triu_indices(len(selected_indices), k=1)]
    return {
        "feature_mean_pairwise_distance": float(nonzero.mean()) if nonzero.size else 0.0,
        "feature_min_pairwise_distance": float(nonzero.min()) if nonzero.size else 0.0,
        "feature_mean_center_distance": float(bundle.center_distances[np.array(selected_indices, dtype=int)].mean()),
    }


def _finalize_selection_result(
    *,
    selected_indices: list[int],
    selector_score: float,
    bundle: ScheduleFeatureBundle | None,
    weights: np.ndarray | None,
    epoch_multipliers: np.ndarray | None,
    small_domain_idx: int | None,
    info_matrix: np.ndarray | None = None,
) -> SelectionResult:
    diagnostics: dict[str, float] = {"selector_score": float(selector_score)}
    if bundle is not None:
        diagnostics.update(_compute_feature_subset_diagnostics(bundle, selected_indices))
    if weights is not None:
        diagnostics.update(
            compute_subset_diagnostics(
                weights[selected_indices],
                full_pool_weights=weights,
                epoch_multipliers=epoch_multipliers,
                small_domain_idx=small_domain_idx,
                info_matrix=info_matrix,
            )
        )
    return SelectionResult(selected_indices=selected_indices, info_logdet=float(selector_score), diagnostics=diagnostics)


def _greedy_logdet_rows(
    rows: np.ndarray,
    *,
    k: int,
    ridge: float,
    distance_matrix: np.ndarray | None = None,
    center_distances: np.ndarray | None = None,
) -> tuple[list[int], np.ndarray]:
    n_points, n_params = rows.shape
    if k <= 0 or n_points == 0:
        return [], ridge * np.eye(n_params)
    if k >= n_points:
        return list(range(n_points)), ridge * np.eye(n_params) + rows.T @ rows

    precision = ridge * np.eye(n_params)
    precision_inv = np.eye(n_params) / ridge
    selected: list[int] = []
    min_distance = np.full(n_points, np.inf, dtype=float)

    while len(selected) < k:
        projected = rows @ precision_inv
        gains = np.einsum("ij,ij->i", projected, rows)
        gains[selected] = -np.inf
        best_gain = float(np.max(gains))
        candidate_indices = np.flatnonzero(np.isclose(gains, best_gain, rtol=1e-10, atol=1e-12))
        chosen = _tie_break_candidate(
            candidate_indices,
            selected=selected,
            min_distance=min_distance if distance_matrix is not None else None,
            center_distances=center_distances,
        )
        selected.append(chosen)

        if distance_matrix is not None:
            min_distance = np.minimum(min_distance, distance_matrix[chosen])
            min_distance[selected] = -np.inf

        row = rows[chosen]
        denom = 1.0 + float(row @ precision_inv @ row)
        update = (precision_inv @ np.outer(row, row) @ precision_inv) / denom
        precision += np.outer(row, row)
        precision_inv -= update

    return selected, precision


def greedy_d_optimal_indices(
    jacobian: np.ndarray,
    *,
    k: int,
    weights: np.ndarray | None = None,
    ridge: float = DEFAULT_INFO_RIDGE,
) -> SelectionResult:
    """Greedily maximize logdet(J^T J + ridge I) with distance tie-breaking."""
    distance_matrix = pairwise_distance_matrix(weights) if weights is not None else None
    selected, precision = _greedy_logdet_rows(
        jacobian,
        k=k,
        ridge=ridge,
        distance_matrix=distance_matrix,
    )
    diagnostics = (
        compute_subset_diagnostics(
            weights[selected],
            full_pool_weights=weights,
            info_matrix=precision,
        )
        if weights is not None and selected
        else {}
    )
    return SelectionResult(
        selected_indices=selected,
        info_logdet=_logdet(precision),
        diagnostics=diagnostics,
    )


def greedy_feature_maximin_indices(
    bundle: ScheduleFeatureBundle,
    *,
    k: int,
) -> SelectionResult:
    """Greedy maximin selection in standardized feature space."""
    n_points = bundle.standardized_matrix.shape[0]
    if k <= 0 or n_points == 0:
        return SelectionResult([], float("nan"), {})
    if k >= n_points:
        selected = list(range(n_points))
        subset_dist = bundle.distance_matrix[np.ix_(selected, selected)]
        nonzero = subset_dist[np.triu_indices(len(selected), k=1)]
        selector_score = float(nonzero.min()) if nonzero.size else 0.0
        return SelectionResult(selected, selector_score, _compute_feature_subset_diagnostics(bundle, selected))

    first = _tie_break_candidate(
        np.arange(n_points, dtype=int),
        selected=[],
        min_distance=None,
        center_distances=bundle.center_distances,
    )
    selected = [first]
    min_distance = bundle.distance_matrix[first].copy()
    min_distance[first] = -np.inf

    while len(selected) < k:
        best_gain = float(np.max(min_distance))
        candidate_indices = np.flatnonzero(np.isclose(min_distance, best_gain, rtol=1e-10, atol=1e-12))
        chosen = _tie_break_candidate(
            candidate_indices,
            selected=selected,
            min_distance=min_distance,
            center_distances=None,
        )
        selected.append(chosen)
        min_distance = np.minimum(min_distance, bundle.distance_matrix[chosen])
        min_distance[selected] = -np.inf

    subset_dist = bundle.distance_matrix[np.ix_(selected, selected)]
    nonzero = subset_dist[np.triu_indices(len(selected), k=1)]
    selector_score = float(nonzero.min()) if nonzero.size else 0.0
    return SelectionResult(selected, selector_score, _compute_feature_subset_diagnostics(bundle, selected))


def greedy_feature_dpp_indices(
    bundle: ScheduleFeatureBundle,
    *,
    k: int,
    jitter: float = KERNEL_JITTER,
) -> SelectionResult:
    """Deterministic greedy MAP k-DPP over an RBF kernel in standardized feature space."""
    n_points = bundle.standardized_matrix.shape[0]
    if k <= 0 or n_points == 0:
        return SelectionResult([], float("nan"), {})

    sqdist = _pairwise_sqdist(bundle.standardized_matrix)
    nonzero_sq = sqdist[sqdist > FEATURE_EPS]
    sigma_sq = float(np.median(nonzero_sq)) if nonzero_sq.size else 1.0
    kernel = np.exp(-sqdist / max(sigma_sq, FEATURE_EPS))
    kernel[np.diag_indices_from(kernel)] += jitter

    if k >= n_points:
        selected = list(range(n_points))
        score = _logdet(kernel)
        diagnostics = _compute_feature_subset_diagnostics(bundle, selected)
        diagnostics["rbf_sigma_sq"] = sigma_sq
        return SelectionResult(selected, score, diagnostics)

    selected: list[int] = []
    min_distance = np.full(n_points, np.inf, dtype=float)
    kernel_diag = np.diag(kernel).copy()
    fallback_steps = 0

    while len(selected) < k:
        remaining_mask = np.ones(n_points, dtype=bool)
        remaining_mask[selected] = False
        remaining = np.flatnonzero(remaining_mask)
        if len(remaining) == 0:
            break

        if not selected:
            gains = kernel_diag.copy()
        else:
            kernel_sel = kernel[np.ix_(selected, selected)]
            chol = _stable_cholesky(kernel_sel, jitter=jitter)
            if chol is None:
                chosen = _choose_remaining_maximin(bundle, selected=selected, min_distance=min_distance)
                fallback_steps += 1
                selected.append(chosen)
                min_distance = np.minimum(min_distance, bundle.distance_matrix[chosen])
                min_distance[selected] = -np.inf
                continue

            cross = kernel[np.ix_(remaining, selected)]
            solved = np.linalg.solve(chol, cross.T)
            gains = np.full(n_points, -np.inf, dtype=float)
            gains_remaining = kernel_diag[remaining] - np.sum(solved * solved, axis=0)
            gains_remaining = np.where(
                np.isfinite(gains_remaining),
                np.maximum(gains_remaining, 0.0),
                -np.inf,
            )
            gains[remaining] = gains_remaining
        gains[selected] = -np.inf

        remaining_gains = gains[remaining]
        finite_mask = np.isfinite(remaining_gains)
        if finite_mask.any():
            best_gain = float(np.max(remaining_gains[finite_mask]))
            candidate_indices = remaining[finite_mask & np.isclose(remaining_gains, best_gain, rtol=1e-10, atol=1e-12)]
        else:
            candidate_indices = np.array([], dtype=int)

        if len(candidate_indices) == 0:
            chosen = _choose_remaining_maximin(bundle, selected=selected, min_distance=min_distance)
            fallback_steps += 1
        else:
            chosen = _tie_break_candidate(
                candidate_indices,
                selected=selected,
                min_distance=min_distance,
                center_distances=bundle.center_distances,
            )

        selected.append(chosen)
        min_distance = np.minimum(min_distance, bundle.distance_matrix[chosen])
        min_distance[selected] = -np.inf

    score = _logdet(kernel[np.ix_(selected, selected)])
    diagnostics = _compute_feature_subset_diagnostics(bundle, selected)
    diagnostics["rbf_sigma_sq"] = sigma_sq
    diagnostics["dpp_fallback_steps"] = float(fallback_steps)
    return SelectionResult(selected, score, diagnostics)


def greedy_feature_bayes_linear_indices(
    bundle: ScheduleFeatureBundle,
    *,
    k: int,
    ridge: float = DEFAULT_INFO_RIDGE,
) -> SelectionResult:
    """Greedy Bayesian linear design over the generic feature map."""
    phi = np.column_stack([np.ones(len(bundle.standardized_matrix)), bundle.standardized_matrix])
    selected, precision = _greedy_logdet_rows(
        phi,
        k=k,
        ridge=ridge,
        distance_matrix=bundle.distance_matrix,
        center_distances=bundle.center_distances,
    )
    diagnostics = _compute_feature_subset_diagnostics(bundle, selected)
    diagnostics["feature_info_condition"] = float(np.linalg.cond(precision))
    return SelectionResult(selected, _logdet(precision), diagnostics)


def compute_subset_diagnostics(
    subset_weights: np.ndarray,
    *,
    full_pool_weights: np.ndarray | None = None,
    epoch_multipliers: np.ndarray | None = None,
    small_domain_idx: int | None = None,
    info_matrix: np.ndarray | None = None,
) -> dict[str, float]:
    """Compute descriptive statistics for one selected subset."""
    if subset_weights.size == 0:
        return {}

    subset_dist = pairwise_distance_matrix(subset_weights)
    nonzero = subset_dist[np.triu_indices(len(subset_weights), k=1)]
    diagnostics = {
        "subset_mean_pairwise_distance": float(nonzero.mean()) if nonzero.size else 0.0,
        "subset_min_pairwise_distance": float(nonzero.min()) if nonzero.size else 0.0,
        "boundary_mass_005": float(np.mean((subset_weights <= 0.05) | (subset_weights >= 0.95))),
    }

    if full_pool_weights is not None:
        coverage = np.min(
            pairwise_distance_matrix(np.concatenate([subset_weights, full_pool_weights], axis=0))[
                : len(subset_weights), len(subset_weights) :
            ],
            axis=0,
        )
        diagnostics["full_pool_mean_coverage_distance"] = float(np.mean(coverage))
        diagnostics["full_pool_max_coverage_distance"] = float(np.max(coverage))

    if epoch_multipliers is not None and small_domain_idx is not None:
        multipliers = _broadcast_epoch_multipliers(
            np.asarray(epoch_multipliers, dtype=float),
            subset_weights.shape[1],
            subset_weights.shape[2],
        )
        total_small_epochs = (subset_weights[:, :, small_domain_idx] * multipliers[:, small_domain_idx]).sum(axis=1)
        diagnostics["high_starcoder_epoch_fraction"] = float(
            np.mean(total_small_epochs >= np.quantile(total_small_epochs, 0.75))
        )

    if info_matrix is not None:
        diagnostics["info_logdet"] = _logdet(info_matrix)
        diagnostics["info_condition"] = float(np.linalg.cond(info_matrix))

    return diagnostics


def fit_anchor_artifacts(
    spec: DatasetSpec,
    *,
    seed: int = 0,
    n_restarts: int = 8,
    maxiter: int = 500,
) -> DsreCeqArtifacts:
    """Fit the DS-RE-CEQ anchor model used by the static selector."""
    return fit_dsre_ceq_artifacts(
        spec,
        seed=seed,
        n_restarts=n_restarts,
        maxiter=maxiter,
    )


def retrospective_d_optimal_selection(
    spec: DatasetSpec,
    *,
    k: int,
    seed: int = 0,
    anchor_artifacts: DsreCeqArtifacts | None = None,
    ridge: float = DEFAULT_INFO_RIDGE,
) -> SelectionResult:
    """Select observed rows directly from the run table."""
    anchor = anchor_artifacts or fit_anchor_artifacts(spec, seed=seed)
    jacobian = anchor.jacobian(spec.weights)
    selection = greedy_d_optimal_indices(
        jacobian,
        k=k,
        weights=spec.weights,
        ridge=ridge,
    )
    diagnostics = dict(selection.diagnostics)
    if spec.small_domains:
        diagnostics.update(
            compute_subset_diagnostics(
                spec.weights[selection.selected_indices],
                full_pool_weights=spec.weights,
                epoch_multipliers=np.asarray(spec.epoch_multipliers, dtype=float),
                small_domain_idx=spec.small_domains[0],
            )
        )
    return SelectionResult(selection.selected_indices, selection.info_logdet, diagnostics)


def retrospective_generic_selection(
    spec: DatasetSpec,
    *,
    method: GenericSelectionMethod,
    k: int,
    seed: int = 0,
    ridge: float = DEFAULT_INFO_RIDGE,
) -> SelectionResult:
    """Select observed rows using a generic feature-space selector."""
    del seed

    bundle = build_schedule_feature_matrix(spec.weights, spec.epoch_multipliers, spec.small_domains)
    if method == "feature_maximin":
        base = greedy_feature_maximin_indices(bundle, k=k)
    elif method == "feature_dpp":
        base = greedy_feature_dpp_indices(bundle, k=k)
    elif method == "feature_bayes_linear":
        base = greedy_feature_bayes_linear_indices(bundle, k=k, ridge=ridge)
    else:
        raise ValueError(f"Unknown generic selection method: {method}")

    info_matrix = None
    if method == "feature_bayes_linear":
        phi = np.column_stack([np.ones(len(bundle.standardized_matrix)), bundle.standardized_matrix])
        info_matrix = ridge * np.eye(phi.shape[1]) + phi[base.selected_indices].T @ phi[base.selected_indices]

    result = _finalize_selection_result(
        selected_indices=base.selected_indices,
        selector_score=base.info_logdet,
        bundle=bundle,
        weights=spec.weights,
        epoch_multipliers=np.asarray(spec.epoch_multipliers, dtype=float),
        small_domain_idx=spec.small_domains[0] if spec.small_domains else None,
        info_matrix=info_matrix,
    )
    diagnostics = dict(base.diagnostics)
    diagnostics.update(result.diagnostics)
    return SelectionResult(result.selected_indices, result.info_logdet, diagnostics)


def sobol_weight_configs(
    experiment: MixtureExperiment,
    *,
    n: int,
    seed: int,
    existing_configs: Sequence[WeightConfig] | None = None,
    oversample_factor: int = 4,
    min_accepted: int | None = None,
) -> list[WeightConfig]:
    """Generate quasi-random simplex schedules that respect experiment constraints.

    When `min_accepted` is provided, the generator may return fewer than `n`
    configs if the hard distance constraints make the full pool infeasible.
    """
    del oversample_factor

    sampler = experiment.create_weight_sampler(seed=seed)
    existing = list(existing_configs) if existing_configs else []
    accepted: list[WeightConfig] = []
    seen_hashes: set[str] = set()
    min_dist = sampler.params.min_config_distance
    total_dims = sampler.n_phases * sampler.n_domains
    sobol = qmc.Sobol(d=total_dims, scramble=True, seed=seed)
    attempts = 0
    max_attempts = max(n * sampler.SAMPLE_MULTIPLIER, n * 50)
    batch_size = 1 << max(3, math.ceil(math.log2(max_attempts)))
    raw_points = np.clip(sobol.random_base2(int(math.log2(batch_size))), 1e-9, 1.0 - 1e-9)
    cursor = 0

    while len(accepted) < n and attempts < max_attempts and cursor < len(raw_points):
        row = raw_points[cursor].reshape(sampler.n_phases, sampler.n_domains)
        cursor += 1
        phase_weights: dict[str, dict[str, float]] = {}
        for phase_idx, phase_name in enumerate(sampler.phase_names):
            phase_raw = -np.log(row[phase_idx])
            phase_simplex = phase_raw / phase_raw.sum()
            weights = dict(zip(sampler.domain_names, phase_simplex, strict=True))
            phase_weights[phase_name] = sampler._normalize(weights)

        candidate = WeightConfig(run_id=len(accepted), phase_weights=phase_weights)
        config_hash = sampler._config_hash(candidate, precision=4)
        if config_hash in seen_hashes:
            attempts += 1
            continue

        if min_dist > 0:
            distance_pool = [*existing, *accepted]
            if distance_pool:
                nearest = min(sampler._config_distance(candidate, prev) for prev in distance_pool)
                if nearest < min_dist:
                    attempts += 1
                    continue

        seen_hashes.add(config_hash)
        accepted.append(candidate)
        attempts += 1

    min_required = n if min_accepted is None else int(min_accepted)
    if min_required <= 0:
        raise ValueError(f"min_accepted must be positive when provided, got {min_accepted}")
    if len(accepted) < min_required:
        raise ValueError(
            f"Could only generate {len(accepted)}/{n} Sobol configs after {attempts} attempts "
            f"(required at least {min_required})"
        )

    return accepted


def weight_configs_to_tensor(
    configs: Sequence[WeightConfig],
    *,
    phase_names: Sequence[str],
    domain_names: Sequence[str],
) -> np.ndarray:
    """Convert weight configs to a dense tensor."""
    tensor = np.zeros((len(configs), len(phase_names), len(domain_names)), dtype=float)
    for run_idx, config in enumerate(configs):
        for phase_idx, phase_name in enumerate(phase_names):
            for domain_idx, domain_name in enumerate(domain_names):
                tensor[run_idx, phase_idx, domain_idx] = config.phase_weights[phase_name][domain_name]
    return tensor


def prospective_d_optimal_selection(
    spec: DatasetSpec,
    experiment: MixtureExperiment,
    *,
    n_select: int,
    seed: int,
    pool_size: int,
    existing_configs: Sequence[WeightConfig] | None = None,
    anchor_artifacts: DsreCeqArtifacts | None = None,
    ridge: float = DEFAULT_INFO_RIDGE,
) -> tuple[list[WeightConfig], SelectionResult]:
    """Select new schedules from a Sobol candidate pool using D-optimal gains."""
    anchor = anchor_artifacts or fit_anchor_artifacts(spec, seed=seed)
    pool_configs = sobol_weight_configs(
        experiment,
        n=pool_size,
        seed=seed,
        existing_configs=existing_configs,
        min_accepted=n_select,
    )
    pool_weights = weight_configs_to_tensor(
        pool_configs,
        phase_names=spec.phase_names,
        domain_names=spec.domain_names,
    )
    jacobian = anchor.jacobian(pool_weights)
    selection = greedy_d_optimal_indices(
        jacobian,
        k=n_select,
        weights=pool_weights,
        ridge=ridge,
    )
    diagnostics = dict(selection.diagnostics)
    diagnostics["candidate_pool_size"] = float(len(pool_configs))
    diagnostics["candidate_pool_requested"] = float(pool_size)
    if spec.small_domains:
        diagnostics.update(
            compute_subset_diagnostics(
                pool_weights[selection.selected_indices],
                full_pool_weights=pool_weights,
                epoch_multipliers=np.asarray(spec.epoch_multipliers, dtype=float),
                small_domain_idx=spec.small_domains[0],
            )
        )
    selection = SelectionResult(selection.selected_indices, selection.info_logdet, diagnostics)
    chosen = [pool_configs[idx] for idx in selection.selected_indices]
    return chosen, selection


def prospective_generic_selection(
    spec: DatasetSpec,
    experiment: MixtureExperiment,
    *,
    method: GenericSelectionMethod,
    n_select: int,
    seed: int,
    pool_size: int,
    existing_configs: Sequence[WeightConfig] | None = None,
    ridge: float = DEFAULT_INFO_RIDGE,
) -> tuple[list[WeightConfig], SelectionResult]:
    """Select new schedules from a Sobol candidate pool using a generic selector."""
    pool_configs = sobol_weight_configs(
        experiment,
        n=pool_size,
        seed=seed,
        existing_configs=existing_configs,
        min_accepted=n_select,
    )
    pool_weights = weight_configs_to_tensor(
        pool_configs,
        phase_names=spec.phase_names,
        domain_names=spec.domain_names,
    )
    pseudo_spec = DatasetSpec(
        weights=pool_weights,
        y=np.zeros(len(pool_weights), dtype=float),
        epoch_multipliers=np.asarray(spec.epoch_multipliers, dtype=float),
        domain_names=spec.domain_names,
        phase_names=spec.phase_names,
        small_domains=spec.small_domains,
        name=f"{spec.name}_prospective_pool",
    )
    selection = retrospective_generic_selection(
        pseudo_spec,
        method=method,
        k=n_select,
        seed=seed,
        ridge=ridge,
    )
    diagnostics = dict(selection.diagnostics)
    diagnostics["candidate_pool_size"] = float(len(pool_configs))
    diagnostics["candidate_pool_requested"] = float(pool_size)
    selection = SelectionResult(selection.selected_indices, selection.info_logdet, diagnostics)
    chosen = [pool_configs[idx] for idx in selection.selected_indices]
    return chosen, selection


def replay_proposals_to_observed(
    proposal_weights: np.ndarray,
    observed_weights: np.ndarray,
) -> ReplayMatch:
    """Map each proposal to the nearest unused observed run."""
    if len(proposal_weights) > len(observed_weights):
        raise ValueError("Cannot replay more proposals than observed runs")

    available = set(range(len(observed_weights)))
    selected: list[int] = []
    distances: list[float] = []

    for proposal in proposal_weights:
        order = np.argsort(average_phase_tv_distance(observed_weights, proposal[None, :, :]))
        for candidate_idx in order:
            idx = int(candidate_idx)
            if idx in available:
                available.remove(idx)
                selected.append(idx)
                distances.append(
                    float(
                        average_phase_tv_distance(
                            observed_weights[idx : idx + 1],
                            proposal[None, :, :],
                        )[0]
                    )
                )
                break
        else:
            raise ValueError("No unused observed run available for replay")

    return ReplayMatch(
        selected_indices=selected,
        mean_distance=float(np.mean(distances)) if distances else 0.0,
        max_distance=float(np.max(distances)) if distances else 0.0,
    )


def sampler_replay_selection(
    spec: DatasetSpec,
    experiment: MixtureExperiment,
    *,
    n_select: int,
    seed: int,
) -> ReplayMatch:
    """Replay the current experiment sampler onto the observed pool."""
    sampler = experiment.create_weight_sampler(seed=seed)
    proposals = sampler.sample_n_configs(n_select)
    proposal_weights = weight_configs_to_tensor(
        proposals,
        phase_names=spec.phase_names,
        domain_names=spec.domain_names,
    )
    return replay_proposals_to_observed(proposal_weights, spec.weights)
