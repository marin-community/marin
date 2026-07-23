# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "cvxpy",
#   "fsspec",
#   "gcsfs",
#   "numpy",
#   "pandas",
#   "plotly",
#   "scikit-learn",
#   "scipy",
#   "tabulate",
# ]
# ///
"""Fit HPR with a cross-fitted same-seed random-effect likelihood.

The response surface is unchanged. Exact aggregate-matched one-phase and
two-phase checkpoints that reuse a data seed are treated as a two-observation
block with a shared seed random effect. The block covariance is estimated only
from nested grouped-OOF residuals on the selected training panel.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import dataclass, replace
from enum import StrEnum
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.optimize import lsq_linear

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    analyze_delphi_3e18_fixed_budget_frontier_composition as composition,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_heterogeneous_design_aware_hpr_20260719 as fitting,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_hierarchical_coverage_grp_20260715 as hierarchical,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_matched_pair_heterogeneous_hpr_20260720 as matched_fit,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    fit_production_grp_quality_variants as family_grp,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.mechanistic_surrogate_discovery_20260717 import (  # noqa: E402
    audit_raw_optima as raw_optima,
)

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "reference_outputs/paired_random_effects_hpr_20260720"
PREREGISTRATION_PATH = DEFAULT_OUTPUT_DIR / "preregistered_candidates.json"
ALLOCATION_NAMES = ("p140", "t42_p119")
INNER_FOLDS = 4
COUPLING_GRID = fitting.COUPLING_GRID
PLOT_CONFIG = {"toImageButtonOptions": {"format": "png", "scale": 4}}


class CovarianceMode(StrEnum):
    IDENTITY = "identity"
    DIAGONAL = "diagonal"
    RANDOM_EFFECTS = "random_effects"


class Candidate(StrEnum):
    POOLED_IDENTITY = "pooled_identity"
    PAIRED_DIAGONAL_SHARED = "paired_diagonal_shared"
    PAIRED_RANDOM_EFFECTS_SHARED = "paired_random_effects_shared"
    PAIRED_RANDOM_EFFECTS_PARTIAL_PHASE = "paired_random_effects_partial_phase"


@dataclass(frozen=True)
class CovarianceEstimate:
    mode: CovarianceMode
    matrix: np.ndarray
    aggregate_variance: float
    phase_variance: float
    shared_variance: float
    correlation: float
    pair_count: int


@dataclass(frozen=True)
class OOFResult:
    prediction: np.ndarray
    aggregate_coefficients: tuple[np.ndarray, ...]
    phase_coefficients: tuple[np.ndarray, ...]
    covariance: tuple[CovarianceEstimate, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seeds", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--allocations", default=",".join(ALLOCATION_NAMES))
    parser.add_argument("--with-optima", action="store_true")
    parser.add_argument("--optimizer-starts", type=int, default=6)
    return parser.parse_args()


def selected_allocations(raw: str) -> tuple[matched_fit.PairAllocation, ...]:
    by_name = {allocation.name: allocation for allocation in matched_fit.ALLOCATIONS}
    names = tuple(value.strip() for value in raw.split(",") if value.strip())
    unknown = sorted(set(names) - set(ALLOCATION_NAMES))
    if unknown:
        raise ValueError(f"Only preregistered allocations may be used: {unknown}")
    return tuple(by_name[name] for name in names)


def pair_ids(frame: pd.DataFrame, indices: np.ndarray) -> tuple[str, ...]:
    local = frame.iloc[np.asarray(indices, dtype=int)]
    ids = local.loc[local["pair_id"].astype(str).ne(""), "pair_id"].astype(str)
    return tuple(sorted(ids.unique()))


def pair_fold_map(ids: tuple[str, ...], salt: str) -> dict[str, int]:
    ranked = sorted(
        ids,
        key=lambda pair_id: hashlib.sha256(f"{salt}::{pair_id}".encode()).digest(),
    )
    return {pair_id: index % INNER_FOLDS for index, pair_id in enumerate(ranked)}


def role_index(frame: pd.DataFrame, selected: np.ndarray, pair_id: str, role: str) -> int:
    local = frame.iloc[selected]
    match = local["pair_id"].astype(str).eq(pair_id) & local["pair_role"].astype(str).eq(role)
    indices = local.index[match].to_numpy(dtype=int)
    if len(indices) != 1:
        raise ValueError(f"Expected one {role} member for pair {pair_id}, found {len(indices)}")
    return int(indices[0])


def baseline_pair_residuals(
    dataset: family_grp.Dataset,
    frame: pd.DataFrame,
    config: hierarchical.Config,
    indices: np.ndarray,
    salt: str,
) -> np.ndarray:
    """Return nested-OOF residual pairs ordered as aggregate, phase."""
    selected = np.asarray(indices, dtype=int)
    ids = pair_ids(frame, selected)
    if len(ids) < 8:
        raise ValueError(f"At least eight complete pairs are required, found {len(ids)}")
    folds = pair_fold_map(ids, salt)
    prediction: dict[int, float] = {}
    selected_frame = frame.iloc[selected]
    singleton = selected_frame["pair_id"].astype(str).eq("").to_numpy()
    singleton_indices = selected[singleton]
    for fold in range(INNER_FOLDS):
        test_ids = {pair_id for pair_id, value in folds.items() if value == fold}
        train_ids = set(ids) - test_ids
        train_pair_mask = selected_frame["pair_id"].astype(str).isin(train_ids).to_numpy()
        train = np.unique(np.concatenate([selected[train_pair_mask], singleton_indices]))
        test_mask = selected_frame["pair_id"].astype(str).isin(test_ids).to_numpy()
        test = selected[test_mask]
        model = fitting.baseline_model(dataset, config, train)
        for row_index, value in zip(test, model.predict(dataset.weights[test]), strict=True):
            prediction[int(row_index)] = float(value)
    residuals = []
    for pair_id in ids:
        aggregate = role_index(frame, selected, pair_id, "aggregate")
        phase = role_index(frame, selected, pair_id, "phase")
        residuals.append(
            [
                float(dataset.target[aggregate] - prediction[aggregate]),
                float(dataset.target[phase] - prediction[phase]),
            ]
        )
    return np.asarray(residuals, dtype=float)


def covariance_estimate(residuals: np.ndarray, mode: CovarianceMode) -> CovarianceEstimate:
    empirical = np.cov(np.asarray(residuals, dtype=float), rowvar=False, ddof=1)
    aggregate_variance = float(empirical[0, 0])
    phase_variance = float(empirical[1, 1])
    covariance = float(empirical[0, 1])
    correlation = covariance / math.sqrt(max(aggregate_variance * phase_variance, 1e-30))
    shared_variance = 0.0
    if mode is CovarianceMode.IDENTITY:
        matrix = np.eye(2)
    elif mode is CovarianceMode.DIAGONAL:
        matrix = np.diag([aggregate_variance, phase_variance])
    else:
        cap = (1.0 - 1e-6) * min(aggregate_variance, phase_variance)
        shared_variance = min(max(covariance, 0.0), cap)
        matrix = np.asarray(
            [
                [aggregate_variance, shared_variance],
                [shared_variance, phase_variance],
            ],
            dtype=float,
        )
    matrix /= float(np.trace(matrix) / 2.0)
    if np.min(np.linalg.eigvalsh(matrix)) <= 0.0:
        raise ValueError(f"Non-positive pair covariance for {mode}: {matrix}")
    return CovarianceEstimate(
        mode=mode,
        matrix=matrix,
        aggregate_variance=aggregate_variance,
        phase_variance=phase_variance,
        shared_variance=shared_variance,
        correlation=correlation,
        pair_count=len(residuals),
    )


def inverse_sqrt(matrix: np.ndarray) -> np.ndarray:
    values, vectors = np.linalg.eigh(matrix)
    return (vectors * (1.0 / np.sqrt(values))[None, :]) @ vectors.T


def candidate_design(
    dataset: family_grp.Dataset,
    config: hierarchical.Config,
    partial_phase: bool,
) -> tuple[np.ndarray, np.ndarray]:
    actual = hierarchical.build_design(dataset, config)
    tied = hierarchical.build_design(
        replace(dataset, weights=fitting.tied_weights(dataset)),
        config,
    ).values
    if partial_phase:
        values = np.column_stack([tied, actual.values - tied])
        ridge = np.concatenate([actual.ridge_multipliers, actual.ridge_multipliers])
        return values, ridge
    return actual.values, actual.ridge_multipliers


def fit_pair_gls(
    dataset: family_grp.Dataset,
    frame: pd.DataFrame,
    config: hierarchical.Config,
    indices: np.ndarray,
    covariance: CovarianceEstimate,
    partial_phase: bool,
    coupling: float,
) -> fitting.StructuredModel:
    selected = np.asarray(indices, dtype=int)
    design, ridge_multipliers = candidate_design(dataset, config, partial_phase)
    width = design.shape[1]
    augmented_rows: list[np.ndarray] = []
    target_rows: list[np.ndarray] = []
    whitening = inverse_sqrt(covariance.matrix)
    used = np.zeros(len(frame), dtype=bool)
    for pair_id in pair_ids(frame, selected):
        aggregate = role_index(frame, selected, pair_id, "aggregate")
        phase = role_index(frame, selected, pair_id, "phase")
        block = np.column_stack([design[[aggregate, phase]], np.ones(2)])
        augmented_rows.append(whitening @ block)
        target_rows.append(whitening @ dataset.target[[aggregate, phase]])
        used[[aggregate, phase]] = True
    singleton = selected[~used[selected]]
    if len(singleton):
        augmented_rows.append(np.column_stack([design[singleton], np.ones(len(singleton))]))
        target_rows.append(dataset.target[singleton])
    fit_design = np.vstack(augmented_rows)
    fit_target = np.concatenate(target_rows)

    if config.l2 > 0.0:
        ridge = np.sqrt(config.l2 * ridge_multipliers)
        ridge_rows = np.column_stack([np.diag(ridge), np.zeros(width)])
        fit_design = np.vstack([fit_design, ridge_rows])
        fit_target = np.concatenate([fit_target, np.zeros(width)])
        if partial_phase and coupling > 0.0:
            feature_width = width // 2
            coupling_scale = np.sqrt(config.l2 * coupling * ridge_multipliers[:feature_width])
            coupling_rows = np.zeros((feature_width, width + 1))
            coupling_rows[:, :feature_width] = -np.diag(coupling_scale)
            coupling_rows[:, feature_width:width] = np.diag(coupling_scale)
            fit_design = np.vstack([fit_design, coupling_rows])
            fit_target = np.concatenate([fit_target, np.zeros(feature_width)])

    lower = np.concatenate([np.zeros(width), [-np.inf]])
    upper = np.full(width + 1, np.inf)
    result = lsq_linear(
        fit_design,
        fit_target,
        bounds=(lower, upper),
        method="trf",
        lsmr_tol="auto",
        max_iter=5_000,
    )
    if not result.success:
        raise RuntimeError(f"Pair GLS failed: {result.message}")
    coefficients = np.asarray(result.x[:width], dtype=float)
    if partial_phase:
        feature_width = width // 2
        aggregate_coefficients = coefficients[:feature_width]
        phase_coefficients = coefficients[feature_width:]
        estimator = fitting.Estimator.PARTIAL_PHASE_ORTHOGONAL_MOMENTS
    else:
        aggregate_coefficients = coefficients
        phase_coefficients = coefficients
        estimator = fitting.Estimator.SHARED_ORTHOGONAL_MOMENTS
    return fitting.StructuredModel(
        dataset=dataset,
        config=config,
        estimator=estimator,
        intercept=float(result.x[-1]),
        aggregate_coefficients=aggregate_coefficients,
        phase_coefficients=phase_coefficients,
        coupling=coupling,
    )


def outer_covariances(
    dataset: family_grp.Dataset,
    frame: pd.DataFrame,
    config: hierarchical.Config,
    mode: CovarianceMode,
    salt: str,
) -> tuple[np.ndarray, tuple[CovarianceEstimate, ...]]:
    folds = fitting.fold_ids(frame)
    estimates = []
    for fold in range(fitting.FOLDS):
        train = np.flatnonzero(folds != fold)
        residuals = baseline_pair_residuals(dataset, frame, config, train, f"{salt}::outer::{fold}")
        estimates.append(covariance_estimate(residuals, mode))
    return folds, tuple(estimates)


def pair_oof(
    dataset: family_grp.Dataset,
    frame: pd.DataFrame,
    config: hierarchical.Config,
    mode: CovarianceMode,
    partial_phase: bool,
    coupling: float,
    salt: str,
) -> OOFResult:
    folds, estimates = outer_covariances(dataset, frame, config, mode, salt)
    prediction = np.full(dataset.n, np.nan, dtype=float)
    aggregate_coefficients: list[np.ndarray] = []
    phase_coefficients: list[np.ndarray] = []
    for fold, covariance in enumerate(estimates):
        train = np.flatnonzero(folds != fold)
        test = np.flatnonzero(folds == fold)
        model = fit_pair_gls(dataset, frame, config, train, covariance, partial_phase, coupling)
        prediction[test] = model.predict(dataset.weights[test])
        aggregate_coefficients.append(model.aggregate_coefficients)
        phase_coefficients.append(model.phase_coefficients)
    if not np.isfinite(prediction).all():
        raise RuntimeError("Incomplete pair-GLS OOF prediction")
    return OOFResult(
        prediction=prediction,
        aggregate_coefficients=tuple(aggregate_coefficients),
        phase_coefficients=tuple(phase_coefficients),
        covariance=estimates,
    )


def baseline_oof(
    dataset: family_grp.Dataset,
    frame: pd.DataFrame,
    config: hierarchical.Config,
) -> OOFResult:
    result = fitting.oof_candidate(
        dataset,
        frame,
        config,
        "uncheatable",
        fitting.Estimator.POOLED_LEVELS,
        math.inf,
    )
    return OOFResult(
        prediction=result.prediction,
        aggregate_coefficients=result.aggregate_coefficients,
        phase_coefficients=result.phase_coefficients,
        covariance=(),
    )


def select_partial_coupling(
    dataset: family_grp.Dataset,
    frame: pd.DataFrame,
    config: hierarchical.Config,
    salt: str,
) -> tuple[float, OOFResult, list[dict[str, float]]]:
    rows = []
    results = {}
    for coupling in COUPLING_GRID:
        result = pair_oof(
            dataset,
            frame,
            config,
            CovarianceMode.RANDOM_EFFECTS,
            True,
            coupling,
            f"{salt}::coupling::{coupling}",
        )
        results[coupling] = result
        rows.append({"coupling": coupling, **composition.prediction_metrics(dataset.target, result.prediction)})
    table = pd.DataFrame(rows).sort_values(["rmse", "coupling"], ascending=[True, False])
    best_rmse = float(table.iloc[0]["rmse"])
    selected = table.loc[table["rmse"] <= 1.01 * best_rmse].sort_values("coupling", ascending=False).iloc[0]
    coupling = float(selected["coupling"])
    return coupling, results[coupling], rows


def full_covariance(
    dataset: family_grp.Dataset,
    frame: pd.DataFrame,
    config: hierarchical.Config,
    mode: CovarianceMode,
    salt: str,
) -> CovarianceEstimate:
    residuals = baseline_pair_residuals(dataset, frame, config, np.arange(dataset.n), f"{salt}::full")
    return covariance_estimate(residuals, mode)


def candidate_full_model(
    candidate: Candidate,
    dataset: family_grp.Dataset,
    frame: pd.DataFrame,
    config: hierarchical.Config,
    coupling: float,
    salt: str,
) -> tuple[fitting.StructuredModel, CovarianceEstimate | None]:
    indices = np.arange(dataset.n)
    if candidate is Candidate.POOLED_IDENTITY:
        return fitting.baseline_model(dataset, config, indices), None
    mode = CovarianceMode.DIAGONAL if candidate is Candidate.PAIRED_DIAGONAL_SHARED else CovarianceMode.RANDOM_EFFECTS
    covariance = full_covariance(dataset, frame, config, mode, salt)
    partial = candidate is Candidate.PAIRED_RANDOM_EFFECTS_PARTIAL_PHASE
    return fit_pair_gls(dataset, frame, config, indices, covariance, partial, coupling), covariance


def unused_pair_metrics(
    model: fitting.StructuredModel,
    matched: matched_fit.MatchedSources,
    selected: pd.DataFrame,
    target: str,
) -> list[dict[str, Any]]:
    selected_ids = set(selected.loc[selected["pair_id"].astype(str).ne(""), "pair_id"].astype(str))
    unused = matched.pair_frame.loc[~matched.pair_frame["pair_id"].astype(str).isin(selected_ids)]
    if len(unused) < 3:
        return []
    broad_indices = unused["broad_index"].to_numpy(dtype=int)
    single_indices = unused["single_index"].to_numpy(dtype=int)
    aggregate_observed = matched.sources.single.frame.iloc[single_indices][fitting.TARGET_COLUMNS[target]].to_numpy(
        dtype=float
    )
    phase_observed = matched.sources.broad.frame.iloc[broad_indices][fitting.TARGET_COLUMNS[target]].to_numpy(
        dtype=float
    )
    aggregate_prediction = model.predict(matched.sources.single.weights[single_indices])
    phase_prediction = model.predict(matched.sources.broad.weights[broad_indices])
    return [
        {
            "scope": "unused_pair_aggregate",
            **composition.prediction_metrics(aggregate_observed, aggregate_prediction),
        },
        {
            "scope": "unused_pair_phase",
            **composition.prediction_metrics(phase_observed, phase_prediction),
        },
        {
            "scope": "unused_pair_delta",
            **composition.prediction_metrics(
                phase_observed - aggregate_observed,
                phase_prediction - aggregate_prediction,
            ),
        },
    ]


def candidate_oof_results(
    dataset: family_grp.Dataset,
    frame: pd.DataFrame,
    config: hierarchical.Config,
    target: str,
    salt: str,
) -> tuple[dict[Candidate, tuple[OOFResult, float]], list[dict[str, Any]]]:
    partial_coupling, partial, grid = select_partial_coupling(dataset, frame, config, salt)
    results = {
        Candidate.POOLED_IDENTITY: (baseline_oof(dataset, frame, config), math.inf),
        Candidate.PAIRED_DIAGONAL_SHARED: (
            pair_oof(dataset, frame, config, CovarianceMode.DIAGONAL, False, math.inf, f"{salt}::diagonal"),
            math.inf,
        ),
        Candidate.PAIRED_RANDOM_EFFECTS_SHARED: (
            pair_oof(
                dataset,
                frame,
                config,
                CovarianceMode.RANDOM_EFFECTS,
                False,
                math.inf,
                f"{salt}::random_effects",
            ),
            math.inf,
        ),
        Candidate.PAIRED_RANDOM_EFFECTS_PARTIAL_PHASE: (partial, partial_coupling),
    }
    return results, [{"target": target, **row} for row in grid]


def covariance_record(
    target: str,
    allocation: str,
    seed: int,
    candidate: Candidate,
    scope: str,
    fold: int,
    estimate: CovarianceEstimate,
) -> dict[str, Any]:
    return {
        "target": target,
        "allocation": allocation,
        "seed": seed,
        "candidate": candidate.value,
        "scope": scope,
        "fold": fold,
        "mode": estimate.mode.value,
        "pair_count": estimate.pair_count,
        "aggregate_variance": estimate.aggregate_variance,
        "phase_variance": estimate.phase_variance,
        "shared_variance": estimate.shared_variance,
        "residual_correlation": estimate.correlation,
        "normalized_covariance_json": json.dumps(estimate.matrix.tolist()),
    }


def optimum_record(
    model: fitting.StructuredModel,
    dataset: family_grp.Dataset,
    sources: composition.Sources,
    target: str,
    allocation: str,
    seed: int,
    candidate: Candidate,
    starts: int,
) -> dict[str, Any]:
    initial = raw_optima.optimization_starts(dataset, "two_phase", seed, starts)
    weights, prediction, converged = raw_optima.optimize(
        raw_optima.Fitted(candidate.value, model),
        dataset,
        "two_phase",
        initial,
    )
    exposure = weights[0] * dataset.c0 + weights[1] * dataset.c1
    return {
        "target": target,
        "allocation": allocation,
        "seed": seed,
        "candidate": candidate.value,
        "predicted_bpb": prediction,
        "optimizer_converged": converged,
        "max_bucket_weight": float(weights.max()),
        "max_simulated_epochs": float(exposure.max()),
        "phase_total_variation": float(0.5 * np.abs(weights[0] - weights[1]).sum()),
        "fit_support_distance": raw_optima.support_distance(dataset, weights),
        "nearest_common_policy_tv": float(
            np.min(0.25 * np.abs(sources.common.weights - weights[None, :, :]).sum(axis=(1, 2)))
        ),
        "phase_0_weights_json": json.dumps(dict(zip(dataset.domains, weights[0].tolist(), strict=True))),
        "phase_1_weights_json": json.dumps(dict(zip(dataset.domains, weights[1].tolist(), strict=True))),
    }


def render(metrics: pd.DataFrame, output_dir: Path) -> None:
    common = metrics.loc[metrics["scope"].eq("common_all")]
    figure = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=("Common archive RMSE", "Calibration slope", "Regret@1", "Worst optimism"),
    )
    colors = {"uncheatable": "#d73027", "table9": "#1a9850"}
    for target in fitting.TARGETS:
        local = common.loc[common["target"].eq(target)]
        for position, metric in enumerate(("rmse", "calibration_slope", "regret_at_1", "worst_optimism")):
            row, column = divmod(position, 2)
            figure.add_trace(
                go.Box(
                    x=local["candidate"],
                    y=local[metric],
                    name=target,
                    legendgroup=target,
                    marker_color=colors[target],
                    boxpoints="all",
                    jitter=0.2,
                    showlegend=position == 0,
                ),
                row=row + 1,
                col=column + 1,
            )
    figure.add_hline(y=1.0, line_dash="dot", line_color="#666", row=1, col=2)
    figure.update_layout(
        title="Same-seed pair GLS: frozen common-archive diagnostics",
        template="plotly_white",
        width=1600,
        height=1000,
        legend={"orientation": "h", "y": 1.08},
    )
    figure.write_html(output_dir / "common_archive_metrics.html", include_plotlyjs="cdn", config=PLOT_CONFIG)


def write_report(metrics: pd.DataFrame, covariance: pd.DataFrame, output_dir: Path) -> None:
    common = metrics.loc[metrics["scope"].eq("common_all")]
    summary = (
        common.groupby(["target", "allocation", "candidate"], sort=True)
        .agg(
            replicates=("seed", "size"),
            rmse=("rmse", "mean"),
            spearman=("spearman", "mean"),
            calibration_slope=("calibration_slope", "mean"),
            regret_at_1=("regret_at_1", "mean"),
            optimism_gt_0p05=("optimism_gt_0p05", "mean"),
            worst_optimism=("worst_optimism", "mean"),
        )
        .reset_index()
    )
    summary.to_csv(output_dir / "common_archive_summary.csv", index=False)
    covariance_summary = (
        covariance.loc[covariance["scope"].eq("full")]
        .groupby(["target", "allocation", "candidate"], sort=True)
        .agg(
            replicates=("seed", "size"),
            residual_correlation=("residual_correlation", "mean"),
            shared_variance=("shared_variance", "mean"),
            shared_variance_min=("shared_variance", "min"),
        )
        .reset_index()
    )
    covariance_summary.to_csv(output_dir / "covariance_summary.csv", index=False)
    lines = [
        "# Paired random-effects HPR",
        "",
        "## Mechanism",
        "",
        "For exact aggregate-matched policies that reuse a data seed, the observation model is ",
        "`Y[j,r] = f(w[j,r]) + u[j] + epsilon[j,r]`. The response `f` is the frozen HPR surface; only ",
        "the likelihood changes. Nested grouped-OOF residuals estimate the shared seed variance. Covariance ",
        "matrices are normalized to mean marginal variance one, preserving the existing ridge scale. With zero ",
        "shared variance and equal marginal variances, the estimator reduces to pooled HPR.",
        "",
        "## Common archive",
        "",
        summary.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Training-only covariance",
        "",
        covariance_summary.to_markdown(index=False, floatfmt=".6f"),
    ]
    (output_dir / "report.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    if args.output_dir == DEFAULT_OUTPUT_DIR and not PREREGISTRATION_PATH.exists():
        raise FileNotFoundError(f"Missing frozen preregistration {PREREGISTRATION_PATH}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    seeds = tuple(int(value) for value in args.seeds.split(",") if value.strip())
    allocations = selected_allocations(args.allocations)
    matched = matched_fit.matched_sources()
    metric_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    covariance_rows: list[dict[str, Any]] = []
    coupling_rows: list[dict[str, Any]] = []
    stability_rows: list[dict[str, Any]] = []
    optimum_rows: list[dict[str, Any]] = []

    for target in fitting.TARGETS:
        config = composition.hpr_config(target)
        common_observed = matched.sources.common.frame[fitting.TARGET_COLUMNS[target]].to_numpy(dtype=float)
        for allocation in allocations:
            for seed in seeds:
                print(f"Fitting {target}/{allocation.name}/seed={seed}", flush=True)
                frame, weights = matched_fit.sampled_rows(matched, allocation, target, seed)
                dataset = composition.custom_dataset(
                    matched.sources.reference,
                    frame,
                    weights,
                    target,
                    f"paired_gls_{target}_{allocation.name}_{seed}",
                )
                salt = f"{target}::{allocation.name}::{seed}"
                oof_results, coupling_grid = candidate_oof_results(dataset, frame, config, target, salt)
                for row in coupling_grid:
                    coupling_rows.append({"allocation": allocation.name, "seed": seed, **row})

                for candidate, (oof, coupling) in oof_results.items():
                    full, full_cov = candidate_full_model(candidate, dataset, frame, config, coupling, salt)
                    base = {
                        "target": target,
                        "allocation": allocation.name,
                        "seed": seed,
                        "candidate": candidate.value,
                        "coupling": coupling,
                        "parameter_count": (
                            len(full.aggregate_coefficients)
                            + (
                                len(full.phase_coefficients)
                                if candidate is Candidate.PAIRED_RANDOM_EFFECTS_PARTIAL_PHASE
                                else 0
                            )
                            + 1
                        ),
                    }
                    metric_rows.append(
                        {**base, "scope": "train_oof", **composition.prediction_metrics(dataset.target, oof.prediction)}
                    )
                    common_prediction = full.predict(matched.sources.common.weights)
                    fitting.append_metrics(
                        metric_rows,
                        base,
                        matched.sources.common.frame,
                        common_observed,
                        common_prediction,
                        target,
                    )
                    for row in matched_fit.source_holdout_metrics(full, matched, frame, target):
                        metric_rows.append({**base, **row})
                    for row in unused_pair_metrics(full, matched, frame, target):
                        metric_rows.append({**base, **row})
                    for index, (observed, predicted) in enumerate(zip(common_observed, common_prediction, strict=True)):
                        prediction_rows.append(
                            {
                                **base,
                                "row_id": matched.sources.common.frame.iloc[index]["row_id"],
                                "training_series": matched.sources.common.frame.iloc[index]["training_series"],
                                "policy_class": matched.sources.common.frame.iloc[index]["policy_class"],
                                "objective": matched.sources.common.frame.iloc[index]["objective"],
                                "observed": observed,
                                "predicted": predicted,
                                "residual": predicted - observed,
                            }
                        )
                    stability_rows.append(
                        {
                            **base,
                            "coefficient_block": "aggregate",
                            **fitting.coefficient_stability(oof.aggregate_coefficients),
                        }
                    )
                    stability_rows.append(
                        {**base, "coefficient_block": "phase", **fitting.coefficient_stability(oof.phase_coefficients)}
                    )
                    for fold, estimate in enumerate(oof.covariance):
                        covariance_rows.append(
                            covariance_record(target, allocation.name, seed, candidate, "outer_train", fold, estimate)
                        )
                    if full_cov is not None:
                        covariance_rows.append(
                            covariance_record(target, allocation.name, seed, candidate, "full", -1, full_cov)
                        )
                    if args.with_optima:
                        optimum_rows.append(
                            optimum_record(
                                full,
                                dataset,
                                matched.sources,
                                target,
                                allocation.name,
                                seed,
                                candidate,
                                args.optimizer_starts,
                            )
                        )

    metrics = pd.DataFrame(metric_rows)
    predictions = pd.DataFrame(prediction_rows)
    covariance = pd.DataFrame(covariance_rows)
    coupling = pd.DataFrame(coupling_rows)
    stability = pd.DataFrame(stability_rows)
    optima = pd.DataFrame(optimum_rows)
    metrics.to_csv(args.output_dir / "metric_runs.csv", index=False)
    predictions.to_csv(args.output_dir / "common_archive_predictions.csv", index=False)
    covariance.to_csv(args.output_dir / "covariance_estimates.csv", index=False)
    coupling.to_csv(args.output_dir / "coupling_selection.csv", index=False)
    stability.to_csv(args.output_dir / "coefficient_stability.csv", index=False)
    optima.to_csv(args.output_dir / "raw_optima.csv", index=False)
    render(metrics, args.output_dir)
    write_report(metrics, covariance, args.output_dir)
    (args.output_dir / "run_metadata.json").write_text(
        json.dumps(
            {
                "allocations": [allocation.name for allocation in allocations],
                "seeds": seeds,
                "inner_folds": INNER_FOLDS,
                "outer_folds": fitting.FOLDS,
                "coupling_grid": COUPLING_GRID,
                "data_use": "Exposed development outcomes were evaluated only after preregistration.",
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
