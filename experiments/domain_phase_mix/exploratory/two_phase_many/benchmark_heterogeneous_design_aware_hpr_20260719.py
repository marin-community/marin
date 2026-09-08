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
"""Fit HPR using the acquisition structure of a heterogeneous swarm.

The estimator preserves the frozen Hierarchical Phase Replay (HPR) feature
map. It separates a phase-tied aggregate spine from an exact-zero phase
residual and uses same-seed frontier-fiber differences as contrast equations.
No heldout outcome is used to choose the feature map or coupling strength.
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
    benchmark_hierarchical_coverage_grp_20260715 as hierarchical,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    fit_production_grp_quality_variants as family_grp,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.mechanistic_surrogate_discovery_20260717 import (  # noqa: E402
    audit_raw_optima as raw_optima,
)

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "reference_outputs/heterogeneous_design_aware_hpr_20260719"
PREREGISTRATION_PATH = DEFAULT_OUTPUT_DIR / "preregistered_candidates.json"
TARGETS = composition.TARGETS
TARGET_COLUMNS = composition.TARGET_COLUMNS
COUPLING_GRID = (0.0, 0.1, 1.0, 10.0, 100.0)
FOLDS = 4
PLOT_CONFIG = {"toImageButtonOptions": {"format": "png", "scale": 4}}


class Estimator(StrEnum):
    POOLED_LEVELS = "pooled_levels"
    SHARED_ORTHOGONAL_MOMENTS = "shared_orthogonal_moments"
    PARTIAL_PHASE_ORTHOGONAL_MOMENTS = "partial_phase_orthogonal_moments"


@dataclass(frozen=True)
class StructuredModel:
    """An aggregate HPR spine plus an exact-zero phase residual."""

    dataset: family_grp.Dataset
    config: hierarchical.Config
    estimator: Estimator
    intercept: float
    aggregate_coefficients: np.ndarray
    phase_coefficients: np.ndarray
    coupling: float

    def designs(self, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        candidate = replace(
            self.dataset,
            weights=np.asarray(weights, dtype=float),
            target=np.zeros(len(weights), dtype=float),
        )
        actual = hierarchical.build_design(candidate, self.config).values
        tied = hierarchical.build_design(replace(candidate, weights=tied_weights(candidate)), self.config).values
        return tied, actual - tied

    def predict(self, weights: np.ndarray) -> np.ndarray:
        aggregate, phase = self.designs(weights)
        return np.asarray(
            self.intercept + aggregate @ self.aggregate_coefficients + phase @ self.phase_coefficients,
            dtype=float,
        )

    def predict_phase_delta(self, weights: np.ndarray) -> np.ndarray:
        _aggregate, phase = self.designs(weights)
        return np.asarray(phase @ self.phase_coefficients, dtype=float)


@dataclass(frozen=True)
class FitRows:
    """Absolute-level and same-seed contrast equations for one selected panel."""

    absolute_design: np.ndarray
    absolute_target: np.ndarray
    contrast_design: np.ndarray
    contrast_target: np.ndarray
    ridge_multipliers: np.ndarray


@dataclass(frozen=True)
class OOFResult:
    prediction: np.ndarray
    phase_delta_prediction: np.ndarray
    phase_delta_observed: np.ndarray
    phase_delta_mask: np.ndarray
    aggregate_coefficients: tuple[np.ndarray, ...]
    phase_coefficients: tuple[np.ndarray, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seeds", default="0,1,2,3,4,5,6,7")
    parser.add_argument(
        "--allocations",
        default="b100_s80_f100_both,s180_f100_matched",
        help="Comma-separated fixed-budget allocations from the composition audit.",
    )
    parser.add_argument("--optimizer-starts", type=int, default=6)
    parser.add_argument("--skip-optima", action="store_true")
    return parser.parse_args()


def selected_allocations(raw: str) -> tuple[composition.Allocation, ...]:
    by_name = {allocation.name: allocation for allocation in composition.ALLOCATIONS}
    names = tuple(value.strip() for value in raw.split(",") if value.strip())
    unknown = sorted(set(names) - set(by_name))
    if unknown:
        raise ValueError(f"Unknown allocations: {unknown}")
    return tuple(by_name[name] for name in names)


def stable_fold(value: str) -> int:
    digest = hashlib.sha256(value.encode()).digest()
    return int.from_bytes(digest[:4], "little") % FOLDS


def fold_ids(frame: pd.DataFrame) -> np.ndarray:
    """Keep each same-seed fiber block wholly inside one fold."""
    result = np.empty(len(frame), dtype=int)
    for index, row in frame.iterrows():
        pair_id = str(row.get("pair_id", ""))
        if pair_id and pair_id != "nan":
            key = f"pair::{pair_id}"
        elif row["source_pool"] == "frontier_fiber":
            key = f"fiber::{row['anchor_id']}::{int(row['seed_block'])}"
        else:
            key = f"{row['source_pool']}::{row['coordinate_hash']}"
        result[index] = stable_fold(key)
    if set(result.tolist()) != set(range(FOLDS)):
        raise ValueError(f"Incomplete fold allocation: {sorted(set(result.tolist()))}")
    return result


def tied_weights(dataset: family_grp.Dataset) -> np.ndarray:
    phase_fraction = float(np.median(dataset.c0 / np.maximum(dataset.c0 + dataset.c1, 1e-12)))
    aggregate = phase_fraction * dataset.weights[:, 0, :] + (1.0 - phase_fraction) * dataset.weights[:, 1, :]
    return np.stack([aggregate, aggregate], axis=1)


def fiber_delta_column(target: str) -> str:
    if target == "uncheatable":
        return "uncheatable_delta_vs_same_seed_center"
    if target == "table9":
        return "table9_delta_vs_same_seed_center"
    raise ValueError(f"Unknown target {target}")


def inverse_sqrt_shared_center_covariance(size: int) -> np.ndarray:
    """Whiten deltas whose independent runs share one center observation.

    If every run has variance sigma^2, Cov(y_i-y_c, y_j-y_c) is
    sigma^2 (I + 11^T). The common sigma cancels against level equations.
    """
    covariance = np.eye(size) + np.ones((size, size))
    values, vectors = np.linalg.eigh(covariance)
    return (vectors * (1.0 / np.sqrt(values))[None, :]) @ vectors.T


def fit_rows(
    dataset: family_grp.Dataset,
    frame: pd.DataFrame,
    config: hierarchical.Config,
    indices: np.ndarray,
    target: str,
    estimator: Estimator,
) -> FitRows:
    design = hierarchical.build_design(dataset, config)
    tied_dataset = replace(dataset, weights=tied_weights(dataset))
    tied = hierarchical.build_design(tied_dataset, config).values
    phase = design.values - tied
    selected = np.asarray(indices, dtype=int)
    selected_frame = frame.iloc[selected]

    source = selected_frame["source_pool"].astype(str).to_numpy()
    pair_role = selected_frame.get("pair_role", pd.Series(index=selected_frame.index, dtype=object)).astype(str)
    phase_member = pair_role.eq("phase").to_numpy()
    center = (
        selected_frame.get("contrast_family", pd.Series(index=selected_frame.index, dtype=object))
        .eq("center_control")
        .to_numpy()
    )
    absolute_mask = ((source != "frontier_fiber") | center) & ~phase_member
    absolute_indices = selected[absolute_mask]
    if estimator is Estimator.SHARED_ORTHOGONAL_MOMENTS:
        absolute_design = design.values[absolute_indices]
    else:
        aggregate = tied[absolute_indices]
        phase_piece = phase[absolute_indices]
        absolute_design = np.column_stack([aggregate, phase_piece])
    absolute_target = dataset.target[absolute_indices]

    contrast_designs: list[np.ndarray] = []
    contrast_targets: list[np.ndarray] = []
    paired_phase = selected_frame.loc[phase_member]
    if len(paired_phase):
        pair_indices = paired_phase.index.to_numpy(dtype=int)
        pair_design = phase[pair_indices] / np.sqrt(2.0)
        pair_target = paired_phase["matched_delta"].to_numpy(dtype=float) / np.sqrt(2.0)
        if estimator is Estimator.SHARED_ORTHOGONAL_MOMENTS:
            contrast_designs.append(pair_design)
        else:
            contrast_designs.append(np.column_stack([np.zeros_like(pair_design), pair_design]))
        contrast_targets.append(pair_target)

    tilted_frame = selected_frame.loc[(source == "frontier_fiber") & ~center]
    delta_column = fiber_delta_column(target)
    selected_centers = selected_frame.loc[(source == "frontier_fiber") & center]
    for (anchor, block), local in tilted_frame.groupby(["anchor_id", "seed_block"], sort=True):
        local_indices = local.index.to_numpy(dtype=int)
        matching_center = selected_centers.loc[
            selected_centers["anchor_id"].eq(anchor) & selected_centers["seed_block"].eq(block)
        ]
        if len(matching_center) != 1:
            raise ValueError(
                f"Expected one selected center for frontier fiber {anchor}/{block}, found {len(matching_center)}"
            )
        center_index = int(matching_center.index[0])
        # Fiber targets are measured against the same-seed center, so their
        # estimating equation must use the identical feature contrast.
        local_design = phase[local_indices] - phase[center_index][None, :]
        local_target = local[delta_column].to_numpy(dtype=float)
        whitening = inverse_sqrt_shared_center_covariance(len(local))
        if estimator is Estimator.SHARED_ORTHOGONAL_MOMENTS:
            contrast_designs.append(whitening @ local_design)
        else:
            contrast_designs.append(np.column_stack([np.zeros_like(local_design), whitening @ local_design]))
        contrast_targets.append(whitening @ local_target)

    width = absolute_design.shape[1]
    contrast_design = np.concatenate(contrast_designs, axis=0) if contrast_designs else np.empty((0, width))
    contrast_target = np.concatenate(contrast_targets) if contrast_targets else np.empty(0)
    ridge = design.ridge_multipliers
    ridge_multipliers = ridge if estimator is Estimator.SHARED_ORTHOGONAL_MOMENTS else np.concatenate([ridge, ridge])
    return FitRows(absolute_design, absolute_target, contrast_design, contrast_target, ridge_multipliers)


def solve_structured(
    dataset: family_grp.Dataset,
    frame: pd.DataFrame,
    config: hierarchical.Config,
    indices: np.ndarray,
    target: str,
    estimator: Estimator,
    coupling: float,
) -> StructuredModel:
    rows = fit_rows(dataset, frame, config, indices, target, estimator)
    width = rows.absolute_design.shape[1]
    level_mean = rows.absolute_design.mean(axis=0)
    target_mean = float(rows.absolute_target.mean())
    level_design = rows.absolute_design - level_mean[None, :]
    level_target = rows.absolute_target - target_mean
    fit_design = np.column_stack([level_design, np.ones(len(level_design))])
    fit_target = level_target.copy()
    if len(rows.contrast_target):
        fit_design = np.vstack(
            [fit_design, np.column_stack([rows.contrast_design, np.zeros(len(rows.contrast_design))])]
        )
        fit_target = np.concatenate([fit_target, rows.contrast_target])

    if config.l2 > 0.0:
        ridge = np.sqrt(config.l2 * rows.ridge_multipliers)
        ridge_rows = np.column_stack([np.diag(ridge), np.zeros(width)])
        fit_design = np.vstack([fit_design, ridge_rows])
        fit_target = np.concatenate([fit_target, np.zeros(width)])
        if estimator is Estimator.PARTIAL_PHASE_ORTHOGONAL_MOMENTS and coupling > 0.0:
            feature_width = width // 2
            coupling_scale = np.sqrt(config.l2 * coupling * rows.ridge_multipliers[:feature_width])
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
        raise RuntimeError(f"Structured fit failed: {result.message}")
    coefficients = np.asarray(result.x[:width], dtype=float)
    fitted_offset = float(result.x[-1])
    if estimator is Estimator.SHARED_ORTHOGONAL_MOMENTS:
        aggregate_coefficients = coefficients
        phase_coefficients = coefficients
    else:
        feature_width = width // 2
        aggregate_coefficients = coefficients[:feature_width]
        phase_coefficients = coefficients[feature_width:]
    intercept = target_mean + fitted_offset - float(level_mean @ coefficients)
    return StructuredModel(
        dataset=dataset,
        config=config,
        estimator=estimator,
        intercept=intercept,
        aggregate_coefficients=aggregate_coefficients,
        phase_coefficients=phase_coefficients,
        coupling=coupling,
    )


def baseline_model(
    dataset: family_grp.Dataset,
    config: hierarchical.Config,
    indices: np.ndarray,
) -> StructuredModel:
    fitted = hierarchical.fit_model(dataset, config, indices)
    return StructuredModel(
        dataset=dataset,
        config=config,
        estimator=Estimator.POOLED_LEVELS,
        intercept=fitted.intercept,
        aggregate_coefficients=fitted.coefficients,
        phase_coefficients=fitted.coefficients,
        coupling=math.inf,
    )


def fit_candidate(
    dataset: family_grp.Dataset,
    frame: pd.DataFrame,
    config: hierarchical.Config,
    indices: np.ndarray,
    target: str,
    estimator: Estimator,
    coupling: float,
) -> StructuredModel:
    if estimator is Estimator.POOLED_LEVELS:
        return baseline_model(dataset, config, indices)
    return solve_structured(dataset, frame, config, indices, target, estimator, coupling)


def oof_candidate(
    dataset: family_grp.Dataset,
    frame: pd.DataFrame,
    config: hierarchical.Config,
    target: str,
    estimator: Estimator,
    coupling: float,
) -> OOFResult:
    folds = fold_ids(frame)
    prediction = np.full(dataset.n, np.nan, dtype=float)
    phase_delta_prediction = np.full(dataset.n, np.nan, dtype=float)
    aggregate_coefficients: list[np.ndarray] = []
    phase_coefficients: list[np.ndarray] = []
    for fold in range(FOLDS):
        train = np.flatnonzero(folds != fold)
        test = np.flatnonzero(folds == fold)
        model = fit_candidate(dataset, frame, config, train, target, estimator, coupling)
        prediction[test] = model.predict(dataset.weights[test])
        phase_delta_prediction[test] = model.predict_phase_delta(dataset.weights[test])
        aggregate_coefficients.append(model.aggregate_coefficients)
        phase_coefficients.append(model.phase_coefficients)
    if not np.isfinite(prediction).all() or not np.isfinite(phase_delta_prediction).all():
        raise RuntimeError("Incomplete OOF prediction")
    source = frame["source_pool"].astype(str)
    contrast = frame.get("contrast_family", pd.Series(index=frame.index, dtype=object)).astype(str)
    pair_role = frame.get("pair_role", pd.Series(index=frame.index, dtype=object)).astype(str)
    fiber_delta = source.eq("frontier_fiber") & ~contrast.eq("center_control")
    pair_delta = pair_role.eq("phase")
    delta_mask = fiber_delta | pair_delta
    observed_delta = np.full(dataset.n, np.nan, dtype=float)
    observed_delta[fiber_delta] = frame.loc[fiber_delta, fiber_delta_column(target)].to_numpy(dtype=float)
    observed_delta[pair_delta] = frame.loc[pair_delta, "matched_delta"].to_numpy(dtype=float)
    return OOFResult(
        prediction=prediction,
        phase_delta_prediction=phase_delta_prediction,
        phase_delta_observed=observed_delta,
        phase_delta_mask=delta_mask.to_numpy(),
        aggregate_coefficients=tuple(aggregate_coefficients),
        phase_coefficients=tuple(phase_coefficients),
    )


def select_coupling(
    dataset: family_grp.Dataset,
    frame: pd.DataFrame,
    config: hierarchical.Config,
    target: str,
) -> tuple[float, OOFResult, list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    results: dict[float, OOFResult] = {}
    for coupling in COUPLING_GRID:
        result = oof_candidate(
            dataset,
            frame,
            config,
            target,
            Estimator.PARTIAL_PHASE_ORTHOGONAL_MOMENTS,
            coupling,
        )
        results[coupling] = result
        level = composition.prediction_metrics(dataset.target, result.prediction)
        mask = result.phase_delta_mask
        delta_rmse = float(
            np.sqrt(np.mean((result.phase_delta_prediction[mask] - result.phase_delta_observed[mask]) ** 2))
        )
        rows.append({"coupling": coupling, "delta_rmse": delta_rmse, **level})
    table = pd.DataFrame(rows).sort_values(["rmse", "delta_rmse", "coupling"], ascending=[True, True, False])
    best_rmse = float(table.iloc[0]["rmse"])
    eligible = table.loc[table["rmse"] <= 1.01 * best_rmse]
    selected = eligible.sort_values(["coupling", "delta_rmse"], ascending=[False, True]).iloc[0]
    coupling = float(selected["coupling"])
    return coupling, results[coupling], rows


def coefficient_stability(values: tuple[np.ndarray, ...]) -> dict[str, float]:
    matrix = np.stack(values)
    normalized = matrix / np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12)
    similarities = normalized @ normalized.T
    upper = similarities[np.triu_indices(len(matrix), k=1)]
    active = np.abs(matrix.mean(axis=0)) > 1e-8
    coefficient_cv = np.std(matrix[:, active], axis=0) / np.maximum(np.abs(np.mean(matrix[:, active], axis=0)), 1e-8)
    return {
        "fold_cosine_mean": float(np.mean(upper)),
        "fold_cosine_min": float(np.min(upper)),
        "active_coefficient_count": int(np.sum(active)),
        "active_coefficient_cv_median": float(np.median(coefficient_cv)) if np.any(active) else math.nan,
    }


def append_metrics(
    rows: list[dict[str, Any]],
    base: dict[str, Any],
    frame: pd.DataFrame,
    observed: np.ndarray,
    predicted: np.ndarray,
    target: str,
) -> None:
    for scope, mask in composition.scope_masks(frame, target).items():
        if np.sum(mask) < 3:
            continue
        rows.append({**base, "scope": scope, **composition.prediction_metrics(observed[mask], predicted[mask])})
    for series, indices in frame.groupby("training_series", sort=True).indices.items():
        local = np.asarray(indices, dtype=int)
        if len(local) < 3:
            continue
        rows.append(
            {
                **base,
                "scope": f"series::{series}",
                **composition.prediction_metrics(observed[local], predicted[local]),
            }
        )


def optimum_record(
    model: StructuredModel,
    dataset: family_grp.Dataset,
    sources: composition.Sources,
    target: str,
    allocation: str,
    seed: int,
    estimator: Estimator,
    starts: int,
) -> dict[str, Any]:
    initial = raw_optima.optimization_starts(dataset, "two_phase", seed, starts)
    weights, prediction, converged = raw_optima.optimize(
        raw_optima.Fitted(estimator.value, model), dataset, "two_phase", initial
    )
    exposure = weights[0] * dataset.c0 + weights[1] * dataset.c1
    return {
        "target": target,
        "allocation": allocation,
        "seed": seed,
        "estimator": estimator.value,
        "predicted_bpb": prediction,
        "optimizer_converged": converged,
        "max_bucket_weight": float(weights.max()),
        "max_simulated_epochs": float(exposure.max()),
        "phase_total_variation": float(0.5 * np.abs(weights[0] - weights[1]).sum()),
        "fit_support_distance": raw_optima.support_distance(dataset, weights),
        "phase_0_weights_json": json.dumps(dict(zip(dataset.domains, weights[0].tolist(), strict=True))),
        "phase_1_weights_json": json.dumps(dict(zip(dataset.domains, weights[1].tolist(), strict=True))),
        "nearest_common_policy_tv": float(
            np.min(0.25 * np.abs(sources.common.weights - weights[None, :, :]).sum(axis=(1, 2)))
        ),
    }


def render(metrics: pd.DataFrame, predictions: pd.DataFrame, output_dir: Path) -> None:
    common = metrics.loc[metrics["scope"].eq("common_all")].copy()
    figure = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=("Common archive RMSE", "Regret@1", "Calibration slope", "Worst optimism"),
    )
    colors = {"uncheatable": "#d73027", "table9": "#1a9850"}
    for target in TARGETS:
        local = common.loc[common["target"].eq(target)]
        for position, metric in enumerate(("rmse", "regret_at_1", "calibration_slope", "worst_optimism")):
            row, column = divmod(position, 2)
            figure.add_trace(
                go.Box(
                    x=local["estimator"],
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
    figure.add_hline(y=1.0, line_dash="dot", line_color="#666", row=2, col=1)
    figure.update_layout(
        title="Design-aware HPR: frozen common-archive diagnostics",
        template="plotly_white",
        width=1500,
        height=950,
        legend={"orientation": "h", "y": 1.08},
    )
    figure.write_html(output_dir / "common_archive_metrics.html", include_plotlyjs="cdn", config=PLOT_CONFIG)

    scatter = make_subplots(rows=1, cols=2, subplot_titles=("Uncheatable", "Table-9"))
    symbols = {
        Estimator.POOLED_LEVELS.value: "circle",
        Estimator.SHARED_ORTHOGONAL_MOMENTS.value: "diamond",
        Estimator.PARTIAL_PHASE_ORTHOGONAL_MOMENTS.value: "square",
    }
    for column, target in enumerate(TARGETS, start=1):
        local = predictions.loc[predictions["target"].eq(target)]
        for estimator, group in local.groupby("estimator", sort=False):
            scatter.add_trace(
                go.Scatter(
                    x=group["observed"],
                    y=group["predicted"],
                    mode="markers",
                    name=estimator,
                    legendgroup=estimator,
                    marker={"symbol": symbols[estimator], "size": 7, "opacity": 0.5},
                    showlegend=column == 1,
                    customdata=np.column_stack([group["row_id"], group["training_series"]]),
                    hovertemplate="%{customdata[0]}<br>%{customdata[1]}<br>obs=%{x:.5f}<br>pred=%{y:.5f}<extra></extra>",
                ),
                row=1,
                col=column,
            )
        low = min(local["observed"].min(), local["predicted"].min())
        high = max(local["observed"].max(), local["predicted"].max())
        scatter.add_trace(
            go.Scatter(
                x=[low, high],
                y=[low, high],
                mode="lines",
                line={"dash": "dash", "color": "#666"},
                showlegend=False,
            ),
            row=1,
            col=column,
        )
    scatter.update_layout(
        title="Frozen common archive: observed versus predicted",
        template="plotly_white",
        width=1500,
        height=700,
        legend={"orientation": "h", "y": 1.12},
    )
    scatter.write_html(output_dir / "common_archive_calibration.html", include_plotlyjs="cdn", config=PLOT_CONFIG)


def write_report(metrics: pd.DataFrame, coupling: pd.DataFrame, stability: pd.DataFrame, output_dir: Path) -> None:
    common = metrics.loc[metrics["scope"].eq("common_all")].copy()
    summary = (
        common.groupby(["target", "allocation", "estimator"], sort=True)
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
    lines = [
        "# Heterogeneous design-aware HPR",
        "",
        "## Frozen estimator",
        "",
        "Let `Phi(w)` be the frozen HPR physical feature map and let `w_tied` preserve the policy's aggregate weights. "
        "The fitted equation is `Y = b + Phi(w_tied) beta + [Phi(w)-Phi(w_tied)] gamma`, with nonnegative "
        "`beta` and `gamma`. It is exactly phase-tied when `w0=w1`. Same-seed fiber differences identify `gamma`; "
        "their reused-center covariance is handled analytically. A ridge coupling shrinks `gamma` toward `beta`; "
        "the pooled HPR and shared-moment fits are nested ablations.",
        "",
        "Coupling is selected only by four-fold training-panel OOF RMSE. Among settings within 1% of the best, the "
        "largest coupling is selected. The common archive is evaluated only after this choice.",
        "",
        "## Common archive",
        "",
        summary.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Coupling selection",
        "",
        coupling.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Fold stability",
        "",
        stability.to_markdown(index=False, floatfmt=".6f"),
    ]
    (output_dir / "report.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    if not PREREGISTRATION_PATH.exists() and output_dir == DEFAULT_OUTPUT_DIR:
        raise FileNotFoundError(f"Missing frozen preregistration: {PREREGISTRATION_PATH}")
    seeds = tuple(int(value) for value in args.seeds.split(",") if value.strip())
    allocations = selected_allocations(args.allocations)
    sources = composition.load_sources()
    metric_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    coupling_rows: list[dict[str, Any]] = []
    stability_rows: list[dict[str, Any]] = []
    optimum_rows: list[dict[str, Any]] = []

    for target in TARGETS:
        config = composition.hpr_config(target)
        common_observed = sources.common.frame[TARGET_COLUMNS[target]].to_numpy(dtype=float)
        for allocation in allocations:
            for seed in composition.allocation_seeds(allocation, seeds):
                print(f"Fitting {target}/{allocation.name}/seed={seed}", flush=True)
                frame, weights = composition.selected_rows(sources, allocation, target, seed)
                dataset = composition.custom_dataset(
                    sources.reference,
                    frame,
                    weights,
                    target,
                    f"heterogeneous_hpr_{target}_{allocation.name}_{seed}",
                )
                selected_coupling, partial_oof, grid = select_coupling(dataset, frame, config, target)
                for row in grid:
                    coupling_rows.append(
                        {
                            "target": target,
                            "allocation": allocation.name,
                            "seed": seed,
                            "selected": float(row["coupling"]) == selected_coupling,
                            **row,
                        }
                    )
                oof_by_estimator = {
                    Estimator.POOLED_LEVELS: oof_candidate(
                        dataset, frame, config, target, Estimator.POOLED_LEVELS, math.inf
                    ),
                    Estimator.SHARED_ORTHOGONAL_MOMENTS: oof_candidate(
                        dataset, frame, config, target, Estimator.SHARED_ORTHOGONAL_MOMENTS, math.inf
                    ),
                    Estimator.PARTIAL_PHASE_ORTHOGONAL_MOMENTS: partial_oof,
                }
                for estimator, oof in oof_by_estimator.items():
                    coupling_value = (
                        selected_coupling if estimator is Estimator.PARTIAL_PHASE_ORTHOGONAL_MOMENTS else math.inf
                    )
                    full = fit_candidate(
                        dataset,
                        frame,
                        config,
                        np.arange(dataset.n),
                        target,
                        estimator,
                        coupling_value,
                    )
                    base = {
                        "target": target,
                        "allocation": allocation.name,
                        "seed": seed,
                        "estimator": estimator.value,
                        "coupling": coupling_value,
                        "parameter_count": (
                            len(full.aggregate_coefficients)
                            + (
                                0
                                if estimator is not Estimator.PARTIAL_PHASE_ORTHOGONAL_MOMENTS
                                else len(full.phase_coefficients)
                            )
                            + 1
                        ),
                    }
                    metric_rows.append(
                        {**base, "scope": "train_oof", **composition.prediction_metrics(dataset.target, oof.prediction)}
                    )
                    delta_mask = oof.phase_delta_mask
                    if np.any(delta_mask):
                        metric_rows.append(
                            {
                                **base,
                                "scope": "train_fiber_delta_oof",
                                **composition.prediction_metrics(
                                    oof.phase_delta_observed[delta_mask], oof.phase_delta_prediction[delta_mask]
                                ),
                            }
                        )
                    common_prediction = full.predict(sources.common.weights)
                    append_metrics(
                        metric_rows,
                        base,
                        sources.common.frame,
                        common_observed,
                        common_prediction,
                        target,
                    )
                    for index, (observed, predicted) in enumerate(zip(common_observed, common_prediction, strict=True)):
                        prediction_rows.append(
                            {
                                **base,
                                "row_id": sources.common.frame.iloc[index]["row_id"],
                                "training_series": sources.common.frame.iloc[index]["training_series"],
                                "policy_class": sources.common.frame.iloc[index]["policy_class"],
                                "objective": sources.common.frame.iloc[index]["objective"],
                                "observed": observed,
                                "predicted": predicted,
                                "residual": predicted - observed,
                            }
                        )
                    stability_rows.append(
                        {
                            **base,
                            "coefficient_block": "aggregate",
                            **coefficient_stability(oof.aggregate_coefficients),
                        }
                    )
                    stability_rows.append(
                        {
                            **base,
                            "coefficient_block": "phase",
                            **coefficient_stability(oof.phase_coefficients),
                        }
                    )
                    if not args.skip_optima and seed == seeds[0]:
                        optimum_rows.append(
                            optimum_record(
                                full,
                                dataset,
                                sources,
                                target,
                                allocation.name,
                                seed,
                                estimator,
                                args.optimizer_starts,
                            )
                        )

    metrics = pd.DataFrame(metric_rows)
    predictions = pd.DataFrame(prediction_rows)
    coupling = pd.DataFrame(coupling_rows)
    stability = pd.DataFrame(stability_rows)
    optima = pd.DataFrame(optimum_rows)
    metrics.to_csv(output_dir / "metric_runs.csv", index=False)
    predictions.to_csv(output_dir / "common_archive_predictions.csv", index=False)
    coupling.to_csv(output_dir / "coupling_selection.csv", index=False)
    stability.to_csv(output_dir / "coefficient_stability.csv", index=False)
    optima.to_csv(output_dir / "raw_optima.csv", index=False)
    render(metrics, predictions, output_dir)
    write_report(metrics, coupling, stability, output_dir)
    (output_dir / "run_metadata.json").write_text(
        json.dumps(
            {
                "allocations": [allocation.name for allocation in allocations],
                "seeds": seeds,
                "coupling_grid": COUPLING_GRID,
                "folds": FOLDS,
                "common_policy_count": len(sources.common.frame),
                "data_use": "Fiber and common-archive outcomes are exposed development evidence.",
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
