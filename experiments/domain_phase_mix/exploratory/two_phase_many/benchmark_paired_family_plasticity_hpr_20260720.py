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
"""Test a low-dimensional family-retention phase response under pair GLS."""

from __future__ import annotations

import argparse
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
    benchmark_paired_random_effects_hpr_20260720 as paired,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    fit_production_grp_quality_variants as family_grp,
)

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "reference_outputs/paired_family_plasticity_hpr_20260720"
PREREGISTRATION_PATH = DEFAULT_OUTPUT_DIR / "preregistered_candidates.json"
PLOT_CONFIG = {"toImageButtonOptions": {"format": "png", "scale": 4}}


class Candidate(StrEnum):
    PAIRED_RANDOM_EFFECTS_SHARED = "paired_random_effects_shared"
    IDENTITY_FAMILY_COVERAGE_GAIN = "identity_family_coverage_gain"
    PAIRED_RANDOM_EFFECTS_GLOBAL_COVERAGE_GAIN = "paired_random_effects_global_coverage_gain"
    PAIRED_RANDOM_EFFECTS_FAMILY_COVERAGE_GAIN = "paired_random_effects_family_coverage_gain"


@dataclass(frozen=True)
class FamilyPlasticityModel:
    dataset: family_grp.Dataset
    config: hierarchical.Config
    candidate: Candidate
    intercept: float
    base_coefficients: np.ndarray
    plasticity_coefficients: np.ndarray

    def phase_channels(self, weights: np.ndarray) -> np.ndarray:
        candidate = replace(
            self.dataset,
            weights=np.asarray(weights, dtype=float),
            target=np.zeros(len(weights), dtype=float),
        )
        actual = hierarchical.build_design(candidate, self.config)
        tied = hierarchical.build_design(replace(candidate, weights=fitting.tied_weights(candidate)), self.config)
        coverage = coverage_indices(actual)
        family_channels = actual.values[:, coverage] - tied.values[:, coverage]
        if self.candidate is Candidate.PAIRED_RANDOM_EFFECTS_GLOBAL_COVERAGE_GAIN:
            return family_channels.sum(axis=1, keepdims=True)
        if self.candidate in {
            Candidate.IDENTITY_FAMILY_COVERAGE_GAIN,
            Candidate.PAIRED_RANDOM_EFFECTS_FAMILY_COVERAGE_GAIN,
        }:
            return family_channels
        return np.empty((len(weights), 0), dtype=float)

    def predict(self, weights: np.ndarray) -> np.ndarray:
        candidate = replace(
            self.dataset,
            weights=np.asarray(weights, dtype=float),
            target=np.zeros(len(weights), dtype=float),
        )
        base = hierarchical.build_design(candidate, self.config).values
        channels = self.phase_channels(weights)
        return np.asarray(
            self.intercept + base @ self.base_coefficients + channels @ self.plasticity_coefficients,
            dtype=float,
        )


@dataclass(frozen=True)
class OOFResult:
    prediction: np.ndarray
    base_coefficients: tuple[np.ndarray, ...]
    plasticity_coefficients: tuple[np.ndarray, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seeds", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--allocations", default=",".join(paired.ALLOCATION_NAMES))
    return parser.parse_args()


def coverage_indices(design: hierarchical.Design) -> np.ndarray:
    indices = np.asarray(
        [index for index, name in enumerate(design.names) if name.startswith("family_coverage_signal:")],
        dtype=int,
    )
    if len(indices) != 3:
        raise ValueError(f"Expected three family retained-coverage channels, found {len(indices)}")
    return indices


def candidate_matrix(
    dataset: family_grp.Dataset,
    config: hierarchical.Config,
    candidate: Candidate,
) -> tuple[np.ndarray, np.ndarray, int]:
    actual = hierarchical.build_design(dataset, config)
    if candidate is Candidate.PAIRED_RANDOM_EFFECTS_SHARED:
        return actual.values, actual.ridge_multipliers, 0
    tied = hierarchical.build_design(replace(dataset, weights=fitting.tied_weights(dataset)), config)
    channels = actual.values[:, coverage_indices(actual)] - tied.values[:, coverage_indices(actual)]
    if candidate is Candidate.PAIRED_RANDOM_EFFECTS_GLOBAL_COVERAGE_GAIN:
        channels = channels.sum(axis=1, keepdims=True)
    values = np.column_stack([actual.values, channels])
    ridge = np.concatenate([actual.ridge_multipliers, np.ones(channels.shape[1])])
    return values, ridge, channels.shape[1]


def candidate_covariance_mode(candidate: Candidate) -> paired.CovarianceMode:
    if candidate is Candidate.IDENTITY_FAMILY_COVERAGE_GAIN:
        return paired.CovarianceMode.IDENTITY
    return paired.CovarianceMode.RANDOM_EFFECTS


def fit_model(
    dataset: family_grp.Dataset,
    frame: pd.DataFrame,
    config: hierarchical.Config,
    indices: np.ndarray,
    candidate: Candidate,
    covariance: paired.CovarianceEstimate,
) -> FamilyPlasticityModel:
    selected = np.asarray(indices, dtype=int)
    design, ridge_multipliers, plasticity_width = candidate_matrix(dataset, config, candidate)
    width = design.shape[1]
    whitening = paired.inverse_sqrt(covariance.matrix)
    fit_designs: list[np.ndarray] = []
    fit_targets: list[np.ndarray] = []
    used = np.zeros(len(frame), dtype=bool)
    for pair_id in paired.pair_ids(frame, selected):
        aggregate = paired.role_index(frame, selected, pair_id, "aggregate")
        phase = paired.role_index(frame, selected, pair_id, "phase")
        block = np.column_stack([design[[aggregate, phase]], np.ones(2)])
        fit_designs.append(whitening @ block)
        fit_targets.append(whitening @ dataset.target[[aggregate, phase]])
        used[[aggregate, phase]] = True
    singleton = selected[~used[selected]]
    if len(singleton):
        fit_designs.append(np.column_stack([design[singleton], np.ones(len(singleton))]))
        fit_targets.append(dataset.target[singleton])
    fit_design = np.vstack(fit_designs)
    fit_target = np.concatenate(fit_targets)
    if config.l2 > 0.0:
        ridge_rows = np.column_stack([np.diag(np.sqrt(config.l2 * ridge_multipliers)), np.zeros(width)])
        fit_design = np.vstack([fit_design, ridge_rows])
        fit_target = np.concatenate([fit_target, np.zeros(width)])
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
        raise RuntimeError(f"Family plasticity fit failed: {result.message}")
    coefficients = np.asarray(result.x[:width], dtype=float)
    base_width = width - plasticity_width
    return FamilyPlasticityModel(
        dataset=dataset,
        config=config,
        candidate=candidate,
        intercept=float(result.x[-1]),
        base_coefficients=coefficients[:base_width],
        plasticity_coefficients=coefficients[base_width:],
    )


def covariance_estimates(
    dataset: family_grp.Dataset,
    frame: pd.DataFrame,
    config: hierarchical.Config,
    candidate: Candidate,
    salt: str,
) -> tuple[np.ndarray, tuple[paired.CovarianceEstimate, ...]]:
    mode = candidate_covariance_mode(candidate)
    if mode is paired.CovarianceMode.RANDOM_EFFECTS:
        return paired.outer_covariances(dataset, frame, config, mode, salt)
    folds = fitting.fold_ids(frame)
    estimate = paired.CovarianceEstimate(mode, np.eye(2), 1.0, 1.0, 0.0, 0.0, 0)
    return folds, tuple(estimate for _fold in range(fitting.FOLDS))


def oof_model(
    dataset: family_grp.Dataset,
    frame: pd.DataFrame,
    config: hierarchical.Config,
    candidate: Candidate,
    salt: str,
) -> OOFResult:
    folds, estimates = covariance_estimates(dataset, frame, config, candidate, salt)
    prediction = np.full(dataset.n, np.nan, dtype=float)
    base_coefficients = []
    plasticity_coefficients = []
    for fold, covariance in enumerate(estimates):
        train = np.flatnonzero(folds != fold)
        test = np.flatnonzero(folds == fold)
        model = fit_model(dataset, frame, config, train, candidate, covariance)
        prediction[test] = model.predict(dataset.weights[test])
        base_coefficients.append(model.base_coefficients)
        plasticity_coefficients.append(model.plasticity_coefficients)
    if not np.isfinite(prediction).all():
        raise RuntimeError("Incomplete family-plasticity OOF prediction")
    return OOFResult(prediction, tuple(base_coefficients), tuple(plasticity_coefficients))


def full_model(
    dataset: family_grp.Dataset,
    frame: pd.DataFrame,
    config: hierarchical.Config,
    candidate: Candidate,
    salt: str,
) -> FamilyPlasticityModel:
    mode = candidate_covariance_mode(candidate)
    if mode is paired.CovarianceMode.IDENTITY:
        covariance = paired.CovarianceEstimate(mode, np.eye(2), 1.0, 1.0, 0.0, 0.0, 0)
    else:
        covariance = paired.full_covariance(dataset, frame, config, mode, salt)
    return fit_model(dataset, frame, config, np.arange(dataset.n), candidate, covariance)


def extra_stability(values: tuple[np.ndarray, ...]) -> dict[str, float | int]:
    if not len(values[0]):
        return {
            "active_plasticity_count": 0,
            "plasticity_fold_cosine": math.nan,
            "plasticity_coefficient_cv": math.nan,
        }
    matrix = np.stack(values)
    norms = np.linalg.norm(matrix, axis=1)
    active = np.mean(matrix, axis=0) > 1e-8
    pairwise = matrix @ matrix.T / np.maximum(norms[:, None] * norms[None, :], 1e-12)
    upper = pairwise[np.triu_indices(len(matrix), k=1)]
    coefficient_cv = np.std(matrix[:, active], axis=0) / np.maximum(np.mean(matrix[:, active], axis=0), 1e-12)
    return {
        "active_plasticity_count": int(np.sum(active)),
        "plasticity_fold_cosine": float(np.mean(upper)),
        "plasticity_coefficient_cv": float(np.median(coefficient_cv)) if np.any(active) else math.nan,
    }


def render(metrics: pd.DataFrame, output_dir: Path) -> None:
    scopes = ("unused_pair_delta", "common_all", "adversarial_target_matched")
    figure = make_subplots(
        rows=2,
        cols=3,
        subplot_titles=tuple(f"{target}: {scope}" for target in fitting.TARGETS for scope in scopes),
    )
    colors = {
        Candidate.PAIRED_RANDOM_EFFECTS_SHARED.value: "#d73027",
        Candidate.IDENTITY_FAMILY_COVERAGE_GAIN.value: "#fc8d59",
        Candidate.PAIRED_RANDOM_EFFECTS_GLOBAL_COVERAGE_GAIN.value: "#91cf60",
        Candidate.PAIRED_RANDOM_EFFECTS_FAMILY_COVERAGE_GAIN.value: "#1a9850",
    }
    for row, target in enumerate(fitting.TARGETS, start=1):
        for column, scope in enumerate(scopes, start=1):
            local = metrics.loc[metrics["target"].eq(target) & metrics["scope"].eq(scope)]
            for candidate, group in local.groupby("candidate", sort=False):
                figure.add_trace(
                    go.Box(
                        x=group["candidate"],
                        y=group["rmse"],
                        name=candidate,
                        legendgroup=candidate,
                        marker_color=colors[candidate],
                        boxpoints="all",
                        jitter=0.2,
                        showlegend=row == 1 and column == 1,
                    ),
                    row=row,
                    col=column,
                )
    figure.update_layout(
        title="Pair-GLS family plasticity: frozen batch RMSE",
        template="plotly_white",
        width=1700,
        height=1000,
        legend={"orientation": "h", "y": 1.08},
    )
    figure.write_html(output_dir / "family_plasticity_rmse.html", include_plotlyjs="cdn", config=PLOT_CONFIG)


def write_report(metrics: pd.DataFrame, stability: pd.DataFrame, output_dir: Path) -> None:
    scopes = ["train_oof", "unused_pair_delta", "common_all", "adversarial_target_matched"]
    summary = (
        metrics.loc[metrics["scope"].isin(scopes)]
        .groupby(["target", "allocation", "candidate", "scope"], sort=True)
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
    summary.to_csv(output_dir / "summary.csv", index=False)
    stability_summary = (
        stability.groupby(["target", "allocation", "candidate"], sort=True)
        .agg(
            active_plasticity_count=("active_plasticity_count", "mean"),
            plasticity_fold_cosine=("plasticity_fold_cosine", "mean"),
            plasticity_coefficient_cv=("plasticity_coefficient_cv", "mean"),
        )
        .reset_index()
    )
    stability_summary.to_csv(output_dir / "plasticity_stability_summary.csv", index=False)
    lines = [
        "# Paired family-plasticity HPR",
        "",
        "The response is `Y=b+Phi(w) beta+sum_f delta_f [C_f(w)-C_f(w_tied)]`, where `C_f` is the existing ",
        "family retained-coverage state and every coefficient is nonnegative. The correction vanishes exactly for ",
        "phase-tied policies. Exact same-seed pairs use the frozen random-effects GLS likelihood.",
        "",
        "## Frozen batch",
        "",
        summary.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Plasticity stability",
        "",
        stability_summary.to_markdown(index=False, floatfmt=".6f"),
    ]
    (output_dir / "report.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    if args.output_dir == DEFAULT_OUTPUT_DIR and not PREREGISTRATION_PATH.exists():
        raise FileNotFoundError(f"Missing frozen preregistration {PREREGISTRATION_PATH}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    seeds = tuple(int(value) for value in args.seeds.split(",") if value.strip())
    allocations = paired.selected_allocations(args.allocations)
    matched = matched_fit.matched_sources()
    metric_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    coefficient_rows: list[dict[str, Any]] = []
    stability_rows: list[dict[str, Any]] = []

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
                    f"family_plasticity_{target}_{allocation.name}_{seed}",
                )
                for candidate in Candidate:
                    salt = f"{target}::{allocation.name}::{seed}::{candidate.value}"
                    oof = oof_model(dataset, frame, config, candidate, salt)
                    model = full_model(dataset, frame, config, candidate, salt)
                    base = {
                        "target": target,
                        "allocation": allocation.name,
                        "seed": seed,
                        "candidate": candidate.value,
                        "parameter_count": len(model.base_coefficients) + len(model.plasticity_coefficients) + 1,
                    }
                    metric_rows.append(
                        {**base, "scope": "train_oof", **composition.prediction_metrics(dataset.target, oof.prediction)}
                    )
                    common_prediction = model.predict(matched.sources.common.weights)
                    fitting.append_metrics(
                        metric_rows,
                        base,
                        matched.sources.common.frame,
                        common_observed,
                        common_prediction,
                        target,
                    )
                    for row in matched_fit.source_holdout_metrics(model, matched, frame, target):
                        metric_rows.append({**base, **row})
                    for row in paired.unused_pair_metrics(model, matched, frame, target):
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
                    stability_rows.append({**base, **extra_stability(oof.plasticity_coefficients)})
                    coefficient_rows.append(
                        {
                            **base,
                            "plasticity_coefficients_json": json.dumps(model.plasticity_coefficients.tolist()),
                        }
                    )

    metrics = pd.DataFrame(metric_rows)
    predictions = pd.DataFrame(prediction_rows)
    coefficients = pd.DataFrame(coefficient_rows)
    stability = pd.DataFrame(stability_rows)
    metrics.to_csv(args.output_dir / "metric_runs.csv", index=False)
    predictions.to_csv(args.output_dir / "common_archive_predictions.csv", index=False)
    coefficients.to_csv(args.output_dir / "plasticity_coefficients.csv", index=False)
    stability.to_csv(args.output_dir / "plasticity_stability.csv", index=False)
    render(metrics, args.output_dir)
    write_report(metrics, stability, args.output_dir)
    (args.output_dir / "run_metadata.json").write_text(
        json.dumps(
            {
                "allocations": [allocation.name for allocation in allocations],
                "seeds": seeds,
                "candidate_count": len(Candidate),
                "data_use": "This frozen batch was evaluated once after preregistration.",
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
