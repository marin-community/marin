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
"""Fit HPR with unified same-seed pair and frontier-fiber blocks."""

from __future__ import annotations

import argparse
import json
import sys
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
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "reference_outputs/heterogeneous_block_gls_hpr_20260720"
PREREGISTRATION_PATH = DEFAULT_OUTPUT_DIR / "preregistered_candidates.json"
ALLOCATION_NAMES = ("p100_f80_matched", "p90_f100_matched", "p70_f140_both")
PLOT_CONFIG = {"toImageButtonOptions": {"format": "png", "scale": 4}}


class Candidate(StrEnum):
    POOLED_IDENTITY = "pooled_identity"
    PAIR_ONLY_RANDOM_EFFECTS = "pair_only_random_effects"
    UNIFIED_BLOCK_RANDOM_EFFECTS = "unified_block_random_effects"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seeds", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--allocations", default=",".join(ALLOCATION_NAMES))
    return parser.parse_args()


def selected_allocations(raw: str) -> tuple[matched_fit.PairAllocation, ...]:
    by_name = {allocation.name: allocation for allocation in matched_fit.ALLOCATIONS}
    names = tuple(value.strip() for value in raw.split(",") if value.strip())
    unknown = sorted(set(names) - set(ALLOCATION_NAMES))
    if unknown:
        raise ValueError(f"Only preregistered allocations may be used: {unknown}")
    return tuple(by_name[name] for name in names)


def fiber_covariance(estimate: paired.CovarianceEstimate, size: int) -> np.ndarray:
    idiosyncratic = max(estimate.phase_variance - estimate.shared_variance, 1e-12)
    matrix = idiosyncratic * np.eye(size) + estimate.shared_variance * np.ones((size, size))
    matrix /= float(np.trace(matrix) / size)
    return matrix


def fit_block_model(
    dataset: family_grp.Dataset,
    frame: pd.DataFrame,
    config: hierarchical.Config,
    indices: np.ndarray,
    candidate: Candidate,
    covariance: paired.CovarianceEstimate,
) -> hierarchical.Model:
    selected = np.asarray(indices, dtype=int)
    design = hierarchical.build_design(dataset, config)
    width = design.values.shape[1]
    fit_designs: list[np.ndarray] = []
    fit_targets: list[np.ndarray] = []
    used = np.zeros(len(frame), dtype=bool)
    pair_whitening = paired.inverse_sqrt(covariance.matrix)

    for pair_id in paired.pair_ids(frame, selected):
        aggregate = paired.role_index(frame, selected, pair_id, "aggregate")
        phase = paired.role_index(frame, selected, pair_id, "phase")
        block = np.column_stack([design.values[[aggregate, phase]], np.ones(2)])
        fit_designs.append(pair_whitening @ block)
        fit_targets.append(pair_whitening @ dataset.target[[aggregate, phase]])
        used[[aggregate, phase]] = True

    if candidate is Candidate.UNIFIED_BLOCK_RANDOM_EFFECTS:
        selected_frame = frame.iloc[selected]
        fibers = selected_frame.loc[selected_frame["source_pool"].astype(str).eq("frontier_fiber")]
        for (_anchor, _seed_block), local in fibers.groupby(["anchor_id", "seed_block"], sort=True):
            block_indices = local.index.to_numpy(dtype=int)
            whitening = paired.inverse_sqrt(fiber_covariance(covariance, len(block_indices)))
            block = np.column_stack([design.values[block_indices], np.ones(len(block_indices))])
            fit_designs.append(whitening @ block)
            fit_targets.append(whitening @ dataset.target[block_indices])
            used[block_indices] = True

    singleton = selected[~used[selected]]
    if len(singleton):
        fit_designs.append(np.column_stack([design.values[singleton], np.ones(len(singleton))]))
        fit_targets.append(dataset.target[singleton])
    fit_design = np.vstack(fit_designs)
    fit_target = np.concatenate(fit_targets)
    if config.l2 > 0.0:
        ridge_rows = np.column_stack([np.diag(np.sqrt(config.l2 * design.ridge_multipliers)), np.zeros(width)])
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
        raise RuntimeError(f"Block GLS failed: {result.message}")
    return hierarchical.Model(dataset, config, float(result.x[-1]), np.asarray(result.x[:width], dtype=float))


def candidate_oof(
    dataset: family_grp.Dataset,
    frame: pd.DataFrame,
    config: hierarchical.Config,
    candidate: Candidate,
    salt: str,
) -> tuple[np.ndarray, tuple[np.ndarray, ...]]:
    folds = fitting.fold_ids(frame)
    prediction = np.full(dataset.n, np.nan, dtype=float)
    coefficients = []
    estimates: tuple[paired.CovarianceEstimate, ...] = ()
    if candidate is not Candidate.POOLED_IDENTITY:
        _covariance_folds, estimates = paired.outer_covariances(
            dataset,
            frame,
            config,
            paired.CovarianceMode.RANDOM_EFFECTS,
            salt,
        )
    for fold in range(fitting.FOLDS):
        train = np.flatnonzero(folds != fold)
        test = np.flatnonzero(folds == fold)
        if candidate is Candidate.POOLED_IDENTITY:
            model = hierarchical.fit_model(dataset, config, train)
        else:
            model = fit_block_model(dataset, frame, config, train, candidate, estimates[fold])
        prediction[test] = model.predict(dataset.weights[test])
        coefficients.append(model.coefficients)
    if not np.isfinite(prediction).all():
        raise RuntimeError("Incomplete block-GLS OOF prediction")
    return prediction, tuple(coefficients)


def full_model(
    dataset: family_grp.Dataset,
    frame: pd.DataFrame,
    config: hierarchical.Config,
    candidate: Candidate,
    salt: str,
) -> hierarchical.Model:
    indices = np.arange(dataset.n)
    if candidate is Candidate.POOLED_IDENTITY:
        return hierarchical.fit_model(dataset, config, indices)
    covariance = paired.full_covariance(
        dataset,
        frame,
        config,
        paired.CovarianceMode.RANDOM_EFFECTS,
        salt,
    )
    return fit_block_model(dataset, frame, config, indices, candidate, covariance)


def render(metrics: pd.DataFrame, deltas: pd.DataFrame, output_dir: Path) -> None:
    figure = make_subplots(
        rows=2,
        cols=3,
        subplot_titles=(
            "Uncheatable OOF",
            "Uncheatable common archive",
            "Uncheatable unused fiber delta",
            "Table-9 OOF",
            "Table-9 common archive",
            "Table-9 unused fiber delta",
        ),
    )
    colors = {
        Candidate.POOLED_IDENTITY.value: "#d73027",
        Candidate.PAIR_ONLY_RANDOM_EFFECTS.value: "#fdae61",
        Candidate.UNIFIED_BLOCK_RANDOM_EFFECTS.value: "#1a9850",
    }
    for row, target in enumerate(fitting.TARGETS, start=1):
        for column, scope in enumerate(("train_oof", "common_all"), start=1):
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
        local_delta = deltas.loc[deltas["target"].eq(target) & deltas["scope"].eq("unused_fiber_all")]
        for candidate, group in local_delta.groupby("candidate", sort=False):
            figure.add_trace(
                go.Box(
                    x=group["candidate"],
                    y=group["delta_rmse"],
                    name=candidate,
                    legendgroup=candidate,
                    marker_color=colors[candidate],
                    boxpoints="all",
                    jitter=0.2,
                    showlegend=False,
                ),
                row=row,
                col=3,
            )
    figure.update_layout(
        title="Unified heterogeneous block GLS",
        template="plotly_white",
        width=1700,
        height=1000,
        legend={"orientation": "h", "y": 1.08},
    )
    figure.write_html(output_dir / "block_gls_diagnostics.html", include_plotlyjs="cdn", config=PLOT_CONFIG)


def write_report(metrics: pd.DataFrame, deltas: pd.DataFrame, output_dir: Path) -> None:
    summary = (
        metrics.loc[metrics["scope"].isin(["train_oof", "common_all", "adversarial_target_matched"])]
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
    delta_summary = (
        deltas.groupby(["target", "allocation", "candidate", "scope"], sort=True)
        .agg(
            replicates=("seed", "size"),
            delta_rmse=("delta_rmse", "mean"),
            delta_bias=("delta_bias", "mean"),
            delta_spearman=("delta_spearman", "mean"),
            delta_sign_accuracy=("delta_sign_accuracy", "mean"),
        )
        .reset_index()
    )
    delta_summary.to_csv(output_dir / "fiber_delta_summary.csv", index=False)
    lines = [
        "# Heterogeneous block-GLS HPR",
        "",
        "The HPR response surface is unchanged. Exact aggregate/phase pairs and same-seed frontier-fiber blocks ",
        "are fitted under `Y[b,r]=f(w[b,r])+u[b]+epsilon[b,r]`. Nested pair residuals identify the shared ",
        "variance; fiber blocks reuse it with the phase-policy idiosyncratic variance. No response parameter or ",
        "covariance hyperparameter is added.",
        "",
        "## Frozen batch",
        "",
        summary.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Held-out fiber deltas",
        "",
        delta_summary.to_markdown(index=False, floatfmt=".6f"),
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
    delta_rows: list[dict[str, Any]] = []
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
                    f"block_gls_{target}_{allocation.name}_{seed}",
                )
                for candidate in Candidate:
                    salt = f"{target}::{allocation.name}::{seed}::{candidate.value}"
                    oof, coefficients = candidate_oof(dataset, frame, config, candidate, salt)
                    model = full_model(dataset, frame, config, candidate, salt)
                    base = {
                        "target": target,
                        "allocation": allocation.name,
                        "seed": seed,
                        "candidate": candidate.value,
                        "parameter_count": len(model.coefficients) + 1,
                    }
                    metric_rows.append(
                        {**base, "scope": "train_oof", **composition.prediction_metrics(dataset.target, oof)}
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
                    for row in composition.fiber_delta_metrics(
                        model,
                        matched.sources,
                        frame,
                        target,
                        allocation,
                        seed,
                    ):
                        delta_rows.append({**base, **row})
                    stability_rows.append({**base, **fitting.coefficient_stability(coefficients)})
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

    metrics = pd.DataFrame(metric_rows)
    predictions = pd.DataFrame(prediction_rows)
    deltas = pd.DataFrame(delta_rows)
    stability = pd.DataFrame(stability_rows)
    metrics.to_csv(args.output_dir / "metric_runs.csv", index=False)
    predictions.to_csv(args.output_dir / "common_archive_predictions.csv", index=False)
    deltas.to_csv(args.output_dir / "fiber_delta_metric_runs.csv", index=False)
    stability.to_csv(args.output_dir / "coefficient_stability.csv", index=False)
    render(metrics, deltas, args.output_dir)
    write_report(metrics, deltas, args.output_dir)
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
