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
"""Evaluate HPR regularization selected under the paired likelihood."""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import replace
from enum import StrEnum
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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
    benchmark_matched_pair_heterogeneous_hpr_20260720 as matched_fit,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_paired_random_effects_hpr_20260720 as paired,
)

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "reference_outputs/paired_likelihood_regularized_hpr_20260720"
PREREGISTRATION_PATH = DEFAULT_OUTPUT_DIR / "preregistered_candidates.json"
ALLOCATION_NAMES = ("p140", "t42_p119")
PLOT_CONFIG = {"toImageButtonOptions": {"format": "png", "scale": 4}}


class Candidate(StrEnum):
    LEGACY = "legacy_regularization"
    PAIRED_LIKELIHOOD = "paired_likelihood_regularization"


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


def candidate_config(target: str, candidate: Candidate):
    base = composition.hpr_config(target)
    if candidate is Candidate.LEGACY or target == "table9":
        return base
    return replace(base, l2=0.01, residual_shrink=30.0)


def render(metrics: pd.DataFrame, output_dir: Path) -> None:
    local = metrics.loc[metrics["scope"].isin(["train_oof", "common_all", "adversarial_target_matched"])]
    figure = make_subplots(
        rows=2,
        cols=3,
        subplot_titles=(
            "Uncheatable OOF",
            "Uncheatable common archive",
            "Uncheatable target-matched",
            "Table-9 OOF",
            "Table-9 common archive",
            "Table-9 target-matched",
        ),
    )
    scopes = ("train_oof", "common_all", "adversarial_target_matched")
    colors = {Candidate.LEGACY.value: "#d73027", Candidate.PAIRED_LIKELIHOOD.value: "#1a9850"}
    for row, target in enumerate(fitting.TARGETS, start=1):
        for column, scope in enumerate(scopes, start=1):
            subset = local.loc[local["target"].eq(target) & local["scope"].eq(scope)]
            for candidate, group in subset.groupby("candidate", sort=False):
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
        title="Paired-likelihood regularization: frozen development evaluation",
        template="plotly_white",
        width=1700,
        height=1000,
        legend={"orientation": "h", "y": 1.08},
    )
    figure.write_html(output_dir / "regularization_diagnostics.html", include_plotlyjs="cdn", config=PLOT_CONFIG)


def write_report(metrics: pd.DataFrame, output_dir: Path) -> None:
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
    lines = [
        "# Paired-likelihood-selected HPR regularization",
        "",
        "The HPR state transition and feature map are frozen. Only Uncheatable coefficient regularization changes ",
        "from legacy `L2=0.1, residual_shrink=10` to training-only paired-likelihood-selected ",
        "`L2=0.01, residual_shrink=30`. Table-9 is an exact no-change control.",
        "",
        summary.to_markdown(index=False, floatfmt=".6f"),
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

    for target in fitting.TARGETS:
        common_observed = matched.sources.common.frame[fitting.TARGET_COLUMNS[target]].to_numpy(dtype=float)
        for allocation in allocations:
            for seed in seeds:
                print(f"Fitting {target}/{allocation.name}/seed={seed}", flush=True)
                frame, weights = matched_fit.sampled_rows(matched, allocation, target, seed)
                for candidate in Candidate:
                    config = candidate_config(target, candidate)
                    dataset = composition.custom_dataset(
                        matched.sources.reference,
                        frame,
                        weights,
                        target,
                        f"paired_reg_{target}_{allocation.name}_{seed}_{candidate.value}",
                    )
                    # Candidate comparisons must share covariance-estimation folds.
                    salt = f"{target}::{allocation.name}::{seed}"
                    oof = paired.pair_oof(
                        dataset,
                        frame,
                        config,
                        paired.CovarianceMode.RANDOM_EFFECTS,
                        False,
                        math.inf,
                        salt,
                    )
                    model, _covariance = paired.candidate_full_model(
                        paired.Candidate.PAIRED_RANDOM_EFFECTS_SHARED,
                        dataset,
                        frame,
                        config,
                        math.inf,
                        salt,
                    )
                    base = {
                        "target": target,
                        "allocation": allocation.name,
                        "seed": seed,
                        "candidate": candidate.value,
                        "l2": config.l2,
                        "residual_shrink": config.residual_shrink,
                        "parameter_count": len(model.aggregate_coefficients) + 1,
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

    metrics = pd.DataFrame(metric_rows)
    predictions = pd.DataFrame(prediction_rows)
    metrics.to_csv(args.output_dir / "metric_runs.csv", index=False)
    predictions.to_csv(args.output_dir / "common_archive_predictions.csv", index=False)
    render(metrics, args.output_dir)
    write_report(metrics, args.output_dir)
    (args.output_dir / "run_metadata.json").write_text(
        json.dumps(
            {
                "allocations": [allocation.name for allocation in allocations],
                "seeds": seeds,
                "data_use": "Frozen training-only regularization rule evaluated once on exposed development outcomes.",
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
