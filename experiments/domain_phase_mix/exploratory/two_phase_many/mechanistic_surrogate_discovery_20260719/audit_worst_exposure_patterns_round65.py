# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "numpy>=2.0",
#   "pandas>=2.2",
#   "plotly>=6.0",
#   "scipy>=1.15",
#   "tabulate>=0.9",
# ]
# ///
"""Describe which policy geometries concentrate frozen-baseline optimism."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import spearmanr

REPO_ROOT = Path(__file__).resolve().parents[5]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.domain_phase_mix.exploratory.two_phase_many.mechanistic_surrogate_discovery_20260719.audit_partial_identification_round53 import (  # noqa: E402
    heldout_frame,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.mechanistic_surrogate_discovery_20260719.freeze_pareto_gate import (  # noqa: E402
    BASELINE_MODELS,
    DEFAULT_DASHBOARD,
)

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_ROOT = SCRIPT_DIR.parent / "reference_outputs/mechanistic_surrogate_discovery_20260719"
DEFAULT_OUTPUT = OUTPUT_ROOT / "round65_worst_exposure_patterns"
FEATURES = (
    "support_distance",
    "max_epoch",
    "phase_tv",
    "aggregate_tv_to_proportional",
    "aggregate_kl_to_proportional",
)
TOP_K = 10


def markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    return frame[columns].to_markdown(index=False, floatfmt=".5f")


def correlations(frame: pd.DataFrame, target: str, model: str) -> list[dict[str, Any]]:
    predicted = frame[f"prediction::{model}"].to_numpy(dtype=float)
    observed = frame["observed"].to_numpy(dtype=float)
    optimism = observed - predicted
    absolute_residual = np.abs(predicted - observed)
    rows = []
    for feature in FEATURES:
        values = frame[feature].to_numpy(dtype=float)
        rows.append(
            {
                "target": target,
                "model": model,
                "feature": feature,
                "spearman_with_optimism": float(spearmanr(values, optimism).statistic),
                "spearman_with_absolute_residual": float(spearmanr(values, absolute_residual).statistic),
            }
        )
    return rows


def summarize_correlations(frame: pd.DataFrame) -> pd.DataFrame:
    return (
        frame.groupby(["target", "feature"])
        .agg(
            minimum_optimism_correlation=("spearman_with_optimism", "min"),
            median_optimism_correlation=("spearman_with_optimism", "median"),
            maximum_optimism_correlation=("spearman_with_optimism", "max"),
            minimum_absolute_error_correlation=("spearman_with_absolute_residual", "min"),
            median_absolute_error_correlation=("spearman_with_absolute_residual", "median"),
            maximum_absolute_error_correlation=("spearman_with_absolute_residual", "max"),
        )
        .reset_index()
    )


def worst_rows(frame: pd.DataFrame, target: str, model: str) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    selected = frame.copy()
    selected["target"] = target
    selected["model"] = model
    selected["predicted"] = selected[f"prediction::{model}"].to_numpy(dtype=float)
    selected["optimism"] = selected["observed"] - selected["predicted"]
    selected["absolute_residual"] = np.abs(selected["predicted"] - selected["observed"])
    selected = selected.nlargest(TOP_K, "optimism").copy()
    selected.insert(2, "optimism_rank", np.arange(1, len(selected) + 1))

    enrichment = []
    for feature in FEATURES:
        all_median = float(frame[feature].median())
        top_median = float(selected[feature].median())
        enrichment.append(
            {
                "target": target,
                "model": model,
                "feature": feature,
                "all_median": all_median,
                "top10_optimism_median": top_median,
                "top10_over_all_median": top_median / all_median if all_median > 0.0 else np.nan,
            }
        )
    return selected, enrichment


def render_diagnostics(correlation: pd.DataFrame, worst: pd.DataFrame, output: Path) -> None:
    figure = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Uncheatable: feature correlation with optimism",
            "Table-9: feature correlation with optimism",
            "Uncheatable: worst optimism and support distance",
            "Table-9: worst optimism and support distance",
        ),
        vertical_spacing=0.18,
        horizontal_spacing=0.12,
    )
    for column, target in enumerate(("uncheatable", "table9"), start=1):
        target_correlation = correlation.loc[correlation["target"].eq(target)]
        heatmap = target_correlation.pivot(index="model", columns="feature", values="spearman_with_optimism")
        figure.add_trace(
            go.Heatmap(
                z=heatmap.to_numpy(dtype=float),
                x=heatmap.columns,
                y=heatmap.index,
                zmin=-1,
                zmax=1,
                colorscale="RdYlGn_r",
                showscale=column == 2,
                colorbar={"title": "Spearman"} if column == 2 else None,
                hovertemplate="%{y}<br>%{x}<br>rho=%{z:.3f}<extra></extra>",
            ),
            row=1,
            col=column,
        )

        target_worst = worst.loc[worst["target"].eq(target)]
        summary = (
            target_worst.groupby("model", as_index=False)
            .agg(maximum_optimism=("optimism", "max"), median_support_distance=("support_distance", "median"))
            .sort_values("maximum_optimism")
        )
        figure.add_trace(
            go.Bar(
                x=summary["maximum_optimism"],
                y=summary["model"],
                orientation="h",
                marker={"color": summary["median_support_distance"], "colorscale": "RdYlGn_r"},
                customdata=summary["median_support_distance"],
                hovertemplate="%{y}<br>worst optimism=%{x:.5f}<br>top-10 median support=%{customdata:.3f}<extra></extra>",
                showlegend=False,
            ),
            row=2,
            col=column,
        )
    figure.update_xaxes(title_text="Observed - predicted BPB", row=2)
    figure.update_layout(
        title="Frozen-baseline optimism concentrates in extrapolative policies",
        width=1600,
        height=1100,
        template="plotly_white",
        margin={"l": 210, "r": 80, "t": 100, "b": 140},
    )
    figure.write_html(
        output,
        include_plotlyjs="cdn",
        full_html=True,
        config={"displaylogo": False, "toImageButtonOptions": {"format": "png", "scale": 4}},
    )


def write_report(
    output: Path,
    correlations_frame: pd.DataFrame,
    enrichment: pd.DataFrame,
    worst: pd.DataFrame,
) -> None:
    correlation_summary = summarize_correlations(correlations_frame)
    support_summary = (
        correlations_frame.loc[correlations_frame["feature"].eq("support_distance")]
        .groupby("target", as_index=False)
        .agg(
            minimum_support_optimism_correlation=("spearman_with_optimism", "min"),
            maximum_support_optimism_correlation=("spearman_with_optimism", "max"),
            minimum_support_error_correlation=("spearman_with_absolute_residual", "min"),
            maximum_support_error_correlation=("spearman_with_absolute_residual", "max"),
        )
    )
    support_enrichment = enrichment.loc[enrichment["feature"].eq("support_distance")]
    enrichment_summary = (
        support_enrichment.groupby("target")["top10_over_all_median"]
        .agg(minimum_enrichment="min", maximum_enrichment="max")
        .reset_index()
    )
    policy_summary = (
        worst.assign(two_phase=worst["policy_class"].eq("two_phase"))
        .groupby(["target", "model"], as_index=False)
        .agg(two_phase_fraction=("two_phase", "mean"), maximum_optimism=("optimism", "max"))
    )
    report = f"""# Round 65: worst-residual exposure-pattern audit

## Boundary

This is a descriptive audit of the frozen 710-run development archive and the frozen baseline predictions. It fits no model, selects no hyperparameter, proposes no residual feature, and reads no sealed confirmation outcome.

## Support signal

{markdown_table(support_summary, list(support_summary.columns))}

Across all policy diagnostics:

{markdown_table(correlation_summary, list(correlation_summary.columns))}

The median support distance among each model's ten most optimistic rows is enriched relative to the full archive by:

{markdown_table(enrichment_summary, list(enrichment_summary.columns))}

## Policy-class concentration

{markdown_table(policy_summary, list(policy_summary.columns))}

## Interpretation

The worst optimism is not isolated to one response family. Support distance and phase divergence have positive optimism correlations for every audited model and target. In contrast, max epoch and aggregate tilt are usually negatively correlated with optimism, so this is not a generic over-repetition or concentration failure. The evidence points more specifically to unsupported phase contrast, although these diagnostics are correlated and are not causal response terms. The admissible consequence is abstention or a targeted aggregate/phase-contrast intervention, not residual calibration.

## Artifacts

- `feature_correlations.csv`: rank correlations of policy diagnostics with optimism and absolute error.
- `feature_correlation_summary.csv`: across-model ranges by target and policy diagnostic.
- `top10_feature_enrichment.csv`: top-10 optimism medians relative to the full archive.
- `worst_optimism_rows.csv`: exact ten worst rows for every frozen baseline and target.
- `worst_exposure_pattern_diagnostics.html`: interactive heatmaps and worst-error summaries.
"""
    (output / "report.md").write_text(report)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dashboard", type=Path, default=DEFAULT_DASHBOARD)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    bundle = json.loads(args.dashboard.read_text())
    correlation_rows = []
    enrichment_rows = []
    worst_frames = []
    for target in ("uncheatable", "table9"):
        frame = heldout_frame(bundle, target)
        for model in BASELINE_MODELS:
            correlation_rows.extend(correlations(frame, target, model))
            selected, enrichment = worst_rows(frame, target, model)
            worst_frames.append(selected)
            enrichment_rows.extend(enrichment)

    correlation = pd.DataFrame(correlation_rows)
    enrichment = pd.DataFrame(enrichment_rows)
    worst = pd.concat(worst_frames, ignore_index=True)
    columns = [
        "target",
        "model",
        "optimism_rank",
        "row_id",
        "name",
        "policy_class",
        "panel",
        "development_layer",
        "candidate_target",
        "observed",
        "predicted",
        "optimism",
        "absolute_residual",
        *FEATURES,
    ]
    correlation.to_csv(args.output / "feature_correlations.csv", index=False)
    summarize_correlations(correlation).to_csv(args.output / "feature_correlation_summary.csv", index=False)
    enrichment.to_csv(args.output / "top10_feature_enrichment.csv", index=False)
    worst[columns].to_csv(args.output / "worst_optimism_rows.csv", index=False)
    render_diagnostics(correlation, worst, args.output / "worst_exposure_pattern_diagnostics.html")
    write_report(args.output, correlation, enrichment, worst)
    print((args.output / "report.md").read_text())


if __name__ == "__main__":
    main()
