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
"""Identify which frozen mechanistic-state directions leave fit support.

This is a diagnostic audit, not a model. It decomposes the residual from each
heldout state to its convex projection onto the 280 fit states.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px

from experiments.domain_phase_mix.exploratory.two_phase_many import (
    benchmark_deficit_output_link_20260716 as output_link,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (
    benchmark_hierarchical_coverage_grp_20260715 as base,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (
    benchmark_hierarchical_deficit_response_20260716 as deficit,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.mechanistic_surrogate_discovery_20260717 import (
    audit_convex_support as convex,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.mechanistic_surrogate_discovery_20260717 import (
    freeze_baseline_gate as gate,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.mechanistic_surrogate_discovery_20260717 import (
    screen_kish_collision_invariant as collision,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.mechanistic_surrogate_discovery_20260717 import (
    screen_nested_support_invariants as support,
)

SCRIPT_DIR = Path(__file__).resolve().parent
RESEARCH_DIR = SCRIPT_DIR.parent
ARTIFACT_ROOT = RESEARCH_DIR / "reference_outputs/mechanistic_surrogate_discovery_20260717"
DEFAULT_OUTPUT = ARTIFACT_ROOT / "convex_support_direction_audit"
SOURCE_METRICS = RESEARCH_DIR / "reference_outputs/hierarchical_deficit_response_20260716/metrics.csv"
LINK_METRICS = RESEARCH_DIR / "reference_outputs/deficit_output_link_asymmetric_20260716/metrics.csv"
FAILURE_ATLAS = ARTIFACT_ROOT / "failure_atlas/heldout_failure_atlas.csv"
TARGETS = (
    base.DatasetId.DELPHI_3E18_UNCHEATABLE,
    base.DatasetId.DELPHI_3E18_TABLE9,
)
OPTIMISM_THRESHOLD = 0.05


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def candidate_dataset(dataset: base.Dataset, weights: np.ndarray, target: np.ndarray) -> base.Dataset:
    return replace(dataset, weights=weights, target=target)


def normalized_coefficients(projector: convex.ConvexProjector) -> np.ndarray:
    if projector.coefficients.value is None:
        raise RuntimeError("Convex projection did not return coefficients")
    coefficients = np.maximum(np.asarray(projector.coefficients.value, dtype=float), 0.0)
    return coefficients / coefficients.sum()


def target_directions(
    dataset_id: base.DatasetId,
    source_metrics: pd.DataFrame,
    link_metrics: pd.DataFrame,
    atlas: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    dataset = base.load_dataset(dataset_id)
    config = output_link.selected_deficit_config(dataset_id, collision.DEFICIT_VARIANT, source_metrics)
    link_config = support.selected_link_config(dataset_id, link_metrics)
    model = collision.fit_model(
        dataset,
        config,
        link_config,
        collision.Config(collision.Mechanism.BASELINE),
        np.arange(dataset.n),
    )
    fit_design = deficit.build_design(dataset, config)
    heldout = base.heldout_data(dataset_id, dataset)
    if heldout is None:
        raise ValueError(dataset_id)
    heldout_frame, heldout_weights, heldout_target = heldout
    baseline = atlas.loc[atlas["dataset"].eq(dataset_id.value) & atlas["mechanism"].eq("baseline")].copy()
    retained = heldout_frame["wandb_run_name"].astype(str).isin(baseline["row_id"].astype(str)).to_numpy()
    heldout_frame = heldout_frame.loc[retained].reset_index(drop=True)
    heldout_weights = heldout_weights[retained]
    heldout_target = heldout_target[retained]
    heldout_design = deficit.build_design(candidate_dataset(dataset, heldout_weights, heldout_target), config)

    scale = fit_design.values.std(axis=0)
    active = scale > 1e-10
    mean = fit_design.values[:, active].mean(axis=0)
    standardized_fit = (fit_design.values[:, active] - mean) / scale[active]
    standardized_heldout = (heldout_design.values[:, active] - mean) / scale[active]
    active_names = np.asarray(fit_design.names, dtype=object)[active]
    active_coefficients = model.coefficients[active]
    projector = convex.ConvexProjector(standardized_fit)
    baseline = baseline.set_index("row_id")
    panel_source = dataset.frame["panel_source"].astype(str).to_numpy()

    direction_rows: list[dict[str, object]] = []
    source_rows: list[dict[str, object]] = []
    for index, row in heldout_frame.iterrows():
        row_id = str(row["wandb_run_name"])
        atlas_row = baseline.loc[row_id]
        distance, effective_support, _status = projector.project(standardized_heldout[index])
        projection_weights = normalized_coefficients(projector)
        projection = standardized_fit.T @ projection_weights
        standardized_residual = standardized_heldout[index] - projection
        raw_residual = standardized_residual * scale[active]
        contribution = raw_residual * active_coefficients
        severe = bool(float(atlas_row["optimism"]) > OPTIMISM_THRESHOLD)
        order = np.argsort(-np.abs(standardized_residual))
        rank = np.empty_like(order)
        rank[order] = np.arange(1, len(order) + 1)
        for feature_index, feature in enumerate(active_names):
            direction_rows.append(
                {
                    "dataset": dataset_id.value,
                    "row_id": row_id,
                    "training_series": atlas_row["training_series"],
                    "observed": float(atlas_row["observed"]),
                    "predicted": float(atlas_row["predicted"]),
                    "optimism": float(atlas_row["optimism"]),
                    "severe_optimism": severe,
                    "convex_hull_distance": distance,
                    "convex_effective_support": effective_support,
                    "feature": str(feature),
                    "standardized_support_residual": standardized_residual[feature_index],
                    "absolute_standardized_support_residual": abs(standardized_residual[feature_index]),
                    "support_residual_rank": int(rank[feature_index]),
                    "raw_design_support_residual": raw_residual[feature_index],
                    "fitted_latent_contribution_outside_support": contribution[feature_index],
                }
            )
        for source in sorted(set(panel_source)):
            source_rows.append(
                {
                    "dataset": dataset_id.value,
                    "row_id": row_id,
                    "severe_optimism": severe,
                    "panel_source": source,
                    "projection_weight": float(projection_weights[panel_source == source].sum()),
                }
            )
    return pd.DataFrame(direction_rows), pd.DataFrame(source_rows)


def summarize(directions: pd.DataFrame) -> pd.DataFrame:
    grouped = directions.groupby(["dataset", "feature", "severe_optimism"], as_index=False).agg(
        policies=("row_id", "nunique"),
        median_absolute_residual=("absolute_standardized_support_residual", "median"),
        mean_absolute_residual=("absolute_standardized_support_residual", "mean"),
        median_signed_residual=("standardized_support_residual", "median"),
        median_fitted_latent_contribution=("fitted_latent_contribution_outside_support", "median"),
        top_three_frequency=("support_residual_rank", lambda value: float((value <= 3).mean())),
    )
    severe = grouped.loc[grouped["severe_optimism"]].drop(columns="severe_optimism")
    ordinary = grouped.loc[~grouped["severe_optimism"]].drop(columns="severe_optimism")
    return severe.merge(ordinary, on=["dataset", "feature"], suffixes=("_severe", "_ordinary"), how="left")


def render(summary: pd.DataFrame, output: Path) -> None:
    displayed = (
        summary.sort_values(
            ["dataset", "top_three_frequency_severe", "median_absolute_residual_severe"], ascending=False
        )
        .groupby("dataset", as_index=False)
        .head(12)
    )
    figure = px.bar(
        displayed,
        x="median_absolute_residual_severe",
        y="feature",
        color="top_three_frequency_severe",
        facet_col="dataset",
        orientation="h",
        color_continuous_scale="RdYlGn_r",
        hover_data=[
            "median_signed_residual_severe",
            "median_fitted_latent_contribution_severe",
            "median_absolute_residual_ordinary",
        ],
        title="Mechanistic state directions absent around severe heldout optimism failures",
    )
    figure.update_yaxes(matches=None, showticklabels=True)
    figure.update_layout(template="plotly_white")
    figure.write_html(output, include_plotlyjs="cdn")


def main() -> None:
    args = parse_args()
    for path in (SOURCE_METRICS, LINK_METRICS, FAILURE_ATLAS):
        gate.assert_sealed_absent(path)
    source_metrics = pd.read_csv(SOURCE_METRICS)
    link_metrics = pd.read_csv(LINK_METRICS)
    atlas = pd.read_csv(FAILURE_ATLAS)
    outputs = [target_directions(dataset_id, source_metrics, link_metrics, atlas) for dataset_id in TARGETS]
    directions = pd.concat([output[0] for output in outputs], ignore_index=True)
    sources = pd.concat([output[1] for output in outputs], ignore_index=True)
    summary = summarize(directions)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    directions.to_csv(args.output_dir / "heldout_support_directions.csv", index=False)
    sources.to_csv(args.output_dir / "heldout_projection_source_mass.csv", index=False)
    summary.to_csv(args.output_dir / "support_direction_summary.csv", index=False)
    render(summary, args.output_dir / "convex_support_directions.html")
    top = (
        summary.sort_values(
            ["dataset", "top_three_frequency_severe", "median_absolute_residual_severe"], ascending=False
        )
        .groupby("dataset", as_index=False)
        .head(8)
    )
    source_summary = sources.groupby(["dataset", "severe_optimism", "panel_source"], as_index=False).agg(
        median_projection_weight=("projection_weight", "median"),
        mean_projection_weight=("projection_weight", "mean"),
    )
    report = [
        "# Convex-support direction audit",
        "",
        "This decomposes extrapolation in the frozen mechanistic design. It does not add support distance, nearest "
        "neighbors, or projection residuals to the surrogate.",
        "",
        "## Dominant unsupported directions around severe errors",
        "",
        top.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Projection mass by fit-panel source",
        "",
        source_summary.to_markdown(index=False, floatfmt=".6f"),
    ]
    (args.output_dir / "report.md").write_text("\n".join(report) + "\n")
    print(top.to_string(index=False))
    print(source_summary.to_string(index=False))


if __name__ == "__main__":
    main()
