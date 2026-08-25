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
"""Decompose the strongest baseline's worst frozen-heldout predictions.

Each heldout policy is paired with its nearest fit design in standardized
mechanistic-feature space. Integrated output contributions allocate the exact
predicted BPB delta across existing channels; the remaining observed delta is
the harm that the fitted state and response do not represent.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

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
    freeze_baseline_gate as gate,
)

SCRIPT_DIR = Path(__file__).resolve().parent
RESEARCH_DIR = SCRIPT_DIR.parent
ARTIFACT_ROOT = RESEARCH_DIR / "reference_outputs/mechanistic_surrogate_discovery_20260717"
DEFAULT_OUTPUT = ARTIFACT_ROOT / "worst_policy_feature_decomposition"
SOURCE_METRICS = RESEARCH_DIR / "reference_outputs/hierarchical_deficit_response_20260716/metrics.csv"
FAILURE_ATLAS = ARTIFACT_ROOT / "failure_atlas/heldout_failure_atlas.csv"
VARIANT = deficit.Variant.POWER_DEFICIT_HYBRID_EARLY_FAMILY_ASYMMETRIC
TARGET_CONFIGS = {
    base.DatasetId.DELPHI_3E18_UNCHEATABLE: output_link.LinkConfig(output_link.Link.IDENTITY, 0.0, 1e-3),
    base.DatasetId.DELPHI_3E18_TABLE9: output_link.LinkConfig(output_link.Link.LOG_EXCESS, 0.75, 1e-2),
}
DISPLAY_TOP_K = 10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def feature_category(name: str) -> str:
    if name.startswith("net_pooled_family"):
        return "pooled family response"
    if name.startswith("net_bucket_excess"):
        return "bucket residual response"
    if name.startswith("net_family_coverage"):
        return "family coverage"
    if name.startswith("family_total_replay"):
        return "family total replay"
    if name.startswith("family_member_replay"):
        return "family member replay"
    if name == "shared_literal_replay":
        return "literal replay"
    if name.startswith("phase0_net_family"):
        return "early family state"
    if name == "phase_shift_tv":
        return "phase divergence"
    raise ValueError(f"Unclassified feature {name}")


def candidate_dataset(
    dataset: base.Dataset,
    weights: np.ndarray,
    target: np.ndarray,
) -> base.Dataset:
    return replace(dataset, weights=weights, target=target)


def audit_target(
    dataset_id: base.DatasetId,
    source_metrics: pd.DataFrame,
    atlas: pd.DataFrame,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    dataset = base.load_dataset(dataset_id)
    structural = output_link.selected_deficit_config(dataset_id, VARIANT, source_metrics)
    link_config = TARGET_CONFIGS[dataset_id]
    model = output_link.fit_model(dataset, structural, link_config, np.arange(dataset.n))
    fit_design = deficit.build_design(dataset, model.deficit_config)
    heldout = base.heldout_data(dataset_id, dataset)
    if heldout is None:
        raise ValueError(dataset_id)
    heldout_frame, heldout_weights, heldout_target = heldout
    heldout_dataset = candidate_dataset(dataset, heldout_weights, heldout_target)
    heldout_design = deficit.build_design(heldout_dataset, model.deficit_config)
    row_to_index = {str(name): index for index, name in enumerate(heldout_frame["wandb_run_name"])}
    heldout_rows = (
        atlas.loc[atlas["dataset"].eq(dataset_id.value) & atlas["mechanism"].eq("baseline")]
        .sort_values("optimism", ascending=False)
        .copy()
    )

    scale = fit_design.values.std(axis=0)
    active_scale = scale > 1e-10
    standardized_fit = (fit_design.values[:, active_scale] - fit_design.values[:, active_scale].mean(axis=0)) / scale[
        active_scale
    ]
    summary_rows: list[dict[str, object]] = []
    detail_rows: list[dict[str, object]] = []
    for rank, atlas_row in enumerate(heldout_rows.itertuples(index=False), start=1):
        heldout_index = row_to_index[str(atlas_row.row_id)]
        heldout_vector = (
            heldout_design.values[heldout_index, active_scale] - fit_design.values[:, active_scale].mean(axis=0)
        ) / scale[active_scale]
        distances = np.sqrt(np.mean((standardized_fit - heldout_vector[None, :]) ** 2, axis=1))
        nearest_index = int(np.argmin(distances))
        nearest_name = str(dataset.frame.iloc[nearest_index]["run_name"])

        heldout_x = heldout_design.values[heldout_index]
        nearest_x = fit_design.values[nearest_index]
        latent_contributions = model.coefficients * (heldout_x - nearest_x)
        heldout_latent = float(model.intercept + heldout_x @ model.coefficients)
        nearest_latent = float(model.intercept + nearest_x @ model.coefficients)
        heldout_prediction = float(model.predict(heldout_weights[[heldout_index]])[0])
        nearest_prediction = float(model.predict(dataset.weights[[nearest_index]])[0])
        latent_delta = heldout_latent - nearest_latent
        prediction_delta = heldout_prediction - nearest_prediction
        if abs(latent_delta) < 1e-12:
            output_contributions = np.zeros_like(latent_contributions)
        else:
            output_contributions = latent_contributions * prediction_delta / latent_delta
        if not np.isclose(output_contributions.sum(), prediction_delta, atol=1e-9):
            raise ValueError("Integrated feature contributions do not recover prediction delta")
        observed = float(heldout_target[heldout_index])
        nearest_observed = float(dataset.target[nearest_index])
        observed_delta = observed - nearest_observed
        missing_harm = observed_delta - prediction_delta
        summary_rows.append(
            {
                "dataset": dataset_id.value,
                "optimism_rank": rank,
                "displayed_worst": rank <= DISPLAY_TOP_K,
                "row_id": atlas_row.row_id,
                "nearest_fit_row": nearest_name,
                "mechanistic_design_distance": float(distances[nearest_index]),
                "heldout_observed": observed,
                "heldout_predicted": heldout_prediction,
                "heldout_optimism": observed - heldout_prediction,
                "nearest_observed": nearest_observed,
                "nearest_predicted": nearest_prediction,
                "observed_delta_from_nearest": observed_delta,
                "predicted_delta_from_nearest": prediction_delta,
                "unrepresented_harm_delta": missing_harm,
            }
        )
        for name, coefficient, held_value, near_value, latent, output in zip(
            fit_design.names,
            model.coefficients,
            heldout_x,
            nearest_x,
            latent_contributions,
            output_contributions,
            strict=True,
        ):
            detail_rows.append(
                {
                    "dataset": dataset_id.value,
                    "optimism_rank": rank,
                    "row_id": atlas_row.row_id,
                    "nearest_fit_row": nearest_name,
                    "feature": name,
                    "category": feature_category(name),
                    "coefficient": coefficient,
                    "heldout_feature_value": held_value,
                    "nearest_feature_value": near_value,
                    "latent_contribution_delta": latent,
                    "output_bpb_contribution_delta": output,
                }
            )
    return summary_rows, detail_rows


def render(summary: pd.DataFrame, details: pd.DataFrame, output: Path) -> None:
    displayed_summary = summary.loc[summary["displayed_worst"]].copy()
    displayed_keys = set(zip(displayed_summary["dataset"], displayed_summary["optimism_rank"], strict=True))
    displayed_details = details.loc[
        [
            (dataset, rank) in displayed_keys
            for dataset, rank in zip(details["dataset"], details["optimism_rank"], strict=True)
        ]
    ].copy()
    categories = displayed_details.groupby(["dataset", "optimism_rank", "row_id", "category"], as_index=False)[
        "output_bpb_contribution_delta"
    ].sum()
    figure = go.Figure()
    palette = ["#1a9850", "#91cf60", "#d9ef8b", "#fee08b", "#fc8d59", "#d73027", "#762a83", "#4575b4"]
    category_order = sorted(categories["category"].unique())
    labels = (
        displayed_summary["dataset"].str.replace("delphi_3e18_", "", regex=False)
        + " #"
        + displayed_summary["optimism_rank"].astype(str)
    )
    keys = zip(displayed_summary["dataset"], displayed_summary["optimism_rank"], strict=True)
    key_to_label = dict(zip(keys, labels, strict=True))
    for color, category in zip(palette, category_order, strict=True):
        local = categories.loc[categories["category"].eq(category)].copy()
        local["label"] = [
            key_to_label[(dataset, rank)] for dataset, rank in zip(local["dataset"], local["optimism_rank"], strict=True)
        ]
        figure.add_trace(
            go.Bar(
                x=local["label"],
                y=local["output_bpb_contribution_delta"],
                name=category,
                marker_color=color,
            )
        )
    figure.add_trace(
        go.Scatter(
            x=labels,
            y=displayed_summary["observed_delta_from_nearest"],
            mode="markers",
            name="observed delta",
            marker={"color": "#111827", "size": 10, "symbol": "diamond"},
            customdata=displayed_summary[["row_id", "nearest_fit_row", "unrepresented_harm_delta"]],
            hovertemplate=(
                "%{customdata[0]}<br>nearest=%{customdata[1]}<br>observed delta=%{y:.4f}"
                "<br>unrepresented harm=%{customdata[2]:.4f}<extra></extra>"
            ),
        )
    )
    figure.update_layout(
        barmode="relative",
        title="Existing model channels versus observed harm at the worst heldout policies",
        xaxis_title="Target and optimism rank",
        yaxis_title="BPB delta from nearest fit design",
        template="plotly_white",
        width=1600,
        height=760,
        legend={"orientation": "h", "y": -0.2},
    )
    figure.write_html(output, include_plotlyjs="cdn", config={"toImageButtonOptions": {"scale": 4}})


def main() -> None:
    args = parse_args()
    for path in (SOURCE_METRICS, FAILURE_ATLAS):
        gate.assert_sealed_absent(path)
    source_metrics = pd.read_csv(SOURCE_METRICS)
    atlas = pd.read_csv(FAILURE_ATLAS)
    summary_rows: list[dict[str, object]] = []
    detail_rows: list[dict[str, object]] = []
    for dataset_id in TARGET_CONFIGS:
        summary, details = audit_target(dataset_id, source_metrics, atlas)
        summary_rows.extend(summary)
        detail_rows.extend(details)
    summary = pd.DataFrame(summary_rows)
    details = pd.DataFrame(detail_rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output_dir / "worst_policy_decomposition_summary.csv", index=False)
    details.to_csv(args.output_dir / "worst_policy_feature_contributions.csv", index=False)
    render(summary, details, args.output_dir / "worst_policy_feature_decomposition.html")
    compact = summary.groupby("dataset", as_index=False).agg(
        median_design_distance=("mechanistic_design_distance", "median"),
        median_observed_delta=("observed_delta_from_nearest", "median"),
        median_predicted_delta=("predicted_delta_from_nearest", "median"),
        median_unrepresented_harm=("unrepresented_harm_delta", "median"),
        minimum_unrepresented_harm=("unrepresented_harm_delta", "min"),
        maximum_unrepresented_harm=("unrepresented_harm_delta", "max"),
    )
    report = [
        "# Worst-policy feature decomposition",
        "",
        "The exact fitted output delta from each policy's nearest fit design is allocated across existing mechanistic "
        "channels by integrated gradients of the scalar response link. The observed-minus-predicted delta is not "
        "fitted and is reported as unrepresented harm.",
        "",
        compact.to_markdown(index=False, floatfmt=".6f"),
        "",
        summary.loc[summary["displayed_worst"]].to_markdown(index=False, floatfmt=".6f"),
    ]
    (args.output_dir / "report.md").write_text("\n".join(report) + "\n")
    print(compact.to_string(index=False))


if __name__ == "__main__":
    main()
