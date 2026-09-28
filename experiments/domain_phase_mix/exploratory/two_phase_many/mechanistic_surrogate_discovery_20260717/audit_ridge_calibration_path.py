# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E501

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
"""Separate ridge range compression from structural heldout optimism.

The strongest frozen deficit-response geometry and output link are held fixed.
Only the nonnegative-head ridge coefficient varies. Heldouts are reported as a
diagnostic path and never used to select the coefficient.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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
DEFAULT_OUTPUT = ARTIFACT_ROOT / "ridge_calibration_path_audit"
SOURCE_METRICS = RESEARCH_DIR / "reference_outputs/hierarchical_deficit_response_20260716/metrics.csv"
L2_GRID = (0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0)
VARIANT = deficit.Variant.POWER_DEFICIT_HYBRID_EARLY_FAMILY_ASYMMETRIC
TARGET_CONFIGS = {
    base.DatasetId.DELPHI_3E18_UNCHEATABLE: (output_link.Link.IDENTITY, 0.0, 1e-3),
    base.DatasetId.DELPHI_3E18_TABLE9: (output_link.Link.LOG_EXCESS, 0.75, 1e-2),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def metric_row(
    dataset: str,
    split: str,
    l2: float,
    selected_l2: float,
    observed: np.ndarray,
    predicted: np.ndarray,
) -> dict[str, object]:
    summary, _bins = gate.metrics(observed, predicted)
    return {
        "dataset": dataset,
        "split": split,
        "l2": l2,
        "fit_selected_l2": selected_l2,
        "is_fit_selected": bool(np.isclose(l2, selected_l2)),
        **summary,
    }


def audit_dataset(
    dataset_id: base.DatasetId,
    source_metrics: pd.DataFrame,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    dataset = base.load_dataset(dataset_id)
    splits = base.split_indices(dataset, dataset_id, np.arange(dataset.n), base.SCREEN_SEED)
    structural = output_link.selected_deficit_config(dataset_id, VARIANT, source_metrics)
    link, floor_fraction, selected_l2 = TARGET_CONFIGS[dataset_id]
    heldout = base.heldout_data(dataset_id, dataset)
    if heldout is None:
        raise ValueError(dataset_id)
    heldout_frame, heldout_weights, heldout_target = heldout
    matched = heldout_frame["policy_class"].eq("two_phase").to_numpy(dtype=bool)
    metric_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []
    for l2 in L2_GRID:
        link_config = output_link.LinkConfig(link, floor_fraction, l2)
        oof = output_link.oof_prediction(dataset, structural, link_config, splits)
        metric_rows.append(metric_row(dataset_id.value, "fit_oof", l2, selected_l2, dataset.target, oof))
        model = output_link.fit_model(dataset, structural, link_config, np.arange(dataset.n))
        heldout_prediction = model.predict(heldout_weights)
        metric_rows.append(
            metric_row(
                dataset_id.value,
                "heldout_policy_matched",
                l2,
                selected_l2,
                heldout_target[matched],
                heldout_prediction[matched],
            )
        )
        for row_id, observed, predicted in zip(
            heldout_frame.loc[matched, "wandb_run_name"],
            heldout_target[matched],
            heldout_prediction[matched],
            strict=True,
        ):
            prediction_rows.append(
                {
                    "dataset": dataset_id.value,
                    "l2": l2,
                    "row_id": row_id,
                    "observed": observed,
                    "predicted": predicted,
                    "optimism": observed - predicted,
                }
            )
    return metric_rows, prediction_rows


def render(metrics: pd.DataFrame, output: Path) -> None:
    figure = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Grouped-OOF RMSE",
            "Policy-matched heldout RMSE",
            "Heldout observed-on-predicted slope",
            "Heldout extreme optimism",
        ),
    )
    colors = {
        "delphi_3e18_uncheatable": "#1a9850",
        "delphi_3e18_table9": "#d73027",
    }
    for dataset, local in metrics.groupby("dataset", sort=True):
        color = colors[dataset]
        fit = local.loc[local["split"].eq("fit_oof")]
        heldout = local.loc[local["split"].eq("heldout_policy_matched")]
        traces = (
            (fit, "rmse", 1, 1, True),
            (heldout, "rmse", 1, 2, False),
            (heldout, "calibration_slope_observed_on_predicted", 2, 1, False),
            (heldout, "worst_optimism", 2, 2, False),
        )
        for frame, metric, row, column, show_legend in traces:
            figure.add_trace(
                go.Scatter(
                    x=frame["l2"],
                    y=frame[metric],
                    mode="lines+markers",
                    name=dataset,
                    legendgroup=dataset,
                    showlegend=show_legend,
                    line={"color": color},
                    customdata=np.column_stack(
                        [
                            frame["optimism_gt_0p05_count"],
                            frame["regret_at_1"],
                            frame["is_fit_selected"],
                        ]
                    ),
                    hovertemplate=(
                        "l2=%{x:.2g}<br>value=%{y:.5f}<br>optimism count=%{customdata[0]}"
                        "<br>regret@1=%{customdata[1]:.5f}<br>fit-selected=%{customdata[2]}<extra></extra>"
                    ),
                ),
                row=row,
                col=column,
            )
    figure.add_hline(y=1.0, line_dash="dash", line_color="#64748b", row=2, col=1)
    figure.update_xaxes(type="symlog" if False else "log")
    # Plotly's log axis cannot display zero, so place zero at a small visual sentinel.
    for trace in figure.data:
        trace.x = np.where(np.asarray(trace.x, dtype=float) == 0.0, 1e-7, np.asarray(trace.x, dtype=float))
    figure.update_xaxes(title_text="Ridge coefficient (zero shown at 1e-7)")
    figure.update_layout(
        title="Does coefficient shrinkage cause optimum-region optimism?",
        template="plotly_white",
        width=1450,
        height=900,
        legend={"orientation": "h", "y": -0.1},
    )
    figure.write_html(output, include_plotlyjs="cdn", config={"toImageButtonOptions": {"scale": 4}})


def main() -> None:
    args = parse_args()
    gate.assert_sealed_absent(SOURCE_METRICS)
    source_metrics = pd.read_csv(SOURCE_METRICS)
    metric_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []
    for dataset_id in TARGET_CONFIGS:
        metrics, predictions = audit_dataset(dataset_id, source_metrics)
        metric_rows.extend(metrics)
        prediction_rows.extend(predictions)
    metrics = pd.DataFrame(metric_rows)
    predictions = pd.DataFrame(prediction_rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(args.output_dir / "ridge_path_metrics.csv", index=False)
    predictions.to_csv(args.output_dir / "ridge_path_heldout_predictions.csv", index=False)
    render(metrics, args.output_dir / "ridge_calibration_path.html")

    selected = metrics.loc[metrics["is_fit_selected"]].copy()
    best_heldout = (
        metrics.loc[metrics["split"].eq("heldout_policy_matched")]
        .sort_values(["dataset", "rmse"])
        .groupby("dataset", as_index=False)
        .first()
    )
    report = [
        "# Ridge calibration-path audit",
        "",
        "The strongest frozen deficit geometry and response link are fixed. Only ridge varies; heldouts are diagnostic and never select the coefficient.",
        "",
        "## Fit-selected settings",
        "",
        selected.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Oracle heldout minima (diagnostic upper bound, not a selection rule)",
        "",
        best_heldout.to_markdown(index=False, floatfmt=".6f"),
        "",
        "If no ridge value jointly removes severe optimism and preserves decision metrics, coefficient shrinkage is not the structural cause of the failure.",
    ]
    (args.output_dir / "report.md").write_text("\n".join(report) + "\n")
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
