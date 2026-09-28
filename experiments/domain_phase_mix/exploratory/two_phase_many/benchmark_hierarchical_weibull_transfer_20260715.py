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
"""Test bounded retained-learning and literal replay on the 3e18 panel.

The current hierarchical phase-replay GRP uses an unbounded power response and
a soft replay onset. Compact retained-state GRP instead separates a bounded
Weibull learning curve from literal repeated-data reuse beyond one actual
epoch. This benchmark compares complete, coherent response/replay forms using
nested fit-panel CV, then scores the historical 3e18 validation archive as a
frozen transfer diagnostic.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_hierarchical_calibration_forms_20260715 as calibration,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_hierarchical_coverage_grp_20260715 as base,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_production_grp_retained_hybrids_20260713 as retained,
)

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "reference_outputs/hierarchical_weibull_transfer_20260715"
PLOT_CONFIG = {"toImageButtonOptions": {"format": "png", "scale": 4}}
VARIANT_NAMES = (
    "power_global_tau",
    "weibull_global_tau",
    "compact_weibull_shared_replay",
    "weibull_family_coverage_shared_replay",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        default="delphi_3e18_uncheatable,delphi_3e18_table9",
        help="Comma-separated 3e18 dataset IDs.",
    )
    parser.add_argument("--variants", default=",".join(VARIANT_NAMES))
    parser.add_argument("--num-shapes", type=int, default=32)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def heldout_prediction(
    dataset: retained.family_grp.Dataset,
    model: retained.FittedModel,
    weights: np.ndarray,
) -> np.ndarray:
    candidate = replace(
        dataset,
        weights=np.asarray(weights, dtype=float),
        target=np.zeros(len(weights), dtype=float),
    )
    design, _names = retained.build_design(candidate, model.variant, model.shape, model.family_tau)
    return np.asarray(model.head.predict_design(design), dtype=float)


def fit_and_score(
    dataset_id: base.DatasetId,
    variant_names: tuple[str, ...],
    num_shapes: int,
    output_dir: Path,
    *,
    force: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    dataset = base.load_dataset(dataset_id)
    # The 3e18 panel uses the same panel-stratified split policy as the 300M panel.
    split_kind = retained.DatasetId.THREE_HUNDRED_M_UNCHEATABLE
    metrics: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    full_models: dict[str, Any] = {}
    heldout = base.heldout_data(dataset_id, dataset)
    if heldout is None:
        raise ValueError(f"{dataset_id.value} has no frozen heldout archive")
    heldout_frame, heldout_weights, heldout_target = heldout

    for variant_name in variant_names:
        print(f"{dataset_id.value}: {variant_name}", flush=True)
        variant = retained.VARIANT_BY_NAME[variant_name]
        shapes = retained.shared_shape_candidates(variant, num_shapes)
        variant_dir = output_dir / dataset_id.value
        oof, _folds, selections = retained.nested_oof(
            dataset,
            split_kind,
            variant,
            shapes,
            variant_dir,
            force=force,
        )
        fit_metrics = calibration.calibration_summary(dataset.target, oof)
        metrics.append(
            {
                "dataset": dataset_id.value,
                "model": variant_name,
                "split": "fit_oof",
                "parameter_count": retained.parameter_count(dataset, variant),
                **fit_metrics,
            }
        )
        for index, (observed, predicted) in enumerate(zip(dataset.target, oof, strict=True)):
            predictions.append(
                {
                    "dataset": dataset_id.value,
                    "model": variant_name,
                    "split": "fit_oof",
                    "row_id": str(dataset.frame.iloc[index].get("run_name", index)),
                    "group": str(dataset.frame.iloc[index].get("panel_source", "fit")),
                    "observed": observed,
                    "predicted": predicted,
                }
            )

        model, model_metadata = retained.fit_full_model(dataset, split_kind, variant, shapes)
        heldout_pred = heldout_prediction(dataset, model, heldout_weights)
        selected_index = int(np.argmin(heldout_pred))
        metrics.append(
            {
                "dataset": dataset_id.value,
                "model": variant_name,
                "split": "heldout",
                "parameter_count": retained.parameter_count(dataset, variant),
                **calibration.calibration_summary(heldout_target, heldout_pred),
                **base.grouped_heldout_summary(heldout_frame, heldout_target, heldout_pred),
                "selected_run": str(heldout_frame.iloc[selected_index]["wandb_run_name"]),
                "selected_observed": float(heldout_target[selected_index]),
                "selected_predicted": float(heldout_pred[selected_index]),
                "selected_optimism": float(heldout_target[selected_index] - heldout_pred[selected_index]),
            }
        )
        for index, (observed, predicted) in enumerate(zip(heldout_target, heldout_pred, strict=True)):
            predictions.append(
                {
                    "dataset": dataset_id.value,
                    "model": variant_name,
                    "split": "heldout",
                    "row_id": str(heldout_frame.iloc[index]["wandb_run_name"]),
                    "group": str(heldout_frame.iloc[index]["training_series"]),
                    "observed": observed,
                    "predicted": predicted,
                }
            )
        full_models[variant_name] = {
            **model_metadata,
            "variant": asdict(variant),
            "nested_cv_selections": selections,
        }
    return metrics, predictions, full_models


def render(metrics: pd.DataFrame, predictions: pd.DataFrame, output_dir: Path) -> None:
    colors = {
        "power_global_tau": "#d73027",
        "weibull_global_tau": "#fdae61",
        "compact_weibull_shared_replay": "#66bd63",
        "weibull_family_coverage_shared_replay": "#1a9850",
    }
    figure = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Uncheatable: frozen-heldout residuals",
            "Table-9: frozen-heldout residuals",
            "Fit-panel nested OOF RMSE",
            "Frozen-heldout RMSE",
        ),
    )
    dataset_ids = (base.DatasetId.DELPHI_3E18_UNCHEATABLE, base.DatasetId.DELPHI_3E18_TABLE9)
    for column, dataset_id in enumerate(dataset_ids, start=1):
        local_dataset = predictions.loc[predictions["dataset"].eq(dataset_id.value) & predictions["split"].eq("heldout")]
        for model in VARIANT_NAMES:
            local = local_dataset.loc[local_dataset["model"].eq(model)]
            if local.empty:
                continue
            figure.add_trace(
                go.Scatter(
                    x=local["observed"],
                    y=local["predicted"] - local["observed"],
                    mode="markers",
                    marker={"color": colors[model], "size": 5, "opacity": 0.45},
                    name=model,
                    legendgroup=model,
                    showlegend=column == 1,
                    customdata=np.column_stack([local["row_id"], local["group"]]),
                    hovertemplate=(
                        "%{customdata[0]}<br>%{customdata[1]}<br>observed=%{x:.5f}"
                        "<br>predicted-observed=%{y:.5f}<extra></extra>"
                    ),
                ),
                row=1,
                col=column,
            )
        figure.add_hline(y=0.0, line={"color": "#64748b", "dash": "dash"}, row=1, col=column)

    for column, split in enumerate(("fit_oof", "heldout"), start=1):
        local = metrics.loc[metrics["split"].eq(split)]
        for model in VARIANT_NAMES:
            selected = local.loc[local["model"].eq(model)]
            if selected.empty:
                continue
            figure.add_trace(
                go.Bar(
                    x=selected["dataset"],
                    y=selected["rmse"],
                    marker_color=colors[model],
                    name=model,
                    legendgroup=model,
                    showlegend=False,
                ),
                row=2,
                col=column,
            )
    figure.update_layout(
        title="Bounded retained learning and literal replay at 3e18",
        template="plotly_white",
        barmode="group",
        width=1600,
        height=1000,
        legend={"orientation": "h", "y": 1.08},
    )
    figure.update_xaxes(title_text="Observed BPB", row=1)
    figure.update_yaxes(title_text="Prediction residual (predicted - observed)", row=1)
    figure.update_yaxes(title_text="RMSE", row=2)
    figure.write_html(output_dir / "weibull_transfer.html", include_plotlyjs="cdn", config=PLOT_CONFIG)


def write_report(metrics: pd.DataFrame, output_dir: Path) -> None:
    columns = [
        "dataset",
        "model",
        "split",
        "parameter_count",
        "rmse",
        "spearman",
        "regret_at_1",
        "lower_tail_optimism",
        "optimism_gt_0p05_count",
        "worst_optimism",
        "selected_optimism",
    ]
    (output_dir / "report.md").write_text(
        "# Bounded retained-learning transfer\n\n"
        "All nonlinear shapes and ridge strengths are selected by nested fit-panel CV. Historical 3e18 validations "
        "are scored only after selection.\n\n" + metrics[columns].to_markdown(index=False, floatfmt=".6f") + "\n"
    )


def main() -> None:
    args = parse_args()
    dataset_ids = tuple(base.DatasetId(value) for value in args.datasets.split(",") if value)
    variant_names = tuple(value for value in args.variants.split(",") if value)
    unknown = set(variant_names) - set(retained.VARIANT_BY_NAME)
    if unknown:
        raise ValueError(f"Unknown variants: {sorted(unknown)}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metric_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    model_records: dict[str, Any] = {}
    for dataset_id in dataset_ids:
        metrics, predictions, models = fit_and_score(
            dataset_id,
            variant_names,
            args.num_shapes,
            args.output_dir,
            force=args.force,
        )
        metric_rows.extend(metrics)
        prediction_rows.extend(predictions)
        model_records[dataset_id.value] = models
    metrics = pd.DataFrame(metric_rows)
    predictions = pd.DataFrame(prediction_rows)
    metrics.to_csv(args.output_dir / "metrics.csv", index=False)
    predictions.to_csv(args.output_dir / "predictions.csv", index=False)
    (args.output_dir / "full_models.json").write_text(json.dumps(model_records, indent=2, allow_nan=False) + "\n")
    render(metrics, predictions, args.output_dir)
    write_report(metrics, args.output_dir)
    (args.output_dir / "summary.json").write_text(
        json.dumps(
            {
                "generated_at": datetime.now(UTC).isoformat(),
                "datasets": [dataset.value for dataset in dataset_ids],
                "models": list(variant_names),
                "selection": "nested fit-panel CV",
                "heldout_role": "frozen transfer diagnostic only",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
