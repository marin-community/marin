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
"""Test component-decomposed hierarchical GRP on the Delphi 3e18 Table-9 panel.

The Table-9 macro is exactly the unweighted mean of 51 smooth BPB components.
One aggregate head can therefore hide a severe component failure behind gains on
unrelated components. This benchmark keeps the promoted hierarchical exposure
form fixed, fits one non-negative head per component, and averages predictions
using the metric's exact definition.

Nonlinear settings are selected on fit-panel cross-validation only. Historical
3e18 validations remain a frozen transfer diagnostic.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go

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
    fit_olmo_base_easy_paper_faithful_olmix_300m as paper_olmix,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    fit_production_grp_quality_variants as family_grp,
)

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "reference_outputs/hierarchical_component_heads_20260715"
PLOT_CONFIG = {"toImageButtonOptions": {"format": "png", "scale": 4}}


@dataclass(frozen=True)
class ComponentFit:
    component: str
    config: base.Config
    oof_prediction: np.ndarray
    model: base.Model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--num-shapes", type=int, default=12)
    parser.add_argument("--top-shapes", type=int, default=1)
    return parser.parse_args()


def selected_shape_indices(rows: list[dict[str, Any]], top_shapes: int) -> list[int]:
    best_by_shape: dict[int, float] = {}
    for row in rows:
        shape_index = int(row["shape_index"])
        best_by_shape[shape_index] = min(best_by_shape.get(shape_index, float("inf")), float(row["rmse"]))
    return [index for index, _rmse in sorted(best_by_shape.items(), key=lambda item: item[1])[:top_shapes]]


def fit_component(
    dataset: family_grp.Dataset,
    component: str,
    shapes: tuple[family_grp.Shape, ...],
    splits: list[tuple[np.ndarray, np.ndarray]],
    shape_indices: list[int] | None,
    top_shapes: int,
) -> tuple[ComponentFit, list[dict[str, Any]]]:
    component_dataset = replace(dataset, target=dataset.frame[component].to_numpy(dtype=float))
    screen_rows: list[dict[str, Any]] = []
    if shape_indices is None:
        _baseline, _baseline_prediction, rows = base.score_configs(
            component_dataset,
            base.baseline_configs(shapes),
            splits,
        )
        screen_rows.extend({"stage": "shape", **row} for row in rows)
        local_shape_indices = selected_shape_indices(rows, top_shapes)
    else:
        local_shape_indices = shape_indices
    config, prediction, rows = base.score_configs(
        component_dataset,
        base.structural_configs(base.Variant.HIERARCHICAL_PHASE_BUCKET_REPLAY, shapes, local_shape_indices),
        splits,
    )
    screen_rows.extend({"stage": "head", **row} for row in rows)
    model = base.fit_model(component_dataset, config, np.arange(component_dataset.n))
    return ComponentFit(component, config, prediction, model), screen_rows


def fit_component_family(
    label: str,
    dataset: family_grp.Dataset,
    components: list[str],
    shapes: tuple[family_grp.Shape, ...],
    splits: list[tuple[np.ndarray, np.ndarray]],
    shape_indices: list[int] | None,
    top_shapes: int,
) -> tuple[list[ComponentFit], list[dict[str, Any]]]:
    fits: list[ComponentFit] = []
    screen_rows: list[dict[str, Any]] = []
    for index, component in enumerate(components, start=1):
        print(f"  {label}: component {index}/{len(components)} {component}", flush=True)
        fit, rows = fit_component(dataset, component, shapes, splits, shape_indices, top_shapes)
        fits.append(fit)
        screen_rows.extend({"model": label, "component": component, **row} for row in rows)
    return fits, screen_rows


def prediction_matrix(fits: list[ComponentFit], weights: np.ndarray, *, oof: bool) -> np.ndarray:
    if oof:
        return np.column_stack([fit.oof_prediction for fit in fits])
    return np.column_stack([fit.model.predict(weights) for fit in fits])


def metric_row(
    model: str,
    split: str,
    observed: np.ndarray,
    predicted: np.ndarray,
    heldout_frame: pd.DataFrame | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "model": model,
        "split": split,
        **calibration.calibration_summary(observed, predicted),
    }
    if heldout_frame is not None:
        row.update(base.grouped_heldout_summary(heldout_frame, observed, predicted))
        selected = int(np.argmin(predicted))
        row.update(
            {
                "selected_run": str(heldout_frame.iloc[selected]["wandb_run_name"]),
                "selected_observed": float(observed[selected]),
                "selected_predicted": float(predicted[selected]),
                "selected_optimism": float(observed[selected] - predicted[selected]),
            }
        )
    return row


def render(predictions: pd.DataFrame, metrics: pd.DataFrame, output_dir: Path) -> None:
    colors = {
        "aggregate_head": "#d73027",
        "component_heads_shared_shape": "#fee08b",
        "component_heads_independent_shape": "#1a9850",
    }
    figure = go.Figure()
    heldout = predictions.loc[predictions["split"].eq("heldout")]
    for model, color in colors.items():
        local = heldout.loc[heldout["model"].eq(model)]
        figure.add_trace(
            go.Scatter(
                x=local["observed"],
                y=local["predicted"] - local["observed"],
                mode="markers",
                marker={"color": color, "size": 7, "opacity": 0.55},
                name=model,
                customdata=np.column_stack([local["row_id"], local["group"]]),
                hovertemplate=(
                    "%{customdata[0]}<br>%{customdata[1]}<br>observed=%{x:.5f}"
                    "<br>predicted-observed=%{y:.5f}<extra></extra>"
                ),
            )
        )
    figure.add_hline(y=0.0, line={"color": "#64748b", "dash": "dash"})
    figure.update_layout(
        title="Table-9 aggregate head vs exact 51-component decomposition",
        template="plotly_white",
        width=1500,
        height=900,
        xaxis_title="Observed Table-9 macro BPB",
        yaxis_title="Prediction residual (predicted - observed)",
        legend={"orientation": "h", "y": 1.08},
    )
    figure.write_html(output_dir / "component_head_calibration.html", include_plotlyjs="cdn", config=PLOT_CONFIG)

    metric_columns = [
        "model",
        "split",
        "rmse",
        "spearman",
        "regret_at_1",
        "calibration_slope_observed_on_predicted",
        "optimism_gt_0p05_count",
        "worst_optimism",
        "selected_optimism",
    ]
    (output_dir / "report.md").write_text(
        "# Hierarchical component-head screen\n\n"
        "The macro target is the exact unweighted mean of the 51 component predictions. "
        "Component and aggregate hyperparameters are selected using the fit panel only; "
        "heldouts are frozen transfer data.\n\n"
        + metrics[metric_columns].to_markdown(index=False, floatfmt=".6f")
        + "\n"
    )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dataset_id = base.DatasetId.DELPHI_3E18_TABLE9
    dataset = base.load_dataset(dataset_id)
    components = paper_olmix.table9_component_order()
    missing = sorted(set(components).difference(dataset.frame.columns))
    if missing:
        raise ValueError(f"Missing Table-9 components: {missing}")
    component_macro = dataset.frame[components].mean(axis=1).to_numpy(dtype=float)
    if not np.allclose(component_macro, dataset.target, atol=1e-9):
        raise ValueError("Stored Table-9 macro is not the exact mean of the 51 components")

    shapes = family_grp.shape_candidates(family_grp.Variant.BUCKET_RESOLVED, args.num_shapes)
    splits = base.split_indices(dataset, dataset_id, np.arange(dataset.n), base.SCREEN_SEED)
    aggregate_baseline, _prediction, aggregate_rows = base.score_configs(
        dataset,
        base.baseline_configs(shapes),
        splits,
    )
    aggregate_shape_indices = selected_shape_indices(aggregate_rows, args.top_shapes)
    aggregate_config, aggregate_oof, aggregate_screen = base.score_configs(
        dataset,
        base.structural_configs(
            base.Variant.HIERARCHICAL_PHASE_BUCKET_REPLAY,
            shapes,
            aggregate_shape_indices,
        ),
        splits,
    )
    del aggregate_baseline
    aggregate_model = base.fit_model(dataset, aggregate_config, np.arange(dataset.n))

    shared_fits, shared_screen = fit_component_family(
        "component_heads_shared_shape",
        dataset,
        components,
        shapes,
        splits,
        aggregate_shape_indices,
        args.top_shapes,
    )
    independent_fits, independent_screen = fit_component_family(
        "component_heads_independent_shape",
        dataset,
        components,
        shapes,
        splits,
        None,
        args.top_shapes,
    )

    heldout = base.heldout_data(dataset_id, dataset)
    if heldout is None:
        raise RuntimeError("Delphi 3e18 heldouts are unavailable")
    heldout_frame, heldout_weights, heldout_target = heldout
    fit_predictions = {
        "aggregate_head": aggregate_oof,
        "component_heads_shared_shape": prediction_matrix(shared_fits, dataset.weights, oof=True).mean(axis=1),
        "component_heads_independent_shape": prediction_matrix(independent_fits, dataset.weights, oof=True).mean(axis=1),
    }
    heldout_predictions = {
        "aggregate_head": aggregate_model.predict(heldout_weights),
        "component_heads_shared_shape": prediction_matrix(shared_fits, heldout_weights, oof=False).mean(axis=1),
        "component_heads_independent_shape": (
            prediction_matrix(independent_fits, heldout_weights, oof=False).mean(axis=1)
        ),
    }

    metric_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    for model, prediction in fit_predictions.items():
        metric_rows.append(metric_row(model, "fit_oof", dataset.target, prediction))
        for index, (observed, predicted) in enumerate(zip(dataset.target, prediction, strict=True)):
            prediction_rows.append(
                {
                    "model": model,
                    "split": "fit_oof",
                    "row_id": str(dataset.frame.iloc[index]["run_name"]),
                    "group": str(dataset.frame.iloc[index]["panel_source"]),
                    "observed": observed,
                    "predicted": predicted,
                }
            )
    for model, prediction in heldout_predictions.items():
        metric_rows.append(metric_row(model, "heldout", heldout_target, prediction, heldout_frame))
        for index, (observed, predicted) in enumerate(zip(heldout_target, prediction, strict=True)):
            prediction_rows.append(
                {
                    "model": model,
                    "split": "heldout",
                    "row_id": str(heldout_frame.iloc[index]["wandb_run_name"]),
                    "group": str(heldout_frame.iloc[index]["training_series"]),
                    "observed": observed,
                    "predicted": predicted,
                }
            )

    metrics = pd.DataFrame(metric_rows)
    predictions = pd.DataFrame(prediction_rows)
    metrics.to_csv(args.output_dir / "metrics.csv", index=False)
    predictions.to_csv(args.output_dir / "predictions.csv", index=False)
    screen_rows = [
        *({"model": "aggregate_head", "component": "macro", "stage": "shape", **row} for row in aggregate_rows),
        *({"model": "aggregate_head", "component": "macro", "stage": "head", **row} for row in aggregate_screen),
        *shared_screen,
        *independent_screen,
    ]
    pd.DataFrame(screen_rows).to_csv(args.output_dir / "hyperparameter_screen.csv", index=False)
    component_rows = []
    for model, fits in (
        ("component_heads_shared_shape", shared_fits),
        ("component_heads_independent_shape", independent_fits),
    ):
        component_rows.extend(
            {
                "model": model,
                "component": fit.component,
                "variant": fit.config.variant.value,
                "shape_index": fit.config.shape_index,
                **asdict(fit.config.shape),
                "l2": fit.config.l2,
                "residual_shrink": fit.config.residual_shrink,
            }
            for fit in fits
        )
    pd.DataFrame(component_rows).to_csv(args.output_dir / "selected_component_configs.csv", index=False)
    render(predictions, metrics, args.output_dir)
    (args.output_dir / "summary.json").write_text(
        json.dumps(
            {
                "generated_at": datetime.now(UTC).isoformat(),
                "component_count": len(components),
                "selection": "fit-panel five-fold OOF RMSE per component",
                "heldout_role": "frozen transfer diagnostic only",
                "shared_shape_indices": aggregate_shape_indices,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
