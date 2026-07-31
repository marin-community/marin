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
"""Test exact Table-9 component decomposition with inverse-deficit GRP.

Table-9 macro BPB is the unweighted mean of 51 component BPBs. A scalar
surrogate can trade an unmodeled failure on one component against unrelated
components. This benchmark instead fits one non-negative inverse-deficit head
per component and averages the 51 predictions exactly. Nonlinear response
geometry is shared across components; an ablation additionally tunes only L2
and hierarchical residual shrinkage per component using fit-panel CV.

Historical 3e18 validations are frozen transfer data and never select a model
or hyperparameter.
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
    benchmark_hierarchical_deficit_response_20260716 as deficit,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    fit_olmo_base_easy_paper_faithful_olmix_300m as paper_olmix,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    fit_production_grp_quality_variants as family_grp,
)

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "reference_outputs/deficit_component_heads_20260716"
PLOT_CONFIG = {"toImageButtonOptions": {"format": "png", "scale": 4}}


@dataclass(frozen=True)
class ComponentFit:
    component: str
    config: deficit.Config
    oof_prediction: np.ndarray
    model: deficit.Model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--num-shapes", type=int, default=12)
    parser.add_argument("--top-shape-floor-pairs", type=int, default=3)
    return parser.parse_args()


def fit_component(
    dataset: family_grp.Dataset,
    component: str,
    shared_config: deficit.Config,
    splits: list[tuple[np.ndarray, np.ndarray]],
    *,
    tune_regularization: bool,
) -> tuple[ComponentFit, list[dict[str, Any]]]:
    component_dataset = replace(dataset, target=dataset.frame[component].to_numpy(dtype=float))
    if tune_regularization:
        configs = [
            replace(
                shared_config,
                base=replace(shared_config.base, l2=l2, residual_shrink=residual_shrink),
            )
            for l2 in base.L2_GRID
            for residual_shrink in base.RESIDUAL_SHRINK_GRID
        ]
        config, oof_prediction, rows = deficit.score_configs(component_dataset, configs, splits)
    else:
        config = shared_config
        oof_prediction = deficit.oof_prediction(component_dataset, config, splits)
        rows = [
            deficit.config_record(
                config,
                calibration.calibration_summary(component_dataset.target, oof_prediction),
            )
        ]
    model = deficit.fit_model(component_dataset, config, np.arange(component_dataset.n))
    return ComponentFit(component, config, oof_prediction, model), rows


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
        selected = int(np.argmin(predicted))
        row.update(base.grouped_heldout_summary(heldout_frame, observed, predicted))
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
        "aggregate_inverse_deficit": "#d73027",
        "component_inverse_deficit_shared": "#fdae61",
        "component_inverse_deficit_tuned_regularization": "#1a9850",
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
        title="Table-9 inverse-deficit aggregate vs exact component decomposition",
        template="plotly_white",
        width=1500,
        height=900,
        xaxis_title="Observed Table-9 macro BPB",
        yaxis_title="Prediction residual (predicted - observed)",
        legend={"orientation": "h", "y": 1.08},
    )
    figure.write_html(output_dir / "component_deficit_calibration.html", include_plotlyjs="cdn", config=PLOT_CONFIG)

    columns = [
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
        "# Inverse-deficit Table-9 component-head benchmark\n\n"
        "All nonlinear response geometry is selected on the fit panel. The tuned-component ablation changes only "
        "ridge and hierarchical residual pooling per component. Frozen heldouts do not select any setting.\n\n"
        + metrics[columns].to_markdown(index=False, floatfmt=".6f")
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
    if not np.allclose(dataset.frame[components].mean(axis=1), dataset.target, atol=1e-9):
        raise ValueError("Stored Table-9 macro is not the exact mean of the 51 components")

    shapes = family_grp.shape_candidates(family_grp.Variant.BUCKET_RESOLVED, args.num_shapes)
    splits = base.split_indices(dataset, dataset_id, np.arange(dataset.n), base.SCREEN_SEED)
    configs, first_stage = deficit.deficit_configs(
        deficit.Variant.POWER_DEFICIT_HYBRID_REPLAY,
        shapes,
        args.top_shape_floor_pairs,
        dataset,
        splits,
    )
    shared_config, aggregate_oof, aggregate_screen = deficit.score_configs(dataset, configs, splits)
    aggregate_model = deficit.fit_model(dataset, shared_config, np.arange(dataset.n))

    families: dict[str, list[ComponentFit]] = {}
    screen_rows: list[dict[str, Any]] = [
        *(
            {"model": "aggregate_inverse_deficit", "component": "macro", "stage": "shape_floor", **row}
            for row in first_stage
        ),
        *(
            {"model": "aggregate_inverse_deficit", "component": "macro", "stage": "full", **row}
            for row in aggregate_screen
        ),
    ]
    for label, tune_regularization in (
        ("component_inverse_deficit_shared", False),
        ("component_inverse_deficit_tuned_regularization", True),
    ):
        fits: list[ComponentFit] = []
        for index, component in enumerate(components, start=1):
            print(f"{label}: {index}/{len(components)} {component}", flush=True)
            fit, rows = fit_component(
                dataset,
                component,
                shared_config,
                splits,
                tune_regularization=tune_regularization,
            )
            fits.append(fit)
            screen_rows.extend({"model": label, "component": component, "stage": "head", **row} for row in rows)
        families[label] = fits

    heldout = base.heldout_data(dataset_id, dataset)
    if heldout is None:
        raise RuntimeError("Delphi 3e18 heldouts are unavailable")
    heldout_frame, heldout_weights, heldout_target = heldout
    fit_predictions = {
        "aggregate_inverse_deficit": aggregate_oof,
        **{label: prediction_matrix(fits, dataset.weights, oof=True).mean(axis=1) for label, fits in families.items()},
    }
    heldout_predictions = {
        "aggregate_inverse_deficit": aggregate_model.predict(heldout_weights),
        **{label: prediction_matrix(fits, heldout_weights, oof=False).mean(axis=1) for label, fits in families.items()},
    }

    metric_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    for split, frame, observed, predictions in (
        ("fit_oof", dataset.frame, dataset.target, fit_predictions),
        ("heldout", heldout_frame, heldout_target, heldout_predictions),
    ):
        for model, prediction in predictions.items():
            metric_rows.append(
                metric_row(model, split, observed, prediction, heldout_frame if split == "heldout" else None)
            )
            for index, (actual, estimate) in enumerate(zip(observed, prediction, strict=True)):
                prediction_rows.append(
                    {
                        "model": model,
                        "split": split,
                        "row_id": str(frame.iloc[index].get("wandb_run_name", frame.iloc[index].get("run_name", index))),
                        "group": str(
                            frame.iloc[index].get("training_series", frame.iloc[index].get("panel_source", split))
                        ),
                        "observed": actual,
                        "predicted": estimate,
                    }
                )

    metrics = pd.DataFrame(metric_rows)
    predictions = pd.DataFrame(prediction_rows)
    metrics.to_csv(args.output_dir / "metrics.csv", index=False)
    predictions.to_csv(args.output_dir / "predictions.csv", index=False)
    pd.DataFrame(screen_rows).to_csv(args.output_dir / "hyperparameter_screen.csv", index=False)
    selected_rows = []
    for model, fits in families.items():
        selected_rows.extend(
            {
                "model": model,
                "component": fit.component,
                "variant": fit.config.variant.value,
                "shape_index": fit.config.base.shape_index,
                **asdict(fit.config.base.shape),
                "l2": fit.config.base.l2,
                "residual_shrink": fit.config.base.residual_shrink,
                "deficit_floor": fit.config.deficit_floor,
            }
            for fit in fits
        )
    pd.DataFrame(selected_rows).to_csv(args.output_dir / "selected_component_configs.csv", index=False)
    render(predictions, metrics, args.output_dir)
    (args.output_dir / "summary.json").write_text(
        json.dumps(
            {
                "generated_at": datetime.now(UTC).isoformat(),
                "component_count": len(components),
                "shared_config": deficit.config_record(shared_config, {}),
                "selection": "fit-panel five-fold OOF RMSE only",
                "heldout_role": "frozen transfer diagnostic only",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
