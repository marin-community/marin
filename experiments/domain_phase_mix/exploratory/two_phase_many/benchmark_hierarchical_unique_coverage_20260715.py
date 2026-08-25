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
"""Test occupancy-derived unique-data coverage in hierarchical phase-replay GRP.

For bucket size ``N_i`` and realized epoch count ``e_i``, independent sampling
exposes approximately ``N_i (1 - exp(-e_i))`` unique examples. Concentrating a
fixed token budget in a few buckets therefore reduces unique-data coverage even
when their effective exposure is high. This benchmark adds that mechanism to
the promoted hierarchical phase-replay model without an entropy prior or a
post-hoc calibrator.

All settings are selected on fit-panel cross-validation. Historical 3e18
validations are scored only after selection as frozen transfer diagnostics.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.optimize import nnls

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
    fit_production_grp_quality_variants as family_grp,
)

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "reference_outputs/hierarchical_unique_coverage_20260715"
PLOT_CONFIG = {"toImageButtonOptions": {"format": "png", "scale": 4}}


class Variant(StrEnum):
    CURRENT = "current_phase_replay"
    GLOBAL_UNIQUE = "global_unique_coverage"
    FAMILY_UNIQUE = "family_unique_coverage"
    FAMILY_UNIQUE_BOTTLENECK = "family_unique_coverage_bottleneck"


@dataclass(frozen=True)
class Config:
    variant: Variant
    base: base.Config


@dataclass(frozen=True)
class Design:
    values: np.ndarray
    names: tuple[str, ...]
    ridge_multipliers: np.ndarray


@dataclass(frozen=True)
class Model:
    dataset: family_grp.Dataset
    config: Config
    intercept: float
    coefficients: np.ndarray

    def predict(self, weights: np.ndarray) -> np.ndarray:
        candidate = replace(
            self.dataset,
            weights=np.asarray(weights, dtype=float),
            target=np.zeros(len(weights), dtype=float),
        )
        design = build_design(candidate, self.config)
        return np.asarray(self.intercept + design.values @ self.coefficients, dtype=float)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        default="delphi_3e18_uncheatable,delphi_3e18_table9",
        help="Comma-separated dataset IDs.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--num-shapes", type=int, default=12)
    parser.add_argument("--top-shapes", type=int, default=1)
    return parser.parse_args()


def unique_mass(dataset: family_grp.Dataset, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    epochs = weights[:, 0, :] * dataset.c0[None, :] + weights[:, 1, :] * dataset.c1[None, :]
    bucket_size = 1.0 / np.maximum(dataset.c0 + dataset.c1, 1e-12)
    bucket_unique = bucket_size[None, :] * -np.expm1(-np.maximum(epochs, 0.0))
    family_unique = np.column_stack([bucket_unique[:, members].sum(axis=1) for members in dataset.family_members])
    return bucket_unique, family_unique


def proportional_unique_mass(dataset: family_grp.Dataset) -> tuple[float, np.ndarray]:
    weights = base.proportional_weights(dataset)
    reference = np.stack([weights, weights], axis=0)[None, :, :]
    bucket_unique, family_unique = unique_mass(dataset, reference)
    return float(bucket_unique.sum()), family_unique[0]


def build_design(dataset: family_grp.Dataset, config: Config) -> Design:
    current = base.build_design(dataset, config.base)
    if config.variant is Variant.CURRENT:
        return Design(current.values, current.names, current.ridge_multipliers)

    bucket_unique, family_unique = unique_mass(dataset, dataset.weights)
    reference_total, reference_family = proportional_unique_mass(dataset)
    total_deficit = np.maximum(1.0 - bucket_unique.sum(axis=1) / reference_total, 0.0) ** 2
    family_deficit = np.maximum(1.0 - family_unique / np.maximum(reference_family[None, :], 1e-12), 0.0) ** 2
    pieces = [current.values]
    names = list(current.names)
    ridge = list(current.ridge_multipliers)
    if config.variant is Variant.GLOBAL_UNIQUE:
        pieces.append(total_deficit[:, None])
        names.append("global_unique_data_deficit")
        ridge.append(1.0)
    else:
        pieces.append(family_deficit)
        names.extend(f"family_unique_data_deficit:{name}" for name in dataset.family_names)
        ridge.extend([1.0] * len(dataset.family_names))
        if config.variant is Variant.FAMILY_UNIQUE_BOTTLENECK:
            pieces.append(np.max(family_deficit, axis=1, keepdims=True))
            names.append("weakest_family_unique_data_deficit")
            ridge.append(1.0)
    return Design(np.hstack(pieces), tuple(names), np.asarray(ridge, dtype=float))


def fit_model(dataset: family_grp.Dataset, config: Config, indices: np.ndarray) -> Model:
    design = build_design(dataset, config)
    x = design.values[indices]
    y = dataset.target[indices]
    x_mean = x.mean(axis=0, keepdims=True)
    y_mean = float(y.mean())
    centered_x = x - x_mean
    centered_y = y - y_mean
    if config.base.l2 > 0.0:
        ridge = np.sqrt(config.base.l2 * design.ridge_multipliers)
        centered_x = np.vstack([centered_x, np.diag(ridge)])
        centered_y = np.concatenate([centered_y, np.zeros(len(ridge), dtype=float)])
    coefficients, _residual = nnls(centered_x, centered_y, maxiter=40 * centered_x.shape[1])
    intercept = y_mean - float((x_mean @ coefficients).item())
    return Model(dataset, config, intercept, coefficients)


def oof_prediction(
    dataset: family_grp.Dataset,
    config: Config,
    splits: list[tuple[np.ndarray, np.ndarray]],
) -> np.ndarray:
    prediction = np.full(dataset.n, np.nan, dtype=float)
    for train, test in splits:
        prediction[test] = fit_model(dataset, config, train).predict(dataset.weights[test])
    if not np.isfinite(prediction).all():
        raise RuntimeError(f"Incomplete OOF prediction for {config.variant}")
    return prediction


def config_record(config: Config, metrics: dict[str, float | int]) -> dict[str, Any]:
    return {
        "variant": config.variant.value,
        "shape_index": config.base.shape_index,
        **asdict(config.base.shape),
        "l2": config.base.l2,
        "residual_shrink": config.base.residual_shrink,
        **metrics,
    }


def score_configs(
    dataset: family_grp.Dataset,
    configs: list[Config],
    splits: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[Config, np.ndarray, list[dict[str, Any]]]:
    best: tuple[float, float, Config, np.ndarray] | None = None
    rows: list[dict[str, Any]] = []
    for config in configs:
        prediction = oof_prediction(dataset, config, splits)
        metrics = calibration.calibration_summary(dataset.target, prediction)
        rows.append(config_record(config, metrics))
        candidate = (float(metrics["rmse"]), -float(metrics["spearman"]), config, prediction)
        if best is None or candidate[:2] < best[:2]:
            best = candidate
    if best is None:
        raise RuntimeError("No unique-coverage configurations were scored")
    return best[2], best[3], rows


def selected_shape_indices(rows: list[dict[str, Any]], top_shapes: int) -> list[int]:
    best_by_shape: dict[int, float] = {}
    for row in rows:
        shape_index = int(row["shape_index"])
        best_by_shape[shape_index] = min(best_by_shape.get(shape_index, float("inf")), float(row["rmse"]))
    return [index for index, _rmse in sorted(best_by_shape.items(), key=lambda item: item[1])[:top_shapes]]


def benchmark_dataset(
    dataset_id: base.DatasetId,
    num_shapes: int,
    top_shapes: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    dataset = base.load_dataset(dataset_id)
    shapes = family_grp.shape_candidates(family_grp.Variant.BUCKET_RESOLVED, num_shapes)
    splits = base.split_indices(dataset, dataset_id, np.arange(dataset.n), base.SCREEN_SEED)
    _baseline, _prediction, baseline_rows = base.score_configs(dataset, base.baseline_configs(shapes), splits)
    shape_indices = selected_shape_indices(baseline_rows, top_shapes)
    base_configs = base.structural_configs(base.Variant.HIERARCHICAL_PHASE_BUCKET_REPLAY, shapes, shape_indices)

    selected: dict[Variant, tuple[Config, np.ndarray]] = {}
    screen_rows: list[dict[str, Any]] = []
    for variant in Variant:
        print(f"  screening {dataset_id.value}: {variant.value}", flush=True)
        config, prediction, rows = score_configs(dataset, [Config(variant, item) for item in base_configs], splits)
        selected[variant] = (config, prediction)
        screen_rows.extend({"dataset": dataset_id.value, **row} for row in rows)

    metric_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    heldout = base.heldout_data(dataset_id, dataset)
    for variant, (config, prediction) in selected.items():
        metric_rows.append(
            {
                "dataset": dataset_id.value,
                "variant": variant.value,
                "split": "fit_oof",
                **config_record(config, calibration.calibration_summary(dataset.target, prediction)),
            }
        )
        if heldout is None:
            continue
        frame, weights, target = heldout
        heldout_prediction = fit_model(dataset, config, np.arange(dataset.n)).predict(weights)
        selected_index = int(np.argmin(heldout_prediction))
        metric_rows.append(
            {
                "dataset": dataset_id.value,
                "variant": variant.value,
                "split": "heldout",
                **config_record(config, calibration.calibration_summary(target, heldout_prediction)),
                **base.grouped_heldout_summary(frame, target, heldout_prediction),
                "selected_run": str(frame.iloc[selected_index]["wandb_run_name"]),
                "selected_observed": float(target[selected_index]),
                "selected_predicted": float(heldout_prediction[selected_index]),
                "selected_optimism": float(target[selected_index] - heldout_prediction[selected_index]),
            }
        )
        for index, (observed, predicted) in enumerate(zip(target, heldout_prediction, strict=True)):
            prediction_rows.append(
                {
                    "dataset": dataset_id.value,
                    "variant": variant.value,
                    "row_id": str(frame.iloc[index]["wandb_run_name"]),
                    "group": str(frame.iloc[index]["training_series"]),
                    "observed": observed,
                    "predicted": predicted,
                }
            )
    return metric_rows, screen_rows, prediction_rows


def render(predictions: pd.DataFrame, metrics: pd.DataFrame, output_dir: Path) -> None:
    colors = {
        Variant.CURRENT.value: "#d73027",
        Variant.GLOBAL_UNIQUE.value: "#fee08b",
        Variant.FAMILY_UNIQUE.value: "#66bd63",
        Variant.FAMILY_UNIQUE_BOTTLENECK.value: "#1a9850",
    }
    for dataset in predictions["dataset"].unique():
        figure = go.Figure()
        local_dataset = predictions.loc[predictions["dataset"].eq(dataset)]
        for variant in Variant:
            local = local_dataset.loc[local_dataset["variant"].eq(variant.value)]
            figure.add_trace(
                go.Scatter(
                    x=local["observed"],
                    y=local["predicted"] - local["observed"],
                    mode="markers",
                    marker={"color": colors[variant.value], "size": 7, "opacity": 0.55},
                    name=variant.value,
                    customdata=np.column_stack([local["row_id"], local["group"]]),
                    hovertemplate=(
                        "%{customdata[0]}<br>%{customdata[1]}<br>observed=%{x:.5f}"
                        "<br>predicted-observed=%{y:.5f}<extra></extra>"
                    ),
                )
            )
        figure.add_hline(y=0.0, line={"color": "#64748b", "dash": "dash"})
        figure.update_layout(
            title=f"{dataset}: occupancy-derived unique-data coverage",
            template="plotly_white",
            width=1500,
            height=900,
            xaxis_title="Observed BPB",
            yaxis_title="Prediction residual (predicted - observed)",
            legend={"orientation": "h", "y": 1.08},
        )
        figure.write_html(output_dir / f"{dataset}_calibration.html", include_plotlyjs="cdn", config=PLOT_CONFIG)

    columns = [
        "dataset",
        "variant",
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
        "# Occupancy-derived unique-data coverage\n\n"
        "All structural settings were selected on fit-panel OOF predictions before frozen heldout scoring.\n\n"
        + metrics[columns].to_markdown(index=False, floatfmt=".6f")
        + "\n"
    )


def main() -> None:
    args = parse_args()
    dataset_ids = tuple(base.DatasetId(value) for value in args.datasets.split(",") if value)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metric_rows: list[dict[str, Any]] = []
    screen_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    for dataset_id in dataset_ids:
        metrics, screens, predictions = benchmark_dataset(dataset_id, args.num_shapes, args.top_shapes)
        metric_rows.extend(metrics)
        screen_rows.extend(screens)
        prediction_rows.extend(predictions)
    metrics = pd.DataFrame(metric_rows)
    predictions = pd.DataFrame(prediction_rows)
    metrics.to_csv(args.output_dir / "metrics.csv", index=False)
    pd.DataFrame(screen_rows).to_csv(args.output_dir / "hyperparameter_screen.csv", index=False)
    predictions.to_csv(args.output_dir / "predictions.csv", index=False)
    render(predictions, metrics, args.output_dir)
    (args.output_dir / "summary.json").write_text(
        json.dumps(
            {
                "generated_at": datetime.now(UTC).isoformat(),
                "datasets": [dataset.value for dataset in dataset_ids],
                "selection": "fit-panel five-fold OOF RMSE, Spearman tie-break",
                "heldout_role": "frozen transfer diagnostic only",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
