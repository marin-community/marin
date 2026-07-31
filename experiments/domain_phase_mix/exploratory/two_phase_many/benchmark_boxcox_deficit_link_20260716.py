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
"""Benchmark a scaling-law Box-Cox link for inverse-deficit surrogates.

The remaining heldout residual trend indicates compressed predictions at high
BPB. A Box-Cox link generalizes direct BPB regression and log reducible-loss
regression without adding response features or fitted coefficients:

``z = ((Y - Y_inf)**lambda - 1) / lambda``.

``lambda=1`` is affine raw-BPB regression and ``lambda=0`` is log reducible
BPB. Link curvature, irreducible floor, and ridge are selected exclusively by
fit-panel OOF RMSE; historical 3e18 validations remain frozen heldouts.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.optimize import nnls

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_deficit_output_link_20260716 as output_link,
)
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
    fit_production_grp_quality_variants as family_grp,
)

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_SOURCE_METRICS = SCRIPT_DIR / "reference_outputs/hierarchical_deficit_response_20260716/metrics.csv"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "reference_outputs/boxcox_deficit_link_20260716"
LINK_POWERS = (-1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0)
FLOOR_FRACTIONS = (0.0, 0.5, 0.75, 0.9, 0.95)
L2_GRID = (0.0, 1e-5, 1e-4, 1e-3, 1e-2, 0.1)
PLOT_CONFIG = {"toImageButtonOptions": {"format": "png", "scale": 4}}


@dataclass(frozen=True)
class Config:
    power: float
    floor_fraction: float
    l2: float


@dataclass(frozen=True)
class Model:
    dataset: family_grp.Dataset
    deficit_config: deficit.Config
    config: Config
    floor: float
    intercept: float
    coefficients: np.ndarray

    def predict(self, weights: np.ndarray) -> np.ndarray:
        candidate = replace(
            self.dataset,
            weights=np.asarray(weights, dtype=float),
            target=np.zeros(len(weights), dtype=float),
        )
        design = deficit.build_design(candidate, self.deficit_config)
        latent = np.asarray(self.intercept + design.values @ self.coefficients, dtype=float)
        return inverse_link(latent, self.config.power, self.floor)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-metrics", type=Path, default=DEFAULT_SOURCE_METRICS)
    parser.add_argument(
        "--datasets",
        default="delphi_3e18_uncheatable,delphi_3e18_table9",
        help="Comma-separated dataset IDs.",
    )
    parser.add_argument(
        "--deficit-variants",
        default=deficit.Variant.POWER_DEFICIT_HYBRID_REPLAY.value,
        help="Comma-separated deficit variants present in source metrics.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def transform_target(target: np.ndarray, power: float, floor: float) -> np.ndarray:
    excess = target - floor
    if np.any(excess <= 0.0):
        raise ValueError("Reducible BPB must be positive")
    if power == 0.0:
        return np.log(excess)
    return (excess**power - 1.0) / power


def inverse_link(latent: np.ndarray, power: float, floor: float) -> np.ndarray:
    if power == 0.0:
        return floor + np.exp(latent)
    base_value = 1.0 + power * latent
    if power < 0.0:
        base_value = np.maximum(base_value, 1e-8)
    else:
        base_value = np.maximum(base_value, 0.0)
    return floor + base_value ** (1.0 / power)


def fit_model(
    dataset: family_grp.Dataset,
    deficit_config: deficit.Config,
    config: Config,
    indices: np.ndarray,
) -> Model:
    tuned_deficit = replace(deficit_config, base=replace(deficit_config.base, l2=config.l2))
    design = deficit.build_design(dataset, tuned_deficit)
    floor = config.floor_fraction * float(dataset.target.min())
    transformed = transform_target(dataset.target[indices], config.power, floor)
    x = design.values[indices]
    x_mean = x.mean(axis=0, keepdims=True)
    target_mean = float(transformed.mean())
    centered_x = x - x_mean
    centered_target = transformed - target_mean
    if config.l2 > 0.0:
        ridge = np.sqrt(config.l2 * design.ridge_multipliers)
        centered_x = np.vstack([centered_x, np.diag(ridge)])
        centered_target = np.concatenate([centered_target, np.zeros(len(ridge), dtype=float)])
    coefficients, _residual = nnls(centered_x, centered_target, maxiter=40 * centered_x.shape[1])
    intercept = target_mean - float((x_mean @ coefficients).item())
    return Model(dataset, tuned_deficit, config, floor, intercept, coefficients)


def oof_prediction(
    dataset: family_grp.Dataset,
    deficit_config: deficit.Config,
    config: Config,
    splits: list[tuple[np.ndarray, np.ndarray]],
) -> np.ndarray:
    prediction = np.full(dataset.n, np.nan, dtype=float)
    for train, test in splits:
        prediction[test] = fit_model(dataset, deficit_config, config, train).predict(dataset.weights[test])
    if not np.isfinite(prediction).all():
        raise RuntimeError(f"Incomplete OOF prediction for {config}")
    return prediction


def candidate_configs() -> tuple[Config, ...]:
    return tuple(
        Config(power, floor_fraction, l2)
        for power in LINK_POWERS
        for floor_fraction in ((0.0,) if power == 1.0 else FLOOR_FRACTIONS)
        for l2 in L2_GRID
    )


def config_record(config: Config) -> dict[str, float]:
    return {"link_power": config.power, "floor_fraction": config.floor_fraction, "l2": config.l2}


def benchmark_dataset(
    dataset_id: base.DatasetId,
    variants: tuple[deficit.Variant, ...],
    source_metrics: pd.DataFrame,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    dataset = base.load_dataset(dataset_id)
    splits = base.split_indices(dataset, dataset_id, np.arange(dataset.n), base.SCREEN_SEED)
    deficit_configs = {
        variant: output_link.selected_deficit_config(dataset_id, variant, source_metrics) for variant in variants
    }
    screen_rows: list[dict[str, Any]] = []
    predictions: dict[tuple[deficit.Variant, Config], np.ndarray] = {}
    selected: list[tuple[deficit.Variant, Config]] = []
    for variant, deficit_config in deficit_configs.items():
        for config in candidate_configs():
            prediction = oof_prediction(dataset, deficit_config, config, splits)
            predictions[(variant, config)] = prediction
            screen_rows.append(
                {
                    "dataset": dataset_id.value,
                    "deficit_variant": variant.value,
                    **config_record(config),
                    **calibration.calibration_summary(dataset.target, prediction),
                }
            )
        rows = pd.DataFrame(screen_rows)
        best = (
            rows[rows["deficit_variant"].eq(variant.value)]
            .sort_values(["rmse", "spearman"], ascending=[True, False])
            .iloc[0]
        )
        selected.append(
            (
                variant,
                Config(float(best["link_power"]), float(best["floor_fraction"]), float(best["l2"])),
            )
        )

    heldout = base.heldout_data(dataset_id, dataset)
    metric_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    for variant, config in selected:
        common = {"dataset": dataset_id.value, "deficit_variant": variant.value, **config_record(config)}
        oof = predictions[(variant, config)]
        metric_rows.append({**common, "split": "fit_oof", **calibration.calibration_summary(dataset.target, oof)})
        for index, (observed, predicted) in enumerate(zip(dataset.target, oof, strict=True)):
            prediction_rows.append(
                {
                    **common,
                    "split": "fit_oof",
                    "row_id": str(dataset.frame.iloc[index].get("run_name", index)),
                    "group": str(dataset.frame.iloc[index].get("panel_source", "fit")),
                    "observed": observed,
                    "predicted": predicted,
                }
            )
        if heldout is None:
            continue
        heldout_frame, heldout_weights, heldout_target = heldout
        heldout_prediction = fit_model(dataset, deficit_configs[variant], config, np.arange(dataset.n)).predict(
            heldout_weights
        )
        selected_index = int(np.argmin(heldout_prediction))
        metric_rows.append(
            {
                **common,
                "split": "heldout",
                **calibration.calibration_summary(heldout_target, heldout_prediction),
                **base.grouped_heldout_summary(heldout_frame, heldout_target, heldout_prediction),
                "selected_run": str(heldout_frame.iloc[selected_index]["wandb_run_name"]),
                "selected_observed": float(heldout_target[selected_index]),
                "selected_predicted": float(heldout_prediction[selected_index]),
                "selected_optimism": float(heldout_target[selected_index] - heldout_prediction[selected_index]),
            }
        )
        for index, (observed, predicted) in enumerate(zip(heldout_target, heldout_prediction, strict=True)):
            prediction_rows.append(
                {
                    **common,
                    "split": "heldout",
                    "row_id": str(heldout_frame.iloc[index]["wandb_run_name"]),
                    "group": str(heldout_frame.iloc[index]["training_series"]),
                    "observed": observed,
                    "predicted": predicted,
                }
            )
    return metric_rows, prediction_rows, screen_rows


def render(predictions: pd.DataFrame, output_path: Path) -> None:
    datasets = (base.DatasetId.DELPHI_3E18_UNCHEATABLE, base.DatasetId.DELPHI_3E18_TABLE9)
    figure = make_subplots(rows=1, cols=2, subplot_titles=("Uncheatable", "Table-9"))
    for column, dataset_id in enumerate(datasets, start=1):
        rows = predictions[predictions["dataset"].eq(dataset_id.value) & predictions["split"].eq("heldout")]
        power = float(rows["link_power"].iloc[0])
        figure.add_trace(
            go.Scatter(
                x=rows["observed"],
                y=rows["predicted"] - rows["observed"],
                mode="markers",
                name=f"{dataset_id.value}: lambda={power:g}",
                showlegend=True,
                marker={"size": 7, "color": "#1a9850", "opacity": 0.68},
                customdata=np.column_stack([rows["row_id"], rows["predicted"]]),
                hovertemplate=(
                    "%{customdata[0]}<br>observed=%{x:.5f}<br>predicted=%{customdata[1]:.5f}"
                    "<br>residual=%{y:+.5f}<extra></extra>"
                ),
            ),
            row=1,
            col=column,
        )
        figure.add_hline(y=0.0, line={"color": "#263238", "dash": "dash"}, row=1, col=column)
        figure.update_xaxes(title_text="Observed BPB", row=1, col=column)
        figure.update_yaxes(
            title_text="Prediction residual (predicted - observed)" if column == 1 else None,
            row=1,
            col=column,
        )
    figure.update_layout(
        title="OOF-selected Box-Cox link: frozen 3e18 heldouts",
        template="plotly_white",
        width=1500,
        height=650,
        legend={"orientation": "h", "y": -0.18},
        margin={"l": 80, "r": 30, "t": 90, "b": 110},
    )
    figure.write_html(output_path, include_plotlyjs="cdn", config=PLOT_CONFIG)


def main() -> None:
    args = parse_args()
    source_metrics = pd.read_csv(args.source_metrics)
    dataset_ids = tuple(base.DatasetId(value) for value in args.datasets.split(",") if value)
    variants = tuple(deficit.Variant(value) for value in args.deficit_variants.split(",") if value)
    metric_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    screen_rows: list[dict[str, Any]] = []
    for dataset_id in dataset_ids:
        metrics, predictions, screen = benchmark_dataset(dataset_id, variants, source_metrics)
        metric_rows.extend(metrics)
        prediction_rows.extend(predictions)
        screen_rows.extend(screen)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics = pd.DataFrame(metric_rows)
    predictions = pd.DataFrame(prediction_rows)
    metrics.to_csv(args.output_dir / "metrics.csv", index=False)
    predictions.to_csv(args.output_dir / "predictions.csv", index=False)
    pd.DataFrame(screen_rows).to_csv(args.output_dir / "hyperparameter_screen.csv", index=False)
    render(predictions, args.output_dir / "boxcox_link_calibration.html")
    report_columns = [
        "dataset",
        "split",
        "link_power",
        "floor_fraction",
        "l2",
        "rmse",
        "spearman",
        "regret_at_1",
        "optimism_gt_0p05_count",
        "worst_optimism",
        "selected_optimism",
    ]
    available = [column for column in report_columns if column in metrics.columns]
    report = "# Scaling-law Box-Cox output-link benchmark\n\n"
    report += "Link curvature, floor, and ridge are selected only on fit-panel OOF RMSE.\n\n"
    report += metrics[available].to_markdown(index=False, floatfmt=".6f")
    report += "\n"
    (args.output_dir / "report.md").write_text(report)
    summary = {
        "source_metrics": str(args.source_metrics),
        "datasets": [dataset.value for dataset in dataset_ids],
        "deficit_variants": [variant.value for variant in variants],
        "link_powers": LINK_POWERS,
        "floor_fractions": FLOOR_FRACTIONS,
        "l2_grid": L2_GRID,
        "metrics": metric_rows,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(report)


if __name__ == "__main__":
    main()
