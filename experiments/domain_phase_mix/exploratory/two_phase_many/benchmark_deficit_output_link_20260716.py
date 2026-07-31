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
"""Test whether a scaling-law output link fixes compressed poor-policy predictions.

The inverse-deficit GRP benchmark models BPB directly as an additive function
of exposure deficits. Neural scaling laws instead commonly model reducible
loss above an irreducible floor as a positive quantity. This benchmark keeps
the selected inverse-deficit response geometry fixed and compares

``Y = g(w)`` and ``Y = Y_inf + exp(g(w))``.

The irreducible-floor fraction and ridge strength are selected exclusively by
fit-panel OOF RMSE. Historical 3e18 validation rows remain frozen heldouts.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, replace
from enum import StrEnum
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.colors import sample_colorscale
from plotly.subplots import make_subplots
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
    benchmark_hierarchical_deficit_response_20260716 as deficit,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    fit_production_grp_quality_variants as family_grp,
)

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_SOURCE_METRICS = SCRIPT_DIR / "reference_outputs/hierarchical_deficit_response_20260716/metrics.csv"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "reference_outputs/deficit_output_link_20260716"
FLOOR_FRACTIONS = (0.0, 0.5, 0.75, 0.9, 0.95)
L2_GRID = (0.0, 1e-5, 1e-4, 1e-3, 1e-2, 0.1)
PLOT_CONFIG = {"toImageButtonOptions": {"format": "png", "scale": 4}}
DEFICIT_VARIANTS = (
    deficit.Variant.POWER_DEFICIT_HYBRID_REPLAY,
    deficit.Variant.POWER_DEFICIT_HYBRID_EARLY_FAMILY,
)


class Link(StrEnum):
    IDENTITY = "identity_raw_bpb"
    LOG_EXCESS = "log_reducible_bpb"


@dataclass(frozen=True)
class LinkConfig:
    link: Link
    floor_fraction: float
    l2: float


@dataclass(frozen=True)
class LinkModel:
    dataset: family_grp.Dataset
    deficit_config: deficit.Config
    link_config: LinkConfig
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
        return inverse_link(latent, self.link_config.link, self.floor)


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
        default=",".join(variant.value for variant in DEFICIT_VARIANTS),
        help="Comma-separated deficit variants present in source metrics.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def selected_deficit_config(
    dataset_id: base.DatasetId,
    variant: deficit.Variant,
    source_metrics: pd.DataFrame,
) -> deficit.Config:
    selected = source_metrics[
        source_metrics["dataset"].eq(dataset_id.value)
        & source_metrics["variant"].eq(variant.value)
        & source_metrics["split"].eq("fit_oof")
    ]
    if len(selected) != 1:
        raise ValueError(f"Expected one selected {variant.value} row for {dataset_id.value}; found {len(selected)}")
    row = selected.iloc[0]
    shape = family_grp.Shape(
        exponent=float(row["exponent"]),
        late_multiplier=float(row["late_multiplier"]),
        forgetting_rate=float(row["forgetting_rate"]),
        penalty_threshold=float(row["penalty_threshold"]),
        quality_discount=float(row["quality_discount"]),
    )
    base_config = base.Config(
        variant=base.Variant.HIERARCHICAL_PHASE_BUCKET_REPLAY,
        shape_index=int(row["shape_index"]),
        shape=shape,
        l2=float(row["l2"]),
        residual_shrink=float(row["residual_shrink"]),
        undercoverage_fraction=0.0,
        coverage_gate_ratio=0.0,
    )
    return deficit.Config(
        variant=variant,
        base=base_config,
        deficit_floor=float(row["deficit_floor"]),
        surplus_credit=float(row["surplus_credit"]),
    )


def transformed_target(target: np.ndarray, link: Link, floor: float) -> np.ndarray:
    if link is Link.IDENTITY:
        return target
    excess = target - floor
    if np.any(excess <= 0.0):
        raise ValueError("Reducible BPB must be positive")
    return np.log(excess)


def inverse_link(latent: np.ndarray, link: Link, floor: float) -> np.ndarray:
    if link is Link.IDENTITY:
        return latent
    return floor + np.exp(latent)


def fit_model(
    dataset: family_grp.Dataset,
    deficit_config: deficit.Config,
    link_config: LinkConfig,
    indices: np.ndarray,
) -> LinkModel:
    config = replace(deficit_config, base=replace(deficit_config.base, l2=link_config.l2))
    design = deficit.build_design(dataset, config)
    floor = 0.0 if link_config.link is Link.IDENTITY else link_config.floor_fraction * float(dataset.target.min())
    x = design.values[indices]
    transformed = transformed_target(dataset.target[indices], link_config.link, floor)
    x_mean = x.mean(axis=0, keepdims=True)
    target_mean = float(transformed.mean())
    centered_x = x - x_mean
    centered_target = transformed - target_mean
    if link_config.l2 > 0.0:
        ridge = np.sqrt(link_config.l2 * design.ridge_multipliers)
        centered_x = np.vstack([centered_x, np.diag(ridge)])
        centered_target = np.concatenate([centered_target, np.zeros(len(ridge), dtype=float)])
    coefficients, _residual = nnls(centered_x, centered_target, maxiter=40 * centered_x.shape[1])
    intercept = target_mean - float((x_mean @ coefficients).item())
    return LinkModel(dataset, config, link_config, floor, intercept, coefficients)


def oof_prediction(
    dataset: family_grp.Dataset,
    deficit_config: deficit.Config,
    link_config: LinkConfig,
    splits: list[tuple[np.ndarray, np.ndarray]],
) -> np.ndarray:
    prediction = np.full(dataset.n, np.nan, dtype=float)
    for train, test in splits:
        prediction[test] = fit_model(dataset, deficit_config, link_config, train).predict(dataset.weights[test])
    if not np.isfinite(prediction).all():
        raise RuntimeError(f"Incomplete OOF prediction for {link_config}")
    return prediction


def candidate_configs() -> tuple[LinkConfig, ...]:
    identity = [LinkConfig(Link.IDENTITY, 0.0, l2) for l2 in L2_GRID]
    log_excess = [
        LinkConfig(Link.LOG_EXCESS, floor_fraction, l2) for floor_fraction in FLOOR_FRACTIONS for l2 in L2_GRID
    ]
    return tuple(identity + log_excess)


def benchmark_dataset(
    dataset_id: base.DatasetId,
    deficit_variants: tuple[deficit.Variant, ...],
    source_metrics: pd.DataFrame,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    dataset = base.load_dataset(dataset_id)
    splits = base.split_indices(dataset, dataset_id, np.arange(dataset.n), base.SCREEN_SEED)
    screen_rows: list[dict[str, Any]] = []
    predictions: dict[tuple[deficit.Variant, LinkConfig], np.ndarray] = {}
    selected_deficit_configs = {
        variant: selected_deficit_config(dataset_id, variant, source_metrics) for variant in deficit_variants
    }
    for variant, deficit_config in selected_deficit_configs.items():
        for link_config in candidate_configs():
            prediction = oof_prediction(dataset, deficit_config, link_config, splits)
            predictions[(variant, link_config)] = prediction
            screen_rows.append(
                {
                    "dataset": dataset_id.value,
                    "deficit_variant": variant.value,
                    "link": link_config.link.value,
                    "floor_fraction": link_config.floor_fraction,
                    "l2": link_config.l2,
                    **calibration.calibration_summary(dataset.target, prediction),
                }
            )

    screen = pd.DataFrame(screen_rows)
    selected_configs: list[tuple[deficit.Variant, LinkConfig]] = []
    for variant in deficit_variants:
        for link in Link:
            local = screen[screen["deficit_variant"].eq(variant.value) & screen["link"].eq(link.value)].sort_values(
                ["rmse", "spearman"], ascending=[True, False]
            )
            row = local.iloc[0]
            selected_configs.append((variant, LinkConfig(link, float(row["floor_fraction"]), float(row["l2"]))))

    metric_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    heldout = base.heldout_data(dataset_id, dataset)
    for variant, link_config in selected_configs:
        deficit_config = selected_deficit_configs[variant]
        oof = predictions[(variant, link_config)]
        common = {
            "dataset": dataset_id.value,
            "deficit_variant": variant.value,
            "link": link_config.link.value,
            "floor_fraction": link_config.floor_fraction,
            "l2": link_config.l2,
        }
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
        model = fit_model(dataset, deficit_config, link_config, np.arange(dataset.n))
        heldout_prediction = model.predict(heldout_weights)
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


def render(
    predictions: pd.DataFrame,
    deficit_variants: tuple[deficit.Variant, ...],
    output_path: Path,
) -> None:
    datasets = (base.DatasetId.DELPHI_3E18_UNCHEATABLE, base.DatasetId.DELPHI_3E18_TABLE9)
    figure = make_subplots(rows=1, cols=2, subplot_titles=("Uncheatable", "Table-9"))
    trace_keys = tuple((variant, link) for variant in deficit_variants for link in Link)
    positions = np.linspace(0.05, 0.95, len(trace_keys))
    colors = dict(zip(trace_keys, sample_colorscale("RdYlGn_r", positions), strict=True))
    for column, dataset_id in enumerate(datasets, start=1):
        local = predictions[predictions["dataset"].eq(dataset_id.value) & predictions["split"].eq("heldout")]
        for variant in deficit_variants:
            for link in Link:
                rows = local[local["deficit_variant"].eq(variant.value) & local["link"].eq(link.value)]
                trace_id = f"{variant.value}:{link.value}"
                figure.add_trace(
                    go.Scatter(
                        x=rows["observed"],
                        y=rows["predicted"] - rows["observed"],
                        mode="markers",
                        name=trace_id.replace("inverse_power_deficit_", "").replace("_", " "),
                        legendgroup=trace_id,
                        showlegend=column == 1,
                        marker={"size": 7, "color": colors[(variant, link)], "opacity": 0.68},
                        customdata=np.column_stack([rows["row_id"], rows["predicted"]]),
                        hovertemplate="%{customdata[0]}<br>observed=%{x:.5f}<br>predicted=%{customdata[1]:.5f}<br>residual=%{y:+.5f}<extra></extra>",
                    ),
                    row=1,
                    col=column,
                )
        figure.add_hline(y=0.0, line={"color": "#263238", "dash": "dash"}, row=1, col=column)
        figure.update_xaxes(title_text="Observed BPB", row=1, col=column)
        figure.update_yaxes(
            title_text="Prediction residual (predicted - observed)" if column == 1 else None, row=1, col=column
        )
    figure.update_layout(
        title="Scaling-law output link: frozen 3e18 heldouts",
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
    deficit_variants = tuple(deficit.Variant(value) for value in args.deficit_variants.split(",") if value)
    metric_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    screen_rows: list[dict[str, Any]] = []
    for dataset_id in dataset_ids:
        metrics, predictions, screen = benchmark_dataset(dataset_id, deficit_variants, source_metrics)
        metric_rows.extend(metrics)
        prediction_rows.extend(predictions)
        screen_rows.extend(screen)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics = pd.DataFrame(metric_rows)
    predictions = pd.DataFrame(prediction_rows)
    metrics.to_csv(args.output_dir / "metrics.csv", index=False)
    predictions.to_csv(args.output_dir / "predictions.csv", index=False)
    pd.DataFrame(screen_rows).to_csv(args.output_dir / "hyperparameter_screen.csv", index=False)
    render(predictions, deficit_variants, args.output_dir / "output_link_calibration.html")
    report_columns = [
        "dataset",
        "deficit_variant",
        "link",
        "split",
        "floor_fraction",
        "l2",
        "rmse",
        "spearman",
        "regret_at_1",
        "calibration_slope_observed_on_predicted",
        "optimism_gt_0p05_count",
        "worst_optimism",
        "selected_optimism",
    ]
    report = "# Scaling-law output-link benchmark\n\n"
    report += (
        "The response geometry is fixed before this benchmark. "
        "Floor fraction and ridge are selected only on fit-panel OOF RMSE.\n\n"
    )
    available_report_columns = [column for column in report_columns if column in metrics.columns]
    report += metrics[available_report_columns].to_markdown(index=False, floatfmt=".6f")
    report += "\n"
    (args.output_dir / "report.md").write_text(report)
    summary = {
        "source_metrics": str(args.source_metrics),
        "output_dir": str(args.output_dir),
        "datasets": [dataset.value for dataset in dataset_ids],
        "deficit_variants": [variant.value for variant in deficit_variants],
        "floor_fractions": FLOOR_FRACTIONS,
        "l2_grid": L2_GRID,
        "metrics": metric_rows,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(report)


if __name__ == "__main__":
    main()
