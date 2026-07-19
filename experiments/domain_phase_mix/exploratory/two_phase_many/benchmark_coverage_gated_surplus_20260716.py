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
"""Test whether broad coverage should gate specialist-data benefits.

The inverse-deficit response assigns harm to underexposure and benefit to
overexposure with the same additive channel. This remains optimistic when a
policy concentrates on a few specialist buckets while starving many semantic
families. The proposed response keeps shortage harm unchanged but attenuates
surplus benefit by a generalized mean of clipped semantic-family coverage.

The gate adds no fitted coefficient. Its generalized-mean order and strength,
the BPB output link, and ridge strength are selected exclusively by fit-panel
OOF RMSE. Historical 3e18 validation rows remain frozen heldouts.
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
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "reference_outputs/coverage_gated_surplus_20260716"
GATE_ORDERS = (-4.0, -1.0, 0.0, 1.0)
GATE_POWERS = (0.5, 1.0, 2.0)
PLOT_CONFIG = {"toImageButtonOptions": {"format": "png", "scale": 4}}


class ModelClass(StrEnum):
    UNGATED = "ungated_surplus"
    FAMILY_TOTAL_GATED = "family_total_gated_surplus"
    MEMBER_COVERAGE_GATED = "member_coverage_gated_surplus"
    FOUNDATION_GATED = "foundation_gated_surplus"


@dataclass(frozen=True)
class GateConfig:
    model_class: ModelClass
    mean_order: float
    power: float


@dataclass(frozen=True)
class Config:
    gate: GateConfig
    link: output_link.LinkConfig


@dataclass(frozen=True)
class Model:
    dataset: family_grp.Dataset
    deficit_config: deficit.Config
    config: Config
    floor: float
    intercept: float
    coefficients: np.ndarray

    def predict(self, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        candidate = replace(
            self.dataset,
            weights=np.asarray(weights, dtype=float),
            target=np.zeros(len(weights), dtype=float),
        )
        design, gate = gated_design(candidate, self.deficit_config, self.config.gate)
        latent = np.asarray(self.intercept + design.values @ self.coefficients, dtype=float)
        return output_link.inverse_link(latent, self.config.link.link, self.floor), gate


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


def gate_configs() -> tuple[GateConfig, ...]:
    configs = [GateConfig(ModelClass.UNGATED, 0.0, 0.0)]
    configs.extend(
        GateConfig(model_class, order, power)
        for model_class in (ModelClass.FAMILY_TOTAL_GATED, ModelClass.MEMBER_COVERAGE_GATED)
        for order in GATE_ORDERS
        for power in GATE_POWERS
    )
    configs.extend(GateConfig(ModelClass.FOUNDATION_GATED, 0.0, power) for power in GATE_POWERS)
    return tuple(configs)


def generalized_mean(values: np.ndarray, order: float) -> np.ndarray:
    if order == 0.0:
        return np.exp(np.mean(np.log(values), axis=1))
    return np.mean(values**order, axis=1) ** (1.0 / order)


def semantic_coverage_gate(
    dataset: family_grp.Dataset,
    deficit_config: deficit.Config,
    gate_config: GateConfig,
) -> np.ndarray:
    if gate_config.model_class is ModelClass.UNGATED:
        return np.ones(dataset.n, dtype=float)
    exposure = base.retained_exposure(dataset, deficit_config.base.shape)
    reference = base.proportional_bucket_exposure(dataset, deficit_config.base.shape)
    if gate_config.model_class is ModelClass.FOUNDATION_GATED:
        if "broad_text" not in dataset.family_names:
            raise ValueError("Foundation-gated surplus requires a broad_text family")
        family_index = dataset.family_names.index("broad_text")
        members = dataset.family_members[family_index]
        coverage = exposure[:, members].sum(axis=1) / max(float(reference[members].sum()), 1e-12)
        return np.minimum(np.maximum(coverage, 0.0), 1.0) ** gate_config.power
    if gate_config.model_class is ModelClass.MEMBER_COVERAGE_GATED:
        ratio = exposure / np.maximum(reference[None, :], 1e-12)
    else:
        family_total = np.column_stack([exposure[:, members].sum(axis=1) for members in dataset.family_members])
        family_reference = np.asarray([reference[members].sum() for members in dataset.family_members], dtype=float)
        ratio = family_total / np.maximum(family_reference[None, :], 1e-12)
    clipped = np.minimum(np.maximum(ratio, 0.0), 1.0)
    floor = deficit_config.deficit_floor
    normalized = (clipped + floor) / (1.0 + floor)
    coverage = generalized_mean(normalized, gate_config.mean_order)
    return np.clip(coverage, 0.0, 1.0) ** gate_config.power


def gated_design(
    dataset: family_grp.Dataset,
    deficit_config: deficit.Config,
    gate_config: GateConfig,
) -> tuple[deficit.Design, np.ndarray]:
    design = deficit.build_design(dataset, deficit_config)
    gate = semantic_coverage_gate(dataset, deficit_config, gate_config)
    if gate_config.model_class is ModelClass.UNGATED:
        return design, gate
    values = design.values.copy()
    response_columns = [index for index, name in enumerate(design.names) if name.startswith("net_") or "_net_" in name]
    response = values[:, response_columns]
    values[:, response_columns] = np.maximum(response, 0.0) + gate[:, None] * np.minimum(response, 0.0)
    return deficit.Design(values, design.names, design.ridge_multipliers), gate


def fit_model(
    dataset: family_grp.Dataset,
    deficit_config: deficit.Config,
    config: Config,
    indices: np.ndarray,
) -> Model:
    design, _gate = gated_design(dataset, deficit_config, config.gate)
    floor = 0.0 if config.link.link is output_link.Link.IDENTITY else config.link.floor_fraction * dataset.target.min()
    transformed = output_link.transformed_target(dataset.target[indices], config.link.link, float(floor))
    x = design.values[indices]
    x_mean = x.mean(axis=0, keepdims=True)
    target_mean = float(transformed.mean())
    centered_x = x - x_mean
    centered_target = transformed - target_mean
    if config.link.l2 > 0.0:
        ridge = np.sqrt(config.link.l2 * design.ridge_multipliers)
        centered_x = np.vstack([centered_x, np.diag(ridge)])
        centered_target = np.concatenate([centered_target, np.zeros(len(ridge), dtype=float)])
    coefficients, _residual = nnls(centered_x, centered_target, maxiter=40 * centered_x.shape[1])
    intercept = target_mean - float((x_mean @ coefficients).item())
    return Model(dataset, deficit_config, config, float(floor), intercept, coefficients)


def oof_prediction(
    dataset: family_grp.Dataset,
    deficit_config: deficit.Config,
    config: Config,
    splits: list[tuple[np.ndarray, np.ndarray]],
) -> np.ndarray:
    prediction = np.full(dataset.n, np.nan, dtype=float)
    for train, test in splits:
        prediction[test] = fit_model(dataset, deficit_config, config, train).predict(dataset.weights[test])[0]
    if not np.isfinite(prediction).all():
        raise RuntimeError(f"Incomplete OOF prediction for {config}")
    return prediction


def candidate_configs() -> tuple[Config, ...]:
    return tuple(Config(gate, link) for gate in gate_configs() for link in output_link.candidate_configs())


def config_record(config: Config) -> dict[str, str | float]:
    return {
        "model_class": config.gate.model_class.value,
        "gate_mean_order": config.gate.mean_order,
        "gate_power": config.gate.power,
        "link": config.link.link.value,
        "floor_fraction": config.link.floor_fraction,
        "l2": config.link.l2,
    }


def selected_config(screen: pd.DataFrame, variant: deficit.Variant, model_class: ModelClass) -> Config:
    rows = screen[screen["deficit_variant"].eq(variant.value) & screen["model_class"].eq(model_class.value)].sort_values(
        ["rmse", "spearman"], ascending=[True, False]
    )
    row = rows.iloc[0]
    return Config(
        GateConfig(model_class, float(row["gate_mean_order"]), float(row["gate_power"])),
        output_link.LinkConfig(
            output_link.Link(str(row["link"])),
            float(row["floor_fraction"]),
            float(row["l2"]),
        ),
    )


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
    oof_predictions: dict[tuple[deficit.Variant, Config], np.ndarray] = {}
    for variant, deficit_config in deficit_configs.items():
        for config in candidate_configs():
            prediction = oof_prediction(dataset, deficit_config, config, splits)
            oof_predictions[(variant, config)] = prediction
            screen_rows.append(
                {
                    "dataset": dataset_id.value,
                    "deficit_variant": variant.value,
                    **config_record(config),
                    **calibration.calibration_summary(dataset.target, prediction),
                }
            )
    screen = pd.DataFrame(screen_rows)
    selections = [
        (variant, selected_config(screen, variant, model_class)) for variant in variants for model_class in ModelClass
    ]
    heldout = base.heldout_data(dataset_id, dataset)
    metric_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    for variant, config in selections:
        common = {"dataset": dataset_id.value, "deficit_variant": variant.value, **config_record(config)}
        oof = oof_predictions[(variant, config)]
        fit_gate = semantic_coverage_gate(dataset, deficit_configs[variant], config.gate)
        metric_rows.append(
            {
                **common,
                "split": "fit_oof",
                **calibration.calibration_summary(dataset.target, oof),
                "median_coverage_gate": float(np.median(fit_gate)),
                "minimum_coverage_gate": float(fit_gate.min()),
            }
        )
        for index, (observed, predicted) in enumerate(zip(dataset.target, oof, strict=True)):
            prediction_rows.append(
                {
                    **common,
                    "split": "fit_oof",
                    "row_id": str(dataset.frame.iloc[index].get("run_name", index)),
                    "group": str(dataset.frame.iloc[index].get("panel_source", "fit")),
                    "observed": observed,
                    "predicted": predicted,
                    "coverage_gate": fit_gate[index],
                }
            )
        if heldout is None:
            continue
        heldout_frame, heldout_weights, heldout_target = heldout
        model = fit_model(dataset, deficit_configs[variant], config, np.arange(dataset.n))
        heldout_prediction, heldout_gate = model.predict(heldout_weights)
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
                "selected_coverage_gate": float(heldout_gate[selected_index]),
                "median_coverage_gate": float(np.median(heldout_gate)),
                "minimum_coverage_gate": float(heldout_gate.min()),
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
                    "coverage_gate": heldout_gate[index],
                }
            )
    return metric_rows, prediction_rows, screen_rows


def render(predictions: pd.DataFrame, output_path: Path) -> None:
    datasets = (base.DatasetId.DELPHI_3E18_UNCHEATABLE, base.DatasetId.DELPHI_3E18_TABLE9)
    figure = make_subplots(rows=1, cols=2, subplot_titles=("Uncheatable", "Table-9"))
    positions = np.linspace(0.05, 0.95, len(ModelClass))
    colors = dict(zip((model.value for model in ModelClass), sample_colorscale("RdYlGn_r", positions), strict=True))
    for column, dataset_id in enumerate(datasets, start=1):
        local = predictions[predictions["dataset"].eq(dataset_id.value) & predictions["split"].eq("heldout")]
        for model_class in ModelClass:
            rows = local[local["model_class"].eq(model_class.value)]
            figure.add_trace(
                go.Scatter(
                    x=rows["observed"],
                    y=rows["predicted"] - rows["observed"],
                    mode="markers",
                    name=model_class.value.replace("_", " "),
                    legendgroup=model_class.value,
                    showlegend=column == 1,
                    marker={"size": 7, "color": colors[model_class.value], "opacity": 0.65},
                    customdata=np.column_stack([rows["row_id"], rows["predicted"], rows["coverage_gate"]]),
                    hovertemplate=(
                        "%{customdata[0]}<br>observed=%{x:.5f}<br>predicted=%{customdata[1]:.5f}"
                        "<br>residual=%{y:+.5f}<br>coverage gate=%{customdata[2]:.3f}<extra></extra>"
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
        title="Coverage-gated specialist benefit: frozen 3e18 heldouts",
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
    render(predictions, args.output_dir / "coverage_gated_surplus.html")
    report_columns = [
        "dataset",
        "model_class",
        "split",
        "gate_mean_order",
        "gate_power",
        "link",
        "floor_fraction",
        "l2",
        "rmse",
        "spearman",
        "regret_at_1",
        "optimism_gt_0p05_count",
        "worst_optimism",
        "selected_optimism",
        "selected_coverage_gate",
    ]
    available = [column for column in report_columns if column in metrics.columns]
    report = "# Coverage-gated specialist-benefit benchmark\n\n"
    report += (
        "Gate geometry, BPB output link, and ridge are selected only on fit-panel OOF RMSE. "
        "The gate changes no fitted coefficient count.\n\n"
    )
    report += metrics[available].to_markdown(index=False, floatfmt=".6f")
    report += "\n"
    (args.output_dir / "report.md").write_text(report)
    summary = {
        "source_metrics": str(args.source_metrics),
        "datasets": [dataset.value for dataset in dataset_ids],
        "deficit_variants": [variant.value for variant in variants],
        "gate_orders": GATE_ORDERS,
        "gate_powers": GATE_POWERS,
        "metrics": metric_rows,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(report)


if __name__ == "__main__":
    main()
