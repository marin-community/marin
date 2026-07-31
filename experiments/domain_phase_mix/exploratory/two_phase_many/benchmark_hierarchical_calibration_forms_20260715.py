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
# ]
# ///
"""Screen mechanistic fixes for optimistic high-loss mixture predictions.

The promoted hierarchical phase-replay GRP extrapolates too much utility from
mixtures that concentrate exposure in a few related buckets. This benchmark
changes one mechanism at a time:

* finite Hill responses cap the utility obtainable from repeated exposure;
* a breadth gate makes specialized utility conditional on retaining coverage
  across semantic families;
* convex member replay prevents an extreme member from being diluted by a
  family average.
* novelty-discounted retained exposure assigns a globally learned fraction of
  the value of novel coverage to additional retained repeats.
* joint family undercoverage represents compounding loss when several semantic
  families are simultaneously absent;
* family peak replay prevents one severely repeated member from being diluted
  by a family average.
* literal replay counts actual passes beyond one epoch independently of the
  retained-learning state and its fitted phase value.

All nonlinear settings and ridge strengths are selected using only fit-panel
cross-validation. Historical 3e18 validations are a frozen transfer diagnostic
and never participate in model or hyperparameter selection.
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
from plotly.subplots import make_subplots
from scipy.optimize import nnls

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_hierarchical_coverage_grp_20260715 as base,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    fit_production_grp_quality_variants as family_grp,
)

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "reference_outputs/hierarchical_calibration_forms_20260715"
SATURATION_ONSET_GRID = (1.0, 2.0, 4.0, 8.0)
BREADTH_GAMMA_GRID = (0.5, 1.0, 2.0, 4.0)
REPLAY_POWER_GRID = (2.0, 4.0)
REPEAT_DISCOUNT_GRID = (0.0, 0.01, 0.03, 0.1, 0.3, 0.6, 1.0)
PLOT_CONFIG = {"toImageButtonOptions": {"format": "png", "scale": 4}}


class Variant(StrEnum):
    CURRENT = "current_phase_replay"
    POWER_BREADTH_GATE = "power_breadth_gate"
    HILL = "finite_hill"
    HILL_BREADTH_GATE = "finite_hill_breadth_gate"
    HILL_BREADTH_GATE_CONVEX_REPLAY = "finite_hill_breadth_gate_convex_replay"
    NOVELTY_DISCOUNTED = "novelty_discounted_retained_state"
    NOVELTY_DISCOUNTED_CONVEX_REPLAY = "novelty_discounted_retained_state_convex_replay"
    JOINT_UNDERCOVERAGE = "joint_family_undercoverage"
    FAMILY_PEAK_REPLAY = "family_peak_replay"
    JOINT_UNDERCOVERAGE_PEAK_REPLAY = "joint_family_undercoverage_peak_replay"
    GLOBAL_LITERAL_REPLAY = "global_literal_replay"
    FAMILY_LITERAL_REPLAY = "family_literal_replay"

    @property
    def finite_hill(self) -> bool:
        return self in {
            Variant.HILL,
            Variant.HILL_BREADTH_GATE,
            Variant.HILL_BREADTH_GATE_CONVEX_REPLAY,
        }

    @property
    def breadth_gate(self) -> bool:
        return self in {
            Variant.POWER_BREADTH_GATE,
            Variant.HILL_BREADTH_GATE,
            Variant.HILL_BREADTH_GATE_CONVEX_REPLAY,
        }

    @property
    def convex_replay(self) -> bool:
        return self in {
            Variant.HILL_BREADTH_GATE_CONVEX_REPLAY,
            Variant.NOVELTY_DISCOUNTED_CONVEX_REPLAY,
        }

    @property
    def novelty_discounted(self) -> bool:
        return self in {
            Variant.NOVELTY_DISCOUNTED,
            Variant.NOVELTY_DISCOUNTED_CONVEX_REPLAY,
        }

    @property
    def joint_undercoverage(self) -> bool:
        return self in {
            Variant.JOINT_UNDERCOVERAGE,
            Variant.JOINT_UNDERCOVERAGE_PEAK_REPLAY,
        }

    @property
    def family_peak_replay(self) -> bool:
        return self in {
            Variant.FAMILY_PEAK_REPLAY,
            Variant.JOINT_UNDERCOVERAGE_PEAK_REPLAY,
        }

    @property
    def literal_replay(self) -> bool:
        return self in {
            Variant.GLOBAL_LITERAL_REPLAY,
            Variant.FAMILY_LITERAL_REPLAY,
        }


@dataclass(frozen=True)
class Config:
    variant: Variant
    shape_index: int
    shape: family_grp.Shape
    l2: float
    residual_shrink: float
    saturation_onset: float
    breadth_gamma: float
    replay_power: float
    repeat_discount: float


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
        default=",".join(dataset.value for dataset in base.DatasetId),
        help="Comma-separated dataset IDs.",
    )
    parser.add_argument(
        "--variants",
        default=",".join(variant.value for variant in Variant),
        help="Comma-separated structural variants.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--num-shapes", type=int, default=12)
    parser.add_argument("--top-shapes", type=int, default=1)
    return parser.parse_args()


def hill_response(exposure: np.ndarray, exponent: float, onset: np.ndarray | float) -> np.ndarray:
    powered = np.maximum(exposure, 0.0) ** exponent
    onset_powered = np.asarray(onset, dtype=float) ** exponent
    return powered / np.maximum(powered + onset_powered, 1e-12)


def utility_response(
    exposure: np.ndarray,
    config: Config,
    onset: np.ndarray | float,
) -> np.ndarray:
    if config.variant.finite_hill:
        return hill_response(exposure, config.shape.exponent, onset)
    return base.power_response(exposure, config.shape.exponent)


def breadth_gate(
    dataset: family_grp.Dataset,
    exposure: np.ndarray,
    family_total: np.ndarray,
    config: Config,
) -> tuple[np.ndarray, np.ndarray]:
    if not config.variant.breadth_gate:
        return np.ones(dataset.n, dtype=float), np.zeros(dataset.n, dtype=float)
    reference = base.proportional_family_exposure(dataset, config.shape)
    ratio = family_total / np.maximum(reference[None, :], 1e-12)
    deficit = np.maximum(1.0 - ratio, 0.0)
    bottleneck = np.sqrt(np.mean(deficit**2, axis=1))
    return np.exp(-config.breadth_gamma * bottleneck), bottleneck


def family_member_replay(
    bucket_harm: np.ndarray,
    members: tuple[np.ndarray, ...],
    power: float,
) -> np.ndarray:
    if power == 1.0:
        return np.column_stack([bucket_harm[:, indices].mean(axis=1) for indices in members])
    return np.column_stack([np.mean(bucket_harm[:, indices] ** power, axis=1) ** (1.0 / power) for indices in members])


def novelty_discounted_state(
    dataset: family_grp.Dataset,
    retained: np.ndarray,
    repeat_discount: float,
) -> np.ndarray:
    """Discount retained passes after expected unique coverage is exhausted."""
    phase0_epochs = dataset.weights[:, 0, :] * dataset.c0[None, :]
    phase1_epochs = dataset.weights[:, 1, :] * dataset.c1[None, :]
    unique_coverage = -np.expm1(-np.maximum(phase0_epochs + phase1_epochs, 0.0))
    novel_state = np.minimum(retained, unique_coverage)
    repeated_state = np.maximum(retained - unique_coverage, 0.0)
    return novel_state + repeat_discount * repeated_state


def build_design(dataset: family_grp.Dataset, config: Config) -> Design:
    retained = base.retained_exposure(dataset, config.shape)
    utility_exposure = (
        novelty_discounted_state(dataset, retained, config.repeat_discount)
        if config.variant.novelty_discounted
        else retained
    )
    utility_family_total = np.column_stack(
        [utility_exposure[:, members].sum(axis=1) for members in dataset.family_members]
    )
    replay_family_total = np.column_stack([retained[:, members].sum(axis=1) for members in dataset.family_members])
    family_sizes = np.asarray([len(members) for members in dataset.family_members], dtype=float)
    bucket_signal = utility_response(utility_exposure, config, config.saturation_onset)
    family_signal = utility_response(
        utility_family_total,
        config,
        config.saturation_onset * family_sizes[None, :],
    )
    gate, bottleneck = breadth_gate(dataset, utility_exposure, utility_family_total, config)
    bucket_signal = bucket_signal * gate[:, None]
    family_signal = family_signal * gate[:, None]

    pieces: list[np.ndarray] = []
    names: list[str] = []
    ridge: list[float] = []
    singleton = [members[0] for members in dataset.family_members if len(members) == 1]
    if singleton:
        pieces.append(-bucket_signal[:, singleton])
        names.extend(f"singleton_utility:{dataset.domains[index]}" for index in singleton)
        ridge.extend([1.0] * len(singleton))
    nonsingleton = [
        (family_name, members)
        for family_name, members in zip(dataset.family_names, dataset.family_members, strict=True)
        if len(members) > 1
    ]
    for family_name, members in nonsingleton:
        pieces.append(-bucket_signal[:, members].sum(axis=1, keepdims=True))
        names.append(f"pooled_family_utility:{family_name}")
        ridge.append(1.0)
    if nonsingleton:
        residual_members = np.concatenate([members for _name, members in nonsingleton])
        pieces.append(-bucket_signal[:, residual_members])
        names.extend(f"bucket_excess_utility:{dataset.domains[index]}" for index in residual_members)
        ridge.extend([config.residual_shrink] * len(residual_members))
        nonsingleton_indices = [index for index, members in enumerate(dataset.family_members) if len(members) > 1]
        pieces.append(-family_signal[:, nonsingleton_indices])
        names.extend(f"family_coverage_utility:{dataset.family_names[index]}" for index in nonsingleton_indices)
        ridge.extend([1.0] * len(nonsingleton_indices))

    pieces.append(base.overexposure_harm(replay_family_total, config.shape.penalty_threshold))
    names.extend(f"family_total_replay:{name}" for name in dataset.family_names)
    ridge.extend([1.0] * len(dataset.family_names))

    bucket_harm = base.overexposure_harm(retained, config.shape.penalty_threshold)
    pieces.append(family_member_replay(bucket_harm, dataset.family_members, config.replay_power))
    names.extend(f"family_member_replay:{name}" for name in dataset.family_names)
    ridge.extend([1.0] * len(dataset.family_names))

    if config.variant.family_peak_replay:
        pieces.append(np.column_stack([bucket_harm[:, members].max(axis=1) for members in dataset.family_members]))
        names.extend(f"family_peak_replay:{name}" for name in dataset.family_names)
        ridge.extend([1.0] * len(dataset.family_names))

    if config.variant.joint_undercoverage:
        reference = base.proportional_family_exposure(dataset, config.shape)
        normalized = replay_family_total / np.maximum(reference[None, :], 1e-12)
        deficit = np.maximum(1.0 - normalized, 0.0)
        pair_count = max(deficit.shape[1] * (deficit.shape[1] - 1) / 2.0, 1.0)
        pairwise_deficit = ((deficit.sum(axis=1) ** 2 - np.sum(deficit**2, axis=1)) / 2.0) / pair_count
        pieces.append(pairwise_deficit[:, None])
        names.append("joint_family_undercoverage")
        ridge.append(1.0)

    if config.variant.literal_replay:
        actual_epochs = dataset.weights[:, 0, :] * dataset.c0[None, :] + dataset.weights[:, 1, :] * dataset.c1[None, :]
        literal_replay = np.maximum(actual_epochs - 1.0, 0.0) ** 2
        if config.variant is Variant.GLOBAL_LITERAL_REPLAY:
            pieces.append(literal_replay.sum(axis=1, keepdims=True))
            names.append("global_literal_replay")
            ridge.append(1.0)
        else:
            pieces.append(
                np.column_stack([literal_replay[:, members].sum(axis=1) for members in dataset.family_members])
            )
            names.extend(f"family_literal_replay:{name}" for name in dataset.family_names)
            ridge.extend([1.0] * len(dataset.family_names))

    phase0_weight = dataset.weights[:, 0, :]
    phase1_weight = dataset.weights[:, 1, :]
    pieces.append(0.5 * np.abs(phase0_weight - phase1_weight).sum(axis=1, keepdims=True))
    names.append("phase_shift_tv")
    ridge.append(1.0)

    if config.variant.breadth_gate:
        pieces.append(bottleneck[:, None])
        names.append("global_family_coverage_bottleneck")
        ridge.append(1.0)

    return Design(np.hstack(pieces), tuple(names), np.asarray(ridge, dtype=float))


def fit_model(dataset: family_grp.Dataset, config: Config, indices: np.ndarray) -> Model:
    design = build_design(dataset, config)
    train_design = design.values[indices]
    train_target = dataset.target[indices]
    design_mean = train_design.mean(axis=0, keepdims=True)
    target_mean = float(train_target.mean())
    centered_design = train_design - design_mean
    centered_target = train_target - target_mean
    if config.l2 > 0.0:
        ridge = np.sqrt(config.l2 * design.ridge_multipliers)
        centered_design = np.vstack([centered_design, np.diag(ridge)])
        centered_target = np.concatenate([centered_target, np.zeros(len(ridge), dtype=float)])
    coefficients, _residual = nnls(centered_design, centered_target, maxiter=40 * centered_design.shape[1])
    intercept = target_mean - float((design_mean @ coefficients).item())
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


def calibration_summary(observed: np.ndarray, prediction: np.ndarray) -> dict[str, float | int]:
    residual = prediction - observed
    observed_range = float(np.ptp(observed))
    predicted_range = float(np.ptp(prediction))
    calibration_slope = float(np.polyfit(prediction, observed, 1)[0]) if predicted_range > 1e-12 else float("nan")
    residual_observed_slope = float(np.polyfit(observed, residual, 1)[0]) if observed_range > 1e-12 else float("nan")
    optimism = -residual
    return {
        **base.metric_summary(observed, prediction),
        "calibration_slope_observed_on_predicted": calibration_slope,
        "residual_observed_slope": residual_observed_slope,
        "prediction_range_ratio": predicted_range / max(observed_range, 1e-12),
        "optimism_gt_0p05_count": int(np.sum(optimism > 0.05)),
        "optimism_gt_0p10_count": int(np.sum(optimism > 0.10)),
        "worst_optimism": float(np.max(optimism)),
        "p90_optimism": float(np.quantile(optimism, 0.9)),
    }


def config_record(config: Config, metrics: dict[str, float | int]) -> dict[str, Any]:
    return {
        "variant": config.variant.value,
        "shape_index": config.shape_index,
        **asdict(config.shape),
        "l2": config.l2,
        "residual_shrink": config.residual_shrink,
        "saturation_onset": config.saturation_onset,
        "breadth_gamma": config.breadth_gamma,
        "replay_power": config.replay_power,
        "repeat_discount": config.repeat_discount,
        **metrics,
    }


def structural_configs(
    variant: Variant,
    shapes: tuple[family_grp.Shape, ...],
    shape_indices: list[int],
) -> list[Config]:
    onset_grid = SATURATION_ONSET_GRID if variant.finite_hill else (0.0,)
    gamma_grid = BREADTH_GAMMA_GRID if variant.breadth_gate else (0.0,)
    replay_grid = REPLAY_POWER_GRID if variant.convex_replay else (1.0,)
    repeat_discount_grid = REPEAT_DISCOUNT_GRID if variant.novelty_discounted else (1.0,)
    return [
        Config(
            variant=variant,
            shape_index=shape_index,
            shape=shapes[shape_index],
            l2=l2,
            residual_shrink=residual_shrink,
            saturation_onset=saturation_onset,
            breadth_gamma=breadth_gamma,
            replay_power=replay_power,
            repeat_discount=repeat_discount,
        )
        for shape_index in shape_indices
        for l2 in base.L2_GRID
        for residual_shrink in base.RESIDUAL_SHRINK_GRID
        for saturation_onset in onset_grid
        for breadth_gamma in gamma_grid
        for replay_power in replay_grid
        for repeat_discount in repeat_discount_grid
    ]


def score_configs(
    dataset: family_grp.Dataset,
    configs: list[Config],
    splits: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[Config, np.ndarray, list[dict[str, Any]]]:
    best: tuple[float, float, Config, np.ndarray] | None = None
    rows: list[dict[str, Any]] = []
    for config in configs:
        prediction = oof_prediction(dataset, config, splits)
        metrics = calibration_summary(dataset.target, prediction)
        rows.append(config_record(config, metrics))
        candidate = (float(metrics["rmse"]), -float(metrics["spearman"]), config, prediction)
        if best is None or candidate[:2] < best[:2]:
            best = candidate
    if best is None:
        raise RuntimeError("No calibration-form configurations were scored")
    return best[2], best[3], rows


def benchmark_dataset(
    dataset_id: base.DatasetId,
    variants: tuple[Variant, ...],
    num_shapes: int,
    top_shapes: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    dataset = base.load_dataset(dataset_id)
    shapes = family_grp.shape_candidates(family_grp.Variant.BUCKET_RESOLVED, num_shapes)
    splits = base.split_indices(dataset, dataset_id, np.arange(dataset.n), base.SCREEN_SEED)
    _baseline, _prediction, baseline_rows = base.score_configs(dataset, base.baseline_configs(shapes), splits)
    best_by_shape: dict[int, float] = {}
    for row in baseline_rows:
        shape_index = int(row["shape_index"])
        best_by_shape[shape_index] = min(best_by_shape.get(shape_index, float("inf")), float(row["rmse"]))
    shape_indices = [
        shape_index for shape_index, _score in sorted(best_by_shape.items(), key=lambda item: item[1])[:top_shapes]
    ]

    selected: dict[Variant, tuple[Config, np.ndarray]] = {}
    screen_rows: list[dict[str, Any]] = []
    for variant in variants:
        print(f"  screening {variant.value}", flush=True)
        config, prediction, rows = score_configs(dataset, structural_configs(variant, shapes, shape_indices), splits)
        selected[variant] = (config, prediction)
        screen_rows.extend({"dataset": dataset_id.value, **row} for row in rows)

    metric_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    heldout = base.heldout_data(dataset_id, dataset)
    for variant, (config, oof) in selected.items():
        metric_rows.append(
            {
                "dataset": dataset_id.value,
                "variant": variant.value,
                "split": "fit_oof",
                **config_record(config, calibration_summary(dataset.target, oof)),
            }
        )
        for index, (observed, predicted) in enumerate(zip(dataset.target, oof, strict=True)):
            prediction_rows.append(
                {
                    "dataset": dataset_id.value,
                    "variant": variant.value,
                    "split": "fit_oof",
                    "row_id": str(dataset.frame.iloc[index].get("run_name", index)),
                    "group": str(dataset.frame.iloc[index].get("panel_source", "fit")),
                    "policy_class": "two_phase_fit",
                    "observed": observed,
                    "predicted": predicted,
                }
            )
        if heldout is None:
            continue
        heldout_frame, heldout_weights, heldout_target = heldout
        heldout_prediction = fit_model(dataset, config, np.arange(dataset.n)).predict(heldout_weights)
        heldout_metrics = calibration_summary(heldout_target, heldout_prediction)
        selected_index = int(np.argmin(heldout_prediction))
        metric_rows.append(
            {
                "dataset": dataset_id.value,
                "variant": variant.value,
                "split": "heldout",
                **config_record(config, heldout_metrics),
                **base.grouped_heldout_summary(heldout_frame, heldout_target, heldout_prediction),
                "selected_observed": float(heldout_target[selected_index]),
                "selected_predicted": float(heldout_prediction[selected_index]),
                "selected_optimism": float(heldout_target[selected_index] - heldout_prediction[selected_index]),
                "selected_run": str(heldout_frame.iloc[selected_index]["wandb_run_name"]),
            }
        )
        for index, (observed, predicted) in enumerate(zip(heldout_target, heldout_prediction, strict=True)):
            prediction_rows.append(
                {
                    "dataset": dataset_id.value,
                    "variant": variant.value,
                    "split": "heldout",
                    "row_id": str(heldout_frame.iloc[index]["wandb_run_name"]),
                    "group": str(heldout_frame.iloc[index]["training_series"]),
                    "policy_class": str(heldout_frame.iloc[index]["policy_class"]),
                    "observed": observed,
                    "predicted": predicted,
                }
            )
    return metric_rows, screen_rows, prediction_rows


def render(metrics: pd.DataFrame, predictions: pd.DataFrame, output_dir: Path) -> None:
    colors = {
        Variant.CURRENT.value: "#a50026",
        Variant.POWER_BREADTH_GATE.value: "#f46d43",
        Variant.HILL.value: "#fee08b",
        Variant.HILL_BREADTH_GATE.value: "#66bd63",
        Variant.HILL_BREADTH_GATE_CONVEX_REPLAY.value: "#006837",
        Variant.NOVELTY_DISCOUNTED.value: "#2c7bb6",
        Variant.NOVELTY_DISCOUNTED_CONVEX_REPLAY.value: "#313695",
        Variant.JOINT_UNDERCOVERAGE.value: "#fdae61",
        Variant.FAMILY_PEAK_REPLAY.value: "#66c2a5",
        Variant.JOINT_UNDERCOVERAGE_PEAK_REPLAY.value: "#3288bd",
        Variant.GLOBAL_LITERAL_REPLAY.value: "#abdda4",
        Variant.FAMILY_LITERAL_REPLAY.value: "#5e4fa2",
    }
    figure = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Uncheatable: residual calibration",
            "Table-9: residual calibration",
            "Fit-panel OOF RMSE",
            "Frozen heldout RMSE",
        ),
    )
    dataset_ids = (base.DatasetId.DELPHI_3E18_UNCHEATABLE, base.DatasetId.DELPHI_3E18_TABLE9)
    for column, dataset_id in enumerate(dataset_ids, start=1):
        selected = predictions.loc[predictions["dataset"].eq(dataset_id.value) & predictions["split"].eq("heldout")]
        for variant in Variant:
            local = selected.loc[selected["variant"].eq(variant.value)]
            figure.add_trace(
                go.Scatter(
                    x=local["observed"],
                    y=local["predicted"] - local["observed"],
                    mode="markers",
                    marker={"color": colors[variant.value], "size": 5, "opacity": 0.45},
                    name=variant.value,
                    legendgroup=variant.value,
                    showlegend=column == 1,
                    customdata=np.column_stack([local["row_id"], local["group"]]),
                    hovertemplate=(
                        "%{customdata[0]}<br>%{customdata[1]}<br>obs=%{x:.5f}" "<br>pred-obs=%{y:.5f}<extra></extra>"
                    ),
                ),
                row=1,
                col=column,
            )
        figure.add_hline(y=0.0, line={"color": "#777", "dash": "dash"}, row=1, col=column)
    for column, split in enumerate(("fit_oof", "heldout"), start=1):
        selected = metrics.loc[metrics["split"].eq(split)]
        for variant in Variant:
            local = selected.loc[selected["variant"].eq(variant.value)]
            figure.add_trace(
                go.Bar(
                    x=local["dataset"],
                    y=local["rmse"],
                    marker_color=colors[variant.value],
                    name=variant.value,
                    legendgroup=variant.value,
                    showlegend=False,
                ),
                row=2,
                col=column,
            )
    figure.update_layout(
        title="Mechanistic calibration forms for hierarchical phase-replay GRP",
        template="plotly_white",
        barmode="group",
        width=1500,
        height=1000,
        legend={"orientation": "h", "y": 1.08},
    )
    figure.update_xaxes(title_text="Observed BPB", row=1)
    figure.update_yaxes(title_text="Prediction residual (predicted - observed)", row=1)
    figure.update_yaxes(title_text="RMSE", row=2)
    figure.write_html(output_dir / "calibration_screen.html", include_plotlyjs="cdn", config=PLOT_CONFIG)


def markdown_table(metrics: pd.DataFrame) -> str:
    columns = [
        "dataset",
        "variant",
        "split",
        "rmse",
        "spearman",
        "calibration_slope_observed_on_predicted",
        "optimism_gt_0p05_count",
        "worst_optimism",
        "regret_at_1",
        "selected_optimism",
    ]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for _, row in metrics[columns].iterrows():
        values = []
        for column in columns:
            value = row[column]
            if column in {"dataset", "variant", "split"}:
                values.append(str(value))
            elif column == "optimism_gt_0p05_count":
                values.append(str(int(value)))
            elif pd.isna(value):
                values.append("-")
            else:
                values.append(f"{float(value):.5f}")
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    dataset_ids = tuple(base.DatasetId(value) for value in args.datasets.split(",") if value)
    variants = tuple(Variant(value) for value in args.variants.split(",") if value)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metric_rows: list[dict[str, Any]] = []
    screen_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    for dataset_id in dataset_ids:
        print(f"Benchmarking {dataset_id.value}", flush=True)
        metrics, screens, predictions = benchmark_dataset(
            dataset_id,
            variants,
            args.num_shapes,
            args.top_shapes,
        )
        metric_rows.extend(metrics)
        screen_rows.extend(screens)
        prediction_rows.extend(predictions)
    metrics = pd.DataFrame(metric_rows)
    predictions = pd.DataFrame(prediction_rows)
    metrics.to_csv(args.output_dir / "metrics.csv", index=False)
    pd.DataFrame(screen_rows).to_csv(args.output_dir / "hyperparameter_screen.csv", index=False)
    predictions.to_csv(args.output_dir / "predictions.csv", index=False)
    render(metrics, predictions, args.output_dir)
    (args.output_dir / "summary.json").write_text(
        json.dumps(
            {
                "generated_at": datetime.now(UTC).isoformat(),
                "datasets": [dataset.value for dataset in dataset_ids],
                "variants": [variant.value for variant in variants],
                "selection": "fit-panel five-fold OOF RMSE, Spearman tie-break",
                "heldout_role": "frozen transfer diagnostic only",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    (args.output_dir / "report.md").write_text(
        "# Hierarchical calibration-form screen\n\n"
        "Every nonlinear setting is selected on the fit panel before the frozen 3e18 heldouts are scored. "
        "The residual-versus-observed slope is descriptive because observed appears on both axes; observed-on-predicted "
        "calibration slope, tail counts, and RMSE are the decision diagnostics.\n\n" + markdown_table(metrics) + "\n"
    )


if __name__ == "__main__":
    main()
