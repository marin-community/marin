# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "cvxpy>=1.7",
#   "fsspec>=2025.7",
#   "numpy>=2.0",
#   "pandas>=2.2",
#   "plotly>=6.0",
#   "scikit-learn>=1.6",
#   "scipy>=1.15",
# ]
# ///
"""Audit preregistered phase mechanisms on both dense StarCoder surfaces."""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.stats import spearmanr
from sklearn.model_selection import KFold

REPO_ROOT = Path(__file__).resolve().parents[5]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    export_mixture_fit_observatory as observatory,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.mechanistic_surrogate_discovery_20260719 import (  # noqa: E402
    paired_dynamics_models as models,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.mechanistic_surrogate_discovery_20260719 import (  # noqa: E402
    starcoder_refined_data,
)

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = (
    SCRIPT_DIR.parent / "reference_outputs/mechanistic_surrogate_discovery_20260719/round1_starcoder_shape_refined"
)
N_SPLITS = 5
SEED = 20260719
PLOT_CONFIG = {"toImageButtonOptions": {"format": "png", "scale": 4}}
REFINED_FAMILIES = (
    models.PhaseFamily.PMVT,
    models.PhaseFamily.FAST_SLOW,
    models.PhaseFamily.TERMINAL_EQUILIBRIUM,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


def panel_from_dataset(dataset: Any) -> models.PairedPanel:
    tied = np.max(np.abs(dataset.weights[:, 0, :] - dataset.weights[:, 1, :]), axis=1) < 1e-10
    one_phase = np.full(dataset.n, np.nan, dtype=float)
    one_phase[tied] = dataset.y[tied]
    return models.PairedPanel(
        name=dataset.name,
        target="starcoder_bpb",
        frame=dataset.frame.copy(),
        domain_names=tuple(dataset.domain_names),
        family_names=tuple(dataset.domain_names),
        family_members=tuple(np.asarray([index], dtype=int) for index in range(dataset.m)),
        weights=np.asarray(dataset.weights, dtype=float),
        c0=np.asarray(dataset.c0, dtype=float),
        c1=np.asarray(dataset.c1, dtype=float),
        two_phase_target=np.asarray(dataset.y, dtype=float),
        one_phase_target=one_phase,
    )


def config_grid(family: models.PhaseFamily) -> list[models.PhaseConfig]:
    if family is models.PhaseFamily.PMVT:
        # The remaining-learnability offset was selected independently on all
        # four 39-bucket paired panels; StarCoder may select only its ridge.
        return [models.PMVTConfig(2.0, l2, True, True) for l2 in (0.001, 0.01, 0.1, 1.0)]
    if family is models.PhaseFamily.COMMUTATOR:
        return [models.CommutatorConfig(offset, l2) for offset in (0.25, 0.5, 1.0, 2.0) for l2 in (0.01, 0.1, 1.0)]
    if family is models.PhaseFamily.FAST_SLOW:
        return [
            models.FastSlowConfig(learn, forget, consolidate, slow, l2)
            for learn in (1.0, 4.0, 16.0)
            for forget in (0.5, 2.0, 8.0)
            for consolidate in (0.25, 1.0, 4.0)
            for slow in (0.25, 0.5, 0.75)
            for l2 in (0.1, 1.0)
        ]
    if family is models.PhaseFamily.QUASI_STEADY:
        return [
            models.QuasiSteadyConfig(ratio, consolidate, slow, l2)
            for ratio in (0.03125, 0.0625, 0.125, 0.25, 0.5)
            for consolidate in (0.25, 1.0, 4.0, 16.0)
            for slow in (0.25, 0.5, 0.75)
            for l2 in (0.1, 1.0)
        ]
    if family is models.PhaseFamily.TERMINAL_EQUILIBRIUM:
        return [
            models.TerminalEquilibriumConfig(ratio, l2)
            for ratio in (0.015625, 0.03125, 0.0625, 0.125, 0.25, 0.5, 1.0)
            for l2 in (0.01, 0.1, 1.0)
        ]
    raise ValueError(f"No grid for {family}")


def aggregate_config_grid() -> list[models.AggregateConfig]:
    return [models.AggregateConfig(power, offset, 1.0) for power in (0.25, 0.5, 1.0) for offset in (0.1, 0.3, 1.0)]


def metrics(observed: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    residual = predicted - observed
    slope, intercept = np.polyfit(predicted, observed, 1) if np.std(predicted) > 1e-12 else (np.nan, np.nan)
    return {
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "mae": float(np.mean(np.abs(residual))),
        "spearman": float(spearmanr(observed, predicted).statistic),
        "calibration_slope": float(slope),
        "calibration_intercept": float(intercept),
        "worst_optimism": float(np.max(observed - predicted)),
    }


def kfolds(n: int) -> list[tuple[np.ndarray, np.ndarray]]:
    splitter = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    return [(train, test) for train, test in splitter.split(np.arange(n))]


def oof_prediction(
    panel: models.PairedPanel,
    aggregate_config: models.AggregateConfig,
    family: models.PhaseFamily,
    config: models.PhaseConfig,
) -> np.ndarray:
    prediction = np.full(panel.n, np.nan, dtype=float)
    for train, test in kfolds(panel.n):
        model = models.fit_joint(panel, train, aggregate_config, family, config)
        prediction[test] = model.predict_weights(panel.weights[test])
    return prediction


def select_config(
    panel: models.PairedPanel,
    family: models.PhaseFamily,
) -> tuple[models.AggregateConfig, models.PhaseConfig, np.ndarray, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    predictions: dict[str, np.ndarray] = {}
    for aggregate_config in aggregate_config_grid():
        for config in config_grid(family):
            prediction = oof_prediction(panel, aggregate_config, family, config)
            key = f"{aggregate_config.key}::{config.key}"
            predictions[key] = prediction
            rows.append(
                {
                    "surface": panel.name,
                    "family": family.value,
                    "aggregate_config": aggregate_config.key,
                    "aggregate_config_json": json.dumps(asdict(aggregate_config), sort_keys=True),
                    "config": config.key,
                    "config_json": json.dumps(asdict(config), sort_keys=True),
                    **metrics(panel.two_phase_target, prediction),
                }
            )
    table = pd.DataFrame(rows).sort_values(["rmse", "worst_optimism"])
    best = table.iloc[0]
    values = json.loads(best["config_json"])
    config_class = {
        models.PhaseFamily.PMVT: models.PMVTConfig,
        models.PhaseFamily.COMMUTATOR: models.CommutatorConfig,
        models.PhaseFamily.FAST_SLOW: models.FastSlowConfig,
        models.PhaseFamily.QUASI_STEADY: models.QuasiSteadyConfig,
        models.PhaseFamily.TERMINAL_EQUILIBRIUM: models.TerminalEquilibriumConfig,
    }[family]
    selected_config = config_class(**values)
    selected_aggregate = models.AggregateConfig(**json.loads(best["aggregate_config_json"]))
    return selected_aggregate, selected_config, predictions[f"{best['aggregate_config']}::{best['config']}"], table


def leave_region_out(
    panel: models.PairedPanel,
    aggregate_config: models.AggregateConfig,
    family: models.PhaseFamily,
    config: models.PhaseConfig,
) -> list[dict[str, Any]]:
    rare0 = panel.weights[:, 0, 1]
    rare1 = panel.weights[:, 1, 1]
    difference = rare1 - rare0
    regions = {
        "late_rare_enriched": difference > 0.1,
        "early_rare_enriched": difference < -0.1,
        "near_phase_tied": np.abs(difference) <= 0.1,
    }
    rows: list[dict[str, Any]] = []
    for name, test_mask in regions.items():
        train = np.flatnonzero(~test_mask)
        test = np.flatnonzero(test_mask)
        if len(train) < 10 or len(test) < 3:
            continue
        model = models.fit_joint(panel, train, aggregate_config, family, config)
        prediction = model.predict_weights(panel.weights[test])
        rows.append(
            {
                "surface": panel.name,
                "family": family.value,
                "region": name,
                "n_train": len(train),
                "n_test": len(test),
                **metrics(panel.two_phase_target[test], prediction),
            }
        )
    return rows


def optimum_record(
    panel: models.PairedPanel,
    aggregate_config: models.AggregateConfig,
    family: models.PhaseFamily,
    config: models.PhaseConfig,
) -> tuple[dict[str, Any], pd.DataFrame]:
    model = models.fit_joint(panel, np.arange(panel.n), aggregate_config, family, config)
    grid = np.linspace(0.0, 1.0, 201)
    rare0, rare1 = np.meshgrid(grid, grid, indexing="ij")
    weights = np.stack(
        [
            np.column_stack([1.0 - rare0.ravel(), rare0.ravel()]),
            np.column_stack([1.0 - rare1.ravel(), rare1.ravel()]),
        ],
        axis=1,
    )
    prediction = model.predict_weights(weights)
    best = int(np.argmin(prediction))
    observed_best = int(np.argmin(panel.two_phase_target))
    record = {
        "surface": panel.name,
        "family": family.value,
        "phase0_rare": float(rare0.ravel()[best]),
        "phase1_rare": float(rare1.ravel()[best]),
        "predicted_bpb": float(prediction[best]),
        "observed_best_phase0_rare": float(panel.weights[observed_best, 0, 1]),
        "observed_best_phase1_rare": float(panel.weights[observed_best, 1, 1]),
        "observed_best_bpb": float(panel.two_phase_target[observed_best]),
        "distance_to_observed_best": float(
            np.hypot(
                rare0.ravel()[best] - panel.weights[observed_best, 0, 1],
                rare1.ravel()[best] - panel.weights[observed_best, 1, 1],
            )
        ),
    }
    surface = pd.DataFrame(
        {
            "phase0_rare": rare0.ravel(),
            "phase1_rare": rare1.ravel(),
            "predicted_bpb": prediction,
        }
    )
    return record, surface


def write_surface_plot(
    panel: models.PairedPanel,
    family: models.PhaseFamily,
    surface: pd.DataFrame,
    optimum: dict[str, Any],
    output_path: Path,
) -> None:
    grid_size = round(math.sqrt(len(surface)))
    z = surface["predicted_bpb"].to_numpy().reshape(grid_size, grid_size)
    grid = surface["phase0_rare"].to_numpy().reshape(grid_size, grid_size)[:, 0]
    figure = go.Figure()
    figure.add_trace(
        go.Surface(
            x=grid,
            y=grid,
            z=z.T,
            colorscale="RdYlGn_r",
            opacity=0.72,
            name="Predicted surface",
            showscale=True,
        )
    )
    figure.add_trace(
        go.Scatter3d(
            x=panel.weights[:, 0, 1],
            y=panel.weights[:, 1, 1],
            z=panel.two_phase_target,
            mode="markers",
            marker={"size": 4, "color": panel.two_phase_target, "colorscale": "RdYlGn_r", "line": {"width": 0.5}},
            text=panel.frame.get("run_name", pd.Series(np.arange(panel.n))).astype(str),
            name="Observed checkpoints",
        )
    )
    figure.add_trace(
        go.Scatter3d(
            x=[optimum["phase0_rare"]],
            y=[optimum["phase1_rare"]],
            z=[optimum["predicted_bpb"]],
            mode="markers",
            marker={"size": 9, "symbol": "diamond", "color": "#111827"},
            name="Predicted optimum",
        )
    )
    figure.update_layout(
        title=f"{panel.name}: {family.value}",
        template="plotly_white",
        scene={
            "xaxis_title": "Phase 0 StarCoder weight",
            "yaxis_title": "Phase 1 StarCoder weight",
            "zaxis_title": "BPB",
        },
        height=850,
    )
    figure.write_html(output_path, include_plotlyjs="cdn", config=PLOT_CONFIG)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cosine = observatory.load_cosine_starcoder()
    panels = [
        panel_from_dataset(cosine),
        panel_from_dataset(starcoder_refined_data.load_refined_wsd80_starcoder(cosine)),
    ]
    grid_tables: list[pd.DataFrame] = []
    metric_rows: list[dict[str, Any]] = []
    region_rows: list[dict[str, Any]] = []
    optimum_rows: list[dict[str, Any]] = []
    for panel in panels:
        for family in REFINED_FAMILIES:
            aggregate_config, config, prediction, grid_table = select_config(panel, family)
            grid_tables.append(grid_table)
            metric_rows.append(
                {
                    "surface": panel.name,
                    "family": family.value,
                    "selected_aggregate_config": aggregate_config.key,
                    "selected_config": config.key,
                    **metrics(panel.two_phase_target, prediction),
                }
            )
            region_rows.extend(leave_region_out(panel, aggregate_config, family, config))
            optimum, surface = optimum_record(panel, aggregate_config, family, config)
            optimum_rows.append(optimum)
            surface.to_csv(args.output_dir / f"{panel.name}__{family.value}__surface.csv", index=False)
            write_surface_plot(
                panel,
                family,
                surface,
                optimum,
                args.output_dir / f"{panel.name}__{family.value}__surface.html",
            )
    pd.concat(grid_tables, ignore_index=True).to_csv(args.output_dir / "hyperparameter_grid.csv", index=False)
    metrics_table = pd.DataFrame(metric_rows)
    regions_table = pd.DataFrame(region_rows)
    optima_table = pd.DataFrame(optimum_rows)
    metrics_table.to_csv(args.output_dir / "surface_oof_metrics.csv", index=False)
    regions_table.to_csv(args.output_dir / "leave_region_out_metrics.csv", index=False)
    optima_table.to_csv(args.output_dir / "predicted_optima.csv", index=False)
    print(metrics_table.to_string(index=False))
    print("\nLeave-region-out")
    print(regions_table.to_string(index=False))
    print("\nOptima")
    print(optima_table.to_string(index=False))


if __name__ == "__main__":
    main()
