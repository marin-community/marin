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
"""Screen aggregate-orthogonal phase mechanisms on matched policy outcomes."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import spearmanr

REPO_ROOT = Path(__file__).resolve().parents[5]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.domain_phase_mix.exploratory.two_phase_many import (
    analyze_olmo_base_easy_per_component_dsp_decision_300m as component_dsp,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (
    benchmark_partially_pooled_phase_bowls as pooled,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (
    export_mixture_fit_observatory as observatory,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.mechanistic_surrogate_discovery_20260719 import (
    paired_dynamics_models as models,
)

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR.parent / "reference_outputs/mechanistic_surrogate_discovery_20260719/round1_paired_dynamics"
ONE_PHASE_DELPHI_SERIES = "delphi_one_phase_augmented_swarm_3e18_20260715"
TARGET_COLUMN = {"uncheatable": "uncheatable_bpb", "table9": "table9_macro_bpb"}
N_SPLITS = 5
FOLD_SEED = 20260719
PLOT_CONFIG = {"toImageButtonOptions": {"format": "png", "scale": 4}}
TECH_CODE_DOMAINS = frozenset(
    {
        "dolma3_arxiv",
        "dolma3_finemath_3plus",
        "dolma3_stack_edu",
        "dolmino_stack_edu_fim",
        "dolmino_synth_code",
        "dolmino_synth_math",
    }
)
REASONING_DOMAINS = frozenset({"dolmino_synth_instruction", "dolmino_synth_thinking"})


@dataclass(frozen=True)
class OOFPrediction:
    aggregate: np.ndarray
    combined: np.ndarray
    delta: np.ndarray
    fold_coefficients: tuple[np.ndarray, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


def family_partition(dataset: pooled.Dataset) -> tuple[tuple[str, ...], tuple[np.ndarray, ...]]:
    grouped: dict[str, list[int]] = {"broad_text": [], "tech_code": [], "reasoning": []}
    for index, domain in enumerate(dataset.domain_names):
        if domain in TECH_CODE_DOMAINS:
            grouped["tech_code"].append(index)
        elif domain in REASONING_DOMAINS:
            grouped["reasoning"].append(index)
        else:
            grouped["broad_text"].append(index)
    names = tuple(grouped)
    members = tuple(np.asarray(grouped[name], dtype=int) for name in names)
    if sorted(np.concatenate(members).tolist()) != list(range(dataset.m)):
        raise ValueError("Generic semantic families do not partition the domains")
    return names, members


def aligned_one_phase_300m(target: str, two_phase: pooled.Dataset) -> np.ndarray:
    one_phase = observatory.load_300m_single_phase_dataset(target, two_phase)
    alpha0 = float(np.mean(two_phase.c0 / (two_phase.c0 + two_phase.c1)))
    aggregate = alpha0 * two_phase.weights[:, 0, :] + (1.0 - alpha0) * two_phase.weights[:, 1, :]
    candidates = one_phase.weights[:, 0, :]
    distance = np.max(np.abs(aggregate[:, None, :] - candidates[None, :, :]), axis=2)
    nearest = np.argmin(distance, axis=1)
    minimum = np.min(distance, axis=1)
    if not np.all(minimum < 1e-10) or len(set(nearest.tolist())) != len(two_phase.frame):
        raise ValueError("The 300M one-phase and two-phase panels do not form a bijective aggregate match")
    return np.asarray(one_phase.y[nearest], dtype=float)


def delphi_one_phase_rows(domain_names: tuple[str, ...], target: str) -> list[dict[str, Any]]:
    """Read only the matched one-phase rows, never adversarial target values."""

    rows: list[dict[str, Any]] = []
    with observatory.DELPHI_3E18_HELDOUTS.open(newline="") as source:
        for row in csv.DictReader(source):
            if row["training_series"] != ONE_PHASE_DELPHI_SERIES:
                continue
            if row["policy_class"] != "single_phase_tied":
                continue
            if row["training_state"] != "finished" or row["checkpoint_declared_complete"] != "1":
                continue
            weight_map = json.loads(row["phase_0_weights_json"])
            rows.append(
                {
                    "heldout_id": row["heldout_id"],
                    "weights": np.asarray([float(weight_map[domain]) for domain in domain_names]),
                    "target": float(row[TARGET_COLUMN[target]]),
                }
            )
    if len(rows) != 238:
        raise ValueError(f"Expected 238 coordinate-disjoint one-phase rows, found {len(rows)}")
    return rows


def aligned_one_phase_delphi(target: str, two_phase: pooled.Dataset) -> np.ndarray:
    rows = delphi_one_phase_rows(tuple(two_phase.domain_names), target)
    candidates = np.stack([row["weights"] for row in rows])
    alpha0 = float(np.mean(two_phase.c0 / (two_phase.c0 + two_phase.c1)))
    aggregate = alpha0 * two_phase.weights[:, 0, :] + (1.0 - alpha0) * two_phase.weights[:, 1, :]
    distance = np.max(np.abs(aggregate[:, None, :] - candidates[None, :, :]), axis=2)
    nearest = np.argmin(distance, axis=1)
    minimum = np.min(distance, axis=1)
    matched = minimum < 1e-10
    if int(matched.sum()) != 238 or len(set(nearest[matched].tolist())) != 238:
        raise ValueError("The Delphi one-phase rows do not match 238 unique two-phase aggregate coordinates")
    target_aligned = np.full(two_phase.n, np.nan, dtype=float)
    row_targets = np.asarray([row["target"] for row in rows], dtype=float)
    target_aligned[matched] = row_targets[nearest[matched]]
    return target_aligned


def load_panel(scale: str, target: str) -> models.PairedPanel:
    if scale == "300m":
        raw = pooled.load_300m_dataset(target)
        one_phase_target = aligned_one_phase_300m(target, raw)
    elif scale == "delphi_3e18":
        raw = observatory.load_delphi_3e18_fit_dataset(target)
        one_phase_target = aligned_one_phase_delphi(target, raw)
    else:
        raise ValueError(f"Unknown scale {scale}")
    family_names, family_members = family_partition(raw)
    return models.PairedPanel(
        name=f"{scale}_{target}",
        target=target,
        frame=raw.frame.copy(),
        domain_names=tuple(raw.domain_names),
        family_names=family_names,
        family_members=family_members,
        weights=np.asarray(raw.weights, dtype=float),
        c0=np.asarray(raw.c0, dtype=float),
        c1=np.asarray(raw.c1, dtype=float),
        two_phase_target=np.asarray(raw.y, dtype=float),
        one_phase_target=one_phase_target,
    )


def folds(panel: models.PairedPanel) -> list[tuple[np.ndarray, np.ndarray]]:
    return component_dsp.panel_stratified_folds(panel.frame, n_splits=N_SPLITS, seed=FOLD_SEED)


def scalar_metrics(observed: np.ndarray, predicted: np.ndarray) -> dict[str, float | int]:
    observed = np.asarray(observed, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    valid = np.isfinite(observed) & np.isfinite(predicted)
    observed = observed[valid]
    predicted = predicted[valid]
    residual = predicted - observed
    if len(observed) >= 3 and np.std(predicted) > 1e-12:
        slope, intercept = np.polyfit(predicted, observed, 1)
        rank_correlation = float(spearmanr(observed, predicted).statistic)
    else:
        slope, intercept, rank_correlation = np.nan, np.nan, np.nan
    selected = int(np.argmin(predicted))
    return {
        "n": len(observed),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "mae": float(np.mean(np.abs(residual))),
        "bias": float(np.mean(residual)),
        "spearman": rank_correlation,
        "calibration_slope": float(slope),
        "calibration_intercept": float(intercept),
        "regret_at_1": float(observed[selected] - np.min(observed)),
        "worst_optimism": float(np.max(observed - predicted)),
    }


def aggregate_oof(panel: models.PairedPanel, config: models.AggregateConfig) -> np.ndarray:
    prediction = np.full(panel.n, np.nan, dtype=float)
    for train, test in folds(panel):
        model = models.fit_aggregate(panel, train, config)
        paired_test = test[panel.paired_mask[test]]
        prediction[paired_test] = model.predict_weights(models.tied_weights(panel, panel.weights[paired_test]))
    return prediction


def phase_oof(
    panel: models.PairedPanel,
    aggregate_config: models.AggregateConfig,
    family: models.PhaseFamily,
    phase_config: models.PhaseConfig,
) -> OOFPrediction:
    aggregate_prediction = np.full(panel.n, np.nan, dtype=float)
    combined_prediction = np.full(panel.n, np.nan, dtype=float)
    delta_prediction = np.full(panel.n, np.nan, dtype=float)
    coefficients: list[np.ndarray] = []
    for train, test in folds(panel):
        aggregate_model = models.fit_aggregate(panel, train, aggregate_config)
        phase_model = models.fit_phase(panel, train, family, phase_config)
        aggregate_prediction[test] = aggregate_model.predict_weights(models.tied_weights(panel, panel.weights[test]))
        delta_prediction[test] = phase_model.predict_delta(panel.weights[test])
        combined_prediction[test] = aggregate_prediction[test] + delta_prediction[test]
        coefficients.append(phase_model.head.coefficients_in_natural_units)
    return OOFPrediction(
        aggregate=aggregate_prediction,
        combined=combined_prediction,
        delta=delta_prediction,
        fold_coefficients=tuple(coefficients),
    )


def aggregate_configs() -> list[models.AggregateConfig]:
    return [
        models.AggregateConfig(power, offset, l2)
        for power in (0.25, 0.5, 1.0)
        for offset in (0.1, 0.3, 1.0)
        for l2 in (0.0, 0.01, 0.1, 1.0)
    ]


def phase_configs(family: models.PhaseFamily) -> list[models.PhaseConfig]:
    if family is models.PhaseFamily.PMVT:
        return [
            models.PMVTConfig(offset, l2, signed, quadratic, transport_level, mismatch_level)
            for offset in (0.25, 0.5, 1.0, 2.0)
            for l2 in (0.001, 0.01, 0.1, 1.0)
            for signed, quadratic, transport_level, mismatch_level in (
                (True, False, "family", "family"),
                (False, True, "family", "family"),
                (True, True, "family", "family"),
                (True, True, "bucket", "family"),
                (True, True, "family", "bucket"),
                (True, True, "bucket", "bucket"),
            )
        ]
    if family is models.PhaseFamily.COMMUTATOR:
        return [
            models.CommutatorConfig(offset, l2) for offset in (0.25, 0.5, 1.0, 2.0) for l2 in (0.001, 0.01, 0.1, 1.0)
        ]
    if family is models.PhaseFamily.FAST_SLOW:
        return [
            models.FastSlowConfig(learn, forget, consolidate, slow, l2, state_level)
            for learn in (1.0, 4.0, 16.0, 64.0)
            for forget in (0.5, 2.0, 8.0)
            for consolidate in (0.25, 1.0, 4.0)
            for slow in (0.25, 0.5, 0.75)
            for l2 in (0.1, 1.0)
            for state_level in ("family", "bucket")
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
    raise ValueError(f"No config grid for {family}")


def select_aggregate(panel: models.PairedPanel) -> tuple[models.AggregateConfig, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    for config in aggregate_configs():
        prediction = aggregate_oof(panel, config)
        summary = scalar_metrics(panel.one_phase_target, prediction)
        rows.append({"panel": panel.name, "config": config.key, **asdict(config), **summary})
    table = pd.DataFrame(rows).sort_values(["rmse", "l2", "shortage_power", "shortage_offset"])
    best = table.iloc[0]
    return (
        models.AggregateConfig(
            shortage_power=float(best["shortage_power"]),
            shortage_offset=float(best["shortage_offset"]),
            l2=float(best["l2"]),
        ),
        table,
    )


def select_phase(
    panel: models.PairedPanel,
    aggregate_config: models.AggregateConfig,
    family: models.PhaseFamily,
) -> tuple[models.PhaseConfig, OOFPrediction, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    predictions: dict[str, OOFPrediction] = {}
    paired = panel.paired_mask
    observed_delta = panel.two_phase_target - panel.one_phase_target
    for config in phase_configs(family):
        prediction = phase_oof(panel, aggregate_config, family, config)
        predictions[config.key] = prediction
        combined = scalar_metrics(panel.two_phase_target, prediction.combined)
        delta = scalar_metrics(observed_delta[paired], prediction.delta[paired])
        coefficient_matrix = np.stack(prediction.fold_coefficients)
        rows.append(
            {
                "panel": panel.name,
                "family": family.value,
                "config": config.key,
                "config_json": json.dumps(asdict(config), sort_keys=True),
                "two_phase_rmse": combined["rmse"],
                "two_phase_spearman": combined["spearman"],
                "two_phase_regret_at_1": combined["regret_at_1"],
                "delta_rmse": delta["rmse"],
                "delta_spearman": delta["spearman"],
                "delta_calibration_slope": delta["calibration_slope"],
                "coefficient_fold_cv": float(
                    np.mean(
                        np.std(coefficient_matrix, axis=0)
                        / np.maximum(np.abs(np.mean(coefficient_matrix, axis=0)), 1e-8)
                    )
                ),
                "coefficient_zero_fraction": float(np.mean(np.abs(coefficient_matrix) < 1e-8)),
            }
        )
    table = pd.DataFrame(rows).sort_values(["delta_rmse", "two_phase_rmse", "coefficient_fold_cv"])
    best_row = table.iloc[0]
    config_values = json.loads(best_row["config_json"])
    config_class = {
        models.PhaseFamily.PMVT: models.PMVTConfig,
        models.PhaseFamily.COMMUTATOR: models.CommutatorConfig,
        models.PhaseFamily.FAST_SLOW: models.FastSlowConfig,
        models.PhaseFamily.QUASI_STEADY: models.QuasiSteadyConfig,
        models.PhaseFamily.TERMINAL_EQUILIBRIUM: models.TerminalEquilibriumConfig,
    }[family]
    best_config = config_class(**config_values)
    return best_config, predictions[str(best_row["config"])], table


def prediction_records(
    panel: models.PairedPanel,
    family: models.PhaseFamily,
    aggregate_config: models.AggregateConfig,
    phase_config: models.PhaseConfig,
    prediction: OOFPrediction,
) -> list[dict[str, Any]]:
    observed_delta = panel.two_phase_target - panel.one_phase_target
    records: list[dict[str, Any]] = []
    for index in range(panel.n):
        records.append(
            {
                "panel": panel.name,
                "target": panel.target,
                "row_index": index,
                "run_name": str(panel.frame.iloc[index].get("run_name", index)),
                "panel_source": str(panel.frame.iloc[index].get("panel_source", "unknown")),
                "paired": bool(panel.paired_mask[index]),
                "family": family.value,
                "aggregate_config": aggregate_config.key,
                "phase_config": phase_config.key,
                "observed_two_phase": float(panel.two_phase_target[index]),
                "predicted_two_phase": float(prediction.combined[index]),
                "observed_one_phase": (float(panel.one_phase_target[index]) if panel.paired_mask[index] else math.nan),
                "predicted_aggregate": float(prediction.aggregate[index]),
                "observed_delta": float(observed_delta[index]) if panel.paired_mask[index] else math.nan,
                "predicted_delta": float(prediction.delta[index]),
            }
        )
    return records


def metric_records(
    panel: models.PairedPanel,
    family: models.PhaseFamily,
    prediction: OOFPrediction,
) -> list[dict[str, Any]]:
    paired = panel.paired_mask
    observed_delta = panel.two_phase_target - panel.one_phase_target
    zero_delta = np.zeros(int(paired.sum()))
    output: list[dict[str, Any]] = []
    for split, observed, predicted in (
        ("one_phase_oof", panel.one_phase_target[paired], prediction.aggregate[paired]),
        ("two_phase_oof", panel.two_phase_target, prediction.combined),
        ("paired_delta_oof", observed_delta[paired], prediction.delta[paired]),
        ("paired_delta_zero_baseline", observed_delta[paired], zero_delta),
    ):
        output.append(
            {"panel": panel.name, "family": family.value, "split": split, **scalar_metrics(observed, predicted)}
        )
    return output


def write_prediction_plot(records: pd.DataFrame, output_path: Path) -> None:
    panels = list(dict.fromkeys(records["panel"]))
    families = list(dict.fromkeys(records["family"]))
    figure = make_subplots(
        rows=len(panels),
        cols=len(families),
        subplot_titles=[f"{panel} / {family}" for panel in panels for family in families],
        horizontal_spacing=0.05,
        vertical_spacing=0.1,
    )
    colors = {"paired": "#198754", "unpaired": "#d97706"}
    for row_index, panel in enumerate(panels, start=1):
        for column_index, family in enumerate(families, start=1):
            subset = records.loc[(records["panel"] == panel) & (records["family"] == family)]
            for paired, marker in ((True, "circle"), (False, "diamond")):
                points = subset.loc[subset["paired"] == paired]
                if points.empty:
                    continue
                figure.add_trace(
                    go.Scatter(
                        x=points["observed_two_phase"],
                        y=points["predicted_two_phase"],
                        mode="markers",
                        name="paired" if paired else "unpaired",
                        legendgroup="paired" if paired else "unpaired",
                        showlegend=row_index == 1 and column_index == 1,
                        marker={"color": colors["paired" if paired else "unpaired"], "symbol": marker, "size": 7},
                        customdata=points[["run_name", "panel_source"]],
                        hovertemplate="%{customdata[0]}<br>%{customdata[1]}<br>obs=%{x:.5f}<br>pred=%{y:.5f}<extra></extra>",
                    ),
                    row=row_index,
                    col=column_index,
                )
            low = float(min(subset["observed_two_phase"].min(), subset["predicted_two_phase"].min()))
            high = float(max(subset["observed_two_phase"].max(), subset["predicted_two_phase"].max()))
            figure.add_trace(
                go.Scatter(
                    x=[low, high],
                    y=[low, high],
                    mode="lines",
                    line={"dash": "dash", "color": "#64748b"},
                    showlegend=False,
                ),
                row=row_index,
                col=column_index,
            )
    figure.update_layout(
        title="Round 1: aggregate-identified phase mechanisms",
        template="plotly_white",
        height=430 * len(panels),
        width=520 * len(families),
    )
    figure.update_xaxes(title_text="Observed BPB")
    figure.update_yaxes(title_text="OOF predicted BPB")
    figure.write_html(output_path, include_plotlyjs="cdn", config=PLOT_CONFIG)


def write_delta_plot(records: pd.DataFrame, output_path: Path) -> None:
    paired = records.loc[records["paired"]].copy()
    panels = list(dict.fromkeys(paired["panel"]))
    figure = make_subplots(rows=1, cols=len(panels), subplot_titles=panels)
    colors = {
        models.PhaseFamily.PMVT.value: "#007f5f",
        models.PhaseFamily.COMMUTATOR.value: "#f59e0b",
        models.PhaseFamily.FAST_SLOW.value: "#d62828",
        models.PhaseFamily.QUASI_STEADY.value: "#5b21b6",
        models.PhaseFamily.TERMINAL_EQUILIBRIUM.value: "#0f766e",
    }
    for column, panel in enumerate(panels, start=1):
        for family, points in paired.loc[paired["panel"] == panel].groupby("family", sort=False):
            figure.add_trace(
                go.Scatter(
                    x=points["observed_delta"],
                    y=points["predicted_delta"],
                    mode="markers",
                    name=family,
                    legendgroup=family,
                    showlegend=column == 1,
                    marker={"color": colors[family], "size": 7, "opacity": 0.75},
                    customdata=points[["run_name", "panel_source"]],
                    hovertemplate="%{customdata[0]}<br>%{customdata[1]}<br>observed delta=%{x:.5f}<br>predicted delta=%{y:.5f}<extra></extra>",
                ),
                row=1,
                col=column,
            )
        low = float(min(points["observed_delta"].min(), points["predicted_delta"].min()))
        high = float(max(points["observed_delta"].max(), points["predicted_delta"].max()))
        figure.add_trace(
            go.Scatter(
                x=[low, high], y=[low, high], mode="lines", line={"dash": "dash", "color": "#64748b"}, showlegend=False
            ),
            row=1,
            col=column,
        )
    figure.update_layout(
        title="Can the proposed mechanisms predict paired phase-order effects?",
        template="plotly_white",
        height=560,
        width=660 * len(panels),
    )
    figure.update_xaxes(title_text="Observed two-phase minus one-phase BPB")
    figure.update_yaxes(title_text="OOF predicted phase correction")
    figure.write_html(output_path, include_plotlyjs="cdn", config=PLOT_CONFIG)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    panels = [load_panel(scale, target) for scale in ("300m", "delphi_3e18") for target in ("uncheatable", "table9")]
    aggregate_grid: list[pd.DataFrame] = []
    phase_grid: list[pd.DataFrame] = []
    metric_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    selections: list[dict[str, Any]] = []
    for panel in panels:
        aggregate_config, aggregate_table = select_aggregate(panel)
        aggregate_grid.append(aggregate_table)
        for family in models.PhaseFamily:
            phase_config, prediction, phase_table = select_phase(panel, aggregate_config, family)
            phase_grid.append(phase_table)
            metric_rows.extend(metric_records(panel, family, prediction))
            prediction_rows.extend(prediction_records(panel, family, aggregate_config, phase_config, prediction))
            selections.append(
                {
                    "panel": panel.name,
                    "family": family.value,
                    "aggregate_config": aggregate_config.key,
                    "phase_config": phase_config.key,
                    "paired_rows": int(panel.paired_mask.sum()),
                    "total_two_phase_rows": panel.n,
                }
            )
    aggregate_table = pd.concat(aggregate_grid, ignore_index=True)
    phase_table = pd.concat(phase_grid, ignore_index=True)
    metrics_table = pd.DataFrame(metric_rows)
    predictions_table = pd.DataFrame(prediction_rows)
    selection_table = pd.DataFrame(selections)
    aggregate_table.to_csv(args.output_dir / "aggregate_hyperparameter_grid.csv", index=False)
    phase_table.to_csv(args.output_dir / "phase_hyperparameter_grid.csv", index=False)
    metrics_table.to_csv(args.output_dir / "paired_screen_metrics.csv", index=False)
    predictions_table.to_csv(args.output_dir / "paired_screen_predictions.csv", index=False)
    selection_table.to_csv(args.output_dir / "selected_configs.csv", index=False)
    write_prediction_plot(predictions_table, args.output_dir / "combined_oof_predictions.html")
    write_delta_plot(predictions_table, args.output_dir / "paired_delta_predictions.html")
    summary = metrics_table.pivot_table(index=["panel", "family"], columns="split", values="rmse")
    summary.to_csv(args.output_dir / "rmse_summary.csv")
    print(selection_table.to_string(index=False))
    print("\nRMSE summary")
    print(summary.to_string())


if __name__ == "__main__":
    main()
