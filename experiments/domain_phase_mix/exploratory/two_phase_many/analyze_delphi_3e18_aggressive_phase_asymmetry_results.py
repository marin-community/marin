# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "fsspec",
#   "gcsfs",
#   "numpy",
#   "pandas",
#   "plotly",
#   "scipy",
#   "tabulate",
#   "wandb",
# ]
# ///
"""Analyze the completed Delphi 3e18 aggressive phase-asymmetry panel."""

from __future__ import annotations

import argparse
import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import fsspec
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import wandb
from plotly.subplots import make_subplots
from scipy import stats

SCRIPT_DIR = Path(__file__).resolve().parent
REFERENCE_OUTPUTS = SCRIPT_DIR / "reference_outputs"
DEFAULT_PANEL_DIR = REFERENCE_OUTPUTS / "delphi_3e18_aggressive_phase_asymmetry_20260722"
DEFAULT_OUTPUT_DIR = REFERENCE_OUTPUTS / "delphi_3e18_aggressive_phase_asymmetry_results_20260723"

TRAIN_PROJECT = "marin-community/marin"
EVAL_PROJECT = "marin-community/marin-eval"
TRAIN_TAG = "delphi-3e18-aggressive-phase-asymmetry"
EVAL_GROUP = "olmo_base_eval_table9_delphi_3e18_aggressive_phase_asymmetry_20260722"
UNCHEATABLE_METRIC = "eval/uncheatable_eval/bpb"
TABLE9_METRIC = "olmo_base_easy/table9_51_component_macro_bpb"
EXPECTED_ROWS = 290
PLOT_CONFIG = {"toImageButtonOptions": {"format": "png", "scale": 4}}
TARGETS = {
    "uncheatable": ("uncheatable_bpb", "Uncheatable BPB"),
    "table9": ("table9_macro_bpb", "Table-9 macro BPB"),
}
PRIMARY_TARGET = {
    "uncheatable_frontier": "uncheatable",
    "table9_frontier": "table9",
}
ANCHOR_LABELS = {
    "uncheatable_frontier": "Uncheatable frontier anchor",
    "table9_frontier": "Table-9 frontier anchor",
}
FAMILY_LABELS = {
    "balanced_partition": "Balanced random partition",
    "handcrafted_late_quality": "Handcrafted late-quality",
    "dolmino_late_continuum": "Dolmino-late continuum",
}
FAMILY_COLORS = {
    "balanced_partition": "#4575B4",
    "handcrafted_late_quality": "#F46D43",
    "dolmino_late_continuum": "#1A9850",
}
CURRENT_TWO_PHASE_FRONTIERS = {
    "uncheatable": ("dphase_unch05_eff_e0p005_3e18", 0.9824552536010742),
    "table9": ("dphase_t9b075_can_e0p005_3e18", 1.056690469761157),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel-dir", type=Path, default=DEFAULT_PANEL_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--wandb-timeout", type=int, default=180)
    parser.add_argument("--refresh-wandb", action="store_true")
    return parser.parse_args()


def finite_summary(run: Any, key: str) -> float:
    value = run.summary.get(key)
    if value is None or not math.isfinite(float(value)):
        raise ValueError(f"Run {run.name!r} has no finite {key!r}: {value!r}")
    return float(value)


def training_run(runs: list[Any], expected_name: str) -> Any:
    matches = [run for run in runs if run.name.startswith(f"{expected_name}-")]
    finished = [run for run in matches if run.state == "finished"]
    if len(finished) != 1:
        states = [(run.name, run.id, run.state) for run in matches]
        raise ValueError(f"Expected one finished training run for {expected_name!r}, found {states}")
    return finished[0]


def persisted_table9(run: Any) -> tuple[float, str] | None:
    output_path = str(run.config.get("output_path") or "").rstrip("/")
    if not output_path.startswith("gs://marin-us-east5/"):
        return None
    result_path = f"{output_path}/olmo_base_eval_table9_results.json"
    status_path = f"{output_path}/.executor_status"
    fs, _, paths = fsspec.get_fs_token_paths(result_path)
    status_fs, _, status_paths = fsspec.get_fs_token_paths(status_path)
    if not paths or not status_paths or not fs.exists(paths[0]) or not status_fs.exists(status_paths[0]):
        return None
    with status_fs.open(status_paths[0], "r") as source:
        status = source.read().strip()
    if status != "SUCCESS":
        return None
    with fs.open(paths[0], "r") as source:
        payload = json.load(source)
    value = float(payload["table9_macro_bpb"])
    if not math.isfinite(value):
        raise ValueError(f"Non-finite persisted Table-9 value for {run.name!r}: {value}")
    return value, result_path


def eval_run(runs: list[Any], expected_name: str) -> tuple[Any, float, str, int, int]:
    matches = [run for run in runs if run.name == expected_name]
    finished = [run for run in matches if run.state == "finished"]
    if finished:
        values = [finite_summary(run, TABLE9_METRIC) for run in finished]
        if max(values) - min(values) > 1e-12:
            raise ValueError(f"Finished native Table-9 retries disagree for {expected_name!r}: {values}")
        selected = max(finished, key=lambda run: str(run.created_at))
        return (
            selected,
            values[-1],
            "wandb_finished_summary",
            len(matches),
            sum(run.state != "finished" for run in matches),
        )
    for selected in sorted(matches, key=lambda run: str(run.created_at), reverse=True):
        recovered = persisted_table9(selected)
        if recovered is not None:
            value, result_path = recovered
            return selected, value, result_path, len(matches), len(matches)
    states = [(run.id, run.state) for run in matches]
    raise ValueError(f"No finished or persisted native Table-9 result for {expected_name!r}: {states}")


def collect_results(
    manifest: pd.DataFrame,
    *,
    timeout: int,
) -> tuple[pd.DataFrame, dict[str, object]]:
    api = wandb.Api(timeout=timeout)
    training_runs = list(api.runs(TRAIN_PROJECT, filters={"tags": {"$in": [TRAIN_TAG]}}, per_page=EXPECTED_ROWS + 50))
    eval_runs = list(api.runs(EVAL_PROJECT, filters={"group": EVAL_GROUP}, per_page=EXPECTED_ROWS + 200))

    rows: list[dict[str, object]] = []
    for record in manifest.to_dict(orient="records"):
        run_order = int(record["run_order"])
        base_name = f"agphase_{run_order:03d}_{record['candidate_id']}"
        train = training_run(training_runs, base_name)
        native_eval, table9_value, table9_source, attempt_count, failed_attempt_count = eval_run(
            eval_runs, f"t9_{base_name}"
        )
        rows.append(
            {
                **record,
                "training_wandb_id": train.id,
                "training_wandb_name": train.name,
                "training_wandb_url": train.url,
                "training_state": train.state,
                "training_final_step": int(train.summary["_step"]),
                "eval_wandb_id": native_eval.id,
                "eval_wandb_name": native_eval.name,
                "eval_wandb_url": native_eval.url,
                "eval_state": native_eval.state,
                "eval_attempt_count": attempt_count,
                "eval_failed_attempt_count": failed_attempt_count,
                "table9_metric_source": table9_source,
                "uncheatable_bpb": finite_summary(train, UNCHEATABLE_METRIC),
                "table9_macro_bpb": table9_value,
            }
        )

    results = pd.DataFrame(rows).sort_values("run_order").reset_index(drop=True)
    if len(results) != EXPECTED_ROWS or results["candidate_id"].duplicated().any():
        raise ValueError(f"Expected {EXPECTED_ROWS} unique outcomes, found {len(results)}")
    if set(results["training_final_step"]) != {3006}:
        raise ValueError(f"Unexpected final steps: {sorted(results['training_final_step'].unique())}")
    audit = {
        "queried_at": datetime.now(UTC).isoformat(),
        "training_project": TRAIN_PROJECT,
        "training_tag": TRAIN_TAG,
        "training_attempt_count": len(training_runs),
        "training_attempt_states": pd.Series([run.state for run in training_runs]).value_counts().to_dict(),
        "eval_project": EVAL_PROJECT,
        "eval_group": EVAL_GROUP,
        "eval_attempt_count": len(eval_runs),
        "eval_attempt_states": pd.Series([run.state for run in eval_runs]).value_counts().to_dict(),
        "selected_training_count": len(results),
        "selected_eval_count": len(results),
        "gcs_recovered_eval_count": int(results["table9_metric_source"].ne("wandb_finished_summary").sum()),
        "eval_names_with_multiple_finished_attempts": int(
            results["eval_attempt_count"].sub(results["eval_failed_attempt_count"]).gt(1).sum()
        ),
    }
    return results, audit


def attach_control_deltas(results: pd.DataFrame) -> pd.DataFrame:
    frame = results.copy()
    centers = frame[frame["contrast_family"].eq("center_control")].set_index(["anchor_id", "seed_block"])
    if len(centers) != 32:
        raise ValueError(f"Expected 32 tied controls, found {len(centers)}")
    for target, (column, _) in TARGETS.items():
        control_values = np.asarray(
            [float(centers.loc[(row.anchor_id, int(row.seed_block)), column]) for row in frame.itertuples()]
        )
        frame[f"{target}_same_seed_control_bpb"] = control_values
        frame[f"{target}_delta_vs_control"] = frame[column].to_numpy(float) - control_values
    return frame


def mean_ci(values: np.ndarray) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    mean = float(np.mean(values))
    if len(values) < 2:
        return mean, math.nan, math.nan
    radius = float(stats.t.ppf(0.975, len(values) - 1) * stats.sem(values))
    return mean, mean - radius, mean + radius


def control_summary(results: pd.DataFrame, anchor_audit: pd.DataFrame) -> pd.DataFrame:
    audit = anchor_audit.set_index("anchor_id")
    rows: list[dict[str, object]] = []
    for anchor_id, group in results[results["contrast_family"].eq("center_control")].groupby("anchor_id"):
        for target, (column, _) in TARGETS.items():
            values = group[column].to_numpy(float)
            source_column = "source_uncheatable_bpb" if target == "uncheatable" else "source_table9_macro_bpb"
            source = float(audit.loc[anchor_id, source_column])
            rows.append(
                {
                    "anchor_id": anchor_id,
                    "target": target,
                    "is_primary_target": PRIMARY_TARGET[anchor_id] == target,
                    "source_selected_bpb": source,
                    "fresh_control_mean_bpb": float(np.mean(values)),
                    "fresh_control_sd_bpb": float(np.std(values, ddof=1)),
                    "fresh_control_min_bpb": float(np.min(values)),
                    "fresh_control_max_bpb": float(np.max(values)),
                    "fresh_minus_source_bpb": float(np.mean(values) - source),
                    "n_controls": len(values),
                }
            )
    return pd.DataFrame(rows).sort_values(["anchor_id", "target"]).reset_index(drop=True)


def family_tv_summary(results: pd.DataFrame) -> pd.DataFrame:
    treatments = results[~results["contrast_family"].eq("center_control")]
    rows: list[dict[str, object]] = []
    for (anchor_id, family, tv), group in treatments.assign(
        target_phase_tv=lambda frame: frame["target_phase_tv"].round(6)
    ).groupby(["anchor_id", "contrast_family", "target_phase_tv"], sort=True):
        for target_name in TARGETS:
            delta = group[f"{target_name}_delta_vs_control"].to_numpy(float)
            mean, ci_low, ci_high = mean_ci(delta)
            rows.append(
                {
                    "anchor_id": anchor_id,
                    "target": target_name,
                    "is_primary_target": PRIMARY_TARGET[anchor_id] == target_name,
                    "contrast_family": family,
                    "target_phase_tv": float(tv),
                    "n": len(delta),
                    "mean_delta_bpb": mean,
                    "ci95_low_bpb": ci_low,
                    "ci95_high_bpb": ci_high,
                    "median_delta_bpb": float(np.median(delta)),
                    "fraction_better": float(np.mean(delta < 0.0)),
                    "best_delta_bpb": float(np.min(delta)),
                    "worst_delta_bpb": float(np.max(delta)),
                    "count_gain_ge_0p005": int(np.sum(delta <= -0.005)),
                    "count_gain_ge_0p010": int(np.sum(delta <= -0.010)),
                }
            )
    return (
        pd.DataFrame(rows)
        .sort_values(["anchor_id", "target", "contrast_family", "target_phase_tv"])
        .reset_index(drop=True)
    )


def balanced_antithetic_summary(results: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    balanced = results[results["contrast_family"].eq("balanced_partition")]
    pair_rows: list[dict[str, object]] = []
    for (anchor_id, direction_id, tv), group in balanced.groupby(
        ["anchor_id", "direction_id", "target_phase_tv"], sort=True
    ):
        if len(group) != 2 or set(group["sign"]) != {"plus", "minus"}:
            raise ValueError(f"Incomplete antithetic pair: {anchor_id}/{direction_id}/{tv}")
        plus = group[group["sign"].eq("plus")].iloc[0]
        minus = group[group["sign"].eq("minus")].iloc[0]
        if int(plus["seed_block"]) != int(minus["seed_block"]):
            raise ValueError(f"Seed mismatch: {anchor_id}/{direction_id}/{tv}")
        row: dict[str, object] = {
            "anchor_id": anchor_id,
            "direction_id": direction_id,
            "target_phase_tv": float(tv),
            "seed_block": int(plus["seed_block"]),
        }
        for target, (column, _) in TARGETS.items():
            control = float(plus[f"{target}_same_seed_control_bpb"])
            plus_value = float(plus[column])
            minus_value = float(minus[column])
            row[f"{target}_plus_delta"] = plus_value - control
            row[f"{target}_minus_delta"] = minus_value - control
            row[f"{target}_odd_effect"] = (plus_value - minus_value) / 2.0
            row[f"{target}_curvature"] = (plus_value + minus_value) / 2.0 - control
            row[f"{target}_best_sign_delta"] = min(plus_value, minus_value) - control
            row[f"{target}_better_sign"] = "plus" if plus_value < minus_value else "minus"
        pair_rows.append(row)
    pairs = pd.DataFrame(pair_rows).sort_values(["anchor_id", "direction_id", "target_phase_tv"])
    if len(pairs) != 96:
        raise ValueError(f"Expected 96 direction-TV pairs, found {len(pairs)}")

    summary_rows: list[dict[str, object]] = []
    for (anchor_id, tv), group in pairs.groupby(["anchor_id", "target_phase_tv"], sort=True):
        for target in TARGETS:
            odd = group[f"{target}_odd_effect"].to_numpy(float)
            curvature = group[f"{target}_curvature"].to_numpy(float)
            best = group[f"{target}_best_sign_delta"].to_numpy(float)
            mean_curvature, ci_low, ci_high = mean_ci(curvature)
            summary_rows.append(
                {
                    "anchor_id": anchor_id,
                    "target": target,
                    "is_primary_target": PRIMARY_TARGET[anchor_id] == target,
                    "target_phase_tv": float(tv),
                    "direction_count": len(group),
                    "odd_effect_rms_bpb": float(np.sqrt(np.mean(np.square(odd)))),
                    "mean_curvature_bpb": mean_curvature,
                    "curvature_ci95_low_bpb": ci_low,
                    "curvature_ci95_high_bpb": ci_high,
                    "fraction_mean_pair_better": float(np.mean(curvature < 0.0)),
                    "median_best_sign_delta_bpb": float(np.median(best)),
                    "best_sign_delta_min_bpb": float(np.min(best)),
                }
            )
    return pairs.reset_index(drop=True), pd.DataFrame(summary_rows)


def direction_consistency(pairs: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for anchor_id, anchor_group in pairs.groupby("anchor_id", sort=True):
        for target in TARGETS:
            coherent = 0
            monotone = 0
            any_all_better = 0
            for _, group in anchor_group.groupby("direction_id", sort=True):
                group = group.sort_values("target_phase_tv")
                signs = group[f"{target}_better_sign"].tolist()
                coherent += len(set(signs)) == 1
                signed_best = group[f"{target}_best_sign_delta"].to_numpy(float)
                monotone += bool(np.all(np.diff(signed_best) <= 0.0))
                any_all_better += bool(np.all(signed_best < 0.0))
            rows.append(
                {
                    "anchor_id": anchor_id,
                    "target": target,
                    "is_primary_target": PRIMARY_TARGET[anchor_id] == target,
                    "direction_count": 16,
                    "same_better_sign_all_three_tv": coherent,
                    "fraction_same_better_sign_all_three_tv": coherent / 16.0,
                    "best_sign_gain_monotone_with_tv": monotone,
                    "fraction_best_sign_gain_monotone_with_tv": monotone / 16.0,
                    "best_sign_better_all_three_tv": any_all_better,
                    "fraction_best_sign_better_all_three_tv": any_all_better / 16.0,
                }
            )
    return pd.DataFrame(rows)


def handcrafted_summary(results: pd.DataFrame) -> pd.DataFrame:
    handcrafted = results[results["contrast_family"].eq("handcrafted_late_quality")]
    rows: list[dict[str, object]] = []
    grouped = handcrafted.assign(target_phase_tv=lambda x: x.target_phase_tv.round(6)).groupby(
        ["anchor_id", "target_phase_tv"], sort=True
    )
    for (anchor_id, tv), group in grouped:
        for target_name in TARGETS:
            delta = group[f"{target_name}_delta_vs_control"].to_numpy(float)
            mean, ci_low, ci_high = mean_ci(delta)
            rows.append(
                {
                    "anchor_id": anchor_id,
                    "target": target_name,
                    "is_primary_target": PRIMARY_TARGET[anchor_id] == target_name,
                    "target_phase_tv": float(tv),
                    "recipe_count": len(group),
                    "mean_delta_bpb": mean,
                    "ci95_low_bpb": ci_low,
                    "ci95_high_bpb": ci_high,
                    "fraction_better": float(np.mean(delta < 0.0)),
                    "best_recipe": group.loc[group[f"{target_name}_delta_vs_control"].idxmin(), "direction_id"],
                    "best_recipe_delta_bpb": float(np.min(delta)),
                }
            )
    return pd.DataFrame(rows)


def dolmino_continuum_summary(results: pd.DataFrame) -> pd.DataFrame:
    continuum = results[results["contrast_family"].eq("dolmino_late_continuum")]
    rows: list[dict[str, object]] = []
    for (anchor_id, direction_id), group in continuum.groupby(["anchor_id", "direction_id"], sort=True):
        for target in TARGETS:
            delta = group[f"{target}_delta_vs_control"].to_numpy(float)
            mean, ci_low, ci_high = mean_ci(delta)
            rows.append(
                {
                    "anchor_id": anchor_id,
                    "target": target,
                    "is_primary_target": PRIMARY_TARGET[anchor_id] == target,
                    "direction_id": direction_id,
                    "late_dolmino_share": float(group["phase_1_dolmino_share"].iloc[0]),
                    "phase_tv": float(group["phase_tv"].iloc[0]),
                    "replicate_count": len(group),
                    "replicate_deltas_json": json.dumps([float(value) for value in delta]),
                    "mean_delta_bpb": mean,
                    "sd_delta_bpb": float(np.std(delta, ddof=1)),
                    "ci95_low_bpb": ci_low,
                    "ci95_high_bpb": ci_high,
                    "all_replicates_better": bool(np.all(delta < 0.0)),
                    "best_delta_bpb": float(np.min(delta)),
                    "worst_delta_bpb": float(np.max(delta)),
                }
            )
    return pd.DataFrame(rows)


def selected_extrema(results: pd.DataFrame, controls: pd.DataFrame) -> pd.DataFrame:
    treatments = results[~results["contrast_family"].eq("center_control")]
    control_lookup = controls.set_index(["anchor_id", "target"])
    rows: list[dict[str, object]] = []
    for anchor_id, group in treatments.groupby("anchor_id"):
        for target, (column, _) in TARGETS.items():
            best = group.loc[group[column].idxmin()]
            source = float(control_lookup.loc[(anchor_id, target), "source_selected_bpb"])
            fresh = float(control_lookup.loc[(anchor_id, target), "fresh_control_mean_bpb"])
            rows.append(
                {
                    "anchor_id": anchor_id,
                    "target": target,
                    "is_primary_target": PRIMARY_TARGET[anchor_id] == target,
                    "candidate_id": best["candidate_id"],
                    "contrast_family": best["contrast_family"],
                    "direction_id": best["direction_id"],
                    "phase_tv": float(best["phase_tv"]),
                    "observed_bpb": float(best[column]),
                    "delta_vs_same_seed_control_bpb": float(best[f"{target}_delta_vs_control"]),
                    "delta_vs_fresh_control_mean_bpb": float(best[column]) - fresh,
                    "delta_vs_historical_source_bpb": float(best[column]) - source,
                }
            )
    return pd.DataFrame(rows)


def write_family_plot(results: pd.DataFrame, output_path: Path) -> None:
    treatments = results[~results["contrast_family"].eq("center_control")]
    figure = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=[
            f"{ANCHOR_LABELS[anchor]} / {TARGETS[target][1]}" for anchor in ANCHOR_LABELS for target in TARGETS
        ],
        horizontal_spacing=0.08,
        vertical_spacing=0.13,
    )
    for anchor_index, anchor_id in enumerate(ANCHOR_LABELS, start=1):
        for target_index, target in enumerate(TARGETS, start=1):
            group = treatments[treatments["anchor_id"].eq(anchor_id)]
            for family in FAMILY_LABELS:
                rows = group[group["contrast_family"].eq(family)]
                figure.add_trace(
                    go.Scatter(
                        x=rows["phase_tv"],
                        y=rows[f"{target}_delta_vs_control"],
                        mode="markers",
                        marker={"size": 8, "opacity": 0.65, "color": FAMILY_COLORS[family]},
                        name=FAMILY_LABELS[family],
                        legendgroup=family,
                        showlegend=anchor_index == 1 and target_index == 1,
                        customdata=rows[["candidate_id", "direction_label", "sign", "seed_block"]].to_numpy(),
                        hovertemplate=(
                            "%{customdata[0]}<br>%{customdata[1]}<br>sign=%{customdata[2]}"
                            "<br>seed=%{customdata[3]}<br>phase TV=%{x:.4f}<br>delta=%{y:.6f}<extra></extra>"
                        ),
                    ),
                    row=anchor_index,
                    col=target_index,
                )
            figure.add_hline(
                y=0.0,
                line={"color": "#6C757D", "dash": "dash"},
                row=anchor_index,
                col=target_index,
            )
            figure.update_xaxes(title_text="Phase total variation", row=anchor_index, col=target_index)
            figure.update_yaxes(title_text="BPB minus same-seed tied control", row=anchor_index, col=target_index)
    figure.update_layout(
        title="Aggressive phase-asymmetry outcomes at fixed aggregate mixture",
        template="plotly_white",
        width=1550,
        height=1050,
        legend={"orientation": "h", "y": 1.08, "x": 0.5, "xanchor": "center"},
    )
    figure.write_html(output_path, include_plotlyjs=True, config=PLOT_CONFIG)


def write_continuum_plot(results: pd.DataFrame, output_path: Path) -> None:
    continuum = results[results["contrast_family"].eq("dolmino_late_continuum")]
    figure = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=[
            f"{ANCHOR_LABELS[anchor]} / {TARGETS[target][1]}" for anchor in ANCHOR_LABELS for target in TARGETS
        ],
        horizontal_spacing=0.08,
        vertical_spacing=0.14,
    )
    for anchor_index, anchor_id in enumerate(ANCHOR_LABELS, start=1):
        for target_index, target in enumerate(TARGETS, start=1):
            group = continuum[continuum["anchor_id"].eq(anchor_id)]
            figure.add_trace(
                go.Scatter(
                    x=group["phase_1_dolmino_share"],
                    y=group[f"{target}_delta_vs_control"],
                    mode="markers",
                    marker={
                        "size": 10,
                        "color": group["phase_tv"],
                        "colorscale": "RdYlGn_r",
                        "cmin": 0.3,
                        "cmax": 0.75,
                        "showscale": anchor_index == 1 and target_index == 2,
                        "colorbar": {"title": "Phase TV"},
                    },
                    showlegend=False,
                    customdata=group[["candidate_id", "replicate_index", "seed_block"]].to_numpy(),
                    hovertemplate=(
                        "%{customdata[0]}<br>replicate=%{customdata[1]}<br>seed=%{customdata[2]}"
                        "<br>late Dolmino=%{x:.0%}<br>delta=%{y:.6f}<extra></extra>"
                    ),
                ),
                row=anchor_index,
                col=target_index,
            )
            figure.add_hline(
                y=0.0,
                line={"color": "#6C757D", "dash": "dash"},
                row=anchor_index,
                col=target_index,
            )
            figure.update_xaxes(
                title_text="Dolmino share in the final 20% phase",
                tickformat=".0%",
                row=anchor_index,
                col=target_index,
            )
            figure.update_yaxes(title_text="BPB minus same-seed tied control", row=anchor_index, col=target_index)
    figure.update_layout(
        title="Replicated conventional Dolmino-late schedules",
        template="plotly_white",
        width=1450,
        height=1000,
    )
    figure.write_html(output_path, include_plotlyjs=True, config=PLOT_CONFIG)


def markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    return frame[columns].to_markdown(index=False, floatfmt=".6f")


def write_report(
    output_path: Path,
    controls: pd.DataFrame,
    family_summary: pd.DataFrame,
    antithetic_summary: pd.DataFrame,
    consistency: pd.DataFrame,
    handcrafted: pd.DataFrame,
    continuum: pd.DataFrame,
    extrema: pd.DataFrame,
) -> None:
    primary_controls = controls[controls["is_primary_target"]]
    primary_families = family_summary[family_summary["is_primary_target"]]
    primary_antithetic = antithetic_summary[antithetic_summary["is_primary_target"]]
    primary_consistency = consistency[consistency["is_primary_target"]]
    primary_handcrafted = handcrafted[handcrafted["is_primary_target"]]
    primary_continuum = continuum[continuum["is_primary_target"]]
    primary_extrema = extrema[extrema["is_primary_target"]]
    uncheatable_extreme = primary_extrema[
        primary_extrema["anchor_id"].eq("uncheatable_frontier") & primary_extrema["target"].eq("uncheatable")
    ].iloc[0]
    table9_extreme = primary_extrema[
        primary_extrema["anchor_id"].eq("table9_frontier") & primary_extrema["target"].eq("table9")
    ].iloc[0]
    uncheatable_frontier_name, uncheatable_frontier = CURRENT_TWO_PHASE_FRONTIERS["uncheatable"]
    table9_frontier_name, table9_frontier = CURRENT_TWO_PHASE_FRONTIERS["table9"]

    lines = [
        "# Delphi 3e18 aggressive phase-asymmetry panel: observed results",
        "",
        "## Coverage and estimand",
        "",
        (
            "All 290 training checkpoints reached step 3006 and all 290 native Table-9 names have a complete "
            "evaluation result. One name has only a crashed W&B record but a persisted executor `SUCCESS` result; "
            "the other 289 are joined to finished W&B attempts. Failed native-eval attempts are retained in the "
            "audit. Every treatment is compared with the tied frontier control sharing its anchor and data-seed "
            "block. Negative deltas are improvements."
        ),
        "",
        (
            "The preregistered units are kept separate: 16 balanced random partitions with antithetic phase order "
            "at TV 0.10/0.25/0.50; eight handcrafted late-quality recipes at the same TVs; and three replicated "
            "Dolmino-late schedules. Selected minima are reported only as secondary diagnostics."
        ),
        "",
        "## Headline conclusion",
        "",
        (
            "The panel does not reveal a new frontier or evidence for a generic large-asymmetry gain at 3e18. "
            f"The best absolute Uncheatable treatment is {uncheatable_extreme['observed_bpb']:.6f}, "
            f"{uncheatable_extreme['observed_bpb'] - uncheatable_frontier:+.6f} BPB versus the existing "
            f"`{uncheatable_frontier_name}` frontier ({uncheatable_frontier:.6f}). None of the 129 Uncheatable-anchor "
            "treatments improves its same-seed control by 0.005 BPB. The best paired reduction is only 0.001440 BPB."
        ),
        "",
        (
            f"The best absolute Table-9 treatment is {table9_extreme['observed_bpb']:.6f}, "
            f"{table9_extreme['observed_bpb'] - table9_frontier:+.6f} BPB versus the existing "
            f"`{table9_frontier_name}` frontier ({table9_frontier:.6f}). Its 0.010186 paired reduction is an isolated "
            "TV=0.10 synthetic-data-late point: the same recipe is worse at TV=0.25 and TV=0.50, and it has one seed. "
            "It is therefore a candidate-direction clue, not evidence for 0.01-BPB recoverable headroom."
        ),
        "",
        (
            "Across both objectives, TV=0.50 is decisively too aggressive. The balanced-partition mean curvature "
            "is +0.007207 BPB for Uncheatable and +0.012204 BPB for Table-9, and every antithetic pair is worse than "
            "its tied control on average. Conventional schedules with 90-100% Dolmino in the final phase also "
            "degrade all three target-matched repeats. The useful search region remains near tied or moderately "
            "asymmetric and must be direction-specific."
        ),
        "",
        "## Fresh tied-control variation",
        "",
        markdown_table(
            primary_controls,
            [
                "anchor_id",
                "target",
                "source_selected_bpb",
                "fresh_control_mean_bpb",
                "fresh_control_sd_bpb",
                "fresh_minus_source_bpb",
            ],
        ),
        "",
        "## Balanced random partitions",
        "",
        markdown_table(
            primary_antithetic,
            [
                "anchor_id",
                "target",
                "target_phase_tv",
                "direction_count",
                "odd_effect_rms_bpb",
                "mean_curvature_bpb",
                "curvature_ci95_low_bpb",
                "curvature_ci95_high_bpb",
                "fraction_mean_pair_better",
                "median_best_sign_delta_bpb",
            ],
        ),
        "",
        (
            "Odd effect measures phase-order sensitivity within an antithetic pair. Curvature is the pair mean "
            "minus its tied control; positive values mean aggressive phase asymmetry is harmful on average even "
            "after choosing neither sign."
        ),
        "",
        "### Cross-TV direction consistency",
        "",
        markdown_table(
            primary_consistency,
            [
                "anchor_id",
                "target",
                "direction_count",
                "same_better_sign_all_three_tv",
                "best_sign_gain_monotone_with_tv",
                "best_sign_better_all_three_tv",
            ],
        ),
        "",
        "## Handcrafted late-quality recipes",
        "",
        markdown_table(
            primary_handcrafted,
            [
                "anchor_id",
                "target",
                "target_phase_tv",
                "recipe_count",
                "mean_delta_bpb",
                "ci95_low_bpb",
                "ci95_high_bpb",
                "fraction_better",
                "best_recipe",
                "best_recipe_delta_bpb",
            ],
        ),
        "",
        "## Replicated Dolmino-late continuum",
        "",
        markdown_table(
            primary_continuum,
            [
                "anchor_id",
                "target",
                "late_dolmino_share",
                "phase_tv",
                "replicate_count",
                "mean_delta_bpb",
                "sd_delta_bpb",
                "ci95_low_bpb",
                "ci95_high_bpb",
                "all_replicates_better",
            ],
        ),
        "",
        "The three-repeat confidence intervals are descriptive and wide; replicate signs and effect sizes matter more.",
        "",
        "## Family-level treatment summaries",
        "",
        markdown_table(
            primary_families,
            [
                "anchor_id",
                "target",
                "contrast_family",
                "target_phase_tv",
                "n",
                "mean_delta_bpb",
                "ci95_low_bpb",
                "ci95_high_bpb",
                "fraction_better",
                "count_gain_ge_0p005",
                "count_gain_ge_0p010",
            ],
        ),
        "",
        "## Selected extrema (secondary, selection-biased)",
        "",
        markdown_table(
            primary_extrema,
            [
                "anchor_id",
                "target",
                "candidate_id",
                "contrast_family",
                "phase_tv",
                "observed_bpb",
                "delta_vs_same_seed_control_bpb",
                "delta_vs_fresh_control_mean_bpb",
                "delta_vs_historical_source_bpb",
            ],
        ),
        "",
        "## Interpretation",
        "",
        (
            "This panel tests whether substantially larger phase asymmetry uncovers a robust effect hidden by the "
            "earlier boundary-normalized sampler. It does not estimate the unrestricted global two-phase optimum: "
            "balanced partitions and conventional late-quality schedules are structured subsets of the 38-dimensional "
            "fixed-aggregate contrast space. It materially lowers the plausibility that the missing 0.01 BPB is "
            "available through generic or conventional aggressive annealing at these two aggregate anchors, while "
            "leaving open narrow, model-identifiable low-TV directions and scale-dependent effects."
        ),
    ]
    output_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(args.panel_dir / "launcher_source_panel.csv")
    anchor_audit = pd.read_csv(args.panel_dir / "anchor_audit.csv")
    observed_path = args.output_dir / "observed_results.csv"
    audit_path = args.output_dir / "wandb_join_audit.json"
    if observed_path.exists() and audit_path.exists() and not args.refresh_wandb:
        observed = pd.read_csv(observed_path)
    else:
        observed, audit = collect_results(manifest, timeout=args.wandb_timeout)
        observed.to_csv(observed_path, index=False)
        audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True, default=int) + "\n")

    results = attach_control_deltas(observed)
    controls = control_summary(results, anchor_audit)
    families = family_tv_summary(results)
    pairs, antithetic = balanced_antithetic_summary(results)
    consistency = direction_consistency(pairs)
    handcrafted = handcrafted_summary(results)
    continuum = dolmino_continuum_summary(results)
    extrema = selected_extrema(results, controls)

    results.to_csv(args.output_dir / "observed_results_with_control_deltas.csv", index=False)
    controls.to_csv(args.output_dir / "fresh_control_summary.csv", index=False)
    families.to_csv(args.output_dir / "family_tv_summary.csv", index=False)
    pairs.to_csv(args.output_dir / "balanced_antithetic_pairs.csv", index=False)
    antithetic.to_csv(args.output_dir / "balanced_antithetic_summary.csv", index=False)
    consistency.to_csv(args.output_dir / "balanced_direction_consistency.csv", index=False)
    handcrafted.to_csv(args.output_dir / "handcrafted_summary.csv", index=False)
    continuum.to_csv(args.output_dir / "dolmino_late_continuum_summary.csv", index=False)
    extrema.to_csv(args.output_dir / "selected_extrema.csv", index=False)
    write_family_plot(results, args.output_dir / "phase_asymmetry_outcomes.html")
    write_continuum_plot(results, args.output_dir / "dolmino_late_continuum.html")
    write_report(
        args.output_dir / "report.md",
        controls,
        families,
        antithetic,
        consistency,
        handcrafted,
        continuum,
        extrema,
    )


if __name__ == "__main__":
    main()
