# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "numpy",
#   "pandas",
#   "plotly",
#   "scipy",
#   "tabulate",
#   "wandb",
# ]
# ///
"""Collect and analyze the Delphi 3e18 Compact raw-optimum path validation."""

from __future__ import annotations

import argparse
import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import wandb
from plotly.colors import sample_colorscale
from plotly.subplots import make_subplots
from scipy import stats

SCRIPT_DIR = Path(__file__).resolve().parent
REFERENCE_OUTPUTS = SCRIPT_DIR / "reference_outputs"
DEFAULT_PANEL_DIR = REFERENCE_OUTPUTS / "delphi_compact_optimum_path_validation_panel_20260721"
DEFAULT_OUTPUT_DIR = REFERENCE_OUTPUTS / "delphi_compact_optimum_path_validation_results_20260721"
DEFAULT_NOISE_PANEL = REFERENCE_OUTPUTS / "delphi_3e18_proportional_noise_floor_20260703/noise_panel.csv"
DEFAULT_REFERENCE_SUMMARY = REFERENCE_OUTPUTS / (
    "delphi_3e18_frontier_phase_fiber_results_20260719/center_control_summary.csv"
)
DEFAULT_SCALING_RESULTS = REFERENCE_OUTPUTS / "delphi_scaling_progress_20260625/delphi_scaling_completed_wandb.csv"

TRAIN_PROJECT = "marin-community/marin"
EVAL_PROJECT = "marin-community/marin-eval"
TRAIN_TAG = "delphi-3e18-compact-optimum-path-validation"
EVAL_GROUP = "olmo_base_eval_table9_delphi_compact_optimum_path_validation_20260721"
UNCHEATABLE_METRIC = "eval/uncheatable_eval/bpb"
TABLE9_METRIC = "olmo_base_easy/table9_51_component_macro_bpb"
EXPECTED_CANDIDATES = 15
PLOT_CONFIG = {"toImageButtonOptions": {"format": "png", "scale": 4}}
TARGETS = {
    "uncheatable": ("uncheatable_bpb", "Uncheatable BPB"),
    "table9": ("table9_macro_bpb", "Table-9 macro BPB"),
}
OBSERVED_REFERENCES = {
    "uncheatable": 0.9851201772689819,
    "table9": 1.0575300915544252,
}
REFERENCE_ANCHORS = {
    "uncheatable": "uncheatable_frontier",
    "table9": "table9_frontier",
}
BASELINE_METRICS = {
    "uncheatable": "eval/uncheatable_eval/bpb",
    "table9": "olmo_base_easy_table9_51_component_macro_bpb",
}
BASELINE_SPECS = {
    "uncheatable": (
        ("Proportional", "proportional_3e18", "#64748B", "dot"),
        ("UniMax-8", "unimax8_3e18", "#7C3AED", "dash"),
        (
            "OLMix best-KL",
            "olmix_onephase_uncheatable_d001_kl0p1_cap4_3e18",
            "#D97706",
            "dashdot",
        ),
    ),
    "table9": (
        ("Proportional", "proportional_3e18", "#64748B", "dot"),
        ("UniMax-8", "unimax8_3e18", "#7C3AED", "dash"),
        (
            "OLMix best-KL",
            "olmix_onephase_table9_d001_kl0p005_cap4_3e18",
            "#D97706",
            "dashdot",
        ),
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel-dir", type=Path, default=DEFAULT_PANEL_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--noise-panel", type=Path, default=DEFAULT_NOISE_PANEL)
    parser.add_argument("--reference-summary", type=Path, default=DEFAULT_REFERENCE_SUMMARY)
    parser.add_argument("--scaling-results", type=Path, default=DEFAULT_SCALING_RESULTS)
    parser.add_argument("--wandb-timeout", type=int, default=180)
    return parser.parse_args()


def finite_summary(run: Any, key: str) -> float:
    value = run.summary.get(key)
    if value is None or not math.isfinite(float(value)):
        raise ValueError(f"Run {run.name!r} has no finite {key!r}: {value!r}")
    return float(value)


def one_finished_run(runs: list[Any], *, expected_name: str, hash_suffix: bool) -> Any:
    if hash_suffix:
        matches = [run for run in runs if run.name.startswith(f"{expected_name}-")]
    else:
        matches = [run for run in runs if run.name == expected_name]
    finished = [run for run in matches if run.state == "finished"]
    if len(finished) != 1:
        states = [(run.name, run.id, run.state) for run in matches]
        raise ValueError(f"Expected one finished run for {expected_name!r}, found {states}")
    return finished[0]


def parse_fit_rows(value: object) -> tuple[int, ...]:
    return tuple(int(part) for part in str(value).split(","))


def collect_results(manifest: pd.DataFrame, *, timeout: int) -> tuple[pd.DataFrame, dict[str, object]]:
    api = wandb.Api(timeout=timeout)
    training_runs = list(
        api.runs(TRAIN_PROJECT, filters={"tags": {"$in": [TRAIN_TAG]}}, per_page=EXPECTED_CANDIDATES + 20)
    )
    eval_runs = list(api.runs(EVAL_PROJECT, filters={"group": EVAL_GROUP}, per_page=EXPECTED_CANDIDATES + 20))

    rows: list[dict[str, object]] = []
    for run_order, record in enumerate(manifest.to_dict(orient="records")):
        candidate_id = str(record["candidate_id"])
        base_name = f"crsv_{run_order:03d}_{candidate_id}"
        training_run = one_finished_run(training_runs, expected_name=base_name, hash_suffix=True)
        eval_name = f"t9_{base_name}"
        eval_run = one_finished_run(eval_runs, expected_name=eval_name, hash_suffix=False)
        eval_attempts = [run for run in eval_runs if run.name == eval_name]
        fit_rows = parse_fit_rows(record["source_fit_row_counts"])
        rows.append(
            {
                **record,
                "run_order": run_order,
                "fit_rows_min": min(fit_rows),
                "fit_rows_latest": max(fit_rows),
                "fit_rows_label": "/".join(str(value) for value in fit_rows),
                "training_wandb_id": training_run.id,
                "training_wandb_name": training_run.name,
                "training_wandb_url": training_run.url,
                "training_state": training_run.state,
                "training_final_step": int(training_run.summary["_step"]),
                "data_seed": int(training_run.config["data_seed"]),
                "eval_wandb_id": eval_run.id,
                "eval_wandb_name": eval_run.name,
                "eval_wandb_url": eval_run.url,
                "eval_state": eval_run.state,
                "eval_attempt_count": len(eval_attempts),
                "eval_failed_attempt_count": sum(run.state != "finished" for run in eval_attempts),
                "uncheatable_bpb": finite_summary(training_run, UNCHEATABLE_METRIC),
                "table9_macro_bpb": finite_summary(eval_run, TABLE9_METRIC),
            }
        )

    results = pd.DataFrame(rows).sort_values("run_order").reset_index(drop=True)
    if len(results) != EXPECTED_CANDIDATES or results["candidate_id"].duplicated().any():
        raise ValueError(f"Expected {EXPECTED_CANDIDATES} unique completed candidates, found {len(results)}")
    if set(results["training_final_step"]) != {3006}:
        raise ValueError(f"Unexpected final training steps: {sorted(results['training_final_step'].unique())}")

    results["proposal_predicted_target_bpb"] = results["proposal_predicted_bpb_latest"].astype(float)
    results["observed_target_bpb"] = [
        float(row[TARGETS[str(row["target"])][0]]) for row in results.to_dict(orient="records")
    ]
    results["target_optimism_bpb"] = results["observed_target_bpb"] - results["proposal_predicted_target_bpb"]
    results["gap_vs_selected_reference_bpb"] = [
        float(row["observed_target_bpb"]) - OBSERVED_REFERENCES[str(row["target"])]
        for row in results.to_dict(orient="records")
    ]
    audit = {
        "queried_at": datetime.now(UTC).isoformat(),
        "training_project": TRAIN_PROJECT,
        "training_tag": TRAIN_TAG,
        "training_attempt_count": len(training_runs),
        "eval_project": EVAL_PROJECT,
        "eval_group": EVAL_GROUP,
        "eval_attempt_count": len(eval_runs),
        "selected_candidate_count": len(results),
        "selected_training_run_ids": results["training_wandb_id"].tolist(),
        "selected_eval_run_ids": results["eval_wandb_id"].tolist(),
        "non_finished_eval_attempts": [
            {"name": run.name, "id": run.id, "state": run.state} for run in eval_runs if run.state != "finished"
        ],
    }
    return results, audit


def reference_table(reference_summary: pd.DataFrame, noise_panel: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for target, (metric, _) in TARGETS.items():
        anchor = REFERENCE_ANCHORS[target]
        match = reference_summary[reference_summary["anchor_id"].eq(anchor) & reference_summary["target"].eq(target)]
        if len(match) != 1:
            raise ValueError(f"Expected one repeat reference for {target}, found {len(match)}")
        row = match.iloc[0]
        repeat_sd = float(noise_panel[metric].std(ddof=1))
        rows.append(
            {
                "target": target,
                "metric": metric,
                "selected_low_draw_reference_bpb": OBSERVED_REFERENCES[target],
                "fresh_same_coordinate_mean_bpb": float(row["fresh_center_mean_bpb"]),
                "fresh_same_coordinate_sd_bpb": float(row["fresh_center_sd_bpb"]),
                "fresh_same_coordinate_n": int(row["n_fresh_centers"]),
                "proportional_repeat_sd_bpb": repeat_sd,
                "independent_two_run_difference_sd_bpb": math.sqrt(2.0) * repeat_sd,
            }
        )
    return pd.DataFrame(rows)


def baseline_table(scaling_results: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for target, specs in BASELINE_SPECS.items():
        metric = BASELINE_METRICS[target]
        for label, run_base, color, dash in specs:
            match = scaling_results[scaling_results["run_base"].eq(run_base)]
            if len(match) != 1:
                raise ValueError(f"Expected one scaling result for {run_base!r}, found {len(match)}")
            row = match.iloc[0]
            value = float(row[metric])
            if not math.isfinite(value):
                raise ValueError(f"Scaling result {run_base!r} has no finite {metric!r}: {row[metric]!r}")
            rows.append(
                {
                    "target": target,
                    "label": label,
                    "run_base": run_base,
                    "observed_bpb": value,
                    "wandb_url": row["wandb_url"],
                    "color": color,
                    "dash": dash,
                }
            )
    return pd.DataFrame(rows)


def path_summary(results: pd.DataFrame, references: pd.DataFrame) -> pd.DataFrame:
    reference_lookup = references.set_index("target")
    rows: list[dict[str, object]] = []
    for target, group in results.groupby("target", sort=False):
        path = group[group["design"].eq("two_phase_only")].copy()
        tied = group[group["design"].eq("tied_spine_plus_two_phase")]
        if len(tied) != 1:
            raise ValueError(f"Expected one tied-spine endpoint for {target}, found {len(tied)}")
        best = path.loc[path["observed_target_bpb"].idxmin()]
        predicted_best = path.loc[path["proposal_predicted_target_bpb"].idxmin()]
        slope = float(stats.linregress(path["proposal_predicted_target_bpb"], path["observed_target_bpb"]).slope)
        spearman = float(stats.spearmanr(path["proposal_predicted_target_bpb"], path["observed_target_bpb"]).statistic)
        reference = reference_lookup.loc[target]
        rows.append(
            {
                "target": target,
                "path_candidates": len(path),
                "best_observed_candidate": best["candidate_id"],
                "best_observed_fit_rows": best["fit_rows_label"],
                "best_observed_target_bpb": float(best["observed_target_bpb"]),
                "best_observed_gap_vs_selected_reference_bpb": float(best["gap_vs_selected_reference_bpb"]),
                "best_observed_gap_vs_fresh_reference_mean_bpb": (
                    float(best["observed_target_bpb"]) - float(reference["fresh_same_coordinate_mean_bpb"])
                ),
                "predicted_best_candidate": predicted_best["candidate_id"],
                "predicted_best_fit_rows": predicted_best["fit_rows_label"],
                "predicted_best_bpb": float(predicted_best["proposal_predicted_target_bpb"]),
                "predicted_best_observed_bpb": float(predicted_best["observed_target_bpb"]),
                "predicted_best_optimism_bpb": float(predicted_best["target_optimism_bpb"]),
                "path_rmse_bpb": float(np.sqrt(np.mean(np.square(path["target_optimism_bpb"].to_numpy(float))))),
                "path_mean_optimism_bpb": float(path["target_optimism_bpb"].mean()),
                "path_worst_optimism_bpb": float(path["target_optimism_bpb"].max()),
                "observed_on_predicted_slope": slope,
                "predicted_observed_spearman": spearman,
                "observed_path_range_bpb": float(path["observed_target_bpb"].max() - path["observed_target_bpb"].min()),
                "predicted_path_range_bpb": float(
                    path["proposal_predicted_target_bpb"].max() - path["proposal_predicted_target_bpb"].min()
                ),
                "tied_spine_endpoint_bpb": float(tied.iloc[0]["observed_target_bpb"]),
                "tied_spine_minus_best_path_bpb": (
                    float(tied.iloc[0]["observed_target_bpb"]) - float(best["observed_target_bpb"])
                ),
            }
        )
    return pd.DataFrame(rows)


def plot_target_paths(
    results: pd.DataFrame,
    references: pd.DataFrame,
    baselines: pd.DataFrame,
    output_path: Path,
) -> None:
    colors = sample_colorscale("RdYlGn_r", [0.12, 0.78])
    figure = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Uncheatable-proposed path", "Table-9-proposed path"),
        horizontal_spacing=0.12,
    )
    references_by_target = references.set_index("target")
    for column, (target, (_, label)) in enumerate(TARGETS.items(), start=1):
        group = results[results["target"].eq(target)]
        path = group[group["design"].eq("two_phase_only")].sort_values("fit_rows_latest")
        tied = group[group["design"].eq("tied_spine_plus_two_phase")].iloc[0]
        x_min = float(path["fit_rows_latest"].min())
        x_max = float(max(path["fit_rows_latest"].max(), tied["fit_rows_latest"]))
        custom = np.stack(
            [
                path["fit_rows_label"].to_numpy(),
                path["candidate_id"].to_numpy(),
                path["target_optimism_bpb"].to_numpy(),
                path["phase_total_variation"].to_numpy(),
                path["standardized_fit_support_distance"].to_numpy(),
            ],
            axis=1,
        )
        figure.add_trace(
            go.Scatter(
                x=path["fit_rows_latest"],
                y=path["observed_target_bpb"],
                mode="lines+markers",
                name="Observed raw optimum",
                legendgroup="observed",
                showlegend=column == 1,
                line={"color": colors[0], "width": 3},
                marker={"size": 10},
                customdata=custom,
                hovertemplate=(
                    "Fit rows=%{customdata[0]}<br>%{customdata[1]}<br>Observed=%{y:.6f}<br>"
                    "Optimism=%{customdata[2]:.6f}<br>Phase TV=%{customdata[3]:.4f}<br>"
                    "Support distance=%{customdata[4]:.2f}<extra></extra>"
                ),
            ),
            row=1,
            col=column,
        )
        for baseline in baselines[baselines["target"].eq(target)].to_dict(orient="records"):
            figure.add_trace(
                go.Scatter(
                    x=[x_min, x_max],
                    y=[baseline["observed_bpb"], baseline["observed_bpb"]],
                    mode="lines",
                    name=str(baseline["label"]),
                    legendgroup=f"baseline-{baseline['label']}",
                    showlegend=column == 1,
                    line={
                        "color": baseline["color"],
                        "dash": baseline["dash"],
                        "width": 2,
                    },
                    customdata=[
                        [baseline["run_base"], baseline["observed_bpb"], baseline["wandb_url"]],
                        [baseline["run_base"], baseline["observed_bpb"], baseline["wandb_url"]],
                    ],
                    hovertemplate=(
                        f"{baseline['label']}<br>"
                        "%{customdata[0]}<br>Observed=%{customdata[1]:.6f}<br>"
                        "%{customdata[2]}<extra></extra>"
                    ),
                ),
                row=1,
                col=column,
            )
        figure.add_trace(
            go.Scatter(
                x=path["fit_rows_latest"],
                y=path["proposal_predicted_target_bpb"],
                mode="lines+markers",
                name="Surrogate prediction",
                legendgroup="predicted",
                showlegend=column == 1,
                line={"color": colors[1], "width": 2.5, "dash": "dash"},
                marker={"size": 9, "symbol": "circle-open"},
                customdata=custom,
                hovertemplate=(
                    "Fit rows=%{customdata[0]}<br>%{customdata[1]}<br>Predicted=%{y:.6f}<br>"
                    "Optimism=%{customdata[2]:.6f}<extra></extra>"
                ),
            ),
            row=1,
            col=column,
        )
        figure.add_trace(
            go.Scatter(
                x=[tied["fit_rows_latest"]],
                y=[tied["observed_target_bpb"]],
                mode="markers",
                name="Tied-spine + 2p endpoint (observed)",
                legendgroup="tied-observed",
                showlegend=column == 1,
                marker={"size": 13, "symbol": "diamond", "color": colors[0], "line": {"width": 1}},
                customdata=[[tied["candidate_id"], tied["target_optimism_bpb"]]],
                hovertemplate="%{customdata[0]}<br>Observed=%{y:.6f}<br>Optimism=%{customdata[1]:.6f}<extra></extra>",
            ),
            row=1,
            col=column,
        )
        figure.add_trace(
            go.Scatter(
                x=[tied["fit_rows_latest"]],
                y=[tied["proposal_predicted_target_bpb"]],
                mode="markers",
                name="Tied-spine + 2p endpoint (predicted)",
                legendgroup="tied-predicted",
                showlegend=column == 1,
                marker={
                    "size": 13,
                    "symbol": "diamond-open",
                    "color": colors[1],
                    "line": {"width": 2},
                },
                customdata=[[tied["candidate_id"], tied["target_optimism_bpb"]]],
                hovertemplate="%{customdata[0]}<br>Predicted=%{y:.6f}<br>Optimism=%{customdata[1]:.6f}<extra></extra>",
            ),
            row=1,
            col=column,
        )
        reference = references_by_target.loc[target]
        figure.add_hline(
            y=float(reference["selected_low_draw_reference_bpb"]),
            line={"color": "#334155", "dash": "dot", "width": 1.5},
            annotation_text="selected low draw",
            annotation_position="bottom right",
            row=1,
            col=column,
        )
        figure.add_hline(
            y=float(reference["fresh_same_coordinate_mean_bpb"]),
            line={"color": "#64748B", "dash": "dash", "width": 1.5},
            annotation_text="fresh repeat mean",
            annotation_position="top right",
            row=1,
            col=column,
        )
        figure.update_xaxes(title_text="Unique 3e18 training policies used in fit", row=1, col=column)
        figure.update_yaxes(title_text=f"{label} (lower is better)", row=1, col=column)

    figure.update_layout(
        title="Compact retained-state raw optima: more fit rows stabilize policy, not deployment calibration",
        template="plotly_white",
        height=680,
        width=1420,
        legend={"orientation": "h", "x": 0.5, "xanchor": "center", "y": -0.18},
        margin={"t": 100, "b": 130},
        hovermode="closest",
    )
    figure.write_html(output_path, include_plotlyjs="cdn", config=PLOT_CONFIG)


def plot_cross_target(results: pd.DataFrame, references: pd.DataFrame, output_path: Path) -> None:
    reference_lookup = references.set_index("target")
    symbols = {
        ("uncheatable", "two_phase_only"): "circle",
        ("table9", "two_phase_only"): "square",
        ("uncheatable", "tied_spine_plus_two_phase"): "diamond",
        ("table9", "tied_spine_plus_two_phase"): "diamond",
    }
    figure = go.Figure()
    for (target, design), group in results.groupby(["target", "design"], sort=False):
        label = {
            ("uncheatable", "two_phase_only"): "Uncheatable path",
            ("table9", "two_phase_only"): "Table-9 path",
            ("uncheatable", "tied_spine_plus_two_phase"): "Uncheatable tied-spine endpoint",
            ("table9", "tied_spine_plus_two_phase"): "Table-9 tied-spine endpoint",
        }[(target, design)]
        custom = np.stack(
            [
                group["candidate_id"].to_numpy(),
                group["fit_rows_label"].to_numpy(),
                group["proposal_predicted_target_bpb"].to_numpy(),
                group["target_optimism_bpb"].to_numpy(),
                group["phase_total_variation"].to_numpy(),
            ],
            axis=1,
        )
        figure.add_trace(
            go.Scatter(
                x=group["uncheatable_bpb"],
                y=group["table9_macro_bpb"],
                mode="markers",
                name=label,
                marker={
                    "size": 12 if design == "two_phase_only" else 15,
                    "symbol": symbols[(target, design)],
                    "color": group["fit_rows_latest"],
                    "colorscale": "RdYlGn_r",
                    "cmin": 280,
                    "cmax": 998,
                    "showscale": target == "table9" and design == "two_phase_only",
                    "colorbar": {"title": "Fit rows"},
                    "line": {"color": "#334155", "width": 1},
                },
                customdata=custom,
                hovertemplate=(
                    "%{customdata[0]}<br>Fit rows=%{customdata[1]}<br>Uncheatable=%{x:.6f}<br>"
                    "Table-9=%{y:.6f}<br>Target prediction=%{customdata[2]:.6f}<br>"
                    "Target optimism=%{customdata[3]:.6f}<br>Phase TV=%{customdata[4]:.4f}<extra></extra>"
                ),
            )
        )
    figure.add_vline(
        x=float(reference_lookup.loc["uncheatable", "selected_low_draw_reference_bpb"]),
        line={"color": "#334155", "dash": "dot"},
    )
    figure.add_hline(
        y=float(reference_lookup.loc["table9", "selected_low_draw_reference_bpb"]),
        line={"color": "#334155", "dash": "dot"},
    )
    figure.add_vline(
        x=float(reference_lookup.loc["uncheatable", "fresh_same_coordinate_mean_bpb"]),
        line={"color": "#64748B", "dash": "dash"},
    )
    figure.add_hline(
        y=float(reference_lookup.loc["table9", "fresh_same_coordinate_mean_bpb"]),
        line={"color": "#64748B", "dash": "dash"},
    )
    figure.update_layout(
        title="Compact raw-optimum policies remain target-specialized; none advances either observed reference",
        template="plotly_white",
        width=1050,
        height=760,
        xaxis_title="Observed Uncheatable BPB (lower is better)",
        yaxis_title="Observed Table-9 macro BPB (lower is better)",
        legend={"orientation": "h", "x": 0.5, "xanchor": "center", "y": -0.17},
        margin={"t": 100, "b": 130},
    )
    figure.write_html(output_path, include_plotlyjs="cdn", config=PLOT_CONFIG)


def write_report(
    results: pd.DataFrame,
    summaries: pd.DataFrame,
    references: pd.DataFrame,
    audit: dict[str, object],
    output_dir: Path,
) -> None:
    lookup = summaries.set_index("target")
    uncheatable = lookup.loc["uncheatable"]
    table9 = lookup.loc["table9"]
    target_table = results[
        [
            "candidate_id",
            "target",
            "design",
            "fit_rows_label",
            "proposal_predicted_target_bpb",
            "observed_target_bpb",
            "target_optimism_bpb",
            "gap_vs_selected_reference_bpb",
            "uncheatable_bpb",
            "table9_macro_bpb",
            "training_wandb_url",
            "eval_wandb_url",
        ]
    ]
    lines = [
        "# Compact Retained State raw-optimum path validation results",
        "",
        "## Verdict",
        "",
        "All 15 Delphi 3e18 checkpoints and all 15 native Table-9 evaluations completed. No Compact raw optimum "
        "advances the existing observed references. The best target-matched policies reach "
        f"{uncheatable['best_observed_target_bpb']:.6f} Uncheatable BPB and "
        f"{table9['best_observed_target_bpb']:.6f} Table-9 BPB, respectively "
        f"({uncheatable['best_observed_gap_vs_selected_reference_bpb']:+.6f} and "
        f"{table9['best_observed_gap_vs_selected_reference_bpb']:+.6f} versus the selected low-draw references).",
        "",
        "The proposed policies become much more stable after 340 fit rows, but deployment calibration does not "
        "improve with more evidence. The two-phase-only paths remain optimistically biased by "
        f"{uncheatable['path_mean_optimism_bpb']:.4f} BPB on Uncheatable and "
        f"{table9['path_mean_optimism_bpb']:.4f} BPB on Table-9. The 280-row Table-9 fit is the sharpest failure: "
        f"it predicts {table9['predicted_best_bpb']:.6f}, observes "
        f"{table9['predicted_best_observed_bpb']:.6f}, and is optimistic by "
        f"{table9['predicted_best_optimism_bpb']:.6f} BPB.",
        "",
        "The maximum-evidence tied-spine-plus-two-phase fits are also not rescued: their raw optima are "
        f"{uncheatable['tied_spine_minus_best_path_bpb']:+.6f} and "
        f"{table9['tied_spine_minus_best_path_bpb']:+.6f} BPB worse than the best two-phase-only path points. "
        "This is evidence against using unregularized Compact optima, not evidence against the Compact form as a "
        "local ranking surrogate.",
        "",
        "## Path summary",
        "",
        summaries.to_markdown(index=False),
        "",
        "The selected low-draw references are useful historical comparisons, not certified global frontiers. "
        "Fresh same-coordinate means are included separately because the Table-9 reference in particular was an "
        "unreplicated favorable draw.",
        "",
        "## Reference and noise context",
        "",
        references.to_markdown(index=False),
        "",
        "The proportional repeat standard deviations are descriptive magnitudes only. These candidates use "
        "different mixtures, and no candidate repeats were scheduled.",
        "",
        "## All candidates",
        "",
        target_table.to_markdown(index=False),
        "",
        "## Collection audit",
        "",
        "```json",
        json.dumps(audit, indent=2, sort_keys=True),
        "```",
        "",
        "## Decision",
        "",
        "Do not deploy or scale any raw Compact optimum from this path. Preserve all 15 policies as append-only "
        "3e18 development heldouts. The useful result is a clean falsification: policy convergence under more "
        "fit rows is not sufficient for trustworthy optimum transfer, and the tied-spine evidence design does not "
        "repair the unsupported raw surface.",
    ]
    (output_dir / "report.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(args.panel_dir / "launcher_source_panel.csv")
    if len(manifest) != EXPECTED_CANDIDATES or manifest["candidate_id"].duplicated().any():
        raise ValueError(f"Expected {EXPECTED_CANDIDATES} unique panel rows, found {len(manifest)}")

    results, audit = collect_results(manifest, timeout=args.wandb_timeout)
    references = reference_table(pd.read_csv(args.reference_summary), pd.read_csv(args.noise_panel))
    baselines = baseline_table(pd.read_csv(args.scaling_results))
    summaries = path_summary(results, references)

    results.to_csv(args.output_dir / "observed_results.csv", index=False, float_format="%.17g")
    references.to_csv(args.output_dir / "reference_context.csv", index=False, float_format="%.17g")
    baselines.to_csv(args.output_dir / "baseline_context.csv", index=False, float_format="%.17g")
    summaries.to_csv(args.output_dir / "path_summary.csv", index=False, float_format="%.17g")
    (args.output_dir / "collection_audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    plot_target_paths(
        results,
        references,
        baselines,
        args.output_dir / "compact_optimum_paths_predicted_vs_observed.html",
    )
    plot_cross_target(results, references, args.output_dir / "compact_optimum_paths_cross_target.html")
    write_report(results, summaries, references, audit, args.output_dir)
    print(summaries.to_string(index=False))
    print(f"Wrote results to {args.output_dir}")


if __name__ == "__main__":
    main()
