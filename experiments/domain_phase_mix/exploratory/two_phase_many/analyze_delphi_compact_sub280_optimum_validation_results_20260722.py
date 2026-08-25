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
"""Analyze the Delphi 3e18 Compact sub-280 raw-optimum validation panel."""

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
DEFAULT_PANEL_DIR = REFERENCE_OUTPUTS / "delphi_compact_sub280_optimum_validation_panel_20260721"
DEFAULT_PRIOR_RESULTS = REFERENCE_OUTPUTS / (
    "delphi_compact_optimum_path_validation_results_20260721/observed_results.csv"
)
DEFAULT_OUTPUT_DIR = REFERENCE_OUTPUTS / "delphi_compact_sub280_optimum_validation_results_20260722"

TRAIN_PROJECT = "marin-community/marin"
EVAL_PROJECT = "marin-community/marin-eval"
TRAIN_TAG = "delphi-3e18-compact-sub280-optimum-validation"
EVAL_GROUP = "olmo_base_eval_table9_delphi_compact_sub280_optimum_validation_20260721"
UNCHEATABLE_METRIC = "eval/uncheatable_eval/bpb"
TABLE9_METRIC = "olmo_base_easy/table9_51_component_macro_bpb"
EXPECTED_CANDIDATES = 140
EXPECTED_REPLICATES = 5
EXPECTED_FIT_ROWS = (48, 64, 80, 112, 144, 184, 232)
PLOT_CONFIG = {"toImageButtonOptions": {"format": "png", "scale": 4}}
TARGETS = {
    "uncheatable": ("uncheatable_bpb", "Uncheatable BPB"),
    "table9": ("table9_macro_bpb", "Table-9 macro BPB"),
}
DESIGN_LABELS = {
    "intervention_core": "Intervention core",
    "panel_stratified": "Panel stratified",
}
SELECTED_REFERENCES = {
    "uncheatable": 0.9851201772689819,
    "table9": 1.0575300915544252,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel-dir", type=Path, default=DEFAULT_PANEL_DIR)
    parser.add_argument("--prior-results", type=Path, default=DEFAULT_PRIOR_RESULTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
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


def collect_results(manifest: pd.DataFrame, *, timeout: int) -> tuple[pd.DataFrame, dict[str, object]]:
    api = wandb.Api(timeout=timeout)
    training_runs = list(
        api.runs(TRAIN_PROJECT, filters={"tags": {"$in": [TRAIN_TAG]}}, per_page=EXPECTED_CANDIDATES + 80)
    )
    eval_runs = list(api.runs(EVAL_PROJECT, filters={"group": EVAL_GROUP}, per_page=EXPECTED_CANDIDATES + 100))

    rows: list[dict[str, object]] = []
    for run_order, record in enumerate(manifest.to_dict(orient="records")):
        candidate_id = str(record["candidate_id"])
        base_name = f"crslowv_{run_order:03d}_{candidate_id}"
        training_run = one_finished_run(training_runs, expected_name=base_name, hash_suffix=True)
        eval_name = f"t9_{base_name}"
        eval_run = one_finished_run(eval_runs, expected_name=eval_name, hash_suffix=False)
        eval_attempts = [run for run in eval_runs if run.name == eval_name]
        rows.append(
            {
                **record,
                "run_order": run_order,
                "training_wandb_id": training_run.id,
                "training_wandb_name": training_run.name,
                "training_wandb_url": training_run.url,
                "training_state": training_run.state,
                "training_final_step": int(training_run.summary["_step"]),
                "data_seed": int(training_run.config["data_seed"]),
                "trainer_seed": int(training_run.config["trainer"]["seed"]),
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
    if set(results["fit_rows"]) != set(EXPECTED_FIT_ROWS):
        raise ValueError(f"Unexpected fit-row budgets: {sorted(results['fit_rows'].unique())}")
    group_sizes = results.groupby(["target", "sampling_design", "fit_rows"]).size()
    if set(group_sizes) != {EXPECTED_REPLICATES}:
        raise ValueError(f"Unexpected replicate counts: {group_sizes.to_dict()}")

    results["proposal_predicted_target_bpb"] = results["proposal_predicted_bpb"].astype(float)
    results["observed_target_bpb"] = [
        float(row[TARGETS[str(row["target"])][0]]) for row in results.to_dict(orient="records")
    ]
    results["target_optimism_bpb"] = results["observed_target_bpb"] - results["proposal_predicted_target_bpb"]
    results["gap_vs_selected_reference_bpb"] = [
        float(row["observed_target_bpb"]) - SELECTED_REFERENCES[str(row["target"])]
        for row in results.to_dict(orient="records")
    ]
    audit: dict[str, object] = {
        "queried_at": datetime.now(UTC).isoformat(),
        "training_project": TRAIN_PROJECT,
        "training_tag": TRAIN_TAG,
        "training_attempt_count": len(training_runs),
        "training_attempt_states": {
            str(state): int(count)
            for state, count in pd.Series([run.state for run in training_runs]).value_counts().items()
        },
        "eval_project": EVAL_PROJECT,
        "eval_group": EVAL_GROUP,
        "eval_attempt_count": len(eval_runs),
        "eval_attempt_states": {
            str(state): int(count) for state, count in pd.Series([run.state for run in eval_runs]).value_counts().items()
        },
        "selected_candidate_count": len(results),
        "selected_training_run_ids": results["training_wandb_id"].tolist(),
        "selected_eval_run_ids": results["eval_wandb_id"].tolist(),
    }
    return results, audit


def finite_spearman(x: pd.Series, y: pd.Series) -> float:
    if x.nunique() < 2 or y.nunique() < 2:
        return math.nan
    return float(stats.spearmanr(x, y).statistic)


def finite_slope(x: pd.Series, y: pd.Series) -> float:
    if x.nunique() < 2:
        return math.nan
    return float(stats.linregress(x, y).slope)


def learning_curve_summary(results: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (target, design, fit_rows), group in results.groupby(["target", "sampling_design", "fit_rows"], sort=False):
        group = group.copy()
        selected = group.loc[group["proposal_predicted_target_bpb"].idxmin()]
        best = group.loc[group["observed_target_bpb"].idxmin()]
        rows.append(
            {
                "target": target,
                "sampling_design": design,
                "sampling_design_label": DESIGN_LABELS[str(design)],
                "fit_rows": int(fit_rows),
                "replicates": len(group),
                "predicted_mean_bpb": float(group["proposal_predicted_target_bpb"].mean()),
                "predicted_sd_bpb": float(group["proposal_predicted_target_bpb"].std(ddof=1)),
                "predicted_min_bpb": float(group["proposal_predicted_target_bpb"].min()),
                "predicted_range_bpb": float(
                    group["proposal_predicted_target_bpb"].max() - group["proposal_predicted_target_bpb"].min()
                ),
                "observed_mean_bpb": float(group["observed_target_bpb"].mean()),
                "observed_sd_bpb": float(group["observed_target_bpb"].std(ddof=1)),
                "observed_min_bpb": float(group["observed_target_bpb"].min()),
                "observed_range_bpb": float(group["observed_target_bpb"].max() - group["observed_target_bpb"].min()),
                "mean_optimism_bpb": float(group["target_optimism_bpb"].mean()),
                "rmse_bpb": float(np.sqrt(np.mean(np.square(group["target_optimism_bpb"].to_numpy(float))))),
                "worst_optimism_bpb": float(group["target_optimism_bpb"].max()),
                "observed_on_predicted_slope": finite_slope(
                    group["proposal_predicted_target_bpb"], group["observed_target_bpb"]
                ),
                "predicted_observed_spearman": finite_spearman(
                    group["proposal_predicted_target_bpb"], group["observed_target_bpb"]
                ),
                "predicted_selected_candidate": selected["candidate_id"],
                "predicted_selected_seed": int(selected["subset_seed"]),
                "predicted_selected_predicted_bpb": float(selected["proposal_predicted_target_bpb"]),
                "predicted_selected_observed_bpb": float(selected["observed_target_bpb"]),
                "predicted_selected_optimism_bpb": float(selected["target_optimism_bpb"]),
                "best_observed_candidate": best["candidate_id"],
                "best_observed_seed": int(best["subset_seed"]),
                "best_observed_bpb": float(best["observed_target_bpb"]),
                "selection_regret_within_five_bpb": float(selected["observed_target_bpb"] - best["observed_target_bpb"]),
                "mean_max_bucket_weight": float(group["max_bucket_weight"].mean()),
                "mean_max_simulated_epochs": float(group["max_simulated_epochs"].mean()),
                "mean_phase_total_variation": float(group["phase_total_variation"].mean()),
                "mean_support_distance": float(group["standardized_fit_support_distance"].mean()),
                "corner_optimum_fraction": float(group["max_bucket_weight"].ge(0.999).mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(["target", "sampling_design", "fit_rows"]).reset_index(drop=True)


def target_summary(results: pd.DataFrame, curve: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for target, group in results.groupby("target", sort=False):
        best = group.loc[group["observed_target_bpb"].idxmin()]
        predicted_best = group.loc[group["proposal_predicted_target_bpb"].idxmin()]
        selected_by_group = curve[curve["target"].eq(target)]
        best_group = selected_by_group.loc[selected_by_group["observed_mean_bpb"].idxmin()]
        rows.append(
            {
                "target": target,
                "candidates": len(group),
                "best_observed_candidate": best["candidate_id"],
                "best_observed_design": best["sampling_design"],
                "best_observed_fit_rows": int(best["fit_rows"]),
                "best_observed_bpb": float(best["observed_target_bpb"]),
                "best_observed_gap_vs_reference_bpb": float(best["gap_vs_selected_reference_bpb"]),
                "predicted_best_candidate": predicted_best["candidate_id"],
                "predicted_best_observed_bpb": float(predicted_best["observed_target_bpb"]),
                "predicted_best_optimism_bpb": float(predicted_best["target_optimism_bpb"]),
                "best_mean_design": best_group["sampling_design"],
                "best_mean_fit_rows": int(best_group["fit_rows"]),
                "best_mean_observed_bpb": float(best_group["observed_mean_bpb"]),
                "overall_rmse_bpb": float(np.sqrt(np.mean(np.square(group["target_optimism_bpb"].to_numpy(float))))),
                "overall_mean_optimism_bpb": float(group["target_optimism_bpb"].mean()),
                "overall_worst_optimism_bpb": float(group["target_optimism_bpb"].max()),
                "overall_observed_on_predicted_slope": finite_slope(
                    group["proposal_predicted_target_bpb"], group["observed_target_bpb"]
                ),
                "overall_predicted_observed_spearman": finite_spearman(
                    group["proposal_predicted_target_bpb"], group["observed_target_bpb"]
                ),
            }
        )
    return pd.DataFrame(rows)


def plot_full_learning_curve(
    results: pd.DataFrame,
    curve: pd.DataFrame,
    prior: pd.DataFrame,
    output_path: Path,
) -> None:
    colors = sample_colorscale("RdYlGn_r", [0.12, 0.82])
    figure = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Uncheatable-proposed optima", "Table-9-proposed optima"),
        horizontal_spacing=0.11,
    )
    for column, (target, (_, label)) in enumerate(TARGETS.items(), start=1):
        target_results = results[results["target"].eq(target)]
        for design_index, (design, design_label) in enumerate(DESIGN_LABELS.items()):
            raw = target_results[target_results["sampling_design"].eq(design)]
            summary = curve[curve["target"].eq(target) & curve["sampling_design"].eq(design)]
            color = colors[design_index]
            figure.add_trace(
                go.Scatter(
                    x=raw["fit_rows"],
                    y=raw["observed_target_bpb"],
                    mode="markers",
                    name=f"{design_label}: observed replicates",
                    legendgroup=f"{target}-{design}",
                    showlegend=column == 1,
                    marker={"size": 7, "color": color, "opacity": 0.28},
                    customdata=np.stack([raw["candidate_id"], raw["subset_seed"]], axis=1),
                    hovertemplate=(
                        "fit rows=%{x}<br>observed=%{y:.6f}<br>candidate=%{customdata[0]}"
                        "<br>subset seed=%{customdata[1]}<extra></extra>"
                    ),
                ),
                row=1,
                col=column,
            )
            figure.add_trace(
                go.Scatter(
                    x=summary["fit_rows"],
                    y=summary["observed_mean_bpb"],
                    mode="lines+markers",
                    name=f"{design_label}: observed mean",
                    legendgroup=f"{target}-{design}",
                    showlegend=column == 1,
                    line={"color": color, "width": 3},
                    marker={"size": 9},
                    error_y={"type": "data", "array": summary["observed_sd_bpb"], "visible": True},
                    hovertemplate="fit rows=%{x}<br>observed mean=%{y:.6f}<extra></extra>",
                ),
                row=1,
                col=column,
            )
            figure.add_trace(
                go.Scatter(
                    x=summary["fit_rows"],
                    y=summary["predicted_mean_bpb"],
                    mode="lines+markers",
                    name=f"{design_label}: predicted mean",
                    legendgroup=f"{target}-{design}",
                    showlegend=column == 1,
                    line={"color": color, "width": 2, "dash": "dash"},
                    marker={"symbol": "diamond-open", "size": 8},
                    hovertemplate="fit rows=%{x}<br>predicted mean=%{y:.6f}<extra></extra>",
                ),
                row=1,
                col=column,
            )
        path = prior[prior["target"].eq(target) & prior["design"].eq("two_phase_only")].sort_values("fit_rows_latest")
        figure.add_trace(
            go.Scatter(
                x=path["fit_rows_latest"],
                y=path["observed_target_bpb"],
                mode="lines+markers",
                name="Validated 280+ path: observed",
                legendgroup="prior-observed",
                showlegend=column == 1,
                line={"color": "#172B3A", "width": 3},
                marker={"size": 9, "symbol": "square"},
                hovertemplate="fit rows=%{x}<br>observed=%{y:.6f}<extra></extra>",
            ),
            row=1,
            col=column,
        )
        figure.add_trace(
            go.Scatter(
                x=path["fit_rows_latest"],
                y=path["proposal_predicted_target_bpb"],
                mode="lines+markers",
                name="Validated 280+ path: predicted",
                legendgroup="prior-predicted",
                showlegend=column == 1,
                line={"color": "#172B3A", "width": 2, "dash": "dot"},
                marker={"size": 8, "symbol": "square-open"},
                hovertemplate="fit rows=%{x}<br>predicted=%{y:.6f}<extra></extra>",
            ),
            row=1,
            col=column,
        )
        figure.add_hline(
            y=SELECTED_REFERENCES[target],
            line={"color": "#64748B", "dash": "dash", "width": 1.5},
            annotation_text="selected reference",
            row=1,
            col=column,
        )
        figure.update_xaxes(title_text="Fit rows", type="log", row=1, col=column)
        figure.update_yaxes(title_text=label if column == 1 else None, row=1, col=column)
    figure.update_layout(
        title="Compact raw-optimum transfer across fit-set size",
        template="plotly_white",
        width=1540,
        height=760,
        legend={"orientation": "h", "y": -0.18, "x": 0.0},
        margin={"l": 80, "r": 50, "t": 100, "b": 180},
    )
    output_path.write_text(figure.to_html(full_html=True, include_plotlyjs=True, config=PLOT_CONFIG))


def plot_calibration(results: pd.DataFrame, output_path: Path) -> None:
    figure = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Uncheatable proposals", "Table-9 proposals"),
        horizontal_spacing=0.11,
    )
    for column, (target, (_, label)) in enumerate(TARGETS.items(), start=1):
        group = results[results["target"].eq(target)]
        lower = float(min(group["proposal_predicted_target_bpb"].min(), group["observed_target_bpb"].min()))
        upper = float(max(group["proposal_predicted_target_bpb"].max(), group["observed_target_bpb"].max()))
        for design, design_group in group.groupby("sampling_design", sort=False):
            figure.add_trace(
                go.Scatter(
                    x=design_group["proposal_predicted_target_bpb"],
                    y=design_group["observed_target_bpb"],
                    mode="markers",
                    name=DESIGN_LABELS[str(design)],
                    legendgroup=str(design),
                    showlegend=column == 1,
                    marker={
                        "size": 9,
                        "color": design_group["fit_rows"],
                        "colorscale": "RdYlGn_r",
                        "cmin": min(EXPECTED_FIT_ROWS),
                        "cmax": max(EXPECTED_FIT_ROWS),
                        "symbol": "circle" if design == "intervention_core" else "diamond",
                        "line": {"color": "#172B3A", "width": 0.6},
                    },
                    customdata=np.stack(
                        [design_group["candidate_id"], design_group["fit_rows"], design_group["subset_seed"]],
                        axis=1,
                    ),
                    hovertemplate=(
                        "predicted=%{x:.6f}<br>observed=%{y:.6f}<br>candidate=%{customdata[0]}"
                        "<br>fit rows=%{customdata[1]}<br>subset seed=%{customdata[2]}<extra></extra>"
                    ),
                ),
                row=1,
                col=column,
            )
        figure.add_trace(
            go.Scatter(
                x=[lower, upper],
                y=[lower, upper],
                mode="lines",
                name="identity",
                legendgroup="identity",
                showlegend=column == 1,
                line={"color": "#64748B", "dash": "dash"},
                hoverinfo="skip",
            ),
            row=1,
            col=column,
        )
        figure.update_xaxes(title_text="Predicted target BPB", row=1, col=column)
        figure.update_yaxes(title_text=f"Observed {label}" if column == 1 else None, row=1, col=column)
    figure.update_layout(
        title="Compact sub-280 raw-optimum calibration",
        template="plotly_white",
        width=1450,
        height=700,
        legend={"orientation": "h", "y": -0.15, "x": 0.0},
        margin={"l": 80, "r": 50, "t": 100, "b": 150},
    )
    output_path.write_text(figure.to_html(full_html=True, include_plotlyjs=True, config=PLOT_CONFIG))


def plot_selection(curve: pd.DataFrame, output_path: Path) -> None:
    colors = sample_colorscale("RdYlGn_r", [0.12, 0.82])
    figure = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Uncheatable optimism",
            "Table-9 optimism",
            "Uncheatable within-five selection regret",
            "Table-9 within-five selection regret",
        ),
        vertical_spacing=0.16,
        horizontal_spacing=0.11,
    )
    for column, target in enumerate(TARGETS, start=1):
        for design_index, (design, design_label) in enumerate(DESIGN_LABELS.items()):
            group = curve[curve["target"].eq(target) & curve["sampling_design"].eq(design)]
            color = colors[design_index]
            figure.add_trace(
                go.Scatter(
                    x=group["fit_rows"],
                    y=group["mean_optimism_bpb"],
                    mode="lines+markers",
                    name=design_label,
                    legendgroup=design,
                    showlegend=column == 1,
                    line={"color": color, "width": 3},
                ),
                row=1,
                col=column,
            )
            figure.add_trace(
                go.Scatter(
                    x=group["fit_rows"],
                    y=group["selection_regret_within_five_bpb"],
                    mode="lines+markers",
                    name=design_label,
                    legendgroup=design,
                    showlegend=False,
                    line={"color": color, "width": 3},
                ),
                row=2,
                col=column,
            )
        figure.add_hline(y=0.0, line={"color": "#64748B", "dash": "dash"}, row=1, col=column)
        figure.add_hline(y=0.0, line={"color": "#64748B", "dash": "dash"}, row=2, col=column)
        figure.update_xaxes(title_text="Fit rows", row=2, col=column)
        figure.update_yaxes(title_text="Observed - predicted BPB" if column == 1 else None, row=1, col=column)
        figure.update_yaxes(title_text="Regret BPB" if column == 1 else None, row=2, col=column)
    figure.update_layout(
        title="Raw-optimum optimism and seed-selection quality",
        template="plotly_white",
        width=1450,
        height=1050,
        legend={"orientation": "h", "y": -0.08, "x": 0.0},
        margin={"l": 80, "r": 50, "t": 110, "b": 120},
    )
    output_path.write_text(figure.to_html(full_html=True, include_plotlyjs=True, config=PLOT_CONFIG))


def write_report(
    results: pd.DataFrame,
    curve: pd.DataFrame,
    targets: pd.DataFrame,
    audit: dict[str, object],
    output_path: Path,
) -> None:
    target_rows = targets.set_index("target")
    uncheatable = target_rows.loc["uncheatable"]
    table9 = target_rows.loc["table9"]
    latest = curve[curve["fit_rows"].eq(max(EXPECTED_FIT_ROWS))][
        [
            "target",
            "sampling_design",
            "observed_mean_bpb",
            "observed_sd_bpb",
            "mean_optimism_bpb",
            "selection_regret_within_five_bpb",
            "mean_max_bucket_weight",
            "mean_support_distance",
        ]
    ]
    lines = [
        "# Compact retained-state sub-280 raw-optimum validation results",
        "",
        "## Verdict",
        "",
        f"All {len(results)} Delphi 3e18 checkpoints and all {len(results)} native Table-9 evaluations completed. "
        "The raw optimum is extremely sample-inefficient below 280 fit rows: subset-to-subset policy and predicted "
        "value instability is large, and predicted improvements do not transfer at the claimed magnitude.",
        "",
        f"The best sub-280 Uncheatable proposal observes {uncheatable['best_observed_bpb']:.6f} "
        f"({uncheatable['best_observed_gap_vs_reference_bpb']:+.6f} versus the selected reference). The best Table-9 "
        f"proposal observes {table9['best_observed_bpb']:.6f} "
        f"({table9['best_observed_gap_vs_reference_bpb']:+.6f}). Neither establishes a new frontier.",
        "",
        "This extends the earlier 280+ result: more fit rows make the proposed policy geometry much more stable, but "
        "the unsupported raw objective remains optimistically calibrated. Compact retained state remains useful as a "
        "local ranking model; its unregularized global optimum is not a trustworthy deployment rule.",
        "",
        "## Target summary",
        "",
        targets.to_markdown(index=False),
        "",
        f"## Results at {max(EXPECTED_FIT_ROWS)} fit rows",
        "",
        latest.to_markdown(index=False),
        "",
        "## Per-budget learning curve",
        "",
        curve.to_markdown(index=False),
        "",
        "## Collection audit",
        "",
        "```json",
        json.dumps(audit, indent=2, sort_keys=True),
        "```",
        "",
        "## Decision",
        "",
        "Do not deploy or scale any sub-280 raw Compact optimum. Append all 140 policies to the frozen 3e18 "
        "development heldouts. For sample-efficiency comparisons, use the observed optimum-transfer curve, not only "
        "fit-panel RMSE or apparent policy convergence.",
    ]
    output_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    manifest = pd.read_csv(args.panel_dir / "launcher_source_panel.csv")
    prior = pd.read_csv(args.prior_results)
    if len(manifest) != EXPECTED_CANDIDATES:
        raise ValueError(f"Expected {EXPECTED_CANDIDATES} manifest rows, found {len(manifest)}")
    results, audit = collect_results(manifest, timeout=args.wandb_timeout)
    curve = learning_curve_summary(results)
    targets = target_summary(results, curve)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output_dir / "observed_results.csv", index=False, float_format="%.17g")
    curve.to_csv(args.output_dir / "learning_curve_summary.csv", index=False, float_format="%.17g")
    targets.to_csv(args.output_dir / "target_summary.csv", index=False, float_format="%.17g")
    (args.output_dir / "collection_audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    plot_full_learning_curve(results, curve, prior, args.output_dir / "full_raw_optimum_learning_curve.html")
    plot_calibration(results, args.output_dir / "predicted_vs_observed.html")
    plot_selection(curve, args.output_dir / "optimism_and_selection_regret.html")
    write_report(results, curve, targets, audit, args.output_dir / "report.md")
    print(targets.to_string(index=False))


if __name__ == "__main__":
    main()
