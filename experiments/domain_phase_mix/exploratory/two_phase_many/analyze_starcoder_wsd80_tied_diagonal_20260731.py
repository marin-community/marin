# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "numpy",
#   "pandas",
#   "plotly",
#   "tabulate",
#   "wandb",
# ]
# ///
"""Analyze the completed tied diagonal for the fixed-model WSD80 token ladder."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import wandb
from plotly.colors import sample_colorscale
from plotly.subplots import make_subplots

SCRIPT_DIR = Path(__file__).resolve().parent
REFERENCE_OUTPUTS = SCRIPT_DIR / "reference_outputs"
PANEL_DIR = REFERENCE_OUTPUTS / "starcoder_wsd80_fixed_model_tied_diagonal_20260730"
TOKEN_RESULTS_DIR = REFERENCE_OUTPUTS / "starcoder_wsd80_fixed_model_token_scaling_20260728" / "results_20260730"
SOURCE_SURFACE_DIR = REFERENCE_OUTPUTS / "starcoder_wsd80_surface_refined_20260714"
DEFAULT_OUTPUT_DIR = PANEL_DIR / "results_20260731"

TRAIN_PROJECT = "marin-community/marin"
TRAIN_TAG = "starcoder_wsd80_fixedn_tieddiag"
TARGET_METRIC = "eval/paloma/dolma_100_programing_languages-llama3/bpb"
REFERENCE_SEED = 20260711
TOKEN_BUDGETS = (1_000_000_000, 2_000_000_000, 4_000_000_000, 8_000_000_000)
REGULAR_TIED_WEIGHTS = tuple(index / 20 for index in range(21))
EXPECTED_NEW_RUNS = 60
LOCAL_QUADRATIC_WIDTHS = (0.15, 0.20, 0.25)
PLOT_CONFIG = {"toImageButtonOptions": {"format": "png", "scale": 4}}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel-dir", type=Path, default=PANEL_DIR)
    parser.add_argument("--token-results-dir", type=Path, default=TOKEN_RESULTS_DIR)
    parser.add_argument("--source-surface-dir", type=Path, default=SOURCE_SURFACE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--wandb-timeout", type=int, default=180)
    return parser.parse_args()


def finite_summary(run: Any, key: str) -> float | None:
    try:
        value = float(run.summary.get(key, np.nan))
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def selected_run(runs: list[Any], run_name: str) -> Any:
    candidates = [run for run in runs if str(run.name) == run_name and finite_summary(run, TARGET_METRIC) is not None]
    if len(candidates) != 1:
        raise ValueError(f"{run_name}: expected one finite W&B run, found {len(candidates)}")
    return candidates[0]


def collect_new_observations(panel_dir: Path, timeout: int) -> pd.DataFrame:
    manifest = pd.read_csv(panel_dir / "run_manifest.csv")
    if len(manifest) != EXPECTED_NEW_RUNS or manifest["run_name"].duplicated().any():
        raise ValueError(f"Expected {EXPECTED_NEW_RUNS} unique tied-diagonal manifest rows")

    api = wandb.Api(timeout=timeout)
    runs = list(api.runs(TRAIN_PROJECT, filters={"tags": TRAIN_TAG}, per_page=100))
    rows = []
    for _, spec in manifest.iterrows():
        run = selected_run(runs, str(spec["run_name"]))
        rows.append(
            {
                "token_budget_requested": int(spec["token_budget_requested"]),
                "weight": float(spec["phase_0_starcoder"]),
                "starcoder_bpb": finite_summary(run, TARGET_METRIC),
                "source": "new tied-diagonal completion",
                "run_name": str(run.name),
                "wandb_id": str(run.id),
                "wandb_state": str(run.state),
                "wandb_url": str(run.url),
            }
        )
    result = pd.DataFrame(rows)
    if len(result) != EXPECTED_NEW_RUNS or result["starcoder_bpb"].isna().any():
        raise ValueError("Tied-diagonal W&B collection is incomplete")
    return result


def collect_existing_observations(
    new: pd.DataFrame,
    token_results_dir: Path,
    source_surface_dir: Path,
) -> pd.DataFrame:
    rows = new.to_dict("records")
    surface = pd.read_csv(source_surface_dir / "wsd80_observed_metrics.csv")
    prior = pd.read_csv(token_results_dir / "observations.csv")

    new_one_billion = set(new.loc[new["token_budget_requested"] == 1_000_000_000, "weight"].round(8).tolist())
    for weight in REGULAR_TIED_WEIGHTS:
        if round(weight, 8) in new_one_billion:
            continue
        matched = surface.loc[(surface["phase_0_starcoder"] - weight).abs() < 5e-4]
        matched = matched.loc[(matched["phase_1_starcoder"] - weight).abs() < 5e-4]
        if len(matched) != 1:
            raise ValueError(f"1B tied weight {weight:.2f}: expected one source row, found {len(matched)}")
        row = matched.iloc[0]
        rows.append(
            {
                "token_budget_requested": 1_000_000_000,
                "weight": weight,
                "starcoder_bpb": float(row["wsd80_bpb"]),
                "source": "existing 1B WSD80 surface",
                "run_name": str(row["wandb_run_name"]),
                "wandb_id": str(row["wandb_run_id"]),
                "wandb_state": str(row["wandb_state"]),
                "wandb_url": str(row["wandb_url"]),
            }
        )

    for token_budget in TOKEN_BUDGETS[1:]:
        for weight in (0.10, 0.30, 0.35):
            matched = prior.loc[
                (prior["token_budget_requested"] == token_budget)
                & ((prior["phase_0_starcoder"] - weight).abs() < 1e-12)
                & ((prior["phase_1_starcoder"] - weight).abs() < 1e-12)
                & (prior["trainer_data_seed"] == REFERENCE_SEED)
                & (prior["replicate_kind"] == "reference")
            ]
            if len(matched) != 1:
                raise ValueError(
                    f"{token_budget} tied weight {weight:.2f}: expected one prior reference, found {len(matched)}"
                )
            row = matched.iloc[0]
            rows.append(
                {
                    "token_budget_requested": token_budget,
                    "weight": weight,
                    "starcoder_bpb": float(row["starcoder_bpb"]),
                    "source": "existing token-scaling panel",
                    "run_name": str(row["run_name"]),
                    "wandb_id": str(row["training_wandb_id"]),
                    "wandb_state": "finished",
                    "wandb_url": str(row["training_wandb_url"]),
                }
            )

    observations = pd.DataFrame(rows).sort_values(["token_budget_requested", "weight"]).reset_index(drop=True)
    counts = observations.groupby("token_budget_requested")["weight"].nunique()
    if len(observations) != 84 or len(counts) != 4 or not (counts == 21).all():
        raise ValueError(f"Expected a 4x21 complete diagonal, got {counts.to_dict()}")
    if observations.duplicated(["token_budget_requested", "weight"]).any():
        raise ValueError("Duplicate tied coordinate in completed diagonal")
    return observations


def repeat_noise(token_results_dir: Path) -> pd.DataFrame:
    prior = pd.read_csv(token_results_dir / "observations.csv")
    repeats = prior.loc[
        (prior["coordinate_index"] == 5) & prior["replicate_kind"].isin(["reference", "joint_randomness"])
    ]
    rows = []
    for token_budget, group in repeats.groupby("token_budget_requested", sort=True):
        if len(group) != 5:
            raise ValueError(f"{token_budget}: expected five tied p=0.30 repeats, found {len(group)}")
        rows.append(
            {
                "token_budget_requested": int(token_budget),
                "repeat_count": len(group),
                "repeat_mean_bpb": float(group["starcoder_bpb"].mean()),
                "repeat_sd_bpb": float(group["starcoder_bpb"].std(ddof=1)),
            }
        )
    return pd.DataFrame(rows)


def materialized_tokens_by_budget(panel_dir: Path) -> dict[int, int]:
    design = json.loads((panel_dir / "design_manifest.json").read_text(encoding="utf-8"))
    result: dict[int, int] = {}
    for row in design["runs"]:
        token_budget = int(row["token_budget_requested"])
        materialized_tokens = int(row["materialized_tokens"])
        previous = result.setdefault(token_budget, materialized_tokens)
        if previous != materialized_tokens:
            raise ValueError(f"{token_budget}: inconsistent materialized token counts")
    if set(result) != set(TOKEN_BUDGETS):
        raise ValueError(f"Missing materialized token counts: {result}")
    return result


def local_quadratic_estimates(group: pd.DataFrame, sampled_weight: float) -> list[dict[str, float | int]]:
    x = group["weight"].to_numpy(dtype=float)
    y = group["starcoder_bpb"].to_numpy(dtype=float)
    estimates = []
    for width in LOCAL_QUADRATIC_WIDTHS:
        selected = np.abs(x - sampled_weight) <= width + 1e-12
        coefficients = np.polyfit(x[selected], y[selected], 2)
        if coefficients[0] <= 0:
            continue
        weight = float(
            np.clip(
                -coefficients[1] / (2 * coefficients[0]),
                x[selected].min(),
                x[selected].max(),
            )
        )
        estimates.append(
            {
                "window_half_width": width,
                "points": int(selected.sum()),
                "estimated_weight": weight,
                "estimated_bpb": float(np.polyval(coefficients, weight)),
                "quadratic_coefficient": float(coefficients[0]),
            }
        )
    if len(estimates) != len(LOCAL_QUADRATIC_WIDTHS):
        raise ValueError("Local tied basin was not convex under every diagnostic window")
    return estimates


def summarize_optima(
    observations: pd.DataFrame,
    noise: pd.DataFrame,
    materialized_tokens: dict[int, int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    noise_by_budget = noise.set_index("token_budget_requested")["repeat_sd_bpb"].to_dict()
    optima = []
    quadratic_rows = []
    for token_budget, group in observations.groupby("token_budget_requested", sort=True):
        group = group.sort_values("weight")
        sampled = group.loc[group["starcoder_bpb"].idxmin()]
        repeat_sd = float(noise_by_budget[token_budget])
        estimates = local_quadratic_estimates(group, float(sampled["weight"]))
        estimated_weights = np.array([float(row["estimated_weight"]) for row in estimates])
        near = group.loc[group["starcoder_bpb"] <= float(sampled["starcoder_bpb"]) + repeat_sd]
        optima.append(
            {
                "token_budget_requested": int(token_budget),
                "token_budget_billion": token_budget / 1e9,
                "materialized_tokens": materialized_tokens[int(token_budget)],
                "total_parameter_tpp": materialized_tokens[int(token_budget)] / 157_499_136,
                "sampled_min_weight": float(sampled["weight"]),
                "sampled_min_bpb": float(sampled["starcoder_bpb"]),
                "repeat_sd_bpb": repeat_sd,
                "one_sd_basin_low": float(near["weight"].min()),
                "one_sd_basin_high": float(near["weight"].max()),
                "local_quadratic_median_weight": float(np.median(estimated_weights)),
                "local_quadratic_low_weight": float(estimated_weights.min()),
                "local_quadratic_high_weight": float(estimated_weights.max()),
            }
        )
        for row in estimates:
            quadratic_rows.append({"token_budget_requested": int(token_budget), **row})
    return pd.DataFrame(optima), pd.DataFrame(quadratic_rows)


def compare_sparse_two_phase(
    optima: pd.DataFrame,
    token_results_dir: Path,
) -> pd.DataFrame:
    selected = pd.read_csv(token_results_dir / "selected_optima.csv")
    prior = pd.read_csv(token_results_dir / "observations.csv")
    tied = optima.set_index("token_budget_requested")
    rows = []
    for token_budget in TOKEN_BUDGETS:
        tied_row = tied.loc[token_budget]
        best = selected.loc[
            (selected["token_budget_requested"] == token_budget) & (selected["policy_class"] == "Two phase")
        ]
        c09 = prior.loc[
            (prior["token_budget_requested"] == token_budget)
            & (prior["coordinate_index"] == 9)
            & (prior["trainer_data_seed"] == REFERENCE_SEED)
            & prior["replicate_kind"].isin(["reference", "backfill"])
        ]
        if len(best) != 1 or len(c09) != 1:
            raise ValueError(f"{token_budget}: sparse two-phase comparison is incomplete")
        for candidate_kind, candidate in [
            ("fixed c09 candidate", c09.iloc[0]),
            ("best sparse two-phase", best.iloc[0]),
        ]:
            candidate_bpb = float(candidate["starcoder_bpb"])
            rows.append(
                {
                    "token_budget_requested": token_budget,
                    "token_budget_billion": token_budget / 1e9,
                    "candidate_kind": candidate_kind,
                    "phase_0_starcoder": float(candidate["phase_0_starcoder"]),
                    "phase_1_starcoder": float(candidate["phase_1_starcoder"]),
                    "aggregate_starcoder_nominal": float(candidate["aggregate_starcoder_nominal"]),
                    "candidate_bpb": candidate_bpb,
                    "sampled_tied_min_weight": float(tied_row["sampled_min_weight"]),
                    "sampled_tied_min_bpb": float(tied_row["sampled_min_bpb"]),
                    "candidate_minus_tied_bpb": candidate_bpb - float(tied_row["sampled_min_bpb"]),
                    "delta_over_repeat_sd": (
                        (candidate_bpb - float(tied_row["sampled_min_bpb"])) / float(tied_row["repeat_sd_bpb"])
                    ),
                }
            )
    return pd.DataFrame(rows)


def style_figure(figure: go.Figure, title: str, height: int) -> None:
    figure.update_layout(
        title={"text": title, "x": 0.04, "xanchor": "left"},
        template="plotly_white",
        height=height,
        margin={"l": 80, "r": 45, "t": 115, "b": 80},
        font={"family": "Avenir Next, Helvetica Neue, sans-serif", "color": "#173042"},
        paper_bgcolor="#fbf8f0",
        plot_bgcolor="#fbf8f0",
        hoverlabel={"font_size": 13},
    )
    figure.update_xaxes(gridcolor="#ded8ca", zerolinecolor="#173042")
    figure.update_yaxes(gridcolor="#ded8ca", zerolinecolor="#173042")


def write_diagonal_plot(observations: pd.DataFrame, optima: pd.DataFrame, output_path: Path) -> None:
    figure = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=[f"{budget // 1_000_000_000}B tokens" for budget in TOKEN_BUDGETS],
        horizontal_spacing=0.10,
        vertical_spacing=0.15,
    )
    colors = sample_colorscale("RdYlGn_r", np.linspace(0.12, 0.88, 4))
    for index, (token_budget, group) in enumerate(observations.groupby("token_budget_requested", sort=True)):
        row = index // 2 + 1
        column = index % 2 + 1
        optimum = optima.loc[optima["token_budget_requested"] == token_budget].iloc[0]
        figure.add_trace(
            go.Scatter(
                x=group["weight"],
                y=group["starcoder_bpb"],
                mode="lines+markers",
                line={"color": colors[index], "width": 2.5},
                marker={"size": 8},
                name=f"{token_budget / 1e9:g}B",
                showlegend=False,
                customdata=group[["source", "run_name", "wandb_url"]].to_numpy(),
                hovertemplate=(
                    "tied weight=%{x:.2f}<br>BPB=%{y:.6f}<br>%{customdata[0]}"
                    "<br>%{customdata[1]}<br>%{customdata[2]}<extra></extra>"
                ),
            ),
            row=row,
            col=column,
        )
        figure.add_trace(
            go.Scatter(
                x=[optimum["sampled_min_weight"]],
                y=[optimum["sampled_min_bpb"]],
                mode="markers",
                marker={"size": 15, "symbol": "diamond", "color": "#173042"},
                name="Sampled minimum",
                showlegend=index == 0,
                hovertemplate="sampled min<br>weight=%{x:.2f}<br>BPB=%{y:.6f}<extra></extra>",
            ),
            row=row,
            col=column,
        )
        figure.add_vrect(
            x0=optimum["one_sd_basin_low"],
            x1=optimum["one_sd_basin_high"],
            fillcolor="#2a9d8f",
            opacity=0.10,
            line_width=0,
            row=row,
            col=column,
        )
        figure.add_vline(
            x=optimum["local_quadratic_median_weight"],
            line={"color": "#e76f51", "dash": "dash", "width": 2},
            row=row,
            col=column,
        )
        figure.update_xaxes(title="Tied StarCoder share", row=row, col=column)
        figure.update_yaxes(title="Programming Languages BPB", row=row, col=column)
    style_figure(
        figure,
        "Completed tied diagonal across the fixed-model token ladder"
        "<br><sup>Diamond: sampled minimum. Shading: sampled points within one repeat SD. "
        "Dashed line: median of three post-hoc local quadratic location estimates.</sup>",
        980,
    )
    figure.update_layout(legend={"orientation": "h", "x": 0, "y": 1.06})
    figure.write_html(output_path, include_plotlyjs=True, config=PLOT_CONFIG)


def write_optimum_path_plot(
    optima: pd.DataFrame,
    comparison: pd.DataFrame,
    output_path: Path,
) -> None:
    figure = make_subplots(
        rows=1,
        cols=2,
        column_titles=["Tied optimum location", "Sparse two-phase candidate versus resolved tied minimum"],
        horizontal_spacing=0.14,
    )
    figure.add_trace(
        go.Scatter(
            x=optima["token_budget_billion"],
            y=optima["sampled_min_weight"],
            mode="lines+markers",
            name="Sampled tied minimum",
            line={"color": "#173042", "width": 3},
            marker={"size": 11, "symbol": "diamond"},
        ),
        row=1,
        col=1,
    )
    figure.add_trace(
        go.Scatter(
            x=optima["token_budget_billion"],
            y=optima["local_quadratic_median_weight"],
            error_y={
                "type": "data",
                "symmetric": False,
                "array": optima["local_quadratic_high_weight"] - optima["local_quadratic_median_weight"],
                "arrayminus": optima["local_quadratic_median_weight"] - optima["local_quadratic_low_weight"],
            },
            mode="lines+markers",
            name="Local quadratic sensitivity",
            line={"color": "#e76f51", "width": 2.5, "dash": "dash"},
            marker={"size": 9},
        ),
        row=1,
        col=1,
    )
    colors = {"fixed c09 candidate": "#e9c46a", "best sparse two-phase": "#2a9d8f"}
    for candidate_kind, group in comparison.groupby("candidate_kind", sort=False):
        figure.add_trace(
            go.Scatter(
                x=group["token_budget_billion"],
                y=group["candidate_minus_tied_bpb"],
                mode="lines+markers",
                name=candidate_kind,
                line={"color": colors[candidate_kind], "width": 3},
                marker={"size": 10},
                customdata=group[["phase_0_starcoder", "phase_1_starcoder", "aggregate_starcoder_nominal"]].to_numpy(),
                hovertemplate=(
                    "%{fullData.name}<br>tokens=%{x:g}B<br>candidate - tied=%{y:+.6f} BPB"
                    "<br>p0=%{customdata[0]:.2f}, p1=%{customdata[1]:.2f}, aggregate=%{customdata[2]:.2f}"
                    "<extra></extra>"
                ),
            ),
            row=1,
            col=2,
        )
    figure.add_hline(y=0, line={"color": "#173042", "width": 1.5}, row=1, col=2)
    figure.update_xaxes(type="log", tickvals=[1, 2, 4, 8], title="Materialized tokens (billions)")
    figure.update_yaxes(title="Tied StarCoder share", range=[0.2, 0.85], row=1, col=1)
    figure.update_yaxes(title="Candidate minus tied minimum BPB", row=1, col=2)
    style_figure(
        figure,
        "The tied optimum moves with training duration"
        "<br><sup>Negative candidate deltas favor two phase. Sparse two-phase selection remains secondary and "
        "does not resolve the global two-phase optimum.</sup>",
        690,
    )
    figure.update_layout(legend={"orientation": "h", "x": 0, "y": 1.08})
    figure.write_html(output_path, include_plotlyjs=True, config=PLOT_CONFIG)


def report_text(
    observations: pd.DataFrame,
    optima: pd.DataFrame,
    comparison: pd.DataFrame,
) -> str:
    optimum_table = optima.assign(
        sampled_min_bpb=optima["sampled_min_bpb"].map(lambda value: f"{value:.6f}"),
        repeat_sd_bpb=optima["repeat_sd_bpb"].map(lambda value: f"{value:.6f}"),
        one_sd_basin=lambda frame: frame.apply(
            lambda row: f"[{row['one_sd_basin_low']:.2f}, {row['one_sd_basin_high']:.2f}]",
            axis=1,
        ),
        local_quadratic_range=lambda frame: frame.apply(
            lambda row: (
                f"{row['local_quadratic_median_weight']:.3f} "
                f"[{row['local_quadratic_low_weight']:.3f}, {row['local_quadratic_high_weight']:.3f}]"
            ),
            axis=1,
        ),
    )[
        [
            "token_budget_billion",
            "total_parameter_tpp",
            "sampled_min_weight",
            "sampled_min_bpb",
            "repeat_sd_bpb",
            "one_sd_basin",
            "local_quadratic_range",
        ]
    ]
    comparison_table = comparison.assign(
        candidate_bpb=comparison["candidate_bpb"].map(lambda value: f"{value:.6f}"),
        sampled_tied_min_bpb=comparison["sampled_tied_min_bpb"].map(lambda value: f"{value:.6f}"),
        candidate_minus_tied_bpb=comparison["candidate_minus_tied_bpb"].map(lambda value: f"{value:+.6f}"),
        delta_over_repeat_sd=comparison["delta_over_repeat_sd"].map(lambda value: f"{value:+.2f}"),
    )[
        [
            "token_budget_billion",
            "candidate_kind",
            "phase_0_starcoder",
            "phase_1_starcoder",
            "aggregate_starcoder_nominal",
            "candidate_bpb",
            "sampled_tied_min_bpb",
            "candidate_minus_tied_bpb",
            "delta_over_repeat_sd",
        ]
    ]
    best_sparse = comparison.loc[comparison["candidate_kind"] == "best sparse two-phase"].sort_values(
        "token_budget_billion"
    )
    fixed_c09 = comparison.loc[comparison["candidate_kind"] == "fixed c09 candidate"].sort_values("token_budget_billion")
    crashed = observations.loc[observations["wandb_state"] == "crashed", "run_name"].tolist()
    return (
        "\n".join(
            [
                "# StarCoder WSD80 fixed-model tied-diagonal results",
                "",
                "## Completion",
                "",
                "- Iris parent: succeeded with exit 0, zero failures, and zero preemptions.",
                "- New tied checkpoints: 60/60 with finite final Programming Languages BPB.",
                "- Joined regular diagonal: 21 tied weights at each of 1B, 2B, 4B, and 8B (84 rows).",
                (
                    "- W&B still labels two recovered 8B runs `crashed`, but both contain finite final evaluation "
                    f"summaries and the Iris artifact graph completed: {', '.join(crashed)}."
                    if crashed
                    else "- All joined W&B runs are terminal and finished."
                ),
                "",
                "## Tied optimum",
                "",
                optimum_table.to_markdown(index=False),
                "",
                "The sampled minimum is primary. The one-SD basin uses the five-seed repeat SD measured at tied "
                "`p=0.30` on the same rung. The local quadratic location is a post-hoc sensitivity analysis over "
                "three windows, not a new measured checkpoint.",
                "",
                "## Comparison with the prior sparse two-phase panel",
                "",
                comparison_table.to_markdown(index=False),
                "",
                "## Main read",
                "",
                "- The tied optimum is not scale-invariant. It moves from a StarCoder share near 0.30 at 1B to "
                "0.35--0.40 at 2B, about 0.53--0.55 at 4B, and a broad 0.65--0.80 basin at 8B.",
                "- The earlier conclusion that the tied optimum stayed at 0.35 after 2B was boundary censoring from "
                "the sparse panel. The completed diagonal resolves that error.",
                f"- The best observed sparse two-phase advantage over the resolved tied minimum changes from "
                f"{best_sparse.iloc[0]['candidate_minus_tied_bpb']:+.6f} BPB at 1B to "
                f"{best_sparse.iloc[-1]['candidate_minus_tied_bpb']:+.6f} BPB at 8B. At 8B the gap is only "
                f"{abs(best_sparse.iloc[-1]['delta_over_repeat_sd']):.2f} times the tied-repeat SD, so it is not "
                "a resolved global two-phase advantage.",
                f"- The fixed c09 policy changes from {fixed_c09.iloc[0]['candidate_minus_tied_bpb']:+.6f} BPB "
                f"relative to the resolved tied minimum at 1B to "
                f"{fixed_c09.iloc[-1]['candidate_minus_tied_bpb']:+.6f} BPB at 8B. Its growing phase gain at "
                "aggregate 0.18 does not compensate for the tied aggregate optimum moving upward.",
                "- This does not show that the true two-phase advantage vanishes with tokens. The sparse two-phase "
                "panel was concentrated around aggregate 0.18 and did not track the moving tied optimum. The global "
                "two-phase optimum remains under-resolved.",
                "",
                "## Planned follow-up",
                "",
                "The preregistered next stage is now unblocked: test matched antithetic phase contrasts on the exact "
                "fixed-aggregate fibers of the scale-specific tied basins. The follow-up should use matched seeds and "
                "retain tied controls, with the fiber estimand reported separately from global policy selection.",
                "",
                "Recommended anchors for design review are the measured grid minima `0.30`, `0.35`, `0.55`, and "
                "`0.80` at 1B, 2B, 4B, and 8B. The 2B and 8B basins are broad, while the smooth location diagnostics "
                "cluster near `0.38--0.39` and `0.72--0.74`. Include preregistered `0.40` and `0.75` sensitivity "
                "anchors, or repeat the neighboring tied controls, before interpreting a fiber-optimality result.",
                "",
                "No follow-up training job was submitted by this analysis.",
            ]
        )
        + "\n"
    )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    new = collect_new_observations(args.panel_dir, args.wandb_timeout)
    observations = collect_existing_observations(new, args.token_results_dir, args.source_surface_dir)
    noise = repeat_noise(args.token_results_dir)
    materialized_tokens = materialized_tokens_by_budget(args.panel_dir)
    optima, quadratic = summarize_optima(observations, noise, materialized_tokens)
    comparison = compare_sparse_two_phase(optima, args.token_results_dir)

    observations.to_csv(args.output_dir / "tied_diagonal_observations.csv", index=False)
    noise.to_csv(args.output_dir / "repeat_noise.csv", index=False)
    optima.to_csv(args.output_dir / "tied_optima.csv", index=False)
    quadratic.to_csv(args.output_dir / "local_quadratic_sensitivity.csv", index=False)
    comparison.to_csv(args.output_dir / "sparse_two_phase_comparison.csv", index=False)
    write_diagonal_plot(observations, optima, args.output_dir / "tied_diagonal_curves.html")
    write_optimum_path_plot(optima, comparison, args.output_dir / "tied_optimum_path.html")
    (args.output_dir / "report.md").write_text(
        report_text(observations, optima, comparison),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
