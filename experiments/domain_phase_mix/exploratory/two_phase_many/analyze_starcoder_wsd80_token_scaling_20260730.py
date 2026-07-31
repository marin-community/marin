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
"""Analyze the completed 1B--8B StarCoder WSD80 fixed-model token ladder."""

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
from scipy import stats

SCRIPT_DIR = Path(__file__).resolve().parent
REFERENCE_OUTPUTS = SCRIPT_DIR / "reference_outputs"
PANEL_DIR = REFERENCE_OUTPUTS / "starcoder_wsd80_fixed_model_token_scaling_20260728"
SOURCE_SURFACE_DIR = REFERENCE_OUTPUTS / "starcoder_wsd80_surface_refined_20260714"
DEFAULT_OUTPUT_DIR = PANEL_DIR / "results_20260730"

TRAIN_PROJECT = "marin-community/marin"
TRAIN_TAG = "starcoder_wsd80_fixedn_tokenscale"
TARGET_METRIC = "eval/paloma/dolma_100_programing_languages-llama3/bpb"
EXPECTED_NEW_RUNS = 104
EXPECTED_ANALYSIS_ROWS = 123
REFERENCE_SEED = 20260711
MATCHED_COORDINATES = (2, 5, 9)
PLOT_CONFIG = {"toImageButtonOptions": {"format": "png", "scale": 4}}
ESTIMAND_LABELS = {
    "phase_gain_c09_minus_c02": "Phase gain: c09 - c02",
    "aggregate_penalty_c02_minus_c05": "Aggregate penalty: c02 - c05",
    "net_advantage_c09_minus_c05": "Net 2p advantage: c09 - c05",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel-dir", type=Path, default=PANEL_DIR)
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


def selected_new_run(runs: list[Any], run_name: str) -> Any:
    candidates = []
    for run in runs:
        if str(run.name) != run_name and not str(run.name).startswith(f"{run_name}_recovery"):
            continue
        if finite_summary(run, TARGET_METRIC) is not None:
            candidates.append(run)
    if len(candidates) != 1:
        raise ValueError(f"{run_name}: expected one finite W&B run, found {len(candidates)}")
    return candidates[0]


def source_row(
    base: dict[str, object],
    *,
    coordinate: dict[str, object],
    seed: int,
    bpb: float,
    wandb_run_id: str,
    wandb_run_name: str,
    wandb_url: str,
    replicate_kind: str,
    evidence_source: str,
) -> dict[str, object]:
    return {
        **base,
        "run_name": wandb_run_name,
        "coordinate_index": int(coordinate["index"]),
        "coordinate_role": coordinate["role"],
        "phase_0_starcoder": float(coordinate["phase_0_starcoder"]),
        "phase_1_starcoder": float(coordinate["phase_1_starcoder"]),
        "aggregate_starcoder_nominal": float(coordinate["aggregate_starcoder_nominal"]),
        "phase_contrast": float(coordinate["phase_contrast"]),
        "replicate_kind": replicate_kind,
        "trainer_data_seed": seed,
        "starcoder_bpb": bpb,
        "training_wandb_id": wandb_run_id,
        "training_wandb_url": wandb_url,
        "evidence_source": evidence_source,
    }


def collect_observations(panel_dir: Path, source_dir: Path, timeout: int) -> pd.DataFrame:
    manifest = pd.read_csv(panel_dir / "run_manifest.csv")
    design = json.loads((panel_dir / "design_manifest.json").read_text(encoding="utf-8"))
    if len(manifest) != EXPECTED_NEW_RUNS or manifest["run_name"].duplicated().any():
        raise ValueError(f"Expected {EXPECTED_NEW_RUNS} unique new-run manifest rows")

    api = wandb.Api(timeout=timeout)
    runs = list(api.runs(TRAIN_PROJECT, filters={"tags": TRAIN_TAG}, per_page=180))
    new_rows = []
    for _, spec in manifest.iterrows():
        run = selected_new_run(runs, str(spec["run_name"]))
        new_rows.append(
            {
                **spec.to_dict(),
                "starcoder_bpb": finite_summary(run, TARGET_METRIC),
                "training_wandb_id": run.id,
                "training_wandb_url": run.url,
                "evidence_source": "new token-scaling panel",
            }
        )

    one_billion_template = manifest.loc[manifest["token_budget_requested"] == 1_000_000_000].iloc[0].to_dict()
    for key in [
        "run_name",
        "coordinate_index",
        "coordinate_role",
        "phase_0_starcoder",
        "phase_1_starcoder",
        "aggregate_starcoder_nominal",
        "aggregate_starcoder_realized",
        "phase_contrast",
        "replicate_kind",
        "trainer_data_seed",
        "starcoder_bpb",
        "training_wandb_id",
        "training_wandb_url",
        "evidence_source",
    ]:
        one_billion_template.pop(key, None)

    surface = pd.read_csv(source_dir / "wsd80_observed_metrics.csv")
    coordinates = {int(row["index"]): row for row in design["coordinates"]}
    reused_reference_rows = []
    for coordinate_index in design["design"]["reused_1b_reference_coordinate_indices"]:
        coordinate = coordinates[int(coordinate_index)]
        matched = surface.loc[(surface["phase_0_starcoder"] - float(coordinate["phase_0_starcoder"])).abs() < 0.005,]
        matched = matched.loc[(matched["phase_1_starcoder"] - float(coordinate["phase_1_starcoder"])).abs() < 0.005]
        if len(matched) != 1:
            raise ValueError(f"1B coordinate {coordinate_index}: expected one source row, found {len(matched)}")
        row = matched.iloc[0]
        reused_reference_rows.append(
            source_row(
                one_billion_template,
                coordinate=coordinate,
                seed=REFERENCE_SEED,
                bpb=float(row["wsd80_bpb"]),
                wandb_run_id=str(row["wandb_run_id"]),
                wandb_run_name=str(row["wandb_run_name"]),
                wandb_url=str(row["wandb_url"]),
                replicate_kind="reference",
                evidence_source="reused 1B reference",
            )
        )

    measured = pd.read_csv(source_dir / "wsd80_measured_fiber_observations.csv")
    reused_repeat_rows = []
    for coordinate_index in MATCHED_COORDINATES:
        coordinate = coordinates[coordinate_index]
        matched = measured.loc[(measured["phase_0_starcoder"] - float(coordinate["phase_0_starcoder"])).abs() < 0.005,]
        matched = matched.loc[(matched["phase_1_starcoder"] - float(coordinate["phase_1_starcoder"])).abs() < 0.005]
        matched = matched.loc[matched["data_seed"].isin(range(20260712, 20260716))]
        if len(matched) != 4:
            raise ValueError(f"1B coordinate {coordinate_index}: expected four reused repeats, found {len(matched)}")
        for _, row in matched.iterrows():
            reused_repeat_rows.append(
                source_row(
                    one_billion_template,
                    coordinate=coordinate,
                    seed=int(row["data_seed"]),
                    bpb=float(row["wsd80_bpb"]),
                    wandb_run_id=str(row["wandb_run_id"]),
                    wandb_run_name=str(row["wandb_run_name"]),
                    wandb_url=str(row["wandb_url"]),
                    replicate_kind="joint_randomness",
                    evidence_source="reused 1B matched repeat",
                )
            )

    observations = pd.concat(
        [
            pd.DataFrame(new_rows),
            pd.DataFrame(reused_reference_rows),
            pd.DataFrame(reused_repeat_rows),
        ],
        ignore_index=True,
        sort=False,
    )
    if len(observations) != EXPECTED_ANALYSIS_ROWS or observations["starcoder_bpb"].isna().any():
        raise ValueError(f"Expected {EXPECTED_ANALYSIS_ROWS} finite analysis rows")

    reference = observations.loc[observations["replicate_kind"].isin(["reference", "backfill"])]
    counts = reference.groupby("token_budget_requested")["coordinate_index"].nunique()
    if not (counts == 18).all() or len(counts) != 4:
        raise ValueError(f"Reference coordinate coverage is not 18 per rung: {counts.to_dict()}")
    return observations


def matched_estimands(observations: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    repeat_rows = observations.loc[
        observations["coordinate_index"].isin(MATCHED_COORDINATES)
        & observations["replicate_kind"].isin(["reference", "backfill", "joint_randomness"])
    ]
    seed_rows = []
    summaries = []
    formulas = {
        "phase_gain_c09_minus_c02": (9, 2),
        "aggregate_penalty_c02_minus_c05": (2, 5),
        "net_advantage_c09_minus_c05": (9, 5),
    }
    for budget, group in repeat_rows.groupby("token_budget_requested", sort=True):
        pivot = group.pivot(
            index="trainer_data_seed",
            columns="coordinate_index",
            values="starcoder_bpb",
        )
        if set(pivot.columns) != set(MATCHED_COORDINATES) or len(pivot) != 5:
            raise ValueError(f"{budget}: matched repeat matrix is not 5x3")
        for estimand, (left, right) in formulas.items():
            values = (pivot[left] - pivot[right]).to_numpy(dtype=float)
            sem = float(stats.sem(values))
            interval = stats.t.interval(0.95, len(values) - 1, loc=float(values.mean()), scale=sem)
            summaries.append(
                {
                    "token_budget_requested": int(budget),
                    "token_budget_billion": float(budget / 1e9),
                    "estimand": estimand,
                    "label": ESTIMAND_LABELS[estimand],
                    "n_pairs": len(values),
                    "mean_delta_bpb": float(values.mean()),
                    "sd_delta_bpb": float(values.std(ddof=1)),
                    "ci95_low": float(interval[0]),
                    "ci95_high": float(interval[1]),
                    "negative_pairs": int((values < 0).sum()),
                    "paired_t_p": float(stats.ttest_1samp(values, 0).pvalue),
                }
            )
            for seed, value in zip(pivot.index, values, strict=True):
                seed_rows.append(
                    {
                        "token_budget_requested": int(budget),
                        "token_budget_billion": float(budget / 1e9),
                        "estimand": estimand,
                        "trainer_data_seed": int(seed),
                        "delta_bpb": float(value),
                    }
                )
    return pd.DataFrame(summaries), pd.DataFrame(seed_rows)


def reference_summary(observations: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    reference = observations.loc[observations["replicate_kind"].isin(["reference", "backfill"])].copy()
    reference["policy_class"] = np.where(reference["phase_contrast"].abs() < 1e-12, "Tied", "Two phase")
    best_rows = []
    for (budget, policy_class), group in reference.groupby(["token_budget_requested", "policy_class"], sort=True):
        best = group.loc[group["starcoder_bpb"].idxmin()]
        best_rows.append(
            {
                "token_budget_requested": int(budget),
                "token_budget_billion": float(budget / 1e9),
                "policy_class": policy_class,
                "coordinate_index": int(best["coordinate_index"]),
                "coordinate_role": best["coordinate_role"],
                "phase_0_starcoder": float(best["phase_0_starcoder"]),
                "phase_1_starcoder": float(best["phase_1_starcoder"]),
                "aggregate_starcoder_nominal": float(best["aggregate_starcoder_nominal"]),
                "phase_contrast": float(best["phase_contrast"]),
                "starcoder_bpb": float(best["starcoder_bpb"]),
            }
        )
    return reference, pd.DataFrame(best_rows)


def extension_gate(
    observations: pd.DataFrame,
    reference: pd.DataFrame,
) -> pd.DataFrame:
    budget = 8_000_000_000
    reference_8b = reference.loc[reference["token_budget_requested"] == budget]
    spread = float(reference_8b["starcoder_bpb"].max() - reference_8b["starcoder_bpb"].min())
    repeats = observations.loc[
        (observations["token_budget_requested"] == budget)
        & observations["coordinate_index"].isin(MATCHED_COORDINATES)
        & observations["replicate_kind"].isin(["reference", "joint_randomness"])
    ]
    coordinate_sd = repeats.groupby("coordinate_index")["starcoder_bpb"].std(ddof=1)
    pooled_sd = float(np.sqrt(np.mean(np.square(coordinate_sd))))
    max_sd = float(coordinate_sd.max())
    return pd.DataFrame(
        [
            {
                "token_budget_requested": budget,
                "reference_coordinate_spread_bpb": spread,
                "matched_repeat_pooled_sd_bpb": pooled_sd,
                "matched_repeat_max_coordinate_sd_bpb": max_sd,
                "spread_over_pooled_sd": spread / pooled_sd,
                "spread_over_max_sd": spread / max_sd,
                "preregistered_16b_gate_pass": bool(spread >= 3 * pooled_sd),
                "conservative_max_sd_gate_pass": bool(spread >= 3 * max_sd),
            }
        ]
    )


def style_figure(figure: go.Figure, title: str, height: int = 700) -> None:
    figure.update_layout(
        title={"text": title, "x": 0.04, "xanchor": "left"},
        template="plotly_white",
        height=height,
        margin={"l": 80, "r": 45, "t": 115, "b": 80},
        font={"family": "Avenir Next, Helvetica Neue, sans-serif", "color": "#173042"},
        paper_bgcolor="#fbf8f0",
        plot_bgcolor="#fbf8f0",
        legend={"orientation": "h", "y": 1.08, "x": 0},
        hoverlabel={"font_size": 13},
    )
    figure.update_xaxes(gridcolor="#ded8ca", zerolinecolor="#173042")
    figure.update_yaxes(gridcolor="#ded8ca", zerolinecolor="#173042")


def write_estimand_plot(estimands: pd.DataFrame, output_path: Path) -> None:
    colors = sample_colorscale("RdYlGn_r", [0.12, 0.5, 0.88])
    figure = go.Figure()
    for color, estimand in zip(colors, ESTIMAND_LABELS, strict=True):
        group = estimands.loc[estimands["estimand"] == estimand].sort_values("token_budget_billion")
        figure.add_trace(
            go.Scatter(
                x=group["token_budget_billion"],
                y=group["mean_delta_bpb"],
                error_y={
                    "type": "data",
                    "symmetric": False,
                    "array": group["ci95_high"] - group["mean_delta_bpb"],
                    "arrayminus": group["mean_delta_bpb"] - group["ci95_low"],
                },
                mode="lines+markers",
                name=ESTIMAND_LABELS[estimand],
                line={"color": color, "width": 3},
                marker={"size": 11},
                customdata=np.column_stack([group["negative_pairs"], group["n_pairs"], group["sd_delta_bpb"]]),
                hovertemplate=(
                    "<b>%{fullData.name}</b><br>tokens=%{x:g}B<br>mean Δ=%{y:+.6f} BPB"
                    "<br>negative pairs=%{customdata[0]:.0f}/%{customdata[1]:.0f}"
                    "<br>paired SD=%{customdata[2]:.6f}<extra></extra>"
                ),
            )
        )
    figure.add_hline(y=0, line={"color": "#173042", "width": 2})
    figure.update_xaxes(type="log", tickvals=[1, 2, 4, 8], title="Materialized training tokens (billions)")
    figure.update_yaxes(title="Matched Δ StarCoder BPB (lower is better)")
    style_figure(
        figure,
        "Phase and aggregate effects as token budget grows"
        "<br><sup>Five matched joint-randomness seeds per rung; bars are 95% paired t intervals.</sup>",
        height=690,
    )
    figure.write_html(output_path, include_plotlyjs=True, config=PLOT_CONFIG)


def write_coordinate_plot(reference: pd.DataFrame, output_path: Path) -> None:
    colors = sample_colorscale("RdYlGn_r", np.linspace(0.05, 0.95, 18))
    figure = go.Figure()
    for coordinate_index in range(1, 19):
        group = reference.loc[reference["coordinate_index"] == coordinate_index].sort_values("token_budget_requested")
        first = group.iloc[0]
        tied = abs(float(first["phase_contrast"])) < 1e-12
        figure.add_trace(
            go.Scatter(
                x=group["token_budget_requested"] / 1e9,
                y=group["starcoder_bpb"],
                mode="lines+markers",
                name=f"c{coordinate_index:02d} · {first['coordinate_role']}",
                line={
                    "color": colors[coordinate_index - 1],
                    "width": 2.5 if tied else 1.8,
                    "dash": "solid" if tied else "dot",
                },
                marker={"size": 8, "symbol": "circle" if tied else "diamond"},
                customdata=group[
                    [
                        "phase_0_starcoder",
                        "phase_1_starcoder",
                        "aggregate_starcoder_nominal",
                        "phase_contrast",
                    ]
                ].to_numpy(),
                hovertemplate=(
                    "<b>%{fullData.name}</b><br>tokens=%{x:g}B<br>BPB=%{y:.6f}"
                    "<br>p0=%{customdata[0]:.2f}, p1=%{customdata[1]:.2f}"
                    "<br>aggregate=%{customdata[2]:.2f}, contrast=%{customdata[3]:.2f}<extra></extra>"
                ),
            )
        )
    figure.update_xaxes(type="log", tickvals=[1, 2, 4, 8], title="Materialized training tokens (billions)")
    figure.update_yaxes(title="StarCoder BPB")
    style_figure(
        figure,
        "Coordinate-wise scaling under fixed model size"
        "<br><sup>Solid circles are tied; dotted diamonds are two phase. Simulated target exposure is fixed.</sup>",
        height=820,
    )
    figure.update_layout(legend={"orientation": "v", "x": 1.01, "y": 1})
    figure.write_html(output_path, include_plotlyjs=True, config=PLOT_CONFIG)


def write_optimum_plot(best: pd.DataFrame, output_path: Path) -> None:
    figure = make_subplots(
        rows=1,
        cols=2,
        column_titles=["Best observed BPB", "Location of selected policy"],
        horizontal_spacing=0.12,
    )
    palette = {"Tied": "#e76f51", "Two phase": "#2a9d8f"}
    for policy_class in ["Tied", "Two phase"]:
        group = best.loc[best["policy_class"] == policy_class].sort_values("token_budget_billion")
        figure.add_trace(
            go.Scatter(
                x=group["token_budget_billion"],
                y=group["starcoder_bpb"],
                mode="lines+markers",
                name=policy_class,
                legendgroup=policy_class,
                line={"color": palette[policy_class], "width": 3},
                marker={"size": 11},
                customdata=group[["coordinate_index", "phase_0_starcoder", "phase_1_starcoder"]].to_numpy(),
                hovertemplate=(
                    "<b>%{fullData.name}</b><br>tokens=%{x:g}B<br>BPB=%{y:.6f}"
                    "<br>c%{customdata[0]:02.0f}: p0=%{customdata[1]:.2f}, p1=%{customdata[2]:.2f}"
                    "<extra></extra>"
                ),
            ),
            row=1,
            col=1,
        )
    two_phase = best.loc[best["policy_class"] == "Two phase"].sort_values("token_budget_billion")
    for column, label, dash in [
        ("phase_0_starcoder", "Selected phase 0", "solid"),
        ("phase_1_starcoder", "Selected phase 1", "dash"),
        ("aggregate_starcoder_nominal", "Selected aggregate", "dot"),
    ]:
        figure.add_trace(
            go.Scatter(
                x=two_phase["token_budget_billion"],
                y=two_phase[column],
                mode="lines+markers",
                name=label,
                line={"width": 3, "dash": dash},
                marker={"size": 10},
                showlegend=True,
                hovertemplate="%{fullData.name}<br>tokens=%{x:g}B<br>share=%{y:.3f}<extra></extra>",
            ),
            row=1,
            col=2,
        )
    figure.update_xaxes(type="log", tickvals=[1, 2, 4, 8], title="Tokens (billions)")
    figure.update_yaxes(title="StarCoder BPB", row=1, col=1)
    figure.update_yaxes(title="StarCoder mixture share", range=[0, 0.9], row=1, col=2)
    style_figure(
        figure,
        "Observed optimum path"
        "<br><sup>Secondary selection-biased summary over the 18 preregistered coordinates per rung.</sup>",
        height=680,
    )
    figure.write_html(output_path, include_plotlyjs=True, config=PLOT_CONFIG)


def write_eight_billion_slice(reference: pd.DataFrame, output_path: Path) -> None:
    panel = reference.loc[reference["token_budget_requested"] == 8_000_000_000]
    figure = go.Figure(
        go.Scatter(
            x=panel["phase_0_starcoder"],
            y=panel["phase_1_starcoder"],
            mode="markers+text",
            text=[f"c{index:02d}" for index in panel["coordinate_index"]],
            textposition="top center",
            marker={
                "size": 18,
                "color": panel["starcoder_bpb"],
                "colorscale": "RdYlGn_r",
                "colorbar": {"title": "BPB"},
                "line": {"color": "#173042", "width": 1},
            },
            customdata=panel[
                ["coordinate_role", "aggregate_starcoder_nominal", "phase_contrast", "starcoder_bpb"]
            ].to_numpy(),
            hovertemplate=(
                "<b>%{text} · %{customdata[0]}</b><br>p0=%{x:.2f}, p1=%{y:.2f}"
                "<br>aggregate=%{customdata[1]:.2f}<br>contrast=%{customdata[2]:.2f}"
                "<br>BPB=%{customdata[3]:.6f}<extra></extra>"
            ),
        )
    )
    figure.add_shape(
        type="line",
        x0=0,
        y0=0,
        x1=0.9,
        y1=0.9,
        line={"color": "#8c8c8c", "dash": "dash"},
    )
    figure.update_xaxes(title="Phase 0 StarCoder share", range=[-0.03, 0.4])
    figure.update_yaxes(title="Phase 1 StarCoder share", range=[-0.04, 0.88])
    style_figure(
        figure,
        "8B WSD80 policy slice" "<br><sup>Lower BPB is greener. The dashed diagonal is the tied policy class.</sup>",
        height=720,
    )
    figure.write_html(output_path, include_plotlyjs=True, config=PLOT_CONFIG)


def report_text(
    observations: pd.DataFrame,
    estimands: pd.DataFrame,
    best: pd.DataFrame,
    gate: pd.DataFrame,
) -> str:
    gate_row = gate.iloc[0]
    table = estimands.assign(
        mean_delta_bpb=estimands["mean_delta_bpb"].map(lambda value: f"{value:+.6f}"),
        ci95=lambda frame: frame.apply(lambda row: f"[{row['ci95_low']:+.6f}, {row['ci95_high']:+.6f}]", axis=1),
        paired_t_p=estimands["paired_t_p"].map(lambda value: f"{value:.4g}"),
    )[
        [
            "token_budget_billion",
            "label",
            "mean_delta_bpb",
            "ci95",
            "negative_pairs",
            "paired_t_p",
        ]
    ]
    best_table = best.assign(
        starcoder_bpb=best["starcoder_bpb"].map(lambda value: f"{value:.6f}"),
    )[
        [
            "token_budget_billion",
            "policy_class",
            "coordinate_index",
            "phase_0_starcoder",
            "phase_1_starcoder",
            "starcoder_bpb",
        ]
    ]
    net = estimands.loc[estimands["estimand"] == "net_advantage_c09_minus_c05"].sort_values("token_budget_billion")
    phase = estimands.loc[estimands["estimand"] == "phase_gain_c09_minus_c02"].sort_values("token_budget_billion")
    return (
        "\n".join(
            [
                "# StarCoder WSD80 fixed-model token-scaling results",
                "",
                "## Completion",
                "",
                "- The 1B--8B parent and targeted recovery are terminal and successful.",
                "- New coordinates: 104/104 with finite persisted checkpoints and StarCoder BPB.",
                "- Analysis evidence: 123 rows after adding 7 preregistered reused 1B references and 12 reused "
                "matched-repeat observations.",
                "",
                "## Preregistered matched estimands",
                "",
                "All deltas are left minus right; negative values favor the first policy.",
                "",
                table.to_markdown(index=False),
                "",
                "## Selected coordinates",
                "",
                "These minima are secondary and winner-biased.",
                "",
                best_table.to_markdown(index=False),
                "",
                "## Main read",
                "",
                f"- The matched c09 phase contrast changes from {phase.iloc[0]['mean_delta_bpb']:+.6f} BPB "
                f"at 1B to {phase.iloc[-1]['mean_delta_bpb']:+.6f} BPB at 8B relative to its tied aggregate.",
                f"- Its net advantage over the tied c05 policy changes from {net.iloc[0]['mean_delta_bpb']:+.6f} "
                f"BPB at 1B to {net.iloc[-1]['mean_delta_bpb']:+.6f} BPB at 8B.",
                "- The selected reference-seed optimum moves from (0.10, 0.50) at 1B to (0.02, 0.82) at 8B, while "
                "the selected tied optimum moves only from 0.30 to 0.35. These minima are winner-biased, but their "
                "direction agrees with the preregistered paired effects.",
                "- This is a fixed-model, fixed-target-exposure trend. It does not by itself identify a universal "
                "token or model-size scaling law.",
                "",
                "## Conditional 16B gate",
                "",
                f"- 8B reference-coordinate spread: {gate_row['reference_coordinate_spread_bpb']:.6f} BPB.",
                f"- Pooled matched-repeat SD: {gate_row['matched_repeat_pooled_sd_bpb']:.6f} BPB; "
                f"spread/SD = {gate_row['spread_over_pooled_sd']:.2f}.",
                f"- Largest coordinate-specific repeat SD: "
                f"{gate_row['matched_repeat_max_coordinate_sd_bpb']:.6f} BPB; "
                f"spread/max-SD = {gate_row['spread_over_max_sd']:.2f}.",
                f"- Preregistered 16B extension gate: "
                f"{'PASS' if gate_row['preregistered_16b_gate_pass'] else 'DO NOT EXTEND'}.",
                "- The gate only establishes that the response remains resolvable relative to repeat noise. Its large "
                "margin is partly driven by deliberately poor coordinates in the reference grid, so it does not by "
                "itself establish that a 16B rung is the highest-value next experiment.",
                "",
                "## Artifacts",
                "",
                "- `observations.csv`: all new and reused checkpoint-level observations.",
                "- `matched_estimands.csv` and `matched_seed_deltas.csv`: paired primary effects.",
                "- `reference_coordinates.csv`, `selected_optima.csv`, and `extension_gate.csv`.",
                "- `matched_estimands.html`, `coordinate_curves.html`, `optimum_path.html`, and "
                "`eight_billion_policy_slice.html`.",
                "",
                f"Analysis rows: {len(observations)}.",
            ]
        )
        + "\n"
    )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    observations = collect_observations(args.panel_dir, args.source_surface_dir, args.wandb_timeout)
    estimands, seed_deltas = matched_estimands(observations)
    reference, best = reference_summary(observations)
    gate = extension_gate(observations, reference)

    observations.to_csv(args.output_dir / "observations.csv", index=False)
    estimands.to_csv(args.output_dir / "matched_estimands.csv", index=False)
    seed_deltas.to_csv(args.output_dir / "matched_seed_deltas.csv", index=False)
    reference.to_csv(args.output_dir / "reference_coordinates.csv", index=False)
    best.to_csv(args.output_dir / "selected_optima.csv", index=False)
    gate.to_csv(args.output_dir / "extension_gate.csv", index=False)
    write_estimand_plot(estimands, args.output_dir / "matched_estimands.html")
    write_coordinate_plot(reference, args.output_dir / "coordinate_curves.html")
    write_optimum_plot(best, args.output_dir / "optimum_path.html")
    write_eight_billion_slice(reference, args.output_dir / "eight_billion_policy_slice.html")
    (args.output_dir / "report.md").write_text(
        report_text(observations, estimands, best, gate),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
