# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "kaleido==0.2.1",
#   "numpy",
#   "pandas",
#   "plotly",
#   "scipy",
#   "wandb",
# ]
# ///
"""Render the fixed-model StarCoder 80/20 WSD surfaces across token budgets.

The 1B rung uses the canonical dense 346-coordinate surface. The 2B, 4B, and
8B rungs combine the preregistered token-scaling scaffold, the completed tied
diagonal, the scale-specific fixed-aggregate fibers, and two rounds of local
Bayesian refinement. The 2B rung includes the completed near-boundary grids
through both tied-optimum anchors. Only the reference joint-randomness seed
enters each Delaunay triangulation; repeated seeds are displayed as means with
one-standard-deviation error bars and do not reweight the fitted surface.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.domain_phase_mix.exploratory.two_phase_many.analyze_starcoder_wsd_80_20_surface import (  # noqa: E402
    BOUNDARY_MIN_COLOR,
    COLOR_SCALE,
    DIAGONAL_MIN_COLOR,
    EXPORT_CONFIG,
    FIBER_COLORS,
    GLOBAL_MIN_COLOR,
    PAPER_BACKGROUND,
    PAPER_TEXT,
    PHASE_0_FRACTION,
    PROPORTIONAL_COLOR,
    PROPORTIONAL_STARCODER,
    REFERENCE_DATA_SEED,
    SERIF_FONT,
    _add_fact_sheet,
    _scene_layout,
    _triangle_indices,
)

SCRIPT_DIR = Path(__file__).resolve().parent
REFERENCE_OUTPUTS = SCRIPT_DIR / "reference_outputs"
DEFAULT_OUTPUT_DIR = REFERENCE_OUTPUTS / "starcoder_wsd80_token_budget_surfaces_20260731"

DENSE_SURFACE_PATH = REFERENCE_OUTPUTS / "starcoder_wsd80_surface_refined_20260714" / "wsd80_observed_metrics.csv"
DENSE_FIBERS_PATH = (
    REFERENCE_OUTPUTS / "starcoder_wsd80_surface_refined_20260714" / "wsd80_measured_fiber_observations.csv"
)
TOKEN_SCALING_PATH = (
    REFERENCE_OUTPUTS / "starcoder_wsd80_fixed_model_token_scaling_20260728" / "results_20260730" / "observations.csv"
)
TIED_DIAGONAL_PATH = (
    REFERENCE_OUTPUTS
    / "starcoder_wsd80_fixed_model_tied_diagonal_20260730"
    / "results_20260731"
    / "tied_diagonal_observations.csv"
)
SCALE_FIBERS_PATH = (
    REFERENCE_OUTPUTS
    / "starcoder_wsd80_2b_complete_fibers_20260731"
    / "results_20260731"
    / "merged_scale_fiber_observations.csv"
)
BAYESIAN_REFINEMENT_DIR = REFERENCE_OUTPUTS / "starcoder_wsd80_scale_bayesian_refinement_20260731"
BAYESIAN_STAGE_1_PATH = BAYESIAN_REFINEMENT_DIR / "results_20260801" / "stage1_observations.csv"
BAYESIAN_STAGE_2_PATH = BAYESIAN_REFINEMENT_DIR / "stage2_results_20260801" / "stage2_observations.csv"

BUDGETS = (1_000_000_000, 2_000_000_000, 4_000_000_000, 8_000_000_000)
BUDGET_LABELS = {
    1_000_000_000: "1B",
    2_000_000_000: "2B",
    4_000_000_000: "4B",
    8_000_000_000: "8B",
}
TOTAL_PARAMETER_TPP = {
    1_000_000_000: 6.348081,
    2_000_000_000: 12.697755,
    4_000_000_000: 25.395511,
    8_000_000_000: 50.791984,
}
NON_EMBEDDING_TPP = {
    1_000_000_000: 16.946477,
    2_000_000_000: 33.892955,
    4_000_000_000: 67.785910,
    8_000_000_000: 135.575237,
}
EXPECTED_COORDINATE_COUNTS = {
    1_000_000_000: 352,
    2_000_000_000: 103,
    4_000_000_000: 62,
    8_000_000_000: 66,
}
METRIC_LABEL = "Paloma · Dolma 100 Programming Languages BPB"
SOURCE_PRIORITY = {
    "dense 1B WSD80 panel": 0,
    "tied diagonal": 0,
    "scale-specific fiber": 1,
    "token-scaling scaffold": 2,
    "Bayesian refinement Stage 1": 3,
    "Bayesian refinement Stage 2": 4,
}
COORDINATE_DECIMALS = 8
BPB_DUPLICATE_TOLERANCE = 1e-9


@dataclass(frozen=True)
class BudgetSurface:
    """One token-budget surface and its directly measured fiber overlays."""

    budget: int
    frame: pd.DataFrame
    fibers: tuple[pd.DataFrame, ...]
    replicate_summaries: pd.DataFrame

    @property
    def label(self) -> str:
        return BUDGET_LABELS[self.budget]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def _normalized_frame(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"p0", "p1", "bpb", "source", "wandb_id", "wandb_url"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Surface frame is missing columns: {sorted(missing)}")
    result = frame.copy()
    for column in ("p0", "p1", "bpb"):
        result[column] = pd.to_numeric(result[column], errors="raise")
    if not np.isfinite(result[["p0", "p1", "bpb"]].to_numpy(dtype=float)).all():
        raise ValueError("Surface frame contains non-finite coordinates or BPB")
    if not result["p0"].between(0.0, 1.0).all() or not result["p1"].between(0.0, 1.0).all():
        raise ValueError("Surface frame contains an invalid mixture weight")
    return result


def _bayesian_acquisition_rows(observations: pd.DataFrame, budget: int, *, stage: int) -> pd.DataFrame:
    selected = observations.loc[
        observations["token_budget_requested"].eq(budget)
        & observations["run_kind"].eq("acquisition")
        & observations["replicate_kind"].eq("reference")
        & observations["trainer_data_seed"].eq(REFERENCE_DATA_SEED)
    ].copy()
    selected = selected.rename(
        columns={
            "phase_0_starcoder": "p0",
            "phase_1_starcoder": "p1",
            "starcoder_bpb": "bpb",
        }
    )
    selected["source"] = f"Bayesian refinement Stage {stage}"
    return selected[["p0", "p1", "bpb", "source", "wandb_id", "wandb_url"]]


def _replicate_summaries(
    stage_1: pd.DataFrame,
    stage_2: pd.DataFrame,
    budget: int,
) -> pd.DataFrame:
    if budget in (2_000_000_000, 4_000_000_000):
        selected = stage_1.loc[
            stage_1["token_budget_requested"].eq(budget) & stage_1["run_kind"].eq("incumbent_repeat")
        ].copy()
        selected["summary_label"] = "Stage 1 incumbent repeats"
        selected["summary_role"] = "incumbent"
    elif budget == 8_000_000_000:
        selected = stage_2.loc[
            stage_2["token_budget_requested"].eq(budget)
            & stage_2["run_kind"].isin(("candidate_confirmation", "incumbent_confirmation"))
        ].copy()
        selected["summary_label"] = selected["run_kind"].map(
            {
                "candidate_confirmation": "Stage 2 candidate confirmation",
                "incumbent_confirmation": "Stage 2 incumbent confirmation",
            }
        )
        selected["summary_role"] = selected["run_kind"].str.removesuffix("_confirmation")
    else:
        return pd.DataFrame(columns=["p0", "p1", "bpb_mean", "bpb_sd", "count", "summary_label", "summary_role"])

    if selected.empty:
        raise ValueError(f"No replicate observations found for {BUDGET_LABELS[budget]}")
    summaries = (
        selected.groupby(
            ["phase_0_starcoder", "phase_1_starcoder", "summary_label", "summary_role"],
        )["starcoder_bpb"]
        .agg(bpb_mean="mean", bpb_sd="std", count="size")
        .reset_index()
        .rename(
            columns={
                "phase_0_starcoder": "p0",
                "phase_1_starcoder": "p1",
            }
        )
    )
    if summaries["count"].lt(2).any() or not np.isfinite(summaries["bpb_sd"]).all():
        raise ValueError(f"Replicate summary for {BUDGET_LABELS[budget]} lacks a finite sample SD")
    return summaries


def _dense_one_billion_surface(stage_1: pd.DataFrame) -> BudgetSurface:
    dense = pd.read_csv(DENSE_SURFACE_PATH).rename(
        columns={
            "phase_0_starcoder": "p0",
            "phase_1_starcoder": "p1",
            "wsd80_bpb": "bpb",
            "wandb_run_id": "wandb_id",
        }
    )
    dense["source"] = "dense 1B WSD80 panel"
    dense = _normalized_frame(dense[["p0", "p1", "bpb", "source", "wandb_id", "wandb_url"]])
    if len(dense) != 346 or dense.duplicated(["p0", "p1"]).any():
        raise ValueError("Canonical 1B surface must contain 346 unique coordinates")
    dense = _deduplicated_surface(
        pd.concat(
            [dense, _bayesian_acquisition_rows(stage_1, 1_000_000_000, stage=1)],
            ignore_index=True,
        )
    )

    observations = pd.read_csv(DENSE_FIBERS_PATH)
    observations = observations.loc[observations["data_seed"].eq(REFERENCE_DATA_SEED)].copy()
    fibers = []
    for fiber_id, fiber in observations.groupby("fiber_id", sort=False):
        plot_frame = fiber.rename(
            columns={
                "phase_0_starcoder": "p0",
                "phase_1_starcoder": "p1",
                "wsd80_bpb": "bpb",
            }
        )[
            ["p0", "p1", "bpb"]
        ].sort_values("p1")
        plot_frame["fiber_id"] = fiber_id
        plot_frame["fiber_label"] = str(fiber["fiber_label"].iloc[0])
        fibers.append(plot_frame.reset_index(drop=True))
    return BudgetSurface(
        1_000_000_000,
        dense.sort_values(["p0", "p1"]).reset_index(drop=True),
        tuple(fibers),
        _replicate_summaries(stage_1, pd.DataFrame(), 1_000_000_000),
    )


def _token_scaling_rows(observations: pd.DataFrame, budget: int) -> pd.DataFrame:
    selected = observations.loc[
        observations["token_budget_requested"].eq(budget)
        & observations["trainer_data_seed"].eq(REFERENCE_DATA_SEED)
        & observations["simulated_epoch_subset_seed"].eq(REFERENCE_DATA_SEED)
    ].copy()
    selected = selected.rename(
        columns={
            "phase_0_starcoder": "p0",
            "phase_1_starcoder": "p1",
            "starcoder_bpb": "bpb",
            "training_wandb_id": "wandb_id",
            "training_wandb_url": "wandb_url",
        }
    )
    selected["source"] = "token-scaling scaffold"
    return selected[["p0", "p1", "bpb", "source", "wandb_id", "wandb_url"]]


def _scale_fiber_rows(observations: pd.DataFrame, budget: int) -> pd.DataFrame:
    selected = observations.loc[
        observations["token_budget_requested"].eq(budget)
        & observations["trainer_data_seed"].eq(REFERENCE_DATA_SEED)
        & observations["simulated_epoch_subset_seed"].eq(REFERENCE_DATA_SEED)
    ].copy()
    selected = selected.rename(
        columns={
            "phase_0_starcoder": "p0",
            "phase_1_starcoder": "p1",
            "starcoder_bpb": "bpb",
            "wandb_id": "wandb_id",
            "wandb_url": "wandb_url",
        }
    )
    selected["source"] = "scale-specific fiber"
    return selected[
        [
            "p0",
            "p1",
            "bpb",
            "source",
            "wandb_id",
            "wandb_url",
            "anchor_index",
            "anchor_aggregate_starcoder",
            "anchor_role",
            "signed_contrast_phase1_minus_phase0",
        ]
    ]


def _tied_diagonal_rows(observations: pd.DataFrame, budget: int) -> pd.DataFrame:
    selected = observations.loc[observations["token_budget_requested"].eq(budget)].copy()
    selected["p0"] = selected["weight"]
    selected["p1"] = selected["weight"]
    selected = selected.rename(
        columns={
            "starcoder_bpb": "bpb",
            "wandb_id": "wandb_id",
            "wandb_url": "wandb_url",
        }
    )
    selected["source"] = "tied diagonal"
    return selected[["p0", "p1", "bpb", "source", "wandb_id", "wandb_url"]]


def _deduplicated_surface(rows: pd.DataFrame) -> pd.DataFrame:
    rows = _normalized_frame(rows)
    rows["coordinate_key"] = list(
        zip(rows["p0"].round(COORDINATE_DECIMALS), rows["p1"].round(COORDINATE_DECIMALS), strict=True)
    )
    for coordinate, duplicates in rows.groupby("coordinate_key"):
        if float(duplicates["bpb"].max() - duplicates["bpb"].min()) > BPB_DUPLICATE_TOLERANCE:
            raise ValueError(f"Conflicting BPB values at coordinate {coordinate}: {duplicates.to_dict('records')}")
        if duplicates["wandb_id"].astype(str).nunique() > 1:
            raise ValueError(f"Duplicate coordinate {coordinate} points to multiple W&B runs")
    rows["source_priority"] = rows["source"].map(SOURCE_PRIORITY)
    if rows["source_priority"].isna().any():
        raise ValueError("A surface row has an unknown source priority")
    result = (
        rows.sort_values(["source_priority", "p0", "p1"])
        .drop_duplicates("coordinate_key", keep="first")
        .drop(columns=["coordinate_key", "source_priority"])
        .sort_values(["p0", "p1"])
        .reset_index(drop=True)
    )
    if len(result) < 3:
        raise ValueError("At least three unique coordinates are required for a surface")
    return result


def _higher_budget_surface(
    budget: int,
    token_scaling: pd.DataFrame,
    tied_diagonal: pd.DataFrame,
    scale_fibers: pd.DataFrame,
    stage_1: pd.DataFrame,
    stage_2: pd.DataFrame,
) -> BudgetSurface:
    token_rows = _token_scaling_rows(token_scaling, budget)
    fiber_rows = _scale_fiber_rows(scale_fibers, budget)
    tied_rows = _tied_diagonal_rows(tied_diagonal, budget)
    frame = _deduplicated_surface(
        pd.concat(
            [
                token_rows,
                fiber_rows,
                tied_rows,
                _bayesian_acquisition_rows(stage_1, budget, stage=1),
                _bayesian_acquisition_rows(stage_2, budget, stage=2),
            ],
            ignore_index=True,
        )
    )

    fibers = []
    for anchor_index, fiber in fiber_rows.groupby("anchor_index", sort=True):
        plot_frame = fiber[["p0", "p1", "bpb", "signed_contrast_phase1_minus_phase0"]].sort_values(
            "signed_contrast_phase1_minus_phase0"
        )
        aggregate = float(fiber["anchor_aggregate_starcoder"].iloc[0])
        role = str(fiber["anchor_role"].iloc[0]).replace("_", " ")
        plot_frame["fiber_id"] = f"anchor_{int(anchor_index)}"
        plot_frame["fiber_label"] = f"Measured fiber a={aggregate:.2f} · {role}"
        fibers.append(plot_frame.drop(columns="signed_contrast_phase1_minus_phase0").reset_index(drop=True))
    return BudgetSurface(budget, frame, tuple(fibers), _replicate_summaries(stage_1, stage_2, budget))


def _load_surfaces() -> tuple[BudgetSurface, ...]:
    token_scaling = pd.read_csv(TOKEN_SCALING_PATH)
    tied_diagonal = pd.read_csv(TIED_DIAGONAL_PATH)
    scale_fibers = pd.read_csv(SCALE_FIBERS_PATH)
    stage_1 = pd.read_csv(BAYESIAN_STAGE_1_PATH)
    stage_2 = pd.read_csv(BAYESIAN_STAGE_2_PATH)
    surfaces = [_dense_one_billion_surface(stage_1)]
    for budget in BUDGETS[1:]:
        surfaces.append(_higher_budget_surface(budget, token_scaling, tied_diagonal, scale_fibers, stage_1, stage_2))
    return tuple(surfaces)


def _hover_text(surface: BudgetSurface) -> list[str]:
    frame = surface.frame
    aggregate = PHASE_0_FRACTION * frame["p0"] + (1.0 - PHASE_0_FRACTION) * frame["p1"]
    contrast = frame["p1"] - frame["p0"]
    return [
        "<br>".join(
            [
                f"{surface.label} materialized tokens",
                f"Phase 0 StarCoder: {p0:.4f}",
                f"Phase 1 StarCoder: {p1:.4f}",
                f"80/20 aggregate: {agg:.4f}",
                f"Phase contrast p1-p0: {delta:+.4f}",
                f"{METRIC_LABEL}: {bpb:.6f}",
                f"Source: {source}",
                f"W&B run: {run_id}",
            ]
        )
        for p0, p1, agg, delta, bpb, source, run_id in zip(
            frame["p0"],
            frame["p1"],
            aggregate,
            contrast,
            frame["bpb"],
            frame["source"],
            frame["wandb_id"],
            strict=True,
        )
    ]


def _special_point_trace(
    row: pd.Series,
    *,
    name: str,
    symbol: str,
    color: str,
    visible: bool,
) -> go.Scatter3d:
    return go.Scatter3d(
        x=[row["p0"]],
        y=[row["p1"]],
        z=[row["bpb"]],
        mode="markers",
        marker={"symbol": symbol, "size": 7, "color": color, "line": {"color": "white", "width": 1.2}},
        text=[f"{name}<br>p0={row['p0']:.4f}<br>p1={row['p1']:.4f}<br>{METRIC_LABEL}={row['bpb']:.6f}"],
        hoverinfo="text",
        name=f"{name}: p0={row['p0']:.3f}, p1={row['p1']:.3f}; BPB={row['bpb']:.3f}",
        visible=visible,
    )


def _replicate_summary_trace(row: pd.Series, *, visible: bool) -> go.Scatter3d:
    role = str(row["summary_role"])
    color = "#1B5E75" if role == "candidate" else "#6C4E88"
    symbol = "diamond-open" if role == "candidate" else "square-open"
    label = str(row["summary_label"])
    return go.Scatter3d(
        x=[row["p0"]],
        y=[row["p1"]],
        z=[row["bpb_mean"]],
        mode="markers",
        marker={"symbol": symbol, "size": 8, "color": color, "line": {"color": color, "width": 2}},
        error_z={
            "type": "data",
            "array": [row["bpb_sd"]],
            "visible": True,
            "color": color,
            "thickness": 3,
            "width": 6,
        },
        text=[
            "<br>".join(
                [
                    label,
                    f"p0={row['p0']:.4f}",
                    f"p1={row['p1']:.4f}",
                    f"Mean {METRIC_LABEL}: {row['bpb_mean']:.6f}",
                    f"SD: {row['bpb_sd']:.6f}",
                    f"Independent seeds: {int(row['count'])}",
                    "Not used in surface triangulation",
                ]
            )
        ],
        hoverinfo="text",
        name=f"{label}: mean ± 1 SD ({int(row['count'])} seeds)",
        visible=visible,
    )


def _add_budget_traces(
    figure: go.Figure,
    surface: BudgetSurface,
    *,
    visible: bool,
    color_min: float,
    color_max: float,
) -> tuple[int, ...]:
    start = len(figure.data)
    frame = surface.frame
    triangles = _triangle_indices(frame)
    figure.add_trace(
        go.Mesh3d(
            x=frame["p0"],
            y=frame["p1"],
            z=frame["bpb"],
            i=triangles[:, 0],
            j=triangles[:, 1],
            k=triangles[:, 2],
            intensity=frame["bpb"],
            colorscale=COLOR_SCALE,
            cmin=color_min,
            cmax=color_max,
            opacity=0.38,
            showscale=True,
            colorbar={"title": "BPB", "len": 0.58, "thickness": 16},
            hoverinfo="skip",
            name="linear triangulation",
            visible=visible,
        )
    )
    figure.add_trace(
        go.Scatter3d(
            x=frame["p0"],
            y=frame["p1"],
            z=frame["bpb"],
            mode="markers",
            marker={
                "size": 4.0,
                "color": frame["bpb"],
                "colorscale": COLOR_SCALE,
                "cmin": color_min,
                "cmax": color_max,
                "line": {"color": "white", "width": 0.8},
                "showscale": False,
            },
            text=_hover_text(surface),
            hoverinfo="text",
            name=f"observed runs ({len(frame)} coordinates)",
            visible=visible,
        )
    )

    best = frame.loc[frame["bpb"].idxmin()]
    tied = frame.loc[np.isclose(frame["p0"], frame["p1"], atol=1e-10)]
    if tied.empty:
        raise ValueError(f"{surface.label} surface has no tied coordinates")
    tied_best = tied.loc[tied["bpb"].idxmin()]
    boundary = frame.loc[np.isclose(frame["p0"], 0.0, atol=1e-10)]
    if boundary.empty:
        raise ValueError(f"{surface.label} surface has no p0=0 boundary coordinate")
    boundary_best = boundary.loc[boundary["bpb"].idxmin()]
    proportional = frame.iloc[
        np.argmin((frame["p0"] - PROPORTIONAL_STARCODER) ** 2 + (frame["p1"] - PROPORTIONAL_STARCODER) ** 2)
    ]
    figure.add_trace(
        _special_point_trace(
            best,
            name="best observed",
            symbol="diamond",
            color=GLOBAL_MIN_COLOR,
            visible=visible,
        )
    )
    figure.add_trace(
        _special_point_trace(
            tied_best,
            name="best sampled constant mixture",
            symbol="x",
            color=DIAGONAL_MIN_COLOR,
            visible=visible,
        )
    )
    figure.add_trace(
        _special_point_trace(
            boundary_best,
            name="best p0=0 boundary point",
            symbol="cross",
            color=BOUNDARY_MIN_COLOR,
            visible=visible,
        )
    )
    figure.add_trace(
        _special_point_trace(
            proportional,
            name="nearest sampled proportional point",
            symbol="circle",
            color=PROPORTIONAL_COLOR,
            visible=visible,
        )
    )

    tied_line = tied.sort_values("p0")
    figure.add_trace(
        go.Scatter3d(
            x=tied_line["p0"],
            y=tied_line["p1"],
            z=tied_line["bpb"],
            mode="lines+markers",
            line={"color": "#51606F", "width": 4, "dash": "dash"},
            marker={"color": "#51606F", "size": 2.5},
            hovertext=[
                f"Tied diagonal<br>p0=p1={weight:.3f}<br>{METRIC_LABEL}={bpb:.6f}"
                for weight, bpb in zip(tied_line["p0"], tied_line["bpb"], strict=True)
            ],
            hoverinfo="text",
            name="measured tied diagonal",
            visible=visible,
        )
    )
    for fiber_index, fiber in enumerate(surface.fibers):
        color = FIBER_COLORS[fiber_index % len(FIBER_COLORS)]
        label = str(fiber["fiber_label"].iloc[0])
        figure.add_trace(
            go.Scatter3d(
                x=fiber["p0"],
                y=fiber["p1"],
                z=fiber["bpb"],
                mode="lines+markers",
                line={"color": color, "width": 6},
                marker={"color": color, "size": 3},
                hovertext=[
                    f"{label}<br>p0={p0:.4f}<br>p1={p1:.4f}<br>{METRIC_LABEL}={bpb:.6f}"
                    for p0, p1, bpb in zip(fiber["p0"], fiber["p1"], fiber["bpb"], strict=True)
                ],
                hoverinfo="text",
                name=label,
                visible=visible,
            )
        )
    for _, replicate_summary in surface.replicate_summaries.iterrows():
        figure.add_trace(_replicate_summary_trace(replicate_summary, visible=visible))
    return tuple(range(start, len(figure.data)))


def _metric_range(frame: pd.DataFrame) -> tuple[float, float]:
    minimum = float(frame["bpb"].min())
    maximum = float(frame["bpb"].max())
    span = max(maximum - minimum, 0.005)
    return minimum - 0.04 * span, maximum + 0.04 * span


def _title(surface: BudgetSurface) -> str:
    repeat_text = ""
    if not surface.replicate_summaries.empty:
        repeat_count = int(surface.replicate_summaries["count"].sum())
        repeat_text = f" · {repeat_count} repeat observations"
    return (
        "StarCoder response under 80/20 WSD"
        f"<br><sup>{surface.label} materialized tokens · {len(surface.frame)} coordinates · "
        f"total-parameter TPP {TOTAL_PARAMETER_TPP[surface.budget]:.2f}{repeat_text}</sup>"
    )


def _render(surfaces: tuple[BudgetSurface, ...], output_dir: Path) -> go.Figure:
    all_bpb = pd.concat([surface.frame["bpb"] for surface in surfaces], ignore_index=True)
    color_min = float(all_bpb.min())
    color_max = float(all_bpb.quantile(0.96))
    figure = go.Figure()
    trace_indices = {}
    metric_ranges = {}
    for surface_index, surface in enumerate(surfaces):
        trace_indices[surface.budget] = _add_budget_traces(
            figure,
            surface,
            visible=surface_index == 0,
            color_min=color_min,
            color_max=color_max,
        )
        metric_ranges[surface.budget] = _metric_range(surface.frame)

    first = surfaces[0]
    _add_fact_sheet(
        figure,
        (
            (
                ("Model", "Fixed Llama: 10 layers, d=768, FFN=1536; 157.5M total parameters"),
                ("Tokenizer / sequence", "Llama 3.1 tokenizer; 2,048 tokens"),
                ("Metric", METRIC_LABEL),
            ),
            (
                ("Token ladder", "1B, 2B, 4B, and 8B materialized tokens"),
                ("Total-parameter TPP", "6.35, 12.70, 25.40, and 50.79"),
                ("Non-embedding TPP", "16.95, 33.89, 67.79, and 135.58"),
            ),
            (
                ("Phases", "80% stable / 20% cosine decay; phase weights shown on horizontal axes"),
                ("1B surface", "Canonical dense 346-coordinate WSD80 panel"),
                (
                    "2B-8B surfaces",
                    "Reference-seed scaffold + tied diagonal + measured fibers + two Bayesian-refinement stages",
                ),
            ),
            (
                ("Interpolation", "Linear Delaunay triangulation inside each observed convex hull"),
                (
                    "Uncertainty",
                    "Repeat means ±1 SD are overlaid but excluded from triangulation",
                ),
                ("Color", "One absolute BPB scale is shared across every token budget"),
            ),
        ),
    )

    num_traces = len(figure.data)
    steps = []
    for surface in surfaces:
        visibility = [False] * num_traces
        for trace_index in trace_indices[surface.budget]:
            visibility[trace_index] = True
        steps.append(
            {
                "label": surface.label,
                "method": "update",
                "args": [
                    {"visible": visibility},
                    {
                        "title.text": _title(surface),
                        "scene.zaxis.range": list(metric_ranges[surface.budget]),
                    },
                ],
            }
        )

    figure.update_layout(
        template="plotly_white",
        width=1120,
        height=1110,
        paper_bgcolor=PAPER_BACKGROUND,
        font={"family": SERIF_FONT, "size": 17, "color": PAPER_TEXT},
        title={"text": _title(first), "x": 0.5, "y": 0.98, "font": {"size": 26}},
        legend={"x": 0.01, "y": 0.88, "bgcolor": "rgba(255,255,255,0.88)", "font": {"size": 12}},
        scene=_scene_layout(*metric_ranges[first.budget], z_title=METRIC_LABEL),
        sliders=[
            {
                "active": 0,
                "steps": steps,
                "x": 0.18,
                "len": 0.68,
                "y": 1.03,
                "xanchor": "left",
                "yanchor": "bottom",
                "currentvalue": {
                    "prefix": "Materialized tokens: ",
                    "font": {"family": "Arial, sans-serif", "size": 13, "color": PAPER_TEXT},
                },
                "font": {"family": "Arial, sans-serif", "size": 12, "color": PAPER_TEXT},
                "pad": {"t": 8, "b": 0},
                "transition": {"duration": 0},
            }
        ],
        annotations=[
            *figure.layout.annotations,
            {
                "text": "<b>TOKEN BUDGET</b>",
                "x": 0.02,
                "xref": "paper",
                "y": 1.075,
                "yref": "paper",
                "showarrow": False,
                "xanchor": "left",
                "font": {"family": "Arial, sans-serif", "size": 12, "color": "#C94F2D"},
            },
        ],
        margin={"l": 20, "r": 80, "t": 155, "b": 220},
        transition={"duration": 0},
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    figure.write_html(
        output_dir / "starcoder_wsd80_token_budget_surfaces.html",
        include_plotlyjs=True,
        include_mathjax="cdn",
        config=EXPORT_CONFIG,
    )
    figure.write_image(output_dir / "starcoder_wsd80_token_budget_surfaces.png", scale=2)
    return figure


def _write_data(surfaces: tuple[BudgetSurface, ...], output_dir: Path) -> None:
    rows = []
    replicate_rows = []
    summaries = []
    for surface in surfaces:
        frame = surface.frame.copy()
        frame.insert(0, "token_budget_requested", surface.budget)
        frame.insert(1, "token_budget_label", surface.label)
        frame["total_parameter_tpp"] = TOTAL_PARAMETER_TPP[surface.budget]
        frame["non_embedding_parameter_tpp"] = NON_EMBEDDING_TPP[surface.budget]
        rows.append(frame)
        if not surface.replicate_summaries.empty:
            replicate_frame = surface.replicate_summaries.copy()
            replicate_frame.insert(0, "token_budget_requested", surface.budget)
            replicate_frame.insert(1, "token_budget_label", surface.label)
            replicate_rows.append(replicate_frame)
        best = frame.loc[frame["bpb"].idxmin()]
        tied = frame.loc[np.isclose(frame["p0"], frame["p1"], atol=1e-10)]
        tied_best = tied.loc[tied["bpb"].idxmin()]
        summaries.append(
            {
                "token_budget_requested": surface.budget,
                "token_budget_label": surface.label,
                "coordinate_count": len(frame),
                "measured_fiber_count": len(surface.fibers),
                "replicate_summary_count": len(surface.replicate_summaries),
                "replicate_observation_count": int(surface.replicate_summaries["count"].sum()),
                "total_parameter_tpp": TOTAL_PARAMETER_TPP[surface.budget],
                "non_embedding_parameter_tpp": NON_EMBEDDING_TPP[surface.budget],
                "best_observed_p0": float(best["p0"]),
                "best_observed_p1": float(best["p1"]),
                "best_observed_bpb": float(best["bpb"]),
                "best_tied_weight": float(tied_best["p0"]),
                "best_tied_bpb": float(tied_best["bpb"]),
            }
        )
    pd.concat(rows, ignore_index=True).to_csv(output_dir / "surface_coordinates.csv", index=False)
    pd.concat(replicate_rows, ignore_index=True).to_csv(output_dir / "replicate_summaries.csv", index=False)
    (output_dir / "surface_summary.json").write_text(json.dumps(summaries, indent=2) + "\n")


def main() -> None:
    args = _parse_args()
    surfaces = _load_surfaces()
    if tuple(surface.budget for surface in surfaces) != BUDGETS:
        raise ValueError("Unexpected token-budget ordering")
    coordinate_counts = {surface.budget: len(surface.frame) for surface in surfaces}
    if coordinate_counts != EXPECTED_COORDINATE_COUNTS:
        raise ValueError(
            f"Unexpected coordinate counts: expected {EXPECTED_COORDINATE_COUNTS}, observed {coordinate_counts}"
        )
    _render(surfaces, args.output_dir)
    _write_data(surfaces, args.output_dir)
    print(args.output_dir / "starcoder_wsd80_token_budget_surfaces.html")


if __name__ == "__main__":
    main()
