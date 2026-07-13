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
"""Collect and compare the completed StarCoder 80/20 WSD surface."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import wandb
from plotly.colors import get_colorscale
from plotly.subplots import make_subplots
from scipy import stats
from scipy.spatial import Delaunay

SCRIPT_DIR = Path(__file__).resolve().parent
EXPLORATORY_DIR = SCRIPT_DIR.parent
PAPER_PLOTS_DIR = EXPLORATORY_DIR / "paper_plots"

COORDINATE_MANIFEST = (
    SCRIPT_DIR
    / "reference_outputs"
    / "starcoder_80_20_wsd_coordinate_selection_boundary_20260711"
    / "selected_coordinates_64.csv"
)
COSINE_CSV = PAPER_PLOTS_DIR / "data" / "two_phase_starcoder_combined_143_from_wandb.csv"
WSD50_CSV = (
    EXPLORATORY_DIR
    / "starcoder_wsd_boundary_aligned_repeat_outputs"
    / "two_phase_feature_bayes_linear_20260313_211537"
    / "proxy_results.csv"
)
REPEAT_CSV = (
    EXPLORATORY_DIR
    / "reference_outputs"
    / "starcoder_heteroskedastic_snr_20260523"
    / "collected_train_only_metrics_live.csv"
)
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "reference_outputs" / "starcoder_wsd80_surface_analysis_20260711"

WANDB_PATH = "marin-community/marin"
WSD80_GROUP = "pinlin_calvin_xu/data_mixture/two_phase_starcoder_wsd80_20_surface64_20260711_retry2"
WSD80_REPEAT_GROUP = "pinlin_calvin_xu/data_mixture/two_phase_starcoder_wsd80_20_repeat3x4_20260711"
WSD80_TARGET = "eval/paloma/dolma_100_programing_languages-llama3/bpb"
HISTORICAL_TARGET = "eval/paloma/dolma_100_programing_languages/bpb"
RUN_NAME_PATTERN = re.compile(r"surface64_r(?P<rank>\d+)_p0_.*")
REPEAT_RUN_NAME_PATTERN = re.compile(r"wsd80_repeat_(?P<schedule>global|boundary|constant)_seed(?P<seed>\d+)")

PHASE_0_FRACTION = 0.8
PHASE_1_FRACTION = 0.2
PROPORTIONAL_STARCODER = 0.036419434769597664
REFERENCE_DATA_SEED = 20_260_711
REPEAT_DATA_SEEDS = (20_260_712, 20_260_713, 20_260_714, 20_260_715)
REPEAT_SCHEDULES = {
    "global": (0.1452468603730965, 0.517364768878253),
    "boundary": (0.0, 0.6),
    "constant": (0.3, 0.3),
}
REPEAT_SCHEDULE_LABELS = {
    "global": "Off-diagonal surface minimum",
    "boundary": "Best sampled p0=0 boundary",
    "constant": "Best sampled constant",
}

COLOR_SCALE = get_colorscale("RdYlGn_r")
PAPER_TEXT = "#23395D"
PAPER_GRID = "#DCE6F2"
PAPER_BACKGROUND = "#FFFFFF"
PANE_BACKGROUND = "#E8EEF6"
GLOBAL_MIN_COLOR = "#E64B35"
DIAGONAL_MIN_COLOR = "#F2B701"
BOUNDARY_MIN_COLOR = "#2C7FB8"
PROPORTIONAL_COLOR = "#FFD700"
SERIF_FONT = "Times New Roman, Times, serif"
EXPORT_CONFIG = {"toImageButtonOptions": {"format": "png", "scale": 4}}


@dataclass(frozen=True)
class Surface:
    """One observed phase-mixture response surface."""

    name: str
    frame: pd.DataFrame
    phase_0_fraction: float


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def _collect_wsd80() -> pd.DataFrame:
    api = wandb.Api(timeout=60)
    runs = list(api.runs(WANDB_PATH, filters={"group": WSD80_GROUP}, per_page=200))
    if len(runs) != 64:
        raise ValueError(f"Expected 64 W&B runs in {WSD80_GROUP!r}, got {len(runs)}")

    rows: list[dict[str, object]] = []
    for run in runs:
        match = RUN_NAME_PATTERN.fullmatch(run.name)
        if match is None:
            raise ValueError(f"Unexpected W&B run name: {run.name!r}")
        target = run.summary.get(WSD80_TARGET)
        if run.state != "finished" or target is None:
            raise ValueError(f"Incomplete W&B run {run.name}: state={run.state}, target={target}")
        rows.append(
            {
                "selection_rank": int(match.group("rank")),
                "wandb_run_id": run.id,
                "wandb_run_name": run.name,
                "wandb_url": run.url,
                "wandb_state": run.state,
                "wsd80_bpb": float(target),
                "eval_loss": float(run.summary["eval/loss"]),
            }
        )

    manifest = pd.read_csv(COORDINATE_MANIFEST)
    observed = pd.DataFrame(rows)
    merged = manifest.merge(observed, on="selection_rank", how="left", validate="one_to_one")
    if len(merged) != 64 or merged["wsd80_bpb"].isna().any():
        raise ValueError("The 64-coordinate manifest did not join one-to-one with completed W&B results")
    return merged.sort_values("selection_rank").reset_index(drop=True)


def _matched_surface_row(wsd80: pd.DataFrame, phase_0: float, phase_1: float) -> pd.Series:
    matched = wsd80[
        np.isclose(wsd80["phase_0_starcoder"], phase_0, atol=1e-10)
        & np.isclose(wsd80["phase_1_starcoder"], phase_1, atol=1e-10)
    ]
    if len(matched) != 1:
        raise ValueError(f"Expected one surface row for ({phase_0}, {phase_1}), got {len(matched)}")
    return matched.iloc[0]


def _collect_wsd80_repeats(wsd80: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for schedule, (phase_0, phase_1) in REPEAT_SCHEDULES.items():
        surface_row = _matched_surface_row(wsd80, phase_0, phase_1)
        rows.append(
            {
                "schedule": schedule,
                "schedule_label": REPEAT_SCHEDULE_LABELS[schedule],
                "data_seed": REFERENCE_DATA_SEED,
                "phase_0_starcoder": phase_0,
                "phase_1_starcoder": phase_1,
                "bpb": float(surface_row["wsd80_bpb"]),
                "wandb_run_id": surface_row["wandb_run_id"],
                "wandb_run_name": surface_row["wandb_run_name"],
                "wandb_url": surface_row["wandb_url"],
                "wandb_group": WSD80_GROUP,
                "source": "surface reference seed",
            }
        )

    api = wandb.Api(timeout=60)
    runs = list(api.runs(WANDB_PATH, filters={"group": WSD80_REPEAT_GROUP}, per_page=100))
    if len(runs) != 12:
        raise ValueError(f"Expected 12 W&B runs in {WSD80_REPEAT_GROUP!r}, got {len(runs)}")
    for run in runs:
        match = REPEAT_RUN_NAME_PATTERN.fullmatch(run.name)
        if match is None:
            raise ValueError(f"Unexpected repeat W&B run name: {run.name!r}")
        schedule = match.group("schedule")
        data_seed = int(match.group("seed"))
        target = run.summary.get(WSD80_TARGET)
        if run.state != "finished" or target is None:
            raise ValueError(f"Incomplete repeat W&B run {run.name}: state={run.state}, target={target}")
        phase_0, phase_1 = REPEAT_SCHEDULES[schedule]
        rows.append(
            {
                "schedule": schedule,
                "schedule_label": REPEAT_SCHEDULE_LABELS[schedule],
                "data_seed": data_seed,
                "phase_0_starcoder": phase_0,
                "phase_1_starcoder": phase_1,
                "bpb": float(target),
                "wandb_run_id": run.id,
                "wandb_run_name": run.name,
                "wandb_url": run.url,
                "wandb_group": WSD80_REPEAT_GROUP,
                "source": "paired repeat panel",
            }
        )

    frame = pd.DataFrame(rows).sort_values(["data_seed", "schedule"]).reset_index(drop=True)
    expected_seeds = {REFERENCE_DATA_SEED, *REPEAT_DATA_SEEDS}
    if set(frame["data_seed"]) != expected_seeds:
        raise ValueError(f"Repeat panel seeds do not match: {sorted(frame['data_seed'].unique())}")
    counts = frame.groupby(["data_seed", "schedule"]).size()
    if len(frame) != 15 or len(counts) != 15 or not counts.eq(1).all():
        raise ValueError("Expected exactly one observation for every one of 5 seeds x 3 schedules")
    return frame


def _exact_sign_flip_pvalue(deltas: np.ndarray, *, alternative: str) -> float:
    observed = float(deltas.mean())
    null_means = np.asarray(
        [np.mean(deltas * np.asarray(signs, dtype=float)) for signs in product((-1.0, 1.0), repeat=len(deltas))]
    )
    tolerance = 1e-15
    if alternative == "two-sided":
        return float(np.mean(np.abs(null_means) >= abs(observed) - tolerance))
    if alternative == "less":
        return float(np.mean(null_means <= observed + tolerance))
    raise ValueError(f"Unsupported sign-flip alternative: {alternative!r}")


def _paired_contrast(
    pivot: pd.DataFrame,
    schedule_a: str,
    schedule_b: str,
) -> tuple[dict[str, object], pd.DataFrame]:
    deltas = (pivot[schedule_a] - pivot[schedule_b]).to_numpy(dtype=float)
    n = len(deltas)
    mean = float(deltas.mean())
    sd = float(deltas.std(ddof=1))
    standard_error = sd / np.sqrt(n)
    critical = float(stats.t.ppf(0.975, df=n - 1))
    t_statistic = mean / standard_error
    two_sided_p = float(stats.t.sf(abs(t_statistic), df=n - 1) * 2.0)
    summary = {
        "schedule_a": schedule_a,
        "schedule_b": schedule_b,
        "contrast": f"{schedule_a} - {schedule_b}",
        "n_pairs": n,
        "mean_delta_bpb": mean,
        "paired_sd": sd,
        "standard_error": standard_error,
        "ci_95_low": mean - critical * standard_error,
        "ci_95_high": mean + critical * standard_error,
        "t_statistic": t_statistic,
        "paired_t_two_sided_p": two_sided_p,
        "paired_t_one_sided_a_lower_p": float(stats.t.cdf(t_statistic, df=n - 1)),
        "sign_flip_two_sided_p": _exact_sign_flip_pvalue(deltas, alternative="two-sided"),
        "sign_flip_one_sided_a_lower_p": _exact_sign_flip_pvalue(deltas, alternative="less"),
        "a_lower_b_count": int(np.sum(deltas < 0.0)),
    }
    raw = pd.DataFrame(
        {
            "data_seed": pivot.index.astype(int),
            "schedule_a": schedule_a,
            "schedule_b": schedule_b,
            "delta_bpb_a_minus_b": deltas,
        }
    )
    return summary, raw


def _paired_repeat_summary(repeats: pd.DataFrame) -> tuple[dict[str, object], pd.DataFrame]:
    pivot = repeats.pivot(index="data_seed", columns="schedule", values="bpb")
    schedule_summaries = {
        schedule: {
            "phase_0_starcoder": REPEAT_SCHEDULES[schedule][0],
            "phase_1_starcoder": REPEAT_SCHEDULES[schedule][1],
            "n": len(values),
            "mean_bpb": float(values.mean()),
            "sd_bpb": float(values.std(ddof=1)),
        }
        for schedule, values in pivot.items()
    }
    contrasts: list[dict[str, object]] = []
    raw_contrasts: list[pd.DataFrame] = []
    for schedule_a, schedule_b in [
        ("global", "constant"),
        ("boundary", "constant"),
        ("global", "boundary"),
    ]:
        contrast, raw = _paired_contrast(pivot, schedule_a, schedule_b)
        contrasts.append(contrast)
        raw_contrasts.append(raw)
    seed_means = pivot.mean(axis=1)
    summary = {
        "wandb_repeat_group": WSD80_REPEAT_GROUP,
        "seeds": [int(seed) for seed in pivot.index],
        "schedule_summaries": schedule_summaries,
        "paired_contrasts": contrasts,
        "seed_common_mode": {
            "lowest_seed_mean": float(seed_means.min()),
            "highest_seed_mean": float(seed_means.max()),
            "range": float(seed_means.max() - seed_means.min()),
        },
    }
    return summary, pd.concat(raw_contrasts, ignore_index=True)


def _surface_frame(
    frame: pd.DataFrame,
    *,
    target: str,
    url_column: str | None = None,
) -> pd.DataFrame:
    result = pd.DataFrame(
        {
            "p0": frame["phase_0_starcoder"].astype(float),
            "p1": frame["phase_1_starcoder"].astype(float),
            "bpb": frame[target].astype(float),
        }
    )
    result["url"] = frame[url_column].astype(str) if url_column is not None else ""
    return result.dropna(subset=["p0", "p1", "bpb"]).drop_duplicates(["p0", "p1"]).reset_index(drop=True)


def _load_surfaces(wsd80: pd.DataFrame) -> list[Surface]:
    cosine = pd.read_csv(COSINE_CSV)
    cosine = cosine[cosine["status"].eq("completed") & cosine[HISTORICAL_TARGET].notna()].copy()
    wsd50 = pd.read_csv(WSD50_CSV)
    wsd50 = wsd50[wsd50["actual_bpb"].notna()].copy()
    return [
        Surface("Cosine, 50/50 phases", _surface_frame(cosine, target=HISTORICAL_TARGET), 0.5),
        Surface(
            "WSD, 50/50 phases",
            _surface_frame(wsd50, target="actual_bpb", url_column="wandb_url"),
            0.5,
        ),
        Surface(
            "WSD, 80/20 phases",
            _surface_frame(wsd80, target="wsd80_bpb", url_column="wandb_url"),
            PHASE_0_FRACTION,
        ),
    ]


def _triangle_indices(frame: pd.DataFrame) -> np.ndarray:
    points = frame[["p0", "p1"]].to_numpy(dtype=float)
    return Delaunay(points).simplices


def _hover_text(frame: pd.DataFrame, phase_0_fraction: float) -> list[str]:
    aggregate = phase_0_fraction * frame["p0"] + (1.0 - phase_0_fraction) * frame["p1"]
    return [
        "<br>".join(
            [
                f"Phase 0 StarCoder: {p0:.4f}",
                f"Phase 1 StarCoder: {p1:.4f}",
                f"Aggregate StarCoder: {agg:.4f}",
                f"BPB: {bpb:.6f}",
            ]
        )
        for p0, p1, agg, bpb in zip(frame["p0"], frame["p1"], aggregate, frame["bpb"], strict=True)
    ]


def _add_surface(
    fig: go.Figure,
    surface: Surface,
    *,
    scene: str,
    color_min: float,
    color_max: float,
    show_scale: bool,
    show_legend: bool,
) -> None:
    frame = surface.frame
    triangles = _triangle_indices(frame)
    fig.add_trace(
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
            showscale=show_scale,
            colorbar={"title": "BPB", "len": 0.58, "thickness": 16} if show_scale else None,
            hoverinfo="skip",
            name="linear triangulation",
            showlegend=show_legend,
            scene=scene,
        )
    )
    fig.add_trace(
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
            text=_hover_text(frame, surface.phase_0_fraction),
            hoverinfo="text",
            name="observed runs",
            showlegend=show_legend,
            scene=scene,
        )
    )
    best = frame.loc[frame["bpb"].idxmin()]
    fig.add_trace(
        go.Scatter3d(
            x=[best["p0"]],
            y=[best["p1"]],
            z=[best["bpb"]],
            mode="markers",
            marker={"symbol": "diamond", "size": 7, "color": GLOBAL_MIN_COLOR, "line": {"color": "white", "width": 1.2}},
            text=[f"best observed<br>p0={best['p0']:.4f}<br>p1={best['p1']:.4f}<br>BPB={best['bpb']:.6f}"],
            hoverinfo="text",
            name="best observed",
            showlegend=show_legend,
            scene=scene,
        )
    )


def _scene_layout(z_min: float, z_max: float) -> dict[str, object]:
    axis = {
        "range": [0, 1],
        "gridcolor": "white",
        "backgroundcolor": PANE_BACKGROUND,
        "showbackground": True,
        "zeroline": False,
    }
    return {
        "xaxis": {**axis, "title": "Phase 0 StarCoder"},
        "yaxis": {**axis, "title": "Phase 1 StarCoder"},
        "zaxis": {
            "title": "Dolma 100 Programming Languages BPB",
            "range": [z_min, z_max],
            "gridcolor": "white",
            "backgroundcolor": PANE_BACKGROUND,
            "showbackground": True,
            "zeroline": False,
        },
        "camera": {"eye": {"x": -1.55, "y": -1.55, "z": 1.25}},
        "aspectmode": "manual",
        "aspectratio": {"x": 1.0, "y": 1.0, "z": 0.8},
    }


def _write_figure(fig: go.Figure, stem: Path, *, static_scale: int = 2) -> None:
    fig.write_html(
        stem.with_suffix(".html"),
        include_plotlyjs="cdn",
        include_mathjax="cdn",
        config=EXPORT_CONFIG,
    )
    fig.write_image(stem.with_suffix(".png"), scale=static_scale)


def _render_wsd80_surface(surface: Surface, output_dir: Path) -> None:
    frame = surface.frame
    color_min = float(frame["bpb"].min())
    color_max = float(np.quantile(frame["bpb"], 0.96))
    z_max = float(max(2.3, frame["bpb"].max() * 1.02))
    fig = go.Figure()
    _add_surface(
        fig,
        surface,
        scene="scene",
        color_min=color_min,
        color_max=color_max,
        show_scale=True,
        show_legend=True,
    )

    diagonal = frame[np.isclose(frame["p0"], frame["p1"], atol=1e-10)]
    diagonal_best = diagonal.loc[diagonal["bpb"].idxmin()]
    boundary = frame[np.isclose(frame["p0"], 0.0, atol=1e-10)]
    boundary_best = boundary.loc[boundary["bpb"].idxmin()]
    proportional = frame.iloc[
        np.argmin((frame["p0"] - PROPORTIONAL_STARCODER) ** 2 + (frame["p1"] - PROPORTIONAL_STARCODER) ** 2)
    ]
    for row, name, symbol, color in [
        (diagonal_best, "best sampled constant mixture", "x", DIAGONAL_MIN_COLOR),
        (boundary_best, "best p0=0 boundary point", "cross", BOUNDARY_MIN_COLOR),
        (proportional, "proportional", "circle", PROPORTIONAL_COLOR),
    ]:
        fig.add_trace(
            go.Scatter3d(
                x=[row["p0"]],
                y=[row["p1"]],
                z=[row["bpb"]],
                mode="markers",
                marker={"symbol": symbol, "size": 7, "color": color, "line": {"color": "white", "width": 1.2}},
                text=[f"{name}<br>p0={row['p0']:.4f}<br>p1={row['p1']:.4f}<br>BPB={row['bpb']:.6f}"],
                hoverinfo="text",
                name=name,
            )
        )
    fig.update_layout(
        template="plotly_white",
        width=1100,
        height=900,
        paper_bgcolor=PAPER_BACKGROUND,
        font={"family": SERIF_FONT, "size": 17, "color": PAPER_TEXT},
        title={"text": "StarCoder response under 80/20 WSD", "x": 0.5, "font": {"size": 26}},
        legend={"x": 0.01, "y": 0.97, "bgcolor": "rgba(255,255,255,0.9)"},
        scene=_scene_layout(max(0.88, color_min - 0.03), z_max),
        margin={"l": 20, "r": 80, "t": 75, "b": 30},
    )
    _write_figure(fig, output_dir / "starcoder_wsd80_surface")


def _render_comparison(surfaces: list[Surface], output_dir: Path) -> None:
    all_bpb = np.concatenate([surface.frame["bpb"].to_numpy(dtype=float) for surface in surfaces])
    color_min = float(all_bpb.min())
    color_max = float(np.quantile(all_bpb, 0.94))
    z_min = max(0.88, color_min - 0.03)
    z_max = float(max(2.6, all_bpb.max() * 1.02))
    fig = make_subplots(
        rows=1,
        cols=3,
        specs=[[{"type": "scene"}, {"type": "scene"}, {"type": "scene"}]],
        subplot_titles=[surface.name for surface in surfaces],
        horizontal_spacing=0.015,
    )
    for index, surface in enumerate(surfaces, start=1):
        scene = "scene" if index == 1 else f"scene{index}"
        before = len(fig.data)
        _add_surface(
            fig,
            surface,
            scene=scene,
            color_min=color_min,
            color_max=color_max,
            show_scale=index == 3,
            show_legend=index == 1,
        )
        # make_subplots needs explicit axis assignment after adding traces by scene name.
        for trace in fig.data[before:]:
            trace.update(scene=scene)
    scenes = {_scene: _scene_layout(z_min, z_max) for _scene in ("scene", "scene2", "scene3")}
    fig.update_layout(
        template="plotly_white",
        width=1800,
        height=700,
        paper_bgcolor=PAPER_BACKGROUND,
        font={"family": SERIF_FONT, "size": 14, "color": PAPER_TEXT},
        title={
            "text": "StarCoder phase-mixture surfaces across learning-rate schedules",
            "x": 0.5,
            "font": {"size": 24},
        },
        legend={"x": 0.01, "y": 0.93, "bgcolor": "rgba(255,255,255,0.9)"},
        margin={"l": 10, "r": 80, "t": 85, "b": 25},
        **scenes,
    )
    _write_figure(fig, output_dir / "starcoder_schedule_surface_comparison", static_scale=2)


def _render_slices(wsd80: pd.DataFrame, output_dir: Path) -> None:
    diagonal = wsd80[np.isclose(wsd80["phase_0_starcoder"], wsd80["phase_1_starcoder"], atol=1e-10)].copy()
    diagonal["aggregate"] = diagonal["phase_0_starcoder"]
    diagonal["label"] = "Constant mixture"
    boundary = wsd80[np.isclose(wsd80["phase_0_starcoder"], 0.0, atol=1e-10)].copy()
    boundary["aggregate"] = PHASE_1_FRACTION * boundary["phase_1_starcoder"]
    boundary["label"] = "Code only in final 20%"

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=["Aggregate exposure slices", "Fixed-aggregate ordering contrasts"],
        horizontal_spacing=0.12,
    )
    for frame, color, symbol in [
        (diagonal.sort_values("aggregate"), "#E67E22", "circle"),
        (boundary.sort_values("aggregate"), "#2C7FB8", "diamond"),
    ]:
        fig.add_trace(
            go.Scatter(
                x=frame["aggregate"],
                y=frame["wsd80_bpb"],
                mode="lines+markers",
                name=str(frame["label"].iloc[0]),
                marker={"color": color, "symbol": symbol, "size": 8},
                line={"color": color, "width": 2},
                customdata=np.column_stack([frame["phase_0_starcoder"], frame["phase_1_starcoder"]]),
                hovertemplate="aggregate=%{x:.4f}<br>p0=%{customdata[0]:.4f}<br>p1=%{customdata[1]:.4f}<br>BPB=%{y:.6f}<extra></extra>",
            ),
            row=1,
            col=1,
        )

    contrast_groups = [
        ("aggregate=0.140704", [28, 25, 15]),
        ("aggregate=0.170050", [30, 27, 18]),
    ]
    text_positions = [
        ["top right", "top left", "bottom left"],
        ["bottom left", "bottom right", "top right"],
    ]
    for group_index, (label, ranks) in enumerate(contrast_groups):
        frame = wsd80.set_index("selection_rank").loc[ranks].sort_values("ordering_contrast_p1_minus_p0")
        fig.add_trace(
            go.Scatter(
                x=frame["ordering_contrast_p1_minus_p0"],
                y=frame["wsd80_bpb"],
                mode="lines+markers+text",
                text=["early only", "tied", "late only"],
                textposition=text_positions[group_index],
                name=label,
                marker={"size": 9},
                line={"width": 2},
                customdata=np.column_stack([frame["phase_0_starcoder"], frame["phase_1_starcoder"]]),
                hovertemplate="p1-p0=%{x:.4f}<br>p0=%{customdata[0]:.4f}<br>p1=%{customdata[1]:.4f}<br>BPB=%{y:.6f}<extra></extra>",
            ),
            row=1,
            col=2,
        )
    fig.update_xaxes(title_text="Aggregate StarCoder share", range=[-0.01, 0.55], row=1, col=1)
    fig.update_xaxes(title_text="Ordering contrast p1 - p0", range=[-0.32, 1.02], row=1, col=2)
    fig.update_yaxes(title_text="Dolma 100 Programming Languages BPB", row=1, col=1)
    fig.update_yaxes(title_text="BPB", row=1, col=2)
    fig.update_layout(
        template="plotly_white",
        width=1500,
        height=620,
        paper_bgcolor=PAPER_BACKGROUND,
        plot_bgcolor=PAPER_BACKGROUND,
        font={"family": SERIF_FONT, "size": 16, "color": PAPER_TEXT},
        title={"text": "80/20 WSD: exposure and ordering diagnostics", "x": 0.5, "font": {"size": 24}},
        legend={"orientation": "h", "x": 0.5, "xanchor": "center", "y": -0.18},
        margin={"l": 80, "r": 40, "t": 90, "b": 120},
    )
    _write_figure(fig, output_dir / "starcoder_wsd80_slices_and_fixed_aggregate", static_scale=2)


def _render_repeats(repeats: pd.DataFrame, repeat_summary: dict[str, object], output_dir: Path) -> None:
    schedule_order = ["constant", "boundary", "global"]
    schedule_colors = {
        "constant": DIAGONAL_MIN_COLOR,
        "boundary": BOUNDARY_MIN_COLOR,
        "global": GLOBAL_MIN_COLOR,
    }
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=["Matched schedules across five seeds", "Paired BPB differences (95% t interval)"],
        column_widths=[0.54, 0.46],
        horizontal_spacing=0.13,
    )
    pivot = repeats.pivot(index="data_seed", columns="schedule", values="bpb")
    x_positions = list(range(len(schedule_order)))
    for seed, values in pivot.iterrows():
        fig.add_trace(
            go.Scatter(
                x=x_positions,
                y=[values[schedule] for schedule in schedule_order],
                mode="lines+markers",
                line={"color": "rgba(80,95,115,0.34)", "width": 1.5},
                marker={
                    "size": 7,
                    "color": [schedule_colors[schedule] for schedule in schedule_order],
                    "line": {"color": "white", "width": 0.7},
                },
                customdata=[seed] * len(schedule_order),
                hovertemplate="seed=%{customdata}<br>BPB=%{y:.6f}<extra></extra>",
                showlegend=False,
            ),
            row=1,
            col=1,
        )
    means = [float(pivot[schedule].mean()) for schedule in schedule_order]
    fig.add_trace(
        go.Scatter(
            x=x_positions,
            y=means,
            mode="markers",
            marker={
                "symbol": "diamond",
                "size": 13,
                "color": [schedule_colors[schedule] for schedule in schedule_order],
                "line": {"color": PAPER_TEXT, "width": 1.3},
            },
            name="five-seed mean",
            hovertemplate="mean BPB=%{y:.6f}<extra></extra>",
        ),
        row=1,
        col=1,
    )

    contrast_labels = {
        "global - constant": "Off-diagonal - constant",
        "boundary - constant": "p0=0 boundary - constant",
        "global - boundary": "Off-diagonal - p0=0 boundary",
    }
    contrasts = repeat_summary["paired_contrasts"]
    for contrast in contrasts:
        mean = float(contrast["mean_delta_bpb"])
        fig.add_trace(
            go.Scatter(
                x=[mean],
                y=[contrast_labels[str(contrast["contrast"])]],
                mode="markers",
                marker={
                    "size": 12,
                    "color": GLOBAL_MIN_COLOR if float(contrast["ci_95_high"]) < 0 else BOUNDARY_MIN_COLOR,
                    "line": {"color": "white", "width": 1.0},
                },
                error_x={
                    "type": "data",
                    "symmetric": False,
                    "array": [float(contrast["ci_95_high"]) - mean],
                    "arrayminus": [mean - float(contrast["ci_95_low"])],
                    "thickness": 2,
                    "width": 7,
                },
                customdata=[
                    [
                        contrast["paired_t_two_sided_p"],
                        contrast["sign_flip_two_sided_p"],
                        contrast["a_lower_b_count"],
                    ]
                ],
                hovertemplate=(
                    "mean delta=%{x:+.6f}<br>paired t p=%{customdata[0]:.4f}<br>"
                    "exact sign-flip p=%{customdata[1]:.4f}<br>A lower in %{customdata[2]}/5 seeds<extra></extra>"
                ),
                showlegend=False,
            ),
            row=1,
            col=2,
        )
    fig.add_vline(x=0.0, line={"color": PAPER_TEXT, "width": 1.4, "dash": "dash"}, row=1, col=2)
    fig.update_xaxes(
        tickmode="array",
        tickvals=x_positions,
        ticktext=[REPEAT_SCHEDULE_LABELS[schedule] for schedule in schedule_order],
        tickangle=-12,
        row=1,
        col=1,
    )
    fig.update_yaxes(title_text="Dolma 100 Programming Languages BPB", row=1, col=1)
    fig.update_xaxes(title_text="Paired delta BPB (negative means first schedule is better)", row=1, col=2)
    fig.update_layout(
        template="plotly_white",
        width=1500,
        height=650,
        paper_bgcolor=PAPER_BACKGROUND,
        plot_bgcolor=PAPER_BACKGROUND,
        font={"family": SERIF_FONT, "size": 16, "color": PAPER_TEXT},
        title={"text": "StarCoder 80/20 WSD: paired schedule repeats", "x": 0.5, "font": {"size": 25}},
        legend={"x": 0.02, "y": 0.98, "bgcolor": "rgba(255,255,255,0.85)"},
        margin={"l": 90, "r": 45, "t": 95, "b": 105},
    )
    _write_figure(fig, output_dir / "starcoder_wsd80_paired_repeats", static_scale=2)


def _best_summary(surface: Surface) -> dict[str, object]:
    frame = surface.frame
    best = frame.loc[frame["bpb"].idxmin()]
    diagonal = frame[np.isclose(frame["p0"], frame["p1"], atol=1e-10)]
    diagonal_best = diagonal.loc[diagonal["bpb"].idxmin()]
    boundary = frame[np.isclose(frame["p0"], 0.0, atol=1e-10)]
    boundary_best = boundary.loc[boundary["bpb"].idxmin()]
    phase_1_fraction = 1.0 - surface.phase_0_fraction
    return {
        "name": surface.name,
        "n": len(frame),
        "best_observed": {
            "p0": float(best["p0"]),
            "p1": float(best["p1"]),
            "aggregate": float(surface.phase_0_fraction * best["p0"] + phase_1_fraction * best["p1"]),
            "bpb": float(best["bpb"]),
        },
        "best_sampled_diagonal": {
            "p": float(diagonal_best["p0"]),
            "bpb": float(diagonal_best["bpb"]),
            "gap_from_best": float(diagonal_best["bpb"] - best["bpb"]),
            "n_diagonal": len(diagonal),
        },
        "best_sampled_p0_zero": {
            "p1": float(boundary_best["p1"]),
            "aggregate": float(phase_1_fraction * boundary_best["p1"]),
            "bpb": float(boundary_best["bpb"]),
            "gap_from_best": float(boundary_best["bpb"] - best["bpb"]),
            "n_boundary": len(boundary),
        },
    }


def _round_key(frame: pd.DataFrame, p0: str, p1: str) -> pd.Series:
    return pd.Series(list(zip(frame[p0].round(9), frame[p1].round(9), strict=True)), index=frame.index)


def _overlap_summary(wsd80: pd.DataFrame, surfaces: list[Surface]) -> dict[str, object]:
    new = wsd80[["phase_0_starcoder", "phase_1_starcoder", "wsd80_bpb"]].copy()
    new["key"] = _round_key(new, "phase_0_starcoder", "phase_1_starcoder")
    result: dict[str, object] = {}
    for surface in surfaces[:2]:
        old = surface.frame.copy()
        old["key"] = _round_key(old, "p0", "p1")
        matched = new.merge(old[["key", "bpb"]], on="key", validate="one_to_one")
        delta = matched["wsd80_bpb"] - matched["bpb"]
        frontier = matched[matched["bpb"].le(0.95)]
        result[surface.name] = {
            "n_exact_matches": len(matched),
            "mean_wsd80_minus_historical_bpb": float(delta.mean()),
            "median_wsd80_minus_historical_bpb": float(delta.median()),
            "spearman": float(matched[["wsd80_bpb", "bpb"]].corr(method="spearman").iloc[0, 1]),
            "historical_bpb_le_0p95": {
                "n": len(frontier),
                "mean_wsd80_minus_historical_bpb": float((frontier["wsd80_bpb"] - frontier["bpb"]).mean()),
            },
        }
    return result


def _repeat_noise_summary() -> dict[str, object]:
    frame = pd.read_csv(REPEAT_CSV)
    final_step = int(frame["latest_step"].dropna().max())
    frame = frame[frame["latest_step"].eq(final_step) & frame[HISTORICAL_TARGET].notna()].copy()
    result: dict[str, object] = {}
    for anchor, rows in frame.groupby("anchor_id"):
        result[str(anchor)] = {
            "mean": float(rows[HISTORICAL_TARGET].mean()),
            "sd": float(rows[HISTORICAL_TARGET].std(ddof=1)),
            "n": len(rows),
        }
    return result


def _fixed_aggregate_summary(wsd80: pd.DataFrame) -> list[dict[str, object]]:
    indexed = wsd80.set_index("selection_rank")
    groups = [
        (0.14070404240108944, {"early_only": 28, "tied": 25, "late_only": 15}),
        (0.17005, {"early_only": 30, "tied": 27, "late_only": 18}),
    ]
    result = []
    for aggregate, ranks in groups:
        values = {label: float(indexed.loc[rank, "wsd80_bpb"]) for label, rank in ranks.items()}
        result.append(
            {
                "aggregate_starcoder_share": aggregate,
                **values,
                "late_minus_tied": values["late_only"] - values["tied"],
                "early_minus_tied": values["early_only"] - values["tied"],
            }
        )
    return result


def _surface_table_row(surface: dict[str, object]) -> str:
    best = surface["best_observed"]
    diagonal = surface["best_sampled_diagonal"]
    cells = [
        str(surface["name"]),
        str(surface["n"]),
        f"`({best['p0']:.4f}, {best['p1']:.4f})`",
        f"{best['bpb']:.6f}",
        f"{diagonal['p']:.4f}",
        f"{diagonal['bpb']:.6f}",
        f"{diagonal['gap_from_best']:.6f}",
    ]
    return "| " + " | ".join(cells) + " |"


def _fixed_table_row(contrast: dict[str, object]) -> str:
    cells = [
        f"{contrast['aggregate_starcoder_share']:.6f}",
        f"{contrast['early_only']:.6f}",
        f"{contrast['tied']:.6f}",
        f"{contrast['late_only']:.6f}",
        f"{contrast['late_minus_tied']:+.6f}",
        f"{contrast['early_minus_tied']:+.6f}",
    ]
    return "| " + " | ".join(cells) + " |"


def _repeat_schedule_table_row(schedule: str, summary: dict[str, object]) -> str:
    phase_0, phase_1 = REPEAT_SCHEDULES[schedule]
    cells = [
        REPEAT_SCHEDULE_LABELS[schedule],
        f"`({phase_0:.4f}, {phase_1:.4f})`",
        str(summary["n"]),
        f"{summary['mean_bpb']:.6f}",
        f"{summary['sd_bpb']:.6f}",
    ]
    return "| " + " | ".join(cells) + " |"


def _repeat_contrast_table_row(contrast: dict[str, object]) -> str:
    cells = [
        str(contrast["contrast"]),
        f"{contrast['mean_delta_bpb']:+.6f}",
        f"[{contrast['ci_95_low']:+.6f}, {contrast['ci_95_high']:+.6f}]",
        f"{contrast['paired_t_two_sided_p']:.4f}",
        f"{contrast['sign_flip_two_sided_p']:.4f}",
        f"{contrast['a_lower_b_count']}/{contrast['n_pairs']}",
    ]
    return "| " + " | ".join(cells) + " |"


def _overlap_bullet(name: str, overlap: dict[str, object]) -> str:
    frontier = overlap["historical_bpb_le_0p95"]
    return (
        f"- Exact overlap with {name}: `{overlap['n_exact_matches']}` coordinates; "
        f"all-coordinate mean delta `{overlap['mean_wsd80_minus_historical_bpb']:+.6f}` and "
        f"Spearman `{overlap['spearman']:.3f}`. Among the `{frontier['n']}` historically strong "
        f"coordinates with BPB at most 0.95, 80/20 WSD is worse by "
        f"`{frontier['mean_wsd80_minus_historical_bpb']:+.6f}` on average."
    )


def _write_report(summary: dict[str, object], output_dir: Path) -> None:
    surfaces = summary["surfaces"]
    wsd80 = surfaces[2]
    fixed = summary["fixed_aggregate_contrasts"]
    overlap = summary["exact_coordinate_overlaps"]
    repeats = summary["paired_schedule_repeats"]
    repeat_schedules = repeats["schedule_summaries"]
    repeat_contrasts = repeats["paired_contrasts"]
    contrast_by_name = {contrast["contrast"]: contrast for contrast in repeat_contrasts}
    global_constant = contrast_by_name["global - constant"]
    boundary_constant = contrast_by_name["boundary - constant"]
    global_boundary = contrast_by_name["global - boundary"]
    lines = [
        "# StarCoder 80/20 WSD surface analysis",
        "",
        "## Design",
        "",
        "- Original RegMix 60M architecture, 1B materialized tokens, and the historical datasets.",
        "- Phase 0 is 80% of training at the WSD plateau; phase 1 is the final 20% cosine decay.",
        "- All 64 coordinates completed with one shared configured seed.",
        "- Surfaces use linear triangulation; no smoothed or model-predicted optimum is called observed.",
        "- Target: Dolma 100 Programming Languages BPB, lower is better.",
        "",
        "## Observed minima",
        "",
        "| Schedule | Points | Best observed `(p0,p1)` | Best BPB | Best sampled constant | Constant BPB | Gap |",
        "|---|---:|---:|---:|---:|---:|---:|",
        *[_surface_table_row(surface) for surface in surfaces],
        "",
        (
            "The historical diagonal grids were sparse, so their diagonal gaps are not estimates of "
            "the true one-phase optimum. The new 80/20 panel deliberately sampled "
            f"{wsd80['best_sampled_diagonal']['n_diagonal']} diagonal points."
        ),
        "",
        (
            "For 80/20 WSD, the best observed point is off-diagonal, but its advantage over the best "
            f"sampled constant mixture is only {wsd80['best_sampled_diagonal']['gap_from_best']:.6f} BPB. "
            f"The best `p0=0` point is `p1={wsd80['best_sampled_p0_zero']['p1']:.2f}` with BPB "
            f"{wsd80['best_sampled_p0_zero']['bpb']:.6f}, only "
            f"{wsd80['best_sampled_p0_zero']['gap_from_best']:.6f} above the observed minimum."
        ),
        (
            "The one-seed surface alone does not resolve these differences. The paired repeat panel "
            "below is the inferential basis for comparing the three selected schedules."
        ),
        "",
        "## Paired five-seed schedule comparison",
        "",
        "| Schedule | `(p0,p1)` | Seeds | Mean BPB | Marginal SD |",
        "|---|---:|---:|---:|---:|",
        *[
            _repeat_schedule_table_row(schedule, repeat_schedules[schedule])
            for schedule in ("global", "boundary", "constant")
        ],
        "",
        (
            "| Paired contrast | Mean delta | 95% paired t CI | Paired t p (two-sided) | "
            "Exact sign-flip p (two-sided) | First lower |"
        ),
        "|---|---:|---:|---:|---:|---:|",
        *[_repeat_contrast_table_row(contrast) for contrast in repeat_contrasts],
        "",
        (
            "Both phased schedules beat the best sampled constant candidate in all five matched seeds. "
            f"The off-diagonal candidate lowers BPB by {-global_constant['mean_delta_bpb']:.6f} on average "
            f"(95% paired t CI [{global_constant['ci_95_low']:+.6f}, "
            f"{global_constant['ci_95_high']:+.6f}]); the `p0=0` boundary candidate lowers it by "
            f"{-boundary_constant['mean_delta_bpb']:.6f}."
        ),
        (
            "With only five pairs, the exact two-sided sign-flip test has coarse resolution: five "
            "same-direction differences give p=0.0625. The directional one-sided exact "
            f"p-value is {global_constant['sign_flip_one_sided_a_lower_p']:.5f} for off-diagonal versus "
            f"constant and {boundary_constant['sign_flip_one_sided_a_lower_p']:.5f} for boundary versus "
            "constant."
        ),
        (
            "The off-diagonal surface minimum and `p0=0` boundary remain indistinguishable: their mean "
            f"difference is {global_boundary['mean_delta_bpb']:+.6f} BPB with 95% CI "
            f"[{global_boundary['ci_95_low']:+.6f}, {global_boundary['ci_95_high']:+.6f}] and paired "
            f"t p={global_boundary['paired_t_two_sided_p']:.3f}."
        ),
        (
            "Seed means across the three schedules span "
            f"{repeats['seed_common_mode']['range']:.6f} BPB. This large common-mode shift explains why "
            "paired contrasts are much more informative than independent comparisons of marginal SDs."
        ),
        "",
        "## Fixed-aggregate ordering contrasts",
        "",
        "| Aggregate StarCoder | Early only | Tied | Late only | Late - tied | Early - tied |",
        "|---:|---:|---:|---:|---:|---:|",
        *[_fixed_table_row(contrast) for contrast in fixed],
        "",
        (
            "Timing matters strongly conditional on total exposure, but not monotonically. At aggregate "
            f"0.1407, final-20%-only code beats tied by {-fixed[0]['late_minus_tied']:.6f} BPB. At "
            f"aggregate 0.1701, it overshoots and loses by {fixed[1]['late_minus_tied']:.6f}."
        ),
        (
            "Early-only code is severely worse at both doses. This is evidence for an "
            "aggregate-by-recency interaction, not a universal late-data bonus."
        ),
        "",
        "## Comparison with historical surfaces",
        "",
        _overlap_bullet("cosine", overlap["Cosine, 50/50 phases"]),
        _overlap_bullet("50/50 WSD", overlap["WSD, 50/50 phases"]),
        "",
        (
            "All-coordinate means are dominated by catastrophic high-code schedules, which 80/20 WSD "
            "makes less bad. They are not evidence that 80/20 WSD improves the optimum region."
        ),
        (
            "Absolute cross-generation shifts are not clean schedule-only effects: the panels used "
            "different seeds and different generations of the training/eval path."
        ),
        "",
        "## Hypotheses",
        "",
        (
            "1. **A phased 80/20 WSD schedule beats the best sampled constant schedule:** supported for "
            "the fixed candidates. Both phased schedules win 5/5 paired seeds by about 0.005 BPB. This "
            "is not yet a proof about the continuous policy-class suprema because only the selected "
            "constant `p=0.30` was repeated."
        ),
        (
            "2. **Later training has a distinct response to code:** supported. Fixed-dose triplets show "
            "large order effects and strong early/late asymmetry."
        ),
        (
            "3. **Gradient conflict causes the effect:** not directly tested. Final losses establish path "
            "dependence, not gradient directions."
        ),
        (
            "4. **The interior off-diagonal minimum is better than the `p0=0` boundary:** not supported. "
            "Their five-seed paired difference is only about 0.0004 BPB and changes sign across seeds."
        ),
        (
            "5. **Larger models or lower token-per-parameter ratios increase phase benefit:** not tested. "
            "This motivates the planned `(N,D)` comparison."
        ),
        (
            "6. **Residual uncertainty about the one-phase optimum:** a local constant-only refinement "
            "around `p=0.25-0.35` would test whether the finite diagonal grid missed a better constant "
            "mixture. The current evidence shows the sampled `p=0.30` candidate is worse, not that every possible "
            "constant schedule."
        ),
        "",
    ]
    (output_dir / "report.md").write_text("\n".join(lines))


def main() -> None:
    args = _parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    wsd80 = _collect_wsd80()
    wsd80.to_csv(args.output_dir / "wsd80_observed_metrics.csv", index=False)
    repeats = _collect_wsd80_repeats(wsd80)
    repeats.to_csv(args.output_dir / "wsd80_repeat_observations.csv", index=False)
    repeat_summary, raw_contrasts = _paired_repeat_summary(repeats)
    raw_contrasts.to_csv(args.output_dir / "wsd80_paired_contrasts.csv", index=False)
    surfaces = _load_surfaces(wsd80)
    summary = {
        "wsd80_wandb_group": WSD80_GROUP,
        "target": WSD80_TARGET,
        "surfaces": [_best_summary(surface) for surface in surfaces],
        "paired_schedule_repeats": repeat_summary,
        "fixed_aggregate_contrasts": _fixed_aggregate_summary(wsd80),
        "exact_coordinate_overlaps": _overlap_summary(wsd80, surfaces),
        "historical_repeat_noise": _repeat_noise_summary(),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    _render_wsd80_surface(surfaces[2], args.output_dir)
    _render_comparison(surfaces, args.output_dir)
    _render_slices(wsd80, args.output_dir)
    _render_repeats(repeats, repeat_summary, args.output_dir)
    _write_report(summary, args.output_dir)
    print(args.output_dir)


if __name__ == "__main__":
    main()
