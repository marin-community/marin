# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E501

# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "matplotlib",
#   "numpy",
#   "pandas",
#   "pillow",
#   "plotly",
# ]
# ///

"""Animate repetition-conditioned StarCoder WSD80 optima across token horizons."""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D
from PIL import Image
from plot_starcoder_wsd80_dense_horizon_replay_gain_20260811 import (
    DEFAULT_COVERAGE_OBSERVATIONS,
    DEFAULT_DESIGN,
    DEFAULT_SELECTED_POLICIES,
    EXPECTED_BLOCKS,
    EXPECTED_CELLS,
    SUPPORT_LABELS,
    SUPPORT_MARKER_LABELS,
    SUPPORT_ORDER,
    load_summary,
)

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = (
    SCRIPT_DIR / "reference_outputs" / "starcoder_wsd80_dense_horizon_replay_optimum_animations_20260811"
)

EXPECTED_COORDINATES_PER_BLOCK = 125
FRAME_DURATIONS_MS = (1800, 1800, 1800, 3000)
SURFACE_MIN = -0.012
SURFACE_MAX = 0.050

PAPER = "#F7F3E8"
PANEL = "#FFFDF8"
INK = "#17324D"
MUTED = "#657786"
GRID = "#D8D1C2"
TIED_COLOR = "#1F5A85"
TWO_PHASE_COLOR = "#D85F3D"
ALIAS_COLOR = "#6E7D89"
GAIN_POSITIVE = "#238B57"
GAIN_NEGATIVE = "#C54D3C"
CMAP = mpl.colormaps["RdYlGn_r"]
SURFACE_NORM = mpl.colors.TwoSlopeNorm(vmin=SURFACE_MIN, vcenter=0.0, vmax=SURFACE_MAX)

MASTER_GIF_FILENAME = "starcoder_wsd80_all_repetition_regimes_optimum_motion.gif"
MASTER_ROW_LABELS = {
    "full": "Full StarCoder pool\n(no forced replay)",
    "m0125": "0.125x StarCoder\nrepetition target",
    "m025": "0.25x StarCoder\nrepetition target",
    "m050": "0.5x StarCoder\nrepetition target",
    "m100": "1x StarCoder\nrepetition target",
    "m200": "2x StarCoder\nrepetition target",
    "m400": "4x StarCoder\nrepetition target",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selected-policies", type=Path, default=DEFAULT_SELECTED_POLICIES)
    parser.add_argument("--coverage-observations", type=Path, default=DEFAULT_COVERAGE_OBSERVATIONS)
    parser.add_argument("--design", type=Path, default=DEFAULT_DESIGN)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def load_data(
    selected_path: Path,
    coverage_path: Path,
    design_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and validate the common dense surfaces and selected minima."""
    summary = load_summary(selected_path, coverage_path, design_path)
    observations = pd.read_csv(coverage_path)
    required = {
        "bpb",
        "cell_id",
        "coordinate_id",
        "is_alias",
        "phase_0_starcoder",
        "phase_1_starcoder",
        "run_name",
        "support_id",
    }
    missing = required - set(observations.columns)
    if missing:
        raise ValueError(f"Coverage table is missing fields: {sorted(missing)}")
    expected_rows = EXPECTED_BLOCKS * EXPECTED_COORDINATES_PER_BLOCK
    if len(observations) != expected_rows:
        raise ValueError(f"Expected {expected_rows} observations, found {len(observations)}")
    block_counts = observations.groupby(["cell_id", "support_id"]).size()
    if len(block_counts) != EXPECTED_BLOCKS or not block_counts.eq(EXPECTED_COORDINATES_PER_BLOCK).all():
        raise ValueError("Every horizon-support block must contain the same 125 coordinates")
    if observations.duplicated(["cell_id", "support_id", "coordinate_id"]).any():
        raise ValueError("Coverage rows are not unique by block and coordinate")
    coordinates = observations[["phase_0_starcoder", "phase_1_starcoder", "bpb"]].to_numpy(dtype=float)
    if not np.isfinite(coordinates).all():
        raise ValueError("Coverage table contains non-finite coordinates or BPB")
    for column in ("phase_0_starcoder", "phase_1_starcoder"):
        if not observations[column].between(0.0, 1.0).all():
            raise ValueError(f"{column} must lie in [0,1]")
    if observations["is_alias"].dtype != bool:
        observations["is_alias"] = observations["is_alias"].astype(str).str.lower().eq("true")
    return observations, summary


def _configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "axes.edgecolor": INK,
            "axes.facecolor": PANEL,
            "axes.labelcolor": INK,
            "font.family": "sans-serif",
            "font.sans-serif": ["Avenir Next", "Helvetica Neue", "DejaVu Sans"],
            "text.color": INK,
            "text.usetex": False,
            "xtick.color": INK,
            "ytick.color": INK,
        }
    )


def _render_surface(
    axis: plt.Axes,
    observations: pd.DataFrame,
    support_summary: pd.DataFrame,
    frame_index: int,
    *,
    compact: bool = False,
    show_x_labels: bool = True,
) -> None:
    current = support_summary.iloc[frame_index]
    block = observations.loc[
        observations["cell_id"].eq(current["cell_id"]) & observations["support_id"].eq(current["support_id"])
    ].copy()
    x = block["phase_0_starcoder"].to_numpy(dtype=float)
    y = block["phase_1_starcoder"].to_numpy(dtype=float)
    relative_bpb = block["bpb"].to_numpy(dtype=float) - float(current["tied_bpb"])
    clipped = np.clip(relative_bpb, SURFACE_MIN, SURFACE_MAX)
    triangulation = mtri.Triangulation(x, y)
    levels = np.linspace(SURFACE_MIN, SURFACE_MAX, 18)
    axis.tricontourf(
        triangulation,
        clipped,
        levels=levels,
        cmap=CMAP,
        norm=SURFACE_NORM,
        extend="both",
        alpha=0.86,
    )

    independent = ~block["is_alias"].to_numpy(dtype=bool)
    axis.scatter(
        x[independent],
        y[independent],
        c=clipped[independent],
        cmap=CMAP,
        norm=SURFACE_NORM,
        s=11 if compact else 23,
        edgecolor=PANEL,
        linewidth=0.35 if compact else 0.6,
        zorder=3,
    )
    if (~independent).any():
        axis.scatter(
            x[~independent],
            y[~independent],
            facecolors="none",
            edgecolors=ALIAS_COLOR,
            marker="o",
            s=14 if compact else 29,
            linewidth=0.5 if compact else 0.8,
            alpha=0.68,
            zorder=3,
        )

    axis.plot(
        [0.0, 1.0],
        [0.0, 1.0],
        color=INK,
        linewidth=0.75 if compact else 1.15,
        linestyle="--",
        alpha=0.50,
        zorder=2,
    )
    history = support_summary.iloc[: frame_index + 1]
    axis.plot(
        history["tied_p0"],
        history["tied_p1"],
        color=TIED_COLOR,
        linewidth=1.25 if compact else 2.0,
        linestyle=(0, (3, 3)),
        alpha=0.72,
        zorder=4,
    )
    axis.plot(
        history["untied_p0"],
        history["untied_p1"],
        color=TWO_PHASE_COLOR,
        linewidth=1.25 if compact else 2.0,
        linestyle=(0, (3, 3)),
        alpha=0.72,
        zorder=4,
    )
    if frame_index:
        prior = history.iloc[:-1]
        axis.scatter(
            prior["tied_p0"],
            prior["tied_p1"],
            marker="X",
            s=29 if compact else 65,
            color=TIED_COLOR,
            edgecolor=PANEL,
            linewidth=0.65 if compact else 1.0,
            alpha=0.55,
            zorder=4,
        )
        axis.scatter(
            prior["untied_p0"],
            prior["untied_p1"],
            marker="D",
            s=24 if compact else 55,
            color=TWO_PHASE_COLOR,
            edgecolor=PANEL,
            linewidth=0.65 if compact else 1.0,
            alpha=0.55,
            zorder=4,
        )
    axis.scatter(
        [current["tied_p0"]],
        [current["tied_p1"]],
        marker="X",
        s=90 if compact else 190,
        color=TIED_COLOR,
        edgecolor=PANEL,
        linewidth=1.15 if compact else 1.9,
        zorder=6,
    )
    axis.scatter(
        [current["untied_p0"]],
        [current["untied_p1"]],
        marker="D",
        s=68 if compact else 145,
        color=TWO_PHASE_COLOR,
        edgecolor=PANEL,
        linewidth=1.15 if compact else 1.9,
        zorder=6,
    )
    axis.set_xlim(-0.02, 1.02)
    axis.set_ylim(-0.02, 1.02)
    axis.set_aspect("equal", adjustable="box")
    if compact:
        axis.set_xticks((0.0, 0.5, 1.0))
        axis.set_yticks((0.0, 0.5, 1.0))
        axis.tick_params(labelsize=7.3)
        if show_x_labels:
            axis.set_xlabel("Phase 0 StarCoder weight", fontsize=8.5, labelpad=2)
        else:
            axis.tick_params(axis="x", labelbottom=False)
    else:
        axis.set_xlabel("Phase 0 StarCoder weight")
        axis.set_ylabel("Phase 1 StarCoder weight")
        axis.set_title("Observed response and optimum movement", color=INK, fontsize=14, fontweight="semibold", pad=10)
    axis.grid(True, color=GRID, linewidth=0.45 if compact else 0.65, alpha=0.66)


def _render_scaling_track(
    axis: plt.Axes,
    support_summary: pd.DataFrame,
    frame_index: int,
    *,
    compact: bool = False,
    show_x_labels: bool = True,
) -> None:
    x_all = support_summary["materialized_tokens_b"].to_numpy(dtype=float)
    revealed = support_summary.iloc[: frame_index + 1]
    x = revealed["materialized_tokens_b"].to_numpy(dtype=float)
    tied = revealed["tied_bpb"].to_numpy(dtype=float)
    untied = revealed["untied_bpb"].to_numpy(dtype=float)

    axis.plot(
        x,
        tied,
        color=TIED_COLOR,
        linewidth=1.65 if compact else 2.5,
        marker="X",
        markersize=5.6 if compact else 9.0,
        label="Tied grid minimum",
    )
    axis.plot(
        x,
        untied,
        color=TWO_PHASE_COLOR,
        linewidth=1.65 if compact else 2.5,
        marker="D",
        markersize=4.9 if compact else 7.8,
        label="Untied grid minimum",
    )
    axis.scatter(
        [x[-1]],
        [tied[-1]],
        marker="X",
        s=70 if compact else 145,
        color=TIED_COLOR,
        edgecolor=PANEL,
        linewidth=0.9 if compact else 1.5,
        zorder=5,
    )
    axis.scatter(
        [x[-1]],
        [untied[-1]],
        marker="D",
        s=58 if compact else 120,
        color=TWO_PHASE_COLOR,
        edgecolor=PANEL,
        linewidth=0.9 if compact else 1.5,
        zorder=5,
    )

    all_bpb = support_summary[["tied_bpb", "untied_bpb"]].to_numpy(dtype=float)
    span = float(all_bpb.max() - all_bpb.min())
    padding = max(0.008, 0.10 * span)
    axis.set_ylim(float(all_bpb.min()) - padding, float(all_bpb.max()) + padding)
    axis.set_xscale("log")
    axis.set_xlim(float(x_all.min()) / 1.09, float(x_all.max()) * 1.09)
    axis.set_xticks(x_all)
    axis.set_xticklabels([f"{value:.2f}B" for value in x_all])
    axis.xaxis.set_minor_locator(mpl.ticker.NullLocator())
    if compact:
        axis.tick_params(labelsize=7.5)
        if show_x_labels:
            axis.set_xlabel("Materialized training tokens D", fontsize=8.5, labelpad=2)
        else:
            axis.tick_params(axis="x", labelbottom=False)
    else:
        axis.set_xlabel("Materialized training tokens D")
        axis.set_ylabel("Observed Programming Languages BPB")
        axis.set_title("Global grid minima across token horizon", color=INK, fontsize=14, fontweight="semibold", pad=10)
    axis.grid(True, color=GRID, linewidth=0.5 if compact else 0.8, alpha=0.80)

    gain = float(tied[-1] - untied[-1])
    gain_color = GAIN_POSITIVE if gain > 0.0 else GAIN_NEGATIVE
    if compact:
        axis.text(
            0.985,
            0.88,
            f"gain {gain:+.5f}",
            transform=axis.transAxes,
            ha="right",
            va="top",
            color=gain_color,
            fontsize=7.8,
            fontweight="semibold",
            bbox={"boxstyle": "round,pad=0.22", "facecolor": PAPER, "edgecolor": gain_color, "alpha": 0.94},
            zorder=7,
        )
        return
    axis.text(
        0.035,
        0.045,
        f"global two-phase gain  {gain:+.5f} BPB",
        transform=axis.transAxes,
        ha="left",
        va="bottom",
        color=gain_color,
        fontsize=11.5,
        fontweight="semibold",
        bbox={"boxstyle": "round,pad=0.34", "facecolor": PAPER, "edgecolor": gain_color, "alpha": 0.94},
        zorder=7,
    )
    label_offset = (-8, 12) if frame_index == EXPECTED_CELLS - 1 else (8, 12)
    label_alignment = "right" if frame_index == EXPECTED_CELLS - 1 else "left"
    axis.annotate(
        f"tied {tied[-1]:.4f}",
        xy=(x[-1], tied[-1]),
        xytext=label_offset,
        textcoords="offset points",
        ha=label_alignment,
        color=TIED_COLOR,
        fontsize=9.5,
        fontweight="semibold",
    )
    label_offset = (-8, -18) if frame_index == EXPECTED_CELLS - 1 else (8, -18)
    axis.annotate(
        f"untied {untied[-1]:.4f}",
        xy=(x[-1], untied[-1]),
        xytext=label_offset,
        textcoords="offset points",
        ha=label_alignment,
        color=TWO_PHASE_COLOR,
        fontsize=9.5,
        fontweight="semibold",
    )


def render_frame(
    observations: pd.DataFrame,
    summary: pd.DataFrame,
    support_id: str,
    frame_index: int,
) -> Image.Image:
    """Render one measured token-horizon frame for one replay regime."""
    support_summary = summary.loc[summary["support_id"].eq(support_id)].sort_values("rung").reset_index(drop=True)
    if len(support_summary) != EXPECTED_CELLS:
        raise ValueError(f"{support_id}: expected {EXPECTED_CELLS} horizons")
    current = support_summary.iloc[frame_index]
    _configure_matplotlib()
    figure = plt.figure(figsize=(15.2, 7.9), facecolor=PAPER)
    grid = figure.add_gridspec(
        1,
        2,
        width_ratios=(1.0, 1.12),
        left=0.065,
        right=0.965,
        bottom=0.16,
        top=0.75,
        wspace=0.24,
    )
    _render_surface(figure.add_subplot(grid[0, 0]), observations, support_summary, frame_index)
    _render_scaling_track(figure.add_subplot(grid[0, 1]), support_summary, frame_index)

    gain = float(current["raw_two_phase_gain_bpb"])
    figure.suptitle(
        f"StarCoder WSD80 · {SUPPORT_LABELS[support_id]}",
        x=0.5,
        y=0.978,
        color=INK,
        fontsize=24,
        fontweight="semibold",
    )
    figure.text(
        0.5,
        0.925,
        f"Measured horizon {frame_index + 1}/4 · D={float(current['materialized_tokens_b']):.2f}B · "
        f"total-parameter TPP={float(current['total_parameter_tpp']):.2f} · "
        f"raw global two-phase gain={gain:+.5f} BPB",
        ha="center",
        va="center",
        color=MUTED,
        fontsize=12.5,
    )
    legend_handles = [
        Line2D([0], [0], color=TIED_COLOR, marker="X", linewidth=2.2, markersize=9, label="Observed tied grid minimum"),
        Line2D(
            [0],
            [0],
            color=TWO_PHASE_COLOR,
            marker="D",
            linewidth=2.2,
            markersize=8,
            label="Observed untied grid minimum",
        ),
        Line2D(
            [0],
            [0],
            color=ALIAS_COLOR,
            marker="o",
            markerfacecolor="none",
            linewidth=0,
            markersize=7,
            label="Deterministic materialization alias",
        ),
    ]
    figure.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.865),
        frameon=False,
        ncol=3,
        fontsize=10.8,
    )
    colorbar_axis = figure.add_axes((0.11, 0.075, 0.34, 0.018))
    colorbar = figure.colorbar(ScalarMappable(norm=SURFACE_NORM, cmap=CMAP), cax=colorbar_axis, orientation="horizontal")
    colorbar.set_label("Observed BPB relative to tied grid minimum (clipped)", color=INK, fontsize=10)
    colorbar.ax.tick_params(labelsize=9, colors=INK)
    figure.text(
        0.74,
        0.08,
        "Four measured horizons only; contours use linear triangulation.\n"
        "Raw selected minima are one-seed and selection-biased.",
        ha="center",
        va="center",
        color=MUTED,
        fontsize=10.2,
    )

    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=105, facecolor=PAPER)
    plt.close(figure)
    buffer.seek(0)
    image = Image.open(buffer).convert("RGB")
    return image.copy()


def render_master_frame(
    observations: pd.DataFrame,
    summary: pd.DataFrame,
    frame_index: int,
) -> Image.Image:
    """Render all seven repetition regimes at one measured token horizon."""
    current_rows = summary.loc[summary["rung"].eq(frame_index)]
    if len(current_rows) != len(SUPPORT_ORDER):
        raise ValueError(f"Frame {frame_index}: expected {len(SUPPORT_ORDER)} repetition regimes")
    token_horizons = current_rows["materialized_tokens_b"].unique()
    total_tpps = current_rows["total_parameter_tpp"].unique()
    if len(token_horizons) != 1 or len(total_tpps) != 1:
        raise ValueError(f"Frame {frame_index}: repetition regimes must share one token horizon and TPP")

    _configure_matplotlib()
    figure = plt.figure(figsize=(15.5, 23.6), facecolor=PAPER)
    grid = figure.add_gridspec(
        len(SUPPORT_ORDER),
        2,
        width_ratios=(0.78, 1.22),
        left=0.145,
        right=0.97,
        bottom=0.065,
        top=0.875,
        hspace=0.26,
        wspace=0.16,
    )
    surface_axes: list[tuple[str, plt.Axes]] = []
    for row_index, support_id in enumerate(SUPPORT_ORDER):
        support_summary = summary.loc[summary["support_id"].eq(support_id)].sort_values("rung").reset_index(drop=True)
        if len(support_summary) != EXPECTED_CELLS:
            raise ValueError(f"{support_id}: expected {EXPECTED_CELLS} horizons")
        show_x_labels = row_index == len(SUPPORT_ORDER) - 1
        surface_axis = figure.add_subplot(grid[row_index, 0])
        scaling_axis = figure.add_subplot(grid[row_index, 1])
        _render_surface(
            surface_axis,
            observations,
            support_summary,
            frame_index,
            compact=True,
            show_x_labels=show_x_labels,
        )
        _render_scaling_track(
            scaling_axis,
            support_summary,
            frame_index,
            compact=True,
            show_x_labels=show_x_labels,
        )
        surface_axes.append((support_id, surface_axis))

    figure.canvas.draw()
    for support_id, surface_axis in surface_axes:
        position = surface_axis.get_position()
        figure.text(
            0.018,
            0.5 * (position.y0 + position.y1),
            MASTER_ROW_LABELS[support_id],
            ha="left",
            va="center",
            color=INK,
            fontsize=10.2,
            fontweight="semibold",
            linespacing=1.25,
        )

    figure.suptitle(
        "StarCoder WSD80 global optimum motion by repetition target",
        x=0.5,
        y=0.982,
        color=INK,
        fontsize=25,
        fontweight="semibold",
    )
    figure.text(
        0.5,
        0.949,
        f"Measured horizon {frame_index + 1}/4 · D={float(token_horizons[0]):.2f}B · "
        f"total-parameter TPP={float(total_tpps[0]):.2f}",
        ha="center",
        va="center",
        color=MUTED,
        fontsize=12.5,
    )
    figure.text(
        0.315,
        0.888,
        "Observed response and optimum movement",
        ha="center",
        va="center",
        color=INK,
        fontsize=13.5,
        fontweight="semibold",
    )
    figure.text(
        0.735,
        0.888,
        "Raw global grid minima across token horizon",
        ha="center",
        va="center",
        color=INK,
        fontsize=13.5,
        fontweight="semibold",
    )
    figure.text(
        0.115,
        0.47,
        "Phase 1 StarCoder weight",
        ha="center",
        va="center",
        rotation=90,
        color=INK,
        fontsize=10.5,
        fontweight="semibold",
    )
    figure.text(
        0.515,
        0.47,
        "Observed Programming Languages BPB",
        ha="center",
        va="center",
        rotation=90,
        color=INK,
        fontsize=10.5,
        fontweight="semibold",
    )
    legend_handles = [
        Line2D([0], [0], color=TIED_COLOR, marker="X", linewidth=1.8, markersize=7, label="Observed tied grid minimum"),
        Line2D(
            [0],
            [0],
            color=TWO_PHASE_COLOR,
            marker="D",
            linewidth=1.8,
            markersize=6.5,
            label="Observed untied grid minimum",
        ),
        Line2D(
            [0],
            [0],
            color=ALIAS_COLOR,
            marker="o",
            markerfacecolor="none",
            linewidth=0,
            markersize=6,
            label="Deterministic materialization alias",
        ),
    ]
    figure.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.923),
        frameon=False,
        ncol=3,
        fontsize=10.5,
    )
    colorbar_axis = figure.add_axes((0.18, 0.025, 0.285, 0.008))
    colorbar = figure.colorbar(ScalarMappable(norm=SURFACE_NORM, cmap=CMAP), cax=colorbar_axis, orientation="horizontal")
    colorbar.set_label("Observed BPB relative to tied grid minimum (clipped)", color=INK, fontsize=8.5)
    colorbar.ax.tick_params(labelsize=7.5, colors=INK)
    figure.text(
        0.73,
        0.027,
        "Four measured horizons only; contours use linear triangulation.\n"
        "Raw selected minima are one-seed and selection-biased.",
        ha="center",
        va="center",
        color=MUTED,
        fontsize=8.8,
    )

    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=104, facecolor=PAPER)
    plt.close(figure)
    buffer.seek(0)
    image = Image.open(buffer).convert("RGB")
    return image.copy()


def write_gif(path: Path, frames: list[Image.Image]) -> None:
    """Write a four-frame GIF with a shared adaptive palette."""
    if len(frames) != len(FRAME_DURATIONS_MS):
        raise ValueError("Animation must contain exactly four measured frames")
    first = frames[0].convert("P", palette=Image.Palette.ADAPTIVE, colors=256)
    quantized = [first, *(frame.quantize(palette=first) for frame in frames[1:])]
    first.save(
        path,
        save_all=True,
        append_images=quantized[1:],
        duration=list(FRAME_DURATIONS_MS),
        loop=0,
        optimize=True,
        disposal=2,
    )


def _html_document(frame_data: dict[str, list[dict[str, object]]]) -> str:
    support_options = "\n".join(
        f'<option value="{support_id}">{SUPPORT_LABELS[support_id]}</option>' for support_id in SUPPORT_ORDER
    )
    support_buttons = "\n".join(
        (
            f'<button type="button" class="regime-button" data-support="{support_id}" aria-pressed="false">'
            f"{SUPPORT_MARKER_LABELS[support_id]}</button>"
        )
        for support_id in SUPPORT_ORDER
    )
    payload = json.dumps(frame_data, separators=(",", ":"))
    template = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>StarCoder WSD80 repetition-conditioned optimum motion</title>
<style>
:root { --paper:#f7f3e8; --panel:#fffdf8; --ink:#17324d; --muted:#657786; --grid:#d8d1c2; --accent:#d85f3d; }
* { box-sizing:border-box; }
body { margin:0; background:var(--paper); color:var(--ink); font-family:"Avenir Next","Helvetica Neue",sans-serif; }
main { max-width:1660px; margin:0 auto; padding:28px 30px 44px; }
h1 { margin:0; font:700 clamp(30px,3.2vw,48px)/1.04 Georgia,serif; }
.dek { margin:10px 0 22px; color:var(--muted); font-size:17px; }
.controls { display:grid; grid-template-columns:minmax(280px,1fr) auto auto; gap:16px; align-items:end; padding:16px 0; border-block:1px solid var(--grid); }
label { display:grid; gap:6px; color:var(--muted); font-size:12px; font-weight:700; letter-spacing:.08em; text-transform:uppercase; }
select,button { color:var(--ink); background:var(--panel); border:1px solid var(--grid); font:700 15px/1 "Avenir Next","Helvetica Neue",sans-serif; }
select { min-height:44px; padding:0 42px 0 13px; }
.transport { display:flex; gap:8px; }
.transport button { min-width:48px; min-height:44px; padding:0 14px; cursor:pointer; }
.regimes { display:flex; flex-wrap:wrap; gap:7px; justify-content:flex-end; }
.regime-button { min-width:54px; min-height:36px; padding:0 10px; cursor:pointer; border-width:2px; }
.regime-button[aria-pressed="true"] { border-color:var(--accent); color:var(--accent); }
.stage { margin-top:20px; border:1px solid var(--grid); background:var(--panel); }
.stage img { display:block; width:100%; height:auto; }
.readout { display:flex; justify-content:space-between; gap:20px; align-items:baseline; padding:13px 16px; border-top:1px solid var(--grid); }
.readout strong { font-family:Georgia,serif; font-size:20px; }
.readout span { color:var(--muted); }
.timeline { display:grid; grid-template-columns:repeat(4,1fr); gap:8px; margin-top:14px; }
.timeline button { min-height:42px; cursor:pointer; }
.timeline button[aria-pressed="true"] { color:var(--accent); border-color:var(--accent); border-width:2px; }
.links { margin-top:16px; color:var(--muted); }
.links a { color:var(--ink); font-weight:700; }
@media (max-width:800px) { main{padding:20px 14px 34px}.controls{grid-template-columns:1fr}.regimes{justify-content:flex-start}.readout{display:block}.readout span{display:block;margin-top:5px} }
</style>
</head>
<body>
<main>
  <h1>Repetition-conditioned optimum motion</h1>
  <p class="dek">StarCoder WSD80 · fixed N=210M · four measured token horizons · seven StarCoder simulated-epoching repetition regimes</p>
  <section class="controls" aria-label="Animation controls">
    <label>StarCoder repetition target<select id="support-select">__SUPPORT_OPTIONS__</select></label>
    <div class="transport">
      <button type="button" id="previous" aria-label="Previous measured horizon">←</button>
      <button type="button" id="play" aria-label="Play animation">Play</button>
      <button type="button" id="next" aria-label="Next measured horizon">→</button>
    </div>
    <div class="regimes" aria-label="StarCoder repetition shortcuts">__SUPPORT_BUTTONS__</div>
  </section>
  <section class="stage">
    <img id="frame" alt="Paired response-surface and optimum-scaling visualization">
    <div class="readout"><strong id="frame-title"></strong><span id="frame-detail"></span></div>
  </section>
  <nav class="timeline" aria-label="Measured token horizons" id="timeline"></nav>
  <p class="links"><a id="gif-link">Open the synchronized seven-row GIF</a></p>
</main>
<script>
const DATA=__FRAME_DATA__;
const order=Object.keys(DATA);
const select=document.getElementById('support-select');
const frame=document.getElementById('frame');
const title=document.getElementById('frame-title');
const detail=document.getElementById('frame-detail');
const timeline=document.getElementById('timeline');
const playButton=document.getElementById('play');
const gifLink=document.getElementById('gif-link');
let support=order[0], index=0, timer=null;
function stop(){ if(timer!==null){clearTimeout(timer);timer=null;} playButton.textContent='Play'; }
function schedule(){ if(timer===null)return; const delay=index===3?3000:1800; timer=setTimeout(()=>{index=(index+1)%4;render();schedule();},delay); }
function render(){
  const item=DATA[support][index];
  frame.src=item.src;
  frame.alt=`${item.support_label}, measured horizon ${index+1} of 4`;
  title.textContent=`${item.tokens_label} tokens · global two-phase gain ${item.gain_label} BPB`;
  detail.textContent=`tied ${item.tied_bpb} · untied ${item.untied_bpb} · total-parameter TPP ${item.tpp}`;
  gifLink.href=item.gif;
  document.querySelectorAll('.regime-button').forEach(button=>button.setAttribute('aria-pressed',String(button.dataset.support===support)));
  document.querySelectorAll('.timeline button').forEach((button,i)=>button.setAttribute('aria-pressed',String(i===index)));
  select.value=support;
}
function chooseSupport(value){stop();support=value;index=0;render();}
DATA[support].forEach((item,i)=>{const button=document.createElement('button');button.type='button';button.textContent=item.tokens_label;button.addEventListener('click',()=>{stop();index=i;render();});timeline.appendChild(button);});
select.addEventListener('change',event=>chooseSupport(event.target.value));
document.querySelectorAll('.regime-button').forEach(button=>button.addEventListener('click',()=>chooseSupport(button.dataset.support)));
document.getElementById('previous').addEventListener('click',()=>{stop();index=(index+3)%4;render();});
document.getElementById('next').addEventListener('click',()=>{stop();index=(index+1)%4;render();});
playButton.addEventListener('click',()=>{if(timer!==null){stop();return;}playButton.textContent='Pause';timer=0;schedule();});
document.addEventListener('keydown',event=>{if(event.key==='ArrowLeft'){stop();index=(index+3)%4;render();}if(event.key==='ArrowRight'){stop();index=(index+1)%4;render();}if(event.key===' '){event.preventDefault();playButton.click();}});
Object.values(DATA).flat().forEach(item=>{const image=new Image();image.src=item.src;});
render();
</script>
</body>
</html>
"""
    return (
        template.replace("__SUPPORT_OPTIONS__", support_options)
        .replace("__SUPPORT_BUTTONS__", support_buttons)
        .replace("__FRAME_DATA__", payload)
    )


def write_html(output_dir: Path, summary: pd.DataFrame) -> None:
    """Write the repetition selector and synchronized measured-horizon controls."""
    frame_data: dict[str, list[dict[str, object]]] = {}
    for support_id in SUPPORT_ORDER:
        group = summary.loc[summary["support_id"].eq(support_id)].sort_values("rung")
        rows: list[dict[str, object]] = []
        for row in group.itertuples(index=False):
            rows.append(
                {
                    "src": f"frames/{support_id}_r{int(row.rung)}.png",
                    "gif": f"gifs/{MASTER_GIF_FILENAME}",
                    "support_label": SUPPORT_LABELS[support_id],
                    "tokens_label": f"{float(row.materialized_tokens_b):.2f}B",
                    "gain_label": f"{float(row.raw_two_phase_gain_bpb):+.5f}",
                    "tied_bpb": f"{float(row.tied_bpb):.6f}",
                    "untied_bpb": f"{float(row.untied_bpb):.6f}",
                    "tpp": f"{float(row.total_parameter_tpp):.2f}",
                }
            )
        frame_data[support_id] = rows
    output = output_dir / "starcoder_wsd80_replay_conditioned_optimum_motion.html"
    output.write_text(_html_document(frame_data), encoding="utf-8")


def write_report(output_dir: Path, summary: pd.DataFrame) -> None:
    """Write a concise provenance and interpretation note."""
    final = summary.loc[summary["rung"].eq(3)].sort_values("support_order")
    lines = [
        "# StarCoder WSD80 repetition-conditioned optimum motion",
        "",
        (
            "- Scope: fixed 210M-parameter model, four measured token horizons, and seven StarCoder "
            "simulated-epoching repetition regimes."
        ),
        "- Nemotron always uses its complete physical caches; only StarCoder support is capped.",
        "- Each response surface uses the same 125 policy coordinates.",
        "- The left panel is a linear triangulation of observed Programming Languages BPB relative to the tied grid minimum.",
        "- Hollow observations are deterministic materialization aliases, not independent training outcomes.",
        "- The right panel reveals raw tied and eligible-untied grid minima as D increases.",
        (
            "- Global two-phase gain means the lowest observed tied BPB minus the lowest eligible untied BPB within "
            "the sampled grid. Positive favors the untied policy."
        ),
        "- Frames are the four measured horizons; no synthetic intermediate surfaces are shown.",
        "- All minima are one-seed and selection-biased; they are not continuous-surface or fresh-seed confirmations.",
        "",
        "## Final-horizon global two-phase gain",
        "",
    ]
    for row in final.itertuples(index=False):
        lines.append(f"- {SUPPORT_LABELS[row.support_id]}: {float(row.raw_two_phase_gain_bpb):+.6f} BPB")
    lines.extend(
        [
            "",
            "## Synchronized animation",
            "",
            f"- `{MASTER_GIF_FILENAME}`: all seven StarCoder repetition regimes in aligned rows.",
        ]
    )
    lines.append("")
    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    observations, summary = load_data(args.selected_policies, args.coverage_observations, args.design)
    frames_dir = args.output_dir / "frames"
    gifs_dir = args.output_dir / "gifs"
    master_frames_dir = args.output_dir / "master_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    gifs_dir.mkdir(parents=True, exist_ok=True)
    master_frames_dir.mkdir(parents=True, exist_ok=True)

    for stale_gif in gifs_dir.glob("*.gif"):
        stale_gif.unlink()

    for support_id in SUPPORT_ORDER:
        for frame_index in range(EXPECTED_CELLS):
            frame = render_frame(observations, summary, support_id, frame_index)
            frame.save(frames_dir / f"{support_id}_r{frame_index}.png", optimize=True)

    master_frames: list[Image.Image] = []
    for frame_index in range(EXPECTED_CELLS):
        frame = render_master_frame(observations, summary, frame_index)
        frame.save(master_frames_dir / f"all_repetition_regimes_r{frame_index}.png", optimize=True)
        master_frames.append(frame)
    write_gif(gifs_dir / MASTER_GIF_FILENAME, master_frames)

    summary.to_csv(args.output_dir / "observed_grid_optima_by_horizon_and_replay.csv", index=False)
    write_html(args.output_dir, summary)
    write_report(args.output_dir, summary)


if __name__ == "__main__":
    main()
