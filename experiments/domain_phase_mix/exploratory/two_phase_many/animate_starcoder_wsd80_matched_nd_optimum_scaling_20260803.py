# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "matplotlib",
#   "numpy",
#   "pandas",
#   "pillow",
# ]
# ///

"""Animate observed StarCoder WSD80 policy optima across three scaling tracks."""

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

SCRIPT_DIR = Path(__file__).resolve().parent
PANEL_DIR = SCRIPT_DIR / "reference_outputs" / "starcoder_wsd80_matched_nd_stage1_20260731"
STAGE3_RESULTS = PANEL_DIR / "stage3_dense_surface_results_20260802"
DEFAULT_OBSERVATIONS = STAGE3_RESULTS / "combined_discovery_observations.csv"
DEFAULT_DISCOVERY = STAGE3_RESULTS / "cell_discovery_summary.csv"
DEFAULT_SOURCE_DESIGN = PANEL_DIR / "stage2_results_20260801" / "source_design.json"
DEFAULT_OUTPUT_DIR = PANEL_DIR / "optimum_scaling_20260802"

TRACK_ORDER = ("increase_d", "increase_n", "increase_nd")
TRACK_LABELS = {
    "increase_d": "Fixed N, increase D",
    "increase_n": "Fixed D, increase N",
    "increase_nd": "Increase N and D",
}
SCALING_AXES = {
    "increase_d": ("materialized_tokens", 1e9, "Materialized tokens D (billions)"),
    "increase_n": ("total_parameters", 1e6, "Total parameters N (millions)"),
    "increase_nd": ("compute_flops", 1e18, "Training compute (1e18 FLOPs)"),
}

PAPER = "#F7F3E8"
PANEL = "#FFFDF8"
INK = "#17324D"
MUTED = "#657786"
GRID = "#D8D1C2"
TIED_COLOR = "#1F5A85"
TWO_PHASE_COLOR = "#D85F3D"
CMAP = mpl.colormaps["RdYlGn_r"]
SURFACE_NORM = mpl.colors.Normalize(vmin=0.0, vmax=0.05)
FRAME_DURATIONS_MS = (1700, 1700, 1700, 2800)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observations", type=Path, default=DEFAULT_OBSERVATIONS)
    parser.add_argument("--discovery", type=Path, default=DEFAULT_DISCOVERY)
    parser.add_argument("--source-design", type=Path, default=DEFAULT_SOURCE_DESIGN)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def _track_memberships(value: object) -> tuple[str, ...]:
    if isinstance(value, str):
        value = json.loads(value.replace("'", '"'))
    if not isinstance(value, list):
        raise ValueError(f"Invalid track memberships: {value!r}")
    tracks = tuple(str(item) for item in value)
    unknown = set(tracks) - set(TRACK_ORDER)
    if unknown:
        raise ValueError(f"Unknown track memberships: {sorted(unknown)}")
    return tracks


def load_data(
    observations_path: Path,
    discovery_path: Path,
    source_design_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and validate observed surfaces and raw selected minima."""
    design = json.loads(source_design_path.read_text(encoding="utf-8"))
    cells = pd.DataFrame(design.get("source_cells", design.get("cells")))
    required_cells = {
        "cell_id",
        "compute_flops",
        "materialized_tokens",
        "rung",
        "total_parameters",
        "track_memberships",
    }
    missing = required_cells - set(cells.columns)
    if missing:
        raise ValueError(f"Source design is missing fields: {sorted(missing)}")
    cells["track_memberships"] = cells["track_memberships"].map(_track_memberships)

    observations = pd.read_csv(observations_path)
    required_observations = {
        "cell_id",
        "phase_0_starcoder",
        "phase_1_starcoder",
        "starcoder_bpb",
    }
    missing = required_observations - set(observations.columns)
    if missing:
        raise ValueError(f"Observation table is missing fields: {sorted(missing)}")

    discovery = pd.read_csv(discovery_path)
    required_discovery = {
        "best_tied_bpb",
        "best_tied_weight",
        "best_untied_bpb",
        "best_untied_p0",
        "best_untied_p1",
        "cell_id",
    }
    missing = required_discovery - set(discovery.columns)
    if missing:
        raise ValueError(f"Discovery summary is missing fields: {sorted(missing)}")
    discovery = discovery.merge(
        cells[
            [
                "cell_id",
                "compute_flops",
                "materialized_tokens",
                "rung",
                "total_parameters",
                "track_memberships",
            ]
        ],
        on="cell_id",
        how="left",
        validate="one_to_one",
        suffixes=("", "_design"),
    )
    for column in ("materialized_tokens", "rung", "total_parameters"):
        design_column = f"{column}_design"
        if design_column in discovery:
            if not np.allclose(discovery[column], discovery[design_column], rtol=0.0, atol=0.0):
                raise ValueError(f"Discovery summary disagrees with source design on {column}")
            discovery = discovery.drop(columns=design_column)

    if set(observations["cell_id"]) != set(cells["cell_id"]):
        raise ValueError("Observation and source-design cell IDs disagree")
    if set(discovery["cell_id"]) != set(cells["cell_id"]):
        raise ValueError("Discovery and source-design cell IDs disagree")
    if len(observations) != 714:
        raise ValueError(f"Expected 714 observed checkpoints, found {len(observations)}")
    return observations, discovery


def _track_data(discovery: pd.DataFrame, track: str) -> pd.DataFrame:
    selected = discovery.loc[discovery["track_memberships"].map(lambda tracks: track in tracks)].sort_values("rung")
    if len(selected) != 4 or selected["rung"].tolist() != [0, 1, 2, 3]:
        raise ValueError(f"{track}: expected exactly four ordered scaling rungs")
    return selected.reset_index(drop=True)


def _render_surface(
    axis: plt.Axes,
    cell_observations: pd.DataFrame,
    cell: pd.Series,
    track: str,
) -> None:
    x = cell_observations["phase_0_starcoder"].to_numpy(dtype=float)
    y = cell_observations["phase_1_starcoder"].to_numpy(dtype=float)
    bpb = cell_observations["starcoder_bpb"].to_numpy(dtype=float)
    delta = np.clip(bpb - bpb.min(), 0.0, SURFACE_NORM.vmax)
    triangulation = mtri.Triangulation(x, y)
    levels = np.linspace(SURFACE_NORM.vmin, SURFACE_NORM.vmax, 15)
    axis.tricontourf(triangulation, delta, levels=levels, cmap=CMAP, norm=SURFACE_NORM, extend="max", alpha=0.82)
    axis.scatter(x, y, c=delta, cmap=CMAP, norm=SURFACE_NORM, s=22, edgecolor=PANEL, linewidth=0.55, zorder=3)
    axis.plot([0.0, 1.0], [0.0, 1.0], color=INK, linewidth=1.0, linestyle="--", alpha=0.45, zorder=2)

    axis.scatter(
        [cell["best_tied_weight"]],
        [cell["best_tied_weight"]],
        marker="X",
        s=145,
        color=TIED_COLOR,
        edgecolor=PANEL,
        linewidth=1.6,
        zorder=5,
    )
    axis.scatter(
        [cell["best_untied_p0"]],
        [cell["best_untied_p1"]],
        marker="D",
        s=115,
        color=TWO_PHASE_COLOR,
        edgecolor=PANEL,
        linewidth=1.6,
        zorder=5,
    )
    gain = float(cell["best_tied_bpb"] - cell["best_untied_bpb"])
    axis.text(
        0.03,
        0.04,
        f"raw min-vs-min gain {gain:+.5f} BPB",
        transform=axis.transAxes,
        color=INK,
        fontsize=9.5,
        bbox={"boxstyle": "round,pad=0.32", "facecolor": PAPER, "edgecolor": GRID, "alpha": 0.92},
        zorder=6,
    )
    axis.set_xlim(-0.02, 1.02)
    axis.set_ylim(-0.02, 1.02)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("Phase 0 StarCoder weight")
    axis.set_ylabel("Phase 1 StarCoder weight")
    axis.set_title(
        f"{TRACK_LABELS[track]} · rung {int(cell['rung']) + 1}/4\n"
        f"N={float(cell['total_parameters']) / 1e6:.0f}M · D={float(cell['materialized_tokens']) / 1e9:.3g}B",
        color=INK,
        fontsize=13,
        fontweight="semibold",
        pad=10,
    )


def _render_scaling_track(axis: plt.Axes, track_data: pd.DataFrame, frame_index: int, track: str) -> None:
    x_column, divisor, x_label = SCALING_AXES[track]
    x_all = track_data[x_column].to_numpy(dtype=float) / divisor
    x = x_all[: frame_index + 1]
    tied = track_data["best_tied_bpb"].to_numpy(dtype=float)[: frame_index + 1]
    untied = track_data["best_untied_bpb"].to_numpy(dtype=float)[: frame_index + 1]

    axis.plot(x, tied, color=TIED_COLOR, linewidth=2.2, marker="X", markersize=8.5, label="Observed 1p minimum")
    axis.plot(
        x,
        untied,
        color=TWO_PHASE_COLOR,
        linewidth=2.2,
        marker="D",
        markersize=7.2,
        label="Observed 2p minimum",
    )
    axis.scatter([x[-1]], [tied[-1]], marker="X", s=120, color=TIED_COLOR, edgecolor=PANEL, linewidth=1.2, zorder=4)
    axis.scatter(
        [x[-1]],
        [untied[-1]],
        marker="D",
        s=100,
        color=TWO_PHASE_COLOR,
        edgecolor=PANEL,
        linewidth=1.2,
        zorder=4,
    )
    all_bpb = track_data[["best_tied_bpb", "best_untied_bpb"]].to_numpy(dtype=float)
    padding = max(0.003, 0.08 * float(all_bpb.max() - all_bpb.min()))
    axis.set_ylim(float(all_bpb.min()) - padding, float(all_bpb.max()) + padding)
    axis.set_xscale("log")
    axis.set_xlim(float(x_all.min()) / 1.08, float(x_all.max()) * 1.08)
    axis.set_xticks(x_all)
    axis.set_xticklabels([f"{value:.0f}" if value >= 1000 else f"{value:.3g}" for value in x_all])
    axis.set_xlabel(x_label)
    axis.set_ylabel("Observed Programming BPB")
    axis.set_title("Raw selected optima · lower is better", color=INK, fontsize=13, fontweight="semibold", pad=10)
    axis.xaxis.set_minor_locator(mpl.ticker.NullLocator())
    axis.grid(True, color=GRID, linewidth=0.8, alpha=0.85)


def render_frame(
    observations: pd.DataFrame,
    discovery: pd.DataFrame,
    frame_index: int,
) -> Image.Image:
    """Render one common-rung frame across all three scaling tracks."""
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
    figure = plt.figure(figsize=(15.5, 15.8), facecolor=PAPER)
    grid = figure.add_gridspec(
        3,
        2,
        width_ratios=(1.0, 1.03),
        left=0.065,
        right=0.965,
        bottom=0.10,
        top=0.88,
        wspace=0.27,
        hspace=0.46,
    )
    for row, track in enumerate(TRACK_ORDER):
        track_data = _track_data(discovery, track)
        cell = track_data.iloc[frame_index]
        cell_observations = observations.loc[observations["cell_id"].eq(cell["cell_id"])]
        _render_surface(figure.add_subplot(grid[row, 0]), cell_observations, cell, track)
        _render_scaling_track(figure.add_subplot(grid[row, 1]), track_data, frame_index, track)

    figure.suptitle(
        "StarCoder 80/20 WSD observed phase-optimum scaling",
        x=0.5,
        y=0.965,
        color=INK,
        fontsize=25,
        fontweight="semibold",
    )
    figure.text(
        0.5,
        0.935,
        f"Scaling rung {frame_index + 1}/4 · 714-run dense panel · "
        "raw selected minima are descriptive and selection-biased",
        ha="center",
        va="center",
        color=MUTED,
        fontsize=12.5,
    )
    legend_handles = [
        Line2D([0], [0], color=TIED_COLOR, marker="X", linewidth=2.2, markersize=9, label="Observed 1p minimum"),
        Line2D([0], [0], color=TWO_PHASE_COLOR, marker="D", linewidth=2.2, markersize=8, label="Observed 2p minimum"),
    ]
    figure.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.913),
        frameon=False,
        ncol=2,
        fontsize=11.5,
    )
    colorbar_axis = figure.add_axes((0.105, 0.035, 0.35, 0.014))
    colorbar = figure.colorbar(ScalarMappable(norm=SURFACE_NORM, cmap=CMAP), cax=colorbar_axis, orientation="horizontal")
    colorbar.set_label("Observed BPB above each cell minimum (clipped at +0.05)", color=INK, fontsize=10)
    colorbar.ax.tick_params(labelsize=9, colors=INK)

    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=105, facecolor=PAPER)
    plt.close(figure)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def write_gif(path: Path, frames: list[Image.Image]) -> None:
    """Write a compact GIF with one stable adaptive palette."""
    if len(frames) != len(FRAME_DURATIONS_MS):
        raise ValueError("Animation must contain exactly four scaling-rung frames")
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


def main() -> None:
    args = parse_args()
    observations, discovery = load_data(args.observations, args.discovery, args.source_design)
    frames = [render_frame(observations, discovery, frame_index) for frame_index in range(4)]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "starcoder_wsd80_matched_nd_optimum_scaling.gif"
    write_gif(output, frames)
    discovery.to_csv(args.output_dir / "observed_optimum_animation_data.csv", index=False)


if __name__ == "__main__":
    main()
