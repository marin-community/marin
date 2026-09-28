# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib", "numpy", "pandas", "plotly"]
# ///
"""Render the two-phase StarCoder non-monotonicity figure for the paper."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D, proj3d
from paper_plot_style import GRP_COLOR, PAPER_AXIS, PAPER_GRID, PAPER_MUTED, PAPER_TEXT, UNIFORM_COLOR

SCRIPT_DIR = Path(__file__).resolve().parent
IMG_DIR = SCRIPT_DIR / "img"
SOURCE_CSV = SCRIPT_DIR / "data" / "two_phase_starcoder_combined_143_from_wandb.csv"
REPEAT_METRICS_CSV = (
    SCRIPT_DIR.parent
    / "reference_outputs"
    / "starcoder_heteroskedastic_snr_20260523"
    / "collected_train_only_metrics_live.csv"
)

TARGET = "eval/paloma/dolma_100_programing_languages/bpb"
LANDSCAPE_OUTPUT_STEM = IMG_DIR / "starcoder_two_phase_landscape"
LANDSCAPE_ERROR_BARS_OUTPUT_STEM = IMG_DIR / "starcoder_two_phase_landscape_anchor_error_bars"
SLICE_OUTPUT_STEM = IMG_DIR / "starcoder_two_phase_slice"

LANDSCAPE_FIGSIZE = (4.7, 4.3)
SLICE_FIGSIZE = (4.7, 3.85)
DPI = 300
PROPORTIONAL_GOLD = "#FFD700"


def _projected_marker(
    ax: Axes3D,
    x: float,
    y: float,
    z: float,
    *,
    marker: str,
    markersize: float,
    color: str = UNIFORM_COLOR,
) -> None:
    """Overlay a 2D marker at the projected location of a 3D point."""
    projected_x, projected_y, _ = proj3d.proj_transform(x, y, z, ax.get_proj())
    marker_artist = Line2D(
        [projected_x],
        [projected_y],
        marker=marker,
        linestyle="",
        color=color,
        markerfacecolor=color,
        markeredgecolor="white",
        markeredgewidth=0.9,
        markersize=markersize,
        transform=ax.transData,
        zorder=1000,
        clip_on=False,
    )
    ax.add_artist(marker_artist)


def _completed_frame() -> pd.DataFrame:
    """Load completed StarCoder two-phase runs with the target metric present."""
    frame = pd.read_csv(SOURCE_CSV)
    frame = frame[frame["status"].eq("completed") & frame[TARGET].notna()].copy()
    if frame.empty:
        raise ValueError(f"No completed rows with {TARGET} in {SOURCE_CSV}")
    return frame


def _proportional_anchor() -> pd.Series:
    """Return the repeated proportional anchor for the target benchmark."""
    frame = pd.read_csv(REPEAT_METRICS_CSV)
    final_step = int(frame["latest_step"].dropna().max())
    rows = frame[
        frame["latest_step"].eq(final_step) & frame["anchor_id"].eq("proportional") & frame[TARGET].notna()
    ].copy()
    if rows.empty:
        raise ValueError(f"No proportional repeat-anchor rows with {TARGET} in {REPEAT_METRICS_CSV}")
    return pd.Series(
        {
            "phase_0_starcoder": float(rows["phase_0_starcoder"].iloc[0]),
            "phase_1_starcoder": float(rows["phase_1_starcoder"].iloc[0]),
            TARGET: float(rows[TARGET].mean()),
        }
    )


def _anchor_noise_summary() -> pd.DataFrame:
    """Return repeat-anchor mean and standard deviation for the target benchmark."""
    frame = pd.read_csv(REPEAT_METRICS_CSV)
    final_step = int(frame["latest_step"].dropna().max())
    rows = frame[frame["latest_step"].eq(final_step) & frame[TARGET].notna()].copy()
    if rows.empty:
        raise ValueError(f"No repeat-anchor rows with {TARGET} in {REPEAT_METRICS_CSV}")
    grouped = (
        rows.groupby("anchor_id", as_index=False)
        .agg(
            phase_0_starcoder=("phase_0_starcoder", "first"),
            phase_1_starcoder=("phase_1_starcoder", "first"),
            mean=(TARGET, "mean"),
            std=(TARGET, "std"),
            count=(TARGET, "count"),
        )
        .sort_values(["phase_0_starcoder", "phase_1_starcoder", "anchor_id"])
        .reset_index(drop=True)
    )
    return grouped


def _phase_1_starcoder_epoch_multiplier(frame: pd.DataFrame) -> float:
    """Return StarCoder phase-1 epochs per unit mixture weight."""
    nonzero = frame["phase_1_starcoder"].to_numpy(dtype=float) > 0
    ratios = frame.loc[nonzero, "phase_1_starcoder_epochs"].to_numpy(dtype=float) / frame.loc[
        nonzero, "phase_1_starcoder"
    ].to_numpy(dtype=float)
    if ratios.size == 0:
        raise ValueError("Cannot infer StarCoder epoch multiplier from zero-only phase-1 weights")
    return float(np.median(ratios))


def _slice_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Return the Nemotron-only first-phase slice used in the background figure."""
    slice_rows = frame[frame["phase_0_nemotron_full"].round(4).eq(1.0)].copy()
    if slice_rows.empty:
        raise ValueError("No phase_0_nemotron_full == 1.0 rows found")
    return slice_rows.sort_values("phase_1_starcoder")


def _style_axis(ax: Axes) -> None:
    """Apply shared static styling to a Matplotlib axis."""
    ax.set_facecolor("white")
    ax.grid(color=PAPER_GRID, linewidth=0.8, alpha=0.75)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(PAPER_AXIS)
        ax.spines[side].set_linewidth(0.9)
    ax.tick_params(axis="both", colors=PAPER_TEXT, labelsize=10)


def _style_3d_axis(ax: Axes3D) -> None:
    """Apply paper styling to a 3D axis."""
    pane_color = (0.91, 0.94, 0.98, 1.0)
    ax.xaxis.set_pane_color(pane_color)
    ax.yaxis.set_pane_color(pane_color)
    ax.zaxis.set_pane_color(pane_color)
    ax.xaxis._axinfo["grid"]["color"] = (1.0, 1.0, 1.0, 1.0)
    ax.yaxis._axinfo["grid"]["color"] = (1.0, 1.0, 1.0, 1.0)
    ax.zaxis._axinfo["grid"]["color"] = (1.0, 1.0, 1.0, 1.0)
    ax.xaxis._axinfo["grid"]["linewidth"] = 0.85
    ax.yaxis._axinfo["grid"]["linewidth"] = 0.85
    ax.zaxis._axinfo["grid"]["linewidth"] = 0.85
    ax.tick_params(axis="both", colors=PAPER_TEXT, labelsize=8.8, pad=1)
    ax.tick_params(axis="z", colors=PAPER_TEXT, labelsize=8.8, pad=3)


def _plot_anchor_error_bars(ax: Axes3D, anchors: pd.DataFrame) -> None:
    """Overlay ±1 std vertical error bars for repeated anchors."""
    cap_width = 0.018
    for row in anchors.itertuples(index=False):
        x = float(row.phase_0_starcoder)
        y = float(row.phase_1_starcoder)
        mean = float(row.mean)
        std = float(row.std)
        lower = mean - std
        upper = mean + std
        ax.plot([x, x], [y, y], [lower, upper], color=PAPER_TEXT, linewidth=1.25, alpha=0.84)
        ax.plot([x - cap_width, x + cap_width], [y, y], [lower, lower], color=PAPER_TEXT, linewidth=1.05, alpha=0.84)
        ax.plot([x - cap_width, x + cap_width], [y, y], [upper, upper], color=PAPER_TEXT, linewidth=1.05, alpha=0.84)
        ax.scatter(
            [x],
            [y],
            [mean],
            s=38,
            color="white",
            edgecolors=PAPER_TEXT,
            linewidths=0.65,
            depthshade=False,
            alpha=0.98,
        )


def _plot_landscape(
    ax: Axes3D,
    frame: pd.DataFrame,
    slice_rows: pd.DataFrame,
    *,
    show_anchor_error_bars: bool = False,
) -> None:
    """Plot the full two-phase landscape as a 3D scatter view."""
    global_min = frame.loc[frame[TARGET].idxmin()]
    slice_min = slice_rows.loc[slice_rows[TARGET].idxmin()]
    proportional = _proportional_anchor()
    anchors = _anchor_noise_summary() if show_anchor_error_bars else pd.DataFrame()
    x = frame["phase_0_starcoder"].to_numpy(dtype=float)
    y = frame["phase_1_starcoder"].to_numpy(dtype=float)
    z = frame[TARGET].to_numpy(dtype=float)
    color_norm = Normalize(vmin=float(z.min()), vmax=float(np.percentile(z, 96)))
    triangulation = mtri.Triangulation(x, y)
    slice_x = slice_rows["phase_0_starcoder"].to_numpy(dtype=float)
    slice_y = slice_rows["phase_1_starcoder"].to_numpy(dtype=float)
    slice_z = slice_rows[TARGET].to_numpy(dtype=float)

    ax.plot_trisurf(
        triangulation,
        z,
        cmap="RdYlGn_r",
        norm=color_norm,
        linewidth=0.08,
        edgecolor=(1.0, 1.0, 1.0, 0.24),
        alpha=0.34,
        antialiased=True,
        shade=False,
    )

    ax.scatter(
        x,
        y,
        z,
        c=z,
        cmap="RdYlGn_r",
        norm=color_norm,
        s=24,
        edgecolors="white",
        linewidths=0.3,
        alpha=0.92,
        depthshade=False,
    )
    ax.plot(slice_x, slice_y, slice_z, color=UNIFORM_COLOR, linewidth=1.6, alpha=0.95)
    if show_anchor_error_bars:
        _plot_anchor_error_bars(ax, anchors)
    legend = ax.legend(
        handles=[
            Line2D(
                [0],
                [0],
                marker="D",
                linestyle="",
                markerfacecolor=UNIFORM_COLOR,
                markeredgecolor="white",
                markeredgewidth=0.75,
                markersize=6.3,
                label=rf"global min: $p^{{(0)}}={global_min['phase_0_starcoder']:.2f}$, "
                rf"$p^{{(1)}}={global_min['phase_1_starcoder']:.2f}$; "
                rf"BPB={float(global_min[TARGET]):.3f}",
            ),
            Line2D(
                [0],
                [0],
                marker="*",
                linestyle="",
                markerfacecolor=UNIFORM_COLOR,
                markeredgecolor="white",
                markeredgewidth=0.75,
                markersize=10.5,
                label=rf"$p^{{(0)}}=0$ slice min: $p^{{(1)}}={slice_min['phase_1_starcoder']:.2f}$; "
                rf"BPB={float(slice_min[TARGET]):.3f}",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="",
                markerfacecolor=PROPORTIONAL_GOLD,
                markeredgecolor="white",
                markeredgewidth=0.75,
                markersize=6.5,
                label=rf"proportional: $p^{{(0)}}=p^{{(1)}}={proportional['phase_0_starcoder']:.2f}$; "
                rf"BPB={float(proportional[TARGET]):.3f}",
            ),
            *(
                [
                    Line2D(
                        [0],
                        [0],
                        marker="o",
                        linestyle="-",
                        color=PAPER_TEXT,
                        markerfacecolor="white",
                        markeredgecolor=PAPER_TEXT,
                        markeredgewidth=0.65,
                        linewidth=1.25,
                        markersize=4.8,
                        label=r"repeat anchors: mean $\pm 1$ std",
                    )
                ]
                if show_anchor_error_bars
                else []
            ),
        ],
        loc="upper left",
        bbox_to_anchor=(-0.03, 0.99),
        frameon=True,
        framealpha=0.9,
        facecolor="white",
        edgecolor=PAPER_AXIS,
        fontsize=7.6,
        handlelength=1.0,
        handletextpad=0.5,
        borderpad=0.45,
    )
    legend.get_frame().set_linewidth(0.6)
    for text in legend.get_texts():
        text.set_color(PAPER_TEXT)

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_zlim(float(z.min()) - 0.04, float(z.max()) + 0.08)
    ax.set_xlabel(r"Phase 0 StarCoder ($p^{(0)}$)", fontsize=9.5, color=PAPER_TEXT, labelpad=4)
    ax.set_ylabel(r"Phase 1 StarCoder ($p^{(1)}$)", fontsize=9.5, color=PAPER_TEXT, labelpad=4)
    ax.set_zlabel("")
    ax.set_title("")
    ax.view_init(elev=25, azim=-141)
    ax.set_box_aspect((1.0, 1.0, 0.95), zoom=1.0)
    _style_3d_axis(ax)
    _projected_marker(
        ax,
        float(global_min["phase_0_starcoder"]),
        float(global_min["phase_1_starcoder"]),
        float(global_min[TARGET]),
        marker="D",
        markersize=6.4,
    )
    _projected_marker(
        ax,
        float(slice_min["phase_0_starcoder"]),
        float(slice_min["phase_1_starcoder"]),
        float(slice_min[TARGET]),
        marker="*",
        markersize=13.5,
    )
    _projected_marker(
        ax,
        float(proportional["phase_0_starcoder"]),
        float(proportional["phase_1_starcoder"]),
        float(proportional[TARGET]),
        marker="o",
        markersize=8.2,
        color=PROPORTIONAL_GOLD,
    )
    projected_x, projected_y, _ = proj3d.proj_transform(
        float(proportional["phase_0_starcoder"]),
        float(proportional["phase_1_starcoder"]),
        float(proportional[TARGET]),
        ax.get_proj(),
    )
    ax.text2D(
        projected_x + 0.018,
        projected_y + 0.01,
        "proportional",
        transform=ax.transData,
        fontsize=6.2,
        color=PAPER_TEXT,
        bbox={"boxstyle": "round,pad=0.12", "facecolor": "white", "edgecolor": "none", "alpha": 0.72},
        zorder=1001,
        clip_on=False,
    )


def _plot_slice(
    ax: Axes,
    slice_rows: pd.DataFrame,
    starcoder_epoch_multiplier: float,
) -> None:
    """Plot the U-shaped first-phase-Nemotron slice."""
    x = slice_rows["phase_1_starcoder"].to_numpy(dtype=float)
    y = slice_rows[TARGET].to_numpy(dtype=float)
    slice_min_idx = int(np.argmin(y))
    slice_min_x = float(x[slice_min_idx])
    slice_min_y = float(y[slice_min_idx])
    slice_min_epochs = slice_min_x * starcoder_epoch_multiplier

    ax.plot(x, y, color=PAPER_MUTED, linewidth=1.35, linestyle="--", alpha=0.72, zorder=1)
    ax.scatter(x, y, s=30, color=GRP_COLOR, edgecolors="white", linewidths=0.5, zorder=3)
    ax.scatter(
        [slice_min_x],
        [slice_min_y],
        s=92,
        color=UNIFORM_COLOR,
        marker="*",
        edgecolors="white",
        linewidths=0.55,
        zorder=4,
    )
    ax.axvline(slice_min_x, color=UNIFORM_COLOR, linewidth=1.3, linestyle=":", alpha=0.95, zorder=2)
    ax.annotate(
        rf"slice minimum at $p^{{(1)}}={slice_min_x:.2f}$" "\n" f"({slice_min_epochs:.1f} StarCoder epochs)",
        xy=(slice_min_x, slice_min_y),
        xytext=(0.47, 1.56),
        textcoords="data",
        arrowprops={"arrowstyle": "->", "color": UNIFORM_COLOR, "linewidth": 1.0},
        color=PAPER_TEXT,
        fontsize=9.8,
        ha="left",
        va="center",
        bbox={
            "boxstyle": "round,pad=0.22",
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.86,
        },
    )

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(float(y.min()) - 0.05, float(y.max()) + 0.06)
    ax.set_xlabel(r"Phase 1 StarCoder weight ($p^{(1)}$)", fontsize=11, color=PAPER_TEXT)
    ax.set_ylabel("Code BPB", fontsize=11, color=PAPER_TEXT)
    top_axis = ax.secondary_xaxis(
        "top",
        functions=(
            lambda weight: weight * starcoder_epoch_multiplier,
            lambda epochs: epochs / starcoder_epoch_multiplier,
        ),
    )
    top_axis.set_xlabel("Phase 1 StarCoder epochs", fontsize=10, color=PAPER_TEXT, labelpad=7)
    top_axis.tick_params(labelsize=9, colors=PAPER_TEXT)
    top_axis.spines["top"].set_color(PAPER_AXIS)
    top_axis.spines["top"].set_linewidth(0.9)


def main() -> None:
    """Render F1 artifacts."""
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    frame = _completed_frame()
    slice_rows = _slice_frame(frame)
    starcoder_epoch_multiplier = _phase_1_starcoder_epoch_multiplier(frame)

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "axes.edgecolor": PAPER_AXIS,
            "axes.labelcolor": PAPER_TEXT,
            "text.color": PAPER_TEXT,
            "xtick.color": PAPER_TEXT,
            "ytick.color": PAPER_TEXT,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    landscape_figure = plt.figure(figsize=LANDSCAPE_FIGSIZE, constrained_layout=False)
    landscape_figure.subplots_adjust(left=0.02, right=0.98, bottom=0.09, top=0.96)
    landscape_axis = landscape_figure.add_subplot(1, 1, 1, projection="3d")
    _plot_landscape(landscape_axis, frame, slice_rows)
    landscape_figure.savefig(LANDSCAPE_OUTPUT_STEM.with_suffix(".png"), dpi=DPI, bbox_inches="tight", pad_inches=0.24)
    landscape_figure.savefig(LANDSCAPE_OUTPUT_STEM.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.24)
    plt.close(landscape_figure)

    error_bar_figure = plt.figure(figsize=LANDSCAPE_FIGSIZE, constrained_layout=False)
    error_bar_figure.subplots_adjust(left=0.02, right=0.98, bottom=0.09, top=0.96)
    error_bar_axis = error_bar_figure.add_subplot(1, 1, 1, projection="3d")
    _plot_landscape(error_bar_axis, frame, slice_rows, show_anchor_error_bars=True)
    error_bar_figure.savefig(
        LANDSCAPE_ERROR_BARS_OUTPUT_STEM.with_suffix(".png"), dpi=DPI, bbox_inches="tight", pad_inches=0.24
    )
    error_bar_figure.savefig(LANDSCAPE_ERROR_BARS_OUTPUT_STEM.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.24)
    plt.close(error_bar_figure)

    slice_figure, slice_axis = plt.subplots(figsize=SLICE_FIGSIZE, constrained_layout=True)
    _plot_slice(slice_axis, slice_rows, starcoder_epoch_multiplier)
    _style_axis(slice_axis)
    slice_figure.savefig(SLICE_OUTPUT_STEM.with_suffix(".png"), dpi=DPI, bbox_inches="tight", pad_inches=0.04)
    slice_figure.savefig(SLICE_OUTPUT_STEM.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.04)
    plt.close(slice_figure)

    print(f"Wrote {LANDSCAPE_OUTPUT_STEM.with_suffix('.png')}")
    print(f"Wrote {LANDSCAPE_OUTPUT_STEM.with_suffix('.pdf')}")
    print(f"Wrote {LANDSCAPE_ERROR_BARS_OUTPUT_STEM.with_suffix('.png')}")
    print(f"Wrote {LANDSCAPE_ERROR_BARS_OUTPUT_STEM.with_suffix('.pdf')}")
    print(f"Wrote {SLICE_OUTPUT_STEM.with_suffix('.png')}")
    print(f"Wrote {SLICE_OUTPUT_STEM.with_suffix('.pdf')}")
    print(f"Completed rows: {len(frame)}; slice rows: {len(slice_rows)}")


if __name__ == "__main__":
    main()
