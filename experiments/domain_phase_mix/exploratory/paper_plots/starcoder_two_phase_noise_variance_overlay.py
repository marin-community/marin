# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib", "numpy", "pandas"]
# ///
"""Render slide-friendly StarCoder repeat-noise variance and std overlays."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
from matplotlib.ticker import FuncFormatter, NullFormatter
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d

from experiments.domain_phase_mix.exploratory.paper_plots import (
    starcoder_two_phase_heteroskedastic_landscape as hetero,
)
from experiments.domain_phase_mix.exploratory.paper_plots.paper_plot_style import (
    PAPER_AXIS,
    PAPER_GRID,
    PAPER_MUTED,
    PAPER_TEXT,
)

SCRIPT_DIR = Path(__file__).resolve().parent
IMG_DIR = SCRIPT_DIR / "img"

METRIC = hetero.TARGET
FIGSIZE = (4.7, 4.3)
DPI = 300
VARIANCE_CAP = 1.0e-4
STD_CAP = 1.0e-2
LOW_VARIANCE_LABEL_FLOOR = 1.6e-6
LOW_STD_LABEL_FLOOR = 1.4e-3
PROPORTIONAL_GOLD = "#FFD700"
LABEL_OFFSETS = {
    "early_code_high_late_low": (0.06, 0.025, 2.1),
    "late_code_moderate": (0.035, 0.00, 1.55),
    "observed_global_best": (0.00, 0.035, 1.45),
    "observed_p0_zero_slice_best": (0.045, 0.02, 1.35),
    "nemotron_only": (0.03, -0.02, 1.02),
    "proportional": (0.05, -0.015, 1.45),
}
ANCHOR_DISPLAY_NAMES = {
    "nemotron_only": "Nemotron only",
    "starcoder_only": "StarCoder only",
    "proportional": "proportional",
    "observed_global_best": "global best",
}
SCREEN_LABEL_OFFSETS = {
    "nemotron_only": (0.005, 0.055),
    "proportional": (0.052, 0.020),
    "observed_global_best": (0.020, 0.030),
}


def _style_3d_axis(ax: Axes3D) -> None:
    """Apply the same 3D styling used by the base StarCoder landscape."""
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


def _setup_matplotlib() -> None:
    """Configure shared static plot typography."""
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


def _anchor_summary() -> pd.DataFrame:
    """Return per-anchor Code-BPB repeat noise summaries."""
    repeat_panel = hetero.load_repeat_panel()
    summary = hetero.summarize_anchor_noise(repeat_panel.final_rows, metrics=[METRIC])
    rows = summary[summary["metric"].eq(METRIC)].copy()
    if rows.empty:
        raise ValueError(f"No anchor summaries found for {METRIC}")
    return rows


def _marker_sizes(log_variance: np.ndarray) -> np.ndarray:
    span = float(np.nanmax(log_variance) - np.nanmin(log_variance))
    if not np.isfinite(span) or span <= 0:
        return np.full(log_variance.shape, 82.0)
    normalized = (log_variance - float(np.nanmin(log_variance))) / span
    return 42.0 + 150.0 * normalized


def _scientific_label(value: float) -> str:
    return f"{value:.1e}"


def _capped_scientific_label(value: float, cap: float) -> str:
    if np.isclose(value, cap):
        return rf"$\geq${_scientific_label(cap)}"
    return _scientific_label(value)


def _data_to_axes_coordinates(ax: Axes3D, x: float, y: float, z: float) -> tuple[float, float]:
    projected_x, projected_y, _ = proj3d.proj_transform(x, y, z, ax.get_proj())
    display_x, display_y = ax.transData.transform((projected_x, projected_y))
    axes_x, axes_y = ax.transAxes.inverted().transform((display_x, display_y))
    return float(axes_x), float(axes_y)


def _draw_offscale_callout(ax: Axes3D, anchors: pd.DataFrame, *, value_column: str, cap: float) -> None:
    """Draw true off-chart values above the 3D box."""
    offscale = anchors[anchors[value_column].gt(cap)]
    if offscale.empty:
        return
    row = offscale.sort_values(value_column, ascending=False).iloc[0]
    figure = ax.figure
    base_x, base_y = _data_to_axes_coordinates(
        ax,
        float(row["phase_0_starcoder"]),
        float(row["phase_1_starcoder"]),
        cap,
    )
    marker_x = base_x
    marker_y = min(0.96, base_y + 0.14)
    line = Line2D(
        [marker_x, marker_x],
        [base_y + 0.01, marker_y - 0.025],
        transform=ax.transAxes,
        color=PAPER_MUTED,
        linewidth=1.05,
        alpha=0.75,
        clip_on=False,
    )
    figure.add_artist(line)
    circle = Circle(
        (marker_x, marker_y),
        radius=0.024,
        transform=ax.transAxes,
        facecolor="#B3002D",
        edgecolor="white",
        linewidth=0.75,
        alpha=0.98,
        clip_on=False,
    )
    figure.add_artist(circle)
    ax.text2D(
        marker_x,
        marker_y + 0.045,
        f"{ANCHOR_DISPLAY_NAMES.get(str(row['anchor_id']), str(row['anchor_id']))}\n"
        f"{_scientific_label(float(row[value_column]))}\noff chart",
        transform=ax.transAxes,
        fontsize=6.4,
        color=PAPER_TEXT,
        ha="center",
        va="bottom",
        bbox={
            "boxstyle": "round,pad=0.14",
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.72,
        },
    )


def _plot_noise_overlay(
    ax: Axes3D,
    anchors: pd.DataFrame,
    *,
    value_column: str,
    cap: float,
    label_floor: float,
    axis_label: str,
    span_label: str,
) -> None:
    """Plot sampled anchors with a raw repeat-noise statistic on the z axis."""
    anchor_x = anchors["phase_0_starcoder"].to_numpy(dtype=float)
    anchor_y = anchors["phase_1_starcoder"].to_numpy(dtype=float)
    values = anchors[value_column].to_numpy(dtype=float)
    color_values = np.minimum(values, cap)
    log_values = np.log10(values)
    onscale = values <= cap
    value_floor = float(np.nanmin(values) * 0.45)
    value_norm = LogNorm(vmin=float(np.nanmin(values)), vmax=cap)

    for row in anchors.itertuples(index=False):
        row_value = float(getattr(row, value_column))
        z_value = min(row_value, cap)
        ax.plot(
            [float(row.phase_0_starcoder), float(row.phase_0_starcoder)],
            [float(row.phase_1_starcoder), float(row.phase_1_starcoder)],
            [value_floor, z_value],
            color=PAPER_MUTED,
            linewidth=1.05,
            alpha=0.62,
        )

    scatter = ax.scatter(
        anchor_x[onscale],
        anchor_y[onscale],
        values[onscale],
        c=color_values[onscale],
        cmap="RdYlGn_r",
        norm=value_norm,
        s=_marker_sizes(log_values[onscale]),
        edgecolors="white",
        linewidths=0.85,
        alpha=0.98,
        depthshade=False,
        axlim_clip=False,
    )

    screen_labels: list[tuple[str, float, float, float, str]] = []
    for row in anchors.itertuples(index=False):
        dx, dy, z_multiplier = LABEL_OFFSETS.get(str(row.anchor_id), (0.0, 0.0, 1.16))
        row_value = float(getattr(row, value_column))
        is_offscale = row_value > cap
        if is_offscale:
            continue
        anchor_id = str(row.anchor_id)
        label = _scientific_label(row_value)
        if anchor_id in ANCHOR_DISPLAY_NAMES:
            label = f"{ANCHOR_DISPLAY_NAMES[anchor_id]}\n{label}"
            screen_labels.append(
                (
                    anchor_id,
                    float(row.phase_0_starcoder),
                    float(row.phase_1_starcoder),
                    row_value,
                    label,
                )
            )
            continue
        label_z = row_value * z_multiplier
        label_z = max(label_z, label_floor)
        text = ax.text(
            float(row.phase_0_starcoder) + dx,
            float(row.phase_1_starcoder) + dy,
            label_z,
            label,
            fontsize=5.7,
            color=PAPER_TEXT,
            ha="center",
            va="bottom",
            bbox={
                "boxstyle": "round,pad=0.12",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.68,
            },
        )
        text.set_clip_on(False)

        if anchor_id == "proportional":
            ax.scatter(
                [float(row.phase_0_starcoder)],
                [float(row.phase_1_starcoder)],
                [row_value],
                s=150,
                marker="o",
                facecolors="none",
                edgecolors=PROPORTIONAL_GOLD,
                linewidths=2.2,
                alpha=0.98,
                depthshade=False,
                axlim_clip=False,
            )

    log_span = float(np.nanmax(log_values) - np.nanmin(log_values))
    ax.text2D(
        0.035,
        0.91,
        rf"{span_label} spans $\approx {log_span:.1f}$ orders" "\n" "5 repeats per anchor",
        transform=ax.transAxes,
        fontsize=7.0,
        color=PAPER_TEXT,
        bbox={
            "boxstyle": "round,pad=0.23",
            "facecolor": "white",
            "edgecolor": PAPER_GRID,
            "alpha": 0.88,
        },
    )

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_zlim(value_floor, cap)
    ax.set_zscale("log")
    ax.set_xlabel(r"Phase 0 StarCoder ($p^{(0)}$)", fontsize=9.5, color=PAPER_TEXT, labelpad=4)
    ax.set_ylabel(r"Phase 1 StarCoder ($p^{(1)}$)", fontsize=9.5, color=PAPER_TEXT, labelpad=4)
    ax.set_zlabel("")
    ax.text2D(
        0.018,
        0.45,
        axis_label,
        transform=ax.transAxes,
        rotation=90,
        fontsize=9.5,
        color=PAPER_TEXT,
        ha="left",
        va="center",
    )
    tick_candidates = [10.0**power for power in range(-8, 1)]
    z_ticks = [tick for tick in tick_candidates if value_floor <= tick < cap]
    z_ticks.append(cap)
    ax.set_zticks(z_ticks)
    ax.zaxis.set_major_formatter(FuncFormatter(lambda value, _: _capped_scientific_label(float(value), cap)))
    ax.zaxis.set_minor_formatter(NullFormatter())
    ax.view_init(elev=25, azim=-141)
    ax.set_box_aspect((1.0, 1.0, 0.95), zoom=1.0)
    _style_3d_axis(ax)
    _draw_offscale_callout(ax, anchors, value_column=value_column, cap=cap)
    for anchor_id, x, y, value, label in screen_labels:
        axes_x, axes_y = _data_to_axes_coordinates(ax, x, y, value)
        dx, dy = SCREEN_LABEL_OFFSETS.get(anchor_id, (0.0, 0.04))
        text = ax.text2D(
            axes_x + dx,
            axes_y + dy,
            label,
            transform=ax.transAxes,
            fontsize=5.7,
            color=PAPER_TEXT,
            ha="center",
            va="bottom",
            bbox={
                "boxstyle": "round,pad=0.12",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.80,
            },
            zorder=1005,
            clip_on=False,
        )
        text.set_clip_on(False)
    return scatter


def _render_noise_plot(
    anchors: pd.DataFrame,
    *,
    output_stem: Path,
    value_column: str,
    cap: float,
    label_floor: float,
    axis_label: str,
    colorbar_label: str,
    span_label: str,
    preserve_axis_viewport: bool = False,
    include_colorbar: bool = True,
) -> None:
    figure = plt.figure(figsize=FIGSIZE, constrained_layout=False)
    if preserve_axis_viewport:
        figure.subplots_adjust(left=0.02, right=0.98, bottom=0.09, top=0.96)
    else:
        figure.subplots_adjust(left=0.04, right=0.88, bottom=0.13, top=0.96)
    axis = figure.add_subplot(1, 1, 1, projection="3d")
    scatter = _plot_noise_overlay(
        axis,
        anchors,
        value_column=value_column,
        cap=cap,
        label_floor=label_floor,
        axis_label=axis_label,
        span_label=span_label,
    )
    if include_colorbar:
        if preserve_axis_viewport:
            colorbar_axis = figure.add_axes([0.865, 0.29, 0.026, 0.46])
            colorbar = figure.colorbar(scatter, cax=colorbar_axis)
        else:
            colorbar = figure.colorbar(scatter, ax=axis, shrink=0.64, pad=0.025, fraction=0.055)
        colorbar.set_label(colorbar_label, color=PAPER_TEXT, fontsize=9.2)
        tick_candidates = [10.0**power for power in range(-8, 1)]
        colorbar_ticks = [tick for tick in tick_candidates if float(anchors[value_column].min()) <= tick < cap]
        colorbar_ticks.append(cap)
        colorbar.set_ticks(colorbar_ticks)
        colorbar.ax.yaxis.set_major_formatter(
            FuncFormatter(lambda value, _: _capped_scientific_label(float(value), cap))
        )
        colorbar.ax.tick_params(colors=PAPER_TEXT, labelsize=8.4)
        colorbar.outline.set_edgecolor(PAPER_AXIS)
        colorbar.outline.set_linewidth(0.6)

    figure.savefig(output_stem.with_suffix(".png"), dpi=DPI, bbox_inches="tight", pad_inches=0.16)
    figure.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.16)
    plt.close(figure)
    print(f"Wrote {output_stem.with_suffix('.png')}")
    print(f"Wrote {output_stem.with_suffix('.pdf')}")


def main() -> None:
    """Render static repeat-noise plots."""
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    _setup_matplotlib()
    anchors = _anchor_summary()
    anchors["std"] = np.sqrt(anchors["variance"])

    _render_noise_plot(
        anchors,
        output_stem=IMG_DIR / "starcoder_two_phase_noise_variance_overlay",
        value_column="variance",
        cap=VARIANCE_CAP,
        label_floor=LOW_VARIANCE_LABEL_FLOOR,
        axis_label="Repeat variance",
        colorbar_label=r"$\mathrm{Var}[\epsilon \mid w]$",
        span_label="repeat variance",
    )
    _render_noise_plot(
        anchors,
        output_stem=IMG_DIR / "starcoder_two_phase_noise_std_overlay",
        value_column="std",
        cap=STD_CAP,
        label_floor=LOW_STD_LABEL_FLOOR,
        axis_label="Repeat std",
        colorbar_label=r"$\mathrm{sd}[\epsilon \mid w]$",
        span_label="repeat std",
    )
    _render_noise_plot(
        anchors,
        output_stem=IMG_DIR / "starcoder_two_phase_noise_std_overlay_slide",
        value_column="std",
        cap=STD_CAP,
        label_floor=LOW_STD_LABEL_FLOOR,
        axis_label="Repeat std",
        colorbar_label=r"$\mathrm{sd}[\epsilon \mid w]$",
        span_label="repeat std",
        preserve_axis_viewport=True,
        include_colorbar=False,
    )
    variance_log_span = float(anchors["log10_variance"].max() - anchors["log10_variance"].min())
    std_log_span = variance_log_span / 2.0
    print(f"Code-BPB repeat variance span: {variance_log_span:.3f} orders of magnitude")
    print(f"Code-BPB repeat std span: {std_log_span:.3f} orders of magnitude")


if __name__ == "__main__":
    main()
