# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["matplotlib>=3.9", "numpy", "pandas", "plotly"]
# ///

"""Build the two-panel StarCoder replay motivation figure for the paper."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from matplotlib.axes import Axes
from matplotlib.figure import Figure as MatplotlibFigure
from matplotlib.patches import Rectangle
from matplotlib.ticker import FuncFormatter
from plotly.subplots import make_subplots

SCRIPT_DIR = Path(__file__).resolve().parent
REFERENCE_OUTPUTS = SCRIPT_DIR / "reference_outputs"
DEFAULT_ATLAS_DIR = REFERENCE_OUTPUTS / "starcoder_all_tied_curves_canonical_dsp_20260902"
DEFAULT_DESIGN = SCRIPT_DIR.parents[1] / "starcoder_wsd80_dense_support_surface_design_20260808.json"
DEFAULT_OUTPUT_DIR = REFERENCE_OUTPUTS / "starcoder_replay_background_figure_20260902"

REPLAY_CURVES = ("C15", "C16", "C17", "C18", "C19", "C20", "C21")
SCALE_CURVES = ("C21", "C28", "C35", "C42")
EXPECTED_POINTS = 26

PAPER = "#ffffff"
PANEL = "#ffffff"
INK = "#111111"
MUTED = "#333333"
GRID = "#b8b8b8"
MINIMUM = "#111111"
FIGURE_WIDTH = 1080
FIGURE_HEIGHT = 820
STATIC_FIGURE_WIDTH = 7.2
STATIC_FIGURE_HEIGHT = 4.6
STATIC_DPI = 300
DEEPSEEK_COLORS = (
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
)
REPLAY_DISPLAY_CURVES = tuple(reversed(REPLAY_CURVES))
REPLAY_DISPLAY_COLORS = tuple(reversed(DEEPSEEK_COLORS[: len(REPLAY_CURVES)]))
SCALE_DISPLAY_COLORS = (DEEPSEEK_COLORS[6], "#80cdc1", "#35978f", "#01665e")
PLOTLY_CONFIG = {
    "displaylogo": False,
    "responsive": True,
    "toImageButtonOptions": {"format": "png", "scale": 4},
}


@dataclass(frozen=True)
class CurveMetadata:
    """Physical training and support metadata for one plotted curve."""

    curve_ref: str
    curve_id: str
    training_tokens: int
    support_tokens: int
    support_fraction: float
    epoch_multiplier: float | None

    @property
    def matches_target_burden(self) -> bool:
        """True when the support reproduces the target budget's repetition (epoch multiplier 1)."""
        return self.epoch_multiplier is not None and np.isclose(self.epoch_multiplier, 1.0)

    @property
    def epochs_at_full_share(self) -> float:
        return self.training_tokens / self.support_tokens

    @property
    def label(self) -> str:
        pool_note = " (full pool)" if np.isclose(self.support_fraction, 1.0) else ""
        return (
            f"D={format_training_budget(self.training_tokens)}, "
            f"D/S_StarCoder={format_epoch_count(self.epochs_at_full_share)} epochs, "
            f"S_StarCoder={format_support_size(self.support_tokens)}{pool_note}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--atlas-dir", type=Path, default=DEFAULT_ATLAS_DIR)
    parser.add_argument("--design", type=Path, default=DEFAULT_DESIGN)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def format_training_budget(tokens: int) -> str:
    value = tokens / 1e9
    if np.isclose(value, round(value), atol=0.015):
        return f"{round(value):.0f}B"
    return f"{value:.2f}B"


def format_support_size(tokens: int) -> str:
    if tokens >= 1e9:
        return f"{tokens / 1e9:.1f}B"
    if tokens >= 10e6:
        return f"{tokens / 1e6:.1f}M"
    return f"{tokens / 1e6:.2f}M"


def format_epoch_count(epochs: float) -> str:
    if epochs < 0.01:
        return f"{epochs:.4f}"
    if epochs < 0.1:
        return f"{epochs:.3f}"
    return f"{epochs:.1f}"


def preserve_spaces(text: str) -> str:
    return text.replace(" ", "&nbsp;")


def legend_label(metadata: CurveMetadata, optimum_epochs: float) -> str:
    support = format_support_size(metadata.support_tokens)
    if np.isclose(metadata.support_fraction, 1.0):
        support += " (full)"
    row = (
        f"{format_training_budget(metadata.training_tokens):>5} | "
        f"{support:<13} | "
        f"{format_epoch_count(metadata.epochs_at_full_share):>6} | "
        f"{format_epoch_count(optimum_epochs):>6}"
    )
    return preserve_spaces(row)


def legend_title() -> str:
    header = f"{'D':>5} | {'S_SC':<13} | {'D/S_SC':>6} | {'E*_SC':>6}"
    return preserve_spaces("      " + header)


def curve_parts(curve_id: str) -> tuple[str, str]:
    parts = curve_id.split("__")
    if len(parts) != 4 or parts[0] != "dense_replay" or parts[-1] != "endpoint":
        raise ValueError(f"Unexpected replay curve id: {curve_id}")
    return parts[1], parts[2]


def load_inputs(
    atlas_dir: Path,
    design_path: Path,
) -> tuple[pd.DataFrame, dict[str, CurveMetadata]]:
    references = pd.read_csv(atlas_dir / "curve_reference.csv")
    predictions = pd.read_csv(atlas_dir / "predictions.csv")
    selected_refs = set(REPLAY_CURVES) | set(SCALE_CURVES)
    references = references.loc[references["curve_ref"].isin(selected_refs)].copy()
    if set(references["curve_ref"]) != selected_refs:
        missing = selected_refs - set(references["curve_ref"])
        raise ValueError(f"Atlas is missing required curves: {sorted(missing)}")

    design = json.loads(design_path.read_text(encoding="utf-8"))
    cells = pd.DataFrame(design["cells"]).set_index("cell_id")
    supports = pd.DataFrame(design["supports"]).set_index(["cell_id", "support_id"])

    metadata: dict[str, CurveMetadata] = {}
    for row in references.itertuples(index=False):
        cell_id, support_id = curve_parts(row.curve_id)
        cell = cells.loc[cell_id]
        support = supports.loc[(cell_id, support_id)]
        metadata[row.curve_ref] = CurveMetadata(
            curve_ref=row.curve_ref,
            curve_id=row.curve_id,
            training_tokens=int(cell["materialized_tokens"]),
            support_tokens=int(support["starcoder_realized_support_tokens"]),
            support_fraction=float(support["starcoder_support_fraction"]),
            epoch_multiplier=None if pd.isna(support["epoch_multiplier"]) else float(support["epoch_multiplier"]),
        )

    points = predictions.loc[predictions["curve_ref"].isin(selected_refs)].copy()
    points = points.sort_values(["curve_number", "starcoder_weight"])
    counts = points.groupby("curve_ref").size()
    if not counts.eq(EXPECTED_POINTS).all():
        raise ValueError(f"Expected {EXPECTED_POINTS} observations per curve, got {counts.to_dict()}")
    return points, metadata


def custom_data(group: pd.DataFrame, metadata: CurveMetadata) -> np.ndarray:
    shares = group["starcoder_weight"].to_numpy(dtype=float)
    values = np.empty((len(group), 4), dtype=object)
    values[:, 0] = metadata.curve_ref
    values[:, 1] = metadata.label
    values[:, 2] = shares * metadata.epochs_at_full_share
    values[:, 3] = metadata.epochs_at_full_share
    return values


def add_panel_curves(
    figure: go.Figure,
    *,
    points: pd.DataFrame,
    metadata: dict[str, CurveMetadata],
    curve_refs: tuple[str, ...],
    column: int,
    colors: list[str],
    legend_name: str,
) -> None:
    for curve_ref, color in zip(curve_refs, colors, strict=True):
        group = points.loc[points["curve_ref"].eq(curve_ref)].sort_values("starcoder_weight")
        curve = metadata[curve_ref]
        minimum = group.loc[group["observed_bpb"].idxmin()]
        optimum_epochs = float(minimum["starcoder_weight"]) * curve.epochs_at_full_share
        is_no_repetition = curve_ref == REPLAY_CURVES[0]
        figure.add_trace(
            go.Scatter(
                x=group["starcoder_weight"],
                y=group["observed_bpb"],
                mode="lines+markers",
                name=legend_label(curve, optimum_epochs),
                legend=legend_name,
                legendgroup=f"{legend_name}-{curve_ref}",
                line={
                    "color": color,
                    "width": 3.0 if is_no_repetition else 2.6,
                    "dash": "dash" if is_no_repetition else "solid",
                },
                marker={
                    "color": color,
                    "size": 7 if is_no_repetition else 5.5,
                    "symbol": "circle-open" if is_no_repetition else "circle",
                    "line": {"color": INK, "width": 0.8 if is_no_repetition else 0.5},
                },
                zorder=5 if is_no_repetition else 2,
                customdata=custom_data(group, curve),
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "%{customdata[1]}<br>"
                    "StarCoder fraction p: %{x:.4f}<br>"
                    "Materialized StarCoder epochs: %{customdata[2]:.3f}<br>"
                    "Epochs at p=1: %{customdata[3]:.3f}<br>"
                    "Programming Languages BPB: %{y:.5f}<extra></extra>"
                ),
            ),
            row=1,
            col=column,
        )

        figure.add_trace(
            go.Scatter(
                x=[minimum["starcoder_weight"]],
                y=[minimum["observed_bpb"]],
                mode="markers",
                marker={
                    "color": color,
                    "size": 13,
                    "symbol": "star",
                    "line": {"color": MINIMUM, "width": 1.4},
                },
                legendgroup=f"{legend_name}-{curve_ref}",
                showlegend=False,
                hoverinfo="skip",
            ),
            row=1,
            col=column,
        )


def build_figure(points: pd.DataFrame, metadata: dict[str, CurveMetadata]) -> go.Figure:
    figure = make_subplots(
        rows=1,
        cols=2,
        horizontal_spacing=0.12,
        subplot_titles=(
            "<b>A · Fixed D, scaling downsampling</b>",
            "<b>B · Fixed downsampling, scaling D</b>",
        ),
    )
    add_panel_curves(
        figure,
        points=points,
        metadata=metadata,
        curve_refs=REPLAY_DISPLAY_CURVES,
        column=1,
        colors=list(REPLAY_DISPLAY_COLORS),
        legend_name="legend",
    )
    add_panel_curves(
        figure,
        points=points,
        metadata=metadata,
        curve_refs=SCALE_CURVES,
        column=2,
        colors=list(SCALE_DISPLAY_COLORS),
        legend_name="legend2",
    )
    figure.add_trace(
        go.Scatter(
            x=[0.0, 1.0],
            y=[None, None],
            xaxis="x3",
            yaxis="y2",
            mode="markers",
            showlegend=False,
            hoverinfo="skip",
        )
    )

    figure.update_xaxes(
        title_text="StarCoder mixture fraction, p",
        range=[-0.01, 1.01],
        tickvals=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
        tickformat=".1f",
        showline=True,
        linecolor=INK,
        linewidth=1.4,
        mirror=False,
        gridcolor=GRID,
        zeroline=False,
        row=1,
        col=1,
    )
    figure.update_xaxes(
        title_text="StarCoder mixture fraction, p",
        range=[-0.01, 1.01],
        tickvals=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
        tickformat=".1f",
        showline=True,
        linecolor=INK,
        linewidth=1.4,
        mirror=False,
        gridcolor=GRID,
        zeroline=False,
        row=1,
        col=2,
    )
    figure.update_yaxes(
        title_text="Programming Languages BPB (lower is better)",
        range=[0.78, 4.28],
        showline=True,
        linecolor=INK,
        linewidth=1.4,
        gridcolor=GRID,
        zeroline=False,
        row=1,
        col=1,
    )
    figure.update_yaxes(
        range=[0.78, 4.28],
        showline=True,
        linecolor=INK,
        linewidth=1.4,
        gridcolor=GRID,
        zeroline=False,
        row=1,
        col=2,
    )

    common_epoch_scale = float(np.mean([metadata[curve].epochs_at_full_share for curve in SCALE_CURVES]))
    top_ticks = np.linspace(0.0, 1.0, 6)
    for annotation in figure.layout.annotations:
        annotation.update(y=1.12, font={"size": 20, "color": INK})
    figure.update_layout(
        xaxis3={
            "overlaying": "x2",
            "anchor": "y2",
            "side": "top",
            "domain": figure.layout.xaxis2.domain,
            "range": [-0.01, 1.01],
            "tickmode": "array",
            "tickvals": top_ticks,
            "ticktext": [f"{value * common_epoch_scale:.0f}" for value in top_ticks],
            "title": {"text": "Materialized StarCoder epochs", "standoff": 8},
            "showline": False,
            "showgrid": False,
            "zeroline": False,
            "ticks": "outside",
            "ticklen": 4,
            "tickcolor": MUTED,
        },
        legend={
            "title": {
                "text": legend_title(),
                "font": {"family": "Menlo, Monaco, monospace", "size": 9.5},
            },
            "orientation": "v",
            "x": 0.00,
            "xanchor": "left",
            "y": 0.99,
            "yanchor": "top",
            "font": {"family": "Menlo, Monaco, monospace", "size": 9.5},
            "bgcolor": "rgba(255,255,255,0.97)",
            "bordercolor": GRID,
            "borderwidth": 1,
            "itemsizing": "constant",
        },
        legend2={
            "title": {
                "text": legend_title(),
                "font": {"family": "Menlo, Monaco, monospace", "size": 9.5},
            },
            "orientation": "v",
            "x": 0.56,
            "xanchor": "left",
            "y": 0.99,
            "yanchor": "top",
            "font": {"family": "Menlo, Monaco, monospace", "size": 9.5},
            "bgcolor": "rgba(255,255,255,0.97)",
            "bordercolor": GRID,
            "borderwidth": 1,
            "itemsizing": "constant",
        },
        updatemenus=[
            {
                "type": "buttons",
                "direction": "right",
                "x": 0.5,
                "xanchor": "center",
                "y": -0.14,
                "yanchor": "top",
                "showactive": True,
                "buttons": [
                    {
                        "label": "Full range",
                        "method": "relayout",
                        "args": [{"yaxis.range": [0.78, 4.28], "yaxis2.range": [0.78, 4.28]}],
                    },
                    {
                        "label": "Zoom near minima",
                        "method": "relayout",
                        "args": [{"yaxis.range": [0.78, 1.85], "yaxis2.range": [0.78, 1.85]}],
                    },
                ],
            }
        ],
        annotations=[*figure.layout.annotations],
        width=FIGURE_WIDTH,
        height=FIGURE_HEIGHT,
        margin={"l": 90, "r": 35, "t": 150, "b": 145},
        paper_bgcolor=PAPER,
        plot_bgcolor=PANEL,
        font={"family": "DejaVu Sans, Helvetica, sans-serif", "size": 15, "color": INK},
        hoverlabel={"bgcolor": PANEL, "font": {"family": "DejaVu Sans, Helvetica, sans-serif", "size": 13}},
    )
    return figure


def style_static_axis(axis: Axes) -> None:
    axis.set_xlim(-0.01, 1.01)
    axis.set_ylim(0.78, 4.28)
    axis.set_xticks(np.linspace(0.0, 1.0, 6))
    axis.set_yticks(np.arange(1.0, 4.1, 0.5))
    axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _position: f"{value:g}"))
    axis.set_axisbelow(True)
    axis.grid(color=GRID, linewidth=0.65, linestyle="-", alpha=0.72)
    axis.tick_params(axis="both", colors=INK, labelsize=7.5, width=0.8, length=3)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    for name in ("left", "bottom"):
        axis.spines[name].set_color(INK)
        axis.spines[name].set_linewidth(0.8)


def add_static_legend_table(
    axis: Axes,
    *,
    rows: list[list[str]],
    colors: list[str],
) -> None:
    x = 0.012
    width = 0.78
    table_top = 0.985
    row_height = 0.047
    table_height = row_height * (len(rows) + 1)
    table_bottom = table_top - table_height
    box_bottom = table_bottom - 0.012
    box_top = table_top + 0.006
    spaced_rows = [[row[0], row[1], "", *row[2:]] for row in rows]

    axis.add_patch(
        Rectangle(
            (x, box_bottom),
            width,
            box_top - box_bottom,
            transform=axis.transAxes,
            facecolor="white",
            edgecolor="#b8b8b8",
            linewidth=0.5,
            zorder=9,
        )
    )
    table = axis.table(
        cellText=spaced_rows,
        colLabels=[
            "",
            "$D$",
            "",
            "$S_{\\mathrm{SC}}$",
            "$D/S_{\\mathrm{SC}}$",
            "$E_{\\mathrm{SC}}^*$",
        ],
        colWidths=[0.075, 0.125, 0.09, 0.26, 0.225, 0.225],
        cellLoc="right",
        colLoc="right",
        bbox=[x + 0.006, table_bottom, width - 0.012, table_height],
    )
    table.set_zorder(11)
    table.auto_set_font_size(False)
    table.set_fontsize(6.6)
    for (row_index, column_index), cell in table.get_celld().items():
        cell.set_facecolor("white")
        cell.set_edgecolor("none")
        cell.PAD = 0.035
        text = cell.get_text()
        text.set_color(INK)
        text.set_fontfamily("DejaVu Sans")
        if row_index == 0:
            text.set_fontweight("bold")
            cell.visible_edges = "B"
            cell.set_edgecolor("#9a9a9a")
            cell.set_linewidth(0.45)
        if column_index == 0:
            text.set_ha("center")

    for row_index, color in enumerate(colors, start=1):
        marker = table[(row_index, 0)].get_text()
        marker.set_text("●")
        marker.set_color(color)
        marker.set_fontsize(9.0)
        marker.set_path_effects([path_effects.withStroke(linewidth=0.55, foreground=INK)])


def add_static_panel(
    axis: Axes,
    *,
    points: pd.DataFrame,
    metadata: dict[str, CurveMetadata],
    curve_refs: tuple[str, ...],
    colors: list[str],
) -> None:
    legend_rows: list[list[str]] = []
    for curve_ref, color in zip(curve_refs, colors, strict=True):
        group = points.loc[points["curve_ref"].eq(curve_ref)].sort_values("starcoder_weight")
        curve = metadata[curve_ref]
        minimum = group.loc[group["observed_bpb"].idxmin()]
        optimum_epochs = float(minimum["starcoder_weight"]) * curve.epochs_at_full_share
        is_no_repetition = curve_ref == REPLAY_CURVES[0]
        line_style = (0, (4, 2)) if is_no_repetition else "-"
        if is_no_repetition:
            axis.plot(
                group["starcoder_weight"],
                group["observed_bpb"],
                color="white",
                linewidth=3.4,
                linestyle=line_style,
                zorder=4,
            )
        axis.plot(
            group["starcoder_weight"],
            group["observed_bpb"],
            color=color,
            linewidth=1.9 if is_no_repetition else 1.5,
            linestyle=line_style,
            marker="o",
            markersize=3.7 if is_no_repetition else 3.0,
            markerfacecolor="white" if is_no_repetition else color,
            markeredgecolor=color if is_no_repetition else INK,
            markeredgewidth=0.9 if is_no_repetition else 0.25,
            zorder=5 if is_no_repetition else 2,
        )
        axis.scatter(
            [minimum["starcoder_weight"]],
            [minimum["observed_bpb"]],
            color=color,
            edgecolor=INK,
            linewidth=0.7,
            marker="*",
            s=62,
            zorder=7,
        )
        support = format_support_size(curve.support_tokens)
        if np.isclose(curve.support_fraction, 1.0):
            support += " (full)"
        if curve.matches_target_burden:
            support += " (target)"
        legend_rows.append(
            [
                "",
                format_training_budget(curve.training_tokens),
                support,
                format_epoch_count(curve.epochs_at_full_share),
                format_epoch_count(optimum_epochs),
            ]
        )

    add_static_legend_table(axis, rows=legend_rows, colors=colors)


def build_static_figure(points: pd.DataFrame, metadata: dict[str, CurveMetadata]) -> MatplotlibFigure:
    with plt.rc_context(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "text.usetex": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    ):
        figure, axes = plt.subplots(
            1,
            2,
            figsize=(STATIC_FIGURE_WIDTH, STATIC_FIGURE_HEIGHT),
            sharey=False,
        )
        figure.subplots_adjust(left=0.09, right=0.99, bottom=0.14, top=0.86, wspace=0.24)

        add_static_panel(
            axes[0],
            points=points,
            metadata=metadata,
            curve_refs=REPLAY_DISPLAY_CURVES,
            colors=list(REPLAY_DISPLAY_COLORS),
        )
        add_static_panel(
            axes[1],
            points=points,
            metadata=metadata,
            curve_refs=SCALE_CURVES,
            colors=list(SCALE_DISPLAY_COLORS),
        )

        for axis in axes:
            style_static_axis(axis)
            axis.set_xlabel(r"StarCoder mixture fraction, $p$", fontsize=8.5, color=INK, labelpad=7)
        axes[0].set_ylabel("Programming Languages BPB (lower is better)", fontsize=8.5, color=INK, labelpad=8)

        common_epoch_scale = float(np.mean([metadata[curve].epochs_at_full_share for curve in SCALE_CURVES]))
        top_axis = axes[1].secondary_xaxis(
            "top",
            functions=(lambda share: share * common_epoch_scale, lambda epochs: epochs / common_epoch_scale),
        )
        top_ticks = np.linspace(0.0, common_epoch_scale, 6)
        top_axis.set_xticks(top_ticks, labels=[f"{value:.0f}" for value in top_ticks])
        top_axis.set_xlabel("Materialized StarCoder epochs", fontsize=8.5, color=INK, labelpad=3)
        top_axis.tick_params(colors=INK, labelsize=7.5, width=0.8, length=0, pad=2)
        top_axis.spines["top"].set_visible(False)

        panel_titles = (
            r"A · Fixed $D$, scaling downsampling",
            r"B · Fixed downsampling, scaling $D$",
        )
        for axis, title in zip(axes, panel_titles, strict=True):
            bounds = axis.get_position()
            figure.text(
                (bounds.x0 + bounds.x1) / 2,
                0.955,
                title,
                ha="center",
                va="top",
                color=INK,
                fontsize=9.6,
                fontweight="bold",
                linespacing=1.2,
            )
        return figure


def write_outputs(
    output_dir: Path,
    figure: go.Figure,
    points: pd.DataFrame,
    metadata: dict[str, CurveMetadata],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    html = pio.to_html(
        figure,
        full_html=True,
        include_plotlyjs="inline",
        config=PLOTLY_CONFIG,
    )
    (output_dir / "index.html").write_text(html, encoding="utf-8")
    with plt.rc_context({"pdf.fonttype": 42, "ps.fonttype": 42}):
        static_figure = build_static_figure(points, metadata)
        static_figure.savefig(output_dir / "figure.png", dpi=STATIC_DPI)
        static_figure.savefig(output_dir / "figure.pdf")
        plt.close(static_figure)

    rows = []
    for curve_ref in (*REPLAY_CURVES, *SCALE_CURVES[1:]):
        curve = metadata[curve_ref]
        rows.append(
            {
                "curve_ref": curve_ref,
                "curve_id": curve.curve_id,
                "label": curve.label,
                "training_tokens": curve.training_tokens,
                "starcoder_support_tokens": curve.support_tokens,
                "starcoder_support_fraction": curve.support_fraction,
                "starcoder_epochs_at_p1": curve.epochs_at_full_share,
            }
        )
    pd.DataFrame(rows).to_csv(output_dir / "curve_metadata.csv", index=False)
    points.loc[points["curve_ref"].isin(set(REPLAY_CURVES) | set(SCALE_CURVES))].to_csv(
        output_dir / "observations.csv",
        index=False,
    )


def main() -> None:
    args = parse_args()
    points, metadata = load_inputs(args.atlas_dir, args.design)
    figure = build_figure(points, metadata)
    write_outputs(args.output_dir, figure, points, metadata)


if __name__ == "__main__":
    main()
