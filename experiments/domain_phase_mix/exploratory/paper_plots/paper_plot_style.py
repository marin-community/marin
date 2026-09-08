# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared styling for domain-mixture paper plots."""

from __future__ import annotations

from pathlib import Path

import plotly.graph_objects as go

PAPER_TEXT = "#232B32"
PAPER_MUTED = "#6C6F7D"
PAPER_GRID = "#E6E2DA"
PAPER_AXIS = "#A8A29A"
PAPER_BACKGROUND = "white"

GRP_COLOR = "#232B32"
PROPORTIONAL_COLOR = "#8F6B38"
OLMIX_COLOR = "#4C78A8"
UNIFORM_COLOR = "#E24731"
UNIMAX_COLOR = "#59A14F"

METHOD_COLORS = {
    "grp_no_l2": GRP_COLOR,
    "proportional": PROPORTIONAL_COLOR,
    "olmix": OLMIX_COLOR,
    "uniform": UNIFORM_COLOR,
    "unimax": UNIMAX_COLOR,
}

METHOD_LINE_DASHES = {
    "grp_no_l2": "solid",
    "proportional": "solid",
    "olmix": "solid",
    "uniform": "solid",
    "unimax": "solid",
}

DEFAULT_FONT_FAMILY = "Times New Roman, Times, serif"
STATIC_WIDTH = 1200
STATIC_HEIGHT = 720
INTERACTIVE_WIDTH = 1450
INTERACTIVE_HEIGHT = 840


def method_color(method_id: str) -> str:
    """Return the canonical color for a plotted method."""
    return METHOD_COLORS[method_id]


def method_dash(method_id: str) -> str:
    """Return the canonical line dash for a plotted method."""
    return METHOD_LINE_DASHES.get(method_id, "solid")


def apply_common_layout(fig: go.Figure) -> None:
    """Apply base paper styling shared by static and interactive outputs."""
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor=PAPER_BACKGROUND,
        plot_bgcolor=PAPER_BACKGROUND,
        font={"family": DEFAULT_FONT_FAMILY, "size": 18, "color": PAPER_TEXT},
        hoverlabel={
            "font": {"family": DEFAULT_FONT_FAMILY, "size": 14},
            "bgcolor": "white",
            "bordercolor": PAPER_AXIS,
        },
        legend={
            "title": "",
            "font": {"size": 17},
            "bgcolor": "rgba(255,255,255,0.88)",
            "bordercolor": "rgba(0,0,0,0)",
            "borderwidth": 0,
        },
    )
    fig.update_xaxes(
        showline=True,
        linewidth=1,
        linecolor=PAPER_AXIS,
        mirror=False,
        ticks="outside",
        tickcolor=PAPER_AXIS,
        gridcolor=PAPER_GRID,
        zeroline=False,
        title_font={"size": 19},
        tickfont={"size": 16},
    )
    fig.update_yaxes(
        showline=True,
        linewidth=1,
        linecolor=PAPER_AXIS,
        mirror=False,
        ticks="outside",
        tickcolor=PAPER_AXIS,
        gridcolor=PAPER_GRID,
        zeroline=False,
        title_font={"size": 19},
        tickfont={"size": 16},
    )


def configure_interactive_layout(fig: go.Figure, *, title: str, y_title: str, x_title: str) -> None:
    """Configure an interactive Plotly artifact with room for controls."""
    apply_common_layout(fig)
    fig.update_layout(
        width=INTERACTIVE_WIDTH,
        height=INTERACTIVE_HEIGHT,
        title={
            "text": title,
            "x": 0.02,
            "xanchor": "left",
            "y": 0.985,
            "yanchor": "top",
            "font": {"size": 27, "family": DEFAULT_FONT_FAMILY, "color": PAPER_TEXT},
        },
        xaxis={"title": x_title},
        yaxis={"title": y_title},
        legend={
            "orientation": "v",
            "x": 1.015,
            "y": 0.5,
            "xanchor": "left",
            "yanchor": "middle",
        },
        margin={"l": 86, "r": 210, "t": 150, "b": 120},
    )


def configure_static_layout(fig: go.Figure, *, y_title: str, x_title: str) -> None:
    """Configure a static paper artifact without interactive controls or title."""
    apply_common_layout(fig)
    fig.layout.updatemenus = ()
    fig.update_layout(
        width=STATIC_WIDTH,
        height=STATIC_HEIGHT,
        title=None,
        xaxis={"title": x_title},
        yaxis={"title": y_title},
        legend={
            "orientation": "h",
            "x": 0.5,
            "y": 1.08,
            "xanchor": "center",
            "yanchor": "bottom",
            "entrywidth": 0.18,
            "entrywidthmode": "fraction",
        },
        margin={"l": 82, "r": 34, "t": 88, "b": 118},
    )


def write_static_images(fig: go.Figure, output_stem: Path) -> None:
    """Write review PNG and LaTeX-ready PDF for a static Plotly figure."""
    fig.write_image(output_stem.with_suffix(".png"), scale=2)
    fig.write_image(output_stem.with_suffix(".pdf"))
