# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Render the surrogate design-space quadrant figure for the paper skeleton."""

from __future__ import annotations

from pathlib import Path

import plotly.graph_objects as go

from paper_plot_style import (
    PAPER_AXIS,
    PAPER_BACKGROUND,
    PAPER_GRID,
    PAPER_TEXT,
    write_static_images,
)

OUTPUT_STEM = Path(__file__).resolve().parent / "img" / "f2_surrogate_design_space"


def add_cell(
    fig: go.Figure,
    *,
    x0: float,
    x1: float,
    y0: float,
    y1: float,
    text: str,
    fillcolor: str,
    line_width: float = 1.6,
    bold: bool = False,
) -> None:
    """Add one quadrant cell."""
    fig.add_shape(
        type="rect",
        x0=x0,
        x1=x1,
        y0=y0,
        y1=y1,
        line={"color": PAPER_TEXT, "width": line_width},
        fillcolor=fillcolor,
        layer="below",
    )
    if bold:
        text = f"<b>{text}</b>"
    fig.add_annotation(
        x=(x0 + x1) / 2,
        y=(y0 + y1) / 2,
        text=text,
        showarrow=False,
        align="center",
        font={"size": 17, "color": PAPER_TEXT},
    )


def main() -> None:
    """Write PNG, PDF, and HTML artifacts."""
    OUTPUT_STEM.parent.mkdir(parents=True, exist_ok=True)

    fig = go.Figure()
    muted = "#F7F4ED"
    highlight = "#E4F5E1"

    add_cell(
        fig,
        x0=0,
        x1=1,
        y0=1,
        y1=2,
        text="DML, BiMix,<br>Olmix, etc.",
        fillcolor=muted,
    )
    add_cell(
        fig,
        x0=1,
        x1=2,
        y0=1,
        y1=2,
        text="Epoch features<br>without repetition<br>penalty",
        fillcolor=muted,
    )
    add_cell(
        fig,
        x0=0,
        x1=1,
        y0=0,
        y1=1,
        text="Nonparametric<br>RegMix, CLIMB,<br>ADMIRE",
        fillcolor=muted,
    )
    add_cell(
        fig,
        x0=1,
        x1=2,
        y0=0,
        y1=1,
        text="GRP<br>effective epochs +<br>retention + penalty",
        fillcolor=highlight,
        line_width=2.4,
        bold=True,
    )

    for x, label in [(0.5, "Proportions"), (1.5, "Effective epochs")]:
        fig.add_annotation(
            x=x,
            y=2.22,
            text=f"<b>{label}</b>",
            showarrow=False,
            font={"size": 19, "color": PAPER_TEXT},
        )
    for y, label in [(1.5, "Monotone"), (0.5, "Non-monotone")]:
        fig.add_annotation(
            x=-0.13,
            y=y,
            text=f"<b>{label}</b>",
            showarrow=False,
            textangle=-90,
            font={"size": 17, "color": PAPER_TEXT},
        )

    fig.add_shape(
        type="line",
        x0=-0.02,
        x1=2.02,
        y0=-0.08,
        y1=-0.08,
        line={"color": PAPER_GRID, "width": 1},
    )

    fig.update_layout(
        width=900,
        height=420,
        paper_bgcolor=PAPER_BACKGROUND,
        plot_bgcolor=PAPER_BACKGROUND,
        margin={"l": 70, "r": 20, "t": 70, "b": 30},
        xaxis={"visible": False, "range": [-0.28, 2.05]},
        yaxis={"visible": False, "range": [-0.15, 2.35]},
        font={"family": "Times New Roman, Times, serif", "color": PAPER_TEXT},
    )
    fig.update_shapes(line={"color": PAPER_AXIS})

    fig.write_html(OUTPUT_STEM.with_suffix(".html"), include_plotlyjs="cdn")
    write_static_images(fig, OUTPUT_STEM)


if __name__ == "__main__":
    main()
