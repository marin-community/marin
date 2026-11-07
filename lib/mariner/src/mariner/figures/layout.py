# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The plain layout that every figure redraw shares.

A redraw sets its panels side by side at one panel size, names each quantity and its unit on the axes, and names
every series in a legend, which sits below the panels so that it covers no data. A method has the same color in every
figure. The PDFs carry no creation date and ignore local matplotlib settings, so redrawing them from one snapshot
gives the same bytes.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from contextlib import AbstractContextManager
from enum import StrEnum
from pathlib import Path
from typing import Any

import matplotlib as mpl
import numpy as np
import pandas as pd
from matplotlib.artist import Artist
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.style import context as style_context

PANEL_WIDTH = 3.4  # inches
PANEL_HEIGHT = 2.7
UNDER_PANEL = -0.2  # a panel's legend hangs from this height, in axes coordinates, below its x label
DATA_MARGIN = 0.05  # the space a panel leaves around its data, as a fraction of the data's range
STYLE = {
    "font.family": "DejaVu Sans",
    "font.size": 8.0,
    "axes.titlesize": 8.5,
    "axes.labelsize": 8.0,
    "legend.fontsize": 7.0,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": "#E3E7EC",
    "grid.linewidth": 0.5,
    "axes.axisbelow": True,
    "lines.linewidth": 1.3,
    "lines.markersize": 4.0,
    "legend.frameon": False,
    "pdf.fonttype": 42,  # TrueType: the PDFs keep their text as text
}
PDF_METADATA = {"CreationDate": None}  # matplotlib stamps the time otherwise
NEUTRAL = "#4D4D4D"  # marks that belong to no method: measured points, reference lines, path fits
BAND_ALPHA = 0.16  # opacity of a shaded band: a range or an interval around a curve


class Method(StrEnum):
    """The methods the figures compare, by their names in the dataset's method, model and predictor columns."""

    PROPORTIONAL = "proportional"
    UNIMAX = "unimax8"
    OLMIX = "olmix"
    MARINER = "mariner"
    REGMIX = "regmix"  # tuned RegMix
    REGMIX_RELEASED = "regmix_official"
    SPLINE = "spline"


METHOD_LABELS = {
    Method.PROPORTIONAL: "Proportional",
    Method.UNIMAX: "UniMax-8",
    Method.OLMIX: "Olmix",
    Method.MARINER: "MARINER",
    Method.REGMIX: "RegMix (tuned)",
    Method.REGMIX_RELEASED: "RegMix (released)",
    Method.SPLINE: "Cubic spline",
}
METHOD_COLORS = {
    Method.PROPORTIONAL: "#6C6F7D",
    Method.UNIMAX: "#4C78A8",
    Method.OLMIX: "#CC79A7",
    Method.MARINER: "#469C76",
    Method.REGMIX: "#B35806",
    Method.REGMIX_RELEASED: "#E69F00",
    Method.SPLINE: "#0072B2",
}
OBJECTIVE_TITLES = {"uncheatable": "Uncheatable", "obe_easy": "OlmoBaseEval Easy"}  # a panel's title per objective
SCALE_COLORMAP = "plasma"  # a ladder of model or data scales, smallest first
SCALE_COLOR_RANGE = (0.0, 0.85)  # plasma's lightest yellow would vanish on white


@dataclasses.dataclass(frozen=True)
class FigureData:
    """A redrawn figure and every series it plots, by name."""

    figure: Figure
    values: dict[str, pd.DataFrame]


def style() -> AbstractContextManager[Any]:
    """The redraws' fonts, sizes and PDF settings, for drawing and saving.

    It starts from matplotlib's defaults, so that a local matplotlibrc (one that typesets text with LaTeX, say)
    changes no PDF.
    """
    return style_context(["default", STYLE])


def panels(
    columns: int, legend_width: float = 0.0, legend_height: float = 0.0, **subplot_options: Any
) -> tuple[Figure, np.ndarray]:
    """A figure of one row of ``columns`` panels of the shared size, and its axes from left to right.

    ``legend_width`` and ``legend_height`` are the inches added for legends outside the panels.
    """
    figsize = (columns * PANEL_WIDTH + legend_width, PANEL_HEIGHT + legend_height)
    figure = Figure(figsize=figsize, layout="constrained")
    axes = figure.subplots(1, columns, squeeze=False, **subplot_options)
    return figure, axes.ravel()


def figure_legend(figure: Figure, handles: Sequence[Artist], columns: int) -> None:
    """A legend of ``handles`` under the panels, in ``columns`` columns."""
    figure.legend(handles=handles, loc="outside lower center", ncols=columns)


def mosaic(
    layout: Sequence[Sequence[str]], columns: float, rows: int, legend_height: float = 0.0
) -> tuple[Figure, dict]:
    """A figure of named panels arranged as ``layout`` (matplotlib's subplot mosaic), ``columns`` panels wide and
    ``rows`` panels tall at the shared panel size, plus ``legend_height`` inches for legends outside the panels."""
    figure = Figure(figsize=(columns * PANEL_WIDTH, rows * PANEL_HEIGHT + legend_height), layout="constrained")
    return figure, figure.subplot_mosaic([list(row) for row in layout])


def panel_legend(axis: Axes, handles: Sequence[Artist] | None = None, **options: Any) -> None:
    """A legend of one panel's series, under its x label."""
    axis.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, UNDER_PANEL), **options)


def data_limits(values: Sequence[float] | np.ndarray) -> tuple[float, float]:
    """Axis limits around ``values`` with the shared margin."""
    low, high = float(np.min(values)), float(np.max(values))
    margin = DATA_MARGIN * (high - low)
    return low - margin, high + margin


def rounded_label(value: float) -> str:
    """Two decimals below one and one from one up, as the paper prints epochs and tokens per parameter."""
    return f"{value:.2f}" if value < 1 else f"{value:.1f}"


def flops_label(flops: float) -> str:
    """Training FLOPs to three significant digits, as 2.49e16."""
    return f"{flops:.3g}".replace("e+", "e")


def scale_colors(count: int) -> list[tuple[float, ...]]:
    """One RGBA color per rung of a scale ladder, from the smallest scale to the largest."""
    colormap = mpl.colormaps[SCALE_COLORMAP]
    return [tuple(colormap(float(position))) for position in np.linspace(*SCALE_COLOR_RANGE, count)]


def save(figure: Figure, path: Path) -> None:
    figure.savefig(path, metadata=PDF_METADATA)
