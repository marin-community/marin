# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.12"
# dependencies = ["matplotlib>=3.9", "numpy>=2.0", "pandas>=2.2"]
# ///
"""Grid views of how each proposed mixture reweights the 39 buckets relative to proportional mixing.

Policies: MARINER, the Olmix policy deployed on the scaling ladder (cap 4; KL 0.005 for OlmoBaseEval Easy, 0.05 for
Uncheatable) and Olmix with no cap and no KL, for each objective. Every cell encodes log2(weight / proportional
weight), clipped at eight times either way, either as a colored cell (``--marks heat``) or as a bar from the
proportional baseline colored by policy (``--marks bars``); the cell text is the materialized epoch count. Buckets a
policy empties are marked with a cross. ``--layout tall`` puts buckets down the rows and the six policies across;
``--layout wide`` puts the policies down the rows and the buckets across. Reads the same candidate tables as
``plot_olmix_vs_mariner_weights_compact_20260921``. Run from the repository root:

    PYTHONPATH=. uv run --offline --no-sync python <this file> [--layout tall|wide] [--marks heat|bars]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib as mpl
import numpy as np
import pandas as pd

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import Rectangle

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.domain_phase_mix import launch_delphi_augmented_swarm_3e18 as base_launch  # noqa: E402
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    plot_olmix_vs_mariner_weights_compact_20260921 as bars,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    plot_wspu_worsened_vs_uncheatable_mixtures_20260905 as layout,
)

OUTPUT_DIR = SCRIPT_DIR / "reference_outputs" / "mixture_heatmap_20260921"
OBJECTIVES = (("uncheatable", "Uncheatable"), ("table9", "OlmoBaseEval Easy"))  # the scaling figure's order
POLICIES = (  # row key, label per objective, table, candidate per objective
    (
        "mariner",
        {"table9": "MARINER", "uncheatable": "MARINER"},
        bars.MARINER_TABLE,
        {"table9": "lwspu_t9_snc_cap08", "uncheatable": "lwspu_u_snc_cap06"},
    ),
    (  # the matched primary policies trained on the scaling ladder (Table 2, Figure 6)
        "olmix_deployed",
        {"table9": "Olmix, cap 4, KL 0.005", "uncheatable": "Olmix, cap 4, KL 0.05"},
        bars.OLMIX_TABLE,
        {"table9": "olmixq_t9_kl0p005_cap04", "uncheatable": "olmixq_u_kl0p05_cap04"},
    ),
    (
        "olmix_nocap",
        {"table9": "Olmix, no cap, no KL", "uncheatable": "Olmix, no cap, no KL"},
        bars.OLMIX_UNCAPPED_TABLE,
        bars.OLMIX_POLICIES["nocap"][1],
    ),
)
COLUMN_LABELS = {  # tall layout, three lines per column
    "mariner": {"table9": "MARINER\n\n", "uncheatable": "MARINER\n\n"},
    "olmix_deployed": {"table9": "Olmix\ncap 4\nKL 0.005", "uncheatable": "Olmix\ncap 4\nKL 0.05"},
    "olmix_nocap": {"table9": "Olmix\nno cap\nno KL", "uncheatable": "Olmix\nno cap\nno KL"},
}
LOG2_LIMIT = 3.0  # cells saturate at eight times up or down
# A muted blue-to-rust scale with a warm light center, so removed buckets (pure white with a cross) stay distinct.
DIVERGING = LinearSegmentedColormap.from_list("reweight", ["#2F5E8A", "#8FB3D1", "#F3F1EC", "#E3A48C", "#B8452C"])
NORM = TwoSlopeNorm(vmin=-LOG2_LIMIT, vcenter=0.0, vmax=LOG2_LIMIT)
POLICY_COLORS = {"mariner": "#469C76", "olmix_deployed": "#CC79A7", "olmix_nocap": "#8E4B7A"}  # series colors
REMOVED_EDGE = "#CFCCC6"
REMOVED_MARK = "#A7A49E"
GRID_EDGE = "#E4E2DE"
BASELINE = "#B9B6B0"
INK = "#2B2B2B"
BAR_THICKNESS = 0.56  # share of the cell's cross-axis
BAR_PAD = 0.04  # cell units kept free at the extremes
LINEAR_SCALE_POLICIES = ("mariner", "olmix_deployed")  # a family's bar scale comes from the trained policies
CLIP_MARK = "\u25b8"  # marks a bar cut at the family scale (only the uncapped Olmix optimum exceeds it)
SCALE_TICKS = ([-3, -2, -1, 0, 1, 2, 3], ["⅛", "¼", "½", "1", "2", "4", "8"])
CROSS = "×"  # noqa: RUF001  (the multiplication sign is the intended glyph)
# Two-block layout (the paper's version): epochs are both the color and the label; Uncheatable first.
NOTO_STYLES = ("Regular", "Bold", "Italic")
EPOCH_NORM = TwoSlopeNorm(vmin=-3.0, vcenter=0.0, vmax=4.0)  # 1/8 to 16 epochs on a log2 scale, centered at one
EPOCH_TICKS = ([-3, -2, -1, 0, 1, 2, 3, 4], ["\u2264\u215b", "\u00bc", "\u00bd", "1", "2", "4", "8", "\u226516"])
POLICY_HEADERS = {  # per objective: the deployed Olmix policies differ in their KL coefficient
    "mariner": {"uncheatable": "MARINER\nno cap\nno KL", "table9": "MARINER\nno cap\nno KL"},
    "olmix_deployed": {"uncheatable": "Olmix\ncap 4\nKL \u03bb = 0.05", "table9": "Olmix\ncap 4\nKL \u03bb = 0.005"},
    "olmix_nocap": {"uncheatable": "Olmix\nno cap\nno KL", "table9": "Olmix\nno cap\nno KL"},
}
HEADER_COLORS = {"mariner": "#3E8A69", "olmix_deployed": "#A8578A", "olmix_nocap": "#A8578A"}
FAMILY_TITLES = {
    "Common Crawl, high quality": "Common Crawl \u00b7 high quality",
    "Common Crawl, low quality": "Common Crawl \u00b7 low quality",
    "Dolma 3 other sources": "Dolma 3 \u00b7 other sources",
    "Dolmino": "Dolmino",
}
RULE = "#CFCCC6"
ZERO_MARK = "#B8B5AF"
# Glyph drawn in zero-allocation cells of the stacked and blocks layouts; "0" needs no legend entry.
ZERO_GLYPH = CROSS
BLOCK_CELL_WIDTH = 0.33
BLOCK_ROW_HEIGHT = 0.125
BLOCK_LABEL_GUTTER = 1.0
BLOCK_GAP = 0.32  # inches between the two blocks
BLOCK_OBJECTIVE_GAP = 0.35  # cell widths between the two objectives
BLOCK_FAMILY_ROWS = 1.0
# Stacked layout: one header row; Common Crawl topics with a high-quality and a low-quality cell per policy, then the
# other sources with one full-width cell per policy, so every column lines up top to bottom.
STACK_UNIT = 0.29  # inches per half cell; a policy column is two units wide
STACK_ROW_HEIGHT = 0.104
STACK_LABEL_GUTTER = 1.06
STACK_OBJECTIVE_GAP = 0.7  # units between the two objectives
STACK_FAMILY_ROWS = 1.0
STACK_GROUP_GAP = 0.35  # rows between Common Crawl and the other sources
QUALITY_LABELS = ("high", "low")
# tall layout geometry (inches and row units)
TALL_CELL_WIDTH = 0.5
TALL_ROW_HEIGHT = 0.128
TALL_LABEL_GUTTER = 1.12
TALL_FAMILY_ROWS = 1.25
TALL_BLOCK_GAP = 0.45
# wide layout geometry
WIDE_GROUP_GAP = 0.8  # columns between bucket families
WIDE_BLOCK_GAP = 1.15  # rows between the two objectives
WIDE_HEADERS = {"Dolma 3 other sources": "Dolma 3,\nother sources"}


def bucket_columns(frame: pd.DataFrame) -> list[tuple[str, list[str]]]:
    """Bucket families in the bar figures' order: CC high, CC low, other Dolma 3, Dolmino."""
    return [(title, group) for column in bars.column_layout(frame) for title, group in column]


def load_policy(table: Path, candidate: str, domains: list[str]) -> pd.DataFrame:
    rows = bars.candidate_weights(table, candidate).reindex(domains)
    if rows["weight"].isna().any():
        raise ValueError(f"{candidate}: buckets missing from {table}")
    return rows


def load_all() -> tuple[pd.DataFrame, dict[tuple[str, str], pd.DataFrame]]:
    tokens = base_launch.TOP_LEVEL_DOMAIN_TOKEN_COUNTS
    domains = list(tokens)
    pool = np.asarray([tokens[d] for d in domains], float)
    frame = pd.DataFrame({"proportional_weight": pool / pool.sum()}, index=domains)
    frame["label"] = [bars.bucket_label(d) for d in domains]
    policies = {
        (objective, key): load_policy(table, candidates[objective], domains)
        for objective, _label in OBJECTIVES
        for key, _labels, table, candidates in POLICIES
    }
    return frame, policies


def cell_text(weight: float, epochs: float, proportional: float, mode: str) -> str:
    if weight <= 0.0:
        return ""
    value = epochs if mode == "epochs" else weight / proportional
    if value >= 10:
        return f"{value:.0f}"
    return f"{value:.2f}" if value < 0.1 else f"{value:.1f}"


def family_scales(frame: pd.DataFrame, policies: dict, families: list[tuple[str, list[str]]]) -> dict[str, float]:
    """Per family, the largest weight the proportional mixture or a trained policy gives one of its buckets."""
    scales = {}
    for title, group in families:
        largest = max(float(frame.loc[d, "proportional_weight"]) for d in group)
        for (_objective, key), policy in policies.items():
            if key in LINEAR_SCALE_POLICIES:
                largest = max(largest, max(float(policy.loc[d, "weight"]) for d in group))
        scales[title] = largest
    return scales


def draw_cell(axis, x0, y0, key, weight, text, ratio, marks, along, proportional=None, scale=None):
    """One cell at (x0, y0): a colored square, a bar from the midline (``bars``) or a raw-weight bar (``weights``).

    Bars run in cell units along ``along`` ('x' or 'y'); the y axis is inverted in both layouts, so a bar along 'y'
    rises. ``weights`` bars start at the cell's edge on the family's linear scale, with a tick at the proportional
    weight; a bar beyond the scale is cut and marked.
    """
    if marks == "weights":
        span = 1 - 2 * BAR_PAD
        clipped = weight > scale
        length = min(weight, scale) / scale * span
        tick = BAR_PAD + proportional / scale * span
        color = POLICY_COLORS[key]
        if clipped:
            text = f"{text} ({100 * weight:.0f}%)"
        if along == "x":
            axis.plot([x0 + tick] * 2, [y0 + 0.12, y0 + 0.88], color=INK, lw=0.7, zorder=4)
            if weight <= 0.0:
                axis.text(
                    x0 + 0.5, y0 + 0.5, CROSS, ha="center", va="center", fontsize=5.4, color=REMOVED_MARK, zorder=4
                )
                return
            axis.add_patch(
                Rectangle(
                    (x0 + BAR_PAD, y0 + (1 - BAR_THICKNESS) / 2), length, BAR_THICKNESS, facecolor=color, lw=0, zorder=3
                )
            )
            if clipped:
                axis.text(
                    x0 + BAR_PAD + length,
                    y0 + 0.5,
                    CLIP_MARK,
                    ha="right",
                    va="center",
                    fontsize=6,
                    color="white",
                    zorder=5,
                )
            inside = length > 0.62 * span
            axis.text(
                x0 + BAR_PAD + length + (-(0.14 if clipped else 0.05) if inside else 0.05),
                y0 + 0.5,
                text,
                ha="right" if inside else "left",
                va="center",
                fontsize=4.9 if clipped else 5.2,
                color="white" if inside else INK,
                zorder=5,
                bbox=None
                if inside
                else {"boxstyle": "round,pad=0.08", "facecolor": "white", "edgecolor": "none", "alpha": 0.85},
            )
            return
        axis.plot([x0 + 0.12, x0 + 0.88], [y0 + 1 - tick] * 2, color=INK, lw=0.7, zorder=4)
        if weight <= 0.0:
            axis.text(x0 + 0.5, y0 + 0.5, CROSS, ha="center", va="center", fontsize=5.4, color=REMOVED_MARK, zorder=4)
            return
        axis.add_patch(
            Rectangle(
                (x0 + (1 - BAR_THICKNESS) / 2, y0 + 1 - BAR_PAD - length),
                BAR_THICKNESS,
                length,
                facecolor=color,
                lw=0,
                zorder=3,
            )
        )
        if clipped:
            axis.text(
                x0 + 0.5,
                y0 + 1 - BAR_PAD - length,
                CLIP_MARK,
                ha="center",
                va="top",
                fontsize=6,
                color="white",
                rotation=90,
                zorder=5,
            )
        return
    if weight <= 0.0:
        if marks == "heat":
            axis.add_patch(Rectangle((x0, y0), 1, 1, facecolor="white", edgecolor=REMOVED_EDGE, lw=0.4))
        axis.text(x0 + 0.5, y0 + 0.5, CROSS, ha="center", va="center", fontsize=5.4, color=REMOVED_MARK, zorder=4)
        return
    value = float(np.clip(np.log2(ratio), -LOG2_LIMIT, LOG2_LIMIT))
    if marks == "heat":
        axis.add_patch(Rectangle((x0, y0), 1, 1, facecolor=DIVERGING(NORM(value)), edgecolor="white", lw=0.5))
        size = 5.6 if along == "x" else (5.0 if len(text) <= 3 else 4.3)
        axis.text(
            x0 + 0.5, y0 + 0.5, text, ha="center", va="center", fontsize=size, color="white" if abs(value) > 2.1 else INK
        )
        return
    half = 0.5 - BAR_PAD
    length = value / LOG2_LIMIT * half
    color = POLICY_COLORS[key]
    if along == "x":
        left = x0 + 0.5 + min(0.0, length)
        axis.add_patch(
            Rectangle((left, y0 + (1 - BAR_THICKNESS) / 2), abs(length), BAR_THICKNESS, facecolor=color, lw=0, zorder=3)
        )
        inside = abs(length) > 0.6 * half  # long bars carry their label inside, near the tip
        tip = x0 + 0.5 + length
        sign = 1 if length >= 0 else -1
        axis.text(
            tip - sign * 0.05 if inside else tip + sign * 0.05,
            y0 + 0.5,
            text,
            ha=("right" if sign > 0 else "left") if inside else ("left" if sign > 0 else "right"),
            va="center",
            fontsize=5.2,
            color="white" if inside else INK,
            zorder=4,
        )
    else:
        bottom = y0 + 0.5 - max(0.0, length)
        axis.add_patch(
            Rectangle(
                (x0 + (1 - BAR_THICKNESS) / 2, bottom), BAR_THICKNESS, abs(length), facecolor=color, lw=0, zorder=3
            )
        )


def draw_scale(axis, marks: str) -> None:
    """The key under a grid: a colorbar for colored cells, a bar-length scale for bar cells, a sample for weights."""
    if marks == "weights":  # a sample bar with its proportional tick and the clip mark
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1)
        axis.add_patch(Rectangle((0.0, 0.22), 0.42, 0.56, facecolor="#9A9894", lw=0))
        axis.plot([0.24] * 2, [0.0, 1.0], color=INK, lw=0.7)
        axis.text(0.42, 0.5, CLIP_MARK, ha="right", va="center", fontsize=6, color="white")
        axis.text(
            0.5,
            0.5,
            CLIP_MARK + " bar cut at the family scale",
            ha="left",
            va="center",
            fontsize=5.6,
            color=INK,
        )
        axis.set_axis_off()
        return
    if marks == "heat":
        colorbar = plt.colorbar(plt.cm.ScalarMappable(norm=NORM, cmap=DIVERGING), cax=axis, orientation="horizontal")
        colorbar.set_ticks(SCALE_TICKS[0])
        colorbar.set_ticklabels(SCALE_TICKS[1])
        colorbar.ax.tick_params(labelsize=5.8, length=1.5, pad=1.5)
        colorbar.outline.set_linewidth(0.4)
        return
    axis.set_xlim(-LOG2_LIMIT, LOG2_LIMIT)
    axis.set_ylim(0, 1)
    axis.axvline(0.0, color=BASELINE, lw=0.5)
    # one bar each way: eight times down (left) and eight times up (right)
    axis.add_patch(Rectangle((-LOG2_LIMIT, 0.22), LOG2_LIMIT, 0.56, facecolor="#B5B2AC", lw=0))
    axis.add_patch(Rectangle((0.0, 0.22), LOG2_LIMIT, 0.56, facecolor="#7C7975", lw=0))
    axis.set_xticks(SCALE_TICKS[0])
    axis.set_xticklabels(SCALE_TICKS[1])
    axis.tick_params(axis="x", labelsize=5.8, length=1.5, pad=1.5)
    axis.set_yticks([])
    for name in ("top", "right", "left"):
        axis.spines[name].set_visible(False)
    axis.spines["bottom"].set_linewidth(0.4)


def removed_key(figure, x: float, y: float, height: float, marks: str) -> None:
    """A small legend entry for removed buckets at figure coordinates (x, y), ``height`` tall."""
    width = figure.get_figwidth()
    if marks == "heat":
        figure.patches.append(
            Rectangle(
                (x, y),
                0.09 / width,
                height,
                transform=figure.transFigure,
                facecolor="white",
                edgecolor=REMOVED_EDGE,
                lw=0.4,
            )
        )
    figure.text(x + 0.045 / width, y + height / 2, CROSS, ha="center", va="center", fontsize=5.6, color=REMOVED_MARK)
    figure.text(x + 0.13 / width, y + height / 2, "removed", ha="left", va="center", fontsize=6.0, color=layout.INK)


def build_tall_figure(text_mode: str, marks: str) -> plt.Figure:
    """Buckets down the rows in family groups, the six policies across; every label horizontal."""
    frame, policies = load_all()
    domains = list(frame.index)
    families = bucket_columns(frame)
    y_of: dict[str, float] = {}
    family_rows: list[tuple[str, float]] = []
    y = 0.0
    for title, group in families:
        family_rows.append((title, y))
        y += TALL_FAMILY_ROWS
        for domain in group:
            y_of[domain] = y
            y += 1.0
    height_rows = y
    x_of: dict[tuple[str, str], float] = {}
    x = 0.0
    block_span: list[tuple[str, float, float]] = []
    for objective, objective_label in OBJECTIVES:
        start = x
        for key, _labels, _table, _candidates in POLICIES:
            x_of[(objective, key)] = x
            x += 1.0
        block_span.append((objective_label, start, x))
        x += TALL_BLOCK_GAP
    width_cells = x - TALL_BLOCK_GAP

    grid_width = width_cells * TALL_CELL_WIDTH
    grid_height = height_rows * TALL_ROW_HEIGHT
    top_inch, bottom_inch = 0.72, 0.62
    fig_width = TALL_LABEL_GUTTER + grid_width + 0.12
    fig_height = top_inch + grid_height + bottom_inch
    with plt.rc_context(layout.PLOT_STYLE):
        figure = plt.figure(figsize=(fig_width, fig_height))
        axis = figure.add_axes(
            (TALL_LABEL_GUTTER / fig_width, bottom_inch / fig_height, grid_width / fig_width, grid_height / fig_height)
        )
        rows_top, rows_bottom = min(y_of.values()), max(y_of.values()) + 1
        scales = family_scales(frame, policies, families)
        scale_of = {domain: scales[title] for title, group in families for domain in group}
        for (objective, key), policy in policies.items():
            x0 = x_of[(objective, key)]
            if marks != "heat":  # faint cell grid; for midline bars also the proportional baseline down the column
                for domain in domains:
                    axis.add_patch(Rectangle((x0, y_of[domain]), 1, 1, facecolor="none", edgecolor=GRID_EDGE, lw=0.4))
                if marks == "bars":
                    axis.plot([x0 + 0.5] * 2, [rows_top, rows_bottom], color=BASELINE, lw=0.5, zorder=2)
            for domain in domains:
                weight = float(policy.loc[domain, "weight"])
                proportional = float(frame.loc[domain, "proportional_weight"])
                text = cell_text(weight, float(policy.loc[domain, "materialized_epochs"]), proportional, text_mode)
                draw_cell(
                    axis,
                    x0,
                    y_of[domain],
                    key,
                    weight,
                    text,
                    weight / proportional,
                    marks,
                    "x",
                    proportional,
                    scale_of[domain],
                )
        for domain in domains:
            axis.text(
                -0.12,
                y_of[domain] + 0.5,
                frame.loc[domain, "label"],
                ha="right",
                va="center",
                fontsize=6.0,
                color=layout.INK,
            )
        gutter_left = -TALL_LABEL_GUTTER / TALL_CELL_WIDTH + 0.1
        for title, y0 in family_rows:
            axis.text(
                gutter_left,
                y0 + TALL_FAMILY_ROWS - 0.3,
                title,
                ha="left",
                va="bottom",
                fontsize=6.4,
                fontweight="bold",
                color=layout.INK,
            )
            axis.plot(
                [gutter_left, width_cells], [y0 + TALL_FAMILY_ROWS - 0.12] * 2, color=layout.INK, lw=0.5, clip_on=False
            )
        if marks == "weights":  # each family's linear bar scale, at the right end of its rule
            for title, y0 in family_rows:
                axis.text(
                    width_cells,
                    y0 + TALL_FAMILY_ROWS - 0.3,
                    f"bars to {100 * scales[title]:.0f}%",
                    ha="right",
                    va="bottom",
                    fontsize=5.8,
                    color=INK,
                )
        for objective_label, start, end in block_span:
            axis.text(
                (start + end) / 2, -3.65, objective_label, ha="center", va="bottom", fontsize=7.0, fontweight="bold"
            )
            axis.plot([start + 0.08, end - 0.08], [-3.5] * 2, color=layout.INK, lw=0.5, clip_on=False)
        for (objective, key), x0 in x_of.items():
            axis.text(
                x0 + 0.5, -0.25, COLUMN_LABELS[key][objective], ha="center", va="bottom", fontsize=5.9, linespacing=1.05
            )
        axis.set_xlim(-0.02, width_cells + 0.02)
        axis.set_ylim(height_rows + 0.02, -0.02)
        axis.set_axis_off()
        key_axis = figure.add_axes(
            (TALL_LABEL_GUTTER / fig_width, 0.2 / fig_height, grid_width * 0.56 / fig_width, 0.075 / fig_height)
        )
        draw_scale(key_axis, marks)
        key_axis.set_title(
            "Mixture weight (linear; scale printed per family)"
            if marks == "weights"
            else "Weight relative to proportional",
            fontsize=6.0,
            pad=2.5,
            loc="left",
        )
        key_x = (TALL_LABEL_GUTTER + grid_width * 0.66) / fig_width
        removed_key(figure, key_x, 0.2 / fig_height, 0.075 / fig_height, marks)
        note = ("cell text: epochs" if text_mode == "epochs" else "cell text: weight ratio") + (
            "; tick: proportional" if marks == "weights" else ""
        )
        figure.text(key_x, (0.2 + 0.075 + 0.06) / fig_height, note, ha="left", va="bottom", fontsize=6.0)
        return figure


def build_wide_figure(text_mode: str, marks: str) -> plt.Figure:
    """Policies down the rows (one block per objective), buckets across in family groups."""
    frame, policies = load_all()
    domains = list(frame.index)
    families = bucket_columns(frame)
    x_of: dict[str, float] = {}
    x = 0.0
    spans = []
    for title, group in families:
        start = x
        for domain in group:
            x_of[domain] = x
            x += 1.0
        spans.append((title, start, x))
        x += WIDE_GROUP_GAP
    width = x - WIDE_GROUP_GAP
    y_of: dict[tuple[str, str], float] = {}
    y = 0.0
    block_labels = []
    for objective, objective_label in OBJECTIVES:
        block_start = y
        for key, _labels, _table, _candidates in POLICIES:
            y_of[(objective, key)] = y
            y += 1.0
        block_labels.append((objective_label, block_start))
        y += WIDE_BLOCK_GAP
    height = y - WIDE_BLOCK_GAP
    with plt.rc_context(layout.PLOT_STYLE):
        figure = plt.figure(figsize=(7.4, 3.7))
        axis = figure.add_axes((0.20, 0.43, 0.79, 0.46))
        scales = family_scales(frame, policies, families)
        scale_of = {domain: scales[title] for title, group in families for domain in group}
        for (objective, key), policy in policies.items():
            y0 = y_of[(objective, key)]
            if marks != "heat":  # faint cell grid; for midline bars also the proportional baseline along the row
                for domain in domains:
                    axis.add_patch(Rectangle((x_of[domain], y0), 1, 1, facecolor="none", edgecolor=GRID_EDGE, lw=0.4))
                if marks == "bars":
                    for _title, start, end in spans:
                        axis.plot([start, end], [y0 + 0.5] * 2, color=BASELINE, lw=0.5, zorder=2)
            for domain in domains:
                weight = float(policy.loc[domain, "weight"])
                proportional = float(frame.loc[domain, "proportional_weight"])
                text = cell_text(weight, float(policy.loc[domain, "materialized_epochs"]), proportional, text_mode)
                draw_cell(
                    axis,
                    x_of[domain],
                    y0,
                    key,
                    weight,
                    text,
                    weight / proportional,
                    marks,
                    "y",
                    proportional,
                    scale_of[domain],
                )
        for (objective, key), y0 in y_of.items():
            label = next(labels[objective] for k, labels, _t, _c in POLICIES if k == key)
            axis.text(-0.25, y0 + 0.5, label, ha="right", va="center", fontsize=6.6, color=layout.INK)
        for objective_label, start in block_labels:
            axis.text(-0.25, start - 0.12, objective_label, ha="right", va="bottom", fontsize=7.0, fontweight="bold")
        for domain in domains:
            axis.text(
                x_of[domain] + 0.5,
                height + 0.15,
                frame.loc[domain, "label"],
                ha="right",
                va="center",
                rotation=90,
                rotation_mode="anchor",
                fontsize=5.8,
                color=layout.INK,
            )
        for title, start, end in spans:
            axis.text(
                (start + end) / 2,
                -0.3,
                WIDE_HEADERS.get(title, title),
                ha="center",
                va="bottom",
                fontsize=6.6,
                fontweight="bold",
                linespacing=1.0,
            )
            axis.plot([start + 0.05, end - 0.05], [-0.15] * 2, color=layout.INK, lw=0.6)
            if marks == "weights":
                axis.text(end - 0.05, -0.05, f"bars to {100 * scales[title]:.0f}%", ha="right", va="top", fontsize=5.4)
        axis.set_xlim(-0.05, width + 0.05)
        axis.set_ylim(height + 0.05, -0.05)  # first policy row at the top
        axis.set_axis_off()
        key_axis = figure.add_axes((0.20, 0.09, 0.30, 0.03))
        draw_scale(key_axis, marks)
        key_axis.set_title(
            "Mixture weight (linear; scale printed per family)"
            if marks == "weights"
            else "Weight relative to proportional (log scale)",
            fontsize=6.4,
            pad=3,
            loc="left",
        )
        removed_key(figure, 0.56, 0.085, 0.035, marks)
        if marks == "heat":  # bar cells are too narrow for text in this layout
            note = (
                "cell text: materialized epochs" if text_mode == "epochs" else "cell text: weight / proportional weight"
            )
            figure.text(0.56, 0.15, note, ha="left", va="bottom", fontsize=6.4, color=layout.INK)
        return figure


def use_noto_sans() -> str:
    """Register Noto Sans from the user's font library, as Figure 1's builder does; fall back to DejaVu Sans."""
    found = False
    for style in NOTO_STYLES:
        path = Path.home() / "Library/Fonts" / f"NotoSans-{style}.ttf"
        if path.exists():
            font_manager.fontManager.addfont(str(path))
            found = True
    return "Noto Sans" if found else "DejaVu Sans"


def epoch_label(epochs: float) -> str:
    if epochs <= 0.0:
        return ""
    if epochs < 0.1:
        return "<0.1"
    return f"{epochs:.0f}" if epochs >= 10 else f"{epochs:.1f}"


def build_blocks_figure(text_mode: str) -> plt.Figure:
    """Two aligned blocks (Common Crawl left, the other sources right), six policy columns each, epochs as color and
    label, every label horizontal."""
    del text_mode  # this layout always labels epochs, the quantity it colors
    frame, policies = load_all()
    families = bucket_columns(frame)
    blocks = [families[:2], families[2:]]
    col_x: dict[tuple[str, str], float] = {}
    objective_span: list[tuple[str, float, float]] = []
    x = 0.0
    for objective, objective_label in OBJECTIVES:
        start = x
        for key, _labels, _table, _candidates in POLICIES:
            col_x[(objective, key)] = x
            x += 1.0
        objective_span.append((objective_label, start, x))
        x += BLOCK_OBJECTIVE_GAP
    width_cells = x - BLOCK_OBJECTIVE_GAP
    block_rows = []
    for block in blocks:
        y = 0.0
        y_of: dict[str, float] = {}
        titles: list[tuple[str, float]] = []
        for title, group in block:
            titles.append((title, y))
            y += BLOCK_FAMILY_ROWS
            for domain in group:
                y_of[domain] = y
                y += 1.0
        block_rows.append((y_of, titles, y))
    tallest = max(rows[2] for rows in block_rows)
    grid_width = width_cells * BLOCK_CELL_WIDTH
    top_inch, bottom_inch = 0.64, 0.1
    fig_width = 2 * (BLOCK_LABEL_GUTTER + grid_width) + BLOCK_GAP + 0.08
    fig_height = top_inch + tallest * BLOCK_ROW_HEIGHT + bottom_inch
    style = dict(layout.PLOT_STYLE, **{"font.family": [use_noto_sans(), "DejaVu Sans"]})  # DejaVu supplies odd glyphs
    with plt.rc_context(style):
        figure = plt.figure(figsize=(fig_width, fig_height))
        for index, (block, (y_of, titles, height_rows)) in enumerate(zip(blocks, block_rows, strict=True)):
            left = BLOCK_LABEL_GUTTER + index * (BLOCK_LABEL_GUTTER + grid_width + BLOCK_GAP)
            axis_height = height_rows * BLOCK_ROW_HEIGHT
            axis = figure.add_axes(
                (
                    left / fig_width,
                    (fig_height - top_inch - axis_height) / fig_height,
                    grid_width / fig_width,
                    axis_height / fig_height,
                )
            )
            domains = [domain for _title, group in block for domain in group]
            for (objective, key), policy in policies.items():
                x0 = col_x[(objective, key)]
                for domain in domains:
                    y0 = y_of[domain]
                    epochs = float(policy.loc[domain, "materialized_epochs"])
                    if float(policy.loc[domain, "weight"]) <= 0.0:
                        axis.text(
                            x0 + 0.5, y0 + 0.5, ZERO_GLYPH, ha="center", va="center", fontsize=5.6, color=ZERO_MARK
                        )
                        continue
                    value = float(np.clip(np.log2(epochs), EPOCH_NORM.vmin, EPOCH_NORM.vmax))
                    axis.add_patch(
                        Rectangle((x0, y0), 1, 1, facecolor=DIVERGING(EPOCH_NORM(value)), edgecolor="white", lw=0.35)
                    )
                    dark = value < -2.0 or value > 2.6
                    axis.text(
                        x0 + 0.5,
                        y0 + 0.5,
                        epoch_label(epochs),
                        ha="center",
                        va="center",
                        fontsize=5.4,
                        color="white" if dark else INK,
                    )
            for domain in domains:
                axis.text(-0.12, y_of[domain] + 0.5, frame.loc[domain, "label"], ha="right", va="center", fontsize=5.9)
            gutter_left = -BLOCK_LABEL_GUTTER / BLOCK_CELL_WIDTH + 0.05
            for title, y0 in titles:
                axis.text(
                    gutter_left, y0 + 0.82, FAMILY_TITLES[title], ha="left", va="bottom", fontsize=6.0, color="#555250"
                )
                axis.plot([gutter_left, width_cells], [y0 + 0.95] * 2, color=RULE, lw=0.5, clip_on=False)
            # column headers above the block
            for objective_label, start, end in objective_span:
                axis.text(
                    (start + end) / 2,
                    -3.35,
                    objective_label,
                    ha="center",
                    va="bottom",
                    fontsize=6.3,
                    fontweight="bold",
                    color="#3A3835",
                )
                axis.plot([start + 0.06, end - 0.06], [-3.2] * 2, color=RULE, lw=0.6, clip_on=False)
            for (objective, key), x0 in col_x.items():
                axis.text(
                    x0 + 0.06,
                    -0.2,
                    POLICY_HEADERS[key][objective],
                    ha="left",
                    ma="left",
                    va="bottom",
                    fontsize=5.6,
                    color=HEADER_COLORS[key],
                    linespacing=1.0,
                )
            axis.set_xlim(-0.02, width_cells + 0.02)
            axis.set_ylim(height_rows + 0.02, -0.02)
            axis.set_axis_off()
        # shared key in the free space under the shorter right block
        bar_width = 1.9
        right_left = 2 * BLOCK_LABEL_GUTTER + grid_width + BLOCK_GAP
        right_bottom = fig_height - top_inch - block_rows[1][2] * BLOCK_ROW_HEIGHT
        key_y = right_bottom - 0.62
        cax = figure.add_axes(
            (
                (right_left + (grid_width - bar_width) / 2) / fig_width,
                key_y / fig_height,
                bar_width / fig_width,
                0.07 / fig_height,
            )
        )
        colorbar = plt.colorbar(
            plt.cm.ScalarMappable(norm=EPOCH_NORM, cmap=DIVERGING), cax=cax, orientation="horizontal"
        )
        colorbar.set_ticks(EPOCH_TICKS[0])
        colorbar.set_ticklabels(EPOCH_TICKS[1])
        colorbar.ax.tick_params(labelsize=5.4, length=1.5, pad=1.2)
        colorbar.outline.set_linewidth(0.35)
        cax.set_title("Materialized epochs (color and label)", fontsize=6.0, pad=2.5, loc="left", color="#555250")
        figure.text(
            (right_left + (grid_width - bar_width) / 2) / fig_width,
            (key_y - 0.2) / fig_height,
            (CROSS + "  zero allocation") if ZERO_GLYPH == CROSS else "",
            ha="left",
            va="center",
            fontsize=6.0,
            color="#555250",
        )
        return figure


def draw_epoch_cell(axis, x0: float, y0: float, width: float, weight: float, epochs: float) -> None:
    """One epoch-colored cell of the given width at (x0, y0), or a cross for zero allocation."""
    if weight <= 0.0:
        axis.text(x0 + width / 2, y0 + 0.5, ZERO_GLYPH, ha="center", va="center", fontsize=5.6, color=ZERO_MARK)
        return
    value = float(np.clip(np.log2(epochs), EPOCH_NORM.vmin, EPOCH_NORM.vmax))
    axis.add_patch(Rectangle((x0, y0), width, 1, facecolor=DIVERGING(EPOCH_NORM(value)), edgecolor="white", lw=0.35))
    dark = value < -2.0 or value > 2.6
    axis.text(
        x0 + width / 2,
        y0 + 0.5,
        epoch_label(epochs),
        ha="center",
        va="center",
        fontsize=5.3,
        color="white" if dark else INK,
    )


def build_stacked_figure(text_mode: str) -> plt.Figure:
    """One header row over two stacked blocks: Common Crawl topics with a high- and a low-quality cell per policy,
    then the other sources with one wide cell per policy."""
    del text_mode
    frame, policies = load_all()
    families = bucket_columns(frame)
    (_high_title, high), (_low_title, low), *others = families
    topics = [(bars.bucket_label(h), h, lo) for h, lo in zip(high, low, strict=True)]
    if any(h.rsplit("_", 1)[0] != lo.rsplit("_", 1)[0] for _label, h, lo in topics):
        raise ValueError("Common Crawl high and low cells are not paired by topic")
    col_x: dict[tuple[str, str], float] = {}
    objective_span: list[tuple[str, float, float]] = []
    x = 0.0
    for objective, objective_label in OBJECTIVES:
        start = x
        for key, _labels, _table, _candidates in POLICIES:
            col_x[(objective, key)] = x
            x += 2.0
        objective_span.append((objective_label, start, x))
        x += STACK_OBJECTIVE_GAP
    width_units = x - STACK_OBJECTIVE_GAP
    # rows: the Common Crawl title and topics, a gap, then each other family with its title
    y = STACK_FAMILY_ROWS
    topic_y = {label: y + i for i, (label, _h, _lo) in enumerate(topics)}
    y += len(topics) + STACK_GROUP_GAP
    family_titles = [("Common Crawl \u00b7 high and low quality", 0.0)]
    row_y: dict[str, float] = {}
    for title, group in others:
        family_titles.append((FAMILY_TITLES[title], y))
        y += STACK_FAMILY_ROWS
        for domain in group:
            row_y[domain] = y
            y += 1.0
    height_rows = y
    grid_width = width_units * STACK_UNIT
    grid_height = height_rows * STACK_ROW_HEIGHT
    top_inch, bottom_inch = 0.62, 0.42
    fig_width = STACK_LABEL_GUTTER + grid_width + 0.08
    fig_height = top_inch + grid_height + bottom_inch
    style = dict(layout.PLOT_STYLE, **{"font.family": [use_noto_sans(), "DejaVu Sans"]})
    with plt.rc_context(style):
        figure = plt.figure(figsize=(fig_width, fig_height))
        axis = figure.add_axes(
            (STACK_LABEL_GUTTER / fig_width, bottom_inch / fig_height, grid_width / fig_width, grid_height / fig_height)
        )
        for (objective, key), policy in policies.items():
            x0 = col_x[(objective, key)]
            for label, h, lo in topics:
                y0 = topic_y[label]
                for offset, domain in ((0.0, h), (1.0, lo)):
                    draw_epoch_cell(
                        axis,
                        x0 + offset,
                        y0,
                        1.0,
                        float(policy.loc[domain, "weight"]),
                        float(policy.loc[domain, "materialized_epochs"]),
                    )
            for domain, y0 in row_y.items():
                draw_epoch_cell(
                    axis,
                    x0,
                    y0,
                    2.0,
                    float(policy.loc[domain, "weight"]),
                    float(policy.loc[domain, "materialized_epochs"]),
                )
        for label, y0 in topic_y.items():
            axis.text(-0.15, y0 + 0.5, label, ha="right", va="center", fontsize=5.9)
        for domain, y0 in row_y.items():
            axis.text(-0.15, y0 + 0.5, frame.loc[domain, "label"], ha="right", va="center", fontsize=5.9)
        gutter_left = -STACK_LABEL_GUTTER / STACK_UNIT + 0.05
        for title, y0 in family_titles:
            axis.text(gutter_left, y0 + 0.85, title, ha="left", va="bottom", fontsize=6.0, color="#555250")
            axis.plot([gutter_left, width_units], [y0 + 0.98] * 2, color=RULE, lw=0.5, clip_on=False)
        # header: objective titles, three-line policy labels, and the high/low sub-labels
        for objective_label, start, end in objective_span:
            axis.text(
                (start + end) / 2,
                -4.1,
                objective_label,
                ha="center",
                va="bottom",
                fontsize=6.3,
                fontweight="bold",
                color="#3A3835",
            )
            axis.plot([start + 0.1, end - 0.1], [-3.95] * 2, color=RULE, lw=0.6, clip_on=False)
        for (objective, key), x0 in col_x.items():
            axis.text(
                x0 + 0.08,
                -1.05,
                POLICY_HEADERS[key][objective],
                ha="left",
                ma="left",
                va="bottom",
                fontsize=5.6,
                color=HEADER_COLORS[key],
                linespacing=1.0,
            )
            for offset, quality in enumerate(QUALITY_LABELS):
                axis.text(x0 + offset + 0.5, -0.12, quality, ha="center", va="bottom", fontsize=5.0, color="#8A8782")
        axis.set_xlim(-0.02, width_units + 0.02)
        axis.set_ylim(height_rows + 0.02, -0.02)
        axis.set_axis_off()
        # key at the bottom
        bar_width = 1.9
        key_left = STACK_LABEL_GUTTER + (grid_width - bar_width) / 2 - 0.5
        cax = figure.add_axes((key_left / fig_width, 0.14 / fig_height, bar_width / fig_width, 0.07 / fig_height))
        colorbar = plt.colorbar(
            plt.cm.ScalarMappable(norm=EPOCH_NORM, cmap=DIVERGING), cax=cax, orientation="horizontal"
        )
        colorbar.set_ticks(EPOCH_TICKS[0])
        colorbar.set_ticklabels(EPOCH_TICKS[1])
        colorbar.ax.tick_params(labelsize=5.4, length=1.5, pad=1.2)
        colorbar.outline.set_linewidth(0.35)
        cax.set_title("Materialized epochs (color and label)", fontsize=6.0, pad=2.5, loc="left", color="#555250")
        figure.text(
            (key_left + bar_width + 0.25) / fig_width,
            (0.14 + 0.035) / fig_height,
            (CROSS + "  zero allocation") if ZERO_GLYPH == CROSS else "",
            ha="left",
            va="center",
            fontsize=6.0,
            color="#555250",
        )
        return figure


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--text", choices=("epochs", "multiplier"), default="epochs")
    parser.add_argument("--layout", choices=("wide", "tall", "blocks", "stacked"), default="stacked")
    parser.add_argument(
        "--marks", choices=("heat", "bars", "weights"), default="heat", help="colored cells, midline bars or weight bars"
    )
    parser.add_argument("--zero-mark", choices=("cross", "zero"), default="zero", help="glyph for zero-allocation cells")
    parser.add_argument("--stem", default=None)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    global ZERO_GLYPH
    ZERO_GLYPH = CROSS if args.zero_mark == "cross" else "0"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.stem or f"mixture_{args.marks}_{args.layout}_{args.text}"
    if args.layout == "stacked":
        figure = build_stacked_figure(args.text)
    elif args.layout == "blocks":
        figure = build_blocks_figure(args.text)
    else:
        figure = (build_tall_figure if args.layout == "tall" else build_wide_figure)(args.text, args.marks)
    for extension in ("png", "pdf"):
        figure.savefig(
            args.output_dir / f"{stem}.{extension}", dpi=layout.STATIC_DPI, bbox_inches="tight", pad_inches=0.02
        )
    plt.close(figure)
    print(f"wrote {args.output_dir / stem}.pdf")


if __name__ == "__main__":
    main()
