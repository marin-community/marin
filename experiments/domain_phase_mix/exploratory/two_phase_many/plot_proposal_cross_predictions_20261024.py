# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Paper figure of collect_proposal_cross_predictions_20261024: each trained proposal is a column with its measured
loss (mean and 95% interval over three trainer seeds, in the proposing model's colour) and the four models' predicted
losses, Uncheatable (left) and OlmoBaseEval Easy (right). Lower is better. Printed at 100%: `--layout half` (2.7 in wide)
or `--layout full` (5.5 in, one legend row, each column labelled with its proposing model, as in "MARINER's proposal";
the paper's Figure 7 since 2026-10-25).

usage: uv run --offline --no-sync python -m \\
    experiments.domain_phase_mix.exploratory.two_phase_many.plot_proposal_cross_predictions_20261024 --layout full
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

OUT = Path(__file__).resolve().parent / "reference_outputs" / "proposal_cross_predictions_20261024"
INK, GRID, MEASURED_KEY = "#111111", "#b8b8b8", "#555555"
MODELS = {  # key: (column label, legend name, colour, marker, marker size)
    "mariner": ("MARINER", "MARINER", "#469C76", "o", 4.4),
    "olmix": ("Olmix", "Olmix", "#CC79A7", "D", 4.0),
    "spline": ("Cubic spline", "Cubic spline", "#0072B2", "s", 4.3),
    "lgbm": ("RegMix (tuned)", "RegMix", "#B35806", "^", 5.0),
}
TARGETS = {"uncheatable": "Uncheatable", "table9": "OlmoBaseEval Easy"}
SIZE_IN = {"half": (2.7, 2.45), "full": (5.5, 2.0)}
BAR_HALF_WIDTH = 0.32
# Layout lengths in points.
MARGIN = 1.0  # outermost ink to the canvas edges (trim_figures.py trims the rest)
PANEL_GAP = 7.0  # the left panel's right edge to the right panel's y tick labels
LEGEND_GAP = 5.0  # the legend's last row to the panel titles
XLABEL_GAP = 3.0  # the slanted tick labels to the x label


def draw_panel(ax, measured: pd.DataFrame, predictions: pd.DataFrame, layout: str) -> None:
    for x, (proposal, (_label, _name, color, _marker, _size)) in enumerate(MODELS.items()):
        loss = measured[measured.proposal.eq(proposal)].loss.to_numpy()
        mean, half = loss.mean(), 1.96 * loss.std(ddof=1) / np.sqrt(len(loss))
        span = [x - BAR_HALF_WIDTH, x + BAR_HALF_WIDTH]
        ax.fill_between(span, mean - half, mean + half, color=color, alpha=0.2, lw=0, zorder=1)
        ax.plot(span, [mean, mean], color=color, lw=1.8, solid_capstyle="butt", zorder=2)
        here = predictions[predictions.proposal.eq(proposal)].set_index("predictor").prediction
        for predictor, (_label, _name, mark_color, marker, size) in MODELS.items():
            if predictor == "mariner":
                ax.plot(x, here[predictor], marker, color=mark_color, ms=size, mec="white", mew=0.5, zorder=4)
            else:
                ax.plot(x, here[predictor], marker, mfc="none", mec=mark_color, ms=size, mew=0.9, zorder=3)
    ax.set_xlim(-0.55, len(MODELS) - 0.45)
    if layout == "full":  # horizontal, each column names the model whose proposal it is, so no x label is needed
        ax.set_xticks(range(len(MODELS)), [f"{name}'s\nproposal" for _label, name, *_ in MODELS.values()])
    else:
        labels = [label for label, *_ in MODELS.values()]
        ax.set_xticks(range(len(MODELS)), labels, rotation=35, ha="right", rotation_mode="anchor")
    for label, (_l, _n, color, _m, _s) in zip(ax.get_xticklabels(), MODELS.values(), strict=True):
        label.set_color(color)
    ax.tick_params(labelsize=6.3, length=2.8, pad=2.0)
    ax.grid(True, axis="y", color=GRID, alpha=0.6, linewidth=0.6, linestyle=(0, (4, 3)), zorder=0)
    ax.spines[["top", "right"]].set_visible(False)


def box_pt(artist, renderer, dpi: float) -> tuple[float, float, float, float]:
    box = artist.get_window_extent(renderer)
    return box.x0 * 72 / dpi, box.y0 * 72 / dpi, box.x1 * 72 / dpi, box.y1 * 72 / dpi


def place(fig, axes, legend, xlabel) -> None:
    """Explicit layout: the legend under the top edge, two equal panels whose centred titles stay on the canvas, and
    the tick labels and the x label (half layout only) above the bottom edge."""
    width, height = (size * 72 for size in fig.get_size_inches())
    for _ in range(3):  # tick-label boxes move with the axes, so settle the positions
        fig.canvas.draw()
        renderer, dpi = fig.canvas.get_renderer(), fig.dpi
        frames = [box_pt(ax, renderer, dpi) for ax in axes]  # the axes boxes
        left_ticks = [
            frame[0] - min(box_pt(t, renderer, dpi)[0] for t in ax.get_yticklabels() if t.get_text())
            for frame, ax in zip(frames, axes, strict=True)
        ]
        ylabel = frames[0][0] - box_pt(axes[0].yaxis.label, renderer, dpi)[0]
        title = box_pt(axes[1].title, renderer, dpi)
        x0 = MARGIN + max(left_ticks[0], ylabel)
        gap = PANEL_GAP + left_ticks[1]
        span = width - MARGIN - x0 - gap
        panel = min(span / 2, (2 * span - (title[2] - title[0])) / 3)
        above = max(box_pt(ax.title, renderer, dpi)[3] for ax in axes) - frames[0][3]
        below = max(
            frame[1] - min(box_pt(t, renderer, dpi)[1] for t in ax.get_xticklabels())
            for frame, ax in zip(frames, axes, strict=True)
        )
        key = box_pt(legend, renderer, dpi)
        top = height - (key[1] - LEGEND_GAP) + above
        if xlabel is None:
            bottom = MARGIN + below
        else:
            label = box_pt(xlabel, renderer, dpi)
            bottom = MARGIN + (label[3] - label[1]) + XLABEL_GAP + below
        for ax, left in zip(axes, (x0, x0 + panel + gap), strict=True):
            ax.set_position((left / width, bottom / height, panel / width, (height - top - bottom) / height))
        if xlabel is not None:
            xlabel.set_position(((x0 + (2 * panel + gap) / 2) / width, MARGIN / height))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layout", choices=("half", "full"), default="half")
    layout = parser.parse_args().layout
    size_in = SIZE_IN[layout]
    plt.rcdefaults()
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 6.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "text.color": INK,
            "axes.labelcolor": INK,
        }
    )
    measured = pd.read_csv(OUT / "measured_runs.csv")
    predictions = pd.read_csv(OUT / "cross_predictions.csv")
    fig = plt.figure(figsize=size_in)
    axes = [fig.add_axes((0.15 + 0.45 * i, 0.3, 0.3, 0.4)) for i in range(2)]
    for ax, (target, title) in zip(axes, TARGETS.items(), strict=True):
        draw_panel(ax, measured[measured.target.eq(target)], predictions[predictions.target.eq(target)], layout)
        ax.set_title(title, fontsize=7.0, fontweight="bold", pad=4)
    axes[0].set_ylabel("Loss (BPB)", fontsize=6.5, labelpad=3)
    xlabel = fig.text(0.5, 0.0, "Proposed by", fontsize=6.5, ha="center", va="bottom") if layout == "half" else None
    marks = {
        key: Line2D(
            [],
            [],
            ls="",
            marker=marker,
            ms=size,
            color=color,
            mec="white" if key == "mariner" else color,
            mfc=color if key == "mariner" else "none",
            mew=0.5 if key == "mariner" else 0.9,
        )
        for key, (_label, _name, color, marker, size) in MODELS.items()
    }
    measured_key = (Line2D([], [], color=MEASURED_KEY, lw=1.8), "Measured (95% CI)")
    if layout == "full":  # one row
        entries = [
            measured_key,
            (marks["mariner"], "MARINER's prediction"),
            (marks["olmix"], "Olmix's prediction"),
            (marks["spline"], "Cubic spline's prediction"),
            (marks["lgbm"], "RegMix's prediction"),
        ]
    else:  # ncol=2 fills columns first: rows read (measured, MARINER), (Olmix, spline), (RegMix)
        entries = [
            measured_key,
            (marks["olmix"], "Olmix's prediction"),
            (marks["lgbm"], "RegMix's prediction"),
            (marks["mariner"], "MARINER's prediction"),
            (marks["spline"], "Cubic spline's prediction"),
        ]
    legend = fig.legend(
        *zip(*entries, strict=True),
        loc="upper center",
        bbox_to_anchor=(0.5, 1 - MARGIN / (size_in[1] * 72)),
        ncol=len(entries) if layout == "full" else 2,
        frameon=False,
        fontsize=6.0,
        handlelength=1.6,
        handletextpad=0.4,
        columnspacing=0.55 if layout == "full" else 1.0,
        labelspacing=0.35,
        borderpad=0.0,
        borderaxespad=0.0,
    )
    place(fig, axes, legend, xlabel)
    stem = "proposal_cross_predictions" + ("_full" if layout == "full" else "")
    for suffix in ("pdf", "png"):
        fig.savefig(OUT / f"{stem}.{suffix}", dpi=300)
    print(OUT / f"{stem}.pdf")


if __name__ == "__main__":
    main()
