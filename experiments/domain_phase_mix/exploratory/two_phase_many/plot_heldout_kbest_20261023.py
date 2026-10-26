# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Paper figure of heldout_kbest_prediction_20261023: mean absolute error of each model's predictions on the k best
held-out mixtures, Uncheatable (left) and OlmoBaseEval Easy (right), with 95% bootstrap bands and the noise floor.

The figure prints at 100% beside the on/off-objective figure in one float row, so it repeats that pair's vertical
geometry (the PAIR_* block of the paper folder's revision_notes/20261012_fig8_legend builders): the legend rows, panel
titles, axes boxes, x tick labels and x label sit on the same baselines, and the outermost ink sits PAIR_INK_INSET
inside the left and right edges. Legend entries give each model's response parameters per task (Table 13's counts);
tuned RegMix's is the median task's tree leaves (1,000 trees of 4 leaves). Reads the frozen held-out set's
kbest_metrics.csv; writes heldout_kbest.{pdf,png} beside it.

usage: uv run --offline --no-sync python -m \\
    experiments.domain_phase_mix.exploratory.two_phase_many.plot_heldout_kbest_20261023
"""

import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.text import Text
from matplotlib.ticker import NullLocator
from PIL import Image

OUT = Path(__file__).resolve().parent / "reference_outputs" / "heldout_kbest_prediction_20261023" / "frozen"
GRID, NOISE, NOISE_TEXT = "#b8b8b8", "#8c8c8c", "#666666"
MODELS = {  # model: (legend label, colour, line width); the paper's method colours since 2026-10-13
    "mariner": ("MARINER (83 parameters)", "#469C76", 1.5),
    "olmix": ("Olmix (40)", "#CC79A7", 1.1),
    "spline": ("Cubic spline (158)", "#0072B2", 1.1),
    "regmix": ("RegMix (tuned, 4,000 leaves)", "#B35806", 1.1),
}
LEGEND_ROWS = (("mariner", "olmix"), ("spline", "regmix"))
TARGETS = {"uncheatable": "Uncheatable", "table9": "OlmoBaseEval Easy"}
X_TICKS = (10, 30, 100, 300)
X_LIMITS = (8.5, 470)
Y_TICKS = (0.001, 0.003, 0.01, 0.03)
Y_LIMITS = (0.00052, 0.062)

# The pair's shared geometry; vertical positions are points below the canvas top.
PAIR_SIZE_IN = (2.7, 2.0)
PAIR_INK_INSET = 0.85
PAIR_LEGEND_SIZE = 6.0
PAIR_TICK_SIZE = 6.3
PAIR_LABEL_SIZE = 6.5
PAIR_TITLE_SIZE = 7.0
PAIR_LEGEND_BASELINES = (5.51, 15.01)
PAIR_TITLE_BASELINE = 27.5
PAIR_AXES_TOP = 34.25
PAIR_AXES_BOTTOM = 125.85
PAIR_XTICK_BASELINE = 133.45
PAIR_XLABEL_BASELINE = 141.80
PANEL_GAP = 8.0  # the shared y axis leaves only the gap between the two panels
SPINE_CAP = 0.5  # the bottom spine's projecting cap reaches half its line width past the axes
TICK_PAD = 2.0
INK_DPI = 1200
INK_TOLERANCE = 0.06
LAYOUT_ITERATIONS = 8


@dataclass
class Anchors:
    """Distances from the left and right canvas edges to the outermost boxes, corrected until the ink sits
    PAIR_INK_INSET inside."""

    left: float = PAIR_INK_INSET
    right: float = PAIR_INK_INSET


def box_pt(artist, renderer, dpi: float) -> tuple[float, float, float, float]:
    box = artist.get_window_extent(renderer)
    scale = 72 / dpi
    return box.x0 * scale, box.y0 * scale, box.x1 * scale, box.y1 * scale


def baseline_pt(text: Text, renderer, dpi: float) -> float:
    """Baseline of a one-line text in points above the canvas bottom (after a draw)."""
    box = text.get_window_extent(renderer)
    _, _, descent = text._get_layout(renderer)
    return (box.y0 + descent) * 72 / dpi


def ink_margins_pt(pdf: Path) -> dict[str, float]:
    """Distance from each page edge to the first pixel below 250 on any channel (trim_figures.py's ink rule)."""
    with tempfile.TemporaryDirectory(dir=pdf.parent) as directory:
        subprocess.run(
            ["pdftoppm", "-r", str(INK_DPI), "-png", "-singlefile", str(pdf), f"{directory}/page"], check=True
        )
        pixels = np.asarray(Image.open(f"{directory}/page.png").convert("RGB"))
    rows, cols = np.where((pixels < 250).any(axis=2))
    height, width = pixels.shape[:2]
    scale = 72 / INK_DPI
    return {
        "left": cols.min() * scale,
        "right": (width - 1 - cols.max()) * scale,
        "top": rows.min() * scale,
        "bottom": (height - 1 - rows.max()) * scale,
    }


def place(fig, axes, titles, xlabel, legends, anchors: Anchors) -> None:
    """Puts the pair's baselines in place and splits the width between the anchors into two equal panels."""
    width, height = (size * 72 for size in fig.get_size_inches())
    axes_bottom, axes_top = height - PAIR_AXES_BOTTOM, height - PAIR_AXES_TOP
    for _ in range(3):  # tick-label boxes move with the axes, so settle the positions
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        boxes = [box_pt(label, renderer, fig.dpi) for label in axes[0].get_yticklabels() if label.get_text()]
        boxes.append(box_pt(axes[0].yaxis.label, renderer, fig.dpi))
        left = anchors.left + axes[0].get_position().x0 * width - min(box[0] for box in boxes)
        panel = (width - anchors.right - SPINE_CAP - left - PANEL_GAP) / 2
        lefts = (left, left + panel + PANEL_GAP)
        for ax, x in zip(axes, lefts, strict=True):
            ax.set_position((x / width, axes_bottom / height, panel / width, (axes_top - axes_bottom) / height))
        for text, x in zip(titles, lefts, strict=True):
            text.set_position(((x + panel / 2) / width, (height - PAIR_TITLE_BASELINE) / height))
        xlabel.set_position(((lefts[0] + lefts[1] + panel) / 2 / width, (height - PAIR_XLABEL_BASELINE) / height))
    # Legend anchors and tick pads move their baselines linearly, so one measured correction lands them.
    for _ in range(2):
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        for legend, target in zip(legends, PAIR_LEGEND_BASELINES, strict=True):
            error = (height - baseline_pt(legend.get_texts()[0], renderer, fig.dpi)) - target  # > 0: too low
            anchor = legend.get_bbox_to_anchor().transformed(fig.transFigure.inverted())
            legend.set_bbox_to_anchor((0.5, anchor.y0 + error / height), transform=fig.transFigure)
        for ax in axes:
            label = next(label for label in ax.get_xticklabels() if label.get_text())
            error = (height - baseline_pt(label, renderer, fig.dpi)) - PAIR_XTICK_BASELINE
            ax.tick_params(axis="x", pad=ax.xaxis.get_major_ticks()[0].get_pad() - error)


def draw_panel(ax, part: pd.DataFrame) -> None:
    reference = part.drop_duplicates("k").sort_values("k")
    ax.plot(reference.k, reference.noise_mae, color=NOISE, lw=0.8, ls=(0, (2, 2)), zorder=1)
    for model, (_label, color, width) in MODELS.items():
        line = part[part.model.eq(model)].sort_values("k")
        ax.fill_between(line.k, line.mae_low, line.mae_high, color=color, alpha=0.15, lw=0, zorder=1)
        ax.plot(line.k, line.mae, color=color, lw=width, solid_capstyle="butt", zorder=3 if model == "mariner" else 2)
    end = reference.iloc[-1]
    ax.text(
        float(end.k),
        float(end.noise_mae) * 1.12,
        "noise floor",
        color=NOISE_TEXT,
        fontsize=PAIR_LEGEND_SIZE,
        ha="right",
        va="bottom",
    )
    ax.set(xscale="log", yscale="log", xlim=X_LIMITS, ylim=Y_LIMITS)
    ax.set_xticks(X_TICKS, [str(k) for k in X_TICKS])
    ax.set_yticks(Y_TICKS, [f"{v:g}" for v in Y_TICKS])
    ax.xaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_minor_locator(NullLocator())
    ax.tick_params(labelsize=PAIR_TICK_SIZE, pad=TICK_PAD)
    ax.grid(True, axis="y", color=GRID, alpha=0.6, linewidth=0.6, linestyle=(0, (4, 3)), zorder=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def main() -> None:
    plt.rcdefaults()
    plt.rcParams.update(
        {"font.family": "DejaVu Sans", "font.size": 8, "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.bbox": "standard"}
    )
    table = pd.read_csv(OUT / "kbest_metrics.csv")
    fig = plt.figure(figsize=PAIR_SIZE_IN)
    height = PAIR_SIZE_IN[1] * 72
    axes = [fig.add_axes((0.15 + 0.45 * i, 0.2, 0.35, 0.6)) for i in range(2)]
    for ax, target in zip(axes, TARGETS, strict=True):
        draw_panel(ax, table[table.target.eq(target)])
    axes[1].tick_params(labelleft=False)
    axes[0].set_ylabel("Mean absolute error (BPB)", fontsize=PAIR_LABEL_SIZE, labelpad=3.0)
    titles = [
        fig.text(0.5, 0.5, title, fontsize=PAIR_TITLE_SIZE, fontweight="bold", ha="center", va="baseline")
        for title in TARGETS.values()
    ]
    xlabel = fig.text(0.5, 0.0, "k best held-out mixtures", fontsize=PAIR_LABEL_SIZE, ha="center", va="baseline")
    legends = [
        fig.legend(
            handles=[Line2D([], [], color=MODELS[m][1], lw=MODELS[m][2], label=MODELS[m][0]) for m in row],
            loc="lower center",
            bbox_to_anchor=(0.5, (height - baseline) / height),
            ncol=len(row),
            frameon=False,
            fontsize=PAIR_LEGEND_SIZE,
            handlelength=1.6,
            columnspacing=1.0,
            handletextpad=0.4,
            borderpad=0.0,
            borderaxespad=0.0,
        )
        for row, baseline in zip(LEGEND_ROWS, PAIR_LEGEND_BASELINES, strict=True)
    ]
    anchors = Anchors()
    pdf = OUT / "heldout_kbest.pdf"
    for _ in range(LAYOUT_ITERATIONS):
        place(fig, axes, titles, xlabel, legends, anchors)
        fig.savefig(pdf)
        margins = ink_margins_pt(pdf)
        if all(abs(margins[side] - PAIR_INK_INSET) <= INK_TOLERANCE for side in ("left", "right")):
            break
        for side in ("left", "right"):
            setattr(anchors, side, getattr(anchors, side) + PAIR_INK_INSET - margins[side])
    else:
        raise RuntimeError(f"ink margins did not converge: {margins}")
    fig.savefig(OUT / "heldout_kbest.png", dpi=300)
    print("ink margins (pt):", {side: round(float(value), 2) for side, value in margins.items()})
    print(pdf)


if __name__ == "__main__":
    main()
