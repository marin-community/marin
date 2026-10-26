# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""The Analysis section's figure: where MARINER's ordering ends, and what a local factorial resolves.

Panels A and B plot MARINER's pairwise sign accuracy among the k best-measured held-out optima on each objective
(`delphi_top_band_ordering_20260928/top_band_ordering.csv`) with its source-block bootstrap interval, the noise
ceiling, and the held-out-source bank kernel. Panel C plots the frontier factorial's measured main effects (with SE)
against MARINER's predicted effects at the same mixtures (`factorial_vs_mariner_20260928/effects_comparison.csv`).

Writes `analysis_floor.pdf` and `.png` to `reference_outputs/analysis_floor_figure_20260928/`.

usage: uv run --offline --no-sync python plot_analysis_floor_20260928.py
"""

import sys
from pathlib import Path

import matplotlib as mpl
import numpy as np
import pandas as pd

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, NullLocator

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parents[3]))

from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    plot_scaling_curves_placeholder_20260905 as original,
)

REFERENCE = SCRIPT_DIR / "reference_outputs"
BANDS = REFERENCE / "delphi_top_band_ordering_20260928" / "top_band_ordering.csv"
EFFECTS = REFERENCE / "factorial_vs_mariner_20260928" / "effects_comparison.csv"
OUTPUT = REFERENCE / "analysis_floor_figure_20260928"
KERNEL = "bank kernel LOSO, TV 0.05"
TARGETS = (("uncheatable", "A. Uncheatable"), ("table9", "B. OlmoBaseEval Easy"))
MUTED = "#8a8f98"
TARGET_COLORS = {"uncheatable": "#0072B2", "table9": "#D55E00"}
TARGET_LABELS = {"uncheatable": "Uncheatable", "table9": "OlmoBaseEval Easy"}
CEILING = "#3b3f47"


def draw_bands(axis, table: pd.DataFrame, target: str, title: str) -> None:
    rows = table[table.target.eq(target) & table.method.eq("MARINER")].sort_values("band")
    kernel = table[table.target.eq(target) & table.method.eq(KERNEL)].sort_values("band")
    x = rows.band.to_numpy(float)
    axis.fill_between(
        x, rows.sign_accuracy_ci_low, rows.sign_accuracy_ci_high, color=original.MARINER_COLOR, alpha=0.18, linewidth=0
    )
    axis.plot(x, rows.noise_ceiling, color=CEILING, linewidth=1.0, linestyle=(0, (3, 2)), zorder=3)
    axis.plot(kernel.band, kernel.sign_accuracy, color=MUTED, linewidth=0.9, marker="o", markersize=2.2, zorder=3)
    axis.plot(x, rows.sign_accuracy, color=original.MARINER_COLOR, linewidth=1.4, marker="o", markersize=3.0, zorder=4)
    axis.axhline(0.5, color=original.GRID, linewidth=0.7, zorder=1)
    axis.set_xscale("log")
    axis.xaxis.set_major_locator(FixedLocator(list(x)))
    axis.xaxis.set_minor_locator(NullLocator())
    axis.set_xticklabels([str(int(b)) if b < rows.band.max() else "all" for b in x], fontsize=7)
    # The 100 and "all" ticks are close on the log axis; lean their labels apart.
    labels = axis.xaxis.get_majorticklabels()
    labels[-2].set_ha("right")
    labels[-1].set_ha("left")
    axis.set_ylim(0.0, 1.02)
    axis.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    axis.tick_params(labelsize=7)
    axis.set_title(title, loc="left", fontsize=8, fontweight="bold")
    axis.set_xlabel("$k$ best held-out optima", fontsize=7.5)
    axis.grid(True, axis="y", color=original.GRID, alpha=0.6, linewidth=0.6, linestyle=(0, (4, 3)), zorder=0)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)


def draw_effects(axis, effects: pd.DataFrame) -> None:
    main = effects[effects.kind.eq("main")]
    limit = 1.15 * float(np.abs(main[["effect_bpb", "predicted_effect_bpb"]].to_numpy()).max())
    axis.plot([-limit, limit], [-limit, limit], color=original.GRID, linewidth=0.8, zorder=1)
    axis.axhline(0, color=original.GRID, linewidth=0.5, zorder=1)
    axis.axvline(0, color=original.GRID, linewidth=0.5, zorder=1)
    for target, color in TARGET_COLORS.items():
        rows = main[main.target.eq(target)]
        axis.errorbar(
            rows.effect_bpb,
            rows.predicted_effect_bpb,
            xerr=rows.se,
            fmt="o",
            color=color,
            markersize=4.0,
            markeredgecolor="white",
            markeredgewidth=0.5,
            elinewidth=0.7,
            capsize=1.5,
            zorder=4,
        )
        for row in rows.itertuples():
            axis.annotate(
                row.term,
                (row.effect_bpb, row.predicted_effect_bpb),
                xytext=(3.5, 2.5),
                textcoords="offset points",
                fontsize=6.2,
                color=original.INK,
                zorder=5,
            )
    axis.set_xlim(-limit * 1.06, limit * 1.06)
    axis.set_ylim(-limit * 1.06, limit * 1.06)
    axis.set_aspect("equal")
    ticks = [-0.008, -0.004, 0.0, 0.004, 0.008]
    axis.set_xticks(ticks)
    axis.set_yticks(ticks)
    labels = [f"{1000 * t:+.0f}" if t else "0" for t in ticks]
    axis.set_xticklabels(labels, fontsize=7)
    axis.set_yticklabels(labels, fontsize=7)
    axis.set_title("C. Frontier factorial", loc="left", fontsize=8, fontweight="bold")
    axis.set_xlabel(r"Measured effect ($10^{-3}$ BPB)", fontsize=7)
    axis.set_ylabel(r"Predicted effect ($10^{-3}$ BPB)", fontsize=7)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    # Coloured text instead of marker handles, so the key cannot be read as data points.
    axis.text(
        limit * 0.98,
        -limit * 0.66,
        "Uncheatable",
        fontsize=6.2,
        color=TARGET_COLORS["uncheatable"],
        ha="right",
        va="top",
    )
    axis.text(
        limit * 0.98,
        -limit * 0.84,
        "OlmoBaseEval Easy",
        fontsize=6.2,
        color=TARGET_COLORS["table9"],
        ha="right",
        va="top",
    )


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    bands = pd.read_csv(BANDS)
    effects = pd.read_csv(EFFECTS)
    plt.rcParams.update(original.PLOT_STYLE)
    figure, axes = plt.subplots(1, 3, figsize=(5.5, 1.5), gridspec_kw={"width_ratios": [1, 1, 1.2], "wspace": 0.38})
    for axis, (target, title) in zip(axes[:2], TARGETS, strict=True):
        draw_bands(axis, bands, target, title)
    axes[0].set_ylabel("Pairwise sign accuracy", fontsize=7.5)
    handles = [
        Line2D([], [], color=original.MARINER_COLOR, marker="o", markersize=3, linewidth=1.4, label="MARINER"),
        Line2D([], [], color=MUTED, marker="o", markersize=2.2, linewidth=0.9, label="Neighbour kernel"),
        Line2D([], [], color=CEILING, linewidth=1.0, linestyle=(0, (3, 2)), label="Noise reference"),
    ]
    axes[0].legend(
        handles=handles,
        loc="lower right",
        frameon=False,
        fontsize=5.8,
        handlelength=1.4,
        handletextpad=0.4,
        borderaxespad=0.2,
    )
    draw_effects(axes[2], effects)
    for extension in ("pdf", "png"):
        figure.savefig(OUTPUT / f"analysis_floor.{extension}", dpi=300, bbox_inches="tight", pad_inches=0.02)
    print(OUTPUT / "analysis_floor.pdf")


if __name__ == "__main__":
    main()
