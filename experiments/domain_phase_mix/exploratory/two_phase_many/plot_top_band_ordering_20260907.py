# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Paper figure: where the surrogate's ordering skill ends, from `analyze_delphi_top_band_ordering_20260906.py`.

Two panels (OlmoBaseEval Easy, Uncheatable): pairwise sign accuracy inside the k best-measured coordinates of
the held-out optima stratum for WSPU, DSP, Olmix and the held-out-source bank kernel, with the noise ceiling of a
perfect predictor against single-run measurements. Reads `top_band_ordering.csv`; writes figure.{png,pdf} and,
with --drive-dir, copies them as `a_top_band_ordering`.

usage: uv run python plot_top_band_ordering_20260907.py [--drive-dir DIR]
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import matplotlib as mpl
import pandas as pd

mpl.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
INPUT = SCRIPT_DIR / "reference_outputs" / "delphi_top_band_ordering_20260906" / "top_band_ordering.csv"
OUTPUT_DIR = SCRIPT_DIR / "reference_outputs" / "top_band_ordering_figure_20260907"
DRIVE_STEM = "a_top_band_ordering"
INK = "#111111"
GRID = "#b8b8b8"
PAPER = "white"
SERIES = (
    ("WSPU", "WSPU", "#178A72", "-"),
    ("DSP", "DSP", "#E69F00", "-"),
    ("OLMix", "Olmix", "#CC79A7", "-"),
    ("bank kernel LOSO, TV 0.05", "Measured neighbours (held-out source)", "#0072B2", "-"),
)
PANELS = (("table9", "A. OlmoBaseEval Easy, suite mean (51)"), ("uncheatable", "B. Uncheatable"))
PLOT_STYLE = {
    "font.family": "DejaVu Sans",
    "font.size": 8,
    "text.usetex": False,
    "axes.grid": False,
    "lines.markeredgewidth": 0.8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "figure.facecolor": PAPER,
    "axes.facecolor": PAPER,
    "savefig.facecolor": PAPER,
}
DPI = 300


def style(axis: plt.Axes, sizes: list[int]) -> None:
    axis.set_xscale("log")
    axis.set_xticks(sizes)
    axis.set_xticklabels([str(size) for size in sizes])
    axis.minorticks_off()
    axis.grid(True, axis="y", color=GRID, alpha=0.72, linewidth=0.6, zorder=0)
    axis.tick_params(colors=INK, labelsize=7)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_color(INK)
    axis.set_xlabel("Band: k best-measured mixtures", color=INK, fontsize=7.5)
    axis.set_ylim(0, 1.0)


def draw(table: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.1), constrained_layout=True)
    for axis, (target, title) in zip(axes, PANELS, strict=True):
        sub = table[table.target.eq(target)]
        sizes = sorted(sub.band.unique())
        ceiling = sub[sub.method.eq("WSPU")].sort_values("band")
        axis.plot(ceiling.band, ceiling.noise_ceiling, color=INK, ls="--", lw=0.9, zorder=3, label="Noise ceiling")
        axis.axhline(0.5, color=GRID, lw=0.7, zorder=1)
        for key, label, colour, ls in SERIES:
            line = sub[sub.method.eq(key)].sort_values("band")
            axis.plot(
                line.band, line.sign_accuracy, color=colour, ls=ls, marker="o", ms=2.6, lw=1.1, zorder=4, label=label
            )
        style(axis, sizes)
        axis.set_title(title, loc="left", fontsize=8, fontweight="bold", color=INK)
        axis.text(sizes[0], 0.515, "chance", color=INK, fontsize=6, va="bottom", ha="left")
    axes[0].set_ylabel("Pairwise sign accuracy", color=INK, fontsize=7.5)
    axes[1].legend(frameon=False, fontsize=6, loc="lower right", handlelength=1.8)
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--drive-dir", type=Path, default=None)
    args = parser.parse_args()
    plt.rcParams.update(PLOT_STYLE)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    table = pd.read_csv(INPUT)
    fig = draw(table)
    for suffix in ("png", "pdf"):
        fig.savefig(OUTPUT_DIR / f"figure.{suffix}", dpi=DPI)
        if args.drive_dir is not None:
            shutil.copy(OUTPUT_DIR / f"figure.{suffix}", args.drive_dir / f"{DRIVE_STEM}.{suffix}")
    print(f"wrote {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
