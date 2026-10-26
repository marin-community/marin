# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Figure 12 of the Analysis: predictions against measured loss at the frontier of the held-out optima.

Two rows of five panels. Each column holds the 30 best-measured held-out optima of one objective (Uncheatable,
OlmoBaseEval Easy) or one component (basic-skills arithmetic, arXiv physics, SciQ, in ascending SNR), with MARINER's
predictions in the top row and Olmix's in the bottom row on the same mixtures and the same axes. Objective-level
predictions come from the frozen packages (`analyze_top_band_ordering_refresh_20260928.PACKAGES`), component
predictions from `analyze_heldout_component_ordering_20260929.component_frame`. The grey band is one run standard
deviation around the diagonal, the pair threshold of the ordering analyses; the Spearman correlation inside each panel
is over the 30 optima.

Writes `analysis_frontier_scatter.pdf/.png` to `reference_outputs/analysis_frontier_scatter_20260930/`.

usage: uv run --offline --no-sync python plot_analysis_frontier_scatter_20260930.py [--best 30] [--output-dir DIR]
"""

import argparse
import dataclasses
import sys
from pathlib import Path

import matplotlib as mpl
import numpy as np
import pandas as pd
from scipy import stats

mpl.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parents[3]))

import analyze_delphi_top_band_ordering_20260906 as top_band  # noqa: E402
import analyze_heldout_component_ordering_20260929 as ordering  # noqa: E402
import analyze_top_band_ordering_refresh_20260928 as refresh  # noqa: E402
import plot_analysis_floor_20260928 as floor  # noqa: E402

OUTPUT = floor.REFERENCE / "analysis_frontier_scatter_20260930"
RELIABILITY = floor.REFERENCE / "table9_reliability_mariner_20260908"
OBJECTIVE_SNR_FILES = {"uncheatable": "snr_fit_uncheatable_delphi.csv", "table9": "snr_fit_components_delphi.csv"}
BEST = 30
BAND = "#e3e5e9"
DIAGONAL = "#b4b8c0"


@dataclasses.dataclass(frozen=True)
class Panel:
    title: str
    target: str
    component: str | None
    snr: float | None


PANELS = (  # objectives, then components in ascending SNR
    Panel("Uncheatable", "uncheatable", None, None),
    Panel("OlmoBaseEval Easy", "table9", None, None),
    Panel("Arithmetic", "table9", "basic_skills_arithmetic", None),
    Panel("arXiv physics", "uncheatable", "arxiv_physics", None),
    Panel("SciQ", "table9", "sciq", None),
)
GROUPS = (("Objectives", 0, 1), ("Components", 2, 4))  # header, first column, last column
ROWS = (("MARINER", "MARINER"), ("Olmix", "OLMix"))  # row label, label in the frozen packages
# Method colours as in the paper's other figures (MARINER green, Olmix pink).
MARINER_COLOR = "#469C76"
OLMIX_COLOR = "#CC79A7"
# Geometry in inches: square panels, a narrow gap inside each group, a wider gap between the groups, and no x tick
# labels on the top row (each column shares one axis range).
FIGURE_WIDTH = 5.5
PANEL = 0.83
LEFT_GUTTER = 0.33
COLUMN_GAP = 0.26  # room for the next panel's y tick labels
GROUP_GAP = 0.25
BOTTOM_MARGIN = 0.42  # x tick labels and the x label of the bottom row
ROW_GAP = 0.09
TITLE_BAND = 0.24  # two-line panel titles
HEADER_BAND = 0.2  # group headers and their rules
FIGURE_HEIGHT = BOTTOM_MARGIN + 2 * PANEL + ROW_GAP + TITLE_BAND + HEADER_BAND
COLUMN_X = tuple(
    (LEFT_GUTTER + i * (PANEL + COLUMN_GAP) + (GROUP_GAP if i >= 2 else 0.0)) / FIGURE_WIDTH for i in range(5)
)
ROW_Y = ((BOTTOM_MARGIN + PANEL + ROW_GAP) / FIGURE_HEIGHT, BOTTOM_MARGIN / FIGURE_HEIGHT)
PANEL_WIDTH, PANEL_HEIGHT = PANEL / FIGURE_WIDTH, PANEL / FIGURE_HEIGHT


def objective_snr(target: str) -> float:
    """The objective aggregate's SNR from the reliability tables (its macro row has no component)."""
    table = pd.read_csv(RELIABILITY / OBJECTIVE_SNR_FILES[target])
    return float(table[table.component.isna()].snr.iloc[0])


def objective_frontier(target: str, frozen) -> tuple[np.ndarray, dict[str, np.ndarray], float]:
    labels = ordering.held_out_labels(target)
    predicted = {
        label: frozen[frozen.target.eq(target) & frozen.label.eq(label)]
        .set_index("row_id")
        .prediction.reindex(labels.coordinate_id)
        .to_numpy(float)
        for _row, label in ROWS
    }
    return labels.measured_mean_bpb.to_numpy(float), predicted, top_band.RUN_SD[target]


def draw(axis, measured, predicted, span, run_sd, color, title: str | None, x_label: bool, best_count: int) -> None:
    best = np.argsort(measured, kind="stable")[:best_count]
    y, p = measured[best], predicted[best]
    axis.fill_between(span, span - run_sd, span + run_sd, color=BAND, linewidth=0, zorder=1)
    axis.plot(span, span, color=DIAGONAL, linewidth=0.8, zorder=2)
    axis.scatter(y, p, s=8, color=color, edgecolor="#222222", linewidths=0.3, zorder=3)
    axis.set_xlim(*span)
    axis.set_ylim(*span)
    axis.set_aspect("equal")
    axis.xaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=2))
    axis.yaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=2))
    axis.tick_params(labelsize=5.8)
    if title is not None:
        axis.set_title(title, loc="left", fontsize=5.9, fontweight="bold", linespacing=1.1)
    rho = float(stats.spearmanr(y, p).statistic)
    axis.text(
        0.96,
        0.05,
        f"Spearman {rho:.2f}",
        transform=axis.transAxes,
        fontsize=5.6,
        ha="right",
        va="bottom",
        zorder=4,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 0.8},
    )
    if x_label:
        axis.set_xlabel("Measured loss (BPB)", fontsize=6.2)
    else:
        axis.tick_params(axis="x", labelbottom=False)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)


def column_span(measured: np.ndarray, predictions: dict[str, np.ndarray], best_count: int) -> np.ndarray:
    """One axis range per column: the best mixtures' measured losses and both models' predictions of them."""
    best = np.argsort(measured, kind="stable")[:best_count]
    values = np.concatenate([measured[best], *(p[best] for p in predictions.values())])
    pad = 0.08 * np.ptp(values)
    return np.array([values.min() - pad, values.max() + pad])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--best", type=int, default=BEST, help="number of best-measured held-out mixtures per panel")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    top_band.PACKAGES = refresh.PACKAGES
    frozen = top_band.load_predictions()
    plt.rcParams.update(floor.original.PLOT_STYLE)
    plt.rcParams.update({"text.usetex": False})
    figure = plt.figure(figsize=(FIGURE_WIDTH, FIGURE_HEIGHT), facecolor="white")
    columns = []
    for panel in PANELS:
        if panel.component is None:
            measured, predicted, run_sd = objective_frontier(panel.target, frozen)
            snr = objective_snr(panel.target)
        else:
            frame = ordering.component_frame(panel.target, panel.component)
            measured, run_sd, snr = frame.measured, frame.run_sd, frame.snr
            predicted = {"MARINER": frame.mariner, "OLMix": frame.olmix}
        columns.append((dataclasses.replace(panel, snr=snr), measured, predicted, run_sd))
    for row_index, ((row_label, label), y0) in enumerate(zip(ROWS, ROW_Y, strict=True)):
        for column_index, (panel, measured, predicted, run_sd) in enumerate(columns):
            axis = figure.add_axes((COLUMN_X[column_index], y0, PANEL_WIDTH, PANEL_HEIGHT))
            color = MARINER_COLOR if row_label == "MARINER" else OLMIX_COLOR
            title = f"{panel.title}\nSNR {panel.snr:.0f}, {args.best} best" if row_index == 0 else None
            span = column_span(measured, predicted, args.best)
            draw(axis, measured, predicted[label], span, run_sd, color, title, row_index == 1, args.best)
            if column_index == 0:
                axis.set_ylabel(f"{row_label}\npredicted (BPB)", fontsize=6.2, linespacing=1.1)
    rule_y = 1.0 - (HEADER_BAND - 0.03) / FIGURE_HEIGHT
    for header, first, last in GROUPS:
        left, right = COLUMN_X[first], COLUMN_X[last] + PANEL_WIDTH
        figure.text((left + right) / 2, 0.995, header, ha="center", va="top", fontsize=6.6, fontweight="bold")
        figure.add_artist(
            plt.Line2D([left, right], [rule_y, rule_y], transform=figure.transFigure, color="#777777", linewidth=0.6)
        )
    stem = "analysis_frontier_scatter" if args.best == BEST else f"analysis_frontier_scatter_best{args.best}"
    for extension in ("pdf", "png"):
        figure.savefig(args.output_dir / f"{stem}.{extension}", dpi=300, bbox_inches="tight", pad_inches=0.02)
    print(args.output_dir / f"{stem}.pdf")


if __name__ == "__main__":
    main()
