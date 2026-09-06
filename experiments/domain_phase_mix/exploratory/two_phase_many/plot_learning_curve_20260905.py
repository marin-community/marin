# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Learning-curve figure: WSPU versus OLMix as a function of the number of fitted runs.

Reads the summaries written by ``learning_curve_metrics_20260905`` and draws, per target, the mean over
draws with its 95% t interval for out-of-fold rank correlation, held-out-bank rank correlation and
held-out regret at 1 (with the random-ranking expectation as reference). An appendix variant adds
complement rank correlation, out-of-fold RMSE in repeat-SD units, fold-mean regret and top-5 regret.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import matplotlib as mpl
import numpy as np
import pandas as pd

mpl.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    learning_curve_fits_20260905 as fits,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    learning_curve_metrics_20260905 as metrics_module,
)

INK = "#111111"
GRID = "#b8b8b8"
PAPER = "white"
WSPU_COLOR = "#178A72"
OLMIX_COLOR = "#CC79A7"
RANDOM_COLOR = "#6C6F7D"
RANDOM_LABELS = {"regret_at_1": "random pick", "top5_regret": "best of 5 random picks"}
MODEL_LABELS = {"wspu": "WSPU", "olmix": "Olmix"}
MODEL_COLORS = {"wspu": WSPU_COLOR, "olmix": OLMIX_COLOR}
TARGET_LABELS = {"uncheatable": "Uncheatable", "table9": "OlmoBaseEval Easy"}
PLOT_STYLE = {
    "font.family": "DejaVu Sans",
    "font.size": 8,
    "text.usetex": False,
    "axes.grid": False,
    "lines.markeredgewidth": 1.0,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "figure.facecolor": PAPER,
    "axes.facecolor": PAPER,
    "savefig.facecolor": PAPER,
}
DPI = 300
X_TICKS = {2: (20, 80, 140, 200, 260), 3: (20, 80, 140, 200, 260), 4: (20, 100, 180, 260)}
DRIVE_STEMS = {
    "learning_curve_paper": "r6_learning_curve",
    "learning_curve_strip": "r6_learning_curve",
    "learning_curve": "a_learning_curve_full",
    "learning_curve_appendix": "a_learning_curve_diagnostics",
}
LOG_X_TICKS = (20, 40, 80, 160, 280)
COMPLEMENT_MAX_K = 250
# Band columns of summary.csv per band type: mean with a 95% t interval, mean with the draw-level bootstrap
# percentile interval, or median with the interquartile range.
BAND_COLUMNS = {
    "t": ("mean", "t_low", "t_high"),
    "boot": ("mean", "boot_low", "boot_high"),
    "iqr": ("median", "q25", "q75"),
}
# (evaluation, stratum, metric, panel title, y label, draw the random-ranking reference)
MAIN_PANELS = (
    ("oof", "pooled", "spearman", "out-of-fold rank", r"Spearman $\rho$", False),
    ("heldout", "pooled", "spearman", "held-out bank rank", r"Spearman $\rho$", False),
    ("heldout", "pooled", "regret_at_1", "held-out selection", "Regret@1 (BPB)", True),
)
# The paper layout: the two columns that carry the claim, with the data-efficiency crossing marked.
PAPER_PANELS = (
    ("oof", "pooled", "spearman", "out-of-fold rank", r"Spearman $\rho$", False),
    ("heldout", "pooled", "regret_at_1", "held-out selection", "Regret@1 (BPB)", True),
)
CROSSING_PANEL = ("oof", "pooled", "spearman")
APPENDIX_PANELS = (
    ("complement", "pooled", "spearman", "unseen swarm runs", r"Spearman $\rho$", False),
    ("oof", "pooled", "rmse", "out-of-fold error", "RMSE (BPB)", False),
    ("oof", "fold_mean", "regret_at_1", "per-fold selection", "Mean fold regret@1 (BPB)", False),
    ("heldout", "pooled", "top5_regret", "held-out top-5", "Top-5 regret (BPB)", True),
)


def series(summary: pd.DataFrame, target: str, model: str, evaluation: str, stratum: str, metric: str) -> pd.DataFrame:
    frame = summary[
        summary["target"].eq(target)
        & summary["model"].eq(model)
        & summary["evaluation"].eq(evaluation)
        & summary["stratum"].eq(stratum)
        & summary["fold"].eq(-1)
        & summary["metric"].eq(metric)
    ]
    if evaluation == "complement":
        # Above this size the complement holds fewer than 30 runs and its metrics are dominated by noise.
        frame = frame[frame["k"] <= COMPLEMENT_MAX_K]
    return frame.sort_values("k")


def random_reference(metrics: pd.DataFrame, target: str, stratum: str, metric: str) -> float:
    """The random-ranking expectation of a held-out regret metric (constant across draws)."""
    column = {"regret_at_1": "random_regret_at_1", "top5_regret": "random_best_of_5_regret"}[metric]
    rows = metrics[metrics["target"].eq(target) & metrics["evaluation"].eq("heldout") & metrics["stratum"].eq(stratum)]
    return float(rows[column].mean()) if column in rows and len(rows) else float("nan")


def draw_panel(
    axis: plt.Axes,
    summary: pd.DataFrame,
    metrics: pd.DataFrame,
    target: str,
    panel: tuple[str, str, str, str, str, bool],
    letter: str,
    columns: int,
    log_x: bool,
    band: str,
    efficiency: pd.DataFrame | None,
    fill: bool = False,
    row_index: int = 0,
) -> list[dict[str, object]]:
    evaluation, stratum, metric, title, y_label, with_random = panel
    center_column, low_column, high_column = BAND_COLUMNS[band]
    points: list[dict[str, object]] = []
    for model in fits.MODEL_KEYS:
        frame = series(summary, target, model, evaluation, stratum, metric)
        if frame.empty:
            continue
        k = frame["k"].to_numpy(float)
        mean = frame[center_column].to_numpy(float)
        low = frame[low_column].to_numpy(float)
        high = frame[high_column].to_numpy(float)
        band = np.isfinite(low) & np.isfinite(high)
        if band.any():
            axis.fill_between(
                k[band], low[band], high[band], color=MODEL_COLORS[model], alpha=0.16, linewidth=0, zorder=2
            )
        axis.plot(
            k,
            mean,
            color=MODEL_COLORS[model],
            linewidth=1.3,
            marker="o",
            markersize=2.6,
            markerfacecolor=PAPER,
            markeredgecolor=MODEL_COLORS[model],
            zorder=4,
            label=MODEL_LABELS[model],
        )
        for row in frame.itertuples():
            points.append(
                {
                    "target": target,
                    "model": model,
                    "evaluation": evaluation,
                    "stratum": stratum,
                    "metric": metric,
                    "k": int(row.k),
                    "n_draws": int(row.n_draws),
                    "center": getattr(row, center_column),
                    "low": getattr(row, low_column),
                    "high": getattr(row, high_column),
                    "band": band,
                }
            )
    if efficiency is not None and (evaluation, stratum, metric) == CROSSING_PANEL:
        draw_crossing(axis, summary, efficiency, target, evaluation, stratum, metric, center_column)
    if with_random:
        reference = random_reference(metrics, target, stratum, metric)
        if np.isfinite(reference):
            axis.axhline(reference, color=RANDOM_COLOR, linestyle=(0, (4, 2)), linewidth=0.9, zorder=1)
            axis.annotate(
                RANDOM_LABELS[metric],
                xy=(0.98, reference),
                xycoords=("axes fraction", "data"),
                xytext=(0, 2),
                textcoords="offset points",
                ha="right",
                va="bottom",
                fontsize=6.5,
                color=RANDOM_COLOR,
            )
    if fill:
        # Half-column layout: the letter sits inside the axes (top right where the regret bands crowd the top
        # left), and each column carries one header, the panel title over the y quantity, on the top row only.
        axis.text(
            0.96 if with_random else 0.04,
            0.96,
            letter,
            transform=axis.transAxes,
            ha="right" if with_random else "left",
            va="top",
            fontsize=8,
            fontweight="bold",
            color=INK,
        )
        if row_index == 0:
            axis.set_title(
                f"{title[0].upper()}{title[1:]}", loc="center", fontsize=7.5, fontweight="bold", color=INK, pad=12
            )
            axis.annotate(
                y_label,
                xy=(0.5, 1.0),
                xycoords="axes fraction",
                xytext=(0, 2),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=7,
                color=INK,
            )
    else:
        axis.set_title(f"{letter}. {title[0].upper()}{title[1:]}", loc="left", fontsize=8, fontweight="bold", color=INK)
        axis.set_ylabel(y_label, color=INK, fontsize=7.5)
    if log_x:
        axis.set_xscale("log")
        axis.set_xlim(17, 320)
        axis.set_xticks(LOG_X_TICKS)
        # At half-column width the 160 and 280 labels touch, so 160 keeps its tick and loses its label.
        axis.set_xticklabels(["" if fill and tick == 160 else str(tick) for tick in LOG_X_TICKS])
        axis.minorticks_off()
    else:
        axis.set_xlim(10, 290)
        axis.set_xticks(X_TICKS[columns])
    axis.grid(True, axis="y", color=GRID, alpha=0.72, linewidth=0.6, zorder=0)
    axis.tick_params(colors=INK, labelsize=6.5 if fill else 7)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_color(INK)
    return points


def draw_crossing(
    axis: plt.Axes,
    summary: pd.DataFrame,
    efficiency: pd.DataFrame,
    target: str,
    evaluation: str,
    stratum: str,
    metric: str,
    center_column: str,
) -> None:
    """Dotted line at OLMix's full-panel value; the crossing k is reported in the caption, not drawn."""
    del summary, center_column
    row = efficiency[
        efficiency["target"].eq(target)
        & efficiency["evaluation"].eq(evaluation)
        & efficiency["stratum"].eq(stratum)
        & efficiency["metric"].eq(metric)
        & efficiency["model"].eq("wspu")
    ]
    if row.empty:
        return
    axis.axhline(
        float(row["reference_value"].iloc[0]), color=OLMIX_COLOR, linestyle=(0, (1.5, 2.5)), linewidth=0.9, zorder=1
    )


def build_figure(
    summary: pd.DataFrame,
    metrics: pd.DataFrame,
    panels: tuple[tuple[str, str, str, str, str, bool], ...],
    width: float,
    log_x: bool,
    band: str,
    efficiency: pd.DataFrame | None = None,
    row_height: float = 1.85,
    fill: bool = False,
) -> tuple[plt.Figure, pd.DataFrame]:
    """Draw the rows-by-panels grid.

    With ``fill`` the figure is laid out to its nominal width for a half-column placement: constrained layout
    with tight pads, one header per column instead of per-panel titles and y labels, and the panel letters
    inside the axes, so the axes take the width that decorations would otherwise consume.
    """
    rows = len(fits.TARGETS)
    columns = len(panels)
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(width, row_height * rows + 0.35),
        squeeze=False,
        layout="constrained" if fill else None,
    )
    if fill:
        figure.get_layout_engine().set(w_pad=0.02, h_pad=0.02, wspace=0.06, hspace=0.04)
    points: list[dict[str, object]] = []
    letters = iter("ABCDEFGHIJKLMNOP")
    for row, target in enumerate(fits.TARGETS):
        for column, panel in enumerate(panels):
            axis = axes[row][column]
            points.extend(
                draw_panel(
                    axis, summary, metrics, target, panel, next(letters), columns, log_x, band, efficiency, fill, row
                )
            )
            if row == rows - 1:
                axis.set_xlabel("Fitted runs k", color=INK, fontsize=7.5)
            elif fill:
                axis.tick_params(labelbottom=False)
        axes[row][0].annotate(
            TARGET_LABELS[target],
            xy=(0, 0.5),
            xycoords="axes fraction",
            xytext=(-30 if fill else -46, 0),
            textcoords="offset points",
            rotation=90,
            ha="center",
            va="center",
            fontsize=8.5,
            fontweight="bold",
            color=INK,
        )
    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        axes[0][0].legend(handles, labels, loc="lower right", frameon=False, fontsize=7, handlelength=1.6)
    if not fill:
        figure.tight_layout(w_pad=1.2, h_pad=1.0)
    return figure, pd.DataFrame(points)


def build_strip_figure(
    summary: pd.DataFrame,
    metrics: pd.DataFrame,
    panels: tuple[tuple[str, str, str, str, str, bool], ...],
    width: float,
    height: float,
    log_x: bool,
    band: str,
    efficiency: pd.DataFrame | None,
) -> tuple[plt.Figure, pd.DataFrame]:
    """One row for a full-width placement: each target's panels side by side under a header naming the target.

    Panels use the compact decoration of the half-column layout (letter inside, metric and unit as a two-line
    header) and share one x label, so the axes take most of the width.
    """
    targets = fits.TARGETS
    count = len(targets) * len(panels)
    figure, axes = plt.subplots(1, count, figsize=(width, height), squeeze=False, layout="constrained")
    figure.get_layout_engine().set(w_pad=0.02, h_pad=0.02, wspace=0.04, rect=(0, 0, 1, 0.89))
    points: list[dict[str, object]] = []
    letters = iter("ABCDEFGHIJKLMNOP")
    for target_index, target in enumerate(targets):
        for column, panel in enumerate(panels):
            axis = axes[0][target_index * len(panels) + column]
            points.extend(
                draw_panel(
                    axis, summary, metrics, target, panel, next(letters), len(panels), log_x, band, efficiency, True, 0
                )
            )
    figure.supxlabel("Fitted runs k", color=INK, fontsize=7.5)
    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        axes[0][0].legend(handles, labels, loc="lower right", frameon=False, fontsize=7, handlelength=1.6)
    # Headers are centred over each target's pair once the layout has fixed the axes positions; they stay
    # out of the layout so adding them does not move the axes.
    figure.canvas.draw()
    for target_index, target in enumerate(targets):
        first = axes[0][target_index * len(panels)].get_position()
        last = axes[0][target_index * len(panels) + len(panels) - 1].get_position()
        figure.text(
            (first.x0 + last.x1) / 2,
            0.985,
            TARGET_LABELS[target],
            ha="center",
            va="top",
            fontsize=8.5,
            fontweight="bold",
            color=INK,
            in_layout=False,
        )
    return figure, pd.DataFrame(points)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, default=fits.OUTPUT_DIR)
    parser.add_argument("--drive-dir", type=Path, default=None, help="copy the figures here as r6_learning_curve*")
    parser.add_argument("--log-x", action="store_true", help="logarithmic k axis; outputs get a _logx suffix")
    parser.add_argument("--band", choices=tuple(BAND_COLUMNS), default="t", help="band type; non-t bands add a suffix")
    parser.add_argument("--paper-width", type=float, default=4.6, help="figure width in inches for the paper layout")
    parser.add_argument("--row-height", type=float, default=1.85, help="panel row height in inches")
    parser.add_argument("--strip-height", type=float, default=1.75, help="figure height in inches for the strip layout")
    parser.add_argument(
        "--layout",
        choices=("full", "paper", "strip"),
        default="full",
        help="full: the 2x3 figure and the 2x4 appendix; paper: the 2x2 half-column figure; strip: the same "
        "four panels in one full-width row",
    )
    args = parser.parse_args()
    summary = pd.read_csv(args.output_dir / metrics_module.SUMMARY)
    metrics = pd.read_csv(args.output_dir / metrics_module.METRICS_LONG)
    plt.rcParams.update(PLOT_STYLE)
    suffix = ("_logx" if args.log_x else "") + ("" if args.band == "t" else f"_{args.band}")
    efficiency = pd.read_csv(args.output_dir / metrics_module.EFFICIENCY)
    if args.layout == "strip":
        figure, points = build_strip_figure(
            summary, metrics, PAPER_PANELS, args.paper_width, args.strip_height, args.log_x, args.band, efficiency
        )
        name = f"learning_curve_strip{suffix}"
        for extension in ("png", "pdf"):
            figure.savefig(args.output_dir / f"{name}.{extension}", dpi=DPI)
        points.to_csv(args.output_dir / f"{name}_points.csv", index=False)
        plt.close(figure)
        if args.drive_dir is not None:
            stem = DRIVE_STEMS[name.replace(suffix, "")] + suffix
            for extension in ("png", "pdf"):
                shutil.copyfile(args.output_dir / f"{name}.{extension}", args.drive_dir / f"{stem}.{extension}")
        print(f"wrote figures to {args.output_dir}")
        return
    if args.layout == "paper":
        layouts = ((f"learning_curve_paper{suffix}", PAPER_PANELS, args.paper_width, efficiency),)
    else:
        layouts = (
            (f"learning_curve{suffix}", MAIN_PANELS, 7.0, None),
            (f"learning_curve_appendix{suffix}", APPENDIX_PANELS, 7.2, None),
        )
    for name, panels, width, crossing in layouts:
        fill = args.layout == "paper"
        figure, points = build_figure(
            summary, metrics, panels, width, args.log_x, args.band, crossing, row_height=args.row_height, fill=fill
        )
        for extension in ("png", "pdf"):
            figure.savefig(
                args.output_dir / f"{name}.{extension}", dpi=DPI, bbox_inches="tight", pad_inches=0.02 if fill else 0.1
            )
        points.to_csv(args.output_dir / f"{name}_points.csv", index=False)
        plt.close(figure)
        if args.drive_dir is not None:
            stem = DRIVE_STEMS[name.replace(suffix, "")] + suffix
            for extension in ("png", "pdf"):
                shutil.copyfile(args.output_dir / f"{name}.{extension}", args.drive_dir / f"{stem}.{extension}")
    print(f"wrote figures to {args.output_dir}")


if __name__ == "__main__":
    main()
