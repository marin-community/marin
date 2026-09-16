# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Learning-curve figure: surrogate performance as a function of the number of fitted runs.

Reads the summaries written by ``learning_curve_metrics_20260905`` and draws, per target, the mean over
draws with its 95% t interval for out-of-fold rank correlation, retrospective-bank rank correlation and
retrospective regret at 1 (with the random-ranking expectation as reference). An appendix variant adds
complement rank correlation, out-of-fold RMSE in BPB (median and IQR), fold-mean regret and top-5 regret.
The main regret layout retains the complete positive-valued curves and intervals on logarithmic axes.
The x axis counts distinct mixtures, including the pinned anchor in the MARINER study; the ten
calibration repeats are additional runs. Source CSVs retain their original non-anchor count ``k``.
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
WSPU_COLOR = "#469C76"
OLMIX_COLOR = "#CC79A7"
RANDOM_COLOR = "#6C6F7D"
RANDOM_LABELS = {"regret_at_1": "random pick", "top5_regret": "best of 5 random picks"}
PER_BUCKET_COLOR = "#E69F00"
QUADRATIC_COLOR = "#56B4E9"
SPLINE_COLOR = "#D55E00"
REGMIX_COLOR = "#0072B2"
MODEL_LABELS = {
    "wspu": "MARINER",
    "mariner": "MARINER",
    "mariner_per_bucket": "MARINER, one shape per bucket",
    "olmix": "Olmix",
    "quadratic": "Quadratic in log-epochs, floor link",
    "spline": "Natural cubic spline in log-epochs, floor link",
    "regmix": "RegMix",
}
MODEL_COLORS = {
    "wspu": WSPU_COLOR,
    "mariner": WSPU_COLOR,
    "mariner_per_bucket": PER_BUCKET_COLOR,
    "olmix": OLMIX_COLOR,
    "quadratic": QUADRATIC_COLOR,
    "spline": SPLINE_COLOR,
    "regmix": REGMIX_COLOR,
}
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
    "learning_curve_regret": "r6_learning_curve",
    "learning_curve": "a_learning_curve_full",
    "learning_curve_appendix": "a_learning_curve_diagnostics",
}
LOG_X_TICKS = (20, 40, 80, 160, 280)
STUDY_ANCHOR_RUNS = {"legacy": 0, "mariner": 1}
COMPLEMENT_MAX_K = 250
# Nominal parameters per task at M=39: MARINER 2M+5, Olmix's log-linear law M+1. Figure 3 names them in its legend.
NOMINAL_PARAMETERS = {"mariner": 83, "olmix": 40}
REGRET_MODELS = ("mariner", "olmix", "regmix")
# Band columns of summary.csv per band type: mean with a 95% t interval, mean with the draw-level bootstrap
# percentile interval, or median with the interquartile range.
BAND_COLUMNS = {
    "t": ("mean", "t_low", "t_high"),
    "boot": ("mean", "boot_low", "boot_high"),
    "iqr": ("median", "q25", "q75"),
}
# Heavy-tailed metrics are drawn as the median over draws with the interquartile band whatever --band says: on
# subsets of 50 runs or fewer a single draw can produce an out-of-fold prediction of astronomical size (both
# surrogates extrapolate through an exponential), and a mean over draws would show nothing else.
ROBUST_METRICS = ("rmse",)
# (evaluation, stratum, metric, panel title, y label, draw the random-ranking reference)
MAIN_PANELS = (
    ("oof", "pooled", "spearman", "out-of-fold rank", r"Spearman $\rho$", False),
    ("heldout", "pooled", "spearman", "retrospective bank rank", r"Spearman $\rho$", False),
    ("heldout", "pooled", "regret_at_1", "retrospective selection", "Regret@1 (BPB)", True),
)
# The paper layout: the two columns that carry the claim, with the data-efficiency crossing marked.
PAPER_PANELS = (
    ("oof", "pooled", "spearman", "out-of-fold rank", r"Spearman $\rho$", False),
    ("heldout", "pooled", "regret_at_1", "retrospective selection", "Regret@1 (BPB)", True),
)
CROSSING_PANEL = ("oof", "pooled", "spearman")
APPENDIX_PANELS = (
    ("complement", "pooled", "spearman", "unseen swarm runs", r"Spearman $\rho$", False),
    ("oof", "pooled", "rmse", "out-of-fold error", "RMSE (BPB)", False),
    ("oof", "fold_mean", "regret_at_1", "per-fold selection", "Mean fold regret@1 (BPB)", False),
    ("heldout", "pooled", "top5_regret", "retrospective top-5", "Top-5 regret (BPB)", True),
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
    models: tuple[str, ...] | None = None,
) -> list[dict[str, object]]:
    # Resolved at call time: ``fits`` is rebound to the selected study in main(), so a default bound at import
    # would always be the legacy study's model list.
    if models is None:
        models = fits.MODEL_KEYS
    evaluation, stratum, metric, title, y_label, with_random = panel
    effective_band = "iqr" if metric in ROBUST_METRICS else band
    center_column, low_column, high_column = BAND_COLUMNS[effective_band]
    if metric in ROBUST_METRICS:
        y_label = f"{y_label}, median"
        # The spline's small-subset errors reach 1e11 BPB in a quarter of the draws; only a log axis keeps the
        # other models legible while showing that.
        axis.set_yscale("log")
    points: list[dict[str, object]] = []
    for model in models:
        frame = series(summary, target, model, evaluation, stratum, metric)
        if frame.empty:
            continue
        mixture_count = frame["mixtures"].to_numpy(float)
        mean = frame[center_column].to_numpy(float)
        low = frame[low_column].to_numpy(float)
        high = frame[high_column].to_numpy(float)
        valid_band = np.isfinite(low) & np.isfinite(high)
        if valid_band.any():
            axis.fill_between(
                mixture_count[valid_band],
                low[valid_band],
                high[valid_band],
                color=MODEL_COLORS[model],
                alpha=0.16,
                linewidth=0,
                zorder=2,
            )
        axis.plot(
            mixture_count,
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
                    "mixtures": int(row.mixtures),
                    "n_draws": int(row.n_draws),
                    "center": getattr(row, center_column),
                    "low": getattr(row, low_column),
                    "high": getattr(row, high_column),
                    "band": effective_band,
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
    anchor_runs = int(summary["mixtures"].iloc[0] - summary["k"].iloc[0])
    if log_x:
        axis.set_xscale("log")
        axis.set_xlim(17, 320)
        ticks = [*(tick + anchor_runs for tick in LOG_X_TICKS[:-1]), int(summary["mixtures"].max())]
        axis.set_xticks(ticks)
        # The penultimate and full-swarm labels touch in narrow panels.
        axis.set_xticklabels(["" if (fill or columns == 4) and tick == ticks[-2] else str(tick) for tick in ticks])
        axis.minorticks_off()
    else:
        axis.set_xlim(10, 290)
        axis.set_xticks([tick + anchor_runs for tick in X_TICKS[columns]])
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
        & efficiency["model"].eq(metrics_module.PRIMARY)
        & efficiency["reference_model"].eq(metrics_module.COMPARATOR)
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
                axis.set_xlabel("Distinct mixtures", color=INK, fontsize=7.5)
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
        # Four models no longer fit inside a panel; the legend spans the top of the figure.
        figure.legend(
            handles, labels, loc="upper center", ncol=len(handles), frameon=False, fontsize=7, handlelength=1.6
        )
    if not fill:
        figure.tight_layout(w_pad=1.2, h_pad=1.0, rect=(0, 0, 1, 0.96))
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
    figure.supxlabel("Distinct mixtures", color=INK, fontsize=7.5)
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


def build_regret_figure(
    summary: pd.DataFrame,
    metrics: pd.DataFrame,
    width: float,
    height: float,
    log_x: bool,
    band: str,
) -> tuple[plt.Figure, pd.DataFrame]:
    """Show both objectives' retrospective-bank regret with complete intervals on a log y axis."""
    missing_models = set(REGRET_MODELS) - set(summary["model"])
    if missing_models:
        raise ValueError(f"Bank-regret summaries are missing models: {sorted(missing_models)}")
    figure, axes = plt.subplots(1, 2, figsize=(width, height))
    figure.subplots_adjust(left=0.10, right=0.985, bottom=0.25, top=0.79, wspace=0.32)
    all_points: list[dict[str, object]] = []
    panel = ("heldout", "pooled", "regret_at_1", "retrospective selection", "Regret@1 (BPB)", False)
    for index, target in enumerate(fits.TARGETS):
        axis = axes[index]
        points = draw_panel(
            axis, summary, metrics, target, panel, "AB"[index], 2, log_x, band, None, models=REGRET_MODELS
        )
        assert points and all(float(point["low"]) > 0 for point in points), "Log regret requires positive bands."
        all_points.extend(points)
        axis.set_title(f"{'AB'[index]} · {TARGET_LABELS[target]}", loc="left", fontsize=8, fontweight="bold", pad=4)
        axis.set_yscale("log")
        minimum = min(float(point["low"]) for point in points)
        maximum = max(float(point["high"]) for point in points)
        axis.set_ylim(minimum / 1.35, maximum * 1.35)
        y_ticks = (0.002, 0.01, 0.05, 0.2) if target == "uncheatable" else (0.01, 0.03, 0.1, 0.3)
        axis.set_yticks([tick for tick in y_ticks if axis.get_ylim()[0] <= tick <= axis.get_ylim()[1]])
        axis.set_yticklabels([f"{tick:g}" for tick in axis.get_yticks()])
        axis.minorticks_off()
        axis.tick_params(labelsize=6.5, length=2.5, width=0.6)
        axis.set_ylabel("Regret@1 (BPB)", fontsize=7, labelpad=3)
        axis.set_xlabel("")
    handles, _labels = axes[0].get_legend_handles_labels()
    labels = [
        (
            f"{MODEL_LABELS[model]} ({NOMINAL_PARAMETERS[model]} parameters)"
            if model in NOMINAL_PARAMETERS
            else MODEL_LABELS[model]
        )
        for model in REGRET_MODELS
    ]
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.53, 1.04),
        ncol=len(handles),
        frameon=False,
        fontsize=7,
        handlelength=1.6,
        columnspacing=1.4,
    )
    figure.text(0.53, 0.045, "Distinct mixtures (including anchor)", ha="center", fontsize=7)
    return figure, pd.DataFrame(all_points)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--study", choices=tuple(metrics_module.STUDIES), default="legacy")
    parser.add_argument("--output-dir", type=Path, default=None, help="defaults to the study's output directory")
    parser.add_argument("--drive-dir", type=Path, default=None, help="copy the figures here as r6_learning_curve*")
    parser.add_argument("--log-x", action="store_true", help="logarithmic k axis; outputs get a _logx suffix")
    parser.add_argument("--band", choices=tuple(BAND_COLUMNS), default="t", help="band type; non-t bands add a suffix")
    parser.add_argument("--paper-width", type=float, default=4.6, help="figure width in inches for the paper layout")
    parser.add_argument("--row-height", type=float, default=1.85, help="panel row height in inches")
    parser.add_argument("--strip-height", type=float, default=1.75, help="figure height in inches for the strip layout")
    parser.add_argument(
        "--layout",
        choices=("full", "paper", "strip", "regret"),
        default="full",
        help="full: the 2x3 figure and the 2x4 appendix; paper: the 2x2 half-column figure; strip: the same "
        "four panels in one full-width row; regret: two full-range retrospective-bank regret panels",
    )
    args = parser.parse_args()
    if args.layout == "regret" and args.study != "mariner":
        parser.error("--layout regret uses the frozen MARINER study; pass --study mariner")
    metrics_module.select_study(args.study)
    global fits
    fits = metrics_module.fits
    args.output_dir = args.output_dir or fits.OUTPUT_DIR
    summary = pd.read_csv(args.output_dir / metrics_module.SUMMARY)
    summary["mixtures"] = summary["k"] + STUDY_ANCHOR_RUNS[args.study]
    metrics = pd.read_csv(args.output_dir / metrics_module.METRICS_LONG)
    plt.rcParams.update(PLOT_STYLE)
    suffix = ("_logx" if args.log_x else "") + ("" if args.band == "t" else f"_{args.band}")
    efficiency = pd.read_csv(args.output_dir / metrics_module.EFFICIENCY)
    if args.layout == "regret":
        figure, points = build_regret_figure(
            summary, metrics, args.paper_width, args.strip_height, args.log_x, args.band
        )
        name = f"learning_curve_regret{suffix}"
        for extension in ("png", "pdf"):
            figure.savefig(args.output_dir / f"{name}.{extension}", dpi=DPI)
        points.to_csv(args.output_dir / f"{name}_points.csv", index=False)
        plt.close(figure)
        if args.drive_dir is not None:
            stem = DRIVE_STEMS["learning_curve_regret"] + suffix
            for extension in ("png", "pdf"):
                shutil.copyfile(args.output_dir / f"{name}.{extension}", args.drive_dir / f"{stem}.{extension}")
        print(f"wrote figures to {args.output_dir}")
        return
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
