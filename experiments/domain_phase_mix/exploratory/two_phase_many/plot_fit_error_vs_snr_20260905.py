# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.12"
# dependencies = ["matplotlib", "numpy", "pandas"]
# ///

"""Surrogate fit error against evaluation signal-to-noise, per Table-9 component and Uncheatable component.

Panel A: out-of-fold RMSE as a fraction of the panel spread, with the floor 1/SNR that run noise alone imposes
(the points reach it only below SNR 3; within the Table-9 suite the ratio is flat in SNR, and the slope across the
whole panel is the Uncheatable components being both nearly noise-free and better modeled).
Panel B: the same RMSE in proportional-repeat-SD units, which is panel A times SNR; the band below 2 marks the
components whose error is within a factor two of the run-noise floor. The `r2` variant replaces panel B with
out-of-fold predictive R^2 (1 - MSE/Var) against the ceiling 1 - 1/SNR^2.
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

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

INPUT_DIR = SCRIPT_DIR / "reference_outputs" / "table9_reliability_mariner_20260908"
INK = "#111111"
GRID = "#b8b8b8"
PAPER = "white"
MARINER_COLOR = "#469C76"
OLMIX_COLOR = "#CC79A7"
BAND_COLOR = "#6C6F7D"
REGMIX_COLOR = "#0072B2"
MODEL_CHOICES = {
    "mariner": ("MARINER", MARINER_COLOR),
    "olmix": ("Olmix", OLMIX_COLOR),
    "regmix": ("RegMix", REGMIX_COLOR),
}
DEFAULT_MODELS = ("mariner", "olmix")
# (key, label, colour) of the models drawn; main() replaces it when --models is given.
MODELS = tuple((key, *MODEL_CHOICES[key]) for key in DEFAULT_MODELS)
NOISE_LIMIT = 2.0
LOW_SNR = 2.0
SUBTASK_LABELS = {
    "Simple Pattern Recognition": "Basic Skills: pattern recognition",
    "String Manipulation": "Basic Skills: string manipulation",
}
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


def model_columns(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    return frame[[f"{model}_{column}" for model, _label, _color in MODELS]]


def load(input_dir: Path = INPUT_DIR) -> pd.DataFrame:
    # The 23 Table-9 tasks (subtasks collapsed as in the SNR table) and the 7 Uncheatable components.
    table9 = pd.read_csv(input_dir / "snr_fit_tasks_delphi.csv")
    table9 = table9[table9["group"].notna() & (table9["group"] != "")].assign(evaluation="OlmoBaseEval Easy")
    uncheatable = pd.read_csv(input_dir / "snr_fit_uncheatable_delphi.csv")
    uncheatable = uncheatable[uncheatable["component"].notna() & (uncheatable["component"] != "")].assign(
        evaluation="Uncheatable"
    )
    # The two Table-9 components with SNR below 2 (both Basic Skills subtasks), shown uncollapsed.
    components = pd.read_csv(input_dir / "snr_fit_components_delphi.csv")
    components = components[components["component"].notna() & (components["component"] != "")]
    floor = components[components["snr"] < LOW_SNR].assign(evaluation="OlmoBaseEval Easy", uncollapsed=True)
    frame = pd.concat([table9, uncheatable, floor], ignore_index=True)
    for model, _label, _color in MODELS:
        frame[f"{model}_rmse_over_panel_sd"] = frame[f"{model}_rmse"] / frame["panel_sd"]
        # Predictive R^2 (one minus MSE over the swarm variance), the quantity the 1 - 1/SNR^2 ceiling bounds.
        frame[f"{model}_r2"] = 1.0 - frame[f"{model}_mse"] / frame["panel_sd"] ** 2
    return frame


def style(axis: plt.Axes) -> None:
    axis.set_xscale("log")
    axis.set_xticks([1, 2, 5, 10, 20, 40])
    axis.set_xticklabels(["1", "2", "5", "10", "20", "40"])
    axis.minorticks_off()
    axis.grid(True, axis="y", color=GRID, alpha=0.72, linewidth=0.6, zorder=0)
    axis.tick_params(colors=INK, labelsize=7)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_color(INK)
    axis.set_xlabel("Evaluation SNR (swarm SD / proportional SD)", color=INK, fontsize=7.5)


def scatter(axis: plt.Axes, frame: pd.DataFrame, column: str, label_points: bool) -> None:
    # One thin segment per task spans the surrogates, so the paired difference is readable.
    axis.vlines(
        frame["snr"],
        model_columns(frame, column).min(axis=1),
        model_columns(frame, column).max(axis=1),
        color=BAND_COLOR,
        alpha=0.45,
        linewidth=0.8,
        zorder=3,
    )
    for model, label, color in MODELS:
        for evaluation, marker, size in (("OlmoBaseEval Easy", "o", 22), ("Uncheatable", "D", 24)):
            part = frame[frame["evaluation"].eq(evaluation)]
            axis.scatter(
                part["snr"],
                part[f"{model}_{column}"],
                s=size,
                marker=marker,
                facecolor=color,
                edgecolor=PAPER,
                linewidth=0.5,
                alpha=0.95,
                zorder=4,
                label=f"{label}, {evaluation}",
            )
    # The uncollapsed sub-2 SNR subtasks are named individually, staggered above and below their points.
    if label_points:
        floor = frame[frame["uncollapsed"].fillna(False)].sort_values("snr")
        for index, (_, row) in enumerate(floor.iterrows()):
            below = column != "r2"
            axis.annotate(
                SUBTASK_LABELS.get(row["subtask"], row["subtask"]),
                xy=(row["snr"], row[f"mariner_{column}"] if below else row[f"olmix_{column}"]),
                xytext=(6, -5 - 11 * index) if below else (6, 3),
                textcoords="offset points",
                ha="left",
                va="top" if below else "bottom",
                fontsize=6,
                color=INK,
            )


def draw_r2_panel(
    axis: plt.Axes,
    frame: pd.DataFrame,
    *,
    title: str | None = "B. Explained variance against its noise ceiling",
    label_points: bool = True,
) -> None:
    """Out-of-fold R^2 against SNR with the ceiling a noise-free predictor of the run mean would reach."""
    scatter(axis, frame, "r2", label_points=label_points)
    grid = np.geomspace(1.0, 40.0, 200)
    axis.plot(grid, 1 - 1 / grid**2, color=BAND_COLOR, linestyle=(0, (4, 2)), linewidth=0.9, zorder=2)
    if label_points:
        axis.annotate(
            r"ceiling $1 - 1/\mathrm{SNR}^2$",
            xy=(2.2, 1 - 1 / 2.2**2),
            xytext=(-4, 6),
            textcoords="offset points",
            ha="right",
            va="bottom",
            fontsize=6.5,
            color=BAND_COLOR,
        )
    axis.set_ylabel(r"Out-of-fold $R^2$ ($1-\mathrm{MSE}/\mathrm{Var}$)", color=INK, fontsize=7.5)
    # The axis shows every task, including those with negative out-of-fold R^2.
    lowest = float(min(model_columns(frame, "r2").min().min(), 0.0))
    axis.set_ylim(lowest - 0.06, 1.02)
    axis.axhline(0.0, color=INK, linewidth=0.5, zorder=1)
    if title is not None:
        axis.set_title(title, loc="left", fontsize=8, fontweight="bold", color=INK)


def build_r2_only_figure(frame: pd.DataFrame) -> plt.Figure:
    """Main-text panel with separate keys for method colors and evaluation shapes."""
    figure = plt.figure(figsize=(3.2, 2.9))
    axis = figure.add_axes((0.16, 0.23, 0.82, 0.58))
    draw_r2_panel(axis, frame, title=None, label_points=False)
    style(axis)
    axis.set_xlim(0.9, 42)
    lower_tick = np.floor(model_columns(frame, "r2").min().min() * 2) / 2
    axis.set_yticks(np.arange(lower_tick, 1.01, 0.5))
    axis.set_ylim(lower_tick - 0.06, 1.20)
    axis.grid(False, axis="y")
    for tick in axis.get_yticks():
        if tick not in (0.0, 1.0):
            axis.axhline(tick, color=GRID, alpha=0.72, linewidth=0.6, zorder=0)
    axis.set_ylabel(r"Out-of-fold $R^2$", color=INK, fontsize=8)
    axis.set_xlabel("Signal-to-noise ratio (SNR)\nSwarm SD / proportional-repeat SD", color=INK, fontsize=7.5)
    axis.annotate(
        "Constant-noise ceiling\n" + r"$1-1/\mathrm{SNR}^2$",
        xy=(2.15, 1 - 1 / 2.15**2),
        xytext=(1.03, 1.15),
        ha="left",
        va="top",
        fontsize=6.6,
        color=BAND_COLOR,
        arrowprops={"arrowstyle": "-", "color": BAND_COLOR, "linewidth": 0.65},
    )
    negative = frame[model_columns(frame, "r2").min(axis=1) < 0]
    for _, row in negative.iterrows():
        subtask_label = SUBTASK_LABELS.get(row["subtask"])
        label = subtask_label.replace(": ", ":\n") if subtask_label else row["task"]
        axis.annotate(
            label,
            xy=(row["snr"], model_columns(negative.loc[[row.name]], "r2").min().min()),
            xytext=(5, -3) if subtask_label else (5, 2),
            textcoords="offset points",
            ha="left",
            va="top" if subtask_label else "bottom",
            fontsize=6.3,
            color=INK,
        )
    method_handles = [
        mpl.lines.Line2D([], [], color=color, marker="s", linestyle="", markersize=4) for _model, _label, color in MODELS
    ]
    figure.legend(
        method_handles,
        [label for _model, label, _color in MODELS],
        loc="upper center",
        bbox_to_anchor=(0.57, 1.0),
        ncol=len(MODELS),
        fontsize=7.5,
        frameon=False,
        handletextpad=0.3,
        columnspacing=1.6,
    )
    group_handles = [
        mpl.lines.Line2D([], [], color=BAND_COLOR, marker=marker, linestyle="", markersize=4) for marker in ("o", "D")
    ]
    figure.legend(
        group_handles,
        ["OlmoBaseEval Easy", "Uncheatable"],
        loc="upper center",
        bbox_to_anchor=(0.57, 0.925),
        ncol=2,
        fontsize=6.6,
        frameon=False,
        handletextpad=0.3,
        columnspacing=1.0,
    )
    return figure


def build_figure(frame: pd.DataFrame, variant: str) -> plt.Figure:
    figure, (left, right) = plt.subplots(1, 2, figsize=(7.0, 2.6))
    table9 = frame[frame["evaluation"].eq("OlmoBaseEval Easy") & ~frame["uncollapsed"].fillna(False)]
    scatter(left, frame, "rmse_over_panel_sd", label_points=False)
    for model, _label, color in MODELS:
        median = float(table9[f"{model}_rmse_over_panel_sd"].median())
        left.axhline(median, color=color, linestyle=(0, (4, 2)), linewidth=0.9, zorder=2)
        left.annotate(
            f"median {median:.2f}",
            xy=(1.0, median),
            xytext=(2, 2 if model == "olmix" else -2),
            textcoords="offset points",
            ha="left",
            va="bottom" if model == "olmix" else "top",
            fontsize=6.5,
            color=color,
        )
    # Run noise alone bounds the ratio below by 1/SNR: the panel-B ceiling in RMSE units.
    grid = np.geomspace(1.0, 40.0, 200)
    left.plot(grid, 1 / grid, color=BAND_COLOR, linestyle=(0, (4, 2)), linewidth=0.9, zorder=2)
    left.annotate(
        r"floor $1/\mathrm{SNR}$",
        xy=(16.0, 1 / 16.0),
        xytext=(0, 4),
        textcoords="offset points",
        ha="left",
        va="bottom",
        fontsize=6.5,
        color=BAND_COLOR,
    )
    left.set_ylabel("Out-of-fold RMSE / swarm SD", color=INK, fontsize=7.5)
    left.set_ylim(0, 1.15)
    left.set_title("A. Fit error as a fraction of the spread", loc="left", fontsize=8, fontweight="bold", color=INK)
    if variant == "r2":
        draw_r2_panel(right, frame)
        for axis in (left, right):
            style(axis)
            axis.set_xlim(0.9, 42)
        handles, labels = left.get_legend_handles_labels()
        figure.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 1.01),
            ncol=4,
            frameon=False,
            fontsize=7,
            handletextpad=0.3,
            columnspacing=1.4,
        )
        figure.tight_layout(w_pad=1.5)
        return figure
    scatter(right, frame, "rmse_over_repeat_sd", label_points=True)
    grid = np.array([1.0, 40.0])
    for model, _label, color in MODELS:
        slope = float(np.sum(table9["snr"] * table9[f"{model}_rmse_over_repeat_sd"]) / np.sum(table9["snr"] ** 2))
        right.plot(grid, slope * grid, color=color, linestyle=(0, (4, 2)), linewidth=0.9, zorder=2)
        right.annotate(
            f"{_label}: {slope:.2f}" + r"$\,\times\,$SNR",
            xy=(0.03, 0.97 if model == "olmix" else 0.89),
            xycoords="axes fraction",
            ha="left",
            va="top",
            fontsize=6.5,
            color=color,
        )
    right.axhspan(0, NOISE_LIMIT, color=BAND_COLOR, alpha=0.10, zorder=1, linewidth=0)
    right.annotate(
        "within 2 SD of the noise floor",
        xy=(40, NOISE_LIMIT),
        xytext=(-2, -3),
        textcoords="offset points",
        ha="right",
        va="top",
        fontsize=6.5,
        color=BAND_COLOR,
    )
    right.set_ylabel("Out-of-fold RMSE / proportional SD", color=INK, fontsize=7.5)
    right.set_yscale("log")
    right.set_yticks([1, 2, 5, 10, 20])
    right.set_yticklabels(["1", "2", "5", "10", "20"])
    right.set_title("B. The same error in noise units", loc="left", fontsize=8, fontweight="bold", color=INK)
    for axis in (left, right):
        style(axis)
        axis.set_xlim(0.9, 42)
    handles, labels = left.get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=4,
        frameon=False,
        fontsize=7,
        handletextpad=0.3,
        columnspacing=1.4,
    )
    figure.tight_layout(w_pad=1.5)
    return figure


def group_table(input_dir: Path = INPUT_DIR) -> pd.DataFrame:
    """Component-level medians by task group: error as a fraction of the swarm SD, R^2, and the noise share of MSE."""
    table9 = pd.read_csv(input_dir / "snr_fit_components_delphi.csv")
    table9 = table9[table9["component"].notna() & (table9["component"] != "")].copy()
    uncheatable = pd.read_csv(input_dir / "snr_fit_uncheatable_delphi.csv")
    uncheatable = uncheatable[uncheatable["component"].notna() & (uncheatable["component"] != "")].copy()
    rows = []
    groups = [(f"OlmoBaseEval Easy, {g}", table9[table9["group"].eq(g)]) for g in ("Math", "Code", "QA")]
    for name, part in [
        ("OlmoBaseEval Easy, all components", table9),
        *groups,
        ("Uncheatable, all components", uncheatable),
    ]:
        row = {"components": name, "n": len(part), "median SNR": float(part["snr"].median())}
        for model, label, _color in MODELS:
            row[f"{label} RMSE/swarm SD"] = float((part[f"{model}_rmse"] / part["panel_sd"]).median())
            row[f"{label} R2"] = float((1.0 - part[f"{model}_mse"] / part["panel_sd"] ** 2).median())
            row[f"{label} noise share"] = float((part["repeat_sd"] ** 2 / part[f"{model}_mse"]).median())
        row["noise-limited (MARINER RMSE/SD < 2)"] = int((part["mariner_rmse_over_repeat_sd"] < NOISE_LIMIT).sum())
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-dir", type=Path, default=INPUT_DIR, help="table9_reliability_fit_metrics output")
    parser.add_argument("--output-dir", type=Path, default=None, help="defaults to the input directory")
    parser.add_argument(
        "--variant",
        choices=("noise", "r2", "r2_only"),
        default="noise",
        help="panel B: noise units or R^2 vs ceiling; r2_only draws the R^2 panel alone at half width",
    )
    parser.add_argument(
        "--models", nargs="+", choices=tuple(MODEL_CHOICES), default=list(DEFAULT_MODELS), help="models to draw"
    )
    args = parser.parse_args()
    global MODELS
    MODELS = tuple((key, *MODEL_CHOICES[key]) for key in args.models)
    args.output_dir = args.output_dir or args.input_dir
    frame = load(args.input_dir)
    plt.rcParams.update(PLOT_STYLE)
    figure = build_r2_only_figure(frame) if args.variant == "r2_only" else build_figure(frame, args.variant)
    suffix = {"noise": "", "r2": "_r2", "r2_only": "_r2_only"}[args.variant]
    for extension in ("png", "pdf"):
        figure.savefig(args.output_dir / f"fit_error_vs_snr{suffix}.{extension}", dpi=DPI, bbox_inches="tight")
    table = group_table(args.input_dir)
    table.to_csv(args.output_dir / "fit_error_vs_snr_groups.csv", index=False)
    print(table.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
