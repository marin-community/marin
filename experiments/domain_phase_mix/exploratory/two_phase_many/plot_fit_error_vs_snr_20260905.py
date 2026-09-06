# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Surrogate fit error against evaluation signal-to-noise, per Table-9 component and Uncheatable component.

Panel A: out-of-fold RMSE as a fraction of the panel spread, with the floor 1/SNR that run noise alone imposes
(the points reach it only below SNR 3; within the Table-9 suite the ratio is flat in SNR, and the slope across the
whole panel is the Uncheatable components being both nearly noise-free and better modeled).
Panel B: the same RMSE in proportional-repeat-SD units, which is panel A times SNR; the band below 2 marks the
components whose error is within a factor two of the run-noise floor. The `r2` variant replaces panel B with
out-of-fold R^2 against the ceiling 1 - 1/SNR^2.
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

INPUT_DIR = SCRIPT_DIR / "reference_outputs" / "table9_reliability_20260905"
INK = "#111111"
GRID = "#b8b8b8"
PAPER = "white"
WSPU_COLOR = "#178A72"
OLMIX_COLOR = "#CC79A7"
BAND_COLOR = "#6C6F7D"
MODELS = (("wspu", "WSPU", WSPU_COLOR), ("olmix", "Olmix", OLMIX_COLOR))
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


def load() -> pd.DataFrame:
    # The 23 Table-9 tasks (subtasks collapsed as in the SNR table) and the 7 Uncheatable components.
    table9 = pd.read_csv(INPUT_DIR / "snr_fit_tasks_delphi.csv")
    table9 = table9[table9["group"].notna() & (table9["group"] != "")].assign(evaluation="OlmoBaseEval Easy")
    uncheatable = pd.read_csv(INPUT_DIR / "snr_fit_uncheatable_delphi.csv")
    uncheatable = uncheatable[uncheatable["component"].notna() & (uncheatable["component"] != "")].assign(
        evaluation="Uncheatable"
    )
    # The two Table-9 components with SNR below 2 (both Basic Skills subtasks), shown uncollapsed.
    components = pd.read_csv(INPUT_DIR / "snr_fit_components_delphi.csv")
    components = components[components["component"].notna() & (components["component"] != "")]
    floor = components[components["snr"] < LOW_SNR].assign(evaluation="OlmoBaseEval Easy", uncollapsed=True)
    frame = pd.concat([table9, uncheatable, floor], ignore_index=True)
    for model, _label, _color in MODELS:
        frame[f"{model}_rmse_over_panel_sd"] = frame[f"{model}_rmse"] / frame["panel_sd"]
        frame[f"{model}_r2"] = frame[f"{model}_pearson"] ** 2
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
    # One thin segment per task joins the two surrogates, so the paired difference is readable.
    axis.vlines(
        frame["snr"],
        frame[f"olmix_{column}"],
        frame[f"wspu_{column}"],
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
                xy=(row["snr"], row[f"wspu_{column}"] if below else row[f"olmix_{column}"]),
                xytext=(6, -5 - 11 * index) if below else (6, 3),
                textcoords="offset points",
                ha="left",
                va="top" if below else "bottom",
                fontsize=6,
                color=INK,
            )


def draw_r2_panel(axis: plt.Axes, frame: pd.DataFrame) -> None:
    """Out-of-fold R^2 against SNR with the ceiling a noise-free predictor of the run mean would reach."""
    scatter(axis, frame, "r2", label_points=True)
    grid = np.geomspace(1.0, 40.0, 200)
    axis.plot(grid, 1 - 1 / grid**2, color=BAND_COLOR, linestyle=(0, (4, 2)), linewidth=0.9, zorder=2)
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
    axis.set_ylabel(r"Out-of-fold $R^2$ (Pearson $r^2$)", color=INK, fontsize=7.5)
    axis.set_ylim(0, 1.02)
    axis.set_title(
        "B. Explained variance against its noise ceiling", loc="left", fontsize=8, fontweight="bold", color=INK
    )


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


def group_table() -> pd.DataFrame:
    """Component-level medians by task group: error as a fraction of the swarm SD, R^2, and the noise share of MSE."""
    table9 = pd.read_csv(INPUT_DIR / "snr_fit_components_delphi.csv")
    table9 = table9[table9["component"].notna() & (table9["component"] != "")].copy()
    uncheatable = pd.read_csv(INPUT_DIR / "snr_fit_uncheatable_delphi.csv")
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
            row[f"{label} R2"] = float((part[f"{model}_pearson"] ** 2).median())
            row[f"{label} noise share"] = float((1 / part[f"{model}_rmse_over_repeat_sd"] ** 2).median())
        row["noise-limited (WSPU RMSE/SD < 2)"] = int((part["wspu_rmse_over_repeat_sd"] < NOISE_LIMIT).sum())
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, default=INPUT_DIR)
    parser.add_argument(
        "--variant", choices=("noise", "r2"), default="noise", help="panel B: noise units or R^2 vs ceiling"
    )
    args = parser.parse_args()
    frame = load()
    plt.rcParams.update(PLOT_STYLE)
    figure = build_figure(frame, args.variant)
    suffix = "" if args.variant == "noise" else "_r2"
    for extension in ("png", "pdf"):
        figure.savefig(args.output_dir / f"fit_error_vs_snr{suffix}.{extension}", dpi=DPI, bbox_inches="tight")
    table = group_table()
    table.to_csv(args.output_dir / "fit_error_vs_snr_groups.csv", index=False)
    print(table.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
