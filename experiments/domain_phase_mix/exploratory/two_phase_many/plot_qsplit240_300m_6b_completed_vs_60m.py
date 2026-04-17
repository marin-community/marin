# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib", "numpy", "pandas"]
# ///
"""Plot completed 300M/6B vs 60M rank and BPB relationships."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize

plt.rcParams["text.usetex"] = False

ROOT = Path(__file__).resolve().parent
INPUT_CSV = ROOT / "qsplit240_300m_6b_completed_vs_60m.csv"
INPUT_SUMMARY_JSON = ROOT / "qsplit240_300m_6b_completed_vs_60m_summary.json"
OUTPUT_PNG = ROOT / "qsplit240_300m_6b_completed_vs_60m_rank_shift.png"
OUTPUT_BPB_PNG = ROOT / "qsplit240_300m_6b_completed_vs_60m_bpb_correlation.png"
OUTPUT_SELECTED_CSV = ROOT / "qsplit240_300m_6b_completed_vs_60m_rank_shift_selected.csv"

TOP_MIXTURE_COUNT = 8
BASELINE_LABELS = {
    "baseline_stratified": "Uniform",
    "baseline_unimax": "UniMax",
    "baseline_proportional": "Proportional",
}
PENDING_REFERENCE_ROWS = (
    {
        "run_name": "baseline_olmix_loglinear_uncheatable_bpb",
        "label": "Olmix",
        "bpb_60m": 1.068716,
    },
    {
        "run_name": "baseline_genericfamily_power_family_penalty_raw_optimum",
        "label": "GRP (Power-Family Penalty)",
        "bpb_60m": 1.036191,
    },
)
ANNOTATION_OFFSET_POINTS = (4, -4)
POSITIVE_COLOR = "#1a9850"
NEGATIVE_COLOR = "#d73027"
NEUTRAL_COLOR = "#4d4d4d"


def _display_label(run_name: str) -> str:
    if run_name in BASELINE_LABELS:
        return BASELINE_LABELS[run_name]
    if run_name.startswith("run_"):
        return run_name.replace("run_", "mix_", 1)
    return run_name


def _selected_rows(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    baselines = frame[frame["run_name"].isin(BASELINE_LABELS)].copy()
    mixtures = frame[frame["run_name"].str.startswith("run_")].copy()
    mixtures = mixtures.sort_values("rank_60m_within_completed").head(TOP_MIXTURE_COUNT).copy()
    selected = pd.concat([baselines, mixtures], ignore_index=True)
    selected["label"] = selected["run_name"].map(_display_label)
    selected["group"] = np.where(selected["run_name"].isin(BASELINE_LABELS), "Baseline", "Best 60M mixture")
    selected = selected.sort_values(["group", "rank_60m_within_completed"]).reset_index(drop=True)

    pending = pd.DataFrame(PENDING_REFERENCE_ROWS)
    pending["rank_60m_if_included"] = pending["bpb_60m"].apply(lambda value: int((frame["bpb_60m"] < value).sum() + 1))
    return selected, pending


def _bar_colors(rank_shift: pd.Series) -> list[str]:
    colors: list[str] = []
    for value in rank_shift:
        if value > 0:
            colors.append(POSITIVE_COLOR)
        elif value < 0:
            colors.append(NEGATIVE_COLOR)
        else:
            colors.append(NEUTRAL_COLOR)
    return colors


def _bpb_colors(bpb_values: pd.Series, *, vmin: float, vmax: float) -> list[tuple[float, float, float, float]]:
    cmap = plt.get_cmap("RdYlGn_r")
    norm = Normalize(vmin=vmin, vmax=vmax)
    return [cmap(norm(float(value))) for value in bpb_values]


def _plot_scatter(ax: plt.Axes, frame: pd.DataFrame, selected: pd.DataFrame) -> None:
    cmap = plt.get_cmap("RdYlGn_r")
    norm = Normalize(vmin=frame["bpb_300m_6b"].min(), vmax=frame["bpb_300m_6b"].max())
    scatter = ax.scatter(
        frame["rank_60m_within_completed"],
        frame["rank_300m_6b"],
        c=frame["bpb_300m_6b"],
        cmap=cmap,
        norm=norm,
        s=44,
        alpha=0.9,
        edgecolors="none",
    )
    max_rank = int(frame[["rank_60m_within_completed", "rank_300m_6b"]].to_numpy().max())
    ax.plot([1, max_rank], [1, max_rank], linestyle="--", color="0.45", linewidth=1.2)
    ax.set_xlim(0, max_rank + 3)
    ax.set_ylim(0, max_rank + 3)
    ax.set_xlabel("60M rank within completed set")
    ax.set_ylabel("300M/6B rank within completed set")
    ax.set_title("Completed-mixture rank comparison")
    ax.grid(alpha=0.18, linewidth=0.6)

    annotated = selected.sort_values(["rank_300m_6b", "rank_60m_within_completed"])
    for _, row in annotated.iterrows():
        ax.annotate(
            row["label"],
            (row["rank_60m_within_completed"], row["rank_300m_6b"]),
            xytext=ANNOTATION_OFFSET_POINTS,
            textcoords="offset points",
            fontsize=8,
            alpha=0.92,
        )

    cbar = plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.02)
    cbar.set_label("300M/6B BPB")


def _plot_selected_bars(ax: plt.Axes, selected: pd.DataFrame, pending: pd.DataFrame) -> None:
    movers = selected.sort_values("rank_shift").reset_index(drop=True)
    ax.barh(movers["label"], movers["rank_shift"], color=_bar_colors(movers["rank_shift"]), alpha=0.9)
    ax.axvline(0, color="0.4", linewidth=1.0)
    ax.set_xlabel("Rank shift = 60M rank - 300M/6B rank")
    ax.set_title("Baselines + best 60M mixtures")
    ax.grid(axis="x", alpha=0.18, linewidth=0.6)

    xmax = max(abs(float(movers["rank_shift"].min())), abs(float(movers["rank_shift"].max())))
    ax.set_xlim(-xmax - 6, xmax + 14)

    for _, row in movers.iterrows():
        x = float(row["rank_shift"])
        alignment = "left" if x >= 0 else "right"
        x_text = x + 1.0 if x >= 0 else x - 1.0
        ax.text(
            x_text,
            row["label"],
            f"{int(row['rank_300m_6b'])}",
            va="center",
            ha=alignment,
            fontsize=8,
            color="0.25",
        )

    pending_note = ", ".join(
        f"{row['label']} (60M rank {int(row['rank_60m_if_included'])})" for _, row in pending.iterrows()
    )
    ax.text(
        0.03,
        0.03,
        f"Pending 300M reruns: {pending_note}",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.5,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.9, "edgecolor": "0.8"},
    )


def _plot_bpb_scatter(ax: plt.Axes, frame: pd.DataFrame, selected: pd.DataFrame) -> None:
    cmap = plt.get_cmap("RdYlGn_r")
    norm = Normalize(vmin=frame["bpb_300m_6b"].min(), vmax=frame["bpb_300m_6b"].max())
    scatter = ax.scatter(
        frame["bpb_60m"],
        frame["bpb_300m_6b"],
        c=frame["bpb_300m_6b"],
        cmap=cmap,
        norm=norm,
        s=44,
        alpha=0.9,
        edgecolors="none",
    )
    coeffs = np.polyfit(frame["bpb_60m"], frame["bpb_300m_6b"], deg=1)
    x_line = np.linspace(float(frame["bpb_60m"].min()), float(frame["bpb_60m"].max()), 200)
    y_line = coeffs[0] * x_line + coeffs[1]
    ax.plot(x_line, y_line, linestyle="--", color="0.35", linewidth=1.2, label="least-squares fit")
    ax.set_xlabel("60M uncheatable BPB")
    ax.set_ylabel("300M/6B uncheatable BPB")
    ax.set_title("Completed-mixture BPB correlation")
    ax.grid(alpha=0.18, linewidth=0.6)

    annotated = selected.sort_values(["bpb_300m_6b", "bpb_60m"])
    for _, row in annotated.iterrows():
        ax.annotate(
            row["label"],
            (row["bpb_60m"], row["bpb_300m_6b"]),
            xytext=ANNOTATION_OFFSET_POINTS,
            textcoords="offset points",
            fontsize=8,
            alpha=0.92,
        )

    cbar = plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.02)
    cbar.set_label("300M/6B BPB")
    ax.legend(loc="upper left", frameon=False, fontsize=8.5)


def _plot_bpb_deltas(ax: plt.Axes, selected: pd.DataFrame, pending: pd.DataFrame, frame: pd.DataFrame) -> None:
    movers = selected.sort_values("bpb_delta").reset_index(drop=True)
    ax.barh(
        movers["label"],
        movers["bpb_delta"],
        color=_bpb_colors(
            movers["bpb_300m_6b"],
            vmin=float(frame["bpb_300m_6b"].min()),
            vmax=float(frame["bpb_300m_6b"].max()),
        ),
        alpha=0.9,
    )
    ax.axvline(0, color="0.4", linewidth=1.0)
    ax.set_xlabel("BPB gain = 60M BPB - 300M/6B BPB")
    ax.set_title("Baselines + best 60M mixtures")
    ax.grid(axis="x", alpha=0.18, linewidth=0.6)

    xmax = float(movers["bpb_delta"].max())
    ax.set_xlim(-0.01, xmax + 0.03)

    for _, row in movers.iterrows():
        ax.text(
            float(row["bpb_delta"]) + 0.002,
            row["label"],
            f"{float(row['bpb_300m_6b']):.3f}",
            va="center",
            ha="left",
            fontsize=8,
            color="0.25",
        )

    pending_note = ", ".join(f"{row['label']} (60M BPB {float(row['bpb_60m']):.3f})" for _, row in pending.iterrows())
    ax.text(
        0.03,
        0.03,
        f"Pending 300M reruns: {pending_note}",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.5,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.9, "edgecolor": "0.8"},
    )


def main() -> None:
    frame = pd.read_csv(INPUT_CSV)
    summary = json.loads(INPUT_SUMMARY_JSON.read_text())
    selected, pending = _selected_rows(frame)
    selected["bpb_delta"] = selected["bpb_60m"] - selected["bpb_300m_6b"]

    fig, (ax_scatter, ax_movers) = plt.subplots(
        1,
        2,
        figsize=(13.5, 6.8),
        gridspec_kw={"width_ratios": [1.5, 1.0]},
        constrained_layout=True,
    )

    _plot_scatter(ax_scatter, frame, selected)
    _plot_selected_bars(ax_movers, selected, pending)

    fig.suptitle(
        (
            "QSplit240 completed 300M/6B mixtures vs 60M swarm\n"
            f"n={summary['n_completed']} shared completed runs, "
            f"Spearman={summary['spearman_rho']:.3f}, Kendall={summary['kendall_tau']:.3f}"
        ),
        fontsize=13,
    )
    fig.savefig(OUTPUT_PNG, dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig_bpb, (ax_bpb_scatter, ax_bpb_movers) = plt.subplots(
        1,
        2,
        figsize=(13.5, 6.8),
        gridspec_kw={"width_ratios": [1.5, 1.0]},
        constrained_layout=True,
    )

    _plot_bpb_scatter(ax_bpb_scatter, frame, selected)
    _plot_bpb_deltas(ax_bpb_movers, selected, pending, frame)

    fig_bpb.suptitle(
        (
            "QSplit240 completed 300M/6B mixtures vs 60M swarm\n"
            f"n={summary['n_completed']} shared completed runs, "
            f"Pearson={summary['pearson_r']:.3f}, Spearman={summary['spearman_rho']:.3f}, "
            f"Kendall={summary['kendall_tau']:.3f}"
        ),
        fontsize=13,
    )
    fig_bpb.savefig(OUTPUT_BPB_PNG, dpi=220, bbox_inches="tight")
    plt.close(fig_bpb)

    pending_out = pending.assign(
        group="Pending 300M rerun",
        rank_300m_6b=np.nan,
        rank_shift=np.nan,
        bpb_300m_6b=np.nan,
        bpb_delta=np.nan,
    )
    selected_out = selected[
        [
            "run_name",
            "label",
            "group",
            "bpb_60m",
            "bpb_300m_6b",
            "bpb_delta",
            "rank_60m_within_completed",
            "rank_300m_6b",
            "rank_shift",
        ]
    ].copy()
    pending_out = pending_out[
        [
            "run_name",
            "label",
            "group",
            "bpb_60m",
            "bpb_300m_6b",
            "bpb_delta",
            "rank_60m_if_included",
            "rank_300m_6b",
            "rank_shift",
        ]
    ].rename(columns={"rank_60m_if_included": "rank_60m_within_completed"})
    pd.concat([selected_out, pending_out], ignore_index=True).to_csv(OUTPUT_SELECTED_CSV, index=False)
    print(f"Wrote {OUTPUT_PNG}")
    print(f"Wrote {OUTPUT_BPB_PNG}")
    print(f"Wrote {OUTPUT_SELECTED_CSV}")


if __name__ == "__main__":
    main()
