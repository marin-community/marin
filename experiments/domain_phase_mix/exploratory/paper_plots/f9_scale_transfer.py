# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib", "numpy", "pandas", "scipy"]
# ///
"""Paper plot for 60M-to-100M mixture transfer on the completed QSplit swarm."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import rankdata

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1] / "two_phase_many"
PAPER_ROOT = Path(__file__).resolve().parent
IMG_DIR = PAPER_ROOT / "img"

INPUT_CSV = ROOT / "qsplit240_300m_6b_completed_vs_60m.csv"
SUMMARY_JSON = ROOT / "qsplit240_300m_6b_completed_vs_60m_summary.json"
SELECTED_CSV = ROOT / "qsplit240_300m_6b_completed_vs_60m_rank_shift_selected.csv"
OUTPUT_STEM = IMG_DIR / "f9_scale_transfer"
OUTPUT_CSV = IMG_DIR / "f9_scale_transfer_points.csv"

TEXT_COLOR = "#232B32"
GRID_COLOR = "#E6E2DA"
POINT_EDGE_COLOR = "#8F6B38"
SELECTED_EDGE_COLOR = "#232B32"
BASELINE_COLOR = "#E24731"
EXTRA_COLOR = "#4C78A8"
CMAP_NAME = "RdYlGn_r"


def _configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "axes.labelcolor": TEXT_COLOR,
            "axes.edgecolor": "#A8A29A",
            "xtick.color": TEXT_COLOR,
            "ytick.color": TEXT_COLOR,
            "text.color": TEXT_COLOR,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _selected_points(frame: pd.DataFrame) -> pd.DataFrame:
    selected = pd.read_csv(SELECTED_CSV)
    selected = selected.copy()
    selected["point_type"] = np.select(
        [
            selected["group"].eq("Baseline"),
            selected["group"].eq("Extra 300M final"),
        ],
        ["Baseline", "Deployed mixture"],
        default="Top 60M swarm",
    )
    selected["rank_60m_for_plot"] = selected["rank_60m_within_completed"].astype(float)
    selected["rank_300m_for_plot"] = selected["rank_300m_6b"].astype(float)

    extras = selected.loc[selected["group"].eq("Extra 300M final")].copy()
    if not extras.empty:
        all_60m = pd.concat(
            [
                frame[["bpb_60m"]].assign(label="swarm"),
                extras[["bpb_60m"]].assign(label="extra"),
            ],
            ignore_index=True,
        )
        all_300m = pd.concat(
            [
                frame[["bpb_300m_6b"]].assign(label="swarm"),
                extras[["bpb_300m_6b"]].assign(label="extra"),
            ],
            ignore_index=True,
        )
        extras.loc[:, "rank_60m_for_plot"] = rankdata(all_60m["bpb_60m"], method="min")[-len(extras) :]
        extras.loc[:, "rank_300m_for_plot"] = rankdata(all_300m["bpb_300m_6b"], method="min")[-len(extras) :]
        selected.loc[extras.index, ["rank_60m_for_plot", "rank_300m_for_plot"]] = extras[
            ["rank_60m_for_plot", "rank_300m_for_plot"]
        ]
    return selected


def _annotate(ax: plt.Axes, frame: pd.DataFrame, *, x_col: str, y_col: str) -> None:
    label_frame = frame.loc[frame["point_type"].isin(["Baseline", "Deployed mixture"])].copy()
    for _, row in label_frame.iterrows():
        x_offset = 4
        y_offset = 4
        if row["point_type"] == "Deployed mixture":
            x_offset = 8
            y_offset = 12 if "GRP" in row["label"] else -12
        elif row["run_name"] == "baseline_stratified":
            x_offset = -44
            y_offset = -3
        elif row["run_name"] == "baseline_unimax":
            x_offset = 8
            y_offset = 10
        elif row["run_name"] == "baseline_proportional":
            x_offset = 8
            y_offset = -14
        ax.annotate(
            row["label"],
            (row[x_col], row[y_col]),
            xytext=(x_offset, y_offset),
            textcoords="offset points",
            fontsize=7.6,
            color=TEXT_COLOR,
            alpha=0.92,
            clip_on=False,
            bbox={"boxstyle": "round,pad=0.12", "facecolor": "white", "edgecolor": "none", "alpha": 0.72},
        )


def _plot_category_points(
    ax: plt.Axes,
    selected: pd.DataFrame,
    *,
    x_col: str,
    y_col: str,
    cmap,
    norm,
) -> None:
    for point_type, edge_color, marker in [
        ("Top 60M swarm", SELECTED_EDGE_COLOR, "o"),
        ("Baseline", BASELINE_COLOR, "s"),
        ("Deployed mixture", EXTRA_COLOR, "D"),
    ]:
        subset = selected.loc[selected["point_type"].eq(point_type)]
        if subset.empty:
            continue
        ax.scatter(
            subset[x_col],
            subset[y_col],
            s=52,
            c=subset["bpb_300m_6b"],
            cmap=cmap,
            norm=norm,
            marker=marker,
            edgecolors=edge_color,
            linewidths=1.1,
            zorder=4,
        )


def _legend_handles() -> list[Line2D]:
    return [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="white",
            markeredgecolor=POINT_EDGE_COLOR,
            markersize=7,
            label="Completed swarm",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="white",
            markeredgecolor=SELECTED_EDGE_COLOR,
            markeredgewidth=1.2,
            markersize=7,
            label="Top 60M swarm",
        ),
        Line2D(
            [0],
            [0],
            marker="s",
            color="none",
            markerfacecolor="white",
            markeredgecolor=BASELINE_COLOR,
            markeredgewidth=1.2,
            markersize=7,
            label="Baseline",
        ),
        Line2D(
            [0],
            [0],
            marker="D",
            color="none",
            markerfacecolor="white",
            markeredgecolor=EXTRA_COLOR,
            markeredgewidth=1.2,
            markersize=7,
            label="Deployed mixture",
        ),
    ]


def main() -> None:
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    _configure_matplotlib()
    frame = pd.read_csv(INPUT_CSV)
    summary = json.loads(SUMMARY_JSON.read_text())
    selected = _selected_points(frame)
    selected.to_csv(OUTPUT_CSV, index=False)

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.25), constrained_layout=True)
    ax_bpb, ax_rank = axes
    cmap = plt.get_cmap(CMAP_NAME)
    color_values = pd.concat([frame["bpb_300m_6b"], selected["bpb_300m_6b"]], ignore_index=True)
    norm = Normalize(
        vmin=float(color_values.quantile(0.03)),
        vmax=float(color_values.quantile(0.97)),
        clip=True,
    )

    x_line = np.linspace(float(frame["bpb_60m"].min()), float(frame["bpb_60m"].max()), 200)
    coeffs = np.polyfit(frame["bpb_60m"], frame["bpb_300m_6b"], deg=1)
    base_scatter = ax_bpb.scatter(
        frame["bpb_60m"],
        frame["bpb_300m_6b"],
        s=20,
        c=frame["bpb_300m_6b"],
        cmap=cmap,
        norm=norm,
        alpha=0.82,
        edgecolors="none",
    )
    ax_bpb.plot(x_line, coeffs[0] * x_line + coeffs[1], linestyle="--", color="#A8A29A", linewidth=1.2)
    _plot_category_points(ax_bpb, selected, x_col="bpb_60m", y_col="bpb_300m_6b", cmap=cmap, norm=norm)
    _annotate(ax_bpb, selected, x_col="bpb_60m", y_col="bpb_300m_6b")
    ax_bpb.set_title(f"BPB transfer: Pearson {summary['pearson_r']:.2f}", fontsize=13)
    ax_bpb.set_xlabel("60M/1.2B BPB")
    ax_bpb.set_ylabel("300M/6B BPB")
    ax_bpb.set_xlim(
        min(float(frame["bpb_60m"].min()), float(selected["bpb_60m"].min())) - 0.014,
        max(float(frame["bpb_60m"].max()), float(selected["bpb_60m"].max())) + 0.020,
    )
    ax_bpb.set_ylim(
        min(float(frame["bpb_300m_6b"].min()), float(selected["bpb_300m_6b"].min())) - 0.012,
        max(float(frame["bpb_300m_6b"].max()), float(selected["bpb_300m_6b"].max())) + 0.008,
    )

    max_rank = int(frame[["rank_60m_within_completed", "rank_300m_6b"]].to_numpy().max())
    ax_rank.scatter(
        frame["rank_60m_within_completed"],
        frame["rank_300m_6b"],
        s=20,
        c=frame["bpb_300m_6b"],
        cmap=cmap,
        norm=norm,
        alpha=0.82,
        edgecolors="none",
    )
    _plot_category_points(
        ax_rank,
        selected,
        x_col="rank_60m_for_plot",
        y_col="rank_300m_for_plot",
        cmap=cmap,
        norm=norm,
    )
    ax_rank.plot([1, max_rank], [1, max_rank], linestyle="--", color="#A8A29A", linewidth=1.2)
    ax_rank.set_title(f"Rank transfer: Spearman {summary['spearman_rho']:.2f}", fontsize=13)
    ax_rank.set_xlabel("Rank at 60M/1.2B")
    ax_rank.set_ylabel("Rank at 300M/6B")
    ax_rank.set_xlim(0, max_rank + 7)
    ax_rank.set_ylim(0, max_rank + 7)

    for ax in axes:
        ax.grid(True, color=GRID_COLOR, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.legend(
        _legend_handles(),
        [handle.get_label() for handle in _legend_handles()],
        loc="upper center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 1.08),
        fontsize=9,
        handletextpad=0.4,
        columnspacing=1.2,
    )
    cbar = fig.colorbar(base_scatter, ax=axes, fraction=0.027, pad=0.015)
    cbar.set_label("300M/6B BPB", fontsize=9)
    cbar.ax.tick_params(labelsize=8)
    fig.savefig(OUTPUT_STEM.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(OUTPUT_STEM.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {OUTPUT_STEM.with_suffix('.png')}")
    print(f"Wrote {OUTPUT_STEM.with_suffix('.pdf')}")
    print(f"Wrote {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
