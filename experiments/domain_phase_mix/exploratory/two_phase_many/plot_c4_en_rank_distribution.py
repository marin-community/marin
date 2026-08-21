# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Plot rank-vs-metric distribution for C4-en BPB on two-phase many-domain runs."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CSV_PATH = Path(__file__).with_name("two_phase_many.csv")
OUTPUT_PATH = Path(__file__).with_name("c4_en_rank_distribution.png")
METRIC = "eval/paloma/c4_en/bpb"

EXTRA_BASELINE_ROWS = (
    {
        "run_name": "baseline_clr_ridge_balanced",
        METRIC: 1.1612638235092163,
    },
    {
        "run_name": "baseline_dsre_ceq_st_lite",
        METRIC: 1.175238013267517,
    },
)

BASELINE_STYLES = {
    "baseline_proportional": {"color": "#E15759", "label": "Baseline proportional"},
    "baseline_unimax": {"color": "#4E79A7", "label": "Baseline UniMax"},
    "baseline_olmix_loglinear": {"color": "#59A14F", "label": "Baseline Olmix"},
    "baseline_clr_ridge_balanced": {"color": "#F28E2B", "label": "Baseline CLR-Ridge"},
    "baseline_dsre_ceq_st_lite": {"color": "#B07AA1", "label": "Baseline DS-RE-CEQ-ST(lite)"},
}

BASELINE_ANNOTATION_OFFSETS = {
    "baseline_proportional": (10, 10),
    "baseline_unimax": (10, -24),
    "baseline_olmix_loglinear": (10, -58),
    "baseline_clr_ridge_balanced": (10, -92),
    "baseline_dsre_ceq_st_lite": (10, -126),
}


def _prepare_frame() -> pd.DataFrame:
    df = pd.read_csv(CSV_PATH)
    extra = pd.DataFrame(EXTRA_BASELINE_ROWS)
    augmented = pd.concat([df, extra], ignore_index=True, sort=False)
    return augmented.drop_duplicates(subset=["run_name"], keep="last")


def _ranked_frame(df: pd.DataFrame) -> pd.DataFrame:
    ranked = df[["run_name", METRIC]].dropna().sort_values(METRIC, ascending=True, ignore_index=True).copy()
    ranked.index = np.arange(1, len(ranked) + 1)
    ranked["rank"] = ranked.index
    return ranked


def main() -> None:
    df = _prepare_frame()
    ranked = _ranked_frame(df)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(9.5, 6.2), dpi=180)
    fig.suptitle("Two-phase many-domain swarm: C4-en rank distribution", fontsize=17, y=0.98)

    cmap = plt.colormaps["RdYlGn_r"]
    ranks = ranked["rank"].to_numpy()
    values = ranked[METRIC].to_numpy()
    point_colors = cmap(np.linspace(0.0, 1.0, len(ranked)))

    ax.plot(ranks, values, color="#4C78A8", linewidth=2.0, alpha=0.95, zorder=1)
    ax.scatter(ranks, values, c=point_colors, s=28, edgecolors="none", alpha=0.9, zorder=2)

    for run_name, style in BASELINE_STYLES.items():
        baseline = ranked.loc[ranked["run_name"] == run_name].iloc[0]
        baseline_rank = int(baseline["rank"])
        baseline_value = float(baseline[METRIC])
        offset_x, offset_y = BASELINE_ANNOTATION_OFFSETS[run_name]
        ax.scatter(
            [baseline_rank],
            [baseline_value],
            marker="D",
            s=72,
            color=style["color"],
            edgecolors="black",
            linewidths=0.7,
            zorder=4,
            label=style["label"],
        )
        ax.annotate(
            f"{style['label']}\nrank {baseline_rank}",
            xy=(baseline_rank, baseline_value),
            xytext=(offset_x, offset_y),
            textcoords="offset points",
            fontsize=9,
            color=style["color"],
            arrowprops={"arrowstyle": "-", "color": style["color"], "lw": 1.0},
            bbox={
                "boxstyle": "round,pad=0.2",
                "facecolor": "white",
                "alpha": 0.85,
                "edgecolor": style["color"],
            },
        )

    ax.set_title("eval/paloma/c4_en/bpb", fontsize=14)
    ax.set_xlabel("Rank (1 = best)")
    ax.set_ylabel(METRIC)
    ax.set_xlim(1, len(ranked))
    ax.text(
        0.02,
        0.98,
        (
            f"n = {len(ranked)}\n"
            f"min = {values[0]:.4f}\n"
            f"median = {np.median(values):.4f}\n"
            f"max = {values[-1]:.4f}"
        ),
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        bbox={
            "boxstyle": "round,pad=0.35",
            "facecolor": "white",
            "alpha": 0.9,
            "edgecolor": "#CCCCCC",
        },
    )
    ax.legend(loc="lower right", fontsize=10, frameon=True)

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUTPUT_PATH, bbox_inches="tight")
    print(f"Saved plot to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
