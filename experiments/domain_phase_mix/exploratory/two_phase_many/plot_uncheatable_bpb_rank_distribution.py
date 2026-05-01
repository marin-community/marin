# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Plot rank-vs-metric distribution for uncheatable-eval BPB with surrogate baselines."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CSV_PATH = Path(__file__).with_name("two_phase_many.csv")
OUTPUT_PNG = Path(__file__).with_name("uncheatable_bpb_rank_distribution.png")
OUTPUT_CSV = Path(__file__).with_name("uncheatable_bpb_rank_distribution_highlights.csv")
METRIC = "eval/uncheatable_eval/bpb"

EXTRA_BASELINE_ROWS = (
    {
        "run_name": "baseline_power_ridge_single_constant_mix",
        METRIC: 1.1439474821090698,
    },
    {
        "run_name": "baseline_dsre_ceq_predicted",
        METRIC: 1.1207557916641235,
    },
    {
        "run_name": "baseline_dsre_ceq_predicted_quality_collapsed",
        METRIC: 1.1111559867858887,
    },
    {
        "run_name": "baseline_olmix_loglinear_uncheatable_bpb",
        METRIC: 1.0687161684036255,
    },
    {
        "run_name": "baseline_thresholdtotal_overfit_uncheatable_bpb",
        METRIC: 1.0767244100570679,
    },
    {
        "run_name": "baseline_dsre_ceq_predicted_topic_collapsed",
        METRIC: 1.1125632524490356,
    },
)

HIGHLIGHT_STYLES = {
    "baseline_olmix_loglinear_uncheatable_bpb": {
        "color": "#1F77B4",
        "label": "Olmix loglinear (79 params)",
        "marker": "P",
    },
    "baseline_thresholdtotal_overfit_uncheatable_bpb": {
        "color": "#FF9D0A",
        "label": "ThresholdTotal-Overfit (41 params)",
        "marker": "X",
    },
    "baseline_proportional": {
        "color": "#E15759",
        "label": "Proportional (0 params)",
        "marker": "D",
    },
    "baseline_unimax": {
        "color": "#4E79A7",
        "label": "UniMax (0 params)",
        "marker": "D",
    },
    "baseline_power_ridge_single_constant_mix": {
        "color": "#F28E2B",
        "label": "Power-Ridge (79 params)",
        "marker": "o",
    },
    "baseline_dsre_ceq_predicted": {
        "color": "#B07AA1",
        "label": "DS-RE-CEQ predicted (162 params)",
        "marker": "s",
    },
    "baseline_dsre_ceq_predicted_quality_collapsed": {
        "color": "#59A14F",
        "label": "DS-RE-CEQ quality-collapsed (162 params)",
        "marker": "^",
    },
    "baseline_dsre_ceq_predicted_topic_collapsed": {
        "color": "#76B7B2",
        "label": "DS-RE-CEQ topic-collapsed (162 params)",
        "marker": "v",
    },
}

ANNOTATION_OFFSETS = {
    "baseline_olmix_loglinear_uncheatable_bpb": (10, 30),
    "baseline_thresholdtotal_overfit_uncheatable_bpb": (10, 2),
    "baseline_unimax": (10, -30),
    "baseline_proportional": (10, 18),
    "baseline_dsre_ceq_predicted_quality_collapsed": (10, 14),
    "baseline_dsre_ceq_predicted_topic_collapsed": (10, -16),
    "baseline_dsre_ceq_predicted": (10, -48),
    "baseline_power_ridge_single_constant_mix": (10, -82),
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
    fig, ax = plt.subplots(figsize=(10.5, 6.4), dpi=180)
    fig.suptitle("Two-phase many-domain swarm: uncheatable-eval rank distribution", fontsize=17, y=0.98)

    cmap = plt.colormaps["RdYlGn_r"]
    ranks = ranked["rank"].to_numpy()
    values = ranked[METRIC].to_numpy()
    point_colors = cmap(np.linspace(0.0, 1.0, len(ranked)))

    ax.plot(ranks, values, color="#4C78A8", linewidth=2.0, alpha=0.95, zorder=1)
    ax.scatter(ranks, values, c=point_colors, s=28, edgecolors="none", alpha=0.9, zorder=2)

    highlight_rows: list[dict[str, float | int | str]] = []
    for run_name, style in HIGHLIGHT_STYLES.items():
        baseline = ranked.loc[ranked["run_name"] == run_name].iloc[0]
        baseline_rank = int(baseline["rank"])
        baseline_value = float(baseline[METRIC])
        offset_x, offset_y = ANNOTATION_OFFSETS[run_name]
        highlight_rows.append(
            {
                "run_name": run_name,
                "label": style["label"],
                "params": int(style["label"].split("(")[1].split()[0]),
                "rank": baseline_rank,
                METRIC: baseline_value,
            }
        )
        ax.scatter(
            [baseline_rank],
            [baseline_value],
            marker=style["marker"],
            s=74,
            color=style["color"],
            edgecolors="black",
            linewidths=0.8,
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
                "alpha": 0.86,
                "edgecolor": style["color"],
            },
        )

    highlight_df = pd.DataFrame(highlight_rows).sort_values("rank")
    highlight_df.to_csv(OUTPUT_CSV, index=False)

    ax.set_title(METRIC, fontsize=14)
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
    ax.legend(loc="lower right", fontsize=9.5, frameon=True)

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUTPUT_PNG, bbox_inches="tight")
    print(f"Saved plot to {OUTPUT_PNG}")
    print(f"Saved highlights to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
