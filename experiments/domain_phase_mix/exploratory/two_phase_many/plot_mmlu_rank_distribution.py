# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Plot rank-vs-metric distributions for two-phase many-domain runs."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CSV_PATH = Path(__file__).with_name("two_phase_many.csv")
OUTPUT_PATH = Path(__file__).with_name("mmlu_rank_distribution.png")

METRICS: tuple[tuple[str, bool, str], ...] = (
    ("lm_eval/mmlu_5shot/bpb", True, "MMLU 5-shot BPB"),
    ("lm_eval/mmlu_5shot/choice_logprob", False, "MMLU 5-shot choice_logprob"),
)

EXTRA_BASELINE_ROWS = (
    {
        "run_name": "baseline_dsre_ensemble",
        "lm_eval/mmlu_5shot/bpb": 2.1438617064681247,
        "lm_eval/mmlu_5shot/choice_logprob": -1.4739001158830458,
    },
    {
        "run_name": "baseline_dsre_observed_consensus",
        "lm_eval/mmlu_5shot/bpb": 2.318125450562879,
        "lm_eval/mmlu_5shot/choice_logprob": -1.5486506430899756,
    },
    {
        "run_name": "baseline_clr_ridge_balanced",
        "lm_eval/mmlu_5shot/bpb": 2.2052168593323414,
        "lm_eval/mmlu_5shot/choice_logprob": -1.51408915395576,
    },
    {
        "run_name": "baseline_dsre_ceq_st_lite",
        "lm_eval/mmlu_5shot/bpb": 2.2992783191251926,
        "lm_eval/mmlu_5shot/choice_logprob": -1.57270913806665,
    },
)

BASELINE_STYLES = {
    "baseline_proportional": {"color": "#E15759", "label": "Baseline proportional"},
    "baseline_unimax": {"color": "#4E79A7", "label": "Baseline UniMax"},
    "baseline_olmix_loglinear": {"color": "#59A14F", "label": "Baseline Olmix"},
    "baseline_dsre_ensemble": {"color": "#F28E2B", "label": "Baseline DS-RE ensemble"},
    "baseline_clr_ridge_balanced": {"color": "#76B7B2", "label": "Baseline CLR-Ridge"},
    "baseline_dsre_ceq_st_lite": {"color": "#B07AA1", "label": "Baseline DS-RE-CEQ-ST(lite)"},
}

BASELINE_ANNOTATION_OFFSETS = {
    "baseline_proportional": (10, 10),
    "baseline_unimax": (10, -30),
    "baseline_olmix_loglinear": (10, -60),
    "baseline_dsre_ensemble": (10, -90),
    "baseline_clr_ridge_balanced": (10, -120),
    "baseline_dsre_ceq_st_lite": (10, -150),
}

PAIR_STYLES = {
    "run_00097": {"color": "#9C755F", "label": "run_00097", "marker": "o", "facecolors": "none"},
    "baseline_dsre_observed_consensus": {
        "color": "#9C755F",
        "label": "run_00097 rerun",
        "marker": "D",
        "facecolors": "#9C755F",
    },
}

PAIR_ANNOTATION_POSITIONS = {
    "lm_eval/mmlu_5shot/bpb": (0.56, 0.30),
    "lm_eval/mmlu_5shot/choice_logprob": (0.55, 0.28),
}


def _prepare_frame() -> pd.DataFrame:
    df = pd.read_csv(CSV_PATH)
    extra = pd.DataFrame(EXTRA_BASELINE_ROWS)
    augmented = pd.concat([df, extra], ignore_index=True, sort=False)
    return augmented.drop_duplicates(subset=["run_name"], keep="last")


def _ranked_frame(df: pd.DataFrame, metric: str, lower_is_better: bool) -> pd.DataFrame:
    ranked = df[["run_name", metric]].dropna().sort_values(metric, ascending=lower_is_better, ignore_index=True).copy()
    ranked.index = np.arange(1, len(ranked) + 1)
    ranked["rank"] = ranked.index
    return ranked


def main() -> None:
    df = _prepare_frame()

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, len(METRICS), figsize=(15, 6), dpi=180)
    fig.suptitle("Two-phase many-domain swarm: MMLU rank distributions", fontsize=18, y=0.98)

    cmap = plt.colormaps["RdYlGn_r"]

    for ax, (metric, lower_is_better, title) in zip(np.atleast_1d(axes), METRICS, strict=True):
        ranked = _ranked_frame(df, metric, lower_is_better)
        ranks = ranked["rank"].to_numpy()
        values = ranked[metric].to_numpy()

        color_positions = np.linspace(0.0, 1.0, len(ranked))
        point_colors = cmap(color_positions)

        ax.plot(ranks, values, color="#4C78A8", linewidth=2.0, alpha=0.95, zorder=1)
        ax.scatter(ranks, values, c=point_colors, s=26, edgecolors="none", alpha=0.9, zorder=2)

        for run_name, style in BASELINE_STYLES.items():
            baseline = ranked.loc[ranked["run_name"] == run_name].iloc[0]
            baseline_rank = int(baseline["rank"])
            baseline_value = float(baseline[metric])
            offset_x, offset_y = BASELINE_ANNOTATION_OFFSETS[run_name]
            ax.scatter(
                [baseline_rank],
                [baseline_value],
                marker="D",
                s=70,
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

        original = ranked.loc[ranked["run_name"] == "run_00097"].iloc[0]
        rerun = ranked.loc[ranked["run_name"] == "baseline_dsre_observed_consensus"].iloc[0]
        for run_name, point in (("run_00097", original), ("baseline_dsre_observed_consensus", rerun)):
            style = PAIR_STYLES[run_name]
            ax.scatter(
                [int(point["rank"])],
                [float(point[metric])],
                marker=style["marker"],
                s=78,
                facecolors=style["facecolors"],
                edgecolors=style["color"],
                linewidths=1.3,
                zorder=5,
            )

        annotation_xytext = PAIR_ANNOTATION_POSITIONS[metric]
        shared_text = f"Same schedule\n" f"run_00097 rank {int(original['rank'])}\n" f"rerun rank {int(rerun['rank'])}"
        ax.annotate(
            shared_text,
            xy=(int(original["rank"]), float(original[metric])),
            xycoords="data",
            xytext=annotation_xytext,
            textcoords="axes fraction",
            fontsize=9,
            color=PAIR_STYLES["run_00097"]["color"],
            arrowprops={"arrowstyle": "-", "color": PAIR_STYLES["run_00097"]["color"], "lw": 1.0},
            bbox={
                "boxstyle": "round,pad=0.2",
                "facecolor": "white",
                "alpha": 0.9,
                "edgecolor": PAIR_STYLES["run_00097"]["color"],
            },
        )
        ax.annotate(
            "",
            xy=(int(rerun["rank"]), float(rerun[metric])),
            xycoords="data",
            xytext=annotation_xytext,
            textcoords="axes fraction",
            arrowprops={"arrowstyle": "-", "color": PAIR_STYLES["run_00097"]["color"], "lw": 1.0},
        )

        best_label = "min" if lower_is_better else "max"
        ax.set_title(title, fontsize=14)
        ax.set_xlabel("Rank (1 = best)")
        ax.set_ylabel(metric)
        ax.set_xlim(1, len(ranked))
        summary_x = 0.98 if metric == "lm_eval/mmlu_5shot/choice_logprob" else 0.02
        summary_ha = "right" if metric == "lm_eval/mmlu_5shot/choice_logprob" else "left"
        ax.text(
            summary_x,
            0.98,
            (
                f"n = {len(ranked)}\n"
                f"{best_label} = {values[0]:.4f}\n"
                f"median = {np.median(values):.4f}\n"
                f"{'max' if lower_is_better else 'min'} = {values[-1]:.4f}"
            ),
            transform=ax.transAxes,
            va="top",
            ha=summary_ha,
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
