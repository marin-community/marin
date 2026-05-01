# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Plot Group-Retain-Penalty (GRP) against the proportional baseline."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from experiments.domain_phase_mix.exploratory.two_phase_many.surrogate_search import (
    plot_ccpairtotal_vs_ccglobalpremium_vs_best as comparison_plot,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.surrogate_search.structured_epoch_family import (
    MANY_DOMAIN_TARGET,
    load_two_phase_many_packet,
)
from experiments.domain_phase_mix.two_phase_many_genericfamily_tuned_baseline import (
    genericfamily_tuned_summary,
)

SCRIPT_DIR = Path(__file__).resolve().parent
PLOT_PNG = SCRIPT_DIR / "grp_vs_proportional_weights.png"
WEIGHTS_CSV = SCRIPT_DIR / "grp_vs_proportional_weights.csv"
BAR_EDGE_COLOR = "#0f172a"
GRID_COLOR = "#cbd5e1"
TEXT_MUTED_COLOR = "#475569"
GRP_COLOR = "#232b32"
PROPORTIONAL_COLOR = "#dcd0bb"
GRP_REALIZED_BPB = 1.0403348207473755
CC_TOPIC_DISPLAY = {
    "art and design": "art/design",
    "crime and law": "crime/law",
    "education and jobs": "education/jobs",
    "electronics and hardware": "electronics/hardware",
    "finance and business": "finance/business",
    "food and dining": "food/dining",
    "history and geography": "history/geography",
    "science math and technology": "science/math/tech",
}


def _display_non_cc_label(domain_name: str) -> str:
    label = domain_name.removeprefix("dolma3_").removeprefix("dolmino_")
    return label.replace("_", " ")


def _cc_topic_and_quality(domain_name: str) -> tuple[str, str]:
    topic, quality = domain_name[len("dolma3_cc/") :].rsplit("_", maxsplit=1)
    topic_display = topic.replace("_", " ")
    return CC_TOPIC_DISPLAY.get(topic_display, topic_display), quality


def _grp_domain_order(domain_names: list[str], grp_weights: np.ndarray) -> tuple[list[int], list[int]]:
    non_cc_indices, cc_indices = comparison_plot._split_domain_blocks(domain_names)
    non_cc_sorted = sorted(
        non_cc_indices,
        key=lambda idx: max(float(grp_weights[0, idx]), float(grp_weights[1, idx])),
        reverse=True,
    )

    cc_topics: list[tuple[str, int, int]] = []
    for i in range(0, len(cc_indices), 2):
        hi = cc_indices[i]
        lo = cc_indices[i + 1]
        topic, _ = _cc_topic_and_quality(domain_names[hi])
        cc_topics.append((topic, hi, lo))
    cc_topics.sort(key=lambda item: item[0])

    cc_sorted: list[int] = []
    for _, hi, lo in cc_topics:
        cc_sorted.extend([hi, lo])
    return non_cc_sorted, cc_sorted


def _block_xlim(schedules: list[tuple[str, np.ndarray, str]], *, phase_idx: int, indices: list[int]) -> float:
    x_max = 0.0
    for _, weights, _ in schedules:
        x_max = max(x_max, float(np.max(weights[phase_idx, indices])))
    return max(x_max * 1.08 + 0.006, 0.05)


def _plot_non_cc_block(
    *,
    ax,
    indices: list[int],
    labels: list[str],
    schedules: list[tuple[str, np.ndarray, str]],
    phase_idx: int,
    multipliers: np.ndarray,
    title: str,
    show_legend: bool,
) -> None:
    y = np.arange(len(indices))
    bar_height = 0.34
    x_max = _block_xlim(schedules, phase_idx=phase_idx, indices=indices)

    for schedule_idx, (label, weights, color) in enumerate(schedules):
        offsets = y + (schedule_idx - 0.5) * bar_height
        phase_weights = weights[phase_idx, indices]
        epochs = phase_weights * multipliers[indices]
        ax.barh(
            offsets,
            phase_weights,
            height=bar_height,
            color=color,
            alpha=0.97,
            edgecolor=BAR_EDGE_COLOR,
            linewidth=0.45,
            label=label if show_legend else None,
        )
        for offset, weight, epoch in zip(offsets, phase_weights, epochs, strict=True):
            epoch_label = comparison_plot._format_epochs(float(epoch))
            text_pad = max(0.0022, x_max * 0.0055)
            text_x = max(float(weight) + text_pad, x_max * 0.018)
            ax.text(
                text_x,
                float(offset),
                epoch_label,
                va="center",
                ha="left",
                fontsize=13.5,
                color=BAR_EDGE_COLOR,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.88, "pad": 0.16},
            )

    ax.set_xlim(0.0, x_max)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=20.5, fontweight="bold")
    ax.invert_yaxis()
    ax.set_xlabel("Mixture Weight", fontsize=17.5, fontweight="bold")
    ax.set_title(title, fontsize=24, pad=12, fontweight="bold")
    ax.grid(axis="x", linestyle="--", color=GRID_COLOR, alpha=0.65, linewidth=0.85)
    ax.tick_params(axis="x", labelsize=15.5)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_color("#94a3b8")
    ax.spines["left"].set_color("#94a3b8")


def _plot_cc_block(
    *,
    ax,
    domain_names: list[str],
    indices: list[int],
    schedules: list[tuple[str, np.ndarray, str]],
    phase_idx: int,
    multipliers: np.ndarray,
    title: str,
) -> None:
    y = np.arange(len(indices))
    bar_height = 0.34
    x_max = _block_xlim(schedules, phase_idx=phase_idx, indices=indices)
    topic_label_x = -0.078
    quality_label_x = -0.020

    for schedule_idx, (_, weights, color) in enumerate(schedules):
        offsets = y + (schedule_idx - 0.5) * bar_height
        phase_weights = weights[phase_idx, indices]
        epochs = phase_weights * multipliers[indices]
        ax.barh(
            offsets,
            phase_weights,
            height=bar_height,
            color=color,
            alpha=0.97,
            edgecolor=BAR_EDGE_COLOR,
            linewidth=0.45,
        )
        for offset, weight, epoch in zip(offsets, phase_weights, epochs, strict=True):
            epoch_label = comparison_plot._format_epochs(float(epoch))
            text_pad = max(0.0010, x_max * 0.0023)
            text_x = max(float(weight) + text_pad, x_max * 0.014)
            ax.text(
                text_x,
                float(offset),
                epoch_label,
                va="center",
                ha="left",
                fontsize=13.0,
                color=BAR_EDGE_COLOR,
            )

    ax.set_xlim(0.0, x_max)
    ax.set_yticks(y)
    ax.set_yticklabels([""] * len(indices))
    ax.invert_yaxis()
    ax.set_xlabel("Mixture Weight", fontsize=17.5, fontweight="bold")
    ax.set_title(title, fontsize=24, pad=12, fontweight="bold")
    ax.grid(axis="x", linestyle="--", color=GRID_COLOR, alpha=0.65, linewidth=0.85)
    ax.tick_params(axis="x", labelsize=15.5)
    ax.tick_params(axis="y", length=0)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_color("#94a3b8")
    ax.spines["left"].set_color("#94a3b8")

    for pair_start in range(0, len(indices), 2):
        high_idx = indices[pair_start]
        low_idx = indices[pair_start + 1]
        topic_high, quality_high = _cc_topic_and_quality(domain_names[high_idx])
        topic_low, quality_low = _cc_topic_and_quality(domain_names[low_idx])
        assert topic_high == topic_low
        assert quality_high == "high"
        assert quality_low == "low"
        mid_y = pair_start + 0.5
        ax.text(
            topic_label_x,
            mid_y,
            topic_high,
            transform=ax.get_yaxis_transform(),
            ha="right",
            va="center",
            fontsize=19.5,
            fontweight="bold",
            color=BAR_EDGE_COLOR,
            clip_on=False,
        )
        ax.text(
            quality_label_x,
            pair_start,
            "high",
            transform=ax.get_yaxis_transform(),
            ha="right",
            va="center",
            fontsize=14.0,
            fontweight="bold",
            color=TEXT_MUTED_COLOR,
            clip_on=False,
        )
        ax.text(
            quality_label_x,
            pair_start + 1,
            "low",
            transform=ax.get_yaxis_transform(),
            ha="right",
            va="center",
            fontsize=14.0,
            fontweight="bold",
            color=TEXT_MUTED_COLOR,
            clip_on=False,
        )


def main() -> None:
    data = load_two_phase_many_packet(target=MANY_DOMAIN_TARGET)
    grp_summary = genericfamily_tuned_summary()
    grp_weights = comparison_plot._summary_weights(grp_summary, data.domain_names)

    proportional_idx = int(data.frame.index[data.frame["run_name"] == "baseline_proportional"][0])
    proportional_weights = comparison_plot._row_weights(data.frame, data.domain_names, proportional_idx)

    best_idx = int(np.argmin(data.y))
    best_name = str(data.frame.iloc[best_idx][data.name_col])
    best_bpb = float(data.y[best_idx])

    original_ranked = (
        data.frame[["run_name", MANY_DOMAIN_TARGET]]
        .dropna()
        .sort_values(MANY_DOMAIN_TARGET, ascending=True)
        .reset_index(drop=True)
    )
    original_ranked["rank"] = original_ranked.index + 1
    proportional_row = original_ranked[original_ranked["run_name"] == "baseline_proportional"].iloc[0]
    best_row = original_ranked[original_ranked["run_name"] == best_name].iloc[0]
    grp_would_rank = int(1 + np.sum(original_ranked[MANY_DOMAIN_TARGET].to_numpy() < GRP_REALIZED_BPB))

    schedules = [
        ("GRP", grp_weights, GRP_COLOR),
        ("Proportional", proportional_weights, PROPORTIONAL_COLOR),
    ]

    canonical_non_cc_indices, canonical_cc_indices = _grp_domain_order(data.domain_names, grp_weights)

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(26, 22),
        gridspec_kw={"width_ratios": [1.0, 1.62], "hspace": 0.22, "wspace": 0.31},
        facecolor="white",
    )

    _plot_non_cc_block(
        ax=axes[0, 0],
        indices=canonical_non_cc_indices,
        labels=[_display_non_cc_label(data.domain_names[idx]) for idx in canonical_non_cc_indices],
        schedules=schedules,
        phase_idx=0,
        multipliers=data.c0,
        title="Phase 0: Non-CC Domains",
        show_legend=True,
    )
    _plot_cc_block(
        ax=axes[0, 1],
        domain_names=data.domain_names,
        indices=canonical_cc_indices,
        schedules=schedules,
        phase_idx=0,
        multipliers=data.c0,
        title="Phase 0: CC Domains",
    )
    _plot_non_cc_block(
        ax=axes[1, 0],
        indices=canonical_non_cc_indices,
        labels=[_display_non_cc_label(data.domain_names[idx]) for idx in canonical_non_cc_indices],
        schedules=schedules,
        phase_idx=1,
        multipliers=data.c1,
        title="Phase 1: Non-CC Domains",
        show_legend=False,
    )
    _plot_cc_block(
        ax=axes[1, 1],
        domain_names=data.domain_names,
        indices=canonical_cc_indices,
        schedules=schedules,
        phase_idx=1,
        multipliers=data.c1,
        title="Phase 1: CC Domains",
    )

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.suptitle("Group-Retain-Penalty (GRP) vs Proportional", fontsize=34, y=0.996, fontweight="bold")
    fig.text(
        0.5,
        0.952,
        "Many-domain two-phase mixture weights for uncheatable-eval BPB",
        ha="center",
        va="center",
        fontsize=20,
        color=TEXT_MUTED_COLOR,
    )
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, fontsize=18.5, bbox_to_anchor=(0.5, 0.928))
    fig.text(
        0.5,
        0.072,
        (
            f"GRP (realized): {GRP_REALIZED_BPB:.4f} BPB, would place {grp_would_rank}st vs the original 241-run swarm\n"
            "Proportional: "
            f"{float(proportional_row[MANY_DOMAIN_TARGET]):.4f} BPB, rank {int(proportional_row['rank'])}/241   |   "
            f"Best observed: {best_bpb:.4f} BPB ({best_name}), rank {int(best_row['rank'])}/241"
        ),
        ha="center",
        va="center",
        fontsize=17.5,
        color="#0f172a",
        bbox={
            "boxstyle": "round,pad=0.62,rounding_size=0.18",
            "facecolor": "#f8fafc",
            "edgecolor": "#cbd5e1",
            "alpha": 0.97,
        },
    )
    fig.text(
        0.5,
        0.026,
        "Bar-end labels show effective epochs for that domain in that phase. Values below 0.01 are displayed as 0.",
        ha="center",
        va="center",
        fontsize=15,
        color=TEXT_MUTED_COLOR,
    )
    fig.subplots_adjust(top=0.905, left=0.14, right=0.985, bottom=0.13, hspace=0.24, wspace=0.31)
    fig.savefig(PLOT_PNG, dpi=220, bbox_inches="tight")
    plt.close(fig)

    rows = []
    for label, weights, _ in schedules:
        for domain_name, phase0_weight, phase1_weight, c0, c1 in zip(
            data.domain_names,
            weights[0],
            weights[1],
            data.c0,
            data.c1,
            strict=True,
        ):
            rows.append(
                {
                    "schedule": label,
                    "domain": domain_name,
                    "phase0_weight": float(phase0_weight),
                    "phase0_epochs": float(phase0_weight * c0),
                    "phase1_weight": float(phase1_weight),
                    "phase1_epochs": float(phase1_weight * c1),
                }
            )
    pd.DataFrame(rows).to_csv(WEIGHTS_CSV, index=False)
    print(f"Plot: {PLOT_PNG}")
    print(f"Weights: {WEIGHTS_CSV}")


if __name__ == "__main__":
    main()
