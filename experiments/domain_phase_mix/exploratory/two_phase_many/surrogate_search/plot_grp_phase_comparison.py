# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Plot GRP mixture weights with phase-0 versus phase-1 bars."""

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
from experiments.domain_phase_mix.exploratory.two_phase_many.surrogate_search.plot_grp_vs_proportional import (
    _cc_topic_and_quality,
    _display_non_cc_label,
    _grp_domain_order,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.surrogate_search.structured_epoch_family import (
    MANY_DOMAIN_TARGET,
    load_two_phase_many_packet,
)
from experiments.domain_phase_mix.two_phase_many_genericfamily_tuned_baseline import (
    genericfamily_tuned_summary,
)

SCRIPT_DIR = Path(__file__).resolve().parent
PLOT_PNG = SCRIPT_DIR / "grp_phase_comparison.png"
WEIGHTS_CSV = SCRIPT_DIR / "grp_phase_comparison.csv"
PHASE0_COLOR = "#232b32"
PHASE1_COLOR = "#dcd0bb"
BAR_EDGE_COLOR = "#0f172a"
GRID_COLOR = "#cbd5e1"
TEXT_MUTED_COLOR = "#475569"


def _format_epochs(epochs: float) -> str:
    if epochs < 0.01:
        return "0"
    if epochs >= 10.0:
        return f"{epochs:.1f}"
    if epochs >= 1.0:
        return f"{epochs:.1f}"
    return f"{epochs:.2f}"


def _plot_non_cc_block(
    *,
    ax,
    indices: list[int],
    labels: list[str],
    weights: np.ndarray,
    phase0_multipliers: np.ndarray,
    phase1_multipliers: np.ndarray,
    title: str,
) -> None:
    y = np.arange(len(indices))
    bar_height = 0.34
    x_max = max(float(np.max(weights[:, indices])) * 1.08 + 0.006, 0.05)

    ax.barh(
        y - bar_height / 2,
        weights[0, indices],
        height=bar_height,
        color=PHASE0_COLOR,
        edgecolor=BAR_EDGE_COLOR,
        linewidth=0.45,
        label="Phase 0",
    )
    ax.barh(
        y + bar_height / 2,
        weights[1, indices],
        height=bar_height,
        color=PHASE1_COLOR,
        edgecolor=BAR_EDGE_COLOR,
        linewidth=0.45,
        label="Phase 1",
    )

    for offset, weight, multiplier in zip(
        y - bar_height / 2, weights[0, indices], phase0_multipliers[indices], strict=True
    ):
        ax.text(
            float(weight) + max(0.002, x_max * 0.006),
            float(offset),
            _format_epochs(float(weight) * float(multiplier)),
            va="center",
            ha="left",
            fontsize=12,
            color=BAR_EDGE_COLOR,
        )
    for offset, weight, multiplier in zip(
        y + bar_height / 2, weights[1, indices], phase1_multipliers[indices], strict=True
    ):
        ax.text(
            float(weight) + max(0.002, x_max * 0.006),
            float(offset),
            _format_epochs(float(weight) * float(multiplier)),
            va="center",
            ha="left",
            fontsize=12,
            color=BAR_EDGE_COLOR,
        )

    ax.set_xlim(0.0, x_max)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=19, fontweight="bold")
    ax.invert_yaxis()
    ax.set_xlabel("Mixture Weight", fontsize=17, fontweight="bold")
    ax.set_title(title, fontsize=24, fontweight="bold", pad=12)
    ax.grid(axis="x", linestyle="--", color=GRID_COLOR, alpha=0.65, linewidth=0.85)
    ax.tick_params(axis="x", labelsize=15)
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
    weights: np.ndarray,
    phase0_multipliers: np.ndarray,
    phase1_multipliers: np.ndarray,
    title: str,
) -> None:
    y = np.arange(len(indices))
    bar_height = 0.34
    x_max = max(float(np.max(weights[:, indices])) * 1.08 + 0.006, 0.05)
    topic_label_x = -0.07
    quality_label_x = -0.017

    ax.barh(
        y - bar_height / 2,
        weights[0, indices],
        height=bar_height,
        color=PHASE0_COLOR,
        edgecolor=BAR_EDGE_COLOR,
        linewidth=0.45,
        label="Phase 0",
    )
    ax.barh(
        y + bar_height / 2,
        weights[1, indices],
        height=bar_height,
        color=PHASE1_COLOR,
        edgecolor=BAR_EDGE_COLOR,
        linewidth=0.45,
        label="Phase 1",
    )

    for offset, weight, multiplier in zip(
        y - bar_height / 2, weights[0, indices], phase0_multipliers[indices], strict=True
    ):
        ax.text(
            float(weight) + max(0.0015, x_max * 0.0045),
            float(offset),
            _format_epochs(float(weight) * float(multiplier)),
            va="center",
            ha="left",
            fontsize=11.5,
            color=BAR_EDGE_COLOR,
        )
    for offset, weight, multiplier in zip(
        y + bar_height / 2, weights[1, indices], phase1_multipliers[indices], strict=True
    ):
        ax.text(
            float(weight) + max(0.0015, x_max * 0.0045),
            float(offset),
            _format_epochs(float(weight) * float(multiplier)),
            va="center",
            ha="left",
            fontsize=11.5,
            color=BAR_EDGE_COLOR,
        )

    ax.set_xlim(0.0, x_max)
    ax.set_yticks(y)
    ax.set_yticklabels([""] * len(indices))
    ax.invert_yaxis()
    ax.set_xlabel("Mixture Weight", fontsize=17, fontweight="bold")
    ax.set_title(title, fontsize=24, fontweight="bold", pad=12)
    ax.grid(axis="x", linestyle="--", color=GRID_COLOR, alpha=0.65, linewidth=0.85)
    ax.tick_params(axis="x", labelsize=15)
    ax.tick_params(axis="y", length=0)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_color("#94a3b8")
    ax.spines["left"].set_color("#94a3b8")

    for pair_start in range(0, len(indices), 2):
        hi = indices[pair_start]
        lo = indices[pair_start + 1]
        topic_hi, quality_hi = _cc_topic_and_quality(domain_names[hi])
        topic_lo, quality_lo = _cc_topic_and_quality(domain_names[lo])
        assert topic_hi == topic_lo
        assert quality_hi == "high"
        assert quality_lo == "low"
        mid_y = pair_start + 0.5
        ax.text(
            topic_label_x,
            mid_y,
            topic_hi,
            transform=ax.get_yaxis_transform(),
            ha="right",
            va="center",
            fontsize=18.5,
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
            fontsize=13.5,
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
            fontsize=13.5,
            fontweight="bold",
            color=TEXT_MUTED_COLOR,
            clip_on=False,
        )


def main() -> None:
    data = load_two_phase_many_packet(target=MANY_DOMAIN_TARGET)
    grp_summary = genericfamily_tuned_summary()
    grp_weights = comparison_plot._summary_weights(grp_summary, data.domain_names)

    non_cc_indices, cc_indices = _grp_domain_order(data.domain_names, grp_weights)

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(24, 15),
        gridspec_kw={"width_ratios": [1.0, 1.65], "wspace": 0.30},
        facecolor="white",
    )

    _plot_non_cc_block(
        ax=axes[0],
        indices=non_cc_indices,
        labels=[_display_non_cc_label(data.domain_names[idx]) for idx in non_cc_indices],
        weights=grp_weights,
        phase0_multipliers=data.c0,
        phase1_multipliers=data.c1,
        title="Non-CC Domains",
    )
    _plot_cc_block(
        ax=axes[1],
        domain_names=data.domain_names,
        indices=cc_indices,
        weights=grp_weights,
        phase0_multipliers=data.c0,
        phase1_multipliers=data.c1,
        title="CC Domains",
    )

    handles, labels = axes[0].get_legend_handles_labels()
    fig.suptitle("Group-Retain-Penalty (GRP): Phase 0 vs Phase 1", fontsize=32, y=0.985, fontweight="bold")
    fig.text(
        0.5,
        0.947,
        "Many-domain two-phase mixture weights for uncheatable-eval BPB",
        ha="center",
        va="center",
        fontsize=19,
        color=TEXT_MUTED_COLOR,
    )
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, fontsize=18, bbox_to_anchor=(0.5, 0.925))
    fig.text(
        0.5,
        0.07,
        "Epoch labels; 80/20 WSD.",
        ha="center",
        va="center",
        fontsize=14,
        color=TEXT_MUTED_COLOR,
    )
    fig.subplots_adjust(top=0.88, left=0.12, right=0.985, bottom=0.10, wspace=0.30)
    fig.savefig(PLOT_PNG, dpi=220, bbox_inches="tight")
    plt.close(fig)

    rows = []
    for domain_name, phase0_weight, phase1_weight, c0, c1 in zip(
        data.domain_names, grp_weights[0], grp_weights[1], data.c0, data.c1, strict=True
    ):
        rows.append(
            {
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
