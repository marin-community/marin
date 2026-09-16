# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Plot CCPairTotal, CCGlobalPremium, and best observed many-domain mixtures."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from experiments.domain_phase_mix.exploratory.two_phase_many.surrogate_search.structured_epoch_family import (
    MANY_DOMAIN_TARGET,
    load_two_phase_many_packet,
)
from experiments.domain_phase_mix.two_phase_many_ccglobalpremium_baselines import (
    ccglobalpremium_retainedtotal_summary,
)
from experiments.domain_phase_mix.two_phase_many_ccpairtotal_baseline import (
    ccpairtotal_retainedtotal_summary,
)

SCRIPT_DIR = Path(__file__).resolve().parent
PLOT_PNG = SCRIPT_DIR / "ccpairtotal_ccglobalpremium_best_observed_weights.png"
WEIGHTS_CSV = SCRIPT_DIR / "ccpairtotal_ccglobalpremium_best_observed_weights.csv"
BAR_EDGE_COLOR = "#1f2937"
GRID_COLOR = "#d1d5db"
TEXT_MUTED_COLOR = "#4b5563"


def _row_weights(frame: pd.DataFrame, domain_names: list[str], row_idx: int) -> np.ndarray:
    row = frame.iloc[row_idx]
    return np.asarray(
        [
            [float(row[f"phase_0_{domain_name}"]) for domain_name in domain_names],
            [float(row[f"phase_1_{domain_name}"]) for domain_name in domain_names],
        ],
        dtype=float,
    )


def _summary_weights(summary: dict[str, object], domain_names: list[str]) -> np.ndarray:
    phase_weights = summary["phase_weights"]
    phase0 = np.asarray([float(phase_weights["phase_0"][domain_name]) for domain_name in domain_names], dtype=float)
    phase1 = np.asarray([float(phase_weights["phase_1"][domain_name]) for domain_name in domain_names], dtype=float)
    return np.stack([phase0, phase1], axis=0)


def _split_domain_blocks(domain_names: list[str]) -> tuple[list[int], list[int]]:
    cc_topics = sorted(
        {
            domain_name[len("dolma3_cc/") : -len("_high")]
            for domain_name in domain_names
            if domain_name.startswith("dolma3_cc/") and domain_name.endswith("_high")
        }
    )
    name_to_index = {domain_name: idx for idx, domain_name in enumerate(domain_names)}
    cc_order: list[int] = []
    for topic in cc_topics:
        cc_order.append(name_to_index[f"dolma3_cc/{topic}_high"])
        cc_order.append(name_to_index[f"dolma3_cc/{topic}_low"])
    cc_indices = set(cc_order)
    non_cc_order = [idx for idx in range(len(domain_names)) if idx not in cc_indices]
    return non_cc_order, cc_order


def _phase_order(
    *,
    domain_names: list[str],
    phase_idx: int,
    schedules: list[tuple[str, np.ndarray, tuple[float, float, float, float]]],
) -> tuple[list[int], list[int]]:
    non_cc_indices, cc_indices = _split_domain_blocks(domain_names)
    score_by_idx: dict[int, float] = {}
    for idx in range(len(domain_names)):
        score_by_idx[idx] = max(float(weights[phase_idx, idx]) for _, weights, _ in schedules)

    non_cc_sorted = sorted(non_cc_indices, key=lambda idx: score_by_idx[idx], reverse=True)

    cc_topics = []
    for i in range(0, len(cc_indices), 2):
        hi = cc_indices[i]
        lo = cc_indices[i + 1]
        cc_topics.append((hi, lo, max(score_by_idx[hi], score_by_idx[lo])))
    cc_topics.sort(key=lambda item: item[2], reverse=True)

    cc_sorted: list[int] = []
    for hi, lo, _ in cc_topics:
        cc_sorted.extend([hi, lo])
    return non_cc_sorted, cc_sorted


def _display_domain_label(domain_name: str) -> str:
    if domain_name.startswith("dolma3_cc/"):
        topic, quality = domain_name[len("dolma3_cc/") :].rsplit("_", maxsplit=1)
        return f"cc/{topic} {quality}"
    return domain_name.replace("dolma3_", "").replace("dolmino_", "")


def _format_epochs(epochs: float) -> str:
    if epochs < 0.01:
        return "0"
    if epochs >= 10.0:
        return f"{epochs:.1f}"
    if epochs >= 1.0:
        return f"{epochs:.1f}"
    if epochs >= 0.1:
        return f"{epochs:.2f}"
    return f"{epochs:.2f}"


def _plot_block(
    *,
    ax,
    indices: list[int],
    labels: list[str],
    schedules: list[tuple[str, np.ndarray, np.ndarray, tuple[float, float, float, float]]],
    phase_idx: int,
    multipliers: np.ndarray,
    title: str,
    show_legend: bool,
) -> None:
    y = np.arange(len(indices))
    bar_height = 0.23
    x_max = 0.0
    for _, weights, _ in schedules:
        x_max = max(x_max, float(np.max(weights[phase_idx, indices])))
    x_max = max(x_max * 1.45, 0.08)

    for schedule_idx, (label, weights, color) in enumerate(schedules):
        offsets = y + (schedule_idx - 1) * bar_height
        phase_weights = weights[phase_idx, indices]
        epochs = phase_weights * multipliers[indices]
        ax.barh(
            offsets,
            phase_weights,
            height=bar_height,
            color=color,
            alpha=0.95,
            edgecolor=BAR_EDGE_COLOR,
            linewidth=0.35,
            label=label if show_legend else None,
        )
        for offset, weight, epoch in zip(offsets, phase_weights, epochs, strict=True):
            epoch_label = _format_epochs(float(epoch))
            text_x = max(float(weight) + x_max * 0.012, x_max * 0.025)
            ax.text(
                text_x,
                float(offset),
                epoch_label,
                va="center",
                ha="left",
                fontsize=10.5,
                color=BAR_EDGE_COLOR,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 0.18},
            )

    ax.set_xlim(0.0, x_max)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=13.5)
    ax.invert_yaxis()
    ax.set_xlabel("Mixture Weight", fontsize=13.5)
    ax.set_title(title, fontsize=17, pad=10)
    ax.grid(axis="x", linestyle="--", color=GRID_COLOR, alpha=0.55, linewidth=0.8)
    ax.tick_params(axis="x", labelsize=12.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_color("#9ca3af")
    ax.spines["left"].set_color("#9ca3af")


def main() -> None:
    data = load_two_phase_many_packet(target=MANY_DOMAIN_TARGET)
    best_idx = int(np.argmin(data.y))
    best_name = str(data.frame.iloc[best_idx][data.name_col])
    best_weights = _row_weights(data.frame, data.domain_names, best_idx)

    ccglobal_summary = ccglobalpremium_retainedtotal_summary()
    ccpair_summary = ccpairtotal_retainedtotal_summary()
    ccglobal_weights = _summary_weights(ccglobal_summary, data.domain_names)
    ccpair_weights = _summary_weights(ccpair_summary, data.domain_names)
    schedules = [
        (
            "CCGlobalPremium-RetainedTotal",
            ccglobal_weights,
            "#0f766e",
        ),
        (
            "CCPairTotal-RetainedTotal",
            ccpair_weights,
            "#c2410c",
        ),
        (
            f"Best observed ({best_name})",
            best_weights,
            "#334155",
        ),
    ]

    phase0_non_cc_indices, phase0_cc_indices = _phase_order(
        domain_names=data.domain_names,
        phase_idx=0,
        schedules=schedules,
    )
    phase1_non_cc_indices, phase1_cc_indices = _phase_order(
        domain_names=data.domain_names,
        phase_idx=1,
        schedules=schedules,
    )
    phase0_non_cc_labels = [_display_domain_label(data.domain_names[idx]) for idx in phase0_non_cc_indices]
    phase0_cc_labels = [_display_domain_label(data.domain_names[idx]) for idx in phase0_cc_indices]
    phase1_non_cc_labels = [_display_domain_label(data.domain_names[idx]) for idx in phase1_non_cc_indices]
    phase1_cc_labels = [_display_domain_label(data.domain_names[idx]) for idx in phase1_cc_indices]

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(30, 24),
        gridspec_kw={"width_ratios": [1.0, 1.75], "hspace": 0.22, "wspace": 0.22},
        facecolor="white",
    )

    _plot_block(
        ax=axes[0, 0],
        indices=phase0_non_cc_indices,
        labels=phase0_non_cc_labels,
        schedules=schedules,
        phase_idx=0,
        multipliers=data.c0,
        title="Phase 0: Non-CC Domains",
        show_legend=True,
    )
    _plot_block(
        ax=axes[0, 1],
        indices=phase0_cc_indices,
        labels=phase0_cc_labels,
        schedules=schedules,
        phase_idx=0,
        multipliers=data.c0,
        title="Phase 0: CC Domains",
        show_legend=False,
    )
    _plot_block(
        ax=axes[1, 0],
        indices=phase1_non_cc_indices,
        labels=phase1_non_cc_labels,
        schedules=schedules,
        phase_idx=1,
        multipliers=data.c1,
        title="Phase 1: Non-CC Domains",
        show_legend=False,
    )
    _plot_block(
        ax=axes[1, 1],
        indices=phase1_cc_indices,
        labels=phase1_cc_labels,
        schedules=schedules,
        phase_idx=1,
        multipliers=data.c1,
        title="Phase 1: CC Domains",
        show_legend=False,
    )

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, fontsize=14, bbox_to_anchor=(0.5, 0.975))
    fig.suptitle(
        "Many-Domain Mixture Recipes for Uncheatable BPB",
        fontsize=22,
        y=0.992,
    )
    fig.text(
        0.5,
        0.955,
        "Bar-end labels show effective epochs for that domain in that phase. Values below 0.01 are displayed as 0.",
        ha="center",
        va="center",
        fontsize=13,
        color=TEXT_MUTED_COLOR,
    )
    fig.subplots_adjust(top=0.90, left=0.14, right=0.99, bottom=0.04, hspace=0.24, wspace=0.22)
    fig.savefig(PLOT_PNG, dpi=200, bbox_inches="tight")
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
