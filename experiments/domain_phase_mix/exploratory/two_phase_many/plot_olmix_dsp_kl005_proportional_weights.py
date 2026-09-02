# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["cvxpy", "matplotlib", "numpy", "pandas", "plotly", "scipy"]
# ///
"""Plot OLMix cap-4, DSP KL-only, and proportional 300M mixtures."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    fit_olmix_reference_deletion_augmented_300m as olmix,
)

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "reference_outputs" / "dsp_l2_kl_sweep_deletion_augmented_300m_20260625"
OLMIX_HUBER_DELTA = 0.01
OLMIX_KL_REG = 0.05
OLMIX_CAP = 4
DSP_KL_REG = 0.05
OLMIX_DIR = SCRIPT_DIR / "reference_outputs" / "olmix_huber_delta_sweep_300m_20260625" / "delta_0p01"
PLOT_PNG = OUTPUT_DIR / "olmix_delta0p01_cap4_vs_dsp_kl005_vs_proportional_weights.png"
WEIGHTS_CSV = OUTPUT_DIR / "olmix_delta0p01_cap4_vs_dsp_kl005_vs_proportional_weights.csv"

BAR_EDGE_COLOR = "#0f172a"
GRID_COLOR = "#cbd5e1"
TEXT_MUTED_COLOR = "#475569"
OLMIX_COLOR = "#232b32"
DSP_COLOR = "#2f855a"
PROPORTIONAL_COLOR = "#dcd0bb"
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


def read_weights(path: Path, domains: list[str]) -> np.ndarray:
    frame = pd.read_csv(path).set_index("domain")
    missing = sorted(set(domains).difference(frame.index))
    if missing:
        raise ValueError(f"Missing domains in {path}: {missing[:5]}")
    weights = frame.loc[domains, ["phase_0_weight", "phase_1_weight"]].to_numpy(dtype=float).T
    weights = np.clip(weights, 0.0, None)
    return weights / weights.sum(axis=1, keepdims=True)


def split_domain_blocks(domain_names: list[str]) -> tuple[list[int], list[int]]:
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


def format_epochs(epochs: float) -> str:
    if epochs < 0.01:
        return "0"
    if epochs >= 1.0:
        return f"{epochs:.1f}"
    return f"{epochs:.2f}"


def display_non_cc_label(domain_name: str) -> str:
    label = domain_name.removeprefix("dolma3_").removeprefix("dolmino_")
    return label.replace("_", " ")


def cc_topic_and_quality(domain_name: str) -> tuple[str, str]:
    topic, quality = domain_name[len("dolma3_cc/") :].rsplit("_", maxsplit=1)
    topic_display = topic.replace("_", " ")
    return CC_TOPIC_DISPLAY.get(topic_display, topic_display), quality


def block_order(domain_names: list[str], schedules: list[tuple[str, np.ndarray, str]]) -> tuple[list[int], list[int]]:
    non_cc_indices, cc_indices = split_domain_blocks(domain_names)
    non_cc_sorted = sorted(
        non_cc_indices,
        key=lambda idx: max(float(np.max(weights[:, idx])) for _, weights, _ in schedules),
        reverse=True,
    )

    cc_topics: list[tuple[str, int, int]] = []
    for i in range(0, len(cc_indices), 2):
        hi = cc_indices[i]
        lo = cc_indices[i + 1]
        topic, _ = cc_topic_and_quality(domain_names[hi])
        cc_topics.append((topic, hi, lo))
    cc_topics.sort(key=lambda item: item[0])
    cc_sorted: list[int] = []
    for _, hi, lo in cc_topics:
        cc_sorted.extend([hi, lo])
    return non_cc_sorted, cc_sorted


def block_xlim(schedules: list[tuple[str, np.ndarray, str]], *, phase_idx: int, indices: list[int]) -> float:
    x_max = max(float(np.max(weights[phase_idx, indices])) for _, weights, _ in schedules)
    return max(x_max * 1.10 + 0.006, 0.05)


def plot_non_cc_block(
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
    bar_height = 0.23
    x_max = block_xlim(schedules, phase_idx=phase_idx, indices=indices)
    center = (len(schedules) - 1) / 2.0
    for schedule_idx, (label, weights, color) in enumerate(schedules):
        offsets = y + (schedule_idx - center) * bar_height
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
            epoch_label = format_epochs(float(epoch))
            text_pad = max(0.0022, x_max * 0.0055)
            text_x = max(float(weight) + text_pad, x_max * 0.018)
            ax.text(
                text_x,
                float(offset),
                epoch_label,
                va="center",
                ha="left",
                fontsize=11.2,
                color=BAR_EDGE_COLOR,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.88, "pad": 0.12},
            )

    ax.set_xlim(0.0, x_max)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=18.5, fontweight="bold")
    ax.invert_yaxis()
    ax.set_xlabel("Mixture Weight", fontsize=16.0, fontweight="bold")
    ax.set_title(title, fontsize=22, pad=12, fontweight="bold")
    ax.grid(axis="x", linestyle="--", color=GRID_COLOR, alpha=0.65, linewidth=0.85)
    ax.tick_params(axis="x", labelsize=14)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_color("#94a3b8")
    ax.spines["left"].set_color("#94a3b8")


def plot_cc_block(
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
    bar_height = 0.23
    x_max = block_xlim(schedules, phase_idx=phase_idx, indices=indices)
    center = (len(schedules) - 1) / 2.0
    for schedule_idx, (_label, weights, color) in enumerate(schedules):
        offsets = y + (schedule_idx - center) * bar_height
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
            epoch_label = format_epochs(float(epoch))
            text_pad = max(0.0010, x_max * 0.0023)
            text_x = max(float(weight) + text_pad, x_max * 0.014)
            ax.text(text_x, float(offset), epoch_label, va="center", ha="left", fontsize=10.7, color=BAR_EDGE_COLOR)

    ax.set_xlim(0.0, x_max)
    ax.set_yticks(y)
    ax.set_yticklabels([""] * len(indices))
    ax.invert_yaxis()
    ax.set_xlabel("Mixture Weight", fontsize=16.0, fontweight="bold")
    ax.set_title(title, fontsize=22, pad=12, fontweight="bold")
    ax.grid(axis="x", linestyle="--", color=GRID_COLOR, alpha=0.65, linewidth=0.85)
    ax.tick_params(axis="x", labelsize=14)
    ax.tick_params(axis="y", length=0)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_color("#94a3b8")
    ax.spines["left"].set_color("#94a3b8")

    for pair_start in range(0, len(indices), 2):
        high_idx = indices[pair_start]
        low_idx = indices[pair_start + 1]
        topic_high, quality_high = cc_topic_and_quality(domain_names[high_idx])
        topic_low, quality_low = cc_topic_and_quality(domain_names[low_idx])
        assert topic_high == topic_low
        assert quality_high == "high"
        assert quality_low == "low"
        mid_y = pair_start + 0.5
        ax.text(
            -0.078,
            mid_y,
            topic_high,
            transform=ax.get_yaxis_transform(),
            ha="right",
            va="center",
            fontsize=17.2,
            fontweight="bold",
            color=BAR_EDGE_COLOR,
            clip_on=False,
        )
        ax.text(
            -0.020,
            pair_start,
            "high",
            transform=ax.get_yaxis_transform(),
            ha="right",
            va="center",
            fontsize=12.7,
            fontweight="bold",
            color=TEXT_MUTED_COLOR,
            clip_on=False,
        )
        ax.text(
            -0.020,
            pair_start + 1,
            "low",
            transform=ax.get_yaxis_transform(),
            ha="right",
            va="center",
            fontsize=12.7,
            fontweight="bold",
            color=TEXT_MUTED_COLOR,
            clip_on=False,
        )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _signal, _columns, domains, natural = olmix.load_raw_signal_panel()
    target_budget = olmix.load_target_budget()
    token_counts = olmix.load_domain_token_counts(domains)
    phase_multipliers = olmix.PHASE_FRACTIONS[:, None] * float(target_budget) / token_counts[None, :]

    olmix_weights = read_weights(OLMIX_DIR / "uncheatable_eval_bpb_rep_cap4" / "proposed_mixture_weights.csv", domains)
    dsp_weights = read_weights(OUTPUT_DIR / "dsp_l2_0.0001_kl_only_0.05" / "proposed_mixture_weights.csv", domains)
    proportional_weights = np.stack([natural, natural], axis=0)
    olmix_label = f"OLMix delta={OLMIX_HUBER_DELTA:g}, KL={OLMIX_KL_REG:g}, cap={OLMIX_CAP}"
    dsp_label = f"DSP KL={DSP_KL_REG:g}, no cap"
    schedules = [
        (olmix_label, olmix_weights, OLMIX_COLOR),
        (dsp_label, dsp_weights, DSP_COLOR),
        ("Proportional", proportional_weights, PROPORTIONAL_COLOR),
    ]

    non_cc_indices, cc_indices = block_order(domains, schedules)

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(27, 22),
        gridspec_kw={"width_ratios": [1.0, 1.62], "hspace": 0.23, "wspace": 0.31},
        facecolor="white",
    )
    plot_non_cc_block(
        ax=axes[0, 0],
        indices=non_cc_indices,
        labels=[display_non_cc_label(domains[idx]) for idx in non_cc_indices],
        schedules=schedules,
        phase_idx=0,
        multipliers=phase_multipliers[0],
        title="Phase 0: Non-CC Domains",
        show_legend=True,
    )
    plot_cc_block(
        ax=axes[0, 1],
        domain_names=domains,
        indices=cc_indices,
        schedules=schedules,
        phase_idx=0,
        multipliers=phase_multipliers[0],
        title="Phase 0: CC Domains",
    )
    plot_non_cc_block(
        ax=axes[1, 0],
        indices=non_cc_indices,
        labels=[display_non_cc_label(domains[idx]) for idx in non_cc_indices],
        schedules=schedules,
        phase_idx=1,
        multipliers=phase_multipliers[1],
        title="Phase 1: Non-CC Domains",
        show_legend=False,
    )
    plot_cc_block(
        ax=axes[1, 1],
        domain_names=domains,
        indices=cc_indices,
        schedules=schedules,
        phase_idx=1,
        multipliers=phase_multipliers[1],
        title="Phase 1: CC Domains",
    )

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.suptitle(
        "Uncheatable BPB proposals: tuned OLMix vs DSP",
        fontsize=33,
        y=0.996,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.952,
        "300M deletion-augmented fit panel; OLMix shown with Huber delta 0.01, KL coefficient 0.05, and cap-4 aggregate exposure",
        ha="center",
        va="center",
        fontsize=19.5,
        color=TEXT_MUTED_COLOR,
    )
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, fontsize=17.0, bbox_to_anchor=(0.5, 0.918))

    olmix_summary = pd.read_csv(OLMIX_DIR / "summary.csv")
    olmix_row = olmix_summary.loc[olmix_summary["target_name"].eq("uncheatable_eval_bpb_rep_cap4")].iloc[0]
    dsp_summary = pd.read_csv(OUTPUT_DIR / "dsp_kl_only_proposal_sweep_summary.csv")
    dsp_row = dsp_summary.loc[np.isclose(dsp_summary["kl_reg"], 0.05)].iloc[0]
    prop_bpb = float(olmix_row["proportional_actual"])
    fig.text(
        0.5,
        0.073,
        (
            f"OLMix delta={OLMIX_HUBER_DELTA:g}, KL={OLMIX_KL_REG:g}, cap={OLMIX_CAP}: "
            f"pred {float(olmix_row['predicted_objective']):.3f}, max epoch {float(olmix_row['max_simulated_epoch']):.1f}   |   "
            f"DSP KL={DSP_KL_REG:g}, no cap: pred {float(dsp_row['predicted_objective']):.3f}, "
            f"max epoch {float(dsp_row['max_simulated_epoch']):.2f}   |   "
            f"Proportional: observed mean {prop_bpb:.3f}"
        ),
        ha="center",
        va="center",
        fontsize=17.0,
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
        fontsize=14.5,
        color=TEXT_MUTED_COLOR,
    )
    fig.subplots_adjust(top=0.875, left=0.14, right=0.985, bottom=0.13, hspace=0.24, wspace=0.31)
    fig.savefig(PLOT_PNG, dpi=220, bbox_inches="tight")
    plt.close(fig)

    rows = []
    for label, weights, _color in schedules:
        for domain, phase0_weight, phase1_weight, c0, c1 in zip(
            domains,
            weights[0],
            weights[1],
            phase_multipliers[0],
            phase_multipliers[1],
            strict=True,
        ):
            rows.append(
                {
                    "schedule": label,
                    "domain": domain,
                    "phase0_weight": float(phase0_weight),
                    "phase0_epochs": float(phase0_weight * c0),
                    "phase1_weight": float(phase1_weight),
                    "phase1_epochs": float(phase1_weight * c1),
                }
            )
    pd.DataFrame(rows).to_csv(WEIGHTS_CSV, index=False)
    print(json.dumps({"plot": str(PLOT_PNG), "weights": str(WEIGHTS_CSV)}, indent=2))


if __name__ == "__main__":
    main()
