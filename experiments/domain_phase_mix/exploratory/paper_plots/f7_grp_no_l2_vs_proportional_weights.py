# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.11"
# dependencies = ["fsspec", "matplotlib", "numpy", "pandas"]
# ///
"""Paper plot for GRP no-L2 raw optimum weights versus proportional."""

from __future__ import annotations

import json
from pathlib import Path

import fsspec
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from experiments.domain_phase_mix.exploratory.two_phase_many.surrogate_search import (
    plot_grp_vs_proportional as reference_plot,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.surrogate_search.structured_epoch_family import (
    MANY_DOMAIN_TARGET,
    load_two_phase_many_packet,
)
from experiments.domain_phase_mix.two_phase_many_genericfamily_penalty_raw_optima_baselines import (
    GENERICFAMILY_PENALTY_RAW_OPTIMA_SOURCE_EXPERIMENT,
    genericfamily_penalty_raw_optimum_summary,
)

PAPER_ROOT = Path(__file__).resolve().parent
IMG_DIR = PAPER_ROOT / "img"
OUTPUT_STEM = IMG_DIR / "f7_grp_no_l2_vs_proportional_weights"
OUTPUT_CSV = IMG_DIR / "f7_grp_no_l2_vs_proportional_weights_points.csv"

NO_L2_VARIANT = "power_family_penalty_no_l2"
NO_L2_LABEL = "GRP no-$L_2$"
PROPORTIONAL_LABEL = "Proportional"
NO_L2_COLOR = reference_plot.GRP_COLOR
PROPORTIONAL_COLOR = reference_plot.PROPORTIONAL_COLOR
CHECKPOINT_ROOT = f"marin-us-east5/checkpoints/{GENERICFAMILY_PENALTY_RAW_OPTIMA_SOURCE_EXPERIMENT}"


def _weights_array(
    phase_weights: dict[str, dict[str, float]],
    domain_names: list[str],
) -> np.ndarray:
    return np.asarray(
        [
            [float(phase_weights["phase_0"][domain_name]) for domain_name in domain_names],
            [float(phase_weights["phase_1"][domain_name]) for domain_name in domain_names],
        ],
        dtype=float,
    )


def _row_weights(frame: pd.DataFrame, domain_names: list[str], row_idx: int) -> np.ndarray:
    return np.asarray(
        [
            [float(frame.iloc[row_idx][f"phase_0_{domain_name}"]) for domain_name in domain_names],
            [float(frame.iloc[row_idx][f"phase_1_{domain_name}"]) for domain_name in domain_names],
        ],
        dtype=float,
    )


def _validated_bpb(run_name: str) -> float | None:
    fs = fsspec.filesystem("gs")
    matches = sorted(fs.glob(f"{CHECKPOINT_ROOT}/{run_name}-*/checkpoints/eval_metrics.jsonl"))
    if not matches:
        return None
    payload: dict[str, float] | None = None
    with fs.open(matches[-1], "r") as handle:
        for line in handle:
            if line.strip():
                payload = json.loads(line)
    if payload is None:
        return None
    value = payload.get(MANY_DOMAIN_TARGET)
    return None if value is None else float(value)


def _format_bpb(value: float | None) -> str:
    return "pending" if value is None else f"{value:.3f}"


def main() -> None:
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    data = load_two_phase_many_packet(target=MANY_DOMAIN_TARGET)
    no_l2_summary = genericfamily_penalty_raw_optimum_summary(NO_L2_VARIANT)
    no_l2_weights = _weights_array(no_l2_summary.phase_weights, data.domain_names)

    proportional_idx = int(data.frame.index[data.frame["run_name"] == "baseline_proportional"][0])
    proportional_weights = _row_weights(data.frame, data.domain_names, proportional_idx)
    proportional_bpb = float(data.frame.iloc[proportional_idx][MANY_DOMAIN_TARGET])
    no_l2_validated = _validated_bpb(no_l2_summary.run_name)

    schedules = [
        (NO_L2_LABEL, no_l2_weights, NO_L2_COLOR),
        (PROPORTIONAL_LABEL, proportional_weights, PROPORTIONAL_COLOR),
    ]
    non_cc_indices, cc_indices = reference_plot._grp_domain_order(data.domain_names, no_l2_weights)

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(24.0, 18.0),
        gridspec_kw={"width_ratios": [1.0, 1.62], "hspace": 0.21, "wspace": 0.31},
        facecolor="white",
    )

    reference_plot._plot_non_cc_block(
        ax=axes[0, 0],
        indices=non_cc_indices,
        labels=[reference_plot._display_non_cc_label(data.domain_names[idx]) for idx in non_cc_indices],
        schedules=schedules,
        phase_idx=0,
        multipliers=data.c0,
        title="Phase 0: non-CC domains",
        show_legend=True,
    )
    reference_plot._plot_cc_block(
        ax=axes[0, 1],
        domain_names=data.domain_names,
        indices=cc_indices,
        schedules=schedules,
        phase_idx=0,
        multipliers=data.c0,
        title="Phase 0: CC domains",
    )
    reference_plot._plot_non_cc_block(
        ax=axes[1, 0],
        indices=non_cc_indices,
        labels=[reference_plot._display_non_cc_label(data.domain_names[idx]) for idx in non_cc_indices],
        schedules=schedules,
        phase_idx=1,
        multipliers=data.c1,
        title="Phase 1: non-CC domains",
        show_legend=False,
    )
    reference_plot._plot_cc_block(
        ax=axes[1, 1],
        domain_names=data.domain_names,
        indices=cc_indices,
        schedules=schedules,
        phase_idx=1,
        multipliers=data.c1,
        title="Phase 1: CC domains",
    )

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, fontsize=18.5, bbox_to_anchor=(0.5, 0.985))
    fig.text(
        0.5,
        0.048,
        (
            f"GRP no-$L_2$: predicted {no_l2_summary.raw_predicted_optimum_value:.3f}, "
            f"validated {_format_bpb(no_l2_validated)} BPB   |   "
            f"Proportional: {proportional_bpb:.3f} BPB"
        ),
        ha="center",
        va="center",
        fontsize=17.0,
        color="#0f172a",
        bbox={
            "boxstyle": "round,pad=0.52,rounding_size=0.16",
            "facecolor": "#f8fafc",
            "edgecolor": "#cbd5e1",
            "alpha": 0.97,
        },
    )
    fig.text(
        0.5,
        0.018,
        "Bar-end labels show effective epochs for that domain in that phase. Values below 0.01 are displayed as 0.",
        ha="center",
        va="center",
        fontsize=14.0,
        color=reference_plot.TEXT_MUTED_COLOR,
    )
    fig.subplots_adjust(top=0.94, left=0.14, right=0.985, bottom=0.085, hspace=0.24, wspace=0.31)
    fig.savefig(OUTPUT_STEM.with_suffix(".png"), dpi=240, bbox_inches="tight")
    fig.savefig(OUTPUT_STEM.with_suffix(".pdf"), bbox_inches="tight")
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
    pd.DataFrame(rows).to_csv(OUTPUT_CSV, index=False)
    print(f"Wrote {OUTPUT_STEM.with_suffix('.png')}")
    print(f"Wrote {OUTPUT_STEM.with_suffix('.pdf')}")
    print(f"Wrote {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
