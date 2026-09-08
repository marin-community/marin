# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Plot the regularized and no-L2 GRP raw optima side by side."""

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

SCRIPT_DIR = Path(__file__).resolve().parent
PLOT_PNG = SCRIPT_DIR / "grp_power_family_penalty_raw_vs_no_l2_weights.png"
WEIGHTS_CSV = SCRIPT_DIR / "grp_power_family_penalty_raw_vs_no_l2_weights.csv"

REGULARIZED_VARIANT = "power_family_penalty"
NO_L2_VARIANT = "power_family_penalty_no_l2"
REGULARIZED_LABEL = "Regularized GRP"
NO_L2_LABEL = "No-$L_2$ GRP"
REGULARIZED_COLOR = reference_plot.GRP_COLOR
NO_L2_COLOR = reference_plot.PROPORTIONAL_COLOR
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


def _phase_tv(a: np.ndarray, b: np.ndarray) -> float:
    return 0.5 * float(np.abs(a - b).sum())


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
    value = payload.get("eval/uncheatable_eval/bpb")
    return None if value is None else float(value)


def _format_bpb(value: float | None) -> str:
    return "pending" if value is None else f"{value:.3f}"


def main() -> None:
    data = load_two_phase_many_packet(target=MANY_DOMAIN_TARGET)
    regularized_summary = genericfamily_penalty_raw_optimum_summary(REGULARIZED_VARIANT)
    no_l2_summary = genericfamily_penalty_raw_optimum_summary(NO_L2_VARIANT)

    regularized_weights = _weights_array(regularized_summary.phase_weights, data.domain_names)
    no_l2_weights = _weights_array(no_l2_summary.phase_weights, data.domain_names)

    schedules = [
        (REGULARIZED_LABEL, regularized_weights, REGULARIZED_COLOR),
        (NO_L2_LABEL, no_l2_weights, NO_L2_COLOR),
    ]
    canonical_non_cc_indices, canonical_cc_indices = reference_plot._grp_domain_order(
        data.domain_names,
        regularized_weights,
    )

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(26, 22),
        gridspec_kw={"width_ratios": [1.0, 1.62], "hspace": 0.22, "wspace": 0.31},
        facecolor="white",
    )

    reference_plot._plot_non_cc_block(
        ax=axes[0, 0],
        indices=canonical_non_cc_indices,
        labels=[reference_plot._display_non_cc_label(data.domain_names[idx]) for idx in canonical_non_cc_indices],
        schedules=schedules,
        phase_idx=0,
        multipliers=data.c0,
        title="Phase 0: Non-CC Domains",
        show_legend=True,
    )
    reference_plot._plot_cc_block(
        ax=axes[0, 1],
        domain_names=data.domain_names,
        indices=canonical_cc_indices,
        schedules=schedules,
        phase_idx=0,
        multipliers=data.c0,
        title="Phase 0: CC Domains",
    )
    reference_plot._plot_non_cc_block(
        ax=axes[1, 0],
        indices=canonical_non_cc_indices,
        labels=[reference_plot._display_non_cc_label(data.domain_names[idx]) for idx in canonical_non_cc_indices],
        schedules=schedules,
        phase_idx=1,
        multipliers=data.c1,
        title="Phase 1: Non-CC Domains",
        show_legend=False,
    )
    reference_plot._plot_cc_block(
        ax=axes[1, 1],
        domain_names=data.domain_names,
        indices=canonical_cc_indices,
        schedules=schedules,
        phase_idx=1,
        multipliers=data.c1,
        title="Phase 1: CC Domains",
    )

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.suptitle("GRP raw optimum: regularized vs no-$L_2$", fontsize=34, y=0.996, fontweight="bold")
    fig.text(
        0.5,
        0.952,
        "Many-domain two-phase mixture weights for uncheatable-eval BPB",
        ha="center",
        va="center",
        fontsize=20,
        color=reference_plot.TEXT_MUTED_COLOR,
    )
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, fontsize=18.5, bbox_to_anchor=(0.5, 0.928))

    regularized_validated = _validated_bpb(regularized_summary.run_name)
    no_l2_validated = _validated_bpb(no_l2_summary.run_name)
    phase0_tv = _phase_tv(regularized_weights[0], no_l2_weights[0])
    phase1_tv = _phase_tv(regularized_weights[1], no_l2_weights[1])
    fig.text(
        0.5,
        0.072,
        (
            f"Regularized GRP: pred {regularized_summary.raw_predicted_optimum_value:.3f}, "
            f"validated {_format_bpb(regularized_validated)}   |   "
            f"No-$L_2$ GRP: pred {no_l2_summary.raw_predicted_optimum_value:.3f}, "
            f"validated {_format_bpb(no_l2_validated)}\n"
            f"Phase TV distance: phase 0 = {phase0_tv:.3f}, phase 1 = {phase1_tv:.3f}"
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
        color=reference_plot.TEXT_MUTED_COLOR,
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
