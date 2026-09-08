# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Plot the full power-family-penalty raw optimum against the k=220 raw optimum."""

from __future__ import annotations

import json
from pathlib import Path

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
    genericfamily_penalty_raw_optimum_summary,
)

SCRIPT_DIR = Path(__file__).resolve().parent
SUMMARY_JSON = SCRIPT_DIR / "two_phase_many_grp_power_family_penalty_raw_summary.json"
PLOT_PNG = SCRIPT_DIR / "grp_power_family_penalty_raw_vs_k220_weights.png"
WEIGHTS_CSV = SCRIPT_DIR / "grp_power_family_penalty_raw_vs_k220_weights.csv"

VARIANT_NAME = "power_family_penalty"
FULL_LABEL = "Full fit (k=242)"
K220_LABEL = "Subset fit (k=220)"
FULL_COLOR = reference_plot.GRP_COLOR
K220_COLOR = reference_plot.PROPORTIONAL_COLOR


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


def _summary_rows_by_subset_size() -> dict[int, dict[str, object]]:
    return {int(row["subset_size"]): row for row in json.loads(SUMMARY_JSON.read_text())["rows"]}


def main() -> None:
    data = load_two_phase_many_packet(target=MANY_DOMAIN_TARGET)
    full_summary = genericfamily_penalty_raw_optimum_summary(VARIANT_NAME)
    summary_rows = _summary_rows_by_subset_size()
    row220 = summary_rows[220]

    full_weights = _weights_array(full_summary.phase_weights, data.domain_names)
    k220_weights = _weights_array(row220["phase_weights"], data.domain_names)

    schedules = [
        (FULL_LABEL, full_weights, FULL_COLOR),
        (K220_LABEL, k220_weights, K220_COLOR),
    ]
    canonical_non_cc_indices, canonical_cc_indices = reference_plot._grp_domain_order(data.domain_names, full_weights)

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
    fig.suptitle("Power-family penalty raw optimum: full fit vs k=220 fit", fontsize=34, y=0.996, fontweight="bold")
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

    full_validated = float(summary_rows[242]["actual_validated_bpb"])
    k220_validated = float(row220["actual_validated_bpb"])
    phase0_tv = _phase_tv(full_weights[0], k220_weights[0])
    phase1_tv = _phase_tv(full_weights[1], k220_weights[1])
    fig.text(
        0.5,
        0.072,
        (
            "Full fit (k=242): "
            f"pred {full_summary.raw_predicted_optimum_value:.4f}, validated {full_validated:.4f}   |   "
            f"k=220 fit: pred {float(row220['predicted_optimum_value']):.4f}, validated {k220_validated:.4f}\n"
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
