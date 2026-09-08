# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Plot power-family-penalty raw-optimum weights with phase-0 versus phase-1 bars."""

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
    plot_grp_phase_comparison as reference_plot,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.surrogate_search.plot_grp_vs_proportional import (
    _display_non_cc_label,
    _grp_domain_order,
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
PLOT_PNG = SCRIPT_DIR / "grp_power_family_penalty_raw_phase_comparison.png"
WEIGHTS_CSV = SCRIPT_DIR / "grp_power_family_penalty_raw_phase_comparison.csv"

VARIANT_NAME = "power_family_penalty"
RUN_NAME = "baseline_genericfamily_power_family_penalty_raw_optimum"
CHECKPOINT_ROOT = "marin-us-east5/checkpoints/" + GENERICFAMILY_PENALTY_RAW_OPTIMA_SOURCE_EXPERIMENT


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


def _validated_bpb() -> float | None:
    fs = fsspec.filesystem("gs")
    matches = sorted(fs.glob(f"{CHECKPOINT_ROOT}/{RUN_NAME}-*/checkpoints/eval_metrics.jsonl"))
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


def main() -> None:
    data = load_two_phase_many_packet(target=MANY_DOMAIN_TARGET)
    summary = genericfamily_penalty_raw_optimum_summary(VARIANT_NAME)
    weights = _weights_array(summary.phase_weights, data.domain_names)
    validated_bpb = _validated_bpb()

    non_cc_indices, cc_indices = _grp_domain_order(data.domain_names, weights)

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(24, 15),
        gridspec_kw={"width_ratios": [1.0, 1.65], "wspace": 0.30},
        facecolor="white",
    )

    reference_plot._plot_non_cc_block(
        ax=axes[0],
        indices=non_cc_indices,
        labels=[_display_non_cc_label(data.domain_names[idx]) for idx in non_cc_indices],
        weights=weights,
        phase0_multipliers=data.c0,
        phase1_multipliers=data.c1,
        title="Non-CC Domains",
    )
    reference_plot._plot_cc_block(
        ax=axes[1],
        domain_names=data.domain_names,
        indices=cc_indices,
        weights=weights,
        phase0_multipliers=data.c0,
        phase1_multipliers=data.c1,
        title="CC Domains",
    )

    handles, labels = axes[0].get_legend_handles_labels()
    fig.suptitle("Power-family penalty raw optimum: Phase 0 vs Phase 1", fontsize=32, y=0.985, fontweight="bold")
    fig.text(
        0.5,
        0.947,
        "Many-domain two-phase mixture weights for uncheatable-eval BPB",
        ha="center",
        va="center",
        fontsize=19,
        color=reference_plot.TEXT_MUTED_COLOR,
    )
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, fontsize=18, bbox_to_anchor=(0.5, 0.925))
    validated_text = "pending" if validated_bpb is None else f"{validated_bpb:.4f}"
    fig.text(
        0.5,
        0.07,
        (
            f"Predicted raw BPB: {summary.raw_predicted_optimum_value:.4f}   |   "
            f"Validated raw BPB: {validated_text}   |   "
            f"Nearest observed TV: {summary.nearest_observed_tv_distance:.3f}"
        ),
        ha="center",
        va="center",
        fontsize=14,
        color=reference_plot.TEXT_MUTED_COLOR,
    )
    fig.subplots_adjust(top=0.88, left=0.12, right=0.985, bottom=0.10, wspace=0.30)
    fig.savefig(PLOT_PNG, dpi=220, bbox_inches="tight")
    plt.close(fig)

    rows = []
    for domain_name, phase0_weight, phase1_weight, c0, c1 in zip(
        data.domain_names, weights[0], weights[1], data.c0, data.c1, strict=True
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
