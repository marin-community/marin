# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Plot the Uniform and Proportional baseline weights side by side."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from experiments.domain_phase_mix.exploratory.two_phase_many.surrogate_search import (
    plot_ccpairtotal_vs_ccglobalpremium_vs_best as comparison_plot,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.surrogate_search import (
    plot_grp_vs_proportional as reference_plot,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.surrogate_search.structured_epoch_family import (
    MANY_DOMAIN_TARGET,
    load_two_phase_many_packet,
)

SCRIPT_DIR = Path(__file__).resolve().parent
PLOT_PNG = SCRIPT_DIR / "uniform_vs_proportional_weights.png"
WEIGHTS_CSV = SCRIPT_DIR / "uniform_vs_proportional_weights.csv"
INPUT_60M_CSV = SCRIPT_DIR / "two_phase_many_all_60m_1p2b.csv"
INPUT_300M_CSV = SCRIPT_DIR / "qsplit240_300m_6b_completed_vs_60m.csv"

UNIFORM_RUN_NAME = "baseline_stratified"
PROPORTIONAL_RUN_NAME = "baseline_proportional"
UNIFORM_LABEL = "Uniform"
PROPORTIONAL_LABEL = "Proportional"
UNIFORM_COLOR = "#6C6F7D"
PROPORTIONAL_COLOR = reference_plot.PROPORTIONAL_COLOR


def _row_weights(frame: pd.DataFrame, domain_names: list[str], run_name: str) -> np.ndarray:
    row_idx = int(frame.index[frame["run_name"] == run_name][0])
    return comparison_plot._row_weights(frame, domain_names, row_idx)


def _phase_tv(a: np.ndarray, b: np.ndarray) -> float:
    return 0.5 * float(np.abs(a - b).sum())


def _load_60m_bpb(run_name: str) -> float:
    frame = pd.read_csv(INPUT_60M_CSV)
    row = frame.loc[frame["run_name"] == run_name].iloc[0]
    return float(row["eval/uncheatable_eval/bpb"])


def _load_300m_bpb(run_name: str) -> float:
    frame = pd.read_csv(INPUT_300M_CSV)
    row = frame.loc[frame["run_name"] == run_name].iloc[0]
    return float(row["bpb_300m_6b"])


def main() -> None:
    data = load_two_phase_many_packet(target=MANY_DOMAIN_TARGET)
    uniform_weights = _row_weights(data.frame, data.domain_names, UNIFORM_RUN_NAME)
    proportional_weights = _row_weights(data.frame, data.domain_names, PROPORTIONAL_RUN_NAME)

    schedules = [
        (UNIFORM_LABEL, uniform_weights, UNIFORM_COLOR),
        (PROPORTIONAL_LABEL, proportional_weights, PROPORTIONAL_COLOR),
    ]
    canonical_non_cc_indices, canonical_cc_indices = reference_plot._grp_domain_order(
        data.domain_names,
        uniform_weights,
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
    fig.suptitle("Uniform vs Proportional baselines", fontsize=34, y=0.996, fontweight="bold")
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

    uniform_60m = _load_60m_bpb(UNIFORM_RUN_NAME)
    proportional_60m = _load_60m_bpb(PROPORTIONAL_RUN_NAME)
    uniform_300m = _load_300m_bpb(UNIFORM_RUN_NAME)
    proportional_300m = _load_300m_bpb(PROPORTIONAL_RUN_NAME)
    phase0_tv = _phase_tv(uniform_weights[0], proportional_weights[0])
    phase1_tv = _phase_tv(uniform_weights[1], proportional_weights[1])

    fig.text(
        0.5,
        0.072,
        (
            f"Uniform: 60M {uniform_60m:.3f}, 300M {uniform_300m:.3f}   |   "
            f"Proportional: 60M {proportional_60m:.3f}, 300M {proportional_300m:.3f}\n"
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
