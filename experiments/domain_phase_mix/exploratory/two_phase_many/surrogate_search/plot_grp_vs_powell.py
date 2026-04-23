# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Plot current GRP against all-data L-BFGS-B and Powell GRP optima."""

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
from experiments.domain_phase_mix.exploratory.two_phase_many.surrogate_search.plot_grp_vs_proportional import (
    BAR_EDGE_COLOR,
    CC_TOPIC_DISPLAY,
    GRP_COLOR,
    PROPORTIONAL_COLOR,
    TEXT_MUTED_COLOR,
    _display_non_cc_label,
    _grp_domain_order,
    _plot_cc_block,
    _plot_non_cc_block,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.surrogate_search.structured_epoch_family import (
    MANY_DOMAIN_TARGET,
    load_two_phase_many_packet,
)
from experiments.domain_phase_mix.two_phase_many_genericfamily_lbfgsb_baseline import (
    genericfamily_lbfgsb_summary,
)
from experiments.domain_phase_mix.two_phase_many_genericfamily_powell_baseline import (
    genericfamily_powell_summary,
)
from experiments.domain_phase_mix.two_phase_many_genericfamily_tuned_baseline import (
    genericfamily_tuned_summary,
)

plt.rcParams["text.usetex"] = False

SCRIPT_DIR = Path(__file__).resolve().parent
PLOT_PNG = SCRIPT_DIR / "grp_vs_powell_weights.png"
WEIGHTS_CSV = SCRIPT_DIR / "grp_vs_powell_weights.csv"
LBFGSB_COLOR = PROPORTIONAL_COLOR
POWELL_COLOR = "#e54e35"

# Avoid import pruning complaints for style constants/helpers re-exported from the sibling module.
assert CC_TOPIC_DISPLAY
assert BAR_EDGE_COLOR


def main() -> None:
    data = load_two_phase_many_packet(target=MANY_DOMAIN_TARGET)
    grp_summary = genericfamily_tuned_summary()
    lbfgsb_summary = genericfamily_lbfgsb_summary()
    powell_summary = genericfamily_powell_summary()
    grp_weights = comparison_plot._summary_weights(grp_summary, data.domain_names)
    lbfgsb_weights = comparison_plot._summary_weights(lbfgsb_summary, data.domain_names)
    powell_weights = comparison_plot._summary_weights(powell_summary, data.domain_names)

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
    best_row = original_ranked[original_ranked["run_name"] == best_name].iloc[0]
    grp_realized = 1.0403348207473755
    grp_would_rank = int(1 + np.sum(original_ranked[MANY_DOMAIN_TARGET].to_numpy() < grp_realized))
    lbfgsb_predicted = float(lbfgsb_summary["predicted_optimum_value"])
    lbfgsb_chosen = float(lbfgsb_summary["fullswarm_chosen_value"])
    lbfgsb_regret = float(lbfgsb_summary["fullswarm_regret_at_1"])
    lbfgsb_would_rank = int(1 + np.sum(original_ranked[MANY_DOMAIN_TARGET].to_numpy() < lbfgsb_chosen))
    powell_predicted = float(powell_summary["predicted_optimum_value"])
    powell_chosen = float(powell_summary["fullswarm_chosen_value"])
    powell_regret = float(powell_summary["fullswarm_regret_at_1"])
    powell_would_rank = int(1 + np.sum(original_ranked[MANY_DOMAIN_TARGET].to_numpy() < powell_chosen))

    schedules = [
        ("GRP", grp_weights, GRP_COLOR),
        ("L-BFGS-B all-data optimum", lbfgsb_weights, LBFGSB_COLOR),
        ("Powell all-data optimum", powell_weights, POWELL_COLOR),
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
    fig.suptitle("Current GRP vs All-Data Retuned Optima", fontsize=34, y=0.996, fontweight="bold")
    fig.text(
        0.5,
        0.952,
        "Many-domain two-phase mixture weights for uncheatable-eval BPB",
        ha="center",
        va="center",
        fontsize=20,
        color=TEXT_MUTED_COLOR,
    )
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, fontsize=17.0, bbox_to_anchor=(0.5, 0.928))
    fig.text(
        0.5,
        0.072,
        (
            f"Current GRP (realized): {grp_realized:.4f} BPB, "
            f"would place {grp_would_rank}st vs the original 241-run swarm   |   "
            f"L-BFGS-B optimum (predicted): {lbfgsb_predicted:.4f} BPB, "
            f"chosen run {lbfgsb_summary['fullswarm_chosen_run_name']} = {lbfgsb_chosen:.4f} BPB, "
            f"regret {lbfgsb_regret:.4f}, would place {lbfgsb_would_rank}st/241\n"
            f"Powell optimum (predicted): {powell_predicted:.4f} BPB, "
            f"chosen run {powell_summary['fullswarm_chosen_run_name']} = {powell_chosen:.4f} BPB, "
            f"regret {powell_regret:.4f}, would place {powell_would_rank}st/241   |   "
            f"Best observed: {best_bpb:.4f} BPB ({best_name}), rank {int(best_row['rank'])}/241"
        ),
        ha="center",
        va="center",
        fontsize=16.4,
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
