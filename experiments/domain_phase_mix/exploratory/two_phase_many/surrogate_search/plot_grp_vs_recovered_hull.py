# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Plot published GRP deployment against the recovered convex-hull deployment."""

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
from experiments.domain_phase_mix.two_phase_many_genericfamily_recovered_hull_baseline import (
    genericfamily_recovered_hull_summary,
)
from experiments.domain_phase_mix.two_phase_many_genericfamily_tuned_baseline import (
    genericfamily_tuned_summary,
)

plt.rcParams["text.usetex"] = False

SCRIPT_DIR = Path(__file__).resolve().parent
PLOT_PNG = SCRIPT_DIR / "grp_vs_recovered_hull_weights.png"
WEIGHTS_CSV = SCRIPT_DIR / "grp_vs_recovered_hull_weights.csv"
RECOVERED_HULL_COLOR = "#3faa59"

assert CC_TOPIC_DISPLAY
assert BAR_EDGE_COLOR


def main() -> None:
    data = load_two_phase_many_packet(target=MANY_DOMAIN_TARGET)
    grp_summary = genericfamily_tuned_summary()
    recovered_summary = genericfamily_recovered_hull_summary()
    grp_weights = comparison_plot._summary_weights(grp_summary, data.domain_names)
    recovered_weights = comparison_plot._summary_weights(recovered_summary, data.domain_names)

    schedules = [
        ("Published GRP", grp_weights, GRP_COLOR),
        ("Recovered hull", recovered_weights, RECOVERED_HULL_COLOR),
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
    fig.suptitle("Published GRP vs Recovered Hull Deployment", fontsize=34, y=0.996, fontweight="bold")
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
            f"Published GRP: predicted {float(grp_summary['predicted_optimum_value']):.4f} BPB, "
            f"realized 1.0403   |   "
            f"Recovered hull: predicted {float(recovered_summary['predicted_optimum_value']):.4f} BPB, "
            f"TV to published GRP {0.5 * float(np.mean(np.sum(np.abs(recovered_weights - grp_weights), axis=1))):.4f}\n"
            f"Recovered anchor blend: best {recovered_summary['anchor_coefficients']['best_observed']:.3f}, "
            f"global {recovered_summary['anchor_coefficients']['validated_global']:.3f}, "
            f"pair {recovered_summary['anchor_coefficients']['validated_pair']:.3f}, "
            f"prop {recovered_summary['anchor_coefficients']['baseline_proportional']:.3f}"
        ),
        ha="center",
        va="center",
        fontsize=16.6,
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
        "Recovered hull is a convex combination of anchor recipes, not a raw unconstrained simplex optimum.",
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
