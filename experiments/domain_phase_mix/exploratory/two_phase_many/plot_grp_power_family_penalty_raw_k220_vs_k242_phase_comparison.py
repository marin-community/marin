# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compare the k=220 and k=242 power-family-penalty raw-optimum mixtures."""

from __future__ import annotations

import json
from pathlib import Path

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

SCRIPT_DIR = Path(__file__).resolve().parent
SUMMARY_JSON = SCRIPT_DIR / "two_phase_many_grp_power_family_penalty_raw_summary.json"
PLOT_PNG = SCRIPT_DIR / "grp_power_family_penalty_raw_k220_vs_k242_phase_comparison.png"
WEIGHTS_CSV = SCRIPT_DIR / "grp_power_family_penalty_raw_k220_vs_k242_phase_comparison.csv"

K220 = 220
K242 = 242


def _load_rows_by_subset_size() -> dict[int, dict[str, object]]:
    payload = json.loads(SUMMARY_JSON.read_text())
    rows = payload["rows"]
    return {int(row["subset_size"]): row for row in rows}


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


def main() -> None:
    data = load_two_phase_many_packet(target=MANY_DOMAIN_TARGET)
    rows = _load_rows_by_subset_size()
    row220 = rows[K220]
    row242 = rows[K242]

    weights220 = _weights_array(row220["phase_weights"], data.domain_names)
    weights242 = _weights_array(row242["phase_weights"], data.domain_names)
    non_cc_indices, cc_indices = _grp_domain_order(data.domain_names, weights242)

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(24, 28),
        gridspec_kw={"width_ratios": [1.0, 1.65], "wspace": 0.30, "hspace": 0.35},
        facecolor="white",
    )

    row_specs = [
        (
            axes[0, 0],
            axes[0, 1],
            K220,
            row220,
            weights220,
            f"k = {K220} fit (validated BPB = {float(row220['actual_validated_bpb']):.4f})",
        ),
        (
            axes[1, 0],
            axes[1, 1],
            K242,
            row242,
            weights242,
            f"k = {K242} fit (validated BPB = {float(row242['actual_validated_bpb']):.4f})",
        ),
    ]
    for non_cc_ax, cc_ax, _, _, weights, row_title in row_specs:
        reference_plot._plot_non_cc_block(
            ax=non_cc_ax,
            indices=non_cc_indices,
            labels=[_display_non_cc_label(data.domain_names[idx]) for idx in non_cc_indices],
            weights=weights,
            phase0_multipliers=data.c0,
            phase1_multipliers=data.c1,
            title=f"{row_title}: Non-CC Domains",
        )
        reference_plot._plot_cc_block(
            ax=cc_ax,
            domain_names=data.domain_names,
            indices=cc_indices,
            weights=weights,
            phase0_multipliers=data.c0,
            phase1_multipliers=data.c1,
            title=f"{row_title}: CC Domains",
        )

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.suptitle(
        "Power-family penalty raw optimum: k=220 versus k=242",
        fontsize=32,
        y=0.988,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.957,
        "Same phase-comparison layout as the full-fit plot, using the validated k=220 and k=242 mixtures",
        ha="center",
        va="center",
        fontsize=19,
        color=reference_plot.TEXT_MUTED_COLOR,
    )
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, fontsize=18, bbox_to_anchor=(0.5, 0.94))

    phase0_tv = _phase_tv(weights220[0], weights242[0])
    phase1_tv = _phase_tv(weights220[1], weights242[1])
    fig.text(
        0.5,
        0.025,
        (f"Phase TV distance between k={K220} and k={K242}: " f"phase 0 = {phase0_tv:.3f}, phase 1 = {phase1_tv:.3f}"),
        ha="center",
        va="center",
        fontsize=14,
        color=reference_plot.TEXT_MUTED_COLOR,
    )
    fig.subplots_adjust(top=0.90, left=0.12, right=0.985, bottom=0.05, wspace=0.30, hspace=0.35)
    fig.savefig(PLOT_PNG, dpi=220, bbox_inches="tight")
    plt.close(fig)

    records: list[dict[str, object]] = []
    for domain_name, w220_p0, w220_p1, w242_p0, w242_p1, c0, c1 in zip(
        data.domain_names,
        weights220[0],
        weights220[1],
        weights242[0],
        weights242[1],
        data.c0,
        data.c1,
        strict=True,
    ):
        records.append(
            {
                "domain": domain_name,
                "k220_phase0_weight": float(w220_p0),
                "k220_phase0_epochs": float(w220_p0 * c0),
                "k220_phase1_weight": float(w220_p1),
                "k220_phase1_epochs": float(w220_p1 * c1),
                "k242_phase0_weight": float(w242_p0),
                "k242_phase0_epochs": float(w242_p0 * c0),
                "k242_phase1_weight": float(w242_p1),
                "k242_phase1_epochs": float(w242_p1 * c1),
            }
        )
    pd.DataFrame(records).to_csv(WEIGHTS_CSV, index=False)
    print(f"Plot: {PLOT_PNG}")
    print(f"Weights: {WEIGHTS_CSV}")


if __name__ == "__main__":
    main()
