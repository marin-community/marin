# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Plot the power-family-penalty raw optimum against the proportional baseline."""

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
    plot_ccpairtotal_vs_ccglobalpremium_vs_best as comparison_plot,
)
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
PLOT_PNG = SCRIPT_DIR / "grp_power_family_penalty_raw_vs_proportional_weights.png"
WEIGHTS_CSV = SCRIPT_DIR / "grp_power_family_penalty_raw_vs_proportional_weights.csv"

VARIANT_NAME = "power_family_penalty"
RUN_NAME = "baseline_genericfamily_power_family_penalty_raw_optimum"
CHECKPOINT_ROOT = "marin-us-east5/checkpoints/" + GENERICFAMILY_PENALTY_RAW_OPTIMA_SOURCE_EXPERIMENT
RAW_COLOR = reference_plot.GRP_COLOR
PROPORTIONAL_COLOR = reference_plot.PROPORTIONAL_COLOR


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
    raw_weights = _weights_array(summary.phase_weights, data.domain_names)
    validated_bpb = _validated_bpb()

    proportional_idx = int(data.frame.index[data.frame["run_name"] == "baseline_proportional"][0])
    proportional_weights = comparison_plot._row_weights(data.frame, data.domain_names, proportional_idx)

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
    proportional_row = original_ranked[original_ranked["run_name"] == "baseline_proportional"].iloc[0]
    best_row = original_ranked[original_ranked["run_name"] == best_name].iloc[0]
    raw_would_rank = int(1 + np.sum(original_ranked[MANY_DOMAIN_TARGET].to_numpy() < float(validated_bpb or np.inf)))

    schedules = [
        ("Power-family penalty raw", raw_weights, RAW_COLOR),
        ("Proportional", proportional_weights, PROPORTIONAL_COLOR),
    ]

    canonical_non_cc_indices, canonical_cc_indices = reference_plot._grp_domain_order(data.domain_names, raw_weights)

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
    fig.suptitle("Power-family penalty raw optimum vs Proportional", fontsize=34, y=0.996, fontweight="bold")
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
    validated_text = "pending" if validated_bpb is None else f"{validated_bpb:.4f}"
    raw_rank_text = "--" if validated_bpb is None else f"{raw_would_rank}st"
    fig.text(
        0.5,
        0.072,
        (
            f"Power-family penalty raw: pred {summary.raw_predicted_optimum_value:.4f}, "
            f"validated {validated_text}, would rank {raw_rank_text} vs the original 242-run swarm\n"
            "Proportional: "
            f"{float(proportional_row[MANY_DOMAIN_TARGET]):.4f} BPB, rank {int(proportional_row['rank'])}/242   |   "
            f"Best observed: {best_bpb:.4f} BPB ({best_name}), rank {int(best_row['rank'])}/242"
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
