# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib", "pandas"]
# ///
"""Plot a direct comparison of observed-only GRP deployment variants."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

DETAIL_CSV = Path(__file__).resolve().parent / "two_phase_many_grp_deployment_variant_curve_points.csv"
SUMMARY_JSON = Path(__file__).resolve().parent / "two_phase_many_grp_deployment_variant_summary.json"
PLOT_PATH = Path(__file__).resolve().parent / "two_phase_many_grp_deployment_variants.png"

plt.rcParams["text.usetex"] = False

VARIANT_ORDER = [
    "all_observed_hull",
    "top16_actual_hull",
    "top8_actual_hull",
    "top4_actual_hull",
    "all_hull_disp0.01",
]
VARIANT_LABELS = {
    "all_observed_hull": "All observed hull",
    "top16_actual_hull": "Top-16 actual hull",
    "top8_actual_hull": "Top-8 actual hull",
    "top4_actual_hull": "Top-4 actual hull",
    "all_hull_disp0.01": "All hull + dispersion",
}


def main() -> None:
    frame = pd.read_csv(DETAIL_CSV).sort_values(["variant", "subset_size"])
    summary = json.loads(SUMMARY_JSON.read_text())
    best_observed_bpb = 1.0571987628936768
    cmap = plt.colormaps["RdYlGn_r"]
    color_positions = {
        "top8_actual_hull": 0.14,
        "top16_actual_hull": 0.26,
        "all_observed_hull": 0.38,
        "top4_actual_hull": 0.58,
        "all_hull_disp0.01": 0.76,
    }

    fig, (ax_bpb, ax_regret, ax_cvregret, ax_move) = plt.subplots(
        4,
        1,
        figsize=(10.8, 10.8),
        dpi=180,
        sharex=True,
        constrained_layout=True,
        gridspec_kw={"height_ratios": [1.3, 1.0, 1.0, 1.0], "hspace": 0.08},
    )

    for variant in VARIANT_ORDER:
        variant_frame = frame[frame["variant"] == variant].sort_values("subset_size")
        color = cmap(color_positions[variant])
        line_width = 2.8 if variant == "top8_actual_hull" else 1.9
        alpha = 1.0 if variant == "top8_actual_hull" else 0.95
        label = VARIANT_LABELS[variant]

        ax_bpb.plot(
            variant_frame["subset_size"],
            variant_frame["predicted_optimum_value"],
            color=color,
            marker="o",
            linewidth=line_width,
            alpha=alpha,
            label=label,
        )
        ax_regret.plot(
            variant_frame["subset_size"],
            variant_frame["fullswarm_regret_at_1"],
            color=color,
            marker="s",
            linewidth=line_width,
            alpha=alpha,
            label=label,
        )
        ax_cvregret.plot(
            variant_frame["subset_size"],
            variant_frame["tuning_cv_foldmean_regret_at_1"],
            color=color,
            marker="^",
            linewidth=line_width,
            alpha=alpha,
            label=label,
        )
        ax_move.plot(
            variant_frame["subset_size"],
            variant_frame["move_mean_phase_tv_vs_prev"],
            color=color,
            marker="D",
            linewidth=line_width,
            alpha=alpha,
            label=label,
        )

    ax_bpb.axhline(
        best_observed_bpb,
        color="0.6",
        linewidth=1.4,
        linestyle=":",
        label=f"Best observed BPB ({best_observed_bpb:.4f})",
    )
    ax_move.axhline(0.0, color="0.55", linewidth=1.0, linestyle=":")

    ax_bpb.set_title("Two-phase many-domain: GRP deployment regularizer comparison")
    ax_bpb.set_ylabel("Predicted BPB")
    ax_regret.set_ylabel("Retrospective Regret@1")
    ax_cvregret.set_ylabel("CV Fold-Mean Regret@1")
    ax_move.set_ylabel("Mean phase TV")
    ax_move.set_xlabel("Observed runs used for fitting")

    subset_sizes = sorted(frame["subset_size"].unique())
    ax_move.set_xticks(subset_sizes)
    ax_move.set_xlim(min(subset_sizes), max(subset_sizes))

    for axis in (ax_bpb, ax_regret, ax_cvregret, ax_move):
        axis.grid(True, alpha=0.25)

    ax_bpb.legend(loc="upper left", ncol=2, frameon=True)
    ax_regret.legend(loc="upper right", ncol=2, frameon=True)
    ax_cvregret.legend(loc="upper right", ncol=2, frameon=True)
    ax_move.legend(loc="upper left", ncol=2, frameon=True)

    top8_summary = summary["variants"]["top8_actual_hull"]
    all_summary = summary["variants"]["all_observed_hull"]
    fig.text(
        0.5,
        0.003,
        (
            "Top-8 actual frontier hull is the current validation target: "
            "same zero-Regret@1 regime from k>=80, "
            f"near-tied mean predicted BPB after k>=80 ({top8_summary['mean_predicted_value_after80']:.4f} vs "
            f"{all_summary['mean_predicted_value_after80']:.4f} for the full hull), "
            f"with lower movement ({top8_summary['mean_move_after80']:.3f} vs {all_summary['mean_move_after80']:.3f})."
        ),
        ha="center",
        va="bottom",
        fontsize=9.2,
    )

    fig.savefig(PLOT_PATH, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
