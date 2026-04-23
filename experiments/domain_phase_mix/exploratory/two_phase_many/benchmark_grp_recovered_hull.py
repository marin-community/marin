# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib", "numpy", "pandas"]
# ///
"""Plot GRP convergence using the recovered convex-hull deployment procedure."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from experiments.domain_phase_mix.exploratory.two_phase_many.convergence_plot_style import (
    BEST_OBSERVED_BPB_COLOR,
    GRP_COLOR,
    PREDICTED_LINESTYLE,
)
from experiments.domain_phase_mix.two_phase_many_genericfamily_retuned_subset_optima import (
    GENERICFAMILY_RETUNED_SUBSET_OPTIMA_ALL_SUBSET_SIZES,
    OBJECTIVE_METRIC,
)
from experiments.domain_phase_mix.two_phase_many_genericfamily_recovered_hull_subset_optima import (
    genericfamily_recovered_hull_subset_optima_summaries_frame,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.surrogate_search.generic_family_followup import (
    load_generic_family_packet,
)

plt.rcParams["text.usetex"] = False

SCRIPT_DIR = Path(__file__).resolve().parent
CURVE_POINTS_CSV = SCRIPT_DIR / "two_phase_many_grp_recovered_hull_curve_points.csv"
SUMMARY_JSON = SCRIPT_DIR / "two_phase_many_grp_recovered_hull_summary.json"
PLOT_PATH = SCRIPT_DIR / "two_phase_many_grp_recovered_hull_convergence_tracks.png"


def _plot(frame: pd.DataFrame, *, best_observed_bpb: float) -> None:
    frame = frame.sort_values("subset_size")
    cmap = plt.colormaps["RdYlGn_r"]
    fig, (ax_bpb, ax_regret, ax_move) = plt.subplots(
        3,
        1,
        figsize=(10.2, 8.4),
        dpi=180,
        sharex=True,
        constrained_layout=True,
        gridspec_kw={"height_ratios": [1.3, 1.0, 1.0], "hspace": 0.08},
    )

    ax_bpb.plot(
        frame["subset_size"],
        frame["deployment_predicted_value"],
        color=GRP_COLOR,
        marker="o",
        linewidth=2.2,
        linestyle=PREDICTED_LINESTYLE,
        label="Recovered-hull predicted BPB",
    )
    ax_bpb.axhline(
        best_observed_bpb,
        color=BEST_OBSERVED_BPB_COLOR,
        linewidth=1.8,
        linestyle=":",
        label=f"Best observed BPB ({best_observed_bpb:.4f})",
    )
    ax_regret.plot(
        frame["subset_size"],
        frame["fullswarm_regret_at_1"],
        color=cmap(0.82),
        marker="s",
        linewidth=2.2,
        label="Retrospective Regret@1",
    )
    ax_move.plot(
        frame["subset_size"],
        frame["deployment_move_mean_phase_tv_vs_prev"],
        color=cmap(0.36),
        marker="D",
        linewidth=2.2,
        label="Deployment movement (mean phase TV)",
    )
    ax_move.axhline(0.0, color="0.55", linewidth=1.0, linestyle=":")

    ax_bpb.set_title("Two-phase many-domain: recovered-hull GRP convergence")
    ax_bpb.set_ylabel("Predicted BPB")
    ax_regret.set_ylabel("Regret@1")
    ax_move.set_ylabel("Mean phase TV")
    ax_move.set_xlabel("Observed runs used for fitting")
    ax_move.set_xticks(list(GENERICFAMILY_RETUNED_SUBSET_OPTIMA_ALL_SUBSET_SIZES))
    ax_move.set_xlim(
        min(GENERICFAMILY_RETUNED_SUBSET_OPTIMA_ALL_SUBSET_SIZES),
        max(GENERICFAMILY_RETUNED_SUBSET_OPTIMA_ALL_SUBSET_SIZES),
    )
    for axis in (ax_bpb, ax_regret, ax_move):
        axis.grid(True, alpha=0.25)
        handles = axis.get_lines()
        labels = [handle.get_label() for handle in handles if not handle.get_label().startswith("_")]
        if handles:
            axis.legend(handles, labels, loc="best", frameon=True)

    fig.savefig(PLOT_PATH, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    packet = load_generic_family_packet(target=OBJECTIVE_METRIC)
    best_observed_bpb = float(packet.base.y.min())
    curve_points = genericfamily_recovered_hull_subset_optima_summaries_frame(
        GENERICFAMILY_RETUNED_SUBSET_OPTIMA_ALL_SUBSET_SIZES
    ).rename(
        columns={
            "predicted_optimum_value": "deployment_predicted_value",
            "optimum_move_mean_phase_tv_vs_prev": "deployment_move_mean_phase_tv_vs_prev",
        }
    )
    curve_points.to_csv(CURVE_POINTS_CSV, index=False)
    _plot(curve_points, best_observed_bpb=best_observed_bpb)
    SUMMARY_JSON.write_text(
        json.dumps(
            {
                "objective_metric": OBJECTIVE_METRIC,
                "curve_points_csv": str(CURVE_POINTS_CSV),
                "plot": str(PLOT_PATH),
                "best_observed_bpb": best_observed_bpb,
                "rows": curve_points.to_dict(orient="records"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
