# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib", "numpy", "pandas"]
# ///
"""Plot GRP convergence for the power-family observed-only trustblend rule."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments.domain_phase_mix.exploratory.two_phase_many.dataset_metadata import (
    load_two_phase_many_candidate_summary_spec,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.surrogate_search.generic_family_followup import (
    load_generic_family_packet,
)
from experiments.domain_phase_mix.static_batch_selection import retrospective_generic_selection
from experiments.domain_phase_mix.two_phase_many_genericfamily_observed_only_trustblend_subset_optima import (
    OBJECTIVE_METRIC,
)
from experiments.domain_phase_mix.two_phase_many_genericfamily_power_family_observed_only_trustblend_baseline import (
    genericfamily_power_family_observed_only_trustblend_summary,
)
from experiments.domain_phase_mix import (
    two_phase_many_genericfamily_power_family_observed_only_trustblend_subset_optima as power_family_subset_optima,
)
from experiments.domain_phase_mix.two_phase_many_genericfamily_retuned_subset_optima import CSV_PATH

plt.rcParams["text.usetex"] = False

SCRIPT_DIR = Path(__file__).resolve().parent
CURVE_POINTS_CSV = (
    SCRIPT_DIR / "two_phase_many_grp_power_family_observed_only_trustblend_top8actual_cap_curve_points.csv"
)
SUMMARY_JSON = SCRIPT_DIR / "two_phase_many_grp_power_family_observed_only_trustblend_top8actual_cap_summary.json"
PLOT_PATH = SCRIPT_DIR / "two_phase_many_grp_power_family_observed_only_trustblend_top8actual_cap_convergence_tracks.png"


def _curve_points() -> pd.DataFrame:
    packet = load_generic_family_packet(target=OBJECTIVE_METRIC)
    _, spec, _ = load_two_phase_many_candidate_summary_spec(
        CSV_PATH,
        objective_metric=OBJECTIVE_METRIC,
        name="two_phase_many_grp_power_family_observed_only_trustblend_top8actual_cap",
    )
    rows = []
    for (
        summary
    ) in power_family_subset_optima.genericfamily_power_family_observed_only_trustblend_subset_optima_summaries():
        selection = retrospective_generic_selection(spec, method="feature_bayes_linear", k=summary.subset_size, seed=0)
        subset_indices = np.asarray(selection.selected_indices, dtype=int)
        subset_best_idx = int(subset_indices[np.argmin(packet.base.y[subset_indices])])
        rows.append(
            {
                "subset_size": summary.subset_size,
                "predicted_optimum_value": summary.predicted_optimum_value,
                "subset_best_observed_run_name": str(packet.base.frame.iloc[subset_best_idx][packet.base.name_col]),
                "subset_best_observed_bpb": float(packet.base.y[subset_best_idx]),
                "fullswarm_chosen_run_name": summary.fullswarm_chosen_run_name,
                "fullswarm_chosen_value": summary.fullswarm_chosen_value,
                "fullswarm_regret_at_1": summary.fullswarm_regret_at_1,
                "nearest_observed_run_name": summary.nearest_observed_run_name,
                "nearest_observed_value": summary.nearest_observed_value,
                "nearest_observed_tv_distance": summary.nearest_observed_tv_distance,
                "optimum_move_mean_phase_tv_vs_prev": summary.optimum_move_mean_phase_tv_vs_prev,
                "tuning_objective": summary.tuning_objective,
                "tuning_cv_rmse": summary.tuning_cv_rmse,
                "tuning_cv_regret_at_1": summary.tuning_cv_regret_at_1,
                "tuning_cv_foldmean_regret_at_1": summary.tuning_cv_foldmean_regret_at_1,
                "tuning_lower_tail_optimism": summary.tuning_lower_tail_optimism,
                "deployment_delta": summary.deployment_delta,
                "deployment_gain_budget": summary.deployment_gain_budget,
                "deployment_raw_predicted_optimum_value": summary.deployment_raw_predicted_optimum_value,
                "deployment_hull_predicted_optimum_value": summary.deployment_hull_predicted_optimum_value,
                "phase0_broad_text": summary.family_shares["phase0_broad_text"],
                "phase0_tech_code": summary.family_shares["phase0_tech_code"],
                "phase0_reasoning": summary.family_shares["phase0_reasoning"],
                "phase1_broad_text": summary.family_shares["phase1_broad_text"],
                "phase1_tech_code": summary.family_shares["phase1_tech_code"],
                "phase1_reasoning": summary.family_shares["phase1_reasoning"],
            }
        )

    frame = pd.DataFrame(rows)
    full_summary = genericfamily_power_family_observed_only_trustblend_summary()
    best_idx = int(np.argmin(packet.base.y))
    frame = pd.concat(
        [
            frame,
            pd.DataFrame(
                [
                    {
                        "subset_size": len(packet.base.y),
                        "predicted_optimum_value": full_summary.predicted_optimum_value,
                        "subset_best_observed_run_name": str(packet.base.frame.iloc[best_idx][packet.base.name_col]),
                        "subset_best_observed_bpb": float(packet.base.y[best_idx]),
                        "fullswarm_chosen_run_name": full_summary.fullswarm_chosen_run_name,
                        "fullswarm_chosen_value": full_summary.fullswarm_chosen_value,
                        "fullswarm_regret_at_1": full_summary.fullswarm_regret_at_1,
                        "nearest_observed_run_name": full_summary.nearest_observed_run_name,
                        "nearest_observed_value": full_summary.nearest_observed_value,
                        "nearest_observed_tv_distance": full_summary.nearest_observed_tv_distance,
                        "optimum_move_mean_phase_tv_vs_prev": np.nan,
                        "tuning_objective": full_summary.tuning_objective,
                        "tuning_cv_rmse": full_summary.tuning_cv_rmse,
                        "tuning_cv_regret_at_1": full_summary.tuning_cv_regret_at_1,
                        "tuning_cv_foldmean_regret_at_1": full_summary.tuning_cv_foldmean_regret_at_1,
                        "tuning_lower_tail_optimism": full_summary.tuning_lower_tail_optimism,
                        "deployment_delta": full_summary.deployment_delta,
                        "deployment_gain_budget": full_summary.deployment_gain_budget,
                        "deployment_raw_predicted_optimum_value": full_summary.deployment_raw_predicted_optimum_value,
                        "deployment_hull_predicted_optimum_value": full_summary.deployment_hull_predicted_optimum_value,
                        "phase0_broad_text": full_summary.family_shares["phase0_broad_text"],
                        "phase0_tech_code": full_summary.family_shares["phase0_tech_code"],
                        "phase0_reasoning": full_summary.family_shares["phase0_reasoning"],
                        "phase1_broad_text": full_summary.family_shares["phase1_broad_text"],
                        "phase1_tech_code": full_summary.family_shares["phase1_tech_code"],
                        "phase1_reasoning": full_summary.family_shares["phase1_reasoning"],
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    return frame.sort_values("subset_size").reset_index(drop=True)


def _plot(frame: pd.DataFrame) -> None:
    cmap = plt.colormaps["RdYlGn_r"]
    fig, (ax_bpb, ax_regret, ax_cvregret, ax_move) = plt.subplots(
        4,
        1,
        figsize=(10.2, 10.0),
        dpi=180,
        sharex=True,
        constrained_layout=True,
        gridspec_kw={"height_ratios": [1.3, 1.0, 1.0, 1.0], "hspace": 0.08},
    )

    ax_bpb.plot(
        frame["subset_size"],
        frame["predicted_optimum_value"],
        color=cmap(0.18),
        marker="o",
        linewidth=2.2,
        label="Predicted deployment BPB",
    )
    ax_bpb.plot(
        frame["subset_size"],
        frame["subset_best_observed_bpb"],
        color="#4C78A8",
        marker="P",
        linewidth=1.8,
        linestyle=":",
        label="Best observed BPB in subset",
    )
    ax_regret.plot(
        frame["subset_size"],
        frame["fullswarm_regret_at_1"],
        color=cmap(0.82),
        marker="s",
        linewidth=2.2,
        label="Retrospective Regret@1",
    )
    ax_cvregret.plot(
        frame["subset_size"],
        frame["tuning_cv_foldmean_regret_at_1"],
        color=cmap(0.68),
        marker="^",
        linewidth=2.2,
        label="CV Fold-Mean Regret@1",
    )
    ax_move.plot(
        frame["subset_size"],
        frame["optimum_move_mean_phase_tv_vs_prev"],
        color=cmap(0.36),
        marker="D",
        linewidth=2.2,
        label="Deployment movement (mean phase TV)",
    )
    ax_move.axhline(0.0, color="0.55", linewidth=1.0, linestyle=":")

    ax_bpb.set_title("Two-phase many-domain: GRP convergence (power-family observed-only trustblend)")
    ax_bpb.set_ylabel("BPB")
    ax_regret.set_ylabel("Regret@1")
    ax_cvregret.set_ylabel("CV Mean Regret@1")
    ax_move.set_ylabel("Mean phase TV")
    ax_move.set_xlabel("Observed runs used for fitting")
    ax_move.set_xticks(frame["subset_size"].tolist())
    ax_move.set_xlim(int(frame["subset_size"].min()), int(frame["subset_size"].max()))

    for axis in (ax_bpb, ax_regret, ax_cvregret, ax_move):
        axis.grid(True, alpha=0.25)
        handles = axis.get_lines()
        labels = [handle.get_label() for handle in handles if not handle.get_label().startswith("_")]
        if handles:
            axis.legend(handles, labels, loc="best", frameon=True)

    fig.savefig(PLOT_PATH, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    packet = load_generic_family_packet(target=OBJECTIVE_METRIC)
    frame = _curve_points()
    frame.to_csv(CURVE_POINTS_CSV, index=False)
    _plot(frame)
    SUMMARY_JSON.write_text(
        json.dumps(
            {
                "objective_metric": OBJECTIVE_METRIC,
                "curve_points_csv": str(CURVE_POINTS_CSV),
                "plot": str(PLOT_PATH),
                "best_observed_bpb": float(np.min(packet.base.y)),
                "rows": frame.replace({np.nan: None}).to_dict(orient="records"),
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
