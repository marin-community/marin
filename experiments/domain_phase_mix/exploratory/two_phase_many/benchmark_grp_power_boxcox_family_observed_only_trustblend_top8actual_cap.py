# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib", "numpy", "pandas"]
# ///
"""Plot GRP convergence for the mixed power/Box-Cox observed-only trustblend rule."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments.domain_phase_mix.exploratory.two_phase_many.convergence_plot_style import (
    BEST_OBSERVED_BPB_COLOR,
    GRP_COLOR,
    PREDICTED_LINESTYLE,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.dataset_metadata import (
    load_two_phase_many_candidate_summary_spec,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.surrogate_search.generic_family_flexible_signal import (
    build_flexible_signal_surrogate,
    deploy_flexible_signal_gaincapped_topkactual,
    flexible_signal_params_from_metrics,
    tune_flexible_signal_params_observed_only,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.surrogate_search.generic_family_followup import (
    family_shares,
    load_generic_family_packet,
)
from experiments.domain_phase_mix.static_batch_selection import retrospective_generic_selection
from experiments.domain_phase_mix.two_phase_many_genericfamily_observed_only_trustblend_subset_optima import (
    DEFAULT_TUNING_METHOD,
    GENERICFAMILY_OBSERVED_ONLY_TRUSTBLEND_SUBSET_OPTIMA_REPRESENTATIVE_SUBSET_SIZES,
    OBJECTIVE_METRIC,
    TRUSTBLEND_TOPK_ACTUAL,
)
from experiments.domain_phase_mix.two_phase_many_genericfamily_retuned_subset_optima import (
    CSV_PATH,
    _mean_phase_tv_distance,
    _subset_packet,
    _top_domains,
)

plt.rcParams["text.usetex"] = False

SCRIPT_DIR = Path(__file__).resolve().parent
VARIANT_NAME = "power_boxcox_family"
VARIANT_LABEL = "mixed power-singleton / Box-Cox-family"
CURVE_POINTS_CSV = (
    SCRIPT_DIR / "two_phase_many_grp_power_boxcox_family_observed_only_trustblend_top8actual_cap_curve_points.csv"
)
SUMMARY_JSON = SCRIPT_DIR / "two_phase_many_grp_power_boxcox_family_observed_only_trustblend_top8actual_cap_summary.json"
PLOT_PATH = (
    SCRIPT_DIR / "two_phase_many_grp_power_boxcox_family_observed_only_trustblend_top8actual_cap_convergence_tracks.png"
)


def _fullswarm_summary(
    full_packet,
    train_packet,
    *,
    previous_deployment: np.ndarray | None = None,
) -> tuple[dict[str, Any], np.ndarray]:
    _, _, tuning_metrics, _ = tune_flexible_signal_params_observed_only(
        train_packet,
        variant_name=VARIANT_NAME,
        method=DEFAULT_TUNING_METHOD,
        coarse_top_k=4,
        seed=0,
    )
    tuned_params = flexible_signal_params_from_metrics(tuning_metrics, VARIANT_NAME)
    model = build_flexible_signal_surrogate(
        train_packet,
        params=tuned_params,
        variant_name=VARIANT_NAME,
    ).fit(train_packet.base.w, train_packet.base.y)
    deployment = deploy_flexible_signal_gaincapped_topkactual(
        train_packet,
        model,
        tuning_metrics,
        top_k=TRUSTBLEND_TOPK_ACTUAL,
    )
    weights = np.asarray(deployment["weights"], dtype=float)
    fullswarm_predictions = model.predict(full_packet.base.w)
    chosen_idx = int(np.argmin(fullswarm_predictions))
    best_idx = int(np.argmin(full_packet.base.y))
    distances = 0.5 * np.abs(full_packet.base.w - weights[None, :, :]).sum(axis=2).mean(axis=1)
    nearest_idx = int(np.argmin(distances))
    hull_coeffs = np.asarray(deployment["hull_coefficients"], dtype=float)
    coeff_order = np.argsort(hull_coeffs)[::-1]
    top_indices = list(deployment["hull_top_indices"])
    hull_anchor_summaries = [
        {
            "run_name": str(train_packet.base.frame.iloc[top_indices[idx]][train_packet.base.name_col]),
            "actual_value": float(train_packet.base.y[top_indices[idx]]),
            "coefficient": float(hull_coeffs[idx]),
        }
        for idx in coeff_order
    ]
    row = {
        "tuning_objective": float(tuning_metrics["objective"]),
        "tuning_cv_rmse": float(tuning_metrics["cv_rmse"]),
        "tuning_cv_regret_at_1": float(tuning_metrics["cv_regret_at_1"]),
        "tuning_cv_foldmean_regret_at_1": float(tuning_metrics["cv_foldmean_regret_at_1"]),
        "tuning_lower_tail_optimism": float(tuning_metrics["lower_tail_optimism"]),
        "tuning_success": bool(tuning_metrics["success"]),
        "tuning_message": str(tuning_metrics["message"]),
        "tuned_params": tuned_params,
        "predicted_optimum_value": float(deployment["predicted_optimum_value"]),
        "deployment_delta": float(deployment["delta"]),
        "deployment_realized_gain": float(deployment["realized_gain"]),
        "deployment_gain_budget": float(deployment["gain_budget"]),
        "deployment_raw_predicted_optimum_value": float(deployment["raw_predicted_optimum_value"]),
        "deployment_hull_predicted_optimum_value": float(deployment["hull_predicted_optimum_value"]),
        "fullswarm_chosen_run_name": str(full_packet.base.frame.iloc[chosen_idx][full_packet.base.name_col]),
        "fullswarm_chosen_value": float(full_packet.base.y[chosen_idx]),
        "fullswarm_regret_at_1": float(full_packet.base.y[chosen_idx] - full_packet.base.y[best_idx]),
        "observed_best_run_name": str(full_packet.base.frame.iloc[best_idx][full_packet.base.name_col]),
        "observed_best_value": float(full_packet.base.y[best_idx]),
        "gap_below_observed_best": float(deployment["predicted_optimum_value"] - full_packet.base.y[best_idx]),
        "nearest_observed_run_name": str(full_packet.base.frame.iloc[nearest_idx][full_packet.base.name_col]),
        "nearest_observed_value": float(full_packet.base.y[nearest_idx]),
        "nearest_observed_tv_distance": float(distances[nearest_idx]),
        "optimum_move_mean_phase_tv_vs_prev": (
            None if previous_deployment is None else _mean_phase_tv_distance(weights, previous_deployment)
        ),
        "optimizer_success": bool(deployment["optimizer_success"]),
        "optimizer_message": str(deployment["optimizer_message"]),
        "phase0_max_weight": float(np.max(weights[0])),
        "phase1_max_weight": float(np.max(weights[1])),
        "phase0_support_below_1e4": int(np.sum(weights[0] < 1e-4)),
        "phase1_support_below_1e4": int(np.sum(weights[1] < 1e-4)),
        "phase0_top_domains": _top_domains(full_packet.base.domain_names, weights[0], full_packet.base.c0, top_k=8),
        "phase1_top_domains": _top_domains(full_packet.base.domain_names, weights[1], full_packet.base.c1, top_k=8),
        "family_shares": family_shares(full_packet, weights),
        "hull_anchor_count": len(top_indices),
        "hull_anchor_summaries": hull_anchor_summaries,
    }
    return row, weights


def _curve_points() -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    full_packet = load_generic_family_packet(target=OBJECTIVE_METRIC)
    _, spec, _ = load_two_phase_many_candidate_summary_spec(
        CSV_PATH,
        objective_metric=OBJECTIVE_METRIC,
        name="two_phase_many_grp_power_boxcox_family_observed_only_trustblend_top8actual_cap",
    )

    rows: list[dict[str, Any]] = []
    detailed_rows: list[dict[str, Any]] = []
    previous_deployment: np.ndarray | None = None
    for subset_size in GENERICFAMILY_OBSERVED_ONLY_TRUSTBLEND_SUBSET_OPTIMA_REPRESENTATIVE_SUBSET_SIZES:
        selection = retrospective_generic_selection(spec, method="feature_bayes_linear", k=subset_size, seed=0)
        subset_indices = np.asarray(selection.selected_indices, dtype=int)
        train_packet = _subset_packet(full_packet, subset_indices)
        summary_row, weights = _fullswarm_summary(
            full_packet,
            train_packet,
            previous_deployment=previous_deployment,
        )
        previous_deployment = weights
        subset_best_idx = int(subset_indices[np.argmin(full_packet.base.y[subset_indices])])
        rows.append(
            {
                "subset_size": subset_size,
                "predicted_optimum_value": summary_row["predicted_optimum_value"],
                "subset_best_observed_run_name": str(
                    full_packet.base.frame.iloc[subset_best_idx][full_packet.base.name_col]
                ),
                "subset_best_observed_bpb": float(full_packet.base.y[subset_best_idx]),
                "fullswarm_chosen_run_name": summary_row["fullswarm_chosen_run_name"],
                "fullswarm_chosen_value": summary_row["fullswarm_chosen_value"],
                "fullswarm_regret_at_1": summary_row["fullswarm_regret_at_1"],
                "nearest_observed_run_name": summary_row["nearest_observed_run_name"],
                "nearest_observed_value": summary_row["nearest_observed_value"],
                "nearest_observed_tv_distance": summary_row["nearest_observed_tv_distance"],
                "optimum_move_mean_phase_tv_vs_prev": summary_row["optimum_move_mean_phase_tv_vs_prev"],
                "tuning_objective": summary_row["tuning_objective"],
                "tuning_cv_rmse": summary_row["tuning_cv_rmse"],
                "tuning_cv_regret_at_1": summary_row["tuning_cv_regret_at_1"],
                "tuning_cv_foldmean_regret_at_1": summary_row["tuning_cv_foldmean_regret_at_1"],
                "tuning_lower_tail_optimism": summary_row["tuning_lower_tail_optimism"],
                "deployment_delta": summary_row["deployment_delta"],
                "deployment_gain_budget": summary_row["deployment_gain_budget"],
                "deployment_raw_predicted_optimum_value": summary_row["deployment_raw_predicted_optimum_value"],
                "deployment_hull_predicted_optimum_value": summary_row["deployment_hull_predicted_optimum_value"],
                "phase0_broad_text": summary_row["family_shares"]["phase0_broad_text"],
                "phase0_tech_code": summary_row["family_shares"]["phase0_tech_code"],
                "phase0_reasoning": summary_row["family_shares"]["phase0_reasoning"],
                "phase1_broad_text": summary_row["family_shares"]["phase1_broad_text"],
                "phase1_tech_code": summary_row["family_shares"]["phase1_tech_code"],
                "phase1_reasoning": summary_row["family_shares"]["phase1_reasoning"],
            }
        )
        detailed_rows.append({"subset_size": subset_size, **summary_row})

    full_summary, _ = _fullswarm_summary(full_packet, full_packet)
    best_idx = int(np.argmin(full_packet.base.y))
    rows.append(
        {
            "subset_size": len(full_packet.base.y),
            "predicted_optimum_value": full_summary["predicted_optimum_value"],
            "subset_best_observed_run_name": str(full_packet.base.frame.iloc[best_idx][full_packet.base.name_col]),
            "subset_best_observed_bpb": float(full_packet.base.y[best_idx]),
            "fullswarm_chosen_run_name": full_summary["fullswarm_chosen_run_name"],
            "fullswarm_chosen_value": full_summary["fullswarm_chosen_value"],
            "fullswarm_regret_at_1": full_summary["fullswarm_regret_at_1"],
            "nearest_observed_run_name": full_summary["nearest_observed_run_name"],
            "nearest_observed_value": full_summary["nearest_observed_value"],
            "nearest_observed_tv_distance": full_summary["nearest_observed_tv_distance"],
            "optimum_move_mean_phase_tv_vs_prev": np.nan,
            "tuning_objective": full_summary["tuning_objective"],
            "tuning_cv_rmse": full_summary["tuning_cv_rmse"],
            "tuning_cv_regret_at_1": full_summary["tuning_cv_regret_at_1"],
            "tuning_cv_foldmean_regret_at_1": full_summary["tuning_cv_foldmean_regret_at_1"],
            "tuning_lower_tail_optimism": full_summary["tuning_lower_tail_optimism"],
            "deployment_delta": full_summary["deployment_delta"],
            "deployment_gain_budget": full_summary["deployment_gain_budget"],
            "deployment_raw_predicted_optimum_value": full_summary["deployment_raw_predicted_optimum_value"],
            "deployment_hull_predicted_optimum_value": full_summary["deployment_hull_predicted_optimum_value"],
            "phase0_broad_text": full_summary["family_shares"]["phase0_broad_text"],
            "phase0_tech_code": full_summary["family_shares"]["phase0_tech_code"],
            "phase0_reasoning": full_summary["family_shares"]["phase0_reasoning"],
            "phase1_broad_text": full_summary["family_shares"]["phase1_broad_text"],
            "phase1_tech_code": full_summary["family_shares"]["phase1_tech_code"],
            "phase1_reasoning": full_summary["family_shares"]["phase1_reasoning"],
        }
    )
    detailed_rows.append({"subset_size": len(full_packet.base.y), **full_summary})
    frame = pd.DataFrame(rows).sort_values("subset_size").reset_index(drop=True)
    return frame, detailed_rows


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
        color=GRP_COLOR,
        marker="o",
        linewidth=2.2,
        linestyle=PREDICTED_LINESTYLE,
        label="Predicted deployment BPB",
    )
    ax_bpb.plot(
        frame["subset_size"],
        frame["subset_best_observed_bpb"],
        color=BEST_OBSERVED_BPB_COLOR,
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

    ax_bpb.set_title(f"Two-phase many-domain: GRP convergence ({VARIANT_LABEL})")
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
    frame, detailed_rows = _curve_points()
    frame.to_csv(CURVE_POINTS_CSV, index=False)
    _plot(frame)
    SUMMARY_JSON.write_text(
        json.dumps(
            {
                "variant_name": VARIANT_NAME,
                "variant_label": VARIANT_LABEL,
                "objective_metric": OBJECTIVE_METRIC,
                "curve_points_csv": str(CURVE_POINTS_CSV),
                "plot": str(PLOT_PATH),
                "best_observed_bpb": float(np.min(packet.base.y)),
                "rows": frame.replace({np.nan: None}).to_dict(orient="records"),
                "detailed_rows": detailed_rows,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
