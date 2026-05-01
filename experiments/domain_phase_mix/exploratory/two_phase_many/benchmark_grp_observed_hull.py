# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib", "numpy", "pandas"]
# ///
"""Plot GRP convergence with observed-only convex-hull deployment."""

from __future__ import annotations

import json
from pathlib import Path

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
from experiments.domain_phase_mix.exploratory.two_phase_many.surrogate_search.generic_family_followup import (
    GenericFamilyRetainedTotalSurrogate,
    load_generic_family_packet,
    optimize_generic_family_convex_hull,
)
from experiments.domain_phase_mix.two_phase_many_ccglobalpremium_baselines import (
    ccglobalpremium_retainedtotal_summary,
)
from experiments.domain_phase_mix.two_phase_many_ccpairtotal_baseline import (
    ccpairtotal_retainedtotal_summary,
)
from experiments.domain_phase_mix.two_phase_many_genericfamily_retuned_subset_optima import (
    CSV_PATH,
    GENERICFAMILY_RETUNED_SUBSET_OPTIMA_ALL_SUBSET_SIZES,
    OBJECTIVE_METRIC,
    VALIDATED_GLOBAL_BPB,
    VALIDATED_PAIR_BPB,
    _mean_phase_tv_distance,
    _summary_weights,
    _subset_packet,
    tune_genericfamily_subset_params,
)
from experiments.domain_phase_mix.static_batch_selection import retrospective_generic_selection

plt.rcParams["text.usetex"] = False

SCRIPT_DIR = Path(__file__).resolve().parent
CURVE_POINTS_CSV = SCRIPT_DIR / "two_phase_many_grp_observed_hull_curve_points.csv"
SUMMARY_JSON = SCRIPT_DIR / "two_phase_many_grp_observed_hull_summary.json"
PLOT_PATH = SCRIPT_DIR / "two_phase_many_grp_observed_hull_convergence_tracks.png"
HULL_START_COUNT = 8


def _curve_points() -> pd.DataFrame:
    _, spec, _ = load_two_phase_many_candidate_summary_spec(
        CSV_PATH,
        objective_metric=OBJECTIVE_METRIC,
        name="two_phase_many_grp_observed_hull",
    )
    packet = load_generic_family_packet(target=OBJECTIVE_METRIC)
    valid_weights = np.stack(
        [
            _summary_weights(ccglobalpremium_retainedtotal_summary(), packet.base.domain_names),
            _summary_weights(ccpairtotal_retainedtotal_summary(), packet.base.domain_names),
        ],
        axis=0,
    )
    valid_y = np.asarray([VALIDATED_GLOBAL_BPB, VALIDATED_PAIR_BPB], dtype=float)
    best_full_idx = int(np.argmin(packet.base.y))
    best_observed_bpb = float(packet.base.y[best_full_idx])
    previous_deployment: np.ndarray | None = None
    rows: list[dict[str, object]] = []

    for subset_size in GENERICFAMILY_RETUNED_SUBSET_OPTIMA_ALL_SUBSET_SIZES:
        selection = retrospective_generic_selection(spec, method="feature_bayes_linear", k=subset_size, seed=0)
        subset_indices = np.asarray(selection.selected_indices, dtype=int)
        train_packet = _subset_packet(packet, subset_indices)
        tuning_metrics, _ = tune_genericfamily_subset_params(train_packet, valid_weights, valid_y)
        tuned_params = {key: float(tuning_metrics[key]) for key in ("alpha", "eta", "lam", "tau", "reg", "beta")}
        model = GenericFamilyRetainedTotalSurrogate(train_packet, params=tuned_params).fit(
            train_packet.base.w,
            train_packet.base.y,
        )
        subset_predictions = model.predict(train_packet.base.w)
        start_count = min(HULL_START_COUNT, len(subset_predictions))
        start_indices = np.argsort(subset_predictions)[:start_count]
        deployment_predicted_value, anchor_coeffs, deployment = optimize_generic_family_convex_hull(
            model,
            train_packet.base.w,
            start_indices=start_indices,
        )
        fullswarm_predictions = model.predict(packet.base.w)
        chosen_idx = int(np.argmin(fullswarm_predictions))
        distances = 0.5 * np.abs(packet.base.w - deployment[None, :, :]).sum(axis=2).mean(axis=1)
        nearest_idx = int(np.argmin(distances))
        coeff_support = np.asarray(anchor_coeffs > 1e-6, dtype=bool)
        top_coeff_indices = np.argsort(anchor_coeffs)[::-1][: min(5, len(anchor_coeffs))]

        rows.append(
            {
                "subset_size": subset_size,
                "tuning_method": str(tuning_metrics["method"]),
                "tuning_objective_name": str(tuning_metrics["objective_name"]),
                "tuning_objective": float(tuning_metrics["objective"]),
                "tuning_cv_rmse": float(tuning_metrics["cv_rmse"]),
                "tuning_cv_r2": float(tuning_metrics["cv_r2"]),
                "tuning_cv_regret_at_1": float(tuning_metrics["cv_regret_at_1"]),
                "tuning_cv_foldmean_regret_at_1": float(tuning_metrics["cv_foldmean_regret_at_1"]),
                "predicted_optimum_value": float(deployment_predicted_value),
                "fullswarm_chosen_run_name": str(packet.base.frame.iloc[chosen_idx][packet.base.name_col]),
                "fullswarm_chosen_value": float(packet.base.y[chosen_idx]),
                "fullswarm_regret_at_1": float(packet.base.y[chosen_idx] - best_observed_bpb),
                "nearest_observed_run_name": str(packet.base.frame.iloc[nearest_idx][packet.base.name_col]),
                "nearest_observed_value": float(packet.base.y[nearest_idx]),
                "nearest_observed_tv_distance": float(distances[nearest_idx]),
                "optimum_move_mean_phase_tv_vs_prev": (
                    None if previous_deployment is None else _mean_phase_tv_distance(deployment, previous_deployment)
                ),
                "hull_anchor_count": int(train_packet.base.w.shape[0]),
                "hull_start_count": int(start_count),
                "hull_nonzero_coeff_count": int(np.sum(coeff_support)),
                "top_hull_run_names": [
                    str(train_packet.base.frame.iloc[idx][train_packet.base.name_col]) for idx in top_coeff_indices
                ],
                "top_hull_coeffs": [float(anchor_coeffs[idx]) for idx in top_coeff_indices],
            }
        )
        previous_deployment = deployment

    return pd.DataFrame(rows)


def _plot(frame: pd.DataFrame, *, best_observed_bpb: float) -> None:
    frame = frame.sort_values("subset_size")
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
        label="Observed-hull predicted BPB",
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

    ax_bpb.set_title("Two-phase many-domain: GRP convergence (observed-only hull deployment)")
    ax_bpb.set_ylabel("Predicted BPB")
    ax_regret.set_ylabel("Regret@1")
    ax_cvregret.set_ylabel("CV Mean Regret@1")
    ax_move.set_ylabel("Mean phase TV")
    ax_move.set_xlabel("Observed runs used for fitting")
    ax_move.set_xticks(list(GENERICFAMILY_RETUNED_SUBSET_OPTIMA_ALL_SUBSET_SIZES))
    ax_move.set_xlim(
        min(GENERICFAMILY_RETUNED_SUBSET_OPTIMA_ALL_SUBSET_SIZES),
        max(GENERICFAMILY_RETUNED_SUBSET_OPTIMA_ALL_SUBSET_SIZES),
    )
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
    best_observed_bpb = float(np.min(packet.base.y))
    curve_points = _curve_points()
    curve_points.to_csv(CURVE_POINTS_CSV, index=False)
    _plot(curve_points, best_observed_bpb=best_observed_bpb)
    SUMMARY_JSON.write_text(
        json.dumps(
            {
                "objective_metric": OBJECTIVE_METRIC,
                "curve_points_csv": str(CURVE_POINTS_CSV),
                "plot": str(PLOT_PATH),
                "best_observed_bpb": best_observed_bpb,
                "hull_start_count": HULL_START_COUNT,
                "rows": curve_points.to_dict(orient="records"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
