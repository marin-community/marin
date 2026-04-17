# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib", "numpy", "pandas"]
# ///
"""Plot GRP convergence with top-8-actual frontier-hull deployment."""

from __future__ import annotations

import json
import logging
from pathlib import Path
import subprocess

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments.domain_phase_mix.exploratory.two_phase_many.convergence_plot_style import (
    BEST_OBSERVED_BPB_COLOR,
    GRP_COLOR,
    PREDICTED_LINESTYLE,
    VALIDATED_LINESTYLE,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.dataset_metadata import (
    load_two_phase_many_candidate_summary_spec,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.surrogate_search.generic_family_followup import (
    GenericFamilyRetainedTotalSurrogate,
    load_generic_family_packet,
    optimize_generic_family_convex_hull,
)
from experiments.domain_phase_mix.static_batch_selection import retrospective_generic_selection
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
    _subset_packet,
    _summary_weights,
    tune_genericfamily_subset_params,
)
from experiments.domain_phase_mix.two_phase_many_genericfamily_top8actual_hull_subset_optima import (
    GENERICFAMILY_TOP8ACTUAL_HULL_SUBSET_OPTIMA_SOURCE_EXPERIMENT,
    genericfamily_top8actual_hull_subset_optimum_run_name,
)

plt.rcParams["text.usetex"] = False

logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
CURVE_POINTS_CSV = SCRIPT_DIR / "two_phase_many_grp_top8actual_hull_curve_points.csv"
SUMMARY_JSON = SCRIPT_DIR / "two_phase_many_grp_top8actual_hull_summary.json"
PLOT_PATH = SCRIPT_DIR / "two_phase_many_grp_top8actual_hull_convergence_tracks.png"
TOP_ACTUAL_HULL_COUNT = 8
CHECKPOINT_ROOT_GCS = "gs://marin-us-east5/checkpoints/" + GENERICFAMILY_TOP8ACTUAL_HULL_SUBSET_OPTIMA_SOURCE_EXPERIMENT


def _gcloud_text(*args: str) -> str:
    return subprocess.check_output(["gcloud", *args], text=True)


def _load_realized_validated_bpb() -> dict[int, float]:
    """Load realized BPB for completed top-8-actual-hull representative runs."""
    try:
        directories = [
            line.strip()
            for line in _gcloud_text("storage", "ls", f"{CHECKPOINT_ROOT_GCS}/").splitlines()
            if line.strip().startswith("gs://")
        ]
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        logger.warning("Unable to list top-8-actual-hull checkpoints: %s", exc)
        return {}

    by_run_name = {
        directory.rstrip("/").split("/")[-1].rsplit("-", 1)[0]: directory.rstrip("/") for directory in directories
    }
    realized: dict[int, float] = {}
    for subset_size in GENERICFAMILY_RETUNED_SUBSET_OPTIMA_ALL_SUBSET_SIZES:
        run_name = genericfamily_top8actual_hull_subset_optimum_run_name(subset_size)
        checkpoint_root = by_run_name.get(run_name)
        if checkpoint_root is None:
            continue
        eval_metrics_path = f"{checkpoint_root}/checkpoints/eval_metrics.jsonl"
        try:
            payload = _gcloud_text("storage", "cat", eval_metrics_path)
        except subprocess.CalledProcessError as exc:
            logger.warning("Unable to read %s: %s", eval_metrics_path, exc)
            continue
        records = [json.loads(line) for line in payload.splitlines() if line.strip()]
        if not records:
            continue
        metric_value = records[-1].get(OBJECTIVE_METRIC)
        if metric_value is not None:
            realized[subset_size] = float(metric_value)
    return realized


def _curve_points() -> pd.DataFrame:
    _, spec, _ = load_two_phase_many_candidate_summary_spec(
        CSV_PATH,
        objective_metric=OBJECTIVE_METRIC,
        name="two_phase_many_grp_top8actual_hull",
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
    realized_validated_bpb = _load_realized_validated_bpb()
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
        actual_order = np.argsort(train_packet.base.y)
        hull_anchor_indices = actual_order[: min(TOP_ACTUAL_HULL_COUNT, len(actual_order))]
        hull_anchor_weights = train_packet.base.w[hull_anchor_indices]
        start_indices = np.arange(min(TOP_ACTUAL_HULL_COUNT, len(hull_anchor_indices)), dtype=int)
        deployment_predicted_value, anchor_coeffs, deployment = optimize_generic_family_convex_hull(
            model,
            hull_anchor_weights,
            start_indices=start_indices,
        )
        fullswarm_predictions = model.predict(packet.base.w)
        chosen_idx = int(np.argmin(fullswarm_predictions))
        distances = 0.5 * np.abs(packet.base.w - deployment[None, :, :]).sum(axis=2).mean(axis=1)
        nearest_idx = int(np.argmin(distances))
        coeff_support = np.asarray(anchor_coeffs > 1e-6, dtype=bool)
        top_coeff_indices = np.argsort(anchor_coeffs)[::-1][: min(5, len(anchor_coeffs))]

        subset_best_idx = int(np.argmin(train_packet.base.y))
        subset_best_observed_bpb = float(train_packet.base.y[subset_best_idx])

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
                "subset_best_observed_run_name": str(
                    train_packet.base.frame.iloc[subset_best_idx][train_packet.base.name_col]
                ),
                "subset_best_observed_bpb": subset_best_observed_bpb,
                "fullswarm_chosen_run_name": str(packet.base.frame.iloc[chosen_idx][packet.base.name_col]),
                "fullswarm_chosen_value": float(packet.base.y[chosen_idx]),
                "fullswarm_regret_at_1": float(packet.base.y[chosen_idx] - best_observed_bpb),
                "nearest_observed_run_name": str(packet.base.frame.iloc[nearest_idx][packet.base.name_col]),
                "nearest_observed_value": float(packet.base.y[nearest_idx]),
                "nearest_observed_tv_distance": float(distances[nearest_idx]),
                "actual_validated_bpb": realized_validated_bpb.get(subset_size),
                "validated_prediction_error": (
                    None
                    if subset_size not in realized_validated_bpb
                    else float(realized_validated_bpb[subset_size] - deployment_predicted_value)
                ),
                "validated_regret_at_1": (
                    None
                    if subset_size not in realized_validated_bpb
                    else float(realized_validated_bpb[subset_size] - best_observed_bpb)
                ),
                "optimum_move_mean_phase_tv_vs_prev": (
                    None if previous_deployment is None else _mean_phase_tv_distance(deployment, previous_deployment)
                ),
                "hull_anchor_count": len(hull_anchor_indices),
                "hull_nonzero_coeff_count": int(np.sum(coeff_support)),
                "top_hull_run_names": [
                    str(train_packet.base.frame.iloc[hull_anchor_indices[idx]][train_packet.base.name_col])
                    for idx in top_coeff_indices
                ],
                "top_hull_coeffs": [float(anchor_coeffs[idx]) for idx in top_coeff_indices],
            }
        )
        previous_deployment = deployment

    return pd.DataFrame(rows)


def _plot(frame: pd.DataFrame) -> None:
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
        label="Top-8-actual-hull predicted BPB",
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
    validated = frame[frame["actual_validated_bpb"].notna()].copy()
    if not validated.empty:
        ax_bpb.plot(
            validated["subset_size"],
            validated["actual_validated_bpb"],
            color=GRP_COLOR,
            marker="X",
            markersize=8,
            linewidth=1.8,
            linestyle=VALIDATED_LINESTYLE,
            label="Realized validated BPB",
        )
        for row in validated.itertuples(index=False):
            ax_bpb.annotate(
                f"{row.actual_validated_bpb:.4f}",
                (row.subset_size, row.actual_validated_bpb),
                textcoords="offset points",
                xytext=(0, 7),
                ha="center",
                fontsize=8,
                color=GRP_COLOR,
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

    ax_bpb.set_title("Two-phase many-domain: GRP convergence (top-8 actual frontier hull)")
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
    _plot(curve_points)
    SUMMARY_JSON.write_text(
        json.dumps(
            {
                "objective_metric": OBJECTIVE_METRIC,
                "curve_points_csv": str(CURVE_POINTS_CSV),
                "plot": str(PLOT_PATH),
                "best_observed_bpb": best_observed_bpb,
                "hull_anchor_count": TOP_ACTUAL_HULL_COUNT,
                "rows": curve_points.to_dict(orient="records"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
