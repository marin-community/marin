# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib", "numpy", "pandas"]
# ///
"""Plot GRP convergence for the observed-only trustblend deployment rule."""

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
    load_generic_family_packet,
)
from experiments.domain_phase_mix.static_batch_selection import retrospective_generic_selection
from experiments.domain_phase_mix.two_phase_many_genericfamily_observed_only_trustblend_baseline import (
    GENERICFAMILY_OBSERVED_ONLY_TRUSTBLEND_RUN_NAME,
    GENERICFAMILY_OBSERVED_ONLY_TRUSTBLEND_SOURCE_EXPERIMENT,
    genericfamily_observed_only_trustblend_summary,
)
from experiments.domain_phase_mix.two_phase_many_genericfamily_observed_only_trustblend_subset_optima import (
    GENERICFAMILY_OBSERVED_ONLY_TRUSTBLEND_SUBSET_OPTIMA_REPRESENTATIVE_SUBSET_SIZES,
    GENERICFAMILY_OBSERVED_ONLY_TRUSTBLEND_SUBSET_OPTIMA_SOURCE_EXPERIMENT,
    OBJECTIVE_METRIC,
    genericfamily_observed_only_trustblend_subset_optimum_run_name,
)
from experiments.domain_phase_mix.two_phase_many_genericfamily_retuned_subset_optima import CSV_PATH

plt.rcParams["text.usetex"] = False

logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
CURVE_POINTS_CSV = SCRIPT_DIR / "two_phase_many_grp_observed_only_trustblend_top8actual_cap_curve_points.csv"
SUMMARY_JSON = SCRIPT_DIR / "two_phase_many_grp_observed_only_trustblend_top8actual_cap_summary.json"
PLOT_PATH = SCRIPT_DIR / "two_phase_many_grp_observed_only_trustblend_top8actual_cap_convergence_tracks.png"

SUBSET_CHECKPOINT_ROOT_GCS = (
    "gs://marin-us-east5/checkpoints/" + GENERICFAMILY_OBSERVED_ONLY_TRUSTBLEND_SUBSET_OPTIMA_SOURCE_EXPERIMENT
)
FULL_CHECKPOINT_ROOT_GCS = "gs://marin-us-east5/checkpoints/" + GENERICFAMILY_OBSERVED_ONLY_TRUSTBLEND_SOURCE_EXPERIMENT


def _gcloud_text(*args: str) -> str:
    return subprocess.check_output(["gcloud", *args], text=True)


def _load_realized_subset_validated_bpb() -> dict[int, float]:
    try:
        directories = [
            line.strip()
            for line in _gcloud_text("storage", "ls", f"{SUBSET_CHECKPOINT_ROOT_GCS}/").splitlines()
            if line.strip().startswith("gs://")
        ]
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        logger.warning("Unable to list observed-only trustblend subset checkpoints: %s", exc)
        return {}

    by_run_name = {
        directory.rstrip("/").split("/")[-1].rsplit("-", 1)[0]: directory.rstrip("/") for directory in directories
    }
    realized: dict[int, float] = {}
    for subset_size in GENERICFAMILY_OBSERVED_ONLY_TRUSTBLEND_SUBSET_OPTIMA_REPRESENTATIVE_SUBSET_SIZES:
        run_name = genericfamily_observed_only_trustblend_subset_optimum_run_name(subset_size)
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


def _load_realized_full_validated_bpb() -> float | None:
    try:
        directories = [
            line.strip()
            for line in _gcloud_text("storage", "ls", f"{FULL_CHECKPOINT_ROOT_GCS}/").splitlines()
            if line.strip().startswith("gs://")
        ]
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        logger.warning("Unable to list observed-only trustblend full-data checkpoints: %s", exc)
        return None

    target_prefix = f"{GENERICFAMILY_OBSERVED_ONLY_TRUSTBLEND_RUN_NAME}-"
    checkpoint_root = next(
        (
            directory.rstrip("/")
            for directory in directories
            if directory.rstrip("/").split("/")[-1].startswith(target_prefix)
        ),
        None,
    )
    if checkpoint_root is None:
        return None
    eval_metrics_path = f"{checkpoint_root}/checkpoints/eval_metrics.jsonl"
    try:
        payload = _gcloud_text("storage", "cat", eval_metrics_path)
    except subprocess.CalledProcessError as exc:
        logger.warning("Unable to read %s: %s", eval_metrics_path, exc)
        return None
    records = [json.loads(line) for line in payload.splitlines() if line.strip()]
    if not records:
        return None
    metric_value = records[-1].get(OBJECTIVE_METRIC)
    return None if metric_value is None else float(metric_value)


def _load_curve_points() -> pd.DataFrame:
    summary_payload = json.loads(SUMMARY_JSON.read_text())
    frame = pd.DataFrame(summary_payload["rows"]).sort_values("subset_size").reset_index(drop=True)
    frame = frame.drop(
        columns=[
            "subset_best_observed_run_name",
            "subset_best_observed_bpb",
            "subset_best_observed_run_name_x",
            "subset_best_observed_bpb_x",
            "subset_best_observed_run_name_y",
            "subset_best_observed_bpb_y",
            "baseline_proportional_bpb_at_first_appearance",
            "baseline_unimax_bpb_at_first_appearance",
        ],
        errors="ignore",
    )
    best_observed_bpb = float(summary_payload["best_observed_bpb"])

    realized_subset = _load_realized_subset_validated_bpb()
    frame["actual_validated_bpb"] = frame["subset_size"].map(realized_subset)
    frame["validated_prediction_error"] = frame["actual_validated_bpb"] - frame["predicted_optimum_value"]
    frame["validated_regret_at_1"] = frame["actual_validated_bpb"] - best_observed_bpb

    packet = load_generic_family_packet(target=OBJECTIVE_METRIC)
    _, spec, _ = load_two_phase_many_candidate_summary_spec(
        CSV_PATH,
        objective_metric=OBJECTIVE_METRIC,
        name="two_phase_many_grp_observed_only_trustblend_top8actual_cap",
    )
    subset_best_rows: list[dict[str, object]] = []
    for subset_size in frame["subset_size"].astype(int).tolist():
        selection = retrospective_generic_selection(spec, method="feature_bayes_linear", k=subset_size, seed=0)
        subset_indices = np.asarray(selection.selected_indices, dtype=int)
        subset_best_idx = int(subset_indices[np.argmin(packet.base.y[subset_indices])])
        subset_best_rows.append(
            {
                "subset_size": subset_size,
                "subset_best_observed_run_name": str(packet.base.frame.iloc[subset_best_idx][packet.base.name_col]),
                "subset_best_observed_bpb": float(packet.base.y[subset_best_idx]),
            }
        )
    frame = frame.merge(pd.DataFrame(subset_best_rows), on="subset_size", how="left")
    frame = frame[frame["subset_size"] != len(packet.base.y)].reset_index(drop=True)

    full_summary = genericfamily_observed_only_trustblend_summary()
    full_realized_bpb = _load_realized_full_validated_bpb()
    full_row = {
        "subset_size": len(packet.base.y),
        "variant": "trustblend_top8actual_cap",
        "tuning_procedure": "observed_only",
        "predicted_optimum_value": full_summary.predicted_optimum_value,
        "fullswarm_chosen_run_name": full_summary.fullswarm_chosen_run_name,
        "fullswarm_chosen_value": full_summary.fullswarm_chosen_value,
        "fullswarm_regret_at_1": full_summary.fullswarm_regret_at_1,
        "subset_best_observed_run_name": str(
            packet.base.frame.iloc[int(np.argmin(packet.base.y))][packet.base.name_col]
        ),
        "subset_best_observed_bpb": best_observed_bpb,
        "nearest_observed_run_name": full_summary.nearest_observed_run_name,
        "nearest_observed_value": full_summary.nearest_observed_value,
        "nearest_observed_tv_distance": full_summary.nearest_observed_tv_distance,
        "optimum_move_mean_phase_tv_vs_prev": np.nan,
        "tuning_objective": full_summary.tuning_objective,
        "tuning_cv_rmse": full_summary.tuning_cv_rmse,
        "tuning_cv_r2": full_summary.tuning_cv_r2,
        "tuning_cv_regret_at_1": full_summary.tuning_cv_regret_at_1,
        "tuning_cv_foldmean_regret_at_1": full_summary.tuning_cv_foldmean_regret_at_1,
        "alpha": np.nan,
        "eta": np.nan,
        "lam": np.nan,
        "tau": np.nan,
        "reg": np.nan,
        "beta": np.nan,
        "deployment_delta": full_summary.deployment_delta,
        "deployment_realized_gain": full_summary.deployment_realized_gain,
        "deployment_gain_budget": full_summary.deployment_gain_budget,
        "deployment_raw_predicted_optimum_value": full_summary.deployment_raw_predicted_optimum_value,
        "deployment_hull_predicted_optimum_value": full_summary.deployment_hull_predicted_optimum_value,
        "phase0_broad_text": full_summary.family_shares["phase0_broad_text"],
        "phase0_tech_code": full_summary.family_shares["phase0_tech_code"],
        "phase0_reasoning": full_summary.family_shares["phase0_reasoning"],
        "phase1_broad_text": full_summary.family_shares["phase1_broad_text"],
        "phase1_tech_code": full_summary.family_shares["phase1_tech_code"],
        "phase1_reasoning": full_summary.family_shares["phase1_reasoning"],
        "tuning_lower_tail_optimism": full_summary.tuning_lower_tail_optimism,
        "actual_validated_bpb": full_realized_bpb,
        "validated_prediction_error": (
            None if full_realized_bpb is None else full_realized_bpb - full_summary.predicted_optimum_value
        ),
        "validated_regret_at_1": None if full_realized_bpb is None else full_realized_bpb - best_observed_bpb,
    }
    frame = pd.concat([frame, pd.DataFrame([full_row])], ignore_index=True)
    return frame.sort_values("subset_size").reset_index(drop=True)


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

    ax_bpb.set_title("Two-phase many-domain: GRP convergence (observed-only trustblend)")
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
    best_observed_bpb = float(np.min(packet.base.y))
    curve_points = _load_curve_points()
    curve_points.to_csv(CURVE_POINTS_CSV, index=False)
    _plot(curve_points)
    SUMMARY_JSON.write_text(
        json.dumps(
            {
                "objective_metric": OBJECTIVE_METRIC,
                "curve_points_csv": str(CURVE_POINTS_CSV),
                "plot": str(PLOT_PATH),
                "best_observed_bpb": best_observed_bpb,
                "rows": curve_points.replace({np.nan: None}).to_dict(orient="records"),
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
