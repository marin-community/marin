# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "pandas", "scipy", "scikit-learn"]
# ///
"""Benchmark family-curvature GRP variants on the many-domain packet."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from experiments.domain_phase_mix.exploratory.two_phase_many.surrogate_search.generic_family_followup import (
    load_generic_family_packet,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.surrogate_search.generic_family_flexible_signal import (
    FLEXIBLE_VARIANT_NAMES,
    GenericFamilyFlexibleSignalSurrogate,
    build_flexible_signal_surrogate,
    compute_flexible_surrogate_metrics,
    deploy_flexible_signal_gaincapped_topkactual,
    flexible_signal_oof_metrics_observed_only,
    flexible_signal_params_from_metrics,
    optimize_flexible_signal_model,
    tune_flexible_signal_params_observed_only,
)

DEFAULT_VARIANTS = ("log", "power", "boxcox", "power_family", "boxcox_family", "power_boxcox_family")
REPEATED_CV_SEEDS = tuple(range(10))
SCRIPT_DIR = Path(__file__).resolve().parent
REFERENCE_STATE_PATH = SCRIPT_DIR / "chatgpt_pro_grp_recovery_packet" / "data" / "current_reference_state.json"
COARSE_CSV = SCRIPT_DIR / "grp_family_curvature_benchmark_coarse.csv"
REFINE_CSV = SCRIPT_DIR / "grp_family_curvature_benchmark_refine.csv"
BEST_CSV = SCRIPT_DIR / "grp_family_curvature_benchmark_best.csv"
DEPLOY_CSV = SCRIPT_DIR / "grp_family_curvature_benchmark_deployments.csv"
REPEATED_CV_CSV = SCRIPT_DIR / "grp_family_curvature_benchmark_repeated_cv.csv"
REPEATED_SUMMARY_CSV = SCRIPT_DIR / "grp_family_curvature_benchmark_repeated_summary.csv"
SUMMARY_JSON = SCRIPT_DIR / "grp_family_curvature_benchmark_summary.json"


def _validated_anchor_arrays() -> tuple[np.ndarray, np.ndarray]:
    payload = json.loads(REFERENCE_STATE_PATH.read_text())
    weights = np.stack(
        [
            np.asarray(payload["validated_global"]["phase_weights"], dtype=float),
            np.asarray(payload["validated_pair"]["phase_weights"], dtype=float),
        ],
        axis=0,
    )
    targets = np.asarray([payload["validated_global_bpb"], payload["validated_pair_bpb"]], dtype=float)
    return weights, targets


def _model_from_best_row(packet, row: dict[str, Any]) -> GenericFamilyFlexibleSignalSurrogate:
    variant = str(row["variant"])
    return build_flexible_signal_surrogate(
        packet,
        params=flexible_signal_params_from_metrics(row, variant),
        variant_name=variant,
    )


def _deployment_row(packet, best_row: dict[str, Any]) -> dict[str, Any]:
    model = _model_from_best_row(packet, best_row).fit(packet.base.w, packet.base.y)
    deployment = deploy_flexible_signal_gaincapped_topkactual(packet, model, best_row)
    deploy_weights = np.asarray(deployment["weights"], dtype=float)
    distances = 0.5 * np.abs(packet.base.w - deploy_weights[None, :, :]).sum(axis=2).mean(axis=1)
    nearest_idx = int(np.argmin(distances))
    fullswarm_pred = model.predict(packet.base.w)
    chosen_idx = int(np.argmin(fullswarm_pred))
    return {
        "variant": str(best_row["variant"]),
        "predicted_optimum_value": float(deployment["predicted_optimum_value"]),
        "raw_predicted_optimum_value": float(deployment["raw_predicted_optimum_value"]),
        "hull_predicted_optimum_value": float(deployment["hull_predicted_optimum_value"]),
        "gain_budget": float(deployment["gain_budget"]),
        "delta": float(deployment["delta"]),
        "phase0_lt_1e4": int(np.sum(deploy_weights[0] < 1e-4)),
        "phase1_lt_1e4": int(np.sum(deploy_weights[1] < 1e-4)),
        "phase0_max_weight": float(np.max(deploy_weights[0])),
        "phase1_max_weight": float(np.max(deploy_weights[1])),
        "nearest_observed_run_name": str(packet.base.frame.iloc[nearest_idx][packet.base.name_col]),
        "nearest_observed_value": float(packet.base.y[nearest_idx]),
        "nearest_observed_tv_distance": float(distances[nearest_idx]),
        "fullswarm_chosen_run_name": str(packet.base.frame.iloc[chosen_idx][packet.base.name_col]),
        "fullswarm_chosen_value": float(packet.base.y[chosen_idx]),
        "fullswarm_regret_at_1": float(packet.base.y[chosen_idx] - np.min(packet.base.y)),
    }


def _repeated_cv_rows(packet, best_row: dict[str, Any]) -> list[dict[str, Any]]:
    variant = str(best_row["variant"])
    params = flexible_signal_params_from_metrics(best_row, variant)
    rows: list[dict[str, Any]] = []
    for seed in REPEATED_CV_SEEDS:
        metrics = flexible_signal_oof_metrics_observed_only(packet, params, variant_name=variant, seed=seed)
        rows.append({"variant": variant, "seed": int(seed), **metrics})
    return rows


def main() -> None:
    for variant in DEFAULT_VARIANTS:
        if variant not in FLEXIBLE_VARIANT_NAMES:
            raise ValueError(f"Unsupported variant {variant!r}")

    packet = load_generic_family_packet()
    valid_weights, valid_y = _validated_anchor_arrays()

    coarse_frames: list[pd.DataFrame] = []
    refine_frames: list[pd.DataFrame] = []
    best_rows: list[dict[str, Any]] = []
    deploy_rows: list[dict[str, Any]] = []
    repeated_rows: list[dict[str, Any]] = []

    for variant in DEFAULT_VARIANTS:
        coarse, refine, best, _ = tune_flexible_signal_params_observed_only(
            packet,
            variant_name=variant,
            method="Powell",
            coarse_top_k=4,
            seed=0,
        )
        coarse_frames.append(coarse)
        refine_frames.append(refine)

        model = _model_from_best_row(packet, best)
        metrics = compute_flexible_surrogate_metrics(
            packet,
            model,
            seed=0,
            valid_weights=valid_weights,
            valid_y=valid_y,
        )
        raw_result, phase0, phase1 = optimize_flexible_signal_model(
            packet,
            model.fit(packet.base.w, packet.base.y),
            seed=0,
            n_random=8,
        )
        best_row = {
            **best,
            **metrics,
            "raw_predicted_optimum_value": float(raw_result.fun),
            "raw_phase0_lt_1e4": int(np.sum(phase0 < 1e-4)),
            "raw_phase1_lt_1e4": int(np.sum(phase1 < 1e-4)),
        }
        best_rows.append(best_row)
        deploy_rows.append(_deployment_row(packet, best_row))
        repeated_rows.extend(_repeated_cv_rows(packet, best_row))

    coarse_frame = pd.concat(coarse_frames, ignore_index=True)
    refine_frame = pd.concat(refine_frames, ignore_index=True)
    best_frame = pd.DataFrame(best_rows).sort_values(
        ["cv_rmse", "train_rmse", "anchor_mae"],
        ascending=[True, True, True],
    )
    deploy_frame = pd.DataFrame(deploy_rows).sort_values(
        ["predicted_optimum_value", "delta"],
        ascending=[True, True],
    )
    repeated_frame = pd.DataFrame(repeated_rows)
    repeated_summary = (
        repeated_frame.groupby("variant", as_index=False)
        .agg(
            repeated_cv_rmse_mean=("cv_rmse", "mean"),
            repeated_cv_rmse_std=("cv_rmse", "std"),
            repeated_foldmean_regret_mean=("cv_foldmean_regret_at_1", "mean"),
            repeated_foldmean_regret_std=("cv_foldmean_regret_at_1", "std"),
            repeated_lower_tail_mean=("lower_tail_optimism", "mean"),
            repeated_lower_tail_std=("lower_tail_optimism", "std"),
        )
        .sort_values("repeated_cv_rmse_mean")
    )

    coarse_frame.to_csv(COARSE_CSV, index=False)
    refine_frame.to_csv(REFINE_CSV, index=False)
    best_frame.to_csv(BEST_CSV, index=False)
    deploy_frame.to_csv(DEPLOY_CSV, index=False)
    repeated_frame.to_csv(REPEATED_CV_CSV, index=False)
    repeated_summary.to_csv(REPEATED_SUMMARY_CSV, index=False)
    SUMMARY_JSON.write_text(
        json.dumps(
            {
                "best_rows": best_frame.to_dict(orient="records"),
                "deployment_rows": deploy_frame.to_dict(orient="records"),
                "repeated_summary_rows": repeated_summary.to_dict(orient="records"),
                "best_csv": str(BEST_CSV),
                "deployment_csv": str(DEPLOY_CSV),
                "repeated_summary_csv": str(REPEATED_SUMMARY_CSV),
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
