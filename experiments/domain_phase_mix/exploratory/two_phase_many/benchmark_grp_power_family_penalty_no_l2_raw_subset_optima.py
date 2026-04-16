# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.11"
# dependencies = ["fsspec", "numpy", "pandas", "scipy"]
# ///
"""Retune no-L2 GRP on each representative subset size and export raw-optimum summaries."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import as_completed
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from experiments.domain_phase_mix.exploratory.two_phase_many.benchmark_grp_power_family_penalty_no_l2_retune import (
    CV_SEED,
    VARIANT_NAME,
    _pack_no_l2_params,
    _start_bank,
    _unpack_no_l2_params,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.dataset_metadata import (
    load_two_phase_many_candidate_summary_spec,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.surrogate_search.generic_family_followup import (
    family_shares,
    load_generic_family_packet,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.surrogate_search.generic_family_penalty_calibration import (
    build_penalty_calibration_surrogate,
    optimize_penalty_calibration_model,
    penalty_calibration_oof_metrics,
)
from experiments.domain_phase_mix.static_batch_selection import (
    average_phase_tv_distance,
    retrospective_generic_selection,
)
from experiments.domain_phase_mix.two_phase_many_genericfamily_observed_only_trustblend_subset_optima import (
    OBJECTIVE_METRIC,
)
from experiments.domain_phase_mix.two_phase_many_genericfamily_penalty_raw_optima_baselines import (
    genericfamily_penalty_raw_optimum_summary,
)
from experiments.domain_phase_mix.two_phase_many_genericfamily_retuned_subset_optima import (
    CSV_PATH,
    _mean_phase_tv_distance,
    _phase_weights_from_array,
    _subset_packet,
    _top_domains,
)

SCRIPT_DIR = Path(__file__).resolve().parent
CURVE_POINTS_CSV = SCRIPT_DIR / "two_phase_many_grp_power_family_penalty_no_l2_raw_curve_points.csv"
SUMMARY_JSON = SCRIPT_DIR / "two_phase_many_grp_power_family_penalty_no_l2_raw_summary.json"
METHOD = "Powell"
SUBSET_COARSE_TOP_K = 1
FULL_SWARM_COARSE_TOP_K = 3
MAX_WORKERS = 6
SUBSET_SIZES = (20, 40, 60, 80, 100, 140, 180, 220, 242)
POLICY = "feature_bayes_linear_power_family_penalty_no_l2_raw_optimum"
BEST_VARIANT = "power_family_penalty_no_l2"


def _best_observed_in_subset(packet, subset_indices: np.ndarray) -> tuple[str, float]:
    subset_values = packet.base.y[subset_indices]
    best_local_idx = int(np.argmin(subset_values))
    best_idx = int(subset_indices[best_local_idx])
    return str(packet.base.frame.iloc[best_idx][packet.base.name_col]), float(packet.base.y[best_idx])


def _optimize_no_l2_subset(
    train_packet,
    *,
    coarse_top_k: int,
) -> tuple[dict[str, float], dict[str, float | bool], np.ndarray]:
    start_bank = _start_bank()
    coarse_rows: list[dict[str, float | bool]] = []
    for start_id, params in enumerate(start_bank):
        coarse_rows.append(
            {
                "start_id": int(start_id),
                **params,
                **penalty_calibration_oof_metrics(train_packet, params, variant_name=VARIANT_NAME, seed=CV_SEED),
            }
        )
    coarse_frame = pd.DataFrame.from_records(coarse_rows).sort_values(
        ["objective", "cv_rmse", "cv_depopt_best8"],
        ascending=[True, True, True],
    )
    chosen_ids = coarse_frame["start_id"].head(coarse_top_k).tolist()

    best_metrics: dict[str, float | bool] | None = None
    best_objective = float("inf")
    best_weights: np.ndarray | None = None
    for start_id in chosen_ids:
        start = _pack_no_l2_params(start_bank[start_id])
        cache: dict[tuple[float, ...], float] = {}

        def objective(z: np.ndarray, cache: dict[tuple[float, ...], float] = cache) -> float:
            key = tuple(np.round(np.asarray(z, dtype=float), 8))
            if key not in cache:
                params = _unpack_no_l2_params(z)
                metrics = penalty_calibration_oof_metrics(train_packet, params, variant_name=VARIANT_NAME, seed=CV_SEED)
                cache[key] = float(metrics["objective"])
            return cache[key]

        result = minimize(
            objective,
            start,
            method=METHOD,
            options={"maxiter": 30, "xtol": 1e-4, "ftol": 1e-6},
        )
        params = _unpack_no_l2_params(np.asarray(result.x, dtype=float))
        metrics = {
            "success": bool(result.success),
            "message": str(result.message),
            **params,
            **penalty_calibration_oof_metrics(train_packet, params, variant_name=VARIANT_NAME, seed=CV_SEED),
        }
        if float(metrics["objective"]) >= best_objective:
            continue
        model = build_penalty_calibration_surrogate(train_packet, params=params, variant_name=VARIANT_NAME).fit(
            train_packet.base.w,
            train_packet.base.y,
        )
        _, phase0, phase1 = optimize_penalty_calibration_model(train_packet, model, seed=CV_SEED)
        best_objective = float(metrics["objective"])
        best_metrics = metrics
        best_weights = np.stack([phase0, phase1], axis=0)

    if best_metrics is None or best_weights is None:
        raise RuntimeError("No-L2 subset retune failed to produce a best result")
    best_params = {
        key: float(value)
        for key, value in best_metrics.items()
        if key
        in {
            "eta",
            "lam",
            "reg",
            "beta",
            "a_broad_text",
            "a_tech_code",
            "a_reasoning",
            "tau_broad_text",
            "tau_tech_code",
            "tau_reasoning",
        }
    }
    return best_params, best_metrics, best_weights


def _fit_subset_point(subset_size: int) -> dict[str, object]:
    _, spec, _ = load_two_phase_many_candidate_summary_spec(
        CSV_PATH,
        objective_metric=OBJECTIVE_METRIC,
        name="two_phase_many_grp_power_family_penalty_no_l2_raw",
    )
    packet = load_generic_family_packet(target=OBJECTIVE_METRIC)
    best_full_observed_bpb = float(np.min(packet.base.y))
    if subset_size == len(packet.base.y):
        subset_indices = np.arange(len(packet.base.y), dtype=int)
    else:
        selection = retrospective_generic_selection(spec, method="feature_bayes_linear", k=subset_size, seed=0)
        subset_indices = np.asarray(selection.selected_indices, dtype=int)
    train_packet = _subset_packet(packet, subset_indices)
    best_params, tuning_metrics, deployment = _optimize_no_l2_subset(
        train_packet,
        coarse_top_k=(FULL_SWARM_COARSE_TOP_K if subset_size == len(packet.base.y) else SUBSET_COARSE_TOP_K),
    )
    model = build_penalty_calibration_surrogate(train_packet, params=best_params, variant_name=VARIANT_NAME).fit(
        train_packet.base.w,
        train_packet.base.y,
    )
    optimizer_result, _, _ = optimize_penalty_calibration_model(train_packet, model, seed=CV_SEED)
    if subset_size == len(packet.base.y):
        full_summary = genericfamily_penalty_raw_optimum_summary(BEST_VARIANT)
        if not np.isclose(float(optimizer_result.fun), float(full_summary.raw_predicted_optimum_value), atol=1e-6):
            raise ValueError(
                "Full-swarm no-L2 raw optimum mismatch: "
                f"expected {full_summary.raw_predicted_optimum_value}, got {float(optimizer_result.fun)}"
            )
        for key, expected in full_summary.tuned_params.items():
            actual = float(best_params[key])
            if not np.isclose(actual, float(expected), atol=1e-8):
                raise ValueError(f"Full-swarm no-L2 retune mismatch for {key}: expected {expected}, got {actual}")
    fullswarm_predictions = model.predict(packet.base.w)
    chosen_idx = int(np.argmin(fullswarm_predictions))
    distances = average_phase_tv_distance(packet.base.w, deployment[None, :, :])
    nearest_idx = int(np.argmin(distances))
    subset_best_run_name, subset_best_bpb = _best_observed_in_subset(packet, subset_indices)
    phase_weights = _phase_weights_from_array(packet.base.domain_names, deployment)
    return {
        "subset_size": subset_size,
        "run_id": 470 + SUBSET_SIZES.index(subset_size),
        "run_name": f"baseline_genericfamily_power_family_penalty_no_l2_raw_optimum_k{subset_size:03d}_uncheatable_bpb",
        "policy": POLICY,
        "objective_metric": OBJECTIVE_METRIC,
        "variant_name": BEST_VARIANT,
        "tuning_method": METHOD,
        "predicted_optimum_value": float(optimizer_result.fun),
        "subset_best_observed_run_name": subset_best_run_name,
        "subset_best_observed_bpb": subset_best_bpb,
        "fullswarm_chosen_run_name": str(packet.base.frame.iloc[chosen_idx][packet.base.name_col]),
        "fullswarm_chosen_value": float(packet.base.y[chosen_idx]),
        "fullswarm_regret_at_1": float(packet.base.y[chosen_idx] - best_full_observed_bpb),
        "nearest_observed_run_name": str(packet.base.frame.iloc[nearest_idx][packet.base.name_col]),
        "nearest_observed_value": float(packet.base.y[nearest_idx]),
        "nearest_observed_tv_distance": float(distances[nearest_idx]),
        "optimum_move_mean_phase_tv_vs_prev": None,
        "tuning_objective": float(tuning_metrics["objective"]),
        "tuning_cv_rmse": float(tuning_metrics["cv_rmse"]),
        "tuning_cv_regret_at_1": float(tuning_metrics["cv_regret_at_1"]),
        "tuning_cv_foldmean_regret_at_1": float(tuning_metrics["cv_foldmean_regret_at_1"]),
        "tuning_lower_tail_optimism": float(tuning_metrics["lower_tail_optimism"]),
        "tuning_cv_depopt_best8": float(tuning_metrics["cv_depopt_best8"]),
        "tuning_cv_rawopt_nearest_tv": float(tuning_metrics["cv_rawopt_nearest_tv"]),
        "phase0_max_weight": float(np.max(deployment[0])),
        "phase1_max_weight": float(np.max(deployment[1])),
        "phase0_support_below_1e4": int(np.sum(deployment[0] < 1e-4)),
        "phase1_support_below_1e4": int(np.sum(deployment[1] < 1e-4)),
        "phase0_top_domains": _top_domains(packet.base.domain_names, deployment[0], deployment[0] * packet.base.c0),
        "phase1_top_domains": _top_domains(packet.base.domain_names, deployment[1], deployment[1] * packet.base.c1),
        "family_shares": family_shares(packet, deployment),
        "phase_weights": phase_weights,
    }


def build_curve_points(max_workers: int = MAX_WORKERS) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_fit_subset_point, subset_size): subset_size for subset_size in SUBSET_SIZES}
        for future in as_completed(futures):
            rows.append(future.result())
    rows.sort(key=lambda row: int(row["subset_size"]))

    previous_weights: np.ndarray | None = None
    packet = load_generic_family_packet(target=OBJECTIVE_METRIC)
    for row in rows:
        current_weights = np.stack(
            [
                np.asarray(
                    [float(row["phase_weights"]["phase_0"][domain_name]) for domain_name in packet.base.domain_names],
                    dtype=float,
                ),
                np.asarray(
                    [float(row["phase_weights"]["phase_1"][domain_name]) for domain_name in packet.base.domain_names],
                    dtype=float,
                ),
            ],
            axis=0,
        )
        if previous_weights is not None:
            row["optimum_move_mean_phase_tv_vs_prev"] = _mean_phase_tv_distance(current_weights, previous_weights)
        previous_weights = current_weights
    return pd.DataFrame(rows)


def main() -> None:
    frame = build_curve_points()
    frame.to_csv(CURVE_POINTS_CSV, index=False)
    SUMMARY_JSON.write_text(
        json.dumps(
            {
                "variant_name": BEST_VARIANT,
                "objective_metric": OBJECTIVE_METRIC,
                "subset_sizes": list(SUBSET_SIZES),
                "method": METHOD,
                "subset_coarse_top_k": SUBSET_COARSE_TOP_K,
                "full_swarm_coarse_top_k": FULL_SWARM_COARSE_TOP_K,
                "rows": frame.to_dict(orient="records"),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {CURVE_POINTS_CSV}")
    print(f"Wrote {SUMMARY_JSON}")


if __name__ == "__main__":
    main()
