# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "pandas", "scipy"]
# ///
"""Retune no-L2 GRP on random 60M swarm subsets and summarize convergence bands."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.domain_phase_mix.exploratory.two_phase_many import (
    benchmark_grp_power_family_penalty_no_l2_raw_subset_optima as raw_subset_optima,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.benchmark_grp_power_family_penalty_no_l2_retune import (
    VARIANT_NAME,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.surrogate_search.generic_family_followup import (
    load_generic_family_packet,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.surrogate_search.generic_family_penalty_calibration import (
    build_penalty_calibration_surrogate,
    optimize_penalty_calibration_model,
)
from experiments.domain_phase_mix.static_batch_selection import average_phase_tv_distance
from experiments.domain_phase_mix.two_phase_many_genericfamily_observed_only_trustblend_subset_optima import (
    OBJECTIVE_METRIC,
)
from experiments.domain_phase_mix.two_phase_many_genericfamily_retuned_subset_optima import (
    _subset_packet,
)

SCRIPT_DIR = Path(__file__).resolve().parent
RAW_CSV = SCRIPT_DIR / "two_phase_many_grp_power_family_penalty_no_l2_random_subset_raw.csv"
SUMMARY_CSV = SCRIPT_DIR / "two_phase_many_grp_power_family_penalty_no_l2_random_subset_summary.csv"
SUMMARY_JSON = SCRIPT_DIR / "two_phase_many_grp_power_family_penalty_no_l2_random_subset_summary.json"
DEFAULT_REPLICATES = 16
DEFAULT_SEED = 20260427
DEFAULT_MAX_WORKERS = 6
POLICY = "random_power_family_penalty_no_l2_raw_optimum"
BEST_VARIANT = raw_subset_optima.BEST_VARIANT
FULL_SWARM_COARSE_TOP_K = raw_subset_optima.FULL_SWARM_COARSE_TOP_K
METHOD = raw_subset_optima.METHOD
SUBSET_COARSE_TOP_K = raw_subset_optima.SUBSET_COARSE_TOP_K
SUBSET_SIZES = raw_subset_optima.SUBSET_SIZES
_best_observed_in_subset = raw_subset_optima._best_observed_in_subset
_optimize_no_l2_subset = raw_subset_optima._optimize_no_l2_subset
METRIC_COLUMNS = (
    "predicted_optimum_value",
    "subset_best_observed_bpb",
    "fullswarm_chosen_value",
    "fullswarm_regret_at_1",
    "nearest_observed_value",
    "nearest_observed_tv_distance",
    "tuning_objective",
    "tuning_cv_rmse",
    "tuning_cv_regret_at_1",
    "tuning_cv_foldmean_regret_at_1",
    "tuning_lower_tail_optimism",
    "tuning_cv_depopt_best8",
    "tuning_cv_rawopt_nearest_tv",
    "phase0_max_weight",
    "phase1_max_weight",
    "phase0_support_below_1e4",
    "phase1_support_below_1e4",
    "optimum_move_mean_phase_tv_vs_prev",
)


def _random_indices(*, population_size: int, subset_size: int, seed: int) -> np.ndarray:
    if subset_size == population_size:
        return np.arange(population_size, dtype=int)
    rng = np.random.default_rng(seed)
    return np.asarray(sorted(rng.choice(population_size, size=subset_size, replace=False).tolist()), dtype=int)


def _fit_random_subset_point(payload: tuple[int, int, int]) -> dict[str, object]:
    subset_size, bootstrap_seed, base_seed = payload
    packet = load_generic_family_packet(target=OBJECTIVE_METRIC)
    subset_indices = _random_indices(
        population_size=len(packet.base.y),
        subset_size=subset_size,
        seed=base_seed + 1009 * subset_size + bootstrap_seed,
    )
    train_packet = _subset_packet(packet, subset_indices)
    best_params, tuning_metrics, deployment = _optimize_no_l2_subset(
        train_packet,
        coarse_top_k=(FULL_SWARM_COARSE_TOP_K if subset_size == len(packet.base.y) else SUBSET_COARSE_TOP_K),
    )
    model = build_penalty_calibration_surrogate(train_packet, params=best_params, variant_name=VARIANT_NAME).fit(
        train_packet.base.w,
        train_packet.base.y,
    )
    optimizer_result, _, _ = optimize_penalty_calibration_model(train_packet, model, seed=base_seed + bootstrap_seed)
    fullswarm_predictions = model.predict(packet.base.w)
    chosen_idx = int(np.argmin(fullswarm_predictions))
    distances = average_phase_tv_distance(packet.base.w, deployment[None, :, :])
    nearest_idx = int(np.argmin(distances))
    subset_best_run_name, subset_best_bpb = _best_observed_in_subset(packet, subset_indices)
    best_full_observed_bpb = float(np.min(packet.base.y))
    return {
        "subset_size": subset_size,
        "bootstrap_seed": bootstrap_seed,
        "policy": POLICY,
        "objective_metric": OBJECTIVE_METRIC,
        "variant_name": BEST_VARIANT,
        "tuning_method": METHOD,
        "selected_indices": json.dumps(subset_indices.tolist()),
        "predicted_optimum_value": float(optimizer_result.fun),
        "subset_best_observed_run_name": subset_best_run_name,
        "subset_best_observed_bpb": subset_best_bpb,
        "fullswarm_chosen_run_name": str(packet.base.frame.iloc[chosen_idx][packet.base.name_col]),
        "fullswarm_chosen_value": float(packet.base.y[chosen_idx]),
        "fullswarm_regret_at_1": float(packet.base.y[chosen_idx] - best_full_observed_bpb),
        "nearest_observed_run_name": str(packet.base.frame.iloc[nearest_idx][packet.base.name_col]),
        "nearest_observed_value": float(packet.base.y[nearest_idx]),
        "nearest_observed_tv_distance": float(distances[nearest_idx]),
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
    }


def _with_optimum_movements(frame: pd.DataFrame) -> pd.DataFrame:
    """Attach per-seed movement between successive subset-size deployments when available."""
    # The random benchmark does not store full domain weights, so movement is a placeholder unless future
    # runs add deployment vectors. Keep the column to match deterministic curve schemas.
    frame = frame.copy()
    frame["optimum_move_mean_phase_tv_vs_prev"] = np.nan
    return frame


def _summarize(raw: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for subset_size, group in raw.groupby("subset_size", sort=True):
        row: dict[str, object] = {
            "subset_size": int(subset_size),
            "n_bootstrap": int(group["bootstrap_seed"].nunique()),
        }
        for column in METRIC_COLUMNS:
            if column not in group.columns:
                continue
            values = pd.to_numeric(group[column], errors="coerce").dropna().to_numpy(dtype=float)
            if values.size == 0:
                for suffix in ("mean", "std", "median", "q10", "q25", "q75", "q90"):
                    row[f"{column}_{suffix}"] = np.nan
                continue
            row[f"{column}_mean"] = float(np.mean(values))
            row[f"{column}_std"] = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
            row[f"{column}_median"] = float(np.median(values))
            row[f"{column}_q10"] = float(np.percentile(values, 10))
            row[f"{column}_q25"] = float(np.percentile(values, 25))
            row[f"{column}_q75"] = float(np.percentile(values, 75))
            row[f"{column}_q90"] = float(np.percentile(values, 90))
        rows.append(row)
    return pd.DataFrame(rows)


def _parse_subset_sizes(value: str) -> tuple[int, ...]:
    if value == "all":
        return SUBSET_SIZES
    sizes = tuple(int(part) for part in value.split(",") if part.strip())
    invalid = sorted(set(sizes).difference(SUBSET_SIZES))
    if invalid:
        raise ValueError(f"Unsupported subset sizes: {invalid}; expected subset of {SUBSET_SIZES}")
    return sizes


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replicates", type=int, default=DEFAULT_REPLICATES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--max-workers", type=int, default=DEFAULT_MAX_WORKERS)
    parser.add_argument("--subset-sizes", default="all")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    subset_sizes = _parse_subset_sizes(args.subset_sizes)
    tasks = [
        (subset_size, bootstrap_seed, int(args.seed))
        for subset_size in subset_sizes
        for bootstrap_seed in range(int(args.replicates))
        if subset_size != 242 or bootstrap_seed == 0
    ]
    rows: list[dict[str, object]] = []
    with ProcessPoolExecutor(max_workers=int(args.max_workers)) as executor:
        futures = {executor.submit(_fit_random_subset_point, task): task for task in tasks}
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            print(
                f"Finished random GRP subset_size={row['subset_size']} seed={row['bootstrap_seed']} "
                f"cv_rmse={row['tuning_cv_rmse']:.6f}",
                flush=True,
            )

    raw = _with_optimum_movements(pd.DataFrame(rows).sort_values(["subset_size", "bootstrap_seed"]))
    raw.to_csv(RAW_CSV, index=False)
    summary = _summarize(raw)
    summary.to_csv(SUMMARY_CSV, index=False)
    SUMMARY_JSON.write_text(
        json.dumps(
            {
                "objective_metric": OBJECTIVE_METRIC,
                "variant_name": BEST_VARIANT,
                "policy": POLICY,
                "subset_sizes": list(subset_sizes),
                "replicates": int(args.replicates),
                "seed": int(args.seed),
                "raw_csv": str(RAW_CSV),
                "summary_csv": str(SUMMARY_CSV),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {RAW_CSV}")
    print(f"Wrote {SUMMARY_CSV}")
    print(f"Wrote {SUMMARY_JSON}")


if __name__ == "__main__":
    main()
