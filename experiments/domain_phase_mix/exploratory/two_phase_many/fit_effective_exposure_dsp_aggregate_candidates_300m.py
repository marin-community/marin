# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "numpy",
#     "pandas",
#     "scipy",
#     "scikit-learn",
# ]
# ///
"""Fit effective-exposure DSP to clean-slate aggregate candidates.

The aggregate notebook writes candidate scores where higher is better. DSP
minimizes an objective, so this script fits ``objective_metric = -score`` and
reports raw optima back in higher-is-better aggregate-score units.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from experiments.domain_phase_mix.exploratory.two_phase_many.standalone_code.dsp_exact import (
    VARIANTS,
    fit_variant,
    metrics,
    optimize_raw,
    packet_from_frame,
    weights_to_frame,
)

SCRIPT_DIR = Path(__file__).resolve().parent
MATRIX_CSV = SCRIPT_DIR / "metric_registry" / "raw_metric_matrix_300m" / "raw_metric_matrix_300m.csv"
METADATA_CSV = SCRIPT_DIR / "two_phase_many_epoch_metadata.csv"
OUTPUT_DIR = SCRIPT_DIR / "reference_outputs" / "aggregate_metric_clean_slate_20260518"
AGGREGATE_SCORES_CSV = OUTPUT_DIR / "aggregate_candidate_scores.csv"
DEFAULT_SUMMARY_CSV = OUTPUT_DIR / "aggregate_candidate_effective_exposure_dsp_summary.csv"
DEFAULT_WEIGHTS_CSV = OUTPUT_DIR / "aggregate_candidate_effective_exposure_dsp_raw_optimum_weights.csv"
DEFAULT_JSON = OUTPUT_DIR / "aggregate_candidate_effective_exposure_dsp_summary.json"

_RAW_SIGNAL: pd.DataFrame | None = None
_METADATA: pd.DataFrame | None = None
_AGGREGATE_SCORES: pd.DataFrame | None = None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-csv", type=Path, default=MATRIX_CSV)
    parser.add_argument("--metadata-csv", type=Path, default=METADATA_CSV)
    parser.add_argument("--aggregate-scores-csv", type=Path, default=AGGREGATE_SCORES_CSV)
    parser.add_argument("--summary-csv", type=Path, default=DEFAULT_SUMMARY_CSV)
    parser.add_argument("--weights-csv", type=Path, default=DEFAULT_WEIGHTS_CSV)
    parser.add_argument("--summary-json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--maxiter", type=int, default=100)
    parser.add_argument("--coarse-top-k", type=int, default=3)
    parser.add_argument("--basin-hopping-iters", type=int, default=0)
    parser.add_argument("--optimum-starts", type=int, default=200)
    parser.add_argument("--stability-seeds", type=int, default=5)
    parser.add_argument("--stability-starts", type=int, default=80)
    parser.add_argument("--max-observed-starts", type=int, default=242)
    parser.add_argument("--no-observed-starts", action="store_true")
    parser.add_argument("--candidate", action="append", default=None)
    return parser.parse_args()


def _initializer(raw_signal: pd.DataFrame, metadata: pd.DataFrame, aggregate_scores: pd.DataFrame) -> None:
    global _RAW_SIGNAL, _METADATA, _AGGREGATE_SCORES
    _RAW_SIGNAL = raw_signal
    _METADATA = metadata
    _AGGREGATE_SCORES = aggregate_scores


def _top_k_domains(weights: np.ndarray, domain_names: list[str], *, k: int) -> set[str]:
    order = np.argsort(weights)[::-1][:k]
    return {domain_names[index] for index in order}


def _mean_pairwise_jaccard(sets: list[set[str]]) -> float:
    if len(sets) < 2:
        return np.nan
    values = []
    for left_index in range(len(sets)):
        for right_index in range(left_index + 1, len(sets)):
            union = sets[left_index] | sets[right_index]
            values.append(len(sets[left_index] & sets[right_index]) / len(union) if union else 1.0)
    return float(np.mean(values)) if values else np.nan


def _stability_diagnostics(
    model: Any,
    observed_start_weights: np.ndarray | None,
    *,
    stability_seeds: int,
    stability_starts: int,
    max_observed_starts: int,
) -> dict[str, Any]:
    if stability_seeds <= 1:
        return {
            "stability_seed_count": stability_seeds,
            "phase0_top8_jaccard": np.nan,
            "phase1_top8_jaccard": np.nan,
            "stability_best_value_range": np.nan,
        }
    phase0_sets: list[set[str]] = []
    phase1_sets: list[set[str]] = []
    values = []
    for seed in range(stability_seeds):
        result, weights = optimize_raw(
            model,
            num_starts=stability_starts,
            seed=seed,
            observed_start_weights=observed_start_weights,
            max_observed_starts=max_observed_starts,
        )
        values.append(float(result.fun))
        phase0_sets.append(_top_k_domains(weights[0], model.domain_names, k=8))
        phase1_sets.append(_top_k_domains(weights[1], model.domain_names, k=8))
    return {
        "stability_seed_count": stability_seeds,
        "phase0_top8_jaccard": _mean_pairwise_jaccard(phase0_sets),
        "phase1_top8_jaccard": _mean_pairwise_jaccard(phase1_sets),
        "stability_best_value_range": float(np.max(values) - np.min(values)) if values else np.nan,
    }


def _fit_candidate(task: dict[str, Any]) -> tuple[dict[str, Any], pd.DataFrame]:
    if _RAW_SIGNAL is None or _METADATA is None or _AGGREGATE_SCORES is None:
        raise RuntimeError("Worker globals were not initialized")

    candidate = str(task["candidate"])
    start_time = time.monotonic()
    fit_frame = _RAW_SIGNAL.merge(
        _AGGREGATE_SCORES[["registry_run_key", "run_name", candidate]],
        on=["registry_run_key", "run_name"],
        how="inner",
        validate="one_to_one",
    )
    fit_frame = fit_frame[pd.to_numeric(fit_frame[candidate], errors="coerce").notna()].copy()
    fit_frame["objective_metric"] = -pd.to_numeric(fit_frame[candidate], errors="coerce")
    if len(fit_frame) < 40:
        return (
            {
                "candidate": candidate,
                "fit_status": "too_few_rows",
                "fit_row_count": len(fit_frame),
                "fit_seconds": time.monotonic() - start_time,
            },
            pd.DataFrame(),
        )

    packet = packet_from_frame(fit_frame, _METADATA)
    variant = VARIANTS["effective_exposure"]
    model, trace = fit_variant(
        packet,
        variant,
        maxiter=int(task["maxiter"]),
        coarse_top_k=int(task["coarse_top_k"]),
        basin_hopping_iters=int(task["basin_hopping_iters"]),
    )
    observed_start_weights = None if task["no_observed_starts"] else packet.w
    raw_result, raw_weights = optimize_raw(
        model,
        num_starts=int(task["optimum_starts"]),
        observed_start_weights=observed_start_weights,
        max_observed_starts=int(task["max_observed_starts"]),
    )
    model_metrics = metrics(packet, model, raw_result, raw_weights)
    stability_metrics = _stability_diagnostics(
        model,
        observed_start_weights,
        stability_seeds=int(task["stability_seeds"]),
        stability_starts=int(task["stability_starts"]),
        max_observed_starts=int(task["max_observed_starts"]),
    )
    best_observed_idx = int(np.argmin(packet.y))
    optimum_weights = weights_to_frame(model, raw_weights)
    optimum_weights.insert(0, "candidate", candidate)
    gamma = float(model.params["gamma"]) if "gamma" in model.params else np.nan
    tau = np.asarray(model.params.get("tau", []), dtype=float)
    phase0_max = float(model_metrics["phase0_max_weight"])
    phase1_max = float(model_metrics["phase1_max_weight"])
    min_support = min(int(model_metrics["raw_phase0_support_gt_1e3"]), int(model_metrics["raw_phase1_support_gt_1e3"]))
    max_phase_weight = max(phase0_max, phase1_max)
    min_top8_jaccard = np.nanmin([stability_metrics["phase0_top8_jaccard"], stability_metrics["phase1_top8_jaccard"]])
    acceptance_pass = (
        float(model_metrics["oof_spearman"]) >= 0.88
        and float(model_metrics["raw_nearest_observed_tv"]) <= 0.40
        and max_phase_weight <= 0.40
        and min_support >= 8
        and np.isfinite(min_top8_jaccard)
        and min_top8_jaccard >= 0.70
    )
    result = {
        "candidate": candidate,
        "fit_status": "ok",
        "fit_seconds": time.monotonic() - start_time,
        "variant": variant.name,
        "maxiter": int(task["maxiter"]),
        "coarse_top_k": int(task["coarse_top_k"]),
        "basin_hopping_iters": int(task["basin_hopping_iters"]),
        "optimum_starts": int(task["optimum_starts"]),
        "observed_start_count": (
            0 if observed_start_weights is None else min(len(observed_start_weights), int(task["max_observed_starts"]))
        ),
        **model_metrics,
        **stability_metrics,
        "fitted_gamma": gamma,
        "fitted_tau_min": float(np.min(tau)) if len(tau) else np.nan,
        "fitted_tau_median": float(np.median(tau)) if len(tau) else np.nan,
        "fitted_tau_max": float(np.max(tau)) if len(tau) else np.nan,
        "fitted_tau_lower_bound_count": int(np.sum(tau <= -1.999)) if len(tau) else 0,
        "fitted_tau_upper_bound_count": int(np.sum(tau >= 7.999)) if len(tau) else 0,
        "max_phase_weight": max_phase_weight,
        "min_phase_support_gt_1e3": min_support,
        "min_top8_jaccard": min_top8_jaccard,
        "raw_optimum_acceptance_pass": bool(acceptance_pass),
        "raw_optimum_reject_reasons": ";".join(
            reason
            for reason, failed in [
                ("oof_spearman_lt_0.88", float(model_metrics["oof_spearman"]) < 0.88),
                ("nearest_tv_gt_0.40", float(model_metrics["raw_nearest_observed_tv"]) > 0.40),
                ("max_phase_weight_gt_0.40", max_phase_weight > 0.40),
                ("min_support_lt_8", min_support < 8),
                ("top8_jaccard_lt_0.70", (not np.isfinite(min_top8_jaccard)) or min_top8_jaccard < 0.70),
            ]
            if failed
        ),
        "best_observed_run_name": str(packet.frame.iloc[best_observed_idx]["run_name"]),
        "best_observed_aggregate_score": float(-packet.y[best_observed_idx]),
        "proportional_aggregate_score": (
            float(-packet.y[packet.frame["run_name"].eq("baseline_proportional")][0])
            if packet.frame["run_name"].eq("baseline_proportional").any()
            else np.nan
        ),
        "raw_predicted_aggregate_score": float(-model_metrics["raw_predicted_optimum_value"]),
        "raw_nearest_observed_aggregate_score": float(-model_metrics["raw_nearest_observed_value"]),
    }
    return result, optimum_weights


def main() -> None:
    args = _parse_args()
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

    raw = pd.read_csv(args.matrix_csv, low_memory=False)
    raw_signal = raw[raw["status"].eq("completed") & raw["row_kind"].eq("signal")].copy()
    metadata = pd.read_csv(args.metadata_csv)
    aggregate_scores = pd.read_csv(args.aggregate_scores_csv)
    candidates = [
        column
        for column in aggregate_scores.columns
        if column not in {"run_name", "registry_run_key"} and (args.candidate is None or column in args.candidate)
    ]
    tasks = [
        {
            "candidate": candidate,
            "maxiter": args.maxiter,
            "coarse_top_k": args.coarse_top_k,
            "basin_hopping_iters": args.basin_hopping_iters,
            "optimum_starts": args.optimum_starts,
            "stability_seeds": args.stability_seeds,
            "stability_starts": args.stability_starts,
            "max_observed_starts": args.max_observed_starts,
            "no_observed_starts": args.no_observed_starts,
        }
        for candidate in candidates
    ]
    args.summary_csv.parent.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    weights: list[pd.DataFrame] = []
    start_time = time.monotonic()
    with ProcessPoolExecutor(
        max_workers=max(1, int(args.workers)),
        initializer=_initializer,
        initargs=(raw_signal, metadata, aggregate_scores),
    ) as pool:
        futures = [pool.submit(_fit_candidate, task) for task in tasks]
        for index, future in enumerate(as_completed(futures), start=1):
            result, optimum_weights = future.result()
            results.append(result)
            if not optimum_weights.empty:
                weights.append(optimum_weights)
            print(f"[{index}/{len(futures)}] {result['candidate']} {result['fit_status']}", flush=True)

    summary = pd.DataFrame(results).sort_values("candidate")
    summary.to_csv(args.summary_csv, index=False)
    if weights:
        pd.concat(weights, ignore_index=True).to_csv(args.weights_csv, index=False)
    else:
        pd.DataFrame().to_csv(args.weights_csv, index=False)
    payload = {
        "candidates": candidates,
        "elapsed_seconds": time.monotonic() - start_time,
        "rows": len(summary),
        "status_counts": summary["fit_status"].value_counts(dropna=False).to_dict(),
        "workers": int(args.workers),
        "maxiter": int(args.maxiter),
        "optimum_starts": int(args.optimum_starts),
        "stability_seeds": int(args.stability_seeds),
        "stability_starts": int(args.stability_starts),
        "max_observed_starts": int(args.max_observed_starts),
    }
    args.summary_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
