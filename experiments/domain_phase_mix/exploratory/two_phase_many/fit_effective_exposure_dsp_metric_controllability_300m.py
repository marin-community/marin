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
"""Fit effective-exposure DSP controllability diagnostics for 300M metrics.

This is a companion to ``aggregate_metric_clean_slate_20260518.py``. The
notebook computes a cheap linear-ridge controllability score for every metric;
this script fits the nonlinear effective-exposure DSP comparator in parallel and
writes a cache that the notebook can merge reactively.

The default target set is one best metric per eval item from the current
reactive metric table. This avoids spending optimizer time on duplicate metric
variants while still covering the decision-level inventory.
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
from scipy.stats import pearsonr, spearmanr

from experiments.domain_phase_mix.exploratory.two_phase_many.standalone_code.dsp_exact import (
    VARIANTS,
    fit_variant,
    oof_predictions,
    packet_from_frame,
)

SCRIPT_DIR = Path(__file__).resolve().parent
RAW_MATRIX_CSV = SCRIPT_DIR / "metric_registry" / "raw_metric_matrix_300m" / "raw_metric_matrix_300m.csv"
METADATA_CSV = SCRIPT_DIR / "two_phase_many_epoch_metadata.csv"
NOTEBOOK_OUTPUT_DIR = SCRIPT_DIR / "reference_outputs" / "aggregate_metric_clean_slate_20260518"
DEFAULT_METRIC_TABLE_CSV = NOTEBOOK_OUTPUT_DIR / "reactive_metric_table_best_by_item.csv"
DEFAULT_OUTPUT_CSV = NOTEBOOK_OUTPUT_DIR / "metric_controllability_effective_exposure_dsp.csv"
DEFAULT_SUMMARY_JSON = NOTEBOOK_OUTPUT_DIR / "metric_controllability_effective_exposure_dsp_summary.json"

LOWER_IS_BETTER_KINDS = {"bpb", "loss", "nll", "perplexity", "bits"}
LOWER_IS_BETTER_SUFFIXES = ("/bpb", "/loss", "/nll", "/perplexity", "/bits")
HIGHER_IS_BETTER_SUFFIXES = ("/acc", "/acc_norm", "/exact_match", "/pass_at_1", "/choice_prob", "/choice_prob_norm")

_RAW_SIGNAL: pd.DataFrame | None = None
_METADATA: pd.DataFrame | None = None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-matrix-csv", type=Path, default=RAW_MATRIX_CSV)
    parser.add_argument("--metadata-csv", type=Path, default=METADATA_CSV)
    parser.add_argument("--metric-table-csv", type=Path, default=DEFAULT_METRIC_TABLE_CSV)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--summary-json", type=Path, default=DEFAULT_SUMMARY_JSON)
    parser.add_argument("--metric-source", choices=("best_by_item", "all_table"), default="best_by_item")
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--maxiter", type=int, default=36)
    parser.add_argument("--coarse-top-k", type=int, default=3)
    parser.add_argument("--basin-hopping-iters", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def _initializer(raw_signal: pd.DataFrame, metadata: pd.DataFrame) -> None:
    global _RAW_SIGNAL, _METADATA
    _RAW_SIGNAL = raw_signal
    _METADATA = metadata


def _metric_orientation(metric: str, metric_kind: str) -> str:
    metric = str(metric)
    metric_kind = str(metric_kind)
    if metric_kind in LOWER_IS_BETTER_KINDS or metric.endswith(LOWER_IS_BETTER_SUFFIXES):
        return "minimize"
    if metric.endswith(HIGHER_IS_BETTER_SUFFIXES):
        return "maximize"
    return "maximize"


def _finite_corr(y_true: np.ndarray, y_pred: np.ndarray, *, method: str) -> float:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if int(mask.sum()) < 3:
        return float("nan")
    if len(np.unique(y_true[mask])) < 2 or len(np.unique(y_pred[mask])) < 2:
        return float("nan")
    if method == "spearman":
        return float(spearmanr(y_true[mask], y_pred[mask]).statistic)
    if method == "pearson":
        return float(pearsonr(y_true[mask], y_pred[mask]).statistic)
    raise ValueError(f"Unknown correlation method: {method}")


def _fit_one(task: dict[str, Any]) -> dict[str, Any]:
    if _RAW_SIGNAL is None or _METADATA is None:
        raise RuntimeError("Worker globals were not initialized")

    metric = str(task["metric"])
    metric_kind = str(task["primary_metric_kind"])
    orientation = _metric_orientation(metric, metric_kind)
    start_time = time.monotonic()
    if metric not in _RAW_SIGNAL.columns:
        return {
            **task,
            "dsp_fit_status": "missing_metric",
            "dsp_fit_seconds": 0.0,
        }

    target = pd.to_numeric(_RAW_SIGNAL[metric], errors="coerce")
    if orientation == "maximize":
        target = -target
    fit_frame = _RAW_SIGNAL.loc[target.notna()].copy()
    fit_frame["objective_metric"] = target.loc[target.notna()].to_numpy(dtype=float)
    fit_n = len(fit_frame)
    if fit_n < 40:
        return {
            **task,
            "dsp_fit_status": "too_few_rows",
            "dsp_fit_seconds": time.monotonic() - start_time,
            "dsp_fit_n": fit_n,
        }
    target_std = float(fit_frame["objective_metric"].std(ddof=1))
    if not np.isfinite(target_std) or target_std <= 0:
        return {
            **task,
            "dsp_fit_status": "zero_target_variance",
            "dsp_fit_seconds": time.monotonic() - start_time,
            "dsp_fit_n": fit_n,
        }
    target_mean = float(fit_frame["objective_metric"].mean())
    fit_frame["objective_metric"] = (fit_frame["objective_metric"] - target_mean) / target_std

    try:
        packet = packet_from_frame(fit_frame, _METADATA)
        variant = VARIANTS["effective_exposure"]
        model, trace = fit_variant(
            packet,
            variant,
            maxiter=int(task["maxiter"]),
            coarse_top_k=int(task["coarse_top_k"]),
            basin_hopping_iters=int(task["basin_hopping_iters"]),
        )
        oof = oof_predictions(packet, model)
        residual = oof - packet.y
        sse = float(np.sum(residual**2))
        sst = float(np.sum((packet.y - np.mean(packet.y)) ** 2))
        result = {
            **task,
            "dsp_fit_status": "ok",
            "dsp_fit_seconds": time.monotonic() - start_time,
            "dsp_fit_n": fit_n,
            "dsp_variant": variant.name,
            "dsp_target_orientation": orientation,
            "dsp_controllability_score": _finite_corr(packet.y, oof, method="spearman"),
            "dsp_oof_spearman": _finite_corr(packet.y, oof, method="spearman"),
            "dsp_oof_pearson": _finite_corr(packet.y, oof, method="pearson"),
            "dsp_oof_r2": float("nan") if sst <= 0 else float(1.0 - sse / sst),
            "dsp_oof_rmse_z": float(np.sqrt(np.mean(residual**2))),
            "dsp_train_objective": (
                float(trace.loc[trace["stage"].eq("refine"), "objective"].min())
                if "stage" in trace.columns and trace["stage"].eq("refine").any()
                else float("nan")
            ),
            "dsp_gamma": float(model.params.get("gamma", np.nan)),
            "dsp_total_param_count": model.total_param_count,
        }
        return result
    except Exception as exc:
        return {
            **task,
            "dsp_fit_status": "error",
            "dsp_error": repr(exc),
            "dsp_fit_seconds": time.monotonic() - start_time,
            "dsp_fit_n": fit_n,
        }


def _metric_tasks(metric_table: pd.DataFrame, args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.metric_source == "best_by_item":
        source = metric_table.copy()
    else:
        source = metric_table.drop_duplicates("metric").copy()
    if args.limit is not None:
        source = source.head(args.limit).copy()
    tasks = []
    for _, row in source.iterrows():
        tasks.append(
            {
                "metric": str(row["metric"]),
                "item_id": str(row.get("item_id", "")),
                "suite": str(row.get("suite", "")),
                "source_class": str(row.get("source_class", "")),
                "primary_metric_kind": str(row.get("primary_metric_kind", "")),
                "recommended_role": str(row.get("recommended_role", "")),
                "ridge_controllability_score": row.get("controllability_score", np.nan),
                "ridge_oof_r2": row.get("controllability_oof_r2", np.nan),
                "maxiter": args.maxiter,
                "coarse_top_k": args.coarse_top_k,
                "basin_hopping_iters": args.basin_hopping_iters,
            }
        )
    return tasks


def main() -> None:
    args = _parse_args()
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

    raw = pd.read_csv(args.raw_matrix_csv, low_memory=False)
    raw_signal = raw[raw["status"].eq("completed") & raw["row_kind"].eq("signal")].reset_index(drop=True)
    metadata = pd.read_csv(args.metadata_csv)
    metric_table = pd.read_csv(args.metric_table_csv)
    tasks = _metric_tasks(metric_table, args)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)

    start_time = time.monotonic()
    results: list[dict[str, Any]] = []
    max_workers = max(1, int(args.workers))
    with ProcessPoolExecutor(max_workers=max_workers, initializer=_initializer, initargs=(raw_signal, metadata)) as pool:
        futures = [pool.submit(_fit_one, task) for task in tasks]
        for index, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            results.append(result)
            if index % 10 == 0 or index == len(futures):
                print(f"[{index}/{len(futures)}] completed", flush=True)

    out = pd.DataFrame(results).sort_values(["recommended_role", "dsp_controllability_score"], ascending=[True, False])
    out.to_csv(args.output_csv, index=False)
    summary = {
        "rows": len(out),
        "workers": max_workers,
        "metric_source": args.metric_source,
        "maxiter": args.maxiter,
        "coarse_top_k": args.coarse_top_k,
        "basin_hopping_iters": args.basin_hopping_iters,
        "elapsed_seconds": time.monotonic() - start_time,
        "status_counts": out["dsp_fit_status"].value_counts(dropna=False).to_dict(),
        "median_dsp_spearman": float(out["dsp_oof_spearman"].median(skipna=True)),
        "median_ridge_spearman": float(
            pd.to_numeric(out["ridge_controllability_score"], errors="coerce").median(skipna=True)
        ),
    }
    args.summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
