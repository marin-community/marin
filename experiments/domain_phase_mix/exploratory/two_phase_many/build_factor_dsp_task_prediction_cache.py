# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Build dashboard task-delta predictions for the factor-DSP candidate library."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd

from experiments.domain_phase_mix.exploratory.two_phase_many.factor_dsp_constraint_dashboard_helpers import (
    ridge_task_delta_prediction_wide,
)

TWO_PHASE_ROOT = Path("experiments/domain_phase_mix/exploratory/two_phase_many")
REPRO_ROOT = TWO_PHASE_ROOT / "reference_outputs" / "collaborator_grug_v4_aggregate_repro_20260525"
CURRENT_AGG_DIR = REPRO_ROOT / "sent_raw_metric_matrix_300m_zip"
SENT_MATRIX_CSV = REPRO_ROOT / "sent_zip_input" / "raw_metric_matrix_300m" / "raw_metric_matrix_300m.csv"
NOISE_CSV = REPRO_ROOT / "sent_zip_input" / "raw_metric_matrix_300m" / "noise_baseline_run00097_300m.csv"
CANDIDATE_LIBRARY_DIR = TWO_PHASE_ROOT / "reference_outputs" / "factor_dsp_candidate_library_y_factor_20260526"
ENDPOINT_DISCOVERY_DIR = CANDIDATE_LIBRARY_DIR / "endpoint_discovery"
TASK_PREDICTION_WIDE = CANDIDATE_LIBRARY_DIR / "task_prediction_wide.parquet"
TASK_PREDICTION_METRICS_CSV = CANDIDATE_LIBRARY_DIR / "task_prediction_metrics.csv"
TASK_PREDICTION_METRICS_PARQUET = CANDIDATE_LIBRARY_DIR / "task_prediction_metrics.parquet"
TASK_PREDICTION_SUMMARY_JSON = CANDIDATE_LIBRARY_DIR / "task_prediction_summary.json"
RIDGE_ALPHA = 1.0


def dashboard_candidate_name(candidate_id: str) -> str:
    """Map cache candidate IDs to the dashboard's candidate namespace."""
    if candidate_id.startswith("observed_"):
        return candidate_id.removeprefix("observed_")
    if candidate_id.startswith("named_"):
        return candidate_id.removeprefix("named_")
    return candidate_id


def load_candidate_weights() -> pd.DataFrame:
    """Load all candidate weights that should receive task-delta predictions."""
    paths = [
        CANDIDATE_LIBRARY_DIR / "candidate_weights_wide.parquet",
        ENDPOINT_DISCOVERY_DIR / "endpoint_path_weights_wide.parquet",
    ]
    frames = [pd.read_parquet(path) for path in paths if path.exists()]
    if not frames:
        raise FileNotFoundError("no candidate weight cache files found")
    weights = pd.concat(frames, ignore_index=True, sort=False)
    weights["original_candidate_id"] = weights["candidate_id"].astype(str)
    weights["candidate_id"] = weights["original_candidate_id"].map(dashboard_candidate_name)
    if weights["candidate_id"].duplicated().any():
        duplicates = weights.loc[weights["candidate_id"].duplicated(), "candidate_id"].unique().tolist()
        raise ValueError(f"duplicate dashboard candidate names after normalization: {duplicates[:10]}")
    return weights


def main() -> None:
    started_at = time.time()
    print("Loading 300M signal, selected tasks, and candidate weights...", flush=True)
    signal_frame = pd.read_csv(SENT_MATRIX_CSV, low_memory=False)
    if "row_kind" in signal_frame.columns:
        signal_frame = signal_frame.loc[signal_frame["row_kind"].eq("signal")].copy()
    selected_tasks = pd.read_csv(CURRENT_AGG_DIR / "selected_tasks.csv")
    noise_frame = pd.read_csv(NOISE_CSV, low_memory=False) if NOISE_CSV.exists() else None
    candidate_weights = load_candidate_weights()
    print(
        f"Fitting local ridge task surrogate for {len(selected_tasks)} tasks "
        f"and predicting {len(candidate_weights):,} candidates...",
        flush=True,
    )
    predictions, metrics = ridge_task_delta_prediction_wide(
        signal_frame,
        selected_tasks,
        candidate_weights,
        noise_frame=noise_frame,
        alpha=RIDGE_ALPHA,
    )
    print("Writing task prediction cache...", flush=True)
    predictions.to_parquet(TASK_PREDICTION_WIDE, index=False)
    metrics.to_csv(TASK_PREDICTION_METRICS_CSV, index=False)
    metrics.to_parquet(TASK_PREDICTION_METRICS_PARQUET, index=False)
    summary = {
        "prediction_source": "local_weight_ridge",
        "ridge_alpha": RIDGE_ALPHA,
        "num_candidates": len(predictions),
        "num_tasks": len(selected_tasks),
        "task_prediction_wide_parquet": str(TASK_PREDICTION_WIDE),
        "task_prediction_metrics_csv": str(TASK_PREDICTION_METRICS_CSV),
        "task_prediction_metrics_parquet": str(TASK_PREDICTION_METRICS_PARQUET),
        "elapsed_seconds": time.time() - started_at,
    }
    TASK_PREDICTION_SUMMARY_JSON.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
