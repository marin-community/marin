# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.11"
# dependencies = ["pandas"]
# ///
"""Build proportional variable-subset noise matrix exports.

This is intentionally narrower than ``build_raw_metric_matrix_300m.py``: it
does not replace the canonical 300M signal matrix. It only exports the new
proportional-anchor noise panels at each scale so collaborator-facing matrices
and SNR scripts can choose this noise anchor explicitly.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from experiments.domain_phase_mix.exploratory.two_phase_many.build_eval_signal_to_noise_all_metrics_300m import (
    METRICS_WIDE_CSV,
    _add_mmlu_category_smooth_metrics,
    _default_extra_results_csvs,
    _latest_eval_metrics_row,
    _overlay_metrics,
    _read_csv,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.metric_registry.build_raw_metric_matrix_300m import (
    _write_matrix,
)
from experiments.domain_phase_mix.launch_proportional_variable_subset_noise_baseline import (
    ANCHOR_RUN_NAME,
    DEFAULT_SCALES,
    N_SEED_SWEEP_RUNS,
    SEED_SWEEP_START,
    family_for_scale,
    source_experiment_for_scale,
)

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "raw_metric_matrix_300m"
ROW_KIND = "noise_variable_subset_proportional"
NOISE_SUBSET_MODE = "proportional_variable"
TARGET_FINAL_CHECKPOINT_STEP_BY_SCALE = {
    "60m_1p2b": 4576,
    "300m_6b": 22887,
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Write placeholder/partial files instead of requiring 10 rows per scale.",
    )
    return parser.parse_args()


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if pd.isna(value):
        return None
    return value


def _load_metrics_with_overlays() -> pd.DataFrame:
    if not METRICS_WIDE_CSV.exists():
        raise FileNotFoundError(f"Missing metric registry: {METRICS_WIDE_CSV}")
    frame = pd.read_csv(METRICS_WIDE_CSV, low_memory=False)
    for path in _default_extra_results_csvs():
        frame = _overlay_metrics(frame, _read_csv(path), key_column="checkpoint_root")
    return _add_mmlu_category_smooth_metrics(frame)


def _hydrate_training_eval_metrics(frame: pd.DataFrame, *, scale: str) -> pd.DataFrame:
    target_step = TARGET_FINAL_CHECKPOINT_STEP_BY_SCALE[scale]
    rows: list[dict[str, Any]] = []
    for _, row in frame.iterrows():
        out = row.to_dict()
        checkpoint_root = row.get("checkpoint_root")
        if not isinstance(checkpoint_root, str) or checkpoint_root.strip() == "":
            raise ValueError(f"Missing checkpoint_root for proportional noise row {row.get('run_name')}")
        metrics = _latest_eval_metrics_row(
            f"{checkpoint_root.rstrip('/')}/checkpoints/eval_metrics.jsonl",
            required_step=target_step,
        )
        for key, value in metrics.items():
            if not key.startswith("eval/") or not isinstance(value, int | float):
                continue
            if key not in out or pd.isna(out[key]):
                out[key] = float(value)
        rows.append(out)
    return pd.DataFrame.from_records(rows)


def _scale_noise_frame(metrics: pd.DataFrame, scale: str) -> pd.DataFrame:
    source_experiment = source_experiment_for_scale(scale)
    frame = metrics[
        metrics["scale"].eq(scale)
        & metrics["cohort"].eq("seed_sweep")
        & metrics["source_experiment"].eq(source_experiment)
    ].copy()
    if frame.empty:
        return frame

    frame["row_kind"] = ROW_KIND
    frame["noise_subset_mode"] = NOISE_SUBSET_MODE
    frame["noise_anchor_run_name"] = ANCHOR_RUN_NAME
    frame["noise_source_run_name"] = ANCHOR_RUN_NAME
    if "noise_trainer_seed" not in frame.columns:
        frame["noise_trainer_seed"] = pd.to_numeric(frame.get("trainer_seed"), errors="coerce")
    if "noise_data_seed" not in frame.columns:
        frame["noise_data_seed"] = pd.NA
    if "noise_simulated_epoch_subset_seed" not in frame.columns:
        frame["noise_simulated_epoch_subset_seed"] = pd.NA
    frame["noise_data_seed"] = pd.NA
    frame["noise_simulated_epoch_subset_seed"] = pd.NA
    frame["source_cohort"] = frame.get("source_cohort", pd.Series(pd.NA, index=frame.index)).fillna("seed_sweep")
    frame["cohort"] = frame.get("cohort", pd.Series(pd.NA, index=frame.index)).fillna("seed_sweep")
    # Local checkpoint/eval audits are fresher than the stale logical registry
    # status for this seed sweep.
    frame["status"] = "completed"
    frame["family"] = family_for_scale(scale)
    return _hydrate_training_eval_metrics(frame, scale=scale)


def _expected_empty_frame(metrics: pd.DataFrame) -> pd.DataFrame:
    return metrics.iloc[0:0].copy()


def main() -> None:
    args = _parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics = _load_metrics_with_overlays()

    outputs: dict[str, dict[str, Any]] = {}
    for scale in DEFAULT_SCALES:
        frame = _scale_noise_frame(metrics, scale)
        output_path = args.output_dir / f"noise_baseline_proportional_variable_subset_{scale}.csv"
        if len(frame) != N_SEED_SWEEP_RUNS:
            if not args.allow_incomplete:
                raise ValueError(
                    f"Expected {N_SEED_SWEEP_RUNS} proportional variable-subset noise rows for {scale}, "
                    f"found {len(frame)}"
                )
            empty = _expected_empty_frame(metrics) if frame.empty else frame
            empty.to_csv(output_path, index=False)
            outputs[scale] = {
                "rows": len(frame),
                "path": str(output_path),
                "status": "incomplete",
                "expected_rows": N_SEED_SWEEP_RUNS,
                "trainer_seed_start": SEED_SWEEP_START,
            }
            continue
        outputs[scale] = {
            **_write_matrix(frame, output_path),
            "status": "complete",
            "expected_rows": N_SEED_SWEEP_RUNS,
            "trainer_seed_start": SEED_SWEEP_START,
        }

    summary_path = args.output_dir / "proportional_variable_subset_noise_summary.json"
    summary_path.write_text(json.dumps(_json_safe(outputs), indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(_json_safe(outputs), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
