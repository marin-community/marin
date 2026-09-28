# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.11"
# dependencies = ["fsspec", "pandas"]
# ///
"""Collect training-time eval metrics for proportional perturbation checkpoints.

The perturbation downstream overlays cover lm-eval and follow-up eval jobs, but
the training-time `eval/*` slice metrics live in each checkpoint's
`eval_metrics.jsonl` and `tracker_metrics.jsonl`. This script materializes those
facts once into a local overlay that the metric registry can ingest quickly.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import fsspec
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PERTURBATION_DIR = SCRIPT_DIR / "proportional_perturbation_scale_transfer"
DEFAULT_CANDIDATES_CSV = PERTURBATION_DIR / "proportional_perturbation_eval_candidates.csv"
DEFAULT_OUTPUT_CSV = PERTURBATION_DIR / "ppert_training_eval_metrics.csv"
METRIC_PREFIX = "eval/"
METADATA_COLUMNS = (
    "panel",
    "scale",
    "run_name",
    "registry_key",
    "source_experiment",
    "cohort",
    "checkpoint_root",
    "expected_checkpoint_step",
    "intervention_id",
    "intervention_type",
    "target_unit",
    "target_domain",
    "target_family",
    "tv_distance",
)


def _last_jsonl_payload(path: str) -> dict[str, Any]:
    last_line = ""
    with fsspec.open(path, "rt") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                last_line = stripped
    if not last_line:
        raise ValueError(f"{path} is empty")
    payload = json.loads(last_line)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} did not contain a JSON object")
    return payload


def _numeric_eval_metrics(payload: dict[str, Any]) -> dict[str, float]:
    return {
        key: float(value)
        for key, value in payload.items()
        if key.startswith(METRIC_PREFIX) and isinstance(value, int | float)
    }


def _checkpoint_eval_metrics(checkpoint_root: str) -> dict[str, float]:
    metrics: dict[str, float] = {}

    eval_metrics_path = checkpoint_root.rstrip("/") + "/checkpoints/eval_metrics.jsonl"
    eval_payload = _last_jsonl_payload(eval_metrics_path)
    metrics.update(_numeric_eval_metrics(eval_payload))

    tracker_metrics_path = checkpoint_root.rstrip("/") + "/tracker_metrics.jsonl"
    try:
        tracker_payload = _last_jsonl_payload(tracker_metrics_path)
    except FileNotFoundError:
        return metrics
    summary = tracker_payload.get("summary", {})
    if isinstance(summary, dict):
        for key, value in _numeric_eval_metrics(summary).items():
            metrics.setdefault(key, value)
    return metrics


def collect_training_eval_metrics(
    *,
    candidates_csv: Path,
    output_csv: Path,
    allow_missing: bool,
) -> pd.DataFrame:
    candidates = pd.read_csv(candidates_csv, low_memory=False)
    missing_metadata = sorted(set(METADATA_COLUMNS) - set(candidates.columns))
    if missing_metadata:
        raise ValueError(f"{candidates_csv} missing columns: {missing_metadata}")

    records: list[dict[str, Any]] = []
    for _, row in candidates.iterrows():
        record = {column: row[column] for column in METADATA_COLUMNS}
        checkpoint_root = str(row["checkpoint_root"])
        try:
            metrics = _checkpoint_eval_metrics(checkpoint_root)
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            if not allow_missing:
                raise RuntimeError(f"Failed to collect metrics for {checkpoint_root}") from exc
            record["collection_status"] = "error"
            record["collection_error"] = str(exc)
        else:
            record.update(metrics)
            record["collection_status"] = "collected" if metrics else "missing_metrics"
            record["collection_error"] = ""
        records.append(record)

    frame = pd.DataFrame.from_records(records)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_csv, index=False)
    return frame


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates-csv", type=Path, default=DEFAULT_CANDIDATES_CSV)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--allow-missing", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    frame = collect_training_eval_metrics(
        candidates_csv=args.candidates_csv,
        output_csv=args.output_csv,
        allow_missing=args.allow_missing,
    )
    metric_columns = [column for column in frame.columns if column.startswith(METRIC_PREFIX)]
    collected = int(frame["collection_status"].eq("collected").sum())
    print(
        json.dumps(
            {
                "output_csv": str(args.output_csv),
                "rows": len(frame),
                "collected_rows": collected,
                "metric_columns": len(metric_columns),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
