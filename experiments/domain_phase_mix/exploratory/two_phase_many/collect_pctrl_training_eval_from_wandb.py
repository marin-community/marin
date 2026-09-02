# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.11"
# dependencies = ["pandas", "wandb"]
# ///
"""Collect training-time eval/* metrics for 300M pctrl runs from W&B.

The proportional-controllability downstream matrix was assembled from separate
eval collectors and missed the training-time eval summaries, including
``eval/uncheatable_eval/*``.  Those summaries are present on the original
training W&B runs whose IDs are encoded in the checkpoint root leaf:

    gs://.../<run_name>-<hash>

This script pulls scalar ``eval/*`` summary keys for every pctrl row and writes
both a standalone collection table and an augmented pctrl metric matrix.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import pandas as pd
import wandb

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_MATRIX = (
    SCRIPT_DIR
    / "reference_outputs"
    / "proportional_controllability_log_tilt_analysis_20260609"
    / "pctrl_final_metric_matrix.csv"
)
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "reference_outputs" / "pctrl_training_eval_wandb_collect_20260623"
DEFAULT_ENTITY = "marin-community"
DEFAULT_PROJECT = "marin"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-matrix", type=Path, default=DEFAULT_INPUT_MATRIX)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--entity", default=DEFAULT_ENTITY)
    parser.add_argument("--project", default=DEFAULT_PROJECT)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    return parser.parse_args()


def checkpoint_run_id(checkpoint_root: str) -> str:
    leaf = checkpoint_root.rstrip("/").rsplit("/", maxsplit=1)[-1]
    if not leaf:
        raise ValueError(f"Could not parse W&B run id from checkpoint root: {checkpoint_root}")
    return leaf


def scalar_or_none(value: Any) -> float | str | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, int | float):
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return value
    if isinstance(value, str):
        return value
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric):
        return None
    return numeric


def eval_summary(run: wandb.apis.public.Run) -> dict[str, Any]:
    summary = dict(run.summary)
    values: dict[str, Any] = {}
    for key, value in summary.items():
        if not key.startswith("eval/"):
            continue
        scalar = scalar_or_none(value)
        if scalar is not None:
            values[key] = scalar
    return values


def collect(
    input_matrix: Path, output_dir: Path, entity: str, project: str, limit: int | None, sleep_seconds: float
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    matrix = pd.read_csv(input_matrix, low_memory=False)
    if limit is not None:
        matrix = matrix.head(limit).copy()

    api = wandb.Api(timeout=90)
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    run_path_prefix = f"{entity}/{project}"
    for index, row in matrix.iterrows():
        run_name = str(row["run_name"])
        wandb_run_id = checkpoint_run_id(str(row["checkpoint_root"]))
        print(f"[{index + 1:03d}/{len(matrix):03d}] {run_name} -> {wandb_run_id}", flush=True)
        record: dict[str, Any] = {
            "run_name": run_name,
            "checkpoint_root": row["checkpoint_root"],
            "wandb_training_run_id": wandb_run_id,
            "wandb_training_run_path": f"{run_path_prefix}/{wandb_run_id}",
        }
        try:
            run = api.run(record["wandb_training_run_path"])
            record.update(
                {
                    "wandb_training_run_name": run.name,
                    "wandb_training_run_state": run.state,
                    "wandb_training_run_url": str(run.url),
                }
            )
            values = eval_summary(run)
            record["training_eval_metric_count"] = len(values)
            record.update(values)
        except Exception as exc:  # noqa: BLE001 - record per-row API failures for audit.
            record["training_eval_metric_count"] = 0
            record["collection_error"] = f"{type(exc).__name__}: {exc}"
            failures.append(
                {"run_name": run_name, "wandb_training_run_id": wandb_run_id, "error": record["collection_error"]}
            )
        rows.append(record)
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

    collected = pd.DataFrame(rows)
    collect_path = output_dir / "pctrl_training_eval_wandb_summary.csv"
    collected.to_csv(collect_path, index=False)

    eval_columns = sorted(column for column in collected.columns if column.startswith("eval/"))
    merge_columns = [
        "run_name",
        "wandb_training_run_id",
        "wandb_training_run_url",
        "training_eval_metric_count",
        *eval_columns,
    ]
    augmented = pd.read_csv(input_matrix, low_memory=False)
    preexisting_eval_columns = [column for column in eval_columns if column in augmented.columns]
    if preexisting_eval_columns:
        augmented = augmented.drop(columns=preexisting_eval_columns)
    augmented = augmented.merge(collected[merge_columns], on="run_name", how="left", validate="one_to_one")
    augmented_path = output_dir / "pctrl_final_metric_matrix_with_training_eval.csv"
    augmented.to_csv(augmented_path, index=False)

    coverage = {
        "input_matrix": str(input_matrix),
        "collection_csv": str(collect_path),
        "augmented_matrix": str(augmented_path),
        "rows": int(len(augmented)),
        "eval_metric_columns": int(len(eval_columns)),
        "uncheatable_eval_metric_columns": int(sum("uncheatable_eval" in column for column in eval_columns)),
        "rows_with_any_training_eval": int((collected["training_eval_metric_count"] > 0).sum()),
        "failures": failures,
    }
    (output_dir / "summary.json").write_text(json.dumps(coverage, indent=2, sort_keys=True))
    return coverage


def main() -> None:
    args = parse_args()
    coverage = collect(args.input_matrix, args.output_dir, args.entity, args.project, args.limit, args.sleep_seconds)
    print(json.dumps(coverage, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
