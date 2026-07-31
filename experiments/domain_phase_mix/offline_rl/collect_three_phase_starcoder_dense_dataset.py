# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Collect dense three-phase StarCoder telemetry for offline-control v3."""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from experiments.domain_phase_mix.offline_rl.collect_three_phase_starcoder_dataset import (
    _load_phase_weights_from_csv,
    _load_phase_weights_from_weight_configs,
    build_wide_history,
    dedupe_history_rows,
    infer_local_run_id,
    infer_source_experiment,
    phase_weights_to_columns,
)
from experiments.domain_phase_mix.offline_rl.contracts import DEFAULT_OBJECTIVE_METRIC

logger = logging.getLogger(__name__)

DEFAULT_DISPLAY_NAME_PREFIX = "pinlin_calvin_xu/data_mixture/three_phase_starcoder"
DEFAULT_SOURCE_EXPERIMENTS = (
    "pinlin_calvin_xu/data_mixture/three_phase_starcoder_1",
    "pinlin_calvin_xu/data_mixture/three_phase_starcoder_2",
)
EXPECTED_RUN_COUNT = 160
DEFAULT_HISTORY_GROUPS: dict[str, tuple[str, ...]] = {
    "train": ("train/loss", "optim/learning_rate", "optim/adam_lr"),
    "norm": ("grad/norm/total", "params/norm/total"),
    "eval": ("eval/loss", DEFAULT_OBJECTIVE_METRIC),
}


@dataclass(frozen=True)
class CollectThreePhaseDenseConfig:
    """Config for collecting dense three-phase StarCoder histories from W&B."""

    output_dir: str
    wandb_entity: str = "marin-community"
    wandb_project: str = "marin"
    display_name_prefix: str = DEFAULT_DISPLAY_NAME_PREFIX
    objective_metric: str = DEFAULT_OBJECTIVE_METRIC
    expected_run_count: int = EXPECTED_RUN_COUNT
    csv_fallback_path: str = "experiments/domain_phase_mix/exploratory/three_phase_starcoder.csv"
    source_experiments: tuple[str, ...] = DEFAULT_SOURCE_EXPERIMENTS
    history_groups: dict[str, tuple[str, ...]] = field(
        default_factory=lambda: {name: tuple(keys) for name, keys in DEFAULT_HISTORY_GROUPS.items()}
    )
    wandb_timeout_seconds: int = 45
    run_query_retry_attempts: int = 4
    run_query_retry_backoff_seconds: float = 5.0
    history_retry_attempts: int = 4
    history_retry_backoff_seconds: float = 3.0


def _extract_numeric_summary(summary: Any, key: str) -> float | None:
    value = summary.get(key) if hasattr(summary, "get") else None
    if isinstance(value, int | float):
        return float(value)
    return None


def _extract_numeric_config(config: Any, key: str) -> float | int | None:
    if hasattr(config, "get"):
        value = config.get(key)
    elif isinstance(config, dict):
        value = config.get(key)
    else:
        value = None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int | float):
        return value
    return None


def _fetch_finished_runs(api, config: CollectThreePhaseDenseConfig):
    regex = rf"^{re.escape(config.display_name_prefix)}"
    filters = {"display_name": {"$regex": regex}, "state": "finished"}
    for attempt in range(1, config.run_query_retry_attempts + 1):
        try:
            runs = list(api.runs(f"{config.wandb_entity}/{config.wandb_project}", filters=filters))
            break
        except Exception:
            logger.warning("api.runs failed attempt=%d/%d", attempt, config.run_query_retry_attempts, exc_info=True)
            if attempt == config.run_query_retry_attempts:
                raise
            time.sleep(config.run_query_retry_backoff_seconds * attempt)
    runs = [run for run in runs if infer_source_experiment(run.display_name) in set(config.source_experiments)]
    runs.sort(key=lambda run: run.display_name)
    if config.expected_run_count > 0 and len(runs) != config.expected_run_count:
        raise ValueError(f"Expected {config.expected_run_count} runs but found {len(runs)}")
    return runs


def _scan_history_rows(
    run,
    *,
    group_name: str,
    keys: tuple[str, ...],
    retry_attempts: int,
    backoff_seconds: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    scan_index = 0
    for attempt in range(1, retry_attempts + 1):
        try:
            entries = list(run.scan_history(keys=["_step", *keys]))
            break
        except Exception:
            logger.warning(
                "scan_history failed for run=%s group=%s attempt=%d/%d",
                run.id,
                group_name,
                attempt,
                retry_attempts,
                exc_info=True,
            )
            if attempt == retry_attempts:
                raise
            time.sleep(backoff_seconds * attempt)
    for entry in entries:
        step = entry.get("_step")
        if step is None or pd.isna(step):
            continue
        for metric_key in keys:
            value = entry.get(metric_key)
            if value is None or pd.isna(value):
                continue
            rows.append(
                {
                    "wandb_run_id": run.id,
                    "source_experiment": infer_source_experiment(run.display_name),
                    "local_run_id": infer_local_run_id(run.display_name),
                    "run_name": run.display_name,
                    "step": int(step),
                    "total_tokens": None,
                    "metric_key": metric_key,
                    "metric_value": float(value),
                    "history_group": group_name,
                    "_scan_index": scan_index,
                }
            )
            scan_index += 1
    return rows


def _merge_group_histories(group_frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    base_cols = ["wandb_run_id", "source_experiment", "local_run_id", "run_name", "step"]
    merged: pd.DataFrame | None = None
    for frame in group_frames.values():
        if frame.empty:
            continue
        part = frame.copy()
        if "total_tokens" in part.columns:
            part = part.drop(columns=["total_tokens"])
        metric_cols = [column for column in part.columns if column not in base_cols]
        for column in metric_cols:
            part[column] = part[column].astype(float)
        if merged is None:
            merged = part
        else:
            merged = merged.merge(part, on=base_cols, how="outer")
    if merged is None:
        return pd.DataFrame(columns=base_cols)
    return merged.sort_values(["wandb_run_id", "step"]).reset_index(drop=True)


def collect_dense_history_from_run(
    run,
    *,
    objective_metric: str = DEFAULT_OBJECTIVE_METRIC,
    retry_attempts: int = 4,
    backoff_seconds: float = 3.0,
) -> pd.DataFrame:
    """Collect cadence-aware dense history for one completed W&B run."""
    group_frames: dict[str, pd.DataFrame] = {}
    history_groups = {
        "train": ("train/loss", "optim/learning_rate", "optim/adam_lr"),
        "norm": ("grad/norm/total", "params/norm/total"),
        "eval": ("eval/loss", objective_metric),
    }
    for group_name, keys in history_groups.items():
        rows = _scan_history_rows(
            run,
            group_name=group_name,
            keys=keys,
            retry_attempts=retry_attempts,
            backoff_seconds=backoff_seconds,
        )
        long_df = dedupe_history_rows(pd.DataFrame(rows))
        group_frames[group_name] = build_wide_history(long_df)
    return _merge_group_histories(group_frames)


def collect_three_phase_dense_dataset(
    config: CollectThreePhaseDenseConfig,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], pd.DataFrame]:
    """Collect run metadata plus cadence-aware dense histories from W&B."""
    import wandb

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    api = wandb.Api(timeout=config.wandb_timeout_seconds)
    wb_runs = _fetch_finished_runs(api, config)

    gcs_prefix = os.environ.get("MARIN_PREFIX", "gs://marin-us-central1")
    map_from_configs = _load_phase_weights_from_weight_configs(
        source_experiments=config.source_experiments,
        gcs_prefix=gcs_prefix,
    )
    map_from_csv = _load_phase_weights_from_csv(config.csv_fallback_path)

    run_rows: list[dict[str, Any]] = []
    history_rows_by_group: dict[str, list[dict[str, Any]]] = {name: [] for name in config.history_groups}

    for index, wb_run in enumerate(wb_runs, start=1):
        if index == 1 or index % 10 == 0:
            logger.info("Collecting run %d/%d (%s)", index, len(wb_runs), wb_run.id)

        source_experiment = infer_source_experiment(wb_run.display_name)
        local_run_id = infer_local_run_id(wb_run.display_name)
        phase_weights = map_from_configs.get((source_experiment, local_run_id), {}) if local_run_id is not None else {}
        if not phase_weights:
            phase_weights = map_from_csv.get(wb_run.id, {})

        for group_name, keys in config.history_groups.items():
            history_rows_by_group[group_name].extend(
                _scan_history_rows(
                    wb_run,
                    group_name=group_name,
                    keys=keys,
                    retry_attempts=config.history_retry_attempts,
                    backoff_seconds=config.history_retry_backoff_seconds,
                )
            )

        objective_summary = _extract_numeric_summary(wb_run.summary, config.objective_metric)
        run_row = {
            "wandb_run_id": wb_run.id,
            "run_name": wb_run.display_name,
            "source_experiment": source_experiment,
            "run_family": "three_phase_starcoder",
            "local_run_id": local_run_id,
            "num_phases_total": 3,
            "total_steps": 5722,
            "phase_boundaries_json": json.dumps([1888, 3824]),
            "status": wb_run.state,
            config.objective_metric: objective_summary,
            "data_seed": _extract_numeric_config(wb_run.config, "data_seed"),
        }
        run_row.update(phase_weights_to_columns(phase_weights))
        run_rows.append(run_row)

    runs_df = pd.DataFrame(run_rows)
    group_frames: dict[str, pd.DataFrame] = {}
    for group_name, rows in history_rows_by_group.items():
        long_df = dedupe_history_rows(pd.DataFrame(rows))
        wide_df = build_wide_history(long_df)
        group_frames[group_name] = wide_df
        long_df.to_parquet(output_dir / f"history_{group_name}_long.parquet", index=False)
        wide_df.to_parquet(output_dir / f"history_{group_name}.parquet", index=False)

    history_dense_wide = _merge_group_histories(group_frames)

    runs_df.to_parquet(output_dir / "runs.parquet", index=False)
    history_dense_wide.to_parquet(output_dir / "history_dense_wide.parquet", index=False)

    summary = {
        "objective_metric": config.objective_metric,
        "display_name_prefix": config.display_name_prefix,
        "history_groups": {name: list(keys) for name, keys in config.history_groups.items()},
        "n_runs": len(runs_df),
        "group_row_counts": {name: len(frame) for name, frame in group_frames.items()},
        "dense_columns": sorted(history_dense_wide.columns.tolist()),
    }
    with (output_dir / "collector_summary.json").open("w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    return runs_df, group_frames, history_dense_wide


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect dense three-phase StarCoder histories from W&B.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments/domain_phase_mix/offline_rl/artifacts/three_phase_starcoder_dense_v3",
        help="Directory for parquet outputs.",
    )
    parser.add_argument("--objective-metric", type=str, default=DEFAULT_OBJECTIVE_METRIC)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = _parse_args()
    collect_three_phase_dense_dataset(
        CollectThreePhaseDenseConfig(
            output_dir=args.output_dir,
            objective_metric=args.objective_metric,
        )
    )


if __name__ == "__main__":
    main()
