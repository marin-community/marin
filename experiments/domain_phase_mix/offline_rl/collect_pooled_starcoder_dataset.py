# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Collect a pooled offline-control dataset from finished StarCoder mixture runs."""

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
from experiments.domain_phase_mix.offline_rl.contracts import (
    DEFAULT_OBJECTIVE_METRIC,
    PooledDatasetConfig,
    default_pooled_dataset_config,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CollectPooledStarCoderConfig:
    """Config for collecting the pooled v2 StarCoder dataset from W&B."""

    output_dir: str
    dataset_config: PooledDatasetConfig = field(default_factory=default_pooled_dataset_config)
    wandb_entity: str = "marin-community"
    wandb_project: str = "marin"
    wandb_timeout_seconds: int = 45
    run_query_retry_attempts: int = 4
    run_query_retry_backoff_seconds: float = 5.0
    history_batch_size: int = 4
    history_retry_attempts: int = 4
    history_retry_backoff_seconds: float = 3.0
    max_runs_per_family: int = 0


def _chunked(values: tuple[str, ...], chunk_size: int) -> list[tuple[str, ...]]:
    return [values[index : index + chunk_size] for index in range(0, len(values), chunk_size)]


def _resolve_phase_weights(
    *,
    map_from_configs: dict[tuple[str, int], dict[str, dict[str, float]]],
    map_from_csv: dict[str, dict[str, dict[str, float]]],
    source_experiment: str,
    local_run_id: int | None,
    wandb_run_id: str,
) -> dict[str, dict[str, float]]:
    if local_run_id is not None:
        candidates = [local_run_id]
        if local_run_id >= 90000:
            candidates.append(local_run_id - 90000)
        else:
            candidates.append(local_run_id + 90000)
        for candidate in candidates:
            phase_weights = map_from_configs.get((source_experiment, candidate), {})
            if phase_weights:
                return phase_weights
    return map_from_csv.get(wandb_run_id, {})


def _fetch_finished_runs(
    api,
    entity: str,
    project: str,
    family,
    max_runs_per_family: int,
    attempts: int,
    backoff_seconds: float,
):
    regex = rf"^{re.escape(family.display_name_prefix)}"
    filters = {"display_name": {"$regex": regex}, "state": "finished"}
    for attempt in range(1, attempts + 1):
        try:
            runs = list(api.runs(f"{entity}/{project}", filters=filters))
            break
        except Exception:
            logger.warning(
                "api.runs failed for family=%s attempt=%d/%d",
                family.run_family,
                attempt,
                attempts,
                exc_info=True,
            )
            if attempt == attempts:
                raise
            time.sleep(backoff_seconds * attempt)
    runs = [run for run in runs if infer_source_experiment(run.display_name) in set(family.source_experiments)]
    runs.sort(key=lambda run: run.display_name)
    if max_runs_per_family > 0:
        runs = runs[:max_runs_per_family]
    if family.expected_finished_runs > 0 and max_runs_per_family <= 0 and len(runs) != family.expected_finished_runs:
        raise ValueError(
            f"Expected {family.expected_finished_runs} finished runs for {family.run_family} but found {len(runs)}"
        )
    return runs


def _scan_history_batch(run, keys: tuple[str, ...], attempts: int, backoff_seconds: float) -> list[dict[str, Any]]:
    for attempt in range(1, attempts + 1):
        try:
            return list(run.scan_history(keys=["_step", *keys]))
        except Exception:
            logger.warning(
                "scan_history failed for run=%s keys=%s attempt=%d/%d",
                run.id,
                keys,
                attempt,
                attempts,
                exc_info=True,
            )
            if attempt == attempts:
                logger.error(
                    "Skipping history batch after exhausting retries for run=%s keys=%s",
                    run.id,
                    keys,
                )
                return []
            time.sleep(backoff_seconds * attempt)
    raise RuntimeError("unreachable")


def collect_history_long_rows_batched(
    run,
    metric_keys: tuple[str, ...],
    history_batch_size: int,
    retry_attempts: int,
    backoff_seconds: float,
) -> list[dict[str, Any]]:
    """Collect metric history rows from W&B using small-key scan_history batches."""
    rows: list[dict[str, Any]] = []
    scan_index = 0
    for batch in _chunked(metric_keys, max(1, history_batch_size)):
        entries = _scan_history_batch(run, batch, retry_attempts, backoff_seconds)
        for entry in entries:
            step = entry.get("_step")
            if step is None or pd.isna(step):
                continue
            total_tokens = entry.get("throughput/total_tokens")
            total_tokens_value = None
            if total_tokens is not None and not pd.isna(total_tokens):
                total_tokens_value = float(total_tokens)

            for metric_key in batch:
                if metric_key == "throughput/total_tokens":
                    continue
                metric_value = entry.get(metric_key)
                if metric_value is None or pd.isna(metric_value):
                    continue
                rows.append(
                    {
                        "wandb_run_id": run.id,
                        "source_experiment": infer_source_experiment(run.display_name),
                        "local_run_id": infer_local_run_id(run.display_name),
                        "run_name": run.display_name,
                        "step": int(step),
                        "total_tokens": total_tokens_value,
                        "metric_key": metric_key,
                        "metric_value": float(metric_value),
                        "_scan_index": scan_index,
                    }
                )
                scan_index += 1
    return rows


def collect_pooled_dataset(config: CollectPooledStarCoderConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Collect pooled two-phase and three-phase StarCoder run histories from W&B."""
    import wandb

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    api = wandb.Api(timeout=config.wandb_timeout_seconds)
    gcs_prefix = os.environ.get("MARIN_PREFIX", "gs://marin-us-central1")

    runs_rows: list[dict[str, Any]] = []
    history_rows: list[dict[str, Any]] = []
    family_summaries: list[dict[str, Any]] = []

    for family in config.dataset_config.run_families:
        wb_runs = _fetch_finished_runs(
            api,
            config.wandb_entity,
            config.wandb_project,
            family,
            config.max_runs_per_family,
            config.run_query_retry_attempts,
            config.run_query_retry_backoff_seconds,
        )
        family_summaries.append(
            {
                "run_family": family.run_family,
                "display_name_prefix": family.display_name_prefix,
                "n_finished_runs": len(wb_runs),
                "expected_finished_runs": family.expected_finished_runs,
            }
        )
        logger.info("Collecting %d runs for %s", len(wb_runs), family.run_family)

        map_from_configs = _load_phase_weights_from_weight_configs(
            source_experiments=family.source_experiments,
            gcs_prefix=gcs_prefix,
        )
        map_from_csv = _load_phase_weights_from_csv(family.csv_fallback_path) if family.csv_fallback_path else {}

        for index, wb_run in enumerate(wb_runs, start=1):
            if index == 1 or index % 10 == 0:
                logger.info("Collecting %s run %d/%d (%s)", family.run_family, index, len(wb_runs), wb_run.id)

            source_experiment = infer_source_experiment(wb_run.display_name)
            local_run_id = infer_local_run_id(wb_run.display_name)
            phase_weights = _resolve_phase_weights(
                map_from_configs=map_from_configs,
                map_from_csv=map_from_csv,
                source_experiment=source_experiment,
                local_run_id=local_run_id,
                wandb_run_id=wb_run.id,
            )

            history_rows.extend(
                collect_history_long_rows_batched(
                    wb_run,
                    metric_keys=config.dataset_config.candidate_history_keys,
                    history_batch_size=config.history_batch_size,
                    retry_attempts=config.history_retry_attempts,
                    backoff_seconds=config.history_retry_backoff_seconds,
                )
            )

            run_row = {
                "wandb_run_id": wb_run.id,
                "run_name": wb_run.display_name,
                "source_experiment": source_experiment,
                "run_family": family.run_family,
                "local_run_id": local_run_id,
                "num_phases_total": family.num_phases_total,
                "total_steps": family.total_steps,
                "phase_boundaries_json": json.dumps(list(family.phase_boundaries)),
                "status": wb_run.state,
                "objective_metric": config.dataset_config.objective_metric,
                config.dataset_config.objective_metric: None,
                "data_seed": None,
            }
            run_row.update(phase_weights_to_columns(phase_weights))
            runs_rows.append(run_row)

    runs_df = pd.DataFrame(runs_rows)
    history_long_df = dedupe_history_rows(pd.DataFrame(history_rows))
    history_wide_df = build_wide_history(history_long_df)

    runs_df.to_parquet(output_dir / "runs.parquet", index=False)
    history_long_df.to_parquet(output_dir / "history_long.parquet", index=False)
    history_wide_df.to_parquet(output_dir / "history_wide.parquet", index=False)

    summary = {
        "objective_metric": config.dataset_config.objective_metric,
        "candidate_history_keys": list(config.dataset_config.candidate_history_keys),
        "selected_feature_keys": list(config.dataset_config.selected_feature_keys),
        "families": family_summaries,
        "n_runs": len(runs_df),
        "n_history_rows": len(history_long_df),
        "history_columns": sorted(history_wide_df.columns.tolist()),
    }
    with (output_dir / "collector_summary.json").open("w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    return runs_df, history_long_df, history_wide_df


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect pooled StarCoder offline-control data from W&B.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments/domain_phase_mix/offline_rl/artifacts/starcoder_pooled_v2",
        help="Directory for pooled runs/history parquet outputs.",
    )
    parser.add_argument(
        "--objective-metric",
        type=str,
        default=DEFAULT_OBJECTIVE_METRIC,
        help="Objective metric key.",
    )
    parser.add_argument(
        "--max-runs-per-family",
        type=int,
        default=0,
        help="Optional debug limit on the number of finished runs collected per family.",
    )
    parser.add_argument(
        "--history-batch-size",
        type=int,
        default=4,
        help="Number of history keys to fetch per scan_history call.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = _parse_args()
    dataset_config = default_pooled_dataset_config(args.objective_metric)
    collect_pooled_dataset(
        CollectPooledStarCoderConfig(
            output_dir=args.output_dir,
            dataset_config=dataset_config,
            history_batch_size=args.history_batch_size,
            max_runs_per_family=args.max_runs_per_family,
        )
    )


if __name__ == "__main__":
    main()
