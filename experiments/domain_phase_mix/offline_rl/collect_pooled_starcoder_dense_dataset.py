# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Collect a pooled dense StarCoder telemetry dataset for offline-control v4."""

from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from experiments.domain_phase_mix.offline_rl.collect_pooled_starcoder_dataset import (
    _fetch_finished_runs,
    _resolve_phase_weights,
)
from experiments.domain_phase_mix.offline_rl.collect_three_phase_starcoder_dataset import (
    _load_phase_weights_from_csv,
    _load_phase_weights_from_weight_configs,
    infer_local_run_id,
    infer_source_experiment,
    phase_weights_to_columns,
)
from experiments.domain_phase_mix.offline_rl.collect_three_phase_starcoder_dense_dataset import (
    _extract_numeric_config,
    _extract_numeric_summary,
    collect_dense_history_from_run,
)
from experiments.domain_phase_mix.offline_rl.contracts import (
    DEFAULT_OBJECTIVE_METRIC,
    DEFAULT_STARCODER_FAMILIES,
    ExperimentFamilyConfig,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CollectPooledDenseStarCoderConfig:
    """Config for collecting pooled dense two-phase and three-phase histories."""

    output_dir: str
    objective_metric: str = DEFAULT_OBJECTIVE_METRIC
    run_families: tuple[ExperimentFamilyConfig, ...] = DEFAULT_STARCODER_FAMILIES
    reuse_three_phase_dir: str | None = None
    reuse_two_phase_dir: str | None = None
    wandb_entity: str = "marin-community"
    wandb_project: str = "marin"
    wandb_timeout_seconds: int = 45
    run_query_retry_attempts: int = 4
    run_query_retry_backoff_seconds: float = 5.0
    history_retry_attempts: int = 4
    history_retry_backoff_seconds: float = 3.0
    max_runs_per_family: int = 0


def _load_existing_family(
    reuse_dir: str | None,
    run_family: str,
) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    if not reuse_dir:
        return None, None
    base_dir = Path(reuse_dir)
    runs_path = base_dir / "runs.parquet"
    history_path = base_dir / "history_dense_wide.parquet"
    if not runs_path.exists() or not history_path.exists():
        raise FileNotFoundError(f"Missing runs/history parquet in {base_dir}")
    runs = pd.read_parquet(runs_path)
    history = pd.read_parquet(history_path)
    if "run_family" in runs.columns:
        runs = runs[runs["run_family"] == run_family].copy()
    if runs.empty:
        raise ValueError(f"No rows for run_family={run_family} in {base_dir}")
    run_ids = set(runs["wandb_run_id"].astype(str).tolist())
    history = history[history["wandb_run_id"].astype(str).isin(run_ids)].copy()
    return runs.reset_index(drop=True), history.reset_index(drop=True)


def collect_pooled_starcoder_dense_dataset(
    config: CollectPooledDenseStarCoderConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Collect pooled dense two-phase and three-phase telemetry from W&B."""
    import wandb

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    api = wandb.Api(timeout=config.wandb_timeout_seconds)
    gcs_prefix = os.environ.get("MARIN_PREFIX", "gs://marin-us-central1")

    runs_frames: list[pd.DataFrame] = []
    history_frames: list[pd.DataFrame] = []
    family_rows: list[dict[str, object]] = []

    for family in config.run_families:
        reuse_dir = None
        if family.run_family == "three_phase_starcoder":
            reuse_dir = config.reuse_three_phase_dir
        elif family.run_family == "two_phase_starcoder":
            reuse_dir = config.reuse_two_phase_dir

        existing_runs, existing_history = _load_existing_family(reuse_dir, family.run_family)
        if existing_runs is not None and existing_history is not None:
            logger.info("Reusing %s dense data from %s", family.run_family, reuse_dir)
            runs_frames.append(existing_runs)
            history_frames.append(existing_history)
            family_rows.append(
                {
                    "run_family": family.run_family,
                    "n_runs": int(existing_runs["wandb_run_id"].nunique()),
                    "source": "reused",
                }
            )
            continue

        wb_runs = _fetch_finished_runs(
            api,
            config.wandb_entity,
            config.wandb_project,
            family,
            config.max_runs_per_family,
            config.run_query_retry_attempts,
            config.run_query_retry_backoff_seconds,
        )
        logger.info("Collecting %d dense runs for %s", len(wb_runs), family.run_family)

        map_from_configs = _load_phase_weights_from_weight_configs(
            source_experiments=family.source_experiments,
            gcs_prefix=gcs_prefix,
        )
        map_from_csv = _load_phase_weights_from_csv(family.csv_fallback_path) if family.csv_fallback_path else {}

        run_rows: list[dict[str, object]] = []
        family_history_frames: list[pd.DataFrame] = []
        for index, wb_run in enumerate(wb_runs, start=1):
            if index == 1 or index % 10 == 0:
                logger.info("Collecting %s dense run %d/%d (%s)", family.run_family, index, len(wb_runs), wb_run.id)
            source_experiment = infer_source_experiment(wb_run.display_name)
            local_run_id = infer_local_run_id(wb_run.display_name)
            phase_weights = _resolve_phase_weights(
                map_from_configs=map_from_configs,
                map_from_csv=map_from_csv,
                source_experiment=source_experiment,
                local_run_id=local_run_id,
                wandb_run_id=wb_run.id,
            )
            history = collect_dense_history_from_run(
                wb_run,
                objective_metric=config.objective_metric,
                retry_attempts=config.history_retry_attempts,
                backoff_seconds=config.history_retry_backoff_seconds,
            )
            family_history_frames.append(history)
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
                config.objective_metric: _extract_numeric_summary(wb_run.summary, config.objective_metric),
                "data_seed": _extract_numeric_config(wb_run.config, "data_seed"),
            }
            run_row.update(phase_weights_to_columns(phase_weights))
            run_rows.append(run_row)

        family_runs = pd.DataFrame(run_rows)
        family_history = pd.concat(family_history_frames, ignore_index=True) if family_history_frames else pd.DataFrame()
        runs_frames.append(family_runs)
        history_frames.append(family_history)
        family_rows.append(
            {
                "run_family": family.run_family,
                "n_runs": int(family_runs["wandb_run_id"].nunique()),
                "source": "wandb",
            }
        )

    runs_df = (
        pd.concat(runs_frames, ignore_index=True).sort_values(["run_family", "wandb_run_id"]).reset_index(drop=True)
    )
    history_df = (
        pd.concat(history_frames, ignore_index=True).sort_values(["wandb_run_id", "step"]).reset_index(drop=True)
        if history_frames
        else pd.DataFrame()
    )
    runs_df.to_parquet(output_dir / "runs.parquet", index=False)
    history_df.to_parquet(output_dir / "history_dense_wide.parquet", index=False)
    summary = {
        "objective_metric": config.objective_metric,
        "families": family_rows,
        "n_runs": int(runs_df["wandb_run_id"].nunique()),
        "history_columns": sorted(history_df.columns.tolist()),
    }
    with (output_dir / "collector_summary.json").open("w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    return runs_df, history_df


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect pooled dense StarCoder telemetry from W&B.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments/domain_phase_mix/offline_rl/artifacts/starcoder_dense_v4_pooled",
    )
    parser.add_argument("--objective-metric", type=str, default=DEFAULT_OBJECTIVE_METRIC)
    parser.add_argument("--reuse-three-phase-dir", type=str, default=None)
    parser.add_argument("--reuse-two-phase-dir", type=str, default=None)
    parser.add_argument("--max-runs-per-family", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = _parse_args()
    collect_pooled_starcoder_dense_dataset(
        CollectPooledDenseStarCoderConfig(
            output_dir=args.output_dir,
            objective_metric=args.objective_metric,
            reuse_three_phase_dir=args.reuse_three_phase_dir,
            reuse_two_phase_dir=args.reuse_two_phase_dir,
            max_runs_per_family=args.max_runs_per_family,
        )
    )


if __name__ == "__main__":
    main()
