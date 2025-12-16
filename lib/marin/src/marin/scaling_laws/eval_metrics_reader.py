# Copyright 2025 The Marin Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Copyright 2025 Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""
Base infrastructure for eval metrics analysis.

This module provides a base config and utilities for analysis jobs that
read eval_metrics.jsonl files from completed training runs. Specific
analyses (like IsoFlop) should subclass EvalMetricsAnalysisConfig.
"""

import logging
import json
import os
from dataclasses import dataclass
from collections.abc import Callable, Sequence

import fsspec
import pandas as pd

try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

logger = logging.getLogger(__name__)

from marin.execution.executor import ExecutorStep, InputName, output_path_of, this_output_path


def extract_run_name_from_path(path: str) -> str:
    """Extract run name (last component) from a checkpoint path.

    E.g., 'gs://bucket/checkpoints/my-run-abc123' -> 'my-run-abc123'
    """
    return os.path.basename(path.rstrip("/"))


def _backfill_metrics_from_wandb(
    checkpoint_path: str,
    metrics_file: str,
    entity_project: str,
) -> bool:
    """
    Backfill eval_metrics.jsonl from WandB for a training run.

    Args:
        checkpoint_path: Path to the checkpoint directory
        metrics_file: Full path to where eval_metrics.jsonl should be written
        entity_project: WandB entity/project (format: 'entity/project')

    Returns:
        True if backfill succeeded, False otherwise
    """
    if not WANDB_AVAILABLE:
        logger.warning(f"wandb not available, cannot backfill metrics for {checkpoint_path}")
        return False

    try:
        run_id = extract_run_name_from_path(checkpoint_path)
        logger.info(f"Attempting to backfill summary metrics for run_id: {run_id}")

        api = wandb.Api()
        run = api.run(f"{entity_project}/{run_id}")

        # Get summary metrics only
        summary = dict(run.summary)

        eval_metrics = {k: v for k, v in summary.items() if k.startswith("eval/")}
        if not eval_metrics:
            logger.warning(f"No eval summary metrics found in WandB for run {run_id}")
            return False
        record = {
            "step": summary.get("_step", summary.get("trainer/global_step", 0)),
            **eval_metrics,
        }

        fs, _, _ = fsspec.get_fs_token_paths(metrics_file)
        fs.makedirs(os.path.dirname(metrics_file), exist_ok=True)

        with fs.open(metrics_file, "w") as f:
            f.write(json.dumps(record) + "\n")

        logger.info(f"Successfully backfilled summary metrics to {metrics_file}")
        return True

    except Exception as e:
        logger.warning(f"Failed to backfill metrics from WandB: {e}")
        return False


@dataclass(frozen=True)
class EvalMetricsAnalysisConfig:
    """Base config for analyses that read eval metrics from training runs.

    Subclass this to create specific analysis types (e.g., IsoFlopAnalysisConfig).
    The training_runs field creates blocking dependencies on the training jobs.
    """

    training_runs: Sequence[str]
    """List of training run output paths to read eval metrics from (blocks until complete)."""

    output_path: str
    """Where to write analysis outputs."""

    metrics_filename: str = "eval_metrics.jsonl"
    """Name of the metrics file within each checkpoint directory."""

    backfill_from_wandb: bool = True
    """If True, backfill eval_metrics.jsonl from WandB for runs that completed before this feature."""

    wandb_entity_project: str = "marin-community/marin"
    """WandB entity/project to query for backfill (format: 'entity/project')."""


def read_metrics_dataframe(config: EvalMetricsAnalysisConfig) -> pd.DataFrame:
    """
    Read eval metrics from training runs into a DataFrame.

    This is the shared utility that all analysis subtypes use to load metrics.
    It handles reading JSONL files and optional WandB backfill.

    Args:
        config: Analysis config with training_runs and backfill settings

    Returns:
        DataFrame with columns: step, run_index, run_path, + all eval/* metrics
    """
    all_records = []

    for i, run_path in enumerate(config.training_runs):
        metrics_file = os.path.join(run_path, config.metrics_filename)

        fs, _, _ = fsspec.get_fs_token_paths(metrics_file)

        if not fs.exists(metrics_file):
            logger.info(f"{metrics_file} does not exist")

            if config.backfill_from_wandb:
                logger.info("Attempting to backfill from WandB...")

                success = _backfill_metrics_from_wandb(
                    checkpoint_path=run_path,
                    metrics_file=metrics_file,
                    entity_project=config.wandb_entity_project,
                )
                if not success:
                    raise RuntimeError(
                        f"Backfill from WandB failed for run {i} (path={run_path}, metrics_file={metrics_file})"
                    )
            else:
                raise RuntimeError(
                    f"Metrics file missing for run {i} (path={run_path}), and backfill_from_wandb is disabled"
                )

        with fs.open(metrics_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                record = json.loads(line)
                record["run_index"] = i
                record["run_path"] = run_path
                all_records.append(record)

    if not all_records:
        logger.warning("No eval metrics found in any training runs")
        return pd.DataFrame()

    df = pd.DataFrame(all_records)
    logger.info(f"Loaded {len(all_records)} evaluation records from {len(config.training_runs)} runs")
    logger.info(f"Available columns: {list(df.columns)}")
    return df


def create_analysis_step(
    name: str,
    training_runs: Sequence[ExecutorStep | InputName],
    analysis_fn: Callable[[EvalMetricsAnalysisConfig], None],
    config_class: type[EvalMetricsAnalysisConfig],
    description: str | None = None,
    **config_kwargs,
) -> ExecutorStep:
    """
    Create an ExecutorStep for an eval metrics analysis.

    This is the factory for creating analysis steps. It:
    - Converts training ExecutorSteps to blocking dependencies
    - Creates the appropriate config subclass
    - Returns an ExecutorStep that runs the analysis

    Args:
        name: Name for this executor step
        training_runs: Training run ExecutorSteps (creates blocking dependencies)
        analysis_fn: The analysis function to run
        config_class: The config class (EvalMetricsAnalysisConfig or subclass)
        description: Optional description
        **config_kwargs: Additional kwargs passed to config_class

    Returns:
        ExecutorStep configured to run the analysis
    """
    run_paths = [output_path_of(run) if isinstance(run, ExecutorStep) else run for run in training_runs]

    config = config_class(
        training_runs=run_paths,
        output_path=this_output_path(),
        **config_kwargs,
    )

    return ExecutorStep(
        name=name,
        fn=analysis_fn,
        config=config,
        description=description or f"Analyze eval metrics from {len(training_runs)} training runs",
    )
