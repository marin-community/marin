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

"""Analysis utilities for data mixture swarm experiments.

This module provides an ExecutorStep for collecting results from W&B and
joining them with weight configurations to produce a consolidated CSV file
for exploratory analysis.

The analysis step:
1. Reads weight_configs.json from GCS (saved by the experiment)
2. Queries W&B for runs matching the experiment tags
3. Matches runs to configurations by run_id
4. Outputs results.csv with all weights and metrics
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

import fsspec
import numpy as np

from marin.execution.executor import ExecutorStep, output_path_of, this_output_path

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)


# Default metrics to collect from W&B
DEFAULT_METRICS = [
    "eval/loss",
    "eval/paloma/c4_en/bpb",
    "eval/paloma/m2d2_wikipedia_unsplit/bpb",
    "lm_eval/arc_challenge/acc",
    "lm_eval/arc_challenge/acc_norm",
    "lm_eval/hellaswag_0shot/acc",
    "lm_eval/hellaswag_0shot/acc_norm",
    "lm_eval/piqa/acc",
    "lm_eval/boolq/acc",
    "lm_eval/averages/macro_avg_acc",
]


# ============================================================================
# ANALYSIS EXECUTOR STEP
# ============================================================================


@dataclass(frozen=True)
class CollectResultsConfig:
    """Configuration for the analysis executor step."""

    weight_configs_path: str  # Path to weight_configs.json (GCS or local)
    output_path: str  # Where to write results CSV (GCS or local)
    wandb_entity: str = "marin"
    wandb_project: str = "marin"
    wandb_tags: tuple[str, ...] = ()  # Tags to filter runs
    metrics: tuple[str, ...] = tuple(DEFAULT_METRICS)  # Metrics to collect


def collect_results(config: CollectResultsConfig):
    """Collect results from W&B and join with weight configs.

    This is an ExecutorStep function that:
    1. Loads weight configurations from GCS
    2. Queries W&B for matching runs
    3. Matches runs to configs by run_id pattern
    4. Writes consolidated CSV to GCS
    """
    import importlib.util

    if importlib.util.find_spec("pandas") is None or importlib.util.find_spec("wandb") is None:
        raise ImportError("pandas and wandb are required for analysis")

    # 1. Load weight configurations from GCS
    logger.info(f"Loading weight configs from {config.weight_configs_path}")
    weight_configs = load_weight_configs(config.weight_configs_path)

    experiment_name = weight_configs["experiment_name"]
    domains = weight_configs["domains"]
    phases = weight_configs["phases"]
    configs = weight_configs["configs"]

    logger.info(f"Loaded {len(configs)} weight configurations")
    logger.info(f"Domains: {domains}")
    logger.info(f"Phases: {phases}")

    # 2. Query W&B for runs matching tags
    tags = list(config.wandb_tags) if config.wandb_tags else [experiment_name]
    logger.info(f"Querying W&B for runs with tags: {tags}")

    runs = query_wandb_runs(
        entity=config.wandb_entity,
        project=config.wandb_project,
        tags=tags,
        metrics=list(config.metrics),
    )
    logger.info(f"Found {len(runs)} W&B runs")

    # 3. Match runs to configs by run_id
    matched = match_runs_to_configs(runs, configs, experiment_name=experiment_name)
    logger.info(f"Matched {sum(1 for m in matched if m.get('wandb_run_id'))} runs to configs")

    # 4. Build DataFrame with all weights and metrics
    df = build_results_dataframe(matched, domains, phases, list(config.metrics))

    # 5. Write CSV to GCS
    csv_path = os.path.join(config.output_path, "results.csv")
    logger.info(f"Writing results to {csv_path}")
    with fsspec.open(csv_path, "w") as f:
        df.to_csv(f, index=False)

    # 6. Write summary JSON
    summary = {
        "experiment_name": experiment_name,
        "n_configs": len(configs),
        "n_matched": sum(1 for m in matched if m.get("wandb_run_id")),
        "n_completed": sum(1 for m in matched if m.get("status") == "completed"),
        "domains": domains,
        "phases": phases,
        "metrics": list(config.metrics),
    }
    summary_path = os.path.join(config.output_path, "summary.json")
    with fsspec.open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Analysis complete: {summary['n_completed']}/{summary['n_configs']} runs completed")


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def load_weight_configs(path: str) -> dict:
    """Load weight configurations from a JSON file (GCS or local).

    Args:
        path: Path to the weight_configs.json file.

    Returns:
        Dictionary containing experiment metadata and configs.
    """
    with fsspec.open(path) as f:
        return json.load(f)


def query_wandb_runs(
    entity: str,
    project: str,
    tags: list[str],
    metrics: list[str],
) -> list[dict]:
    """Query W&B API for runs with specific tags.

    Args:
        entity: W&B entity name.
        project: W&B project name.
        tags: Tags to filter runs (runs must have at least one of these tags).
        metrics: List of metric keys to collect.

    Returns:
        List of dictionaries with run info and metrics.
    """
    import wandb

    api = wandb.Api()

    # Query runs with any of the specified tags
    filters = {"tags": {"$in": tags}}
    runs = api.runs(f"{entity}/{project}", filters=filters)

    results = []
    for run in runs:
        row = {
            "wandb_run_id": run.id,
            "wandb_run_name": run.name,
            "status": run.state,
        }

        # Extract metrics from run summary
        for metric in metrics:
            row[metric] = run.summary.get(metric)

        results.append(row)

    return results


def match_runs_to_configs(
    runs: list[dict], configs: list[dict], experiment_name: str
) -> list[dict]:
    """Match W&B runs to weight configurations by run_id pattern.

    Extracts run_id from W&B run names and matches to the corresponding config.
    Tries multiple patterns to handle different W&B naming conventions:
    1. Full path: "pinlin_calvin_xu/data_mixture/3_partitions_3_phases/run_00042"
    2. Short name: "run_00042-abc123" (W&B may truncate long names)

    Args:
        runs: List of W&B run dictionaries.
        configs: List of weight configuration dictionaries.
        experiment_name: Experiment name prefix to filter runs (required to avoid false positives).

    Returns:
        List of matched dictionaries with config + run info.
    """
    # Build lookup from run_id to W&B run
    run_by_id: dict[int, dict] = {}

    escaped_name = re.escape(experiment_name)
    run_id_pattern = re.compile(rf"{escaped_name}/run_(\d+)")

    for run in runs:
        name = run.get("wandb_run_name", "")
        match = run_id_pattern.search(name)
        if match:
            run_id = int(match.group(1))
            # Keep the most recent run if there are duplicates
            if run_id not in run_by_id or run["status"] == "finished":
                run_by_id[run_id] = run

    # Match configs to runs
    matched = []
    for config in configs:
        run_id = config["run_id"]
        row = {"run_id": run_id, **config.get("phase_weights", {})}

        if run_id in run_by_id:
            run = run_by_id[run_id]
            row["wandb_run_id"] = run["wandb_run_id"]
            row["wandb_run_name"] = run["wandb_run_name"]
            row["status"] = "completed" if run["status"] == "finished" else run["status"]

            # Copy metrics
            for key, value in run.items():
                if key not in ["wandb_run_id", "wandb_run_name", "status"]:
                    row[key] = value
        else:
            row["wandb_run_id"] = None
            row["wandb_run_name"] = None
            row["status"] = "not_found"

        matched.append(row)

    return matched


def build_results_dataframe(
    matched: list[dict],
    domains: list[str],
    phases: list[str],
    metrics: list[str],
) -> pd.DataFrame:
    """Build a pandas DataFrame from matched results.

    Args:
        matched: List of matched config + run dictionaries.
        domains: List of domain names.
        phases: List of phase names.
        metrics: List of metric names.

    Returns:
        DataFrame with one row per run.
    """
    import pandas as pd

    rows = []
    for m in matched:
        row = {
            "run_id": m["run_id"],
            "wandb_run_id": m.get("wandb_run_id"),
            "status": m.get("status", "not_found"),
        }

        # Flatten phase weights into columns
        for phase in phases:
            phase_weights = m.get(phase, {})
            for domain in domains:
                col_name = f"{phase}_{domain}"
                row[col_name] = phase_weights.get(domain, np.nan)

        # Add metrics
        for metric in metrics:
            row[metric] = m.get(metric)

        rows.append(row)

    return pd.DataFrame(rows)


# ============================================================================
# STEP CREATION HELPER
# ============================================================================


def create_analysis_step(
    weight_configs_step: ExecutorStep,
    name_prefix: str,
    wandb_entity: str = "marin",
    wandb_project: str = "marin",
    metrics: list[str] | None = None,
) -> ExecutorStep:
    """Create an analysis ExecutorStep.

    Args:
        weight_configs_step: The ExecutorStep that saves weight configurations.
        name_prefix: Experiment name prefix (used as W&B tag).
        wandb_entity: W&B entity name.
        wandb_project: W&B project name.
        metrics: Metrics to collect (defaults to DEFAULT_METRICS).

    Returns:
        ExecutorStep that collects results and writes CSV.
    """
    return ExecutorStep(
        name=f"{name_prefix}/analysis",
        description="Collect W&B results and generate analysis CSV",
        fn=collect_results,
        config=CollectResultsConfig(
            weight_configs_path=os.path.join(output_path_of(weight_configs_step), "weight_configs.json"),
            output_path=this_output_path(),
            wandb_entity=wandb_entity,
            wandb_project=wandb_project,
            wandb_tags=(name_prefix,),
            metrics=tuple(metrics or DEFAULT_METRICS),
        ),
    )
