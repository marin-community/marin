# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Recover missing MMLU metrics for the run_00097 seed study from finished checkpoints."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
from dataclasses import dataclass

import fsspec
from fray.cluster import ResourceConfig
from marin.execution.executor import ExecutorStep, InputName, executor_main, output_path_of, this_output_path

from experiments.evals.evals import evaluate_levanter_lm_evaluation_harness
from experiments.evals.task_configs import MMLU_5_SHOT, MMLU_PRO_5_SHOT

logger = logging.getLogger(__name__)

NAME = "pinlin_calvin_xu/data_mixture/ngd3dm2_run00097_missing_mmlu_recovery"
SEED_STUDY_ROOT = "gs://marin-us-east5/checkpoints/pinlin_calvin_xu/data_mixture/ngd3dm2_run00097_seed_study"
RESULTS_JSON = "results.json"
RECOVERED_JSON = "recovered_mmlu_metrics.json"
RECOVERED_CSV = "recovered_mmlu_metrics.csv"

MISSING_RUNS: tuple[tuple[str, str], ...] = (
    ("trainer_seed_10000", "trainer_seed_10000-5bf683"),
    ("trainer_seed_10006", "trainer_seed_10006-c2622a"),
    ("trainer_seed_10008", "trainer_seed_10008-54737f"),
    ("exact_replay_control_a", "exact_replay_control_a-f2b1ac"),
)


@dataclass(frozen=True)
class SaveRecoveredMetricsConfig:
    """Config for flattening recovered eval-harness outputs."""

    output_path: str
    results_by_run: dict[str, InputName]


def _flatten_eval_results(payload: dict) -> dict[str, float]:
    flat: dict[str, float] = {}

    for task_name, task_results in payload.get("results", {}).items():
        for metric_name, metric_value in task_results.items():
            metric_key = metric_name.removesuffix(",none")
            if isinstance(metric_value, int | float):
                flat[f"lm_eval/{task_name}/{metric_key}"] = float(metric_value)

    for metric_name, metric_value in payload.get("averages", {}).items():
        metric_key = metric_name.removesuffix(",none")
        if isinstance(metric_value, int | float):
            flat[f"lm_eval/averages/{metric_key}"] = float(metric_value)

    return flat


def save_recovered_metrics(config: SaveRecoveredMetricsConfig) -> None:
    """Write recovered MMLU metrics as JSON and CSV."""
    rows: list[dict[str, int | float | str]] = []

    for run_name, results_path in config.results_by_run.items():
        with fsspec.open(results_path, "r") as f:
            payload = json.load(f)
        flat = _flatten_eval_results(payload)
        rows.append({"run_name": run_name, **flat})

    fs, _, _ = fsspec.get_fs_token_paths(config.output_path)
    fs.makedirs(config.output_path, exist_ok=True)

    json_path = os.path.join(config.output_path, RECOVERED_JSON)
    with fsspec.open(json_path, "w") as f:
        json.dump(rows, f, indent=2, sort_keys=True)

    csv_path = os.path.join(config.output_path, RECOVERED_CSV)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with fsspec.open(csv_path, "w") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description="Recover missing MMLU metrics for the run_00097 seed study.")
    parser.add_argument("--name-prefix", default=NAME)
    parser.add_argument("--tpu-type", default="v5p-8")
    return parser.parse_known_args()


def main() -> None:
    args, remaining = _parse_args()
    sys.argv = [sys.argv[0], *remaining]

    if os.getenv("CI") is not None:
        logger.info("Skipping missing MMLU recovery launch in CI environment")
        return

    resource_config = ResourceConfig.with_tpu(args.tpu_type, regions=["us-east5"])
    eval_steps: list[ExecutorStep] = []
    results_by_run: dict[str, str] = {}

    for run_name, checkpoint_name in MISSING_RUNS:
        hf_path = f"{SEED_STUDY_ROOT}/{checkpoint_name}/hf/step-4576/"
        eval_step = evaluate_levanter_lm_evaluation_harness(
            model_name=run_name,
            model_path=hf_path,
            evals=[MMLU_5_SHOT, MMLU_PRO_5_SHOT],
            resource_config=resource_config,
            discover_latest_checkpoint=False,
        )
        eval_steps.append(eval_step)
        results_by_run[run_name] = output_path_of(eval_step, RESULTS_JSON)

    harvest_step = ExecutorStep(
        name=f"{args.name_prefix}/collect_recovered_mmlu_metrics",
        description=f"Collect recovered MMLU metrics for {len(MISSING_RUNS)} missing seed-study runs",
        fn=save_recovered_metrics,
        config=SaveRecoveredMetricsConfig(
            output_path=this_output_path(),
            results_by_run=results_by_run,
        ),
    )

    logger.info("Launching MMLU recovery for %d missing run_00097 study runs on %s.", len(MISSING_RUNS), args.tpu_type)
    executor_main(
        steps=[*eval_steps, harvest_step],
        description=f"{args.name_prefix}: recover missing MMLU metrics",
    )


if __name__ == "__main__":
    main()
