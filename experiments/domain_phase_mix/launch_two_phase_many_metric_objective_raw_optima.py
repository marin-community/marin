# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Launch metric-specific no-L2 GRP raw-optimum validations."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import logging
import os
import sys

import fsspec
from fray.cluster import ResourceConfig
from marin.evaluation.evaluation_config import EvalTaskConfig
from marin.execution.executor import ExecutorMainConfig, ExecutorStep, executor_main, this_output_path

from experiments.domain_phase_mix.config import WeightConfig
from experiments.domain_phase_mix.two_phase_dolma3_dolmino_top_level import (
    create_two_phase_dolma3_dolmino_top_level_experiment,
)
from experiments.domain_phase_mix.two_phase_many_metric_objective_raw_optima import (
    METRIC_OBJECTIVE_RAW_OPTIMA_DEFAULT_SLUGS,
    METRIC_OBJECTIVE_RAW_OPTIMA_SOURCE_EXPERIMENT,
    metric_objective_raw_optimum_summaries,
    metric_objective_raw_optimum_summaries_csv,
    metric_objective_raw_optimum_summaries_json,
    parse_metric_objective_raw_optimum_slugs,
)
from experiments.evals.olmo_base_easy_overlap import OLMO_BASE_EASY_OVERLAP_CACHE_PATH

logger = logging.getLogger(__name__)

DEFAULT_NAME_PREFIX = METRIC_OBJECTIVE_RAW_OPTIMA_SOURCE_EXPERIMENT
DEFAULT_SLUGS = ",".join(METRIC_OBJECTIVE_RAW_OPTIMA_DEFAULT_SLUGS)
DEFAULT_TPU_REGION = "us-east5"
DEFAULT_TPU_ZONE = "us-east5-a"
DEFAULT_MAX_CONCURRENT = 2
FIT_SUMMARY_JSON = "metric_objective_grp_no_l2_raw_optima_fit_summaries.json"
FIT_SUMMARY_CSV = "metric_objective_grp_no_l2_raw_optima_fit_summaries.csv"
PIQA_5SHOT_TASK = EvalTaskConfig("piqa", 5, task_alias="piqa_5shot")


@dataclass(frozen=True)
class SaveFitSummariesConfig:
    """Config for persisting metric-specific no-L2 GRP raw-optimum summaries."""

    output_path: str
    fit_summaries_json: str
    fit_summaries_csv: str


def save_fit_summaries(config: SaveFitSummariesConfig) -> None:
    """Write metric-specific no-L2 GRP raw-optimum summaries."""
    fs, _, _ = fsspec.get_fs_token_paths(config.output_path)
    fs.makedirs(config.output_path, exist_ok=True)
    with fsspec.open(os.path.join(config.output_path, FIT_SUMMARY_JSON), "w") as handle:
        handle.write(config.fit_summaries_json)
    with fsspec.open(os.path.join(config.output_path, FIT_SUMMARY_CSV), "w") as handle:
        handle.write(config.fit_summaries_csv)


def _parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name-prefix", default=DEFAULT_NAME_PREFIX)
    parser.add_argument("--slugs", default=DEFAULT_SLUGS)
    parser.add_argument("--tpu-type", default="v5p-8")
    parser.add_argument("--tpu-region", default=DEFAULT_TPU_REGION)
    parser.add_argument("--tpu-zone", default=DEFAULT_TPU_ZONE)
    parser.add_argument("--max-concurrent", type=int, default=DEFAULT_MAX_CONCURRENT)
    parser.add_argument("--eval-datasets-cache-path", default=OLMO_BASE_EASY_OVERLAP_CACHE_PATH)
    return parser.parse_known_args()


def main() -> None:
    args, remaining = _parse_args()
    sys.argv = [sys.argv[0], *remaining]

    if os.getenv("CI") is not None:
        logger.info("Skipping metric-specific no-L2 GRP raw-optimum launch in CI environment")
        return

    slugs = parse_metric_objective_raw_optimum_slugs(args.slugs)
    summaries = metric_objective_raw_optimum_summaries(slugs)
    experiment = create_two_phase_dolma3_dolmino_top_level_experiment(
        name=args.name_prefix,
        resources=ResourceConfig.with_tpu(args.tpu_type, regions=[args.tpu_region], zone=args.tpu_zone),
        eval_harness_tasks=(PIQA_5SHOT_TASK,),
        eval_datasets_cache_path=args.eval_datasets_cache_path,
    )
    fit_summary_step = ExecutorStep(
        name=f"{args.name_prefix}/fit_summaries",
        description="Save metric-specific no-L2 GRP raw-optimum summaries",
        fn=save_fit_summaries,
        config=SaveFitSummariesConfig(
            output_path=this_output_path(),
            fit_summaries_json=metric_objective_raw_optimum_summaries_json(slugs),
            fit_summaries_csv=metric_objective_raw_optimum_summaries_csv(slugs),
        ),
    )
    train_steps: list[ExecutorStep] = []
    for summary in summaries:
        train_steps.append(
            experiment.create_training_step(
                weight_config=WeightConfig(run_id=summary.run_id, phase_weights=summary.phase_weights),
                name_prefix=args.name_prefix,
                run_name=summary.run_name,
            )
        )

    logger.info(
        "Launching %d metric-specific no-L2 GRP raw-optimum validations in %s (%s/%s) with max_concurrent=%d",
        len(train_steps),
        args.tpu_type,
        args.tpu_region,
        args.tpu_zone,
        args.max_concurrent,
    )
    for summary in summaries:
        logger.info(
            "  slug=%s run=%s metric=%s raw_predicted=%.6f best_observed=%s %.6f "
            "predicted_observed=%s %.6f regret=%.6f nearest_tv=%.6f",
            summary.slug,
            summary.run_name,
            summary.metric_key,
            summary.predicted_optimum_metric,
            summary.best_observed_run_name,
            summary.best_observed_metric,
            summary.predicted_observed_run_name,
            summary.predicted_observed_metric,
            summary.predicted_observed_regret,
            summary.raw_nearest_observed_tv,
        )

    executor_main(
        ExecutorMainConfig(max_concurrent=args.max_concurrent),
        steps=[fit_summary_step, *train_steps],
        description=f"{args.name_prefix}: metric-specific no-L2 GRP raw-optimum validations",
    )


if __name__ == "__main__":
    main()
