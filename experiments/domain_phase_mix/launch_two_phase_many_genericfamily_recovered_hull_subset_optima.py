# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Launch recovered-hull GRP subset-fit deployments as a fixed follow-up sweep."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from dataclasses import dataclass

import fsspec
from fray.cluster import ResourceConfig
from marin.execution.executor import ExecutorMainConfig, ExecutorStep, executor_main, this_output_path

from experiments.domain_phase_mix.config import WeightConfig
from experiments.domain_phase_mix.two_phase_dolma3_dolmino_top_level import (
    create_two_phase_dolma3_dolmino_top_level_experiment,
)
from experiments.domain_phase_mix.two_phase_many_genericfamily_recovered_hull_subset_optima import (
    GENERICFAMILY_RECOVERED_HULL_SUBSET_OPTIMA_SOURCE_EXPERIMENT,
    genericfamily_recovered_hull_subset_optima_summaries,
    genericfamily_recovered_hull_subset_optima_summaries_frame,
    genericfamily_recovered_hull_subset_optima_summaries_json,
)
from experiments.domain_phase_mix.two_phase_many_genericfamily_retuned_subset_optima import (
    parse_subset_sizes,
)

logger = logging.getLogger(__name__)

DEFAULT_NAME_PREFIX = GENERICFAMILY_RECOVERED_HULL_SUBSET_OPTIMA_SOURCE_EXPERIMENT
DEFAULT_SUBSET_SIZES = "all"
FIT_SUMMARY_JSON = "genericfamily_recovered_hull_subset_optima_fit_summaries.json"
FIT_SUMMARY_CSV = "genericfamily_recovered_hull_subset_optima_fit_summaries.csv"
DEFAULT_TPU_REGION = "us-east5"
DEFAULT_TPU_ZONE = "us-east5-a"
DEFAULT_MAX_CONCURRENT = 4


@dataclass(frozen=True)
class SaveFitSummariesConfig:
    """Config for persisting recovered-hull subset-fit summaries."""

    output_path: str
    fit_summaries_json: str
    fit_summaries_csv: str


def save_fit_summaries(config: SaveFitSummariesConfig) -> None:
    """Write the recovered-hull subset-fit summaries."""
    fs, _, _ = fsspec.get_fs_token_paths(config.output_path)
    fs.makedirs(config.output_path, exist_ok=True)
    with fsspec.open(os.path.join(config.output_path, FIT_SUMMARY_JSON), "w") as f:
        f.write(config.fit_summaries_json)
    with fsspec.open(os.path.join(config.output_path, FIT_SUMMARY_CSV), "w") as f:
        f.write(config.fit_summaries_csv)


def _parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name-prefix", default=DEFAULT_NAME_PREFIX)
    parser.add_argument("--subset-sizes", default=DEFAULT_SUBSET_SIZES)
    parser.add_argument("--tpu-type", default="v5p-8")
    parser.add_argument("--tpu-region", default=DEFAULT_TPU_REGION)
    parser.add_argument("--tpu-zone", default=DEFAULT_TPU_ZONE)
    parser.add_argument("--max-concurrent", type=int, default=DEFAULT_MAX_CONCURRENT)
    return parser.parse_known_args()


def main() -> None:
    args, remaining = _parse_args()
    sys.argv = [sys.argv[0], *remaining]

    if os.getenv("CI") is not None:
        logger.info("Skipping recovered-hull subset-fit sweep launch in CI environment")
        return

    subset_sizes = parse_subset_sizes(args.subset_sizes)
    experiment = create_two_phase_dolma3_dolmino_top_level_experiment(
        name=args.name_prefix,
        resources=ResourceConfig.with_tpu(args.tpu_type, regions=[args.tpu_region], zone=args.tpu_zone),
    )
    summaries = genericfamily_recovered_hull_subset_optima_summaries(subset_sizes)
    fit_summary_step = ExecutorStep(
        name=f"{args.name_prefix}/fit_summaries",
        description="Save recovered-hull GRP subset-optimum summaries",
        fn=save_fit_summaries,
        config=SaveFitSummariesConfig(
            output_path=this_output_path(),
            fit_summaries_json=genericfamily_recovered_hull_subset_optima_summaries_json(subset_sizes),
            fit_summaries_csv=genericfamily_recovered_hull_subset_optima_summaries_frame(subset_sizes).to_csv(
                index=False
            ),
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
        "Launching %d recovered-hull subset-fit runs in %s with max_concurrent=%d",
        len(train_steps),
        args.tpu_type,
        args.max_concurrent,
    )
    for summary in summaries:
        logger.info(
            "  k=%d run=%s predicted=%.6f tune_obj=%.6f nearest_tv=%.6f",
            summary.subset_size,
            summary.run_name,
            summary.predicted_optimum_value,
            summary.tuning_objective,
            summary.nearest_observed_tv_distance,
        )

    executor_main(
        ExecutorMainConfig(max_concurrent=args.max_concurrent),
        steps=[fit_summary_step, *train_steps],
        description=f"{args.name_prefix}: recovered-hull GRP subset-fit predicted optima",
    )


if __name__ == "__main__":
    main()
