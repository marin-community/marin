# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Launch GRP subset-fit predicted optima as a fixed follow-up sweep."""

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
from experiments.domain_phase_mix.two_phase_many_genericfamily_subset_optima import (
    GENERICFAMILY_SUBSET_OPTIMA_SOURCE_EXPERIMENT,
    genericfamily_subset_optima_summaries,
    genericfamily_subset_optima_summaries_frame,
    genericfamily_subset_optima_summaries_json,
)

logger = logging.getLogger(__name__)

DEFAULT_NAME_PREFIX = GENERICFAMILY_SUBSET_OPTIMA_SOURCE_EXPERIMENT
FIT_SUMMARY_JSON = "genericfamily_subset_optima_fit_summaries.json"
FIT_SUMMARY_CSV = "genericfamily_subset_optima_fit_summaries.csv"
DEFAULT_TPU_REGION = "us-east5"
DEFAULT_TPU_ZONE = "us-east5-a"
DEFAULT_MAX_CONCURRENT = 4


@dataclass(frozen=True)
class SaveFitSummariesConfig:
    """Config for persisting subset-optimum summaries."""

    output_path: str
    fit_summaries_json: str
    fit_summaries_csv: str


def save_fit_summaries(config: SaveFitSummariesConfig) -> None:
    """Write the subset-optimum summaries."""
    fs, _, _ = fsspec.get_fs_token_paths(config.output_path)
    fs.makedirs(config.output_path, exist_ok=True)
    with fsspec.open(os.path.join(config.output_path, FIT_SUMMARY_JSON), "w") as f:
        f.write(config.fit_summaries_json)
    with fsspec.open(os.path.join(config.output_path, FIT_SUMMARY_CSV), "w") as f:
        f.write(config.fit_summaries_csv)


def _parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name-prefix", default=DEFAULT_NAME_PREFIX)
    parser.add_argument("--tpu-type", default="v5p-8")
    parser.add_argument("--tpu-region", default=DEFAULT_TPU_REGION)
    parser.add_argument("--tpu-zone", default=DEFAULT_TPU_ZONE)
    parser.add_argument("--max-concurrent", type=int, default=DEFAULT_MAX_CONCURRENT)
    return parser.parse_known_args()


def main() -> None:
    args, remaining = _parse_args()
    sys.argv = [sys.argv[0], *remaining]

    if os.getenv("CI") is not None:
        logger.info("Skipping GRP subset-optimum sweep launch in CI environment")
        return

    experiment = create_two_phase_dolma3_dolmino_top_level_experiment(
        name=args.name_prefix,
        resources=ResourceConfig.with_tpu(args.tpu_type, regions=[args.tpu_region], zone=args.tpu_zone),
    )
    summaries = genericfamily_subset_optima_summaries()
    fit_summary_step = ExecutorStep(
        name=f"{args.name_prefix}/fit_summaries",
        description="Save GRP subset-optimum summaries",
        fn=save_fit_summaries,
        config=SaveFitSummariesConfig(
            output_path=this_output_path(),
            fit_summaries_json=genericfamily_subset_optima_summaries_json(),
            fit_summaries_csv=genericfamily_subset_optima_summaries_frame().to_csv(index=False),
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
        "Launching %d GRP subset-optimum runs in %s with max_concurrent=%d",
        len(train_steps),
        args.tpu_type,
        args.max_concurrent,
    )
    for summary in summaries:
        logger.info(
            "  k=%d run=%s predicted=%.6f nearest_tv=%.6f move=%.6f",
            summary.subset_size,
            summary.run_name,
            summary.predicted_optimum_value,
            summary.nearest_observed_tv_distance,
            0.0 if summary.optimum_move_mean_phase_tv_vs_prev is None else summary.optimum_move_mean_phase_tv_vs_prev,
        )

    executor_main(
        ExecutorMainConfig(max_concurrent=args.max_concurrent),
        steps=[fit_summary_step, *train_steps],
        description=f"{args.name_prefix}: GRP subset-fit predicted optima",
    )


if __name__ == "__main__":
    main()
