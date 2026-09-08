# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Launch GRP penalty raw-optimum validations."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from dataclasses import dataclass

import fsspec
from fray.cluster import ResourceConfig
from marin.execution.executor import ExecutorMainConfig, ExecutorStep, executor_main, this_output_path

from experiments.domain_phase_mix import (
    two_phase_many_genericfamily_penalty_raw_optima_baselines as raw_optima,
)
from experiments.domain_phase_mix.config import WeightConfig
from experiments.domain_phase_mix.two_phase_dolma3_dolmino_top_level import (
    create_two_phase_dolma3_dolmino_top_level_experiment,
)

logger = logging.getLogger(__name__)

DEFAULT_NAME_PREFIX = raw_optima.GENERICFAMILY_PENALTY_RAW_OPTIMA_SOURCE_EXPERIMENT
DEFAULT_VARIANTS = ",".join(raw_optima.GENERICFAMILY_PENALTY_RAW_OPTIMUM_DEFAULT_VARIANTS)
DEFAULT_TPU_REGION = "us-east5"
DEFAULT_TPU_ZONE = "us-east5-a"
FIT_SUMMARY_JSON = "genericfamily_penalty_raw_optima_fit_summaries.json"
FIT_SUMMARY_CSV = "genericfamily_penalty_raw_optima_fit_summaries.csv"


@dataclass(frozen=True)
class SaveFitSummariesConfig:
    """Config for persisting raw-optimum summaries."""

    output_path: str
    fit_summaries_json: str
    fit_summaries_csv: str


def save_fit_summaries(config: SaveFitSummariesConfig) -> None:
    """Write raw-optimum fit summaries."""
    fs, _, _ = fsspec.get_fs_token_paths(config.output_path)
    fs.makedirs(config.output_path, exist_ok=True)
    with fsspec.open(os.path.join(config.output_path, FIT_SUMMARY_JSON), "w") as handle:
        handle.write(config.fit_summaries_json)
    with fsspec.open(os.path.join(config.output_path, FIT_SUMMARY_CSV), "w") as handle:
        handle.write(config.fit_summaries_csv)


def _parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name-prefix", default=DEFAULT_NAME_PREFIX)
    parser.add_argument("--variants", default=DEFAULT_VARIANTS)
    parser.add_argument("--tpu-type", default="v5p-8")
    parser.add_argument("--tpu-region", default=DEFAULT_TPU_REGION)
    parser.add_argument("--tpu-zone", default=DEFAULT_TPU_ZONE)
    parser.add_argument("--max-concurrent", type=int, default=2)
    return parser.parse_known_args()


def main() -> None:
    args, remaining = _parse_args()
    sys.argv = [sys.argv[0], *remaining]

    if os.getenv("CI") is not None:
        logger.info("Skipping penalty raw-optimum GRP baseline launch in CI environment")
        return

    variants = raw_optima.parse_penalty_raw_optimum_variants(args.variants)
    experiment = create_two_phase_dolma3_dolmino_top_level_experiment(
        name=args.name_prefix,
        resources=ResourceConfig.with_tpu(args.tpu_type, regions=[args.tpu_region], zone=args.tpu_zone),
    )
    summaries = raw_optima.genericfamily_penalty_raw_optimum_summaries(variants)
    fit_summary_step = ExecutorStep(
        name=f"{args.name_prefix}/fit_summaries",
        description="Save penalty raw-optimum GRP fit summaries",
        fn=save_fit_summaries,
        config=SaveFitSummariesConfig(
            output_path=this_output_path(),
            fit_summaries_json=raw_optima.genericfamily_penalty_raw_optimum_summaries_json(variants),
            fit_summaries_csv=raw_optima.genericfamily_penalty_raw_optimum_summaries_frame(variants).to_csv(index=False),
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
        "Launching %d penalty raw-optimum GRP validation runs in %s (%s/%s) with max_concurrent=%d",
        len(train_steps),
        args.tpu_type,
        args.tpu_region,
        args.tpu_zone,
        args.max_concurrent,
    )
    for summary in summaries:
        logger.info(
            "  variant=%s run=%s raw_predicted=%.6f nearest_tv=%.6f",
            summary.variant_name,
            summary.run_name,
            summary.raw_predicted_optimum_value,
            summary.nearest_observed_tv_distance,
        )

    executor_main(
        ExecutorMainConfig(max_concurrent=args.max_concurrent),
        steps=[fit_summary_step, *train_steps],
        description=f"{args.name_prefix}: penalty raw-optimum GRP validations",
    )


if __name__ == "__main__":
    main()
