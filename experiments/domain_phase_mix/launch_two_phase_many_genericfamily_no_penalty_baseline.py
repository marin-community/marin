# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Launch the GRP no-overexposure-penalty predicted baseline."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from dataclasses import dataclass

import fsspec
from fray.cluster import ResourceConfig
from marin.execution.executor import ExecutorStep, executor_main, this_output_path

from experiments.domain_phase_mix.two_phase_dolma3_dolmino_top_level import (
    create_two_phase_dolma3_dolmino_top_level_experiment,
)
from experiments.domain_phase_mix.two_phase_many_genericfamily_no_penalty_baseline import (
    GENERICFAMILY_NO_PENALTY_RUN_NAME,
    GENERICFAMILY_NO_PENALTY_SOURCE_EXPERIMENT,
    create_genericfamily_no_penalty_weight_config,
    genericfamily_no_penalty_summary,
    genericfamily_no_penalty_summary_json,
)

logger = logging.getLogger(__name__)

DEFAULT_NAME_PREFIX = GENERICFAMILY_NO_PENALTY_SOURCE_EXPERIMENT
FIT_SUMMARY_JSON = "genericfamily_no_penalty_fit_summary.json"
DEFAULT_TPU_REGION = "us-east5"
DEFAULT_TPU_ZONE = "us-east5-a"


@dataclass(frozen=True)
class SaveFitSummaryConfig:
    """Config for persisting the no-overexposure-penalty summary."""

    output_path: str
    fit_summary_json: str


def save_fit_summary(config: SaveFitSummaryConfig) -> None:
    """Write the no-overexposure-penalty summary JSON."""
    fs, _, _ = fsspec.get_fs_token_paths(config.output_path)
    fs.makedirs(config.output_path, exist_ok=True)
    with fsspec.open(os.path.join(config.output_path, FIT_SUMMARY_JSON), "w") as f:
        f.write(config.fit_summary_json)


def _parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name-prefix", default=DEFAULT_NAME_PREFIX)
    parser.add_argument("--tpu-type", default="v5p-8")
    parser.add_argument("--tpu-region", default=DEFAULT_TPU_REGION)
    parser.add_argument("--tpu-zone", default=DEFAULT_TPU_ZONE)
    return parser.parse_known_args()


def main() -> None:
    args, remaining = _parse_args()
    sys.argv = [sys.argv[0], *remaining]

    if os.getenv("CI") is not None:
        logger.info("Skipping GRP no-overexposure-penalty baseline launch in CI environment")
        return

    experiment = create_two_phase_dolma3_dolmino_top_level_experiment(
        name=args.name_prefix,
        resources=ResourceConfig.with_tpu(args.tpu_type, regions=[args.tpu_region], zone=args.tpu_zone),
    )
    fit_summary_step = ExecutorStep(
        name=f"{args.name_prefix}/fit_summary",
        description="Save GRP no-overexposure-penalty optimum summary",
        fn=save_fit_summary,
        config=SaveFitSummaryConfig(
            output_path=this_output_path(),
            fit_summary_json=genericfamily_no_penalty_summary_json(),
        ),
    )
    train_step = experiment.create_training_step(
        weight_config=create_genericfamily_no_penalty_weight_config(),
        name_prefix=args.name_prefix,
        run_name=GENERICFAMILY_NO_PENALTY_RUN_NAME,
    )

    summary = genericfamily_no_penalty_summary()
    logger.info(
        "Launching %s with predicted bpb=%.6f, nearest observed TV=%.6f",
        GENERICFAMILY_NO_PENALTY_RUN_NAME,
        float(summary["predicted_optimum_value"]),
        float(summary["nearest_observed_tv_distance"]),
    )
    executor_main(
        steps=[fit_summary_step, train_step],
        description=f"{args.name_prefix}: {GENERICFAMILY_NO_PENALTY_RUN_NAME}",
    )


if __name__ == "__main__":
    main()
