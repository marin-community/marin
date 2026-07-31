# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Launch the constant-mix realization of the Power-Ridge single-equation optimum."""

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
from experiments.domain_phase_mix.two_phase_many_power_ridge_single import (
    POWER_RIDGE_SINGLE_OBJECTIVE_METRIC,
    POWER_RIDGE_SINGLE_RUN_NAME,
    POWER_RIDGE_SINGLE_SOURCE_EXPERIMENT,
    create_power_ridge_single_weight_config,
    power_ridge_single_summary,
    power_ridge_single_summary_json,
)

logger = logging.getLogger(__name__)

DEFAULT_NAME_PREFIX = POWER_RIDGE_SINGLE_SOURCE_EXPERIMENT
FIT_SUMMARY_JSON = "power_ridge_single_fit_summary.json"


@dataclass(frozen=True)
class SaveFitSummaryConfig:
    """Config for persisting the fitted Power-Ridge summary."""

    output_path: str
    fit_summary_json: str


def save_fit_summary(config: SaveFitSummaryConfig) -> None:
    """Write the Power-Ridge fit summary JSON."""
    fs, _, _ = fsspec.get_fs_token_paths(config.output_path)
    fs.makedirs(config.output_path, exist_ok=True)
    with fsspec.open(os.path.join(config.output_path, FIT_SUMMARY_JSON), "w") as f:
        f.write(config.fit_summary_json)


def _parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name-prefix", default=DEFAULT_NAME_PREFIX)
    parser.add_argument("--tpu-type", default="v5p-8")
    return parser.parse_known_args()


def main() -> None:
    args, remaining = _parse_args()
    sys.argv = [sys.argv[0], *remaining]

    if os.getenv("CI") is not None:
        logger.info("Skipping Power-Ridge baseline launch in CI environment")
        return

    experiment = create_two_phase_dolma3_dolmino_top_level_experiment(
        name=args.name_prefix,
        resources=ResourceConfig.with_tpu(args.tpu_type, regions=["us-east5"]),
    )
    summary = power_ridge_single_summary()
    fit_summary_step = ExecutorStep(
        name=f"{args.name_prefix}/fit_summary",
        description="Save Power-Ridge optimum analysis summary",
        fn=save_fit_summary,
        config=SaveFitSummaryConfig(
            output_path=this_output_path(),
            fit_summary_json=power_ridge_single_summary_json(),
        ),
    )
    training_step = experiment.create_training_step(
        weight_config=create_power_ridge_single_weight_config(),
        name_prefix=args.name_prefix,
        run_name=POWER_RIDGE_SINGLE_RUN_NAME,
    )

    logger.info(
        "Launching %s with predicted %s=%.6f and nearest observed TV=%.6f",
        POWER_RIDGE_SINGLE_RUN_NAME,
        POWER_RIDGE_SINGLE_OBJECTIVE_METRIC,
        float(summary["predicted_optimum_value"]),
        float(summary["nearest_observed_tv_distance"]),
    )
    executor_main(
        steps=[fit_summary_step, training_step],
        description=f"{args.name_prefix}: {POWER_RIDGE_SINGLE_RUN_NAME}",
    )


if __name__ == "__main__":
    main()
