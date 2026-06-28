# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Launch the CCGlobalPremium Threshold and RetainedTotal baselines."""

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
from experiments.domain_phase_mix.two_phase_many_ccglobalpremium_baselines import (
    CCGLOBALPREMIUM_RETAINEDTOTAL_RUN_NAME,
    CCGLOBALPREMIUM_SOURCE_EXPERIMENT,
    CCGLOBALPREMIUM_THRESHOLD_RUN_NAME,
    ccglobalpremium_retainedtotal_summary,
    ccglobalpremium_summary_json,
    ccglobalpremium_threshold_summary,
    create_ccglobalpremium_retainedtotal_weight_config,
    create_ccglobalpremium_threshold_weight_config,
)

logger = logging.getLogger(__name__)

DEFAULT_NAME_PREFIX = CCGLOBALPREMIUM_SOURCE_EXPERIMENT
FIT_SUMMARY_JSON = "ccglobalpremium_fit_summary.json"


@dataclass(frozen=True)
class SaveFitSummaryConfig:
    """Config for persisting CCGlobalPremium summaries."""

    output_path: str
    fit_summary_json: str


def save_fit_summary(config: SaveFitSummaryConfig) -> None:
    """Write the CCGlobalPremium summary JSON."""
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
        logger.info("Skipping CCGlobalPremium baseline launch in CI environment")
        return

    experiment = create_two_phase_dolma3_dolmino_top_level_experiment(
        name=args.name_prefix,
        resources=ResourceConfig.with_tpu(args.tpu_type, regions=["us-east5"]),
    )
    fit_summary_step = ExecutorStep(
        name=f"{args.name_prefix}/fit_summary",
        description="Save CCGlobalPremium baseline summary",
        fn=save_fit_summary,
        config=SaveFitSummaryConfig(
            output_path=this_output_path(),
            fit_summary_json=ccglobalpremium_summary_json(),
        ),
    )
    threshold_step = experiment.create_training_step(
        weight_config=create_ccglobalpremium_threshold_weight_config(),
        name_prefix=args.name_prefix,
        run_name=CCGLOBALPREMIUM_THRESHOLD_RUN_NAME,
    )
    retained_step = experiment.create_training_step(
        weight_config=create_ccglobalpremium_retainedtotal_weight_config(),
        name_prefix=args.name_prefix,
        run_name=CCGLOBALPREMIUM_RETAINEDTOTAL_RUN_NAME,
    )

    threshold_summary = ccglobalpremium_threshold_summary()
    retained_summary = ccglobalpremium_retainedtotal_summary()
    logger.info(
        "Launching %s with predicted bpb=%.6f and nearest observed TV=%.6f",
        CCGLOBALPREMIUM_THRESHOLD_RUN_NAME,
        float(threshold_summary["predicted_optimum_value"]),
        float(threshold_summary["nearest_observed_tv_distance"]),
    )
    logger.info(
        "Launching %s with predicted bpb=%.6f and nearest observed TV=%.6f",
        CCGLOBALPREMIUM_RETAINEDTOTAL_RUN_NAME,
        float(retained_summary["predicted_optimum_value"]),
        float(retained_summary["nearest_observed_tv_distance"]),
    )
    run_names = " + ".join((CCGLOBALPREMIUM_THRESHOLD_RUN_NAME, CCGLOBALPREMIUM_RETAINEDTOTAL_RUN_NAME))
    executor_main(
        steps=[fit_summary_step, threshold_step, retained_step],
        description=f"{args.name_prefix}: {run_names}",
    )


if __name__ == "__main__":
    main()
