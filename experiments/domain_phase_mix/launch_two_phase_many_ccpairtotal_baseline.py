# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Launch the CCPairTotal-RetainedTotal baseline."""

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
from experiments.domain_phase_mix.two_phase_many_ccpairtotal_baseline import (
    CCPAIRTOTAL_RETAINEDTOTAL_RUN_NAME,
    CCPAIRTOTAL_RETAINEDTOTAL_SOURCE_EXPERIMENT,
    ccpairtotal_retainedtotal_summary,
    ccpairtotal_retainedtotal_summary_json,
    create_ccpairtotal_retainedtotal_weight_config,
)

logger = logging.getLogger(__name__)

DEFAULT_NAME_PREFIX = CCPAIRTOTAL_RETAINEDTOTAL_SOURCE_EXPERIMENT
FIT_SUMMARY_JSON = "ccpairtotal_retainedtotal_fit_summary.json"


@dataclass(frozen=True)
class SaveFitSummaryConfig:
    """Config for persisting the CCPairTotal summary."""

    output_path: str
    fit_summary_json: str


def save_fit_summary(config: SaveFitSummaryConfig) -> None:
    """Write the CCPairTotal fit summary JSON."""
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
        logger.info("Skipping CCPairTotal baseline launch in CI environment")
        return

    experiment = create_two_phase_dolma3_dolmino_top_level_experiment(
        name=args.name_prefix,
        resources=ResourceConfig.with_tpu(args.tpu_type, regions=["us-east5"]),
    )
    fit_summary_step = ExecutorStep(
        name=f"{args.name_prefix}/fit_summary",
        description="Save CCPairTotal baseline summary",
        fn=save_fit_summary,
        config=SaveFitSummaryConfig(
            output_path=this_output_path(),
            fit_summary_json=ccpairtotal_retainedtotal_summary_json(),
        ),
    )
    train_step = experiment.create_training_step(
        weight_config=create_ccpairtotal_retainedtotal_weight_config(),
        name_prefix=args.name_prefix,
        run_name=CCPAIRTOTAL_RETAINEDTOTAL_RUN_NAME,
    )

    summary = ccpairtotal_retainedtotal_summary()
    logger.info(
        "Launching %s with predicted bpb=%.6f, nearest observed TV=%.6f, support<1e-4=(%d,%d)",
        CCPAIRTOTAL_RETAINEDTOTAL_RUN_NAME,
        float(summary["predicted_optimum_value"]),
        float(summary["nearest_observed_tv_distance"]),
        int(summary["phase0_support_below_1e4"]),
        int(summary["phase1_support_below_1e4"]),
    )
    executor_main(
        steps=[fit_summary_step, train_step],
        description=f"{args.name_prefix}: {CCPAIRTOTAL_RETAINEDTOTAL_RUN_NAME}",
    )


if __name__ == "__main__":
    main()
