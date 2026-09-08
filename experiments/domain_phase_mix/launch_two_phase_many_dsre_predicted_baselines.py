# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Launch the saved DS-RE-CEQ predicted optimum and a quality-collapsed variant."""

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
from experiments.domain_phase_mix.two_phase_many_dsre_predicted_baselines import (
    DSRE_CEQ_PREDICTED_QUALITY_COLLAPSED_RUN_NAME,
    DSRE_CEQ_PREDICTED_RUN_NAME,
    DSRE_PREDICTED_BASELINES_SOURCE_EXPERIMENT,
    create_dsre_ceq_predicted_quality_collapsed_weight_config,
    create_dsre_ceq_predicted_weight_config,
    dsre_ceq_predicted_quality_collapsed_summary,
    dsre_ceq_predicted_summary,
    dsre_ceq_predicted_summary_json,
)

logger = logging.getLogger(__name__)

DEFAULT_NAME_PREFIX = DSRE_PREDICTED_BASELINES_SOURCE_EXPERIMENT
FIT_SUMMARY_JSON = "dsre_predicted_fit_summary.json"


@dataclass(frozen=True)
class SaveFitSummaryConfig:
    """Config for persisting the DS-RE predicted-baseline summary."""

    output_path: str
    fit_summary_json: str


def save_fit_summary(config: SaveFitSummaryConfig) -> None:
    """Write the DS-RE predicted-baseline summary JSON."""
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
        logger.info("Skipping DS-RE predicted baseline launch in CI environment")
        return

    experiment = create_two_phase_dolma3_dolmino_top_level_experiment(
        name=args.name_prefix,
        resources=ResourceConfig.with_tpu(args.tpu_type, regions=["us-east5"]),
    )
    fit_summary_step = ExecutorStep(
        name=f"{args.name_prefix}/fit_summary",
        description="Save DS-RE predicted baseline summary",
        fn=save_fit_summary,
        config=SaveFitSummaryConfig(
            output_path=this_output_path(),
            fit_summary_json=dsre_ceq_predicted_summary_json(),
        ),
    )
    full_step = experiment.create_training_step(
        weight_config=create_dsre_ceq_predicted_weight_config(),
        name_prefix=args.name_prefix,
        run_name=DSRE_CEQ_PREDICTED_RUN_NAME,
    )
    collapsed_step = experiment.create_training_step(
        weight_config=create_dsre_ceq_predicted_quality_collapsed_weight_config(),
        name_prefix=args.name_prefix,
        run_name=DSRE_CEQ_PREDICTED_QUALITY_COLLAPSED_RUN_NAME,
    )

    full_summary = dsre_ceq_predicted_summary()
    collapsed_summary = dsre_ceq_predicted_quality_collapsed_summary()
    logger.info(
        "Launching %s with predicted bpb=%.6f and nearest observed TV=%.6f",
        DSRE_CEQ_PREDICTED_RUN_NAME,
        float(full_summary["predicted_bpb"]),
        float(full_summary["nearest_observed_tv_distance"]),
    )
    logger.info(
        "Launching %s with nearest observed TV=%.6f",
        DSRE_CEQ_PREDICTED_QUALITY_COLLAPSED_RUN_NAME,
        float(collapsed_summary["nearest_observed_tv_distance"]),
    )
    run_names = " + ".join((DSRE_CEQ_PREDICTED_RUN_NAME, DSRE_CEQ_PREDICTED_QUALITY_COLLAPSED_RUN_NAME))
    description = f"{args.name_prefix}: {run_names}"
    executor_main(
        steps=[fit_summary_step, full_step, collapsed_step],
        description=description,
    )


if __name__ == "__main__":
    main()
