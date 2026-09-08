# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Launch the DS-RE predicted baseline on the CC-topic-collapsed topology."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from dataclasses import dataclass

import fsspec
from fray.cluster import ResourceConfig
from marin.execution.executor import ExecutorStep, executor_main, this_output_path

from experiments.domain_phase_mix.two_phase_dolma3_dolmino_top_level_topic_collapsed import (
    create_two_phase_dolma3_dolmino_top_level_topic_collapsed_experiment,
)
from experiments.domain_phase_mix.two_phase_many_dsre_predicted_topic_collapsed import (
    DSRE_CEQ_PREDICTED_TOPIC_COLLAPSED_RUN_NAME,
    DSRE_CEQ_PREDICTED_TOPIC_COLLAPSED_SOURCE_EXPERIMENT,
    create_dsre_ceq_predicted_topic_collapsed_weight_config,
    dsre_ceq_predicted_topic_collapsed_summary,
    dsre_ceq_predicted_topic_collapsed_summary_json,
)

logger = logging.getLogger(__name__)

DEFAULT_NAME_PREFIX = DSRE_CEQ_PREDICTED_TOPIC_COLLAPSED_SOURCE_EXPERIMENT
FIT_SUMMARY_JSON = "dsre_predicted_topic_collapsed_fit_summary.json"


@dataclass(frozen=True)
class SaveFitSummaryConfig:
    """Config for persisting the topic-collapsed DS-RE summary."""

    output_path: str
    fit_summary_json: str


def save_fit_summary(config: SaveFitSummaryConfig) -> None:
    """Write the topic-collapsed DS-RE summary JSON."""
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
        logger.info("Skipping topic-collapsed DS-RE baseline launch in CI environment")
        return

    experiment = create_two_phase_dolma3_dolmino_top_level_topic_collapsed_experiment(
        name=args.name_prefix,
        resources=ResourceConfig.with_tpu(args.tpu_type, regions=["us-east5"]),
    )
    summary = dsre_ceq_predicted_topic_collapsed_summary()
    fit_summary_step = ExecutorStep(
        name=f"{args.name_prefix}/fit_summary",
        description="Save topic-collapsed DS-RE predicted baseline summary",
        fn=save_fit_summary,
        config=SaveFitSummaryConfig(
            output_path=this_output_path(),
            fit_summary_json=dsre_ceq_predicted_topic_collapsed_summary_json(),
        ),
    )
    training_step = experiment.create_training_step(
        weight_config=create_dsre_ceq_predicted_topic_collapsed_weight_config(),
        name_prefix=args.name_prefix,
        run_name=DSRE_CEQ_PREDICTED_TOPIC_COLLAPSED_RUN_NAME,
    )

    logger.info(
        "Launching %s with realized-flat nearest observed TV=%.6f and flat equivalence TV=%.6f",
        DSRE_CEQ_PREDICTED_TOPIC_COLLAPSED_RUN_NAME,
        float(summary["nearest_observed_tv_distance"]),
        float(summary["equivalent_flat_tv_distance"]),
    )
    executor_main(
        steps=[fit_summary_step, training_step],
        description=f"{args.name_prefix}: {DSRE_CEQ_PREDICTED_TOPIC_COLLAPSED_RUN_NAME}",
    )


if __name__ == "__main__":
    main()
