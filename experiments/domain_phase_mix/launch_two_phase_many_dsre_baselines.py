# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Launch the frozen DS-RE-derived baselines for the two-phase many-domain sweep."""

from __future__ import annotations

import argparse
import logging
import os
import sys

from fray.cluster import ResourceConfig
from marin.execution.executor import executor_main

from experiments.domain_phase_mix.two_phase_dolma3_dolmino_top_level import (
    create_two_phase_dolma3_dolmino_top_level_experiment,
)
from experiments.domain_phase_mix.two_phase_many_dsre_baselines import (
    DSRE_BASELINES_SOURCE_EXPERIMENT,
    DSRE_ENSEMBLE_PREDICTED_BPB_MEAN,
    DSRE_ENSEMBLE_PREDICTED_BPB_STD,
    DSRE_ENSEMBLE_RUN_NAME,
    DSRE_OBSERVED_CONSENSUS_ACTUAL_BPB,
    DSRE_OBSERVED_CONSENSUS_RUN_NAME,
    DSRE_OBSERVED_CONSENSUS_SOURCE_RUN_NAME,
    create_dsre_ensemble_weight_config,
    create_dsre_observed_consensus_weight_config,
)

logger = logging.getLogger(__name__)

DEFAULT_NAME_PREFIX = DSRE_BASELINES_SOURCE_EXPERIMENT


def _parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description="Launch the two-phase many-domain DS-RE-derived baselines.")
    parser.add_argument("--name-prefix", default=DEFAULT_NAME_PREFIX)
    return parser.parse_known_args()


def main() -> None:
    args, remaining = _parse_args()
    sys.argv = [sys.argv[0], *remaining]

    if os.getenv("CI") is not None:
        logger.info("Skipping DS-RE baseline launch in CI environment")
        return

    experiment = create_two_phase_dolma3_dolmino_top_level_experiment(
        name=args.name_prefix,
        resources=ResourceConfig.with_tpu("v5p-8", regions=["us-east5"]),
    )
    ensemble_step = experiment.create_training_step(
        weight_config=create_dsre_ensemble_weight_config(),
        name_prefix=args.name_prefix,
        run_name=DSRE_ENSEMBLE_RUN_NAME,
    )
    observed_step = experiment.create_training_step(
        weight_config=create_dsre_observed_consensus_weight_config(),
        name_prefix=args.name_prefix,
        run_name=DSRE_OBSERVED_CONSENSUS_RUN_NAME,
    )

    logger.info(
        "Launching %s with predicted %s=%.6f +/- %.6f",
        DSRE_ENSEMBLE_RUN_NAME,
        "lm_eval/mmlu_5shot/bpb",
        DSRE_ENSEMBLE_PREDICTED_BPB_MEAN,
        DSRE_ENSEMBLE_PREDICTED_BPB_STD,
    )
    logger.info(
        "Launching %s based on observed consensus source %s with historical %s=%.6f",
        DSRE_OBSERVED_CONSENSUS_RUN_NAME,
        DSRE_OBSERVED_CONSENSUS_SOURCE_RUN_NAME,
        "lm_eval/mmlu_5shot/bpb",
        DSRE_OBSERVED_CONSENSUS_ACTUAL_BPB,
    )
    executor_main(
        steps=[ensemble_step, observed_step],
        description=f"{args.name_prefix}: {DSRE_ENSEMBLE_RUN_NAME} + {DSRE_OBSERVED_CONSENSUS_RUN_NAME}",
    )


if __name__ == "__main__":
    main()
