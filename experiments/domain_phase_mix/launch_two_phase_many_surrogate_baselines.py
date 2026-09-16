# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Launch the frozen CLR-Ridge and DS-RE-CEQ-ST(lite) baselines."""

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
from experiments.domain_phase_mix.two_phase_many_surrogate_baselines import (
    CLR_RIDGE_ALPHA,
    CLR_RIDGE_PREDICTED_BPB,
    CLR_RIDGE_RUN_NAME,
    DSRE_CEQ_ST_LITE_PREDICTED_BPB,
    DSRE_CEQ_ST_LITE_RUN_NAME,
    SURROGATE_BASELINES_SOURCE_EXPERIMENT,
    create_clr_ridge_weight_config,
    create_dsre_ceq_st_lite_weight_config,
)

logger = logging.getLogger(__name__)

DEFAULT_NAME_PREFIX = SURROGATE_BASELINES_SOURCE_EXPERIMENT


def _parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description="Launch the two-phase many-domain surrogate-derived baselines.")
    parser.add_argument("--name-prefix", default=DEFAULT_NAME_PREFIX)
    return parser.parse_known_args()


def main() -> None:
    args, remaining = _parse_args()
    sys.argv = [sys.argv[0], *remaining]

    if os.getenv("CI") is not None:
        logger.info("Skipping surrogate baseline launch in CI environment")
        return

    experiment = create_two_phase_dolma3_dolmino_top_level_experiment(
        name=args.name_prefix,
        resources=ResourceConfig.with_tpu("v5p-8", regions=["us-east5"]),
    )
    clr_step = experiment.create_training_step(
        weight_config=create_clr_ridge_weight_config(),
        name_prefix=args.name_prefix,
        run_name=CLR_RIDGE_RUN_NAME,
    )
    st_lite_step = experiment.create_training_step(
        weight_config=create_dsre_ceq_st_lite_weight_config(),
        name_prefix=args.name_prefix,
        run_name=DSRE_CEQ_ST_LITE_RUN_NAME,
    )

    logger.info(
        "Launching %s with alpha=%.2f and predicted %s=%.6f",
        CLR_RIDGE_RUN_NAME,
        CLR_RIDGE_ALPHA,
        "lm_eval/mmlu_5shot/bpb",
        CLR_RIDGE_PREDICTED_BPB,
    )
    logger.info(
        "Launching %s with predicted %s=%.6f",
        DSRE_CEQ_ST_LITE_RUN_NAME,
        "lm_eval/mmlu_5shot/bpb",
        DSRE_CEQ_ST_LITE_PREDICTED_BPB,
    )
    executor_main(
        steps=[clr_step, st_lite_step],
        description=f"{args.name_prefix}: {CLR_RIDGE_RUN_NAME} + {DSRE_CEQ_ST_LITE_RUN_NAME}",
    )


if __name__ == "__main__":
    main()
