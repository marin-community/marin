# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Launch the frozen Olmix log-linear baseline for the two-phase many-domain sweep."""

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
from experiments.domain_phase_mix.two_phase_many_olmix_loglinear import (
    OLMIX_LOGLINEAR_PREDICTED_BPB,
    OLMIX_LOGLINEAR_RUN_NAME,
    create_olmix_loglinear_weight_config,
)

logger = logging.getLogger(__name__)

DEFAULT_NAME_PREFIX = "pinlin_calvin_xu/data_mixture/ngd3dm2_olmix_bpb"


def _parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description="Launch the two-phase many-domain Olmix log-linear baseline.")
    parser.add_argument("--name-prefix", default=DEFAULT_NAME_PREFIX)
    return parser.parse_known_args()


def main() -> None:
    args, remaining = _parse_args()
    sys.argv = [sys.argv[0], *remaining]

    if os.getenv("CI") is not None:
        logger.info("Skipping Olmix baseline launch in CI environment")
        return

    experiment = create_two_phase_dolma3_dolmino_top_level_experiment(
        name=args.name_prefix,
        resources=ResourceConfig.with_tpu("v5p-8", regions=["us-east5"]),
    )
    weight_config = create_olmix_loglinear_weight_config()
    training_step = experiment.create_training_step(
        weight_config=weight_config,
        name_prefix=args.name_prefix,
        run_name=OLMIX_LOGLINEAR_RUN_NAME,
    )

    logger.info(
        "Launching %s with predicted %s=%.6f",
        OLMIX_LOGLINEAR_RUN_NAME,
        "lm_eval/mmlu_5shot/bpb",
        OLMIX_LOGLINEAR_PREDICTED_BPB,
    )
    executor_main(
        steps=[training_step],
        description=f"{args.name_prefix}: {OLMIX_LOGLINEAR_RUN_NAME}",
    )


if __name__ == "__main__":
    main()
