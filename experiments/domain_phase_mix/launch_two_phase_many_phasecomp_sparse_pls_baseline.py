# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Launch the phase-composition sparse PLS predicted baseline."""

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
from experiments.domain_phase_mix.two_phase_many_phasecomp_sparse_pls_baseline import (
    PHASECOMP_SPARSE_PLS_RUN_NAME,
    PHASECOMP_SPARSE_PLS_SOURCE_EXPERIMENT,
    create_phasecomp_sparse_pls_weight_config,
    phasecomp_sparse_pls_summary,
    phasecomp_sparse_pls_summary_json,
)

logger = logging.getLogger(__name__)

DEFAULT_NAME_PREFIX = PHASECOMP_SPARSE_PLS_SOURCE_EXPERIMENT
FIT_SUMMARY_JSON = "phasecomp_sparse_pls_fit_summary.json"


@dataclass(frozen=True)
class SaveFitSummaryConfig:
    """Config for persisting the phase-composition sparse PLS summary."""

    output_path: str
    fit_summary_json: str


def save_fit_summary(config: SaveFitSummaryConfig) -> None:
    """Write the phase-composition sparse PLS summary JSON."""
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
        logger.info("Skipping phase-composition sparse PLS baseline launch in CI environment")
        return

    experiment = create_two_phase_dolma3_dolmino_top_level_experiment(
        name=args.name_prefix,
        resources=ResourceConfig.with_tpu(args.tpu_type, regions=["us-east5"]),
    )
    fit_summary_step = ExecutorStep(
        name=f"{args.name_prefix}/fit_summary",
        description="Save phase-composition sparse PLS optimum summary",
        fn=save_fit_summary,
        config=SaveFitSummaryConfig(
            output_path=this_output_path(),
            fit_summary_json=phasecomp_sparse_pls_summary_json(),
        ),
    )
    train_step = experiment.create_training_step(
        weight_config=create_phasecomp_sparse_pls_weight_config(),
        name_prefix=args.name_prefix,
        run_name=PHASECOMP_SPARSE_PLS_RUN_NAME,
    )

    summary = phasecomp_sparse_pls_summary()
    logger.info(
        "Launching %s with predicted bpb=%.6f, nearest observed TV=%.6f",
        PHASECOMP_SPARSE_PLS_RUN_NAME,
        float(summary["predicted_optimum_value"]),
        float(summary["nearest_observed_tv_distance"]),
    )
    executor_main(
        steps=[fit_summary_step, train_step],
        description=f"{args.name_prefix}: {PHASECOMP_SPARSE_PLS_RUN_NAME}",
    )


if __name__ == "__main__":
    main()
