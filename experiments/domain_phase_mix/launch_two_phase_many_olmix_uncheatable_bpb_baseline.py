# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Launch a newly fitted two-phase-many Olmix baseline on Uncheatable BPB."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from dataclasses import dataclass

import fsspec
import numpy as np
from fray.cluster import ResourceConfig
from marin.execution.executor import ExecutorStep, executor_main, this_output_path

from experiments.domain_phase_mix.config import WeightConfig
from experiments.domain_phase_mix.two_phase_dolma3_dolmino_top_level import (
    create_two_phase_dolma3_dolmino_top_level_experiment,
)
from experiments.domain_phase_mix.two_phase_many_olmix_loglinear_uncheatable import (
    FIT_SUMMARY_JSON,
    RUN_ID,
    RUN_NAME,
    SOURCE_EXPERIMENT,
    fit_summary_json,
    load_fit_from_local_results,
)

logger = logging.getLogger(__name__)

DEFAULT_NAME_PREFIX = SOURCE_EXPERIMENT


@dataclass(frozen=True)
class SaveFitSummaryConfig:
    """Config for persisting the fitted Uncheatable-BPB Olmix summary."""

    output_path: str
    fit_summary_json: str


def save_fit_summary(config: SaveFitSummaryConfig) -> None:
    """Write the fitted Olmix summary JSON."""
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
        logger.info("Skipping Uncheatable-BPB Olmix baseline launch in CI environment")
        return

    experiment = create_two_phase_dolma3_dolmino_top_level_experiment(
        name=args.name_prefix,
        resources=ResourceConfig.with_tpu(args.tpu_type, regions=["us-east5"]),
    )
    natural_proportions = np.asarray([float(domain.total_weight) for domain in experiment.domains], dtype=float)
    normalized_natural_proportions = natural_proportions / natural_proportions.sum()
    phase_fractions = np.asarray(
        [phase.end_fraction - phase.start_fraction for phase in experiment.phase_schedule.phases],
        dtype=float,
    )
    fit = load_fit_from_local_results(
        natural_proportions=normalized_natural_proportions,
        phase_fractions=phase_fractions,
    )
    weight_config = WeightConfig(run_id=RUN_ID, phase_weights=fit.phase_weights)

    fit_summary_step = ExecutorStep(
        name=f"{args.name_prefix}/fit_summary",
        description="Save fitted Uncheatable-BPB Olmix summary",
        fn=save_fit_summary,
        config=SaveFitSummaryConfig(
            output_path=this_output_path(),
            fit_summary_json=fit_summary_json(fit),
        ),
    )
    training_step = experiment.create_training_step(
        weight_config=weight_config,
        name_prefix=args.name_prefix,
        run_name=RUN_NAME,
    )

    logger.info(
        "Launching %s with predicted %s=%.6f and nearest observed TV=%.6f",
        RUN_NAME,
        fit.objective_metric,
        fit.predicted_objective,
        fit.nearest_observed_tv_distance,
    )
    executor_main(
        steps=[fit_summary_step, training_step],
        description=f"{args.name_prefix}: {RUN_NAME}",
    )


if __name__ == "__main__":
    main()
