# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Entrypoint for the next-generation domain/phase mixture loop.

This script provides a single submission surface for:
- initial runs,
- follow-up runs,
- model additions (fit/propose/validate reruns), and
- legacy trajectory imports.
"""

from __future__ import annotations

import argparse
import logging
import os

from marin.execution.executor import executor_main

from experiments.domain_phase_mix.nextgen.contracts import LoopConfig
from experiments.domain_phase_mix.nextgen.import_sources import (
    LegacyDomainPhaseImportSource,
    default_legacy_sources,
)
from experiments.domain_phase_mix.nextgen.model_registry import available_model_names
from experiments.domain_phase_mix.nextgen.pipeline import create_nextgen_steps, summarize_step_names
from experiments.domain_phase_mix.two_phase_starcoder_experiment import create_two_phase_experiment

logger = logging.getLogger("ray")


DEFAULT_OBJECTIVE = "eval/paloma/dolma_100_programing_languages/bpb"
DEFAULT_LOOP_NAME = "pinlin_calvin_xu/data_mixture/nextgen_loop"


def _parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description="Next-gen mixture loop")
    parser.add_argument("--name", type=str, default=DEFAULT_LOOP_NAME, help="Loop name used for run tags and artifacts.")
    parser.add_argument(
        "--objective-metric",
        type=str,
        default=DEFAULT_OBJECTIVE,
        help="Objective metric key to optimize and track.",
    )
    parser.add_argument("--n-new-runs", type=int, default=0, help="Number of additional runs to sample and submit.")
    parser.add_argument(
        "--models",
        type=str,
        default=",".join(sorted(available_model_names())),
        help="Comma-separated model names from the registry.",
    )
    parser.add_argument(
        "--import-legacy",
        action="store_true",
        help="Import configured legacy experiments into the merged trajectory store.",
    )
    parser.add_argument(
        "--state-root",
        type=str,
        default="domain_phase_mix/nextgen",
        help="Persistent root directory for loop state and artifacts.",
    )
    parser.add_argument("--wandb-entity", type=str, default="marin-community")
    parser.add_argument("--wandb-project", type=str, default="marin")
    return parser.parse_known_args()



def _build_loop_config(args: argparse.Namespace) -> LoopConfig:
    model_names = tuple(name.strip() for name in args.models.split(",") if name.strip())
    import_sources: tuple[LegacyDomainPhaseImportSource, ...]
    if args.import_legacy:
        import_sources = default_legacy_sources()
    else:
        import_sources = ()

    return LoopConfig(
        name=args.name,
        objective_metric=args.objective_metric,
        model_names=model_names,
        n_new_runs=args.n_new_runs,
        import_sources=import_sources,
        state_root=args.state_root,
    )



def main() -> None:
    args, _ = _parse_args()

    if os.getenv("CI") is not None:
        logger.info("Skipping execution in CI environment")
        return

    experiment = create_two_phase_experiment(name=args.name)
    loop = _build_loop_config(args)

    steps = create_nextgen_steps(
        loop=loop,
        experiment=experiment,
        wandb_entity=args.wandb_entity,
        wandb_project=args.wandb_project,
    )

    logger.info("Created %d next-gen steps", len(steps))
    for step_name in summarize_step_names(steps):
        logger.info("  %s", step_name)

    executor_main(
        steps=steps,
        description=f"{loop.name}: next-gen mixture loop",
    )


if __name__ == "__main__":
    main()
