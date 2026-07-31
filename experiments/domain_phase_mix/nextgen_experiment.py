# Copyright The Marin Authors
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
import json
import logging
import os
import sys

import fsspec
from marin.execution.executor import executor_main

from experiments.domain_phase_mix.nextgen.contracts import LoopConfig
from experiments.domain_phase_mix.nextgen.import_sources import (
    CsvDomainPhaseImportSource,
    LegacyDomainPhaseImportSource,
    default_legacy_sources,
    source_from_dict,
)
from experiments.domain_phase_mix.nextgen.model_registry import available_model_names
from experiments.domain_phase_mix.nextgen.pipeline import create_nextgen_steps, summarize_step_names
from experiments.domain_phase_mix.three_phase_starcoder_experiment import create_three_phase_experiment
from experiments.domain_phase_mix.two_phase_dolma3_dolmino_top_level import (
    MIN_RECOMMENDED_SWARM_RUNS,
    create_two_phase_dolma3_dolmino_top_level_experiment,
)
from experiments.domain_phase_mix.two_phase_starcoder_experiment import create_two_phase_experiment

logger = logging.getLogger(__name__)


DEFAULT_OBJECTIVE = "eval/paloma/dolma_100_programing_languages/bpb"
DEFAULT_LOOP_NAME = "pinlin_calvin_xu/data_mixture/nextgen_loop"


def _parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description="Next-gen mixture loop")
    parser.add_argument(
        "--experiment",
        type=str,
        default="two_phase_starcoder",
        choices=("two_phase_starcoder", "three_phase_starcoder", "two_phase_dolma3_dolmino_top_level"),
        help="Which domain_phase_mix experiment topology to use for new runs.",
    )
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
        "--import-csv",
        action="append",
        default=[],
        help="Additional CSV import source path (repeatable).",
    )
    parser.add_argument(
        "--import-source-json",
        action="append",
        default=[],
        help="Path to a JSON file containing one serialized import source dict.",
    )
    parser.add_argument(
        "--state-root",
        type=str,
        default="domain_phase_mix/nextgen",
        help="Persistent root directory for loop state and artifacts.",
    )
    parser.add_argument(
        "--candidate-opt-method",
        type=str,
        default="sample",
        choices=("sample", "scipy_minimize"),
        help="Candidate extraction method after model fitting.",
    )
    parser.add_argument(
        "--candidate-opt-restarts",
        type=int,
        default=128,
        help="Number of random restarts for scipy-based candidate optimization.",
    )
    parser.add_argument(
        "--candidate-opt-maxiter",
        type=int,
        default=400,
        help="Maximum optimizer iterations per scipy restart.",
    )
    parser.add_argument(
        "--design-policy",
        type=str,
        default="sampler",
        choices=("sampler", "static_d_optimal"),
        help="How to plan new swarm runs before training.",
    )
    parser.add_argument(
        "--design-candidate-pool-size",
        type=int,
        default=2048,
        help="Candidate pool size used by the static D-optimal design policy.",
    )
    parser.add_argument("--wandb-entity", type=str, default="marin-community")
    parser.add_argument("--wandb-project", type=str, default="marin")
    return parser.parse_known_args()


def _domain_token_counts_from_experiment(experiment) -> dict[str, int]:
    domain_token_counts: dict[str, int] = {}
    for domain in experiment.domains:
        total_weight = domain.total_weight
        if isinstance(total_weight, float) and not total_weight.is_integer():
            raise ValueError(f"Domain '{domain.name}' total_weight is not an exact integer token count: {total_weight}")
        domain_token_counts[domain.name] = int(total_weight)
    return domain_token_counts


def _load_import_source_json(path: str):
    with fsspec.open(path, "r") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Import source JSON must contain one dict, got {type(payload)} from {path}")
    return source_from_dict(payload)


def _build_loop_config(args: argparse.Namespace, experiment) -> LoopConfig:
    model_names = tuple(name.strip() for name in args.models.split(",") if name.strip())
    import_sources: list[LegacyDomainPhaseImportSource | CsvDomainPhaseImportSource] = []
    for csv_path in args.import_csv:
        import_sources.append(
            CsvDomainPhaseImportSource(
                source_experiment=args.name,
                csv_path=csv_path,
            )
        )
    for import_source_path in args.import_source_json:
        import_sources.append(_load_import_source_json(import_source_path))
    if args.import_legacy:
        import_sources.extend(default_legacy_sources())

    phase_fractions = tuple(phase.end_fraction - phase.start_fraction for phase in experiment.phase_schedule.phases)

    return LoopConfig(
        name=args.name,
        objective_metric=args.objective_metric,
        model_names=model_names,
        n_new_runs=args.n_new_runs,
        import_sources=tuple(import_sources),
        state_root=args.state_root,
        domain_token_counts=_domain_token_counts_from_experiment(experiment),
        phase_fractions=phase_fractions,
        target_budget=experiment.target_budget,
        candidate_opt_method=args.candidate_opt_method,
        candidate_opt_restarts=args.candidate_opt_restarts,
        candidate_opt_maxiter=args.candidate_opt_maxiter,
        design_policy=args.design_policy,
        design_candidate_pool_size=args.design_candidate_pool_size,
    )


def _build_experiment(args: argparse.Namespace):
    if args.experiment == "two_phase_dolma3_dolmino_top_level":
        return create_two_phase_dolma3_dolmino_top_level_experiment(name=args.name)
    if args.experiment == "three_phase_starcoder":
        return create_three_phase_experiment(name=args.name)
    return create_two_phase_experiment(name=args.name)


def main() -> None:
    args, remaining = _parse_args()
    sys.argv = [sys.argv[0], *remaining]

    if os.getenv("CI") is not None:
        logger.info("Skipping execution in CI environment")
        return

    experiment = _build_experiment(args)
    if args.experiment == "two_phase_dolma3_dolmino_top_level":
        total_requested_runs = args.n_new_runs + len(experiment.initial_fixed_weight_configs)
        if args.n_new_runs > 0 and total_requested_runs < MIN_RECOMMENDED_SWARM_RUNS:
            logger.warning(
                "Requested %d total runs for %d domains; recommend at least %d total runs (6x domains).",
                total_requested_runs,
                len(experiment.domains),
                MIN_RECOMMENDED_SWARM_RUNS,
            )
    loop = _build_loop_config(args, experiment)

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
