# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Launch a seed-sensitivity repeat study for the exact run_00097 schedule."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import asdict, dataclass
from typing import Literal

import fsspec
from fray.cluster import ResourceConfig
from marin.execution.executor import ExecutorStep, executor_main, this_output_path

from experiments.domain_phase_mix.config import WeightConfig
from experiments.domain_phase_mix.determinism_analysis import (
    RUN_MANIFEST_FILE,
    SWARM_COMPARISON_CSV,
    SWARM_COMPARISON_JSON,
    create_determinism_report_step,
    create_manifest_results_step,
)
from experiments.domain_phase_mix.two_phase_many_observed_runs import (
    TWO_PHASE_MANY_CSV_PATH,
    load_two_phase_many_phase_weights,
)
from experiments.domain_phase_mix.two_phase_dolma3_dolmino_top_level import (
    create_two_phase_dolma3_dolmino_top_level_experiment,
)

logger = logging.getLogger(__name__)

NAME = "pinlin_calvin_xu/data_mixture/ngd3dm2_run00097_seed_study"
SOURCE_RUN_NAME = "run_00097"
OBJECTIVE_METRIC = "lm_eval/mmlu_5shot/bpb"
WANDB_ENTITY = "marin-community"
WANDB_PROJECT = "marin"
SEED_SWEEP_START = 10_000
N_SEED_SWEEP_RUNS = 10
EXACT_CONTROL_NAMES = ("exact_replay_control_a", "exact_replay_control_b")
EXACT_CONTROL_TRAINER_SEED = 0
EXACT_CONTROL_DATA_SEED = 97
SWARM_COMPARISON_METRICS = (
    "lm_eval/mmlu_5shot/bpb",
    "lm_eval/mmlu_5shot/choice_logprob",
    "eval/paloma/c4_en/bpb",
    "eval/paloma/macro_bpb",
    "eval/uncheatable_eval/macro_bpb",
)


@dataclass(frozen=True)
class SeedStudyRunSpec:
    """Manifest entry for one run in the seed-sensitivity study."""

    run_id: int
    run_name: str
    cohort: Literal["seed_sweep", "exact_replay_control"]
    trainer_seed: int
    data_seed: int | None
    source_run_name: str
    phase_weights: dict[str, dict[str, float]]


@dataclass(frozen=True)
class SaveRunManifestConfig:
    """Config for writing the seed-study manifest."""

    output_path: str
    experiment_name: str
    objective_metric: str
    run_specs_json: str


RUN_00097_PHASE_WEIGHTS = load_two_phase_many_phase_weights(SOURCE_RUN_NAME)


def build_run_specs() -> list[SeedStudyRunSpec]:
    """Build the 12-run study manifest."""
    run_specs: list[SeedStudyRunSpec] = []
    next_run_id = 0

    for offset in range(N_SEED_SWEEP_RUNS):
        trainer_seed = SEED_SWEEP_START + offset
        run_specs.append(
            SeedStudyRunSpec(
                run_id=next_run_id,
                run_name=f"trainer_seed_{trainer_seed}",
                cohort="seed_sweep",
                trainer_seed=trainer_seed,
                data_seed=None,
                source_run_name=SOURCE_RUN_NAME,
                phase_weights=RUN_00097_PHASE_WEIGHTS,
            )
        )
        next_run_id += 1

    for run_name in EXACT_CONTROL_NAMES:
        run_specs.append(
            SeedStudyRunSpec(
                run_id=next_run_id,
                run_name=run_name,
                cohort="exact_replay_control",
                trainer_seed=EXACT_CONTROL_TRAINER_SEED,
                data_seed=EXACT_CONTROL_DATA_SEED,
                source_run_name=SOURCE_RUN_NAME,
                phase_weights=RUN_00097_PHASE_WEIGHTS,
            )
        )
        next_run_id += 1

    return run_specs


def save_run_manifest(config: SaveRunManifestConfig) -> None:
    """Persist the study manifest for downstream collection and analysis."""
    run_specs = json.loads(config.run_specs_json)
    payload = {
        "experiment_name": config.experiment_name,
        "objective_metric": config.objective_metric,
        "n_runs": len(run_specs),
        "runs": run_specs,
    }
    fs, _, _ = fsspec.get_fs_token_paths(config.output_path)
    fs.makedirs(config.output_path, exist_ok=True)
    with fsspec.open(os.path.join(config.output_path, RUN_MANIFEST_FILE), "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def create_run_manifest_step(*, name_prefix: str, run_specs: list[SeedStudyRunSpec]) -> ExecutorStep:
    """Create the manifest writer step for this study."""
    return ExecutorStep(
        name=f"{name_prefix}/run_manifest",
        description=f"Save run_00097 seed-sensitivity manifest ({len(run_specs)} runs)",
        fn=save_run_manifest,
        config=SaveRunManifestConfig(
            output_path=this_output_path(),
            experiment_name=name_prefix,
            objective_metric=OBJECTIVE_METRIC,
            run_specs_json=json.dumps([asdict(spec) for spec in run_specs], sort_keys=True),
        ),
    )


def _parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description="Launch the run_00097 seed-sensitivity repeat study.")
    parser.add_argument("--name-prefix", default=NAME)
    parser.add_argument("--source-run-name", default=SOURCE_RUN_NAME)
    parser.add_argument("--tpu-type", default="v5p-8")
    return parser.parse_known_args()


def main() -> None:
    args, remaining = _parse_args()
    sys.argv = [sys.argv[0], *remaining]

    if os.getenv("CI") is not None:
        logger.info("Skipping run_00097 seed study launch in CI environment")
        return
    if args.source_run_name != SOURCE_RUN_NAME:
        raise ValueError(f"Only {SOURCE_RUN_NAME!r} is currently supported, got {args.source_run_name!r}")

    experiment = create_two_phase_dolma3_dolmino_top_level_experiment(
        name=args.name_prefix,
        resources=ResourceConfig.with_tpu(args.tpu_type, regions=["us-east5"]),
    )
    run_specs = build_run_specs()
    run_manifest_step = create_run_manifest_step(name_prefix=args.name_prefix, run_specs=run_specs)

    training_steps: list[ExecutorStep] = []
    for spec in run_specs:
        step = experiment.create_training_step(
            weight_config=WeightConfig(run_id=spec.run_id, phase_weights=spec.phase_weights),
            name_prefix=args.name_prefix,
            run_name=spec.run_name,
            trainer_seed=spec.trainer_seed,
            data_seed=spec.data_seed,
        )
        training_steps.append(step)

    results_step = create_manifest_results_step(
        name_prefix=args.name_prefix,
        run_manifest_step=run_manifest_step,
        wandb_entity=WANDB_ENTITY,
        wandb_project=WANDB_PROJECT,
        extra_metrics=(OBJECTIVE_METRIC, *SWARM_COMPARISON_METRICS),
        depends_on=training_steps,
    )
    report_step = create_determinism_report_step(
        name_prefix=args.name_prefix,
        objective_metric=OBJECTIVE_METRIC,
        run_manifest_step=run_manifest_step,
        analysis_step=results_step,
        wandb_entity=WANDB_ENTITY,
        wandb_project=WANDB_PROJECT,
        swarm_results_csv_path=str(TWO_PHASE_MANY_CSV_PATH),
        comparison_metrics=SWARM_COMPARISON_METRICS,
    )

    logger.info(
        "Launching %d run_00097 repeats on %s. Final report will include %s and %s.",
        len(run_specs),
        args.tpu_type,
        SWARM_COMPARISON_JSON,
        SWARM_COMPARISON_CSV,
    )
    executor_main(
        steps=[run_manifest_step, *training_steps, results_step, report_step],
        description=f"{args.name_prefix}: run_00097 seed-sensitivity repeat study",
    )


if __name__ == "__main__":
    main()
