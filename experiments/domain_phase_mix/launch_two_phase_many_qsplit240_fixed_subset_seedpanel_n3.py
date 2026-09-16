# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Launch a 3-seed fixed-subset replicated swarm over the original qsplit240 candidates."""

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
from marin.execution.executor import ExecutorMainConfig, ExecutorStep, executor_main, this_output_path

from experiments.domain_phase_mix.config import WeightConfig
from experiments.domain_phase_mix.determinism_analysis import (
    FIT_DATASET_CSV,
    FIT_DATASET_SUMMARY_JSON,
    RESULTS_CSV,
    RUN_MANIFEST_FILE,
    create_fit_dataset_export_step,
    create_manifest_results_step,
)
from experiments.domain_phase_mix.launch_two_phase_many_run_00097_fixed_subset_study import (
    SIMULATED_EPOCH_SUBSET_SEED,
    WANDB_ENTITY,
    WANDB_PROJECT,
)
from experiments.domain_phase_mix.launch_two_phase_many_run_00097_seed_study import SEED_SWEEP_START
from experiments.domain_phase_mix.two_phase_dolma3_dolmino_top_level import (
    create_two_phase_dolma3_dolmino_top_level_experiment,
)
from experiments.domain_phase_mix.two_phase_many_observed_runs import load_original_qsplit240_with_core_baselines

logger = logging.getLogger(__name__)

NAME = "pinlin_calvin_xu/data_mixture/ngd3dm2_qsplit240_fixed_subset_seedpanel_n3"
TRAINER_SEEDS = (SEED_SWEEP_START, SEED_SWEEP_START + 1, SEED_SWEEP_START + 2)
DEFAULT_MAX_CONCURRENT = 256


@dataclass(frozen=True)
class ReplicatedSwarmRunSpec:
    """Manifest entry for one replicated qsplit240 fixed-subset run."""

    run_id: int
    run_name: str
    cohort: Literal["replicated_swarm"]
    trainer_seed: int
    data_seed: int | None
    simulated_epoch_subset_seed: int
    candidate_run_id: int
    candidate_run_name: str
    candidate_source_experiment: str
    phase_weights: dict[str, dict[str, float]]


@dataclass(frozen=True)
class SaveRunManifestConfig:
    """Config for writing the replicated swarm manifest."""

    output_path: str
    experiment_name: str
    run_specs_json: str


def build_run_specs() -> list[ReplicatedSwarmRunSpec]:
    """Build the full replicated qsplit240-plus-baselines manifest."""
    candidate_runs = load_original_qsplit240_with_core_baselines()
    run_specs: list[ReplicatedSwarmRunSpec] = []
    next_run_id = 0

    for candidate in candidate_runs:
        for trainer_seed in TRAINER_SEEDS:
            run_specs.append(
                ReplicatedSwarmRunSpec(
                    run_id=next_run_id,
                    run_name=f"seed{trainer_seed}_{candidate.run_name}",
                    cohort="replicated_swarm",
                    trainer_seed=trainer_seed,
                    data_seed=None,
                    simulated_epoch_subset_seed=SIMULATED_EPOCH_SUBSET_SEED,
                    candidate_run_id=candidate.run_id,
                    candidate_run_name=candidate.run_name,
                    candidate_source_experiment=candidate.source_experiment,
                    phase_weights=candidate.phase_weights,
                )
            )
            next_run_id += 1

    return run_specs


def save_run_manifest(config: SaveRunManifestConfig) -> None:
    """Persist the replicated swarm manifest for downstream collection and fitting."""
    run_specs = json.loads(config.run_specs_json)
    payload = {
        "experiment_name": config.experiment_name,
        "n_runs": len(run_specs),
        "trainer_seeds": list(TRAINER_SEEDS),
        "runs": run_specs,
    }
    fs, _, _ = fsspec.get_fs_token_paths(config.output_path)
    fs.makedirs(config.output_path, exist_ok=True)
    with fsspec.open(os.path.join(config.output_path, RUN_MANIFEST_FILE), "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def create_run_manifest_step(*, name_prefix: str, run_specs: list[ReplicatedSwarmRunSpec]) -> ExecutorStep:
    """Create the manifest writer step for the replicated swarm."""
    return ExecutorStep(
        name=f"{name_prefix}/run_manifest",
        description=f"Save qsplit240 replicated swarm manifest ({len(run_specs)} runs)",
        fn=save_run_manifest,
        config=SaveRunManifestConfig(
            output_path=this_output_path(),
            experiment_name=name_prefix,
            run_specs_json=json.dumps([asdict(spec) for spec in run_specs], sort_keys=True),
        ),
    )


def _parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description="Launch a 3-seed fixed-subset qsplit240 replicated swarm.")
    parser.add_argument("--name-prefix", default=NAME)
    parser.add_argument("--tpu-type", default="v5p-8")
    parser.add_argument("--max-concurrent", type=int, default=DEFAULT_MAX_CONCURRENT)
    return parser.parse_known_args()


def main() -> None:
    args, remaining = _parse_args()
    sys.argv = [sys.argv[0], *remaining]

    if os.getenv("CI") is not None:
        logger.info("Skipping qsplit240 fixed-subset replicated swarm launch in CI environment")
        return

    experiment = create_two_phase_dolma3_dolmino_top_level_experiment(
        name=args.name_prefix,
        resources=ResourceConfig.with_tpu(args.tpu_type, regions=["us-east5"]),
    )
    run_specs = build_run_specs()
    run_manifest_step = create_run_manifest_step(name_prefix=args.name_prefix, run_specs=run_specs)

    training_steps: list[ExecutorStep] = []
    for spec in run_specs:
        training_steps.append(
            experiment.create_training_step(
                weight_config=WeightConfig(run_id=spec.run_id, phase_weights=spec.phase_weights),
                name_prefix=args.name_prefix,
                run_name=spec.run_name,
                trainer_seed=spec.trainer_seed,
                data_seed=spec.data_seed,
                simulated_epoch_subset_seed=spec.simulated_epoch_subset_seed,
            )
        )

    results_step = create_manifest_results_step(
        name_prefix=args.name_prefix,
        run_manifest_step=run_manifest_step,
        wandb_entity=WANDB_ENTITY,
        wandb_project=WANDB_PROJECT,
        depends_on=training_steps,
    )
    fit_dataset_step = create_fit_dataset_export_step(
        name_prefix=args.name_prefix,
        run_manifest_step=run_manifest_step,
        analysis_step=results_step,
    )

    logger.info(
        "Launching %d replicated qsplit240 runs on %s with trainer seeds %s, fixed subset %d, and max_concurrent=%d. "
        "Outputs will include %s, %s, and %s.",
        len(run_specs),
        args.tpu_type,
        TRAINER_SEEDS,
        SIMULATED_EPOCH_SUBSET_SEED,
        args.max_concurrent,
        RESULTS_CSV,
        FIT_DATASET_CSV,
        FIT_DATASET_SUMMARY_JSON,
    )
    all_steps = [run_manifest_step, *training_steps, results_step, fit_dataset_step]
    executor_main(
        ExecutorMainConfig(max_concurrent=args.max_concurrent),
        steps=all_steps,
        description=f"{args.name_prefix}: 3-seed fixed-subset replicated swarm over original qsplit240 candidates",
    )


if __name__ == "__main__":
    main()
