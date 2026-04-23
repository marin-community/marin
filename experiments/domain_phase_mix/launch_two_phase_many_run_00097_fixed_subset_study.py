# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Launch a fixed-subset seed-sensitivity study for the exact run_00097 schedule."""

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
    RUN_MANIFEST_FILE,
    create_determinism_report_step,
    create_fixed_subset_noise_report_step,
    create_manifest_results_step,
)
from experiments.domain_phase_mix.launch_two_phase_many_run_00097_seed_study import (
    EXACT_CONTROL_DATA_SEED,
    EXACT_CONTROL_NAMES,
    NAME as BASELINE_SEED_STUDY_NAME,
    OBJECTIVE_METRIC,
    RUN_00097_PHASE_WEIGHTS,
    SEED_SWEEP_START,
    SOURCE_RUN_NAME,
    SWARM_COMPARISON_METRICS,
    WANDB_ENTITY,
    WANDB_PROJECT,
    build_run_specs as build_run_00097_seed_specs,
)
from experiments.domain_phase_mix.two_phase_dolma3_dolmino_top_level import (
    create_two_phase_dolma3_dolmino_top_level_experiment,
)

logger = logging.getLogger(__name__)

NAME = "pinlin_calvin_xu/data_mixture/ngd3dm2_run00097_fixed_subset_study"
N_SEED_SWEEP_RUNS = 10
SIMULATED_EPOCH_SUBSET_SEED = EXACT_CONTROL_DATA_SEED
PRIMARY_METRICS = (
    "lm_eval/mmlu_5shot/bpb",
    "lm_eval/mmlu_5shot/choice_logprob",
    "lm_eval/mmlu_5shot/choice_logprob_norm",
    "lm_eval/mmlu_5shot/choice_prob_norm",
)
SECONDARY_METRICS = (
    "eval/paloma/c4_en/bpb",
    "eval/paloma/macro_bpb",
    "eval/uncheatable_eval/macro_bpb",
)


@dataclass(frozen=True)
class FixedSubsetRunSpec:
    """Manifest entry for one run in the fixed-subset seed study."""

    run_id: int
    run_name: str
    cohort: Literal["seed_sweep", "exact_replay_control"]
    trainer_seed: int
    data_seed: int | None
    simulated_epoch_subset_seed: int
    source_run_name: str
    phase_weights: dict[str, dict[str, float]]


@dataclass(frozen=True)
class SaveRunManifestConfig:
    """Config for writing the fixed-subset study manifest."""

    output_path: str
    experiment_name: str
    objective_metric: str
    run_specs_json: str


def build_run_specs() -> list[FixedSubsetRunSpec]:
    """Build the 12-run fixed-subset study manifest."""
    run_specs: list[FixedSubsetRunSpec] = []
    next_run_id = 0

    for offset in range(N_SEED_SWEEP_RUNS):
        trainer_seed = SEED_SWEEP_START + offset
        run_specs.append(
            FixedSubsetRunSpec(
                run_id=next_run_id,
                run_name=f"trainer_seed_{trainer_seed}",
                cohort="seed_sweep",
                trainer_seed=trainer_seed,
                data_seed=None,
                simulated_epoch_subset_seed=SIMULATED_EPOCH_SUBSET_SEED,
                source_run_name=SOURCE_RUN_NAME,
                phase_weights=RUN_00097_PHASE_WEIGHTS,
            )
        )
        next_run_id += 1

    for run_name in EXACT_CONTROL_NAMES:
        run_specs.append(
            FixedSubsetRunSpec(
                run_id=next_run_id,
                run_name=run_name,
                cohort="exact_replay_control",
                trainer_seed=0,
                data_seed=EXACT_CONTROL_DATA_SEED,
                simulated_epoch_subset_seed=SIMULATED_EPOCH_SUBSET_SEED,
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


def create_run_manifest_step(*, name_prefix: str, run_specs: list[FixedSubsetRunSpec]) -> ExecutorStep:
    """Create the manifest writer step for this study."""
    return ExecutorStep(
        name=f"{name_prefix}/run_manifest",
        description=f"Save run_00097 fixed-subset manifest ({len(run_specs)} runs)",
        fn=save_run_manifest,
        config=SaveRunManifestConfig(
            output_path=this_output_path(),
            experiment_name=name_prefix,
            objective_metric=OBJECTIVE_METRIC,
            run_specs_json=json.dumps([asdict(spec) for spec in run_specs], sort_keys=True),
        ),
    )


def _build_baseline_seed_manifest_json() -> str:
    run_specs = [asdict(spec) for spec in build_run_00097_seed_specs() if spec.cohort == "seed_sweep"]
    payload = {
        "experiment_name": BASELINE_SEED_STUDY_NAME,
        "objective_metric": OBJECTIVE_METRIC,
        "n_runs": len(run_specs),
        "runs": run_specs,
    }
    return json.dumps(payload, sort_keys=True)


def _parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description="Launch the run_00097 fixed-subset repeat study.")
    parser.add_argument("--name-prefix", default=NAME)
    parser.add_argument("--source-run-name", default=SOURCE_RUN_NAME)
    parser.add_argument("--tpu-type", default="v5p-8")
    return parser.parse_known_args()


def main() -> None:
    args, remaining = _parse_args()
    sys.argv = [sys.argv[0], *remaining]

    if os.getenv("CI") is not None:
        logger.info("Skipping run_00097 fixed-subset study launch in CI environment")
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

    all_metrics = tuple(dict.fromkeys((*PRIMARY_METRICS, *SECONDARY_METRICS)))
    results_step = create_manifest_results_step(
        name_prefix=args.name_prefix,
        run_manifest_step=run_manifest_step,
        wandb_entity=WANDB_ENTITY,
        wandb_project=WANDB_PROJECT,
        extra_metrics=all_metrics,
        depends_on=training_steps,
    )
    determinism_step = create_determinism_report_step(
        name_prefix=args.name_prefix,
        objective_metric=OBJECTIVE_METRIC,
        run_manifest_step=run_manifest_step,
        analysis_step=results_step,
        wandb_entity=WANDB_ENTITY,
        wandb_project=WANDB_PROJECT,
        comparison_metrics=tuple(dict.fromkeys((*PRIMARY_METRICS, *SECONDARY_METRICS, *SWARM_COMPARISON_METRICS))),
    )
    fixed_subset_report_step = create_fixed_subset_noise_report_step(
        name_prefix=args.name_prefix,
        analysis_step=determinism_step,
        wandb_entity=WANDB_ENTITY,
        wandb_project=WANDB_PROJECT,
        baseline_manifest_json=_build_baseline_seed_manifest_json(),
        primary_metrics=PRIMARY_METRICS,
        secondary_metrics=SECONDARY_METRICS,
    )

    logger.info(
        "Launching %d run_00097 fixed-subset repeats on %s with simulated_epoch_subset_seed=%d.",
        len(run_specs),
        args.tpu_type,
        SIMULATED_EPOCH_SUBSET_SEED,
    )
    all_steps = [run_manifest_step, *training_steps, results_step, determinism_step, fixed_subset_report_step]
    executor_main(
        ExecutorMainConfig(max_concurrent=len(training_steps)),
        steps=all_steps,
        description=f"{args.name_prefix}: run_00097 fixed-subset repeat study",
    )


if __name__ == "__main__":
    main()
