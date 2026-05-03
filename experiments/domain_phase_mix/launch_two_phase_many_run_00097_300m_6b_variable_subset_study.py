# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Launch a 300M / 6B variable-subset seed study for the exact run_00097 schedule."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import asdict, dataclass, replace
from typing import Literal

import fsspec
from fray.cluster import ResourceConfig
from marin.execution.executor import ExecutorMainConfig, ExecutorStep, executor_main, this_output_path
from marin.training.training import TrainLmOnPodConfig

from experiments.domain_phase_mix.config import WeightConfig
from experiments.domain_phase_mix.determinism_analysis import (
    RUN_MANIFEST_FILE,
    create_manifest_results_step,
    create_model_size_noise_report_step,
)
from experiments.domain_phase_mix.launch_two_phase_many_run_00097_compute_scaling_study import (
    NAME as COMPUTE_SCALING_BASELINE_NAME,
)
from experiments.domain_phase_mix.launch_two_phase_many_run_00097_compute_scaling_study import (
    OBJECTIVE_METRIC,
    REGMIX60M_6B_LADDER,
    WANDB_ENTITY,
    WANDB_PROJECT,
)
from experiments.domain_phase_mix.launch_two_phase_many_run_00097_compute_scaling_study import (
    build_run_specs as build_compute_scaling_run_specs,
)
from experiments.domain_phase_mix.launch_two_phase_many_run_00097_fixed_subset_study import (
    NAME as FIXED_SUBSET_BASELINE_NAME,
)
from experiments.domain_phase_mix.launch_two_phase_many_run_00097_fixed_subset_study import (
    PRIMARY_METRICS,
    SECONDARY_METRICS,
)
from experiments.domain_phase_mix.launch_two_phase_many_run_00097_fixed_subset_study import (
    build_run_specs as build_fixed_subset_run_specs,
)
from experiments.domain_phase_mix.launch_two_phase_many_run_00097_seed_study import (
    RUN_00097_PHASE_WEIGHTS,
    SEED_SWEEP_START,
    SOURCE_RUN_NAME,
)
from experiments.domain_phase_mix.proxy_sweep import (
    get_num_train_steps,
    regmix_300m_muonh_base,
    regmix_300m_proxy,
)
from experiments.domain_phase_mix.two_phase_dolma3_dolmino_top_level import (
    BATCH_SIZE,
    SEQ_LEN,
    create_two_phase_dolma3_dolmino_top_level_experiment,
)

logger = logging.getLogger(__name__)

NAME = "pinlin_calvin_xu/data_mixture/ngd3dm2_run00097_300m_6b_variable_subset"
MODEL_FAMILY = "regmix_300m_proxy"
EXPERIMENT_BUDGET = 6_000_000_000
N_SEED_SWEEP_RUNS = 10
DEFAULT_TPU_REGION = "us-east5"
DEFAULT_TPU_ZONE = "us-east5-a"


@dataclass(frozen=True)
class VariableSubsetRunSpec:
    """Manifest entry for one run in the 300M / 6B variable-subset study."""

    run_id: int
    run_name: str
    cohort: Literal["seed_sweep"]
    model_family: str
    trainer_seed: int
    data_seed: int | None
    simulated_epoch_subset_seed: None
    source_run_name: str
    experiment_budget: int
    num_train_steps: int
    phase_weights: dict[str, dict[str, float]]


@dataclass(frozen=True)
class SaveRunManifestConfig:
    """Config for writing the variable-subset study manifest."""

    output_path: str
    experiment_name: str
    objective_metric: str
    run_specs_json: str


def build_run_specs() -> list[VariableSubsetRunSpec]:
    """Build the 10-run 300M / 6B variable-subset study manifest."""
    num_train_steps = get_num_train_steps(EXPERIMENT_BUDGET, BATCH_SIZE, SEQ_LEN)
    run_specs: list[VariableSubsetRunSpec] = []
    for offset in range(N_SEED_SWEEP_RUNS):
        trainer_seed = SEED_SWEEP_START + offset
        run_specs.append(
            VariableSubsetRunSpec(
                run_id=offset,
                run_name=f"regmix300m_6b_variable_subset_trainer_seed_{trainer_seed}",
                cohort="seed_sweep",
                model_family=MODEL_FAMILY,
                trainer_seed=trainer_seed,
                data_seed=None,
                simulated_epoch_subset_seed=None,
                source_run_name=SOURCE_RUN_NAME,
                experiment_budget=EXPERIMENT_BUDGET,
                num_train_steps=num_train_steps,
                phase_weights=RUN_00097_PHASE_WEIGHTS,
            )
        )
    return run_specs


def validate_run_specs(run_specs: list[VariableSubsetRunSpec]) -> None:
    """Fail fast if the variable-subset study is not the intended ten-row design."""
    if len(run_specs) != N_SEED_SWEEP_RUNS:
        raise ValueError(f"Expected {N_SEED_SWEEP_RUNS} specs, got {len(run_specs)}")
    run_names = [spec.run_name for spec in run_specs]
    if len(set(run_names)) != len(run_names):
        raise ValueError(f"Duplicate run names in variable-subset specs: {run_names}")
    expected_train_steps = get_num_train_steps(EXPERIMENT_BUDGET, BATCH_SIZE, SEQ_LEN)
    for offset, spec in enumerate(run_specs):
        expected_seed = SEED_SWEEP_START + offset
        if spec.source_run_name != SOURCE_RUN_NAME:
            raise ValueError(f"Unexpected source run {spec.source_run_name!r}")
        if spec.trainer_seed != expected_seed:
            raise ValueError(f"Unexpected trainer seed for {spec.run_name}: {spec.trainer_seed}")
        if spec.data_seed is not None:
            raise ValueError(f"Variable-subset run {spec.run_name} must leave data_seed unset")
        if spec.simulated_epoch_subset_seed is not None:
            raise ValueError(f"Variable-subset run {spec.run_name} must not set simulated_epoch_subset_seed")
        if spec.num_train_steps != expected_train_steps:
            raise ValueError(f"Unexpected train steps for {spec.run_name}: {spec.num_train_steps}")
        if spec.phase_weights != RUN_00097_PHASE_WEIGHTS:
            raise ValueError(f"Unexpected phase weights for {spec.run_name}")
        for phase_name, phase_weights in spec.phase_weights.items():
            total = sum(float(weight) for weight in phase_weights.values())
            if abs(total - 1.0) > 1e-6:
                raise ValueError(f"{spec.run_name} {phase_name} weights sum to {total}, not 1.0")


def save_run_manifest(config: SaveRunManifestConfig) -> None:
    """Persist the study manifest for downstream collection and analysis."""
    run_specs = json.loads(config.run_specs_json)
    payload = {
        "experiment_name": config.experiment_name,
        "objective_metric": config.objective_metric,
        "n_runs": len(run_specs),
        "noise_subset_mode": "variable",
        "runs": run_specs,
    }
    fs, _, _ = fsspec.get_fs_token_paths(config.output_path)
    fs.makedirs(config.output_path, exist_ok=True)
    with fsspec.open(os.path.join(config.output_path, RUN_MANIFEST_FILE), "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def create_run_manifest_step(*, name_prefix: str, run_specs: list[VariableSubsetRunSpec]) -> ExecutorStep:
    """Create the manifest writer step for this study."""
    return ExecutorStep(
        name=f"{name_prefix}/run_manifest",
        description=f"Save run_00097 300M / 6B variable-subset manifest ({len(run_specs)} runs)",
        fn=save_run_manifest,
        config=SaveRunManifestConfig(
            output_path=this_output_path(),
            experiment_name=name_prefix,
            objective_metric=OBJECTIVE_METRIC,
            run_specs_json=json.dumps([asdict(spec) for spec in run_specs], sort_keys=True),
        ),
    )


def _set_child_job_name(training_step: ExecutorStep, *, child_job_name: str) -> ExecutorStep:
    config = training_step.config
    if not isinstance(config, TrainLmOnPodConfig):
        raise TypeError(f"Expected TrainLmOnPodConfig for {training_step.name!r}, got {type(config)!r}")
    if config.job_name == "train_lm" and child_job_name == "train_lm":
        raise ValueError(f"{training_step.name} would use non-unique child job name {child_job_name!r}")
    return replace(training_step, config=replace(config, job_name=child_job_name))


def _build_fixed_subset_baseline_manifest_json() -> str:
    run_specs = [asdict(spec) for spec in build_fixed_subset_run_specs() if spec.cohort == "seed_sweep"]
    payload = {
        "experiment_name": FIXED_SUBSET_BASELINE_NAME,
        "objective_metric": OBJECTIVE_METRIC,
        "n_runs": len(run_specs),
        "runs": run_specs,
    }
    return json.dumps(payload, sort_keys=True)


def _build_compute_baseline_manifest_json() -> str:
    run_specs = [asdict(spec) for spec in build_compute_scaling_run_specs() if spec.ladder == REGMIX60M_6B_LADDER]
    payload = {
        "experiment_name": COMPUTE_SCALING_BASELINE_NAME,
        "objective_metric": OBJECTIVE_METRIC,
        "n_runs": len(run_specs),
        "runs": run_specs,
    }
    return json.dumps(payload, sort_keys=True)


def _parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description="Launch the run_00097 300M / 6B variable-subset repeat study.")
    parser.add_argument("--name-prefix", default=NAME)
    parser.add_argument("--source-run-name", default=SOURCE_RUN_NAME)
    parser.add_argument("--tpu-type", default="v5p-8")
    parser.add_argument("--tpu-region", default=DEFAULT_TPU_REGION)
    parser.add_argument("--tpu-zone", default=DEFAULT_TPU_ZONE)
    parser.add_argument(
        "--include-eval-harness",
        action="store_true",
        help="Run the default lm-eval harness during training. By default this launcher is perplexity-only.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_known_args()


def main() -> None:
    args, remaining = _parse_args()
    sys.argv = [sys.argv[0], *remaining]

    if args.source_run_name != SOURCE_RUN_NAME:
        raise ValueError(f"Only {SOURCE_RUN_NAME!r} is currently supported, got {args.source_run_name!r}")

    run_specs = build_run_specs()
    validate_run_specs(run_specs)
    if args.dry_run:
        print(json.dumps([asdict(spec) for spec in run_specs], indent=2, sort_keys=True))
        return

    if os.getenv("CI") is not None:
        logger.info("Skipping run_00097 300M / 6B variable-subset study launch in CI environment")
        return

    experiment = create_two_phase_dolma3_dolmino_top_level_experiment(
        name=args.name_prefix,
        experiment_budget=EXPERIMENT_BUDGET,
        model_config=regmix_300m_proxy,
        optimizer_config=regmix_300m_muonh_base,
        resources=ResourceConfig.with_tpu(args.tpu_type, regions=[args.tpu_region], zone=args.tpu_zone),
        eval_harness_tasks=None if args.include_eval_harness else (),
    )
    run_manifest_step = create_run_manifest_step(name_prefix=args.name_prefix, run_specs=run_specs)

    training_steps: list[ExecutorStep] = []
    for spec in run_specs:
        if spec.simulated_epoch_subset_seed is not None:
            raise ValueError(f"Unexpected simulated_epoch_subset_seed for {spec.run_name}")
        training_step = experiment.create_training_step(
            weight_config=WeightConfig(run_id=spec.run_id, phase_weights=spec.phase_weights),
            name_prefix=args.name_prefix,
            run_name=spec.run_name,
            trainer_seed=spec.trainer_seed,
            data_seed=spec.data_seed,
        )
        training_steps.append(_set_child_job_name(training_step, child_job_name=f"train_lm_{spec.run_name}"))

    all_metrics = tuple(dict.fromkeys((*PRIMARY_METRICS, *SECONDARY_METRICS)))
    results_step = create_manifest_results_step(
        name_prefix=args.name_prefix,
        run_manifest_step=run_manifest_step,
        wandb_entity=WANDB_ENTITY,
        wandb_project=WANDB_PROJECT,
        extra_metrics=all_metrics,
        depends_on=training_steps,
    )
    report_step = create_model_size_noise_report_step(
        name_prefix=args.name_prefix,
        run_manifest_step=run_manifest_step,
        analysis_step=results_step,
        wandb_entity=WANDB_ENTITY,
        wandb_project=WANDB_PROJECT,
        fixed_subset_baseline_manifest_json=_build_fixed_subset_baseline_manifest_json(),
        compute_baseline_manifest_json=_build_compute_baseline_manifest_json(),
        primary_metrics=PRIMARY_METRICS,
        secondary_metrics=SECONDARY_METRICS,
    )

    logger.info(
        "Launching %d run_00097 300M / 6B variable-subset repeats on %s in %s/%s.",
        len(run_specs),
        args.tpu_type,
        args.tpu_region,
        args.tpu_zone,
    )
    all_steps = [run_manifest_step, *training_steps, results_step, report_step]
    executor_main(
        ExecutorMainConfig(max_concurrent=len(training_steps)),
        steps=all_steps,
        description="run_00097 300M / 6B variable-subset noise study",
    )


if __name__ == "__main__":
    main()
