# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Launch proportional variable-subset noise baselines at 60M/1.2B and 100M/6B."""

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
from experiments.domain_phase_mix.determinism_analysis import RUN_MANIFEST_FILE, create_manifest_results_step
from experiments.domain_phase_mix.launch_two_phase_many_run_00097_fixed_subset_study import (
    PRIMARY_METRICS,
    SECONDARY_METRICS,
    WANDB_ENTITY,
    WANDB_PROJECT,
)
from experiments.domain_phase_mix.qsplit240_replay import skip_eval_harness_for_training_step
from experiments.domain_phase_mix.scaling_study_recipes import ScalingStudyScale, resolve_scale_spec
from experiments.domain_phase_mix.two_phase_dolma3_dolmino_top_level import (
    PHASE_NAMES,
    create_initial_fixed_weight_configs,
    create_two_phase_dolma3_dolmino_top_level_experiment,
)

logger = logging.getLogger(__name__)

BASE_NAME = "pinlin_calvin_xu/data_mixture/ngd3dm2_proportional_variable_subset_noise"
ANCHOR_RUN_NAME = "baseline_proportional"
COHORT = "proportional_variable_subset_noise"
N_SEED_SWEEP_RUNS = 10
SEED_SWEEP_START = 10000
DEFAULT_SCALES = (ScalingStudyScale.REGMIX_60M_1P2B, ScalingStudyScale.REGMIX_300M_6B)
DEFAULT_TARGET_BUDGET_MULTIPLIER = 1.0
DEFAULT_TPU_TYPE = "v5p-8"
DEFAULT_TPU_REGION = "us-east5"
DEFAULT_TPU_ZONE = "us-east5-a"


@dataclass(frozen=True)
class ProportionalVariableSubsetRunSpec:
    """Manifest entry for one proportional variable-subset noise run."""

    run_id: int
    run_name: str
    family: str
    scale: str
    cohort: Literal["proportional_variable_subset_noise"]
    model_family: str
    trainer_seed: int
    data_seed: None
    simulated_epoch_subset_seed: None
    noise_anchor_run_name: str
    source_experiment: str
    experiment_budget: int
    target_budget: int
    target_budget_multiplier: float
    num_train_steps: int
    batch_size: int
    seq_len: int
    phase_weights: dict[str, dict[str, float]]


@dataclass(frozen=True)
class SaveRunManifestConfig:
    """Config for writing one scale's proportional variable-subset manifest."""

    output_path: str
    experiment_name: str
    scale: str
    run_specs_json: str


def source_experiment_for_scale(scale: ScalingStudyScale | str, *, base_name: str = BASE_NAME) -> str:
    """Return the checkpoint namespace for one proportional variable-subset scale."""
    scale_value = scale.value if isinstance(scale, ScalingStudyScale) else str(scale)
    return f"{base_name}_{scale_value}"


def family_for_scale(scale: ScalingStudyScale | str) -> str:
    """Return the run-registry family name for one proportional variable-subset scale."""
    scale_value = scale.value if isinstance(scale, ScalingStudyScale) else str(scale)
    return f"proportional_variable_subset_noise_{scale_value}"


def _anchor_phase_weights() -> dict[str, dict[str, float]]:
    baseline_configs = {config.run_name: config.weight_config for config in create_initial_fixed_weight_configs()}
    if ANCHOR_RUN_NAME not in baseline_configs:
        raise ValueError(f"Missing anchor run {ANCHOR_RUN_NAME!r}")
    phase_weights = baseline_configs[ANCHOR_RUN_NAME].phase_weights
    for phase_name in PHASE_NAMES:
        total = sum(float(weight) for weight in phase_weights[phase_name].values())
        if abs(total - 1.0) > 1e-12:
            raise ValueError(f"{ANCHOR_RUN_NAME} {phase_name} weights sum to {total}, not 1")
    for domain_name, phase0_weight in phase_weights["phase_0"].items():
        phase1_weight = phase_weights["phase_1"].get(domain_name)
        if phase1_weight is None or abs(float(phase0_weight) - float(phase1_weight)) > 1e-15:
            raise ValueError(f"{ANCHOR_RUN_NAME} is not phase-constant for {domain_name}")
    return {phase_name: dict(weights) for phase_name, weights in phase_weights.items()}


def _scale_from_value(scale_value: str) -> ScalingStudyScale:
    try:
        return ScalingStudyScale(scale_value)
    except ValueError as exc:
        valid = ", ".join(scale.value for scale in ScalingStudyScale)
        raise ValueError(f"Unknown scale {scale_value!r}; valid values: {valid}") from exc


def _parse_scales(value: str) -> tuple[ScalingStudyScale, ...]:
    scales = tuple(_scale_from_value(part.strip()) for part in value.split(",") if part.strip())
    if not scales:
        raise ValueError("--scales must contain at least one scale")
    return tuple(dict.fromkeys(scales))


def build_run_specs(
    *,
    scales: tuple[ScalingStudyScale, ...] = DEFAULT_SCALES,
    base_name: str = BASE_NAME,
    target_budget_multiplier: float = DEFAULT_TARGET_BUDGET_MULTIPLIER,
) -> list[ProportionalVariableSubsetRunSpec]:
    """Build proportional variable-subset noise specs for the requested scales."""
    phase_weights = _anchor_phase_weights()
    run_specs: list[ProportionalVariableSubsetRunSpec] = []
    for scale in scales:
        scale_spec = resolve_scale_spec(scale)
        experiment_budget = scale_spec.experiment_budget_for_multiplier(target_budget_multiplier)
        target_budget = scale_spec.target_budget_for_multiplier(target_budget_multiplier)
        num_train_steps = scale_spec.num_train_steps_for_multiplier(target_budget_multiplier)
        for offset in range(N_SEED_SWEEP_RUNS):
            trainer_seed = SEED_SWEEP_START + offset
            run_specs.append(
                ProportionalVariableSubsetRunSpec(
                    run_id=offset,
                    run_name=f"propvar_{scale.value}_trainer_seed_{trainer_seed}",
                    family=family_for_scale(scale),
                    scale=scale.value,
                    cohort=COHORT,
                    model_family=scale_spec.model_family,
                    trainer_seed=trainer_seed,
                    data_seed=None,
                    simulated_epoch_subset_seed=None,
                    noise_anchor_run_name=ANCHOR_RUN_NAME,
                    source_experiment=source_experiment_for_scale(scale, base_name=base_name),
                    experiment_budget=experiment_budget,
                    target_budget=target_budget,
                    target_budget_multiplier=target_budget_multiplier,
                    num_train_steps=num_train_steps,
                    batch_size=scale_spec.batch_size,
                    seq_len=scale_spec.seq_len,
                    phase_weights=phase_weights,
                )
            )
    return run_specs


def validate_run_specs(
    run_specs: list[ProportionalVariableSubsetRunSpec],
    *,
    expected_scales: tuple[ScalingStudyScale, ...],
) -> None:
    """Validate that the manifest matches the intended proportional variable-subset design."""
    expected_count = len(expected_scales) * N_SEED_SWEEP_RUNS
    if len(run_specs) != expected_count:
        raise ValueError(f"Expected {expected_count} specs, got {len(run_specs)}")
    run_names = [spec.run_name for spec in run_specs]
    if len(set(run_names)) != len(run_names):
        raise ValueError(f"Duplicate run names: {run_names}")
    source_keys = [(spec.source_experiment, spec.run_name) for spec in run_specs]
    if len(set(source_keys)) != len(source_keys):
        raise ValueError(f"Duplicate output keys: {source_keys}")
    expected_phase_weights = _anchor_phase_weights()
    expected_scale_values = {scale.value for scale in expected_scales}
    for spec in run_specs:
        if spec.scale not in expected_scale_values:
            raise ValueError(f"Unexpected scale in run spec: {spec.scale}")
        if spec.data_seed is not None:
            raise ValueError(f"{spec.run_name} must leave data_seed unset")
        if spec.simulated_epoch_subset_seed is not None:
            raise ValueError(f"{spec.run_name} must leave simulated_epoch_subset_seed unset")
        if spec.noise_anchor_run_name != ANCHOR_RUN_NAME:
            raise ValueError(f"{spec.run_name} has wrong noise anchor {spec.noise_anchor_run_name!r}")
        if spec.phase_weights != expected_phase_weights:
            raise ValueError(f"{spec.run_name} phase weights do not match {ANCHOR_RUN_NAME}")
        for phase_name, weights in spec.phase_weights.items():
            total = sum(float(weight) for weight in weights.values())
            if abs(total - 1.0) > 1e-12:
                raise ValueError(f"{spec.run_name} {phase_name} weights sum to {total}, not 1")


def save_run_manifest(config: SaveRunManifestConfig) -> None:
    """Persist one scale's run manifest."""
    run_specs = json.loads(config.run_specs_json)
    payload = {
        "experiment_name": config.experiment_name,
        "scale": config.scale,
        "cohort": COHORT,
        "noise_subset_mode": "variable",
        "noise_anchor_run_name": ANCHOR_RUN_NAME,
        "n_runs": len(run_specs),
        "runs": run_specs,
    }
    fs, _, _ = fsspec.get_fs_token_paths(config.output_path)
    fs.makedirs(config.output_path, exist_ok=True)
    with fsspec.open(os.path.join(config.output_path, RUN_MANIFEST_FILE), "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def create_run_manifest_step(
    *,
    source_experiment: str,
    scale: ScalingStudyScale,
    run_specs: list[ProportionalVariableSubsetRunSpec],
) -> ExecutorStep:
    """Create a per-scale manifest writer."""
    return ExecutorStep(
        name=f"{source_experiment}/run_manifest",
        description=f"Save proportional variable-subset manifest for {scale.value} ({len(run_specs)} runs)",
        fn=save_run_manifest,
        config=SaveRunManifestConfig(
            output_path=this_output_path(),
            experiment_name=source_experiment,
            scale=scale.value,
            run_specs_json=json.dumps([asdict(spec) for spec in run_specs], sort_keys=True),
        ),
    )


def _set_child_job_name(training_step: ExecutorStep, *, child_job_name: str) -> ExecutorStep:
    config = training_step.config
    if not isinstance(config, TrainLmOnPodConfig):
        raise TypeError(f"Expected TrainLmOnPodConfig for {training_step.name!r}, got {type(config)!r}")
    return replace(training_step, config=replace(config, job_name=child_job_name))


def _parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-name", default=BASE_NAME)
    parser.add_argument(
        "--scales",
        default=",".join(scale.value for scale in DEFAULT_SCALES),
        help="Comma-separated historical scale keys.",
    )
    parser.add_argument("--target-budget-multiplier", type=float, default=DEFAULT_TARGET_BUDGET_MULTIPLIER)
    parser.add_argument("--tpu-type", default=DEFAULT_TPU_TYPE)
    parser.add_argument("--tpu-region", default=DEFAULT_TPU_REGION)
    parser.add_argument("--tpu-zone", default=DEFAULT_TPU_ZONE)
    parser.add_argument("--max-concurrent", type=int, default=len(DEFAULT_SCALES) * N_SEED_SWEEP_RUNS)
    parser.add_argument("--include-eval-harness", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_known_args()


def main() -> None:
    args, remaining = _parse_args()
    sys.argv = [sys.argv[0], *remaining]
    scales = _parse_scales(args.scales)
    run_specs = build_run_specs(
        scales=scales,
        base_name=args.base_name,
        target_budget_multiplier=args.target_budget_multiplier,
    )
    validate_run_specs(run_specs, expected_scales=scales)
    if args.dry_run:
        print(json.dumps([asdict(spec) for spec in run_specs], indent=2, sort_keys=True))
        return
    if os.getenv("CI") is not None:
        logger.info("Skipping proportional variable-subset noise launch in CI environment")
        return

    all_steps: list[ExecutorStep] = []
    for scale in scales:
        scale_spec = resolve_scale_spec(scale)
        source_experiment = source_experiment_for_scale(scale, base_name=args.base_name)
        scale_run_specs = [spec for spec in run_specs if spec.scale == scale.value]
        experiment = create_two_phase_dolma3_dolmino_top_level_experiment(
            name=source_experiment,
            experiment_budget=scale_run_specs[0].experiment_budget,
            target_budget=scale_run_specs[0].target_budget,
            batch_size=scale_spec.batch_size,
            seq_len=scale_spec.seq_len,
            model_config=scale_spec.model_config,
            optimizer_config=scale_spec.optimizer_config,
            resources=ResourceConfig.with_tpu(args.tpu_type, regions=[args.tpu_region], zone=args.tpu_zone),
            eval_harness_tasks=() if not args.include_eval_harness else None,
            eval_datasets_cache_path=None,
            runtime_cache_region=args.tpu_region,
        )
        run_manifest_step = create_run_manifest_step(
            source_experiment=source_experiment,
            scale=scale,
            run_specs=scale_run_specs,
        )
        training_steps: list[ExecutorStep] = []
        for spec in scale_run_specs:
            if spec.simulated_epoch_subset_seed is not None:
                raise ValueError(f"Unexpected simulated_epoch_subset_seed for {spec.run_name}")
            training_step = experiment.create_training_step(
                weight_config=WeightConfig(run_id=spec.run_id, phase_weights=spec.phase_weights),
                name_prefix=source_experiment,
                run_name=spec.run_name,
                trainer_seed=spec.trainer_seed,
                data_seed=spec.data_seed,
            )
            if not args.include_eval_harness:
                training_step = skip_eval_harness_for_training_step(training_step)
            training_steps.append(_set_child_job_name(training_step, child_job_name=f"train_lm_{spec.run_name}"))
        results_step = create_manifest_results_step(
            name_prefix=source_experiment,
            run_manifest_step=run_manifest_step,
            wandb_entity=WANDB_ENTITY,
            wandb_project=WANDB_PROJECT,
            extra_metrics=tuple(dict.fromkeys((*PRIMARY_METRICS, *SECONDARY_METRICS))),
            depends_on=training_steps,
        )
        all_steps.extend([run_manifest_step, *training_steps, results_step])

    logger.info("Launching %d proportional variable-subset noise runs.", len(run_specs))
    executor_main(
        ExecutorMainConfig(max_concurrent=args.max_concurrent),
        steps=all_steps,
        description="proportional variable-subset noise baselines at 60M and 100M/6B",
    )


if __name__ == "__main__":
    main()
