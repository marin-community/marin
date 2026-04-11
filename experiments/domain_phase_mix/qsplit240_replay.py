# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared launcher helpers for qsplit240 replay studies."""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, replace
from typing import cast

import fsspec
from fray.cluster import ResourceConfig
from levanter.main.train_lm import LmConfig
from levanter.optim import MuonHConfig
from marin.evaluation.eval_dataset_cache import create_cache_eval_datasets_step
from marin.evaluation.evaluation_config import EvalTaskConfig
from marin.execution.executor import Executor, ExecutorStep, output_path_of, this_output_path
from marin.training.training import TrainLmOnPodConfig
from rigging.filesystem import marin_prefix

from experiments.domain_phase_mix.config import WeightConfig
from experiments.domain_phase_mix.determinism_analysis import (
    FIT_DATASET_CSV,
    FIT_DATASET_SUMMARY_JSON,
    RESULTS_CSV,
    RUN_MANIFEST_FILE,
    create_fit_dataset_export_step,
    create_manifest_results_step,
)
from experiments.domain_phase_mix.two_phase_dolma3_dolmino_top_level import (
    create_two_phase_dolma3_dolmino_top_level_experiment,
)
from experiments.domain_phase_mix.two_phase_many_observed_runs import (
    REPRESENTATIVE12_PANEL_RUN_NAMES,
    load_original_qsplit240_named_panel,
    load_original_qsplit240_with_core_baselines,
)

DEFAULT_QSPLIT240_EVAL_DATASETS_CACHE_PATH = "gs://marin-us-central1/raw/eval-datasets/qsplit240-300m-6b-expanded-tasks"
EVAL_DATASETS_CACHE_DEP_ENV_VAR = "MARIN_EVAL_DATASETS_CACHE_DEPENDENCY"
ALL_PANEL = "all"
REPRESENTATIVE12_PANEL = "representative12"


@dataclass(frozen=True)
class Qsplit240ReplayRunSpec:
    """Manifest entry for one replayed qsplit240 run."""

    run_id: int
    run_name: str
    cohort: str
    model_family: str
    trainer_seed: int | None
    data_seed: int
    simulated_epoch_subset_seed: int | None
    candidate_run_id: int
    candidate_run_name: str
    candidate_source_experiment: str
    experiment_budget: int
    num_train_steps: int
    phase_weights: dict[str, dict[str, float]]


@dataclass(frozen=True)
class SaveRunManifestConfig:
    """Config for writing a qsplit240 replay manifest."""

    output_path: str
    experiment_name: str
    model_family: str
    experiment_budget: int
    num_train_steps: int
    eval_task_aliases: tuple[str, ...]
    run_specs_json: str


@dataclass(frozen=True)
class Qsplit240ReplayLaunchArtifacts:
    """Resolved steps for one qsplit240 replay launch."""

    run_specs: list[Qsplit240ReplayRunSpec]
    execution_name_prefix: str
    run_manifest_step: ExecutorStep
    cache_eval_datasets_step: ExecutorStep
    training_steps: list[ExecutorStep]
    results_step: ExecutorStep
    fit_dataset_step: ExecutorStep

    @property
    def steps(self) -> list[object]:
        return [
            self.run_manifest_step,
            self.cache_eval_datasets_step,
            *self.training_steps,
            self.results_step,
            self.fit_dataset_step,
        ]


def _region_local_marin_path(default_path: str) -> str:
    """Map a Marin GCS path to the current region bucket when possible."""
    current_prefix = marin_prefix().rstrip("/")
    if not default_path.startswith("gs://marin-") or not current_prefix.startswith("gs://marin-"):
        return default_path

    without_scheme = default_path[len("gs://") :]
    _, sep, object_key = without_scheme.partition("/")
    if not sep:
        return default_path
    return f"{current_prefix}/{object_key}"


def resolve_qsplit240_eval_cache_path(eval_datasets_cache_path: str | None = None) -> str:
    """Resolve the eval-dataset cache path for qsplit240 replay launches."""
    return eval_datasets_cache_path or DEFAULT_QSPLIT240_EVAL_DATASETS_CACHE_PATH


def resolve_qsplit240_eval_cache_path_for_current_region(eval_datasets_cache_path: str | None = None) -> str:
    """Resolve the eval cache path and rewrite it to the current Marin bucket prefix."""
    return _region_local_marin_path(resolve_qsplit240_eval_cache_path(eval_datasets_cache_path))


def load_panel_observed_runs(panel: str):
    """Load the selected observed-run panel in a stable order."""
    if panel == ALL_PANEL:
        return load_original_qsplit240_with_core_baselines()
    if panel == REPRESENTATIVE12_PANEL:
        return load_original_qsplit240_named_panel(REPRESENTATIVE12_PANEL_RUN_NAMES)
    raise ValueError(f"Unsupported qsplit240 panel {panel!r}")


def build_qsplit240_replay_run_specs(
    *,
    cohort: str,
    model_family: str,
    experiment_budget: int,
    num_train_steps: int,
    panel: str = ALL_PANEL,
) -> list[Qsplit240ReplayRunSpec]:
    """Build replay specs for the selected qsplit240 panel."""
    observed_runs = load_panel_observed_runs(panel)
    return [
        Qsplit240ReplayRunSpec(
            run_id=observed.run_id,
            run_name=observed.run_name,
            cohort=cohort,
            model_family=model_family,
            trainer_seed=None,
            data_seed=observed.run_id,
            simulated_epoch_subset_seed=None,
            candidate_run_id=observed.run_id,
            candidate_run_name=observed.run_name,
            candidate_source_experiment=observed.source_experiment,
            experiment_budget=experiment_budget,
            num_train_steps=num_train_steps,
            phase_weights=observed.phase_weights,
        )
        for observed in observed_runs
    ]


def shard_label(*, shard_index: int, shard_count: int) -> str:
    """Return a stable user-facing shard label."""
    return f"shard_{shard_index + 1:02d}of{shard_count:02d}"


def select_run_specs_for_shard(
    run_specs: list[Qsplit240ReplayRunSpec], *, shard_index: int, shard_count: int
) -> list[Qsplit240ReplayRunSpec]:
    """Select one contiguous shard from a replay manifest."""
    if shard_count < 1:
        raise ValueError(f"shard_count must be >= 1, got {shard_count}")
    if not 0 <= shard_index < shard_count:
        raise ValueError(f"shard_index must be in [0, {shard_count}), got {shard_index}")

    shard_size = math.ceil(len(run_specs) / shard_count)
    start = shard_index * shard_size
    end = min(start + shard_size, len(run_specs))
    return run_specs[start:end]


def shard_execution_name_prefix(*, name_prefix: str, shard_index: int, shard_count: int) -> str:
    """Return the executor-side name prefix for shard-local bookkeeping steps."""
    if shard_count == 1:
        return name_prefix
    return f"{name_prefix}/{shard_label(shard_index=shard_index, shard_count=shard_count)}"


def save_run_manifest(config: SaveRunManifestConfig) -> None:
    """Persist the replay manifest for downstream collection and analysis."""
    run_specs = json.loads(config.run_specs_json)
    payload = {
        "experiment_name": config.experiment_name,
        "model_family": config.model_family,
        "experiment_budget": config.experiment_budget,
        "num_train_steps": config.num_train_steps,
        "eval_tasks": list(config.eval_task_aliases),
        "n_runs": len(run_specs),
        "runs": run_specs,
    }
    fs, _, _ = fsspec.get_fs_token_paths(config.output_path)
    fs.makedirs(config.output_path, exist_ok=True)
    with fsspec.open(os.path.join(config.output_path, RUN_MANIFEST_FILE), "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def create_run_manifest_step(
    *,
    step_name_prefix: str,
    experiment_name: str,
    model_family: str,
    experiment_budget: int,
    num_train_steps: int,
    eval_tasks: tuple[EvalTaskConfig, ...],
    run_specs: list[Qsplit240ReplayRunSpec],
) -> ExecutorStep:
    """Create the manifest writer step for a qsplit240 replay launch."""
    return ExecutorStep(
        name=f"{step_name_prefix}/run_manifest",
        description=f"Save qsplit240 replay manifest ({len(run_specs)} runs)",
        fn=save_run_manifest,
        config=SaveRunManifestConfig(
            output_path=this_output_path(),
            experiment_name=experiment_name,
            model_family=model_family,
            experiment_budget=experiment_budget,
            num_train_steps=num_train_steps,
            eval_task_aliases=tuple(task.task_alias for task in eval_tasks),
            run_specs_json=json.dumps([run_spec.__dict__ for run_spec in run_specs], sort_keys=True),
        ),
    )


def create_qsplit240_replay_experiment(
    *,
    name: str,
    experiment_budget: int,
    batch_size: int,
    seq_len: int,
    model_config: LmConfig,
    optimizer_config: MuonHConfig,
    tpu_type: str,
    tpu_region: str,
    tpu_zone: str,
    eval_tasks: tuple[EvalTaskConfig, ...],
    eval_datasets_cache_path: str | None = None,
) -> object:
    """Create a qsplit240 replay experiment for a specific model size and region."""
    resolved_eval_cache_path = resolve_qsplit240_eval_cache_path(eval_datasets_cache_path)
    return create_two_phase_dolma3_dolmino_top_level_experiment(
        name=name,
        experiment_budget=experiment_budget,
        batch_size=batch_size,
        seq_len=seq_len,
        model_config=model_config,
        optimizer_config=optimizer_config,
        resources=ResourceConfig.with_tpu(tpu_type, regions=[tpu_region], zone=tpu_zone),
        eval_harness_tasks=eval_tasks,
        eval_datasets_cache_path=resolved_eval_cache_path,
        runtime_cache_region=tpu_region,
    )


def add_eval_cache_dependency_to_training_step(training_step: ExecutorStep, cache_step: ExecutorStep) -> ExecutorStep:
    """Make a training step block on the eval-dataset cache step."""
    config = training_step.config
    if not isinstance(config, TrainLmOnPodConfig):
        raise TypeError(
            f"Expected TrainLmOnPodConfig for qsplit240 training step {training_step.name!r}, got {type(config)!r}"
        )

    original_output_path = training_step.override_output_path
    if original_output_path is None:
        prefix = marin_prefix()
        executor = Executor(prefix=prefix, executor_info_base_path=os.path.join(prefix, "experiments"))
        executor.compute_version(training_step, is_pseudo_dep=False)
        original_output_path = executor.output_paths[training_step]

    env_vars = dict(config.env_vars or {})
    env_vars[EVAL_DATASETS_CACHE_DEP_ENV_VAR] = output_path_of(cache_step, ".eval_datasets_manifest.json")
    return replace(
        training_step,
        config=replace(config, env_vars=cast(dict[str, str], env_vars)),
        override_output_path=original_output_path,
    )


def build_qsplit240_replay_launch_artifacts(
    *,
    name_prefix: str,
    cohort: str,
    model_family: str,
    experiment_budget: int,
    num_train_steps: int,
    batch_size: int,
    seq_len: int,
    model_config: LmConfig,
    optimizer_config: MuonHConfig,
    tpu_type: str,
    tpu_region: str,
    tpu_zone: str,
    eval_tasks: tuple[EvalTaskConfig, ...],
    panel: str,
    shard_count: int,
    shard_index: int,
    wandb_entity: str,
    wandb_project: str,
    eval_datasets_cache_path: str | None = None,
) -> Qsplit240ReplayLaunchArtifacts:
    """Resolve the full step graph for a qsplit240 replay launch."""
    experiment = create_qsplit240_replay_experiment(
        name=name_prefix,
        experiment_budget=experiment_budget,
        batch_size=batch_size,
        seq_len=seq_len,
        model_config=model_config,
        optimizer_config=optimizer_config,
        tpu_type=tpu_type,
        tpu_region=tpu_region,
        tpu_zone=tpu_zone,
        eval_tasks=eval_tasks,
        eval_datasets_cache_path=eval_datasets_cache_path,
    )
    run_specs = select_run_specs_for_shard(
        build_qsplit240_replay_run_specs(
            cohort=cohort,
            model_family=model_family,
            experiment_budget=experiment_budget,
            num_train_steps=num_train_steps,
            panel=panel,
        ),
        shard_index=shard_index,
        shard_count=shard_count,
    )
    execution_name_prefix = shard_execution_name_prefix(
        name_prefix=name_prefix,
        shard_index=shard_index,
        shard_count=shard_count,
    )
    resolved_eval_cache_path = resolve_qsplit240_eval_cache_path(eval_datasets_cache_path)
    run_manifest_step = create_run_manifest_step(
        step_name_prefix=execution_name_prefix,
        experiment_name=name_prefix,
        model_family=model_family,
        experiment_budget=experiment_budget,
        num_train_steps=num_train_steps,
        eval_tasks=eval_tasks,
        run_specs=run_specs,
    )
    cache_eval_datasets_step = create_cache_eval_datasets_step(
        eval_tasks=eval_tasks,
        gcs_path=resolved_eval_cache_path,
        name_prefix=execution_name_prefix,
    )

    training_steps: list[ExecutorStep] = []
    for spec in run_specs:
        training_step = experiment.create_training_step(
            weight_config=WeightConfig(run_id=spec.run_id, phase_weights=spec.phase_weights),
            name_prefix=name_prefix,
            run_name=spec.run_name,
            data_seed=spec.data_seed,
        )
        training_steps.append(add_eval_cache_dependency_to_training_step(training_step, cache_eval_datasets_step))

    results_step = create_manifest_results_step(
        name_prefix=execution_name_prefix,
        run_manifest_step=run_manifest_step,
        wandb_entity=wandb_entity,
        wandb_project=wandb_project,
        depends_on=training_steps,
    )
    fit_dataset_step = create_fit_dataset_export_step(
        name_prefix=execution_name_prefix,
        run_manifest_step=run_manifest_step,
        analysis_step=results_step,
    )
    return Qsplit240ReplayLaunchArtifacts(
        run_specs=run_specs,
        execution_name_prefix=execution_name_prefix,
        run_manifest_step=run_manifest_step,
        cache_eval_datasets_step=cache_eval_datasets_step,
        training_steps=training_steps,
        results_step=results_step,
        fit_dataset_step=fit_dataset_step,
    )


def replay_description(*, execution_name_prefix: str, label: str) -> str:
    """Return the executor description for one replay launch."""
    return (
        f"{execution_name_prefix}: {label}. Outputs will include "
        f"{RESULTS_CSV}, {FIT_DATASET_CSV}, and {FIT_DATASET_SUMMARY_JSON}."
    )
