# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared launcher helpers for qsplit240 replay studies."""

from __future__ import annotations

import json
import math
import os
import re
from collections.abc import Sequence
from dataclasses import dataclass, replace
from functools import cache
from typing import cast

import fsspec
from fray.cluster import ResourceConfig
from levanter.main.train_lm import LmConfig
from levanter.optim import MuonHConfig
import numpy as np
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
    mirror_marin_path,
)
from experiments.domain_phase_mix.two_phase_many_olmix_loglinear_uncheatable import (
    RUN_ID as OLMIX_UNCHEATABLE_RUN_ID,
    RUN_NAME as OLMIX_UNCHEATABLE_RUN_NAME,
    SOURCE_EXPERIMENT as OLMIX_UNCHEATABLE_SOURCE_EXPERIMENT,
    load_fit_from_local_results,
)
from experiments.domain_phase_mix.two_phase_many_observed_runs import (
    CORE_BASELINE_RUN_NAMES,
    ObservedTwoPhaseManyRun,
    REPRESENTATIVE12_PANEL_RUN_NAMES,
    load_original_qsplit240_named_panel,
    load_original_qsplit240_with_core_baselines,
)

DEFAULT_QSPLIT240_EVAL_DATASETS_CACHE_PATH = "gs://marin-us-central1/raw/eval-datasets/qsplit240-300m-6b-expanded-tasks"
DEFAULT_REGION_AGNOSTIC_TPU_REGIONS = ("us-east5", "us-central1")
EVAL_DATASETS_CACHE_DEP_ENV_VAR = "MARIN_EVAL_DATASETS_CACHE_DEPENDENCY"
ALL_PANEL = "all"
REPRESENTATIVE12_PANEL = "representative12"
BASELINES3_PANEL = "baselines3"
CHECKPOINT_STEP_METADATA_GLOB = "checkpoints/step-*/metadata.json"
CHECKPOINT_STEP_PATTERN = re.compile(r"/checkpoints/step-(\d+)/")


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


def _region_local_marin_path(default_path: str, region: str | None = None) -> str:
    """Map a Marin GCS path to a specific region bucket when possible."""
    if region is None:
        current_prefix = marin_prefix().rstrip("/")
    else:
        current_prefix = f"gs://marin-{region}"
    if not default_path.startswith("gs://marin-") or not current_prefix.startswith("gs://marin-"):
        return default_path

    without_scheme = default_path[len("gs://") :]
    _, sep, object_key = without_scheme.partition("/")
    if not sep:
        return default_path
    return f"{current_prefix}/{object_key}"


def mirror_path(path: str) -> str:
    """Return a mirror-backed filesystem path for a GCS path."""
    return mirror_marin_path(path)


def _checkpoint_root_from_metadata_path(metadata_path: str) -> tuple[str, int] | None:
    match = CHECKPOINT_STEP_PATTERN.search(metadata_path)
    if match is None:
        return None
    step = int(match.group(1))
    checkpoint_root = metadata_path.split("/checkpoints/step-", 1)[0]
    return checkpoint_root, step


def normalize_tpu_regions(tpu_regions: str | Sequence[str]) -> tuple[str, ...]:
    """Normalize a region spec into a stable tuple without duplicates."""
    raw_regions: Sequence[str]
    if isinstance(tpu_regions, str):
        raw_regions = tpu_regions.split(",")
    else:
        raw_regions = tpu_regions
    normalized = tuple(dict.fromkeys(region.strip() for region in raw_regions if region.strip()))
    if not normalized:
        raise ValueError("tpu_regions must contain at least one region")
    return normalized


def resolve_qsplit240_eval_cache_path(eval_datasets_cache_path: str | None = None) -> str:
    """Resolve the eval-dataset cache path for qsplit240 replay launches."""
    return eval_datasets_cache_path or DEFAULT_QSPLIT240_EVAL_DATASETS_CACHE_PATH


def resolve_qsplit240_eval_cache_path_for_current_region(eval_datasets_cache_path: str | None = None) -> str:
    """Resolve the eval cache path and rewrite it to the current Marin bucket prefix."""
    return _region_local_marin_path(resolve_qsplit240_eval_cache_path(eval_datasets_cache_path))


def resolve_qsplit240_eval_cache_path_for_regions(
    tpu_regions: str | Sequence[str],
    eval_datasets_cache_path: str | None = None,
) -> str:
    """Resolve the eval cache path for a single-region or region-agnostic launch."""
    resolved_path = resolve_qsplit240_eval_cache_path(eval_datasets_cache_path)
    normalized_regions = normalize_tpu_regions(tpu_regions)
    if len(normalized_regions) == 1:
        return _region_local_marin_path(resolved_path, normalized_regions[0])
    return mirror_path(resolved_path)


def resolve_latest_checkpoint_root(
    *,
    experiment_name_prefix: str,
    run_name: str,
    checkpoint_regions: str | Sequence[str] = DEFAULT_REGION_AGNOSTIC_TPU_REGIONS,
) -> str | None:
    """Resolve the highest-step checkpoint root for a run across one or more regions."""
    best_checkpoint: tuple[int, str] | None = None
    for region in normalize_tpu_regions(checkpoint_regions):
        pattern = (
            f"gs://marin-{region}/checkpoints/{experiment_name_prefix}/{run_name}-*/{CHECKPOINT_STEP_METADATA_GLOB}"
        )
        fs, _, _ = fsspec.get_fs_token_paths(pattern)
        for match in fs.glob(pattern):
            full_match = match if str(match).startswith("gs://") else f"gs://{match}"
            resolved = _checkpoint_root_from_metadata_path(full_match)
            if resolved is None:
                continue
            checkpoint_root, step = resolved
            if best_checkpoint is None or step > best_checkpoint[0]:
                best_checkpoint = (step, checkpoint_root)

    if best_checkpoint is None:
        return None
    return best_checkpoint[1]


@cache
def _top_level_natural_proportions_and_phase_fractions() -> tuple[np.ndarray, np.ndarray]:
    """Return the deterministic top-level natural proportions and phase fractions."""
    experiment = create_two_phase_dolma3_dolmino_top_level_experiment(name="qsplit240_replay")
    natural_proportions = np.asarray([float(domain.total_weight) for domain in experiment.domains], dtype=float)
    natural_proportions = natural_proportions / natural_proportions.sum()
    phase_fractions = np.asarray(
        [phase.end_fraction - phase.start_fraction for phase in experiment.phase_schedule.phases],
        dtype=float,
    )
    return natural_proportions, phase_fractions


@cache
def _olmix_uncheatable_panel_run() -> ObservedTwoPhaseManyRun:
    """Return the fitted 60M Olmix-uncheatable mixture as a synthetic observed-panel run."""
    natural_proportions, phase_fractions = _top_level_natural_proportions_and_phase_fractions()
    fit = load_fit_from_local_results(
        natural_proportions=natural_proportions,
        phase_fractions=phase_fractions,
    )
    return ObservedTwoPhaseManyRun(
        source_experiment=OLMIX_UNCHEATABLE_SOURCE_EXPERIMENT,
        run_id=OLMIX_UNCHEATABLE_RUN_ID,
        run_name=OLMIX_UNCHEATABLE_RUN_NAME,
        status="completed",
        phase_weights=fit.phase_weights,
    )


def load_panel_observed_runs(panel: str):
    """Load the selected observed-run panel in a stable order."""
    if panel == ALL_PANEL:
        return load_original_qsplit240_with_core_baselines()
    if panel == REPRESENTATIVE12_PANEL:
        return load_original_qsplit240_named_panel(REPRESENTATIVE12_PANEL_RUN_NAMES)
    if panel == BASELINES3_PANEL:
        baseline_runs = [
            run for run in load_original_qsplit240_with_core_baselines() if run.run_name in CORE_BASELINE_RUN_NAMES
        ]
        baseline_runs.sort(key=lambda run: CORE_BASELINE_RUN_NAMES.index(run.run_name))
        return [*baseline_runs, _olmix_uncheatable_panel_run()]
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
    tpu_regions: Sequence[str],
    tpu_zone: str | None,
    eval_tasks: tuple[EvalTaskConfig, ...],
    eval_datasets_cache_path: str | None = None,
) -> object:
    """Create a qsplit240 replay experiment for a specific model size and region."""
    normalized_regions = normalize_tpu_regions(tpu_regions)
    resolved_eval_cache_path = resolve_qsplit240_eval_cache_path_for_regions(
        normalized_regions,
        eval_datasets_cache_path,
    )
    runtime_cache_region: str | tuple[str, ...]
    if len(normalized_regions) == 1:
        runtime_cache_region = normalized_regions[0]
    else:
        runtime_cache_region = normalized_regions
    return create_two_phase_dolma3_dolmino_top_level_experiment(
        name=name,
        experiment_budget=experiment_budget,
        batch_size=batch_size,
        seq_len=seq_len,
        model_config=model_config,
        optimizer_config=optimizer_config,
        resources=ResourceConfig.with_tpu(tpu_type, regions=list(normalized_regions), zone=tpu_zone),
        eval_harness_tasks=eval_tasks,
        eval_datasets_cache_path=resolved_eval_cache_path,
        runtime_cache_region=runtime_cache_region,
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
    tpu_regions: Sequence[str],
    tpu_zone: str | None,
    eval_tasks: tuple[EvalTaskConfig, ...],
    panel: str,
    shard_count: int,
    shard_index: int,
    wandb_entity: str,
    wandb_project: str,
    eval_datasets_cache_path: str | None = None,
    resume_latest_checkpoints: bool = False,
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
        tpu_regions=tpu_regions,
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
    resolved_eval_cache_path = resolve_qsplit240_eval_cache_path_for_regions(tpu_regions, eval_datasets_cache_path)
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
        train_kwargs: dict[str, object] = {}
        if resume_latest_checkpoints:
            latest_checkpoint_root = resolve_latest_checkpoint_root(
                experiment_name_prefix=name_prefix,
                run_name=spec.run_name,
                checkpoint_regions=tpu_regions,
            )
            if latest_checkpoint_root is not None:
                train_kwargs["initialize_from_checkpoint_path"] = mirror_path(latest_checkpoint_root)
                train_kwargs["reset_data_loader_on_init"] = False
        training_step = experiment.create_training_step(
            weight_config=WeightConfig(run_id=spec.run_id, phase_weights=spec.phase_weights),
            name_prefix=name_prefix,
            run_name=spec.run_name,
            data_seed=spec.data_seed,
            **train_kwargs,
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
