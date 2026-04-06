# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Launch the original qsplit240 swarm at 300M / 6B with expanded task evals."""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
from dataclasses import asdict, dataclass, replace
from typing import Literal, cast

import fsspec
from fray.cluster import ResourceConfig
from rigging.filesystem import marin_prefix
from marin.evaluation.eval_dataset_cache import create_cache_eval_datasets_step
from marin.execution.executor import (
    Executor,
    ExecutorMainConfig,
    ExecutorStep,
    executor_main,
    output_path_of,
    this_output_path,
)
from marin.evaluation.evaluation_config import EvalTaskConfig
from marin.training.training import TrainLmOnPodConfig

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
    WANDB_ENTITY,
    WANDB_PROJECT,
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
from experiments.domain_phase_mix.two_phase_many_observed_runs import load_original_qsplit240_with_core_baselines
from experiments.evals.task_configs import MMLU_5_SHOT, MMLU_PRO_5_SHOT, MMLU_SL_VERB_5_SHOT

logger = logging.getLogger(__name__)

NAME = "pinlin_calvin_xu/data_mixture/ngd3dm2_qsplit240_300m_6b"
MODEL_FAMILY = "regmix_300m_proxy"
EXPERIMENT_BUDGET = 6_000_000_000
NUM_TRAIN_STEPS = get_num_train_steps(EXPERIMENT_BUDGET, BATCH_SIZE, SEQ_LEN)
DEFAULT_MAX_CONCURRENT = 256
EVAL_DATASETS_CACHE_PATH = "gs://marin-us-central1/raw/eval-datasets/qsplit240-300m-6b-expanded-tasks"
EVAL_DATASETS_CACHE_DEP_ENV_VAR = "MARIN_EVAL_DATASETS_CACHE_DEPENDENCY"
QSPLIT240_300M_EVAL_TASKS = (
    MMLU_5_SHOT,
    MMLU_SL_VERB_5_SHOT,
    MMLU_PRO_5_SHOT,
    EvalTaskConfig("arc_easy", 10),
    EvalTaskConfig("piqa", 10),
    EvalTaskConfig("sciq", 0, task_alias="sciq_0shot"),
    EvalTaskConfig("hellaswag", 0, task_alias="hellaswag_0shot"),
)


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


@dataclass(frozen=True)
class Qsplit240ModelSizeRunSpec:
    """Manifest entry for one replayed qsplit240 run at 300M / 6B."""

    run_id: int
    run_name: str
    cohort: Literal["original_swarm_300m"]
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
    """Config for writing the replay manifest."""

    output_path: str
    experiment_name: str
    run_specs_json: str


def build_run_specs() -> list[Qsplit240ModelSizeRunSpec]:
    """Build the replay manifest for the original qsplit240 swarm plus two baselines."""
    observed_runs = load_original_qsplit240_with_core_baselines()
    return [
        Qsplit240ModelSizeRunSpec(
            run_id=observed.run_id,
            run_name=observed.run_name,
            cohort="original_swarm_300m",
            model_family=MODEL_FAMILY,
            trainer_seed=None,
            data_seed=observed.run_id,
            simulated_epoch_subset_seed=None,
            candidate_run_id=observed.run_id,
            candidate_run_name=observed.run_name,
            candidate_source_experiment=observed.source_experiment,
            experiment_budget=EXPERIMENT_BUDGET,
            num_train_steps=NUM_TRAIN_STEPS,
            phase_weights=observed.phase_weights,
        )
        for observed in observed_runs
    ]


def shard_label(*, shard_index: int, shard_count: int) -> str:
    """Return a stable user-facing shard label."""
    return f"shard_{shard_index + 1:02d}of{shard_count:02d}"


def select_run_specs_for_shard(
    run_specs: list[Qsplit240ModelSizeRunSpec], *, shard_index: int, shard_count: int
) -> list[Qsplit240ModelSizeRunSpec]:
    """Select one contiguous shard from the full qsplit240 replay manifest."""
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
    """Persist the replay manifest for downstream collection and fitting."""
    run_specs = json.loads(config.run_specs_json)
    payload = {
        "experiment_name": config.experiment_name,
        "model_family": MODEL_FAMILY,
        "experiment_budget": EXPERIMENT_BUDGET,
        "num_train_steps": NUM_TRAIN_STEPS,
        "eval_tasks": [task.task_alias for task in QSPLIT240_300M_EVAL_TASKS],
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
    run_specs: list[Qsplit240ModelSizeRunSpec],
) -> ExecutorStep:
    """Create the manifest writer step for the replayed 300M swarm."""
    return ExecutorStep(
        name=f"{step_name_prefix}/run_manifest",
        description=f"Save qsplit240 300M / 6B swarm manifest ({len(run_specs)} runs)",
        fn=save_run_manifest,
        config=SaveRunManifestConfig(
            output_path=this_output_path(),
            experiment_name=experiment_name,
            run_specs_json=json.dumps([asdict(spec) for spec in run_specs], sort_keys=True),
        ),
    )


def create_experiment(*, name: str, tpu_type: str, eval_datasets_cache_path: str | None = None) -> object:
    """Create the 300M qsplit240 experiment with the expanded task suite."""
    eval_datasets_cache_path = eval_datasets_cache_path or _region_local_marin_path(EVAL_DATASETS_CACHE_PATH)
    return create_two_phase_dolma3_dolmino_top_level_experiment(
        name=name,
        experiment_budget=EXPERIMENT_BUDGET,
        model_config=regmix_300m_proxy,
        optimizer_config=regmix_300m_muonh_base,
        resources=ResourceConfig.with_tpu(tpu_type, regions=["us-east5"], zone="us-east5-a"),
        eval_harness_tasks=QSPLIT240_300M_EVAL_TASKS,
        eval_datasets_cache_path=eval_datasets_cache_path,
    )


def add_eval_cache_dependency_to_training_step(training_step: ExecutorStep, cache_step: ExecutorStep) -> ExecutorStep:
    """Make a training step block on the eval-dataset cache step.

    The qsplit240 launcher passes the cache GCS path as a plain string, which does
    not create an executor dependency edge by itself. Thread a harmless env-var
    marker through the training config so the executor treats the cache step as a
    real dependency.
    """
    config = training_step.config
    if not isinstance(config, TrainLmOnPodConfig):
        raise TypeError(
            f"Expected TrainLmOnPodConfig for qsplit240 training step {training_step.name!r}, got {type(config)!r}"
        )

    # Preserve the original training output path so relaunches resume from the
    # same checkpoint root even though we are adding a new executor dependency.
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


def _parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description="Launch the original qsplit240 swarm at 300M / 6B.")
    parser.add_argument("--name-prefix", default=NAME)
    parser.add_argument("--tpu-type", default="v5p-8")
    parser.add_argument("--max-concurrent", type=int, default=DEFAULT_MAX_CONCURRENT)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    return parser.parse_known_args()


def main() -> None:
    args, remaining = _parse_args()
    sys.argv = [sys.argv[0], *remaining]

    if os.getenv("CI") is not None:
        logger.info("Skipping qsplit240 300M / 6B swarm launch in CI environment")
        return

    eval_datasets_cache_path = _region_local_marin_path(EVAL_DATASETS_CACHE_PATH)
    experiment = create_experiment(
        name=args.name_prefix,
        tpu_type=args.tpu_type,
        eval_datasets_cache_path=eval_datasets_cache_path,
    )
    run_specs = select_run_specs_for_shard(
        build_run_specs(),
        shard_index=args.shard_index,
        shard_count=args.shard_count,
    )
    execution_name_prefix = shard_execution_name_prefix(
        name_prefix=args.name_prefix,
        shard_index=args.shard_index,
        shard_count=args.shard_count,
    )
    run_manifest_step = create_run_manifest_step(
        step_name_prefix=execution_name_prefix,
        experiment_name=args.name_prefix,
        run_specs=run_specs,
    )
    cache_eval_datasets_step = create_cache_eval_datasets_step(
        eval_tasks=QSPLIT240_300M_EVAL_TASKS,
        gcs_path=eval_datasets_cache_path,
        name_prefix=execution_name_prefix,
    )

    training_steps: list[ExecutorStep] = []
    for spec in run_specs:
        training_step = experiment.create_training_step(
            weight_config=WeightConfig(run_id=spec.run_id, phase_weights=spec.phase_weights),
            name_prefix=args.name_prefix,
            run_name=spec.run_name,
            data_seed=spec.data_seed,
        )
        training_steps.append(add_eval_cache_dependency_to_training_step(training_step, cache_eval_datasets_step))

    results_step = create_manifest_results_step(
        name_prefix=execution_name_prefix,
        run_manifest_step=run_manifest_step,
        wandb_entity=WANDB_ENTITY,
        wandb_project=WANDB_PROJECT,
        depends_on=training_steps,
    )
    fit_dataset_step = create_fit_dataset_export_step(
        name_prefix=execution_name_prefix,
        run_manifest_step=run_manifest_step,
        analysis_step=results_step,
    )

    logger.info(
        "Launching shard %d/%d with %d qsplit240 300M / 6B runs on %s with max_concurrent=%d. "
        "Training outputs stay under %s; shard bookkeeping outputs go under %s. Outputs will include %s, %s, and %s.",
        args.shard_index + 1,
        args.shard_count,
        len(run_specs),
        args.tpu_type,
        args.max_concurrent,
        args.name_prefix,
        execution_name_prefix,
        RESULTS_CSV,
        FIT_DATASET_CSV,
        FIT_DATASET_SUMMARY_JSON,
    )
    executor_main(
        ExecutorMainConfig(max_concurrent=args.max_concurrent),
        steps=[run_manifest_step, cache_eval_datasets_step, *training_steps, results_step, fit_dataset_step],
        description=f"{execution_name_prefix}: original qsplit240 swarm replay at 300M / 6B",
    )


if __name__ == "__main__":
    main()
