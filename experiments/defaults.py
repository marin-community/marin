# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""
This file represents the best practices for each stage of the pipeline.
"""

import dataclasses
import logging
import os
from collections.abc import Callable, Sequence
from functools import lru_cache
from typing import Any

import levanter.main.train_lm as levanter_train_lm
from fray import current_client
from fray.cluster import ResourceConfig
from fray.types import Entrypoint, JobRequest, create_environment
from levanter.data.text import (
    LMMixtureDatasetConfig,
)
from levanter.main.train_lm import TrainLmConfig
from levanter.models.lm_model import LmConfig
from marin.defaults import (
    SimpleTrainConfig,
    _build_train_lm_config,
    _prepare_data_config,
    _validate_train_length,
    default_train,
)
from marin.evaluation.evaluation_config import EvalTaskConfig
from marin.execution.executor import compute_output_path, materialize, resolve_local_placeholders
from marin.execution.remote import _sanitize_job_name
from marin.execution.types import ExecutorStep, InputName
from marin.training.run_environment import extras_for_resources
from marin.training.training import (
    bake_output_path,
    check_train_config_paths,
    impute_run_id,
    resolve_training_env,
)

from experiments.evals.exp1600_uncheatable_evals import (
    uncheatable_eval_raw_validation_sets,
)
from experiments.evals.task_configs import CORE_TASKS
from experiments.paloma import paloma_raw_validation_sets

logger = logging.getLogger(__name__)


@lru_cache
def default_raw_validation_sets() -> dict[str, Any]:
    validation_sets = dict(paloma_raw_validation_sets())
    validation_sets.update(uncheatable_eval_raw_validation_sets())
    return validation_sets


def simulated_epoching_train(
    name: str,
    tokenized: InputName | ExecutorStep | LMMixtureDatasetConfig,
    model_config: LmConfig,
    train_config: SimpleTrainConfig,
    target_budget: int,
    tags: Sequence[str] = (),
    use_default_validation: bool = True,
    eval_harness_tasks: Sequence[EvalTaskConfig] = CORE_TASKS,
) -> ExecutorStep:
    """
    Simulates the number of epochs seen in a full training run by sub-sampling individual datasets.
    Otherwise, operates the same as default_train.

    Args:
        name:  The name of the training run. Will form the basis of the output path for the executor step.
        tokenized:  The tokenized data to train on. This can be an InputName, ExecutorStep, or LMMixtureDatasetConfig.
        model_config: Levanter LmConfig for the model to train.
        train_config: SimpleTrainConfig for the training run.
        target_budget: Target token budget to simulate.
        tags: Any additional tags to add to the Wandb tracker.
        use_default_validation: Whether to use the default validation sets (currently Paloma).
        eval_harness_tasks: List of evaluation harness tasks. Defaults to the CORE set of tasks. Use () or [] to disable
    """
    pretraining_data = _prepare_data_config(tokenized, use_default_validation)

    train_length = _validate_train_length(train_config.train_seq_len, model_config)

    # Calculate the experiment token budget
    experiment_budget = train_config.train_batch_size * train_config.num_train_steps * train_length

    simulated_pretraining_data = dataclasses.replace(
        pretraining_data, target_budget=target_budget, experiment_budget=experiment_budget
    )

    logger.info(
        f"Simulating Epoching Behavior, Experiment Tokens {experiment_budget}, "
        + "Simulated Target Tokens {target_budget}"
    )

    return default_train(
        name, simulated_pretraining_data, model_config, train_config, tags, use_default_validation, eval_harness_tasks
    )


def _submit_train_job(
    name: str,
    entrypoint_callable: Callable[..., None],
    args: Sequence[Any],
    resources: ResourceConfig,
    env_vars: dict[str, str] | None,
) -> None:
    """Resolve env, build a JobRequest, submit to Iris, block on completion.

    Args:
        name: Job name (used for the Iris job label after sanitization).
        entrypoint_callable: Top-level callable invoked on the worker. The
            worker is responsible for the resolution chain (compute the output
            path under its own region, bake checkpointer paths, materialize
            placeholders) before running training.
        args: Positional arguments passed to ``entrypoint_callable``. Carries
            placeholder-bearing configs and any other state the worker needs.
        resources: TPU/GPU/CPU resources to request from Iris.
        env_vars: Env vars injected into the Iris worker at startup. Values are
            resolved in the caller's process.
    """
    resolved_env_vars = dict(env_vars or {})
    env = resolve_training_env(resolved_env_vars, resources)

    job_request = JobRequest(
        name=_sanitize_job_name(name),
        entrypoint=Entrypoint.from_callable(entrypoint_callable, args=list(args)),
        resources=resources,
        environment=create_environment(env_vars=env, extras=extras_for_resources(resources)),
    )

    client = current_client()
    handle = client.submit(job_request)
    handle.wait(raise_on_failure=True)


def resolve_lm_train_config(
    name: str,
    raw_config: TrainLmConfig,
    override_output_path: str | None,
    resources: ResourceConfig,
) -> TrainLmConfig:
    """Resolve a placeholder-bearing ``TrainLmConfig`` under the *current* region.

    Runs the full path-baking chain (output path computation, OutputName
    substitution, checkpointer baking, run-id imputation, materialization of
    upstream ExecutorSteps) on the caller. Designed to be invoked on the Iris
    worker so ``marin_prefix()`` reflects the worker's region after a
    cross-region preemption — putting checkpoint paths in the worker's region,
    not the submitter's.
    """
    output_path = compute_output_path(name, raw_config, override_output_path=override_output_path)
    config = resolve_local_placeholders(raw_config, output_path)
    config = bake_output_path(config, output_path)
    config, _ = impute_run_id(config, output_path=output_path)

    # Disable accelerator requirement when running without GPU/TPU resources.
    if resources.device.kind == "cpu":
        config = dataclasses.replace(
            config,
            trainer=dataclasses.replace(config.trainer, require_accelerator=False),
        )

    # Guard against cross-region GCS access; skip on CPU (no region to match).
    check_train_config_paths(config, resources)
    return materialize(config)


def _run_training_on_worker(
    name: str,
    raw_config: TrainLmConfig,
    override_output_path: str | None,
    resources: ResourceConfig,
) -> None:
    """LM training entrypoint: resolve under worker region, then run levanter.

    Top-level so Fray can pickle it as a JobRequest entrypoint.
    """
    config = resolve_lm_train_config(name, raw_config, override_output_path, resources)
    levanter_train_lm.main(config)


def prepare_lm_train(
    name: str,
    tokenized: InputName | ExecutorStep | LMMixtureDatasetConfig,
    model_config: LmConfig,
    train_config: SimpleTrainConfig,
    *,
    tags: Sequence[str] = (),
    use_default_validation: bool = True,
    eval_harness_tasks: Sequence[EvalTaskConfig] = CORE_TASKS,
    wandb_name: str | None = None,
    wandb_group: str | None = None,
) -> tuple[str, TrainLmConfig]:
    """Build the placeholder-bearing trainer config without resolving paths.

    Path resolution is deferred to the worker (via ``resolve_lm_train_config``)
    so cross-region preemption picks up the worker's region instead of the
    submitter's. Does NOT submit any Iris job.

    Returns:
        ``(job_name, raw_config)`` where ``job_name`` is the
        ``checkpoints/<truncated_name>`` string used for the Iris job label and
        ``raw_config`` still has ``OutputName`` / ``InputName`` placeholders.
    """
    truncated_name, inner_config = _build_train_lm_config(
        name,
        tokenized,
        model_config,
        train_config,
        tags=tags,
        use_default_validation=use_default_validation,
        eval_harness_tasks=eval_harness_tasks,
        wandb_name=wandb_name,
        wandb_group=wandb_group,
    )
    return os.path.join("checkpoints", truncated_name), inner_config


def train(
    name: str,
    tokenized: InputName | ExecutorStep | LMMixtureDatasetConfig,
    model_config: LmConfig,
    train_config: SimpleTrainConfig,
    *,
    tags: Sequence[str] = (),
    use_default_validation: bool = True,
    eval_harness_tasks: Sequence[EvalTaskConfig] = CORE_TASKS,
    wandb_name: str | None = None,
    wandb_group: str | None = None,
    override_output_path: str | None = None,
) -> None:
    """Build and immediately submit a Levanter LM training job to Iris.

    Path baking (output path computation, checkpointer paths, run-id stamping)
    is deferred to the worker so a job preempted across regions resolves under
    the new region. Blocks until the Iris job completes. This is the
    single-call alternative to ``prepare_lm_train`` + ``_submit_train_job``.

    Args:
        name: Human-readable identifier; forms the basis of the output path.
        tokenized: Tokenized data to train on (InputName, ExecutorStep, or
            LMMixtureDatasetConfig).
        model_config: Levanter LmConfig for the model architecture.
        train_config: SimpleTrainConfig for the training run.
        tags: Additional W&B tags.
        use_default_validation: Whether to include the default Paloma validation sets.
        eval_harness_tasks: Evaluation harness tasks. Defaults to CORE_TASKS.
            Pass ``()`` or ``[]`` to disable.
        wandb_name: Optional W&B display name. Defaults to W&B's auto-generated name.
        wandb_group: Optional W&B group. Defaults to ``$WANDB_GROUP`` if unset.
        override_output_path: Optional explicit output path, bypassing the hash-based one.
    """
    job_name, inner_config = prepare_lm_train(
        name,
        tokenized,
        model_config,
        train_config,
        tags=tags,
        use_default_validation=use_default_validation,
        eval_harness_tasks=eval_harness_tasks,
        wandb_name=wandb_name,
        wandb_group=wandb_group,
    )

    _submit_train_job(
        name=job_name,
        entrypoint_callable=_run_training_on_worker,
        args=[job_name, inner_config, override_output_path, train_config.resources],
        resources=train_config.resources,
        env_vars=dict(train_config.env_vars or {}),
    )
