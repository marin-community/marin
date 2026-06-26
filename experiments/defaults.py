# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""
This file represents the best practices for each stage of the pipeline.
"""

import dataclasses
import logging
import os
from collections.abc import Callable, Sequence
from datetime import timedelta
from functools import lru_cache
from typing import Any

import jmp
import levanter.main.train_lm as levanter_train_lm
from fray import ResourceConfig, current_client
from fray.types import Entrypoint, JobRequest, create_environment
from haliax.partitioning import ResourceAxis
from haliax.quantization import QuantizationConfig
from levanter.adaptor import AdaptorConfig, LoraAdaptorConfig, NoAdaptorConfig
from levanter.checkpoint import CheckpointerConfig
from levanter.data.text import (
    DEFAULT_LM_DATA_SHUFFLE,
    LMMixtureDatasetConfig,
    PreferenceLmDataConfig,
)
from levanter.eval_harness import LmEvalHarnessConfig
from levanter.main.train_dpo import SeparateReferenceConfig, TrainDpoConfig
from levanter.main.train_lm import TrainLmConfig
from levanter.models.llama import LlamaConfig
from levanter.models.lm_model import LmConfig
from levanter.optim import AdamConfig
from levanter.optim.model_averaging import EmaModelAveragingConfig
from levanter.schedule import BatchSchedule
from levanter.tracker.wandb import WandbConfig, truncate_wandb_run_name
from levanter.trainer import TrainerConfig
from levanter.utils.mesh import MeshConfig
from marin.evaluation.evaluation_config import EvalTaskConfig, convert_to_levanter_task_config
from marin.execution.executor import compute_output_path, materialize, resolve_local_placeholders, unwrap_versioned_value
from marin.execution.remote import _sanitize_job_name
from marin.execution.types import ExecutorStep, InputName, this_output_path, versioned
from marin.processing.tokenize import (
    TokenizerStep,
    add_validation_sets_to_mixture,
    lm_data_config,
    lm_mixture_data_config,
)
from marin.processing.tokenize.tokenize import HfTokenizeConfig, TokenizeConfigBase
from marin.training.run_environment import extras_for_resources
from marin.training.training import (
    TrainDpoOnPodConfig,
    TrainLmOnPodConfig,
    bake_output_path,
    check_train_config_paths,
    impute_run_id,
    resolve_training_env,
    run_levanter_train_dpo,
    run_levanter_train_lm,
)

from experiments.evals.exp1600_uncheatable_evals import (
    uncheatable_eval_raw_validation_sets,
    uncheatable_eval_tokenized,
)
from experiments.evals.task_configs import CORE_TASKS
from experiments.paloma import paloma_raw_validation_sets, paloma_tokenized
from experiments.simple_dpo_config import SimpleDPOConfig
from experiments.simple_sft_config import SimpleSFTConfig
from experiments.simple_train_config import SimpleTrainConfig

logger = logging.getLogger(__name__)


def _resolve_hf_export_steps(steps_per_hf_export: int | None, steps_per_export: int | None) -> int | None:
    """Resolve the HF export step interval: None means same as checkpoint, -1 means disabled."""
    if steps_per_hf_export is None:
        return steps_per_export
    if steps_per_hf_export == -1:
        return None
    return steps_per_hf_export


def _checkpoint_keep(steps_per_export: int | None) -> list[dict]:
    """Build the `keep` list for `CheckpointerConfig`.

    None means keep no permanent intermediate checkpoints (only the final checkpoint
    is saved at end-of-training, plus a rolling temporary checkpoint for resumption).
    """
    if steps_per_export is None:
        return []
    return [dict(every=steps_per_export)]


def _validate_train_length(train_seq_len: int | None, model_config: LmConfig) -> int:
    """Resolve and validate the training sequence length against the model's max."""
    actual = unwrap_versioned_value(model_config)
    train_length = train_seq_len or actual.max_seq_len
    if train_length > actual.max_seq_len:
        raise ValueError(f"train_length {train_length} exceeds model max_seq_len {actual.max_seq_len}.")
    return train_length


@lru_cache  # LRU to make the executor happier
def default_validation_sets(tokenizer: str, base_path: str = "tokenized/") -> dict[str, TokenizerStep]:
    validation_sets = dict(paloma_tokenized(base_path=base_path, tokenizer=tokenizer))
    validation_sets.update(uncheatable_eval_tokenized(base_path=base_path, tokenizer=tokenizer))
    return validation_sets


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


def _build_train_lm_config(
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
    wandb_project: str | None = None,
    adapter: AdaptorConfig | None = None,
) -> tuple[str, TrainLmConfig]:
    """Build the shared ``TrainLmConfig`` body used by ``default_train`` and ``prepare_lm_train``.

    Returns:
        (truncated_name, inner_config) where ``truncated_name`` is the W&B-safe
        version of ``name`` and ``inner_config`` is the fully-populated config.
        The caller is responsible for baking in a concrete ``output_path``,
        resolving placeholders, and imputing a run id.
    """
    pretraining_data = _prepare_data_config(tokenized, use_default_validation)

    if wandb_group is None:
        wandb_group = os.environ.get("WANDB_GROUP")
    if wandb_project is None:
        wandb_project = os.environ.get("WANDB_PROJECT", "marin")

    name = truncate_wandb_run_name(name)

    if eval_harness_tasks:
        harness_config = LmEvalHarnessConfig(task_spec=convert_to_levanter_task_config(eval_harness_tasks))
    else:
        harness_config = None

    steps_per_export = train_config.steps_per_export
    steps_per_export_hf = _resolve_hf_export_steps(train_config.steps_per_hf_export, steps_per_export)

    model_averaging = None
    if train_config.ema_beta is not None:
        model_averaging = EmaModelAveragingConfig(beta=train_config.ema_beta)

    if train_config.per_device_eval_parallelism is None:
        per_device_eval_parallelism = -1
    else:
        per_device_eval_parallelism = train_config.per_device_eval_parallelism

    checkpoint_path_to_load_from = train_config.initialize_from_checkpoint_path
    hf_checkpoint_path_to_load_from = train_config.initialize_from_hf

    if hf_checkpoint_path_to_load_from is not None and checkpoint_path_to_load_from is not None:
        raise ValueError("Cannot specify both initialize_from_checkpoint_path and initialize_from_hf")

    train_length = _validate_train_length(train_config.train_seq_len, model_config)

    inner_config = TrainLmConfig(
        data=pretraining_data,
        trainer=TrainerConfig(
            tracker=WandbConfig(
                project=wandb_project or "marin",
                name=wandb_name,
                tags=[*tags],
                group=wandb_group,
                replicate_path=this_output_path(),
            ),
            mp=jmp.get_policy("p=f32,c=bfloat16"),
            train_batch_size=train_config.train_batch_size,
            per_device_parallelism=train_config.per_device_parallelism,
            num_train_steps=train_config.num_train_steps,
            steps_per_eval=train_config.steps_per_eval if train_config.steps_per_eval is not None else 1000,
            checkpointer=CheckpointerConfig(
                save_interval=timedelta(minutes=10),
                keep=_checkpoint_keep(steps_per_export),
            ),
            model_averaging=model_averaging,
            mesh=MeshConfig(
                axes={"replica": 1, "data": -1, "model": train_config.tensor_parallel_size},
                # Special axes for MoEs
                # TODO: this is actually bad and we should remove, but keeping for now
                compute_mapping={
                    "token": (ResourceAxis.REPLICA_DCN, ResourceAxis.REPLICA, ResourceAxis.DATA),
                    "token_repeat": (ResourceAxis.REPLICA_DCN, ResourceAxis.REPLICA, ResourceAxis.DATA),
                },
            ),
            allow_partial_checkpoint=train_config.allow_partial_checkpoint,
            per_device_eval_parallelism=per_device_eval_parallelism,
            max_eval_batches=train_config.max_eval_batches,
            allow_nondivisible_batch_size=True,
            quantization=QuantizationConfig(int8=train_config.int8) if train_config.int8 else None,
            initialize_from=None if train_config.reset_data_loader_on_init else checkpoint_path_to_load_from,
            watch=train_config.watch,
            profiler=train_config.profiler,
            use_explicit_mesh_axes=train_config.explicit_mesh_axes,
        ),
        initialize_from_checkpoint_path=(
            checkpoint_path_to_load_from if train_config.reset_data_loader_on_init else None
        ),
        checkpoint_init_mode=train_config.checkpoint_init_mode,
        initialize_from_hf=hf_checkpoint_path_to_load_from or False,
        pad_tokenizer_to_match_model=train_config.pad_tokenizer_to_match_model,
        z_loss_weight=train_config.z_loss_weight,
        train_seq_len=train_length,
        model=model_config,
        optimizer=(
            train_config.optimizer_config
            if getattr(train_config, "optimizer_config", None) is not None
            else AdamConfig(
                learning_rate=train_config.learning_rate,
                weight_decay=(
                    train_config.weight_decay if train_config.weight_decay is not None else AdamConfig().weight_decay
                ),
                beta1=(train_config.beta1 if train_config.beta1 is not None else AdamConfig().beta1),
                beta2=(train_config.beta2 if train_config.beta2 is not None else AdamConfig().beta2),
                epsilon=(train_config.epsilon if train_config.epsilon is not None else AdamConfig().epsilon),
                max_grad_norm=(
                    train_config.max_grad_norm if train_config.max_grad_norm is not None else AdamConfig().max_grad_norm
                ),
                warmup=(train_config.warmup if train_config.warmup is not None else AdamConfig().warmup),
                rewarmup=(train_config.rewarmup if train_config.rewarmup is not None else AdamConfig().rewarmup),
                decay=(train_config.decay if train_config.decay is not None else AdamConfig().decay),
                lr_schedule=(
                    train_config.lr_schedule if train_config.lr_schedule is not None else AdamConfig().lr_schedule
                ),
                cycle_length=train_config.cycle_length,  # can be int, list[int], or None
                min_lr_ratio=(
                    train_config.min_lr_ratio if train_config.min_lr_ratio is not None else AdamConfig().min_lr_ratio
                ),
                skip_bad_steps=train_config.skip_bad_steps,
            )
        ),
        hf_save_steps=steps_per_export_hf,
        hf_generation_eos_token_ids=train_config.hf_generation_eos_token_ids,
        data_seed=train_config.data_seed,
        eval_harness_steps=train_config.steps_per_task_eval or 10000,
        eval_harness=harness_config,
        adapter=adapter if adapter is not None else NoAdaptorConfig(),
    )

    return name, inner_config


def default_train(
    name: str,
    tokenized: InputName | ExecutorStep | LMMixtureDatasetConfig,
    model_config: LmConfig,
    train_config: SimpleTrainConfig,
    tags: Sequence[str] = (),
    use_default_validation: bool = True,
    eval_harness_tasks: Sequence[EvalTaskConfig] = CORE_TASKS,
    wandb_name: str | None = None,
    wandb_group: str | None = None,
    wandb_project: str | None = None,
    override_output_path: str | None = None,
    adapter: AdaptorConfig | None = None,
) -> ExecutorStep:
    """
    Train a language model using the default configuration.

    Args:
        name:  The name of the training run. Will form the basis of the output path for the executor step.
        tokenized:  The tokenized data to train on. This can be an InputName, ExecutorStep, or LMMixtureDatasetConfig.
        model_config: Levanter LmConfig for the model to train.
        train_config: SimpleTrainConfig for the training run.
        tags: Any additional tags to add to the Wandb tracker.
        use_default_validation: Whether to use the default validation sets (currently Paloma).
        eval_harness_tasks: List of evaluation harness tasks. Defaults to the CORE set of tasks. Use () or [] to disable
        wandb_name: Optional W&B display name for this run. Defaults to W&B's auto-generated name.
        wandb_group: Optional W&B group to organize related runs (e.g., a sweep). If unset, defaults to $WANDB_GROUP.
        wandb_project: Optional W&B project. If unset, defaults to $WANDB_PROJECT, then "marin".
    """
    name, inner_config = _build_train_lm_config(
        name,
        tokenized,
        model_config,
        train_config,
        tags=tags,
        use_default_validation=use_default_validation,
        eval_harness_tasks=eval_harness_tasks,
        wandb_name=wandb_name,
        wandb_group=wandb_group,
        wandb_project=wandb_project,
        adapter=adapter,
    )

    pretraining_data = inner_config.data
    tokenizer_name = unwrap_versioned_value(pretraining_data.tokenizer)
    train_length = unwrap_versioned_value(inner_config.train_seq_len)
    schedule = BatchSchedule(unwrap_versioned_value(train_config.train_batch_size))
    total_examples = schedule.global_data_offset_by_step(unwrap_versioned_value(train_config.num_train_steps))

    pod_config = train_config.resources

    config = TrainLmOnPodConfig(
        train_config=inner_config,
        resources=pod_config,
        output_path=this_output_path(),
        env_vars=train_config.env_vars,
    )

    model_config = unwrap_versioned_value(model_config)

    return ExecutorStep(
        name=os.path.join("checkpoints", name),
        description=(
            f"Train a model (tokenizer={tokenizer_name}) for "
            f"{unwrap_versioned_value(train_config.num_train_steps)} (steps) * "
            f"{unwrap_versioned_value(train_config.train_batch_size)} (batch_size) * "
            f"{train_length} (train_seq_len) "
            f"= {total_examples * train_length} tokens."
        ),
        fn=run_levanter_train_lm,
        resources=train_config.resources,
        config=config,
        override_output_path=override_output_path,
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
    wandb_project: str | None = None,
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
        wandb_project=wandb_project,
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
    wandb_project: str | None = None,
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
        wandb_project: Optional W&B project. Defaults to ``$WANDB_PROJECT`` if unset.
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
        wandb_project=wandb_project,
    )

    _submit_train_job(
        name=job_name,
        entrypoint_callable=_run_training_on_worker,
        args=[job_name, inner_config, override_output_path, train_config.resources],
        resources=train_config.resources,
        env_vars=dict(train_config.env_vars or {}),
    )


def default_sft(
    name: str,
    tokenized: InputName | ExecutorStep | LMMixtureDatasetConfig,
    model_config: LlamaConfig,
    sft_config: SimpleSFTConfig,
    tags: Sequence[str] = (),
    adapter: AdaptorConfig | None = None,
) -> ExecutorStep:
    """
    Creates an ExecutorStep for supervised fine-tuning of a language model.

    This function provides a unified interface for both single-dataset SFT and mixture-based
    SFT with a simplified configuration approach.

    Args:
        name: The name of the training run, forms the basis of the output path.
        tokenized: The tokenized data to train on:
                  - For single dataset: an InputName or ExecutorStep for a tokenized dataset.
                  - For mixture: a LMMixtureDatasetConfig with multiple datasets.
        model_config: Levanter LlamaConfig for the model architecture to train.
        sft_config: Configuration for the SFT training process.
        tags: Additional tags for WandB logging. Default: ().

    Returns:
        An ExecutorStep configured for supervised fine-tuning.
    """
    if "sft" not in tags:
        tags = [*tags, "sft"]

    if sft_config.initialize_from_hf is not None and sft_config.initialize_from_checkpoint_path is not None:
        raise ValueError("Cannot specify both initialize_from_hf and initialize_from_checkpoint_path!")

    normal_train_config = SimpleTrainConfig(
        resources=sft_config.resources,
        train_batch_size=sft_config.train_batch_size,
        num_train_steps=sft_config.num_train_steps,
        learning_rate=sft_config.learning_rate,
        lr_schedule=sft_config.lr_schedule,
        decay=sft_config.decay,
        weight_decay=sft_config.weight_decay,
        min_lr_ratio=sft_config.min_lr_ratio,
        max_grad_norm=sft_config.max_grad_norm,
        warmup=sft_config.warmup,
        steps_per_eval=sft_config.steps_per_eval,
        steps_per_export=sft_config.steps_per_checkpoint,
        int8=sft_config.int8,
        steps_per_hf_export=sft_config.steps_per_hf_export,
        initialize_from_hf=sft_config.initialize_from_hf,
        initialize_from_checkpoint_path=sft_config.initialize_from_checkpoint_path,
        train_seq_len=sft_config.max_seq_len,
        data_seed=sft_config.seed,
        z_loss_weight=sft_config.z_loss_weight,
        beta1=sft_config.beta1,
        beta2=sft_config.beta2,
        pad_tokenizer_to_match_model=sft_config.pad_tokenizer_to_match_model,
        per_device_parallelism=sft_config.per_device_parallelism,
        hf_generation_eos_token_ids=sft_config.hf_generation_eos_token_ids,
    )

    if sft_config.reinit_tokens:
        raise NotImplementedError("reinit_tokens is not supported by default_train")

    return default_train(
        name=name,
        tokenized=tokenized,
        model_config=model_config,
        train_config=normal_train_config,
        tags=tags,
        eval_harness_tasks=[],
        use_default_validation=False,
        adapter=adapter,
    )


def default_dpo(
    name: str,
    tokenized: InputName | ExecutorStep | LMMixtureDatasetConfig,
    model_config: LlamaConfig,
    dpo_config: SimpleDPOConfig,
    tags: Sequence[str] = (),
    override_output_path: str | None = None,
) -> ExecutorStep:
    """
    Creates an ExecutorStep for DPO fine-tuning.

    Args:
        name: The name of the training run, forms the basis of the output path.
        tokenized: The tokenized preference data to train on.
        model_config: Levanter LlamaConfig for the model architecture to train.
        dpo_config: Configuration for the DPO training process.
        tags: Additional tags for WandB logging. Default: ().
        override_output_path: Optional override for executor output path.
    """
    if "dpo" not in tags:
        tags = [*tags, "dpo"]

    initialize_from_hf = dpo_config.initialize_from_hf

    if initialize_from_hf is None:
        initialize_from_hf = (
            dpo_config.model_name_or_path is not None and dpo_config.initialize_from_checkpoint_path is None
        )
    elif initialize_from_hf is True and dpo_config.model_name_or_path is None:
        raise ValueError("initialize_from_hf is True but model_name_or_path is not set")
    elif initialize_from_hf is False and dpo_config.initialize_from_checkpoint_path is None:
        raise ValueError("initialize_from_hf is False but initialize_from_checkpoint_path is not set")

    pretraining_data = _prepare_data_config(tokenized, use_default_validation=False)
    preference_data = PreferenceLmDataConfig.from_lm_data_config(pretraining_data)
    preference_data = dataclasses.replace(preference_data, permutation_type="feistel")
    dpo_tokenizer_name = unwrap_versioned_value(preference_data.tokenizer)
    lm_validation_data = lm_mixture_data_config(
        default_validation_sets(tokenizer=dpo_tokenizer_name),
        {},
        missing_weights_are_validation=True,
        include_raw_paths=False,
    )

    name = truncate_wandb_run_name(name)

    steps_per_export = dpo_config.steps_per_checkpoint
    steps_per_export_hf = _resolve_hf_export_steps(dpo_config.steps_per_hf_export, steps_per_export)

    train_length = _validate_train_length(dpo_config.train_seq_len, model_config)

    requested_num_train_steps = dpo_config.num_train_steps
    auto_num_epochs = None
    if requested_num_train_steps is None:
        requested_num_train_steps = 1
        auto_num_epochs = dpo_config.num_epochs

    requested_steps_per_eval = dpo_config.steps_per_eval
    auto_validation_runs = None
    if requested_steps_per_eval is None:
        requested_steps_per_eval = 1
        auto_validation_runs = 5

    schedule = BatchSchedule(unwrap_versioned_value(dpo_config.train_batch_size))
    total_examples = schedule.global_data_offset_by_step(requested_num_train_steps)

    reference = dpo_config.reference
    if isinstance(reference, SeparateReferenceConfig) and not reference.model_path:
        reference_model_path = dpo_config.reference_model_path or dpo_config.model_name_or_path
        if reference_model_path is None:
            raise ValueError("reference_model_path must be set for DPO training when using a separate reference.")
        reference = dataclasses.replace(
            reference,
            model_path=reference_model_path,
            is_hf=dpo_config.reference_is_hf,
        )

    # Default DPO LoRA to the topology-stable A=0/B=Gaussian init.
    # Standard LoRA init is fragile here; see
    # https://github.com/marin-community/marin/issues/4755.
    # Users who need paper init can construct TrainDpoConfig directly.
    if isinstance(dpo_config.adapter, LoraAdaptorConfig):
        dpo_config = dataclasses.replace(
            dpo_config,
            adapter=dataclasses.replace(
                dpo_config.adapter,
                a_init_mode="zero",
                zero_init_b=False,
            ),
        )

    hf_save_dtype = dpo_config.hf_save_dtype
    if not isinstance(dpo_config.adapter, NoAdaptorConfig) and hf_save_dtype is not None:
        raise ValueError("hf_save_dtype is not supported with adapter-based DPO exports.")

    inner_config = TrainDpoConfig(
        data=preference_data,
        trainer=TrainerConfig(
            tracker=WandbConfig(
                project=dpo_config.wandb_project or "marin",
                tags=[*tags],
            ),
            mp=jmp.get_policy("p=f32,c=bfloat16"),
            train_batch_size=dpo_config.train_batch_size,
            num_train_steps=requested_num_train_steps,
            steps_per_eval=requested_steps_per_eval,
            checkpointer=CheckpointerConfig(
                save_interval=timedelta(minutes=10),
                keep=_checkpoint_keep(steps_per_export),
            ),
            model_averaging=None,
            mesh=MeshConfig(
                compute_mapping={
                    "token": (ResourceAxis.REPLICA_DCN, ResourceAxis.REPLICA, ResourceAxis.DATA),
                    "token_repeat": (ResourceAxis.REPLICA_DCN, ResourceAxis.REPLICA, ResourceAxis.DATA),
                }
            ),
            per_device_eval_parallelism=dpo_config.per_device_eval_parallelism,
            profiler=dpo_config.profiler,
            allow_partial_checkpoint=dpo_config.allow_partial_checkpoint,
            allow_nondivisible_batch_size=True,
            quantization=QuantizationConfig(int8=dpo_config.int8) if dpo_config.int8 else None,
            initialize_from=None,
        ),
        initialize_from_checkpoint_path=dpo_config.initialize_from_checkpoint_path,
        initialize_from_hf=dpo_config.model_name_or_path if initialize_from_hf else False,
        train_seq_len=train_length,
        model=model_config,
        adapter=dpo_config.adapter,
        optimizer=AdamConfig(
            learning_rate=dpo_config.learning_rate,
            weight_decay=dpo_config.weight_decay,
            warmup=dpo_config.warmup,
            decay=dpo_config.cooldown,
            lr_schedule=dpo_config.lr_schedule,
            min_lr_ratio=dpo_config.min_lr_ratio,
            max_grad_norm=dpo_config.max_grad_norm,
        ),
        reference=reference,
        beta=dpo_config.beta,
        validation_split_fraction=dpo_config.validation_split_fraction,
        reference_eval_cache=dpo_config.reference_eval_cache,
        lm_validation_data=lm_validation_data,
        hf_save_steps=steps_per_export_hf,
        hf_save_dtype=hf_save_dtype,
        hf_generation_eos_token_ids=dpo_config.hf_generation_eos_token_ids,
        data_seed=dpo_config.seed,
    )

    config = TrainDpoOnPodConfig(
        train_config=inner_config,
        resources=dpo_config.resources,
        output_path=this_output_path(),
        auto_num_epochs=auto_num_epochs,
        auto_validation_runs=auto_validation_runs,
    )

    model_config = unwrap_versioned_value(model_config)

    return ExecutorStep(
        name=os.path.join("checkpoints", name),
        description=(
            (
                f"Train a model (tokenizer={dpo_tokenizer_name}) for "
                f"{requested_num_train_steps} (steps) * "
                f"{dpo_config.train_batch_size} (batch_size) * "
                f"{train_length} (train_seq_len) "
                f"= {total_examples * train_length} tokens."
            )
            if auto_num_epochs is None
            else (
                f"Train a model (tokenizer={dpo_tokenizer_name}) for "
                f"{dpo_config.num_epochs:g} epoch(s) with runtime-resolved step count "
                f"and train_seq_len={train_length}."
            )
        ),
        fn=run_levanter_train_dpo,
        config=config,
        override_output_path=override_output_path,
    )


def _prepare_data_config(
    tokenized: InputName | ExecutorStep | LMMixtureDatasetConfig,
    use_default_validation: bool,
) -> LMMixtureDatasetConfig:
    """
    Prepare a tokenized dataset for training. This is mostly just combining the tokenized data with the validation sets.

    Returns:
        The data config to use for training with any validation sets added.
        The evaluation data config for internal evaluation.

    """
    tokenizer = _get_tokenizer_for_train(tokenized)
    if use_default_validation:
        validation_sets = default_validation_sets(tokenizer=tokenizer)
    else:
        validation_sets = {}

    if isinstance(tokenized, InputName | ExecutorStep):
        pretraining_data = lm_data_config(
            training_set=tokenized,
            validation_sets=validation_sets,
            shuffle=versioned(DEFAULT_LM_DATA_SHUFFLE),
        )
    else:
        # TODO: would be better to expose hooks in levanter instead of relying on mixtures
        pretraining_data = tokenized
        if validation_sets:
            pretraining_data = add_validation_sets_to_mixture(pretraining_data, validation_sets)
    return pretraining_data


def _get_tokenizer_for_train(tokenized: InputName | ExecutorStep | LMMixtureDatasetConfig) -> str:
    match tokenized:
        case LMMixtureDatasetConfig(tokenizer=tokenizer):
            pass
        case ExecutorStep(config=config) if isinstance(config, TokenizeConfigBase):
            tokenizer = config.tokenizer
        case ExecutorStep(config=HfTokenizeConfig(tokenizer=tokenizer)):
            pass
        case InputName(step=ExecutorStep(config)) if isinstance(config, TokenizeConfigBase):
            tokenizer = config.tokenizer
        case _:
            raise ValueError(f"Could not determine tokenizer from {tokenized}")

    return tokenizer
