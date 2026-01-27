# Copyright 2025 The Marin Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Default training configuration utilities for language model training.

This module provides `default_train()`, which creates an ExecutorStep for training
a language model with Levanter. Unlike the experiments version, this module requires
explicit parameters for validation sets and eval harness tasks - no defaults are assumed.
"""

import logging
import os
from collections.abc import Sequence
from datetime import timedelta
from functools import lru_cache

import jmp
from haliax.partitioning import ResourceAxis
from haliax.quantization import QuantizationConfig
from levanter.checkpoint import CheckpointerConfig
from levanter.compat.hf_checkpoints import load_tokenizer
from levanter.data.text import LMMixtureDatasetConfig, UrlDatasetSourceConfig
from levanter.eval_harness import LmEvalHarnessConfig
from levanter.main.train_lm import TrainLmConfig
from levanter.models.lm_model import LmConfig
from levanter.optim import AdamConfig
from levanter.schedule import BatchSchedule
from levanter.tracker.wandb import WandbConfig
from levanter.trainer import TrainerConfig
from levanter.utils.mesh import MeshConfig

from marin.evaluation.evaluation_config import EvalTaskConfig
from marin.evaluation.tasks import convert_to_levanter_task_config
from marin.execution.executor import (
    ExecutorStep,
    InputName,
    this_output_path,
    unwrap_versioned_value,
)
from marin.processing.tokenize import (
    TokenizerStep,
    add_validation_sets_to_mixture,
    lm_data_config,
)
from marin.processing.tokenize.tokenize import HfTokenizeConfig, TokenizeConfigBase
from marin.training.simple_train_config import SimpleTrainConfig
from marin.training.training import (
    TrainLmOnPodConfig,
    run_levanter_train_lm,
)

logger = logging.getLogger("ray")


@lru_cache
def _cached_load_tokenizer(tokenizer_name: str):
    return load_tokenizer(tokenizer_name)


def _get_vocab_size(pretraining_data: LMMixtureDatasetConfig) -> int:
    tokenizer = unwrap_versioned_value(pretraining_data.tokenizer)
    vocab_size = _cached_load_tokenizer(tokenizer).vocab_size
    return vocab_size


def get_tokenizer_for_train(
    tokenized: str | InputName | ExecutorStep | LMMixtureDatasetConfig,
    tokenizer: str | None = None,
) -> str:
    """
    Extract the tokenizer name from a tokenized dataset or data config.

    This function inspects the input to determine which tokenizer was used for tokenization.
    It supports string paths (with explicit tokenizer), InputName, ExecutorStep
    (with TokenizeConfig or HfTokenizeConfig), and LMMixtureDatasetConfig inputs.

    Args:
        tokenized: The tokenized data source. Can be:
            - A string path to a pre-tokenized dataset (requires tokenizer parameter)
            - InputName pointing to an ExecutorStep with a tokenize config
            - ExecutorStep with a TokenizeConfigBase config
            - LMMixtureDatasetConfig with a tokenizer field
        tokenizer: Required when tokenized is a string path. The tokenizer name
            (e.g., "meta-llama/Meta-Llama-3.1-8B") used to create the pre-tokenized dataset.

    Returns:
        The tokenizer name (e.g., "meta-llama/Meta-Llama-3.1-8B")

    Raises:
        ValueError: If the tokenizer cannot be determined from the input
    """
    if isinstance(tokenized, str):
        if tokenizer is None:
            raise ValueError(
                "When tokenized is a string path to a pre-tokenized dataset, "
                "the tokenizer parameter must be provided."
            )
        return tokenizer

    match tokenized:
        case LMMixtureDatasetConfig(tokenizer=tok):
            return tok
        case ExecutorStep(config=config) if isinstance(config, TokenizeConfigBase):
            return config.tokenizer
        case ExecutorStep(config=HfTokenizeConfig(tokenizer=tok)):
            return tok
        case InputName(step=ExecutorStep(config)) if isinstance(config, TokenizeConfigBase):
            return config.tokenizer
        case _:
            raise ValueError(f"Could not determine tokenizer from {tokenized}")


def _prepare_data_config(
    tokenized: str | InputName | ExecutorStep | LMMixtureDatasetConfig,
    validation_sets: dict[str, TokenizerStep] | None,
    tokenizer: str | None = None,
) -> LMMixtureDatasetConfig:
    """
    Prepare a tokenized dataset for training by combining it with validation sets.

    Args:
        tokenized: The tokenized data source. Can be a string path to a pre-tokenized
            dataset, an InputName, ExecutorStep, or LMMixtureDatasetConfig.
        validation_sets: Dictionary mapping validation set names to TokenizerStep instances.
        tokenizer: Required when tokenized is a string path. The tokenizer name used
            to create the pre-tokenized dataset.

    Returns:
        The data config to use for training with any validation sets added.
    """
    if validation_sets is None:
        validation_sets = {}

    if isinstance(tokenized, str):
        if tokenizer is None:
            raise ValueError(
                "When tokenized is a string path to a pre-tokenized dataset, "
                "the tokenizer parameter must be provided."
            )
        # Create an LMMixtureDatasetConfig from the pre-tokenized dataset path.
        # The path is used as the cache_dir since it contains pre-built tokenized data.
        dataset_name = os.path.basename(tokenized.rstrip("/"))
        source_config = UrlDatasetSourceConfig(cache_dir=tokenized)
        pretraining_data = LMMixtureDatasetConfig(
            configs={dataset_name: source_config},
            train_weights={dataset_name: 1.0},
            tokenizer=tokenizer,
            cache_dir=None,
            shuffle=True,
            permutation_type="feistel",
        )
        if validation_sets:
            pretraining_data = add_validation_sets_to_mixture(pretraining_data, validation_sets)
    elif isinstance(tokenized, InputName | ExecutorStep):
        pretraining_data = lm_data_config(
            training_set=tokenized,
            validation_sets=validation_sets,
            permutation_type="feistel",
        )
    else:
        pretraining_data = tokenized
        if validation_sets:
            pretraining_data = add_validation_sets_to_mixture(pretraining_data, validation_sets)
    return pretraining_data


def default_train(
    name: str,
    tokenized: str | InputName | ExecutorStep | LMMixtureDatasetConfig,
    model_config: LmConfig,
    train_config: SimpleTrainConfig,
    *,
    tokenizer: str | None = None,
    tags: Sequence[str] = (),
    validation_sets: dict[str, TokenizerStep] | None = None,
    eval_harness_tasks: Sequence[EvalTaskConfig] | None = None,
    wandb_name: str | None = None,
    wandb_group: str | None = None,
    wandb_project: str = "marin",
    override_output_path: str | None = None,
) -> ExecutorStep:
    """
    Create an ExecutorStep for training a language model using Levanter.

    This function creates a training step with explicit configuration. Unlike the experiments
    version, it does not provide default validation sets or eval harness tasks - callers must
    explicitly provide these or pass None/{} to disable.

    Args:
        name: The name of the training run. Will form the basis of the output path for the executor step.
        tokenized: The tokenized data to train on. This can be:
            - A string path to a pre-tokenized dataset (e.g., "hf://marin-community/fineweb-edu-pretokenized-10B")
            - An InputName or ExecutorStep pointing to a tokenization step
            - An LMMixtureDatasetConfig for more complex data setups
        model_config: Levanter LmConfig for the model to train.
        train_config: SimpleTrainConfig for the training run.
        tokenizer: The tokenizer name (e.g., "meta-llama/Meta-Llama-3.1-8B"). Required when
            tokenized is a string path. When tokenized is an ExecutorStep or LMMixtureDatasetConfig,
            the tokenizer is inferred automatically.
        tags: Additional tags to add to the Wandb tracker.
        validation_sets: Dictionary mapping validation set names to TokenizerStep instances.
            Pass None or {} to disable validation.
        eval_harness_tasks: List of evaluation harness tasks. Pass None or [] to disable eval harness.
        wandb_name: Optional W&B display name for this run. Defaults to W&B's auto-generated name.
        wandb_group: Optional W&B group to organize related runs (e.g., a sweep). If unset, defaults to $WANDB_GROUP.
        wandb_project: W&B project name. Defaults to "marin".
        override_output_path: Optional path to override the default output location.

    Returns:
        An ExecutorStep configured for language model training.
    """
    pretraining_data = _prepare_data_config(tokenized, validation_sets, tokenizer)

    vocab_size = _get_vocab_size(pretraining_data)

    steps_per_export = train_config.steps_per_export

    if wandb_group is None:
        wandb_group = os.environ.get("WANDB_GROUP")

    # Max length of 64 characters for WANDB run is 64 characters
    # we don't want to use the first 64 because the UID bit goes at the end. instead, grab the trailing -XXX
    # and add whatever we can fit in the remaining space.
    if len(name) > 64:
        old_name = name
        if "-" not in name:
            name = name[:64]
        else:
            prefix, suffix = name.rsplit("-", 1)
            if len(suffix) >= 64:
                suffix = suffix[:64]
                name = suffix
            else:
                name = prefix[: 63 - len(suffix)] + "-" + suffix
        logger.warning(f"Truncated name from {old_name} to {name} to fit within WANDB limits.")

    if eval_harness_tasks:
        harness_config = LmEvalHarnessConfig(task_spec=convert_to_levanter_task_config(eval_harness_tasks))
    else:
        harness_config = None

    if train_config.steps_per_hf_export is None:
        steps_per_export_hf = steps_per_export
    elif train_config.steps_per_hf_export == -1:
        steps_per_export_hf = None
    else:
        steps_per_export_hf = train_config.steps_per_hf_export

    model_averaging = None
    if train_config.ema_beta is not None:
        from levanter.optim.model_averaging import EmaModelAveragingConfig

        model_averaging = EmaModelAveragingConfig(beta=train_config.ema_beta)

    if train_config.per_device_eval_parallelism is None:
        per_device_eval_parallelism = -1
    else:
        per_device_eval_parallelism = train_config.per_device_eval_parallelism

    schedule = BatchSchedule(unwrap_versioned_value(train_config.train_batch_size))
    total_examples = schedule.global_data_offset_by_step(train_config.num_train_steps)

    checkpoint_path_to_load_from = train_config.initialize_from_checkpoint_path
    hf_checkpoint_path_to_load_from = train_config.initialize_from_hf

    if hf_checkpoint_path_to_load_from is not None and checkpoint_path_to_load_from is not None:
        raise ValueError("Cannot specify both initialize_from_checkpoint_path and initialize_from_hf")

    # Create the inner config
    actual_model_config = unwrap_versioned_value(model_config)
    train_length = train_config.train_seq_len or actual_model_config.max_seq_len
    if train_length > actual_model_config.max_seq_len:
        raise ValueError(f"train_length {train_length} exceeds model max_seq_len {actual_model_config.max_seq_len}.")

    inner_config = TrainLmConfig(
        data=pretraining_data,
        trainer=TrainerConfig(
            tracker=WandbConfig(
                project=wandb_project,
                name=wandb_name,
                tags=[*tags],
                group=wandb_group,
            ),
            mp=jmp.get_policy("p=f32,c=bfloat16"),
            train_batch_size=train_config.train_batch_size,
            num_train_steps=train_config.num_train_steps,
            steps_per_eval=train_config.steps_per_eval if train_config.steps_per_eval is not None else 1000,
            checkpointer=CheckpointerConfig(
                save_interval=timedelta(minutes=10),
                keep=[dict(every=steps_per_export)],
            ),
            model_averaging=model_averaging,
            mesh=MeshConfig(
                compute_mapping={
                    "token": (ResourceAxis.REPLICA_DCN, ResourceAxis.REPLICA, ResourceAxis.DATA),
                    "token_repeat": (ResourceAxis.REPLICA_DCN, ResourceAxis.REPLICA, ResourceAxis.DATA),
                }
            ),
            allow_partial_checkpoint=train_config.allow_partial_checkpoint,
            per_device_eval_parallelism=per_device_eval_parallelism,
            max_eval_batches=train_config.max_eval_batches,
            allow_nondivisible_batch_size=True,
            quantization=QuantizationConfig(int8=train_config.int8) if train_config.int8 else None,
            initialize_from=None if train_config.reset_data_loader_on_init else checkpoint_path_to_load_from,
            watch=train_config.watch,
            profiler=train_config.profiler,
            profiler_start_step=train_config.profiler_start_step,
            profiler_num_steps=train_config.profiler_num_steps,
            use_explicit_mesh_axes=train_config.explicit_mesh_axes,
        ),
        initialize_from_checkpoint_path=(
            checkpoint_path_to_load_from if train_config.reset_data_loader_on_init else None
        ),
        initialize_from_hf=hf_checkpoint_path_to_load_from or False,
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
                cycle_length=train_config.cycle_length,
                min_lr_ratio=(
                    train_config.min_lr_ratio if train_config.min_lr_ratio is not None else AdamConfig().min_lr_ratio
                ),
                skip_bad_steps=train_config.skip_bad_steps,
            )
        ),
        hf_save_steps=steps_per_export_hf,
        data_seed=train_config.data_seed,
        eval_harness_steps=train_config.steps_per_task_eval or 10000,
        eval_harness=harness_config,
    )

    # Create the pod config
    pod_config = train_config.resources

    # Create the full config
    config = TrainLmOnPodConfig(
        train_config=inner_config,
        resources=pod_config,
        output_path=this_output_path(),
    )

    actual_model_config = unwrap_versioned_value(model_config)
    num_params = actual_model_config.total_trainable_params(vocab_size)

    return ExecutorStep(
        name=os.path.join("checkpoints", name),
        description=(
            f"Train a {num_params:,} parameter model for "
            f"{train_config.num_train_steps} (steps) * "
            f"{train_config.train_batch_size} (batch_size) * "
            f"{train_length} (train_seq_len) "
            f"= {total_examples * train_length} tokens."
        ),
        fn=run_levanter_train_lm,
        config=config,
        override_output_path=override_output_path,
    )
