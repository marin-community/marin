# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
import dataclasses
import os
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import timedelta
from functools import lru_cache
from typing import Any

import jmp
from fray import ResourceConfig
from haliax.partitioning import ResourceAxis
from haliax.quantization import QuantizationConfig
from levanter.adaptor import AdaptorConfig, LoraAdaptorConfig, NoAdaptorConfig
from levanter.callbacks.profiler import ProfilerConfig
from levanter.callbacks.watch import WatchConfig
from levanter.checkpoint import CheckpointerConfig
from levanter.data.text import (
    DEFAULT_LM_DATA_SHUFFLE,
    LmDatasetFormatBase,
    LMMixtureDatasetConfig,
    PreferenceLmDataConfig,
    TextLmDatasetFormat,
)
from levanter.dpo import ReferenceEvalCacheConfig
from levanter.eval_harness import LmEvalHarnessConfig
from levanter.main.train_dpo import DpoReferenceConfig, SeparateReferenceConfig, TrainDpoConfig
from levanter.main.train_lm import TrainLmConfig
from levanter.models.llama import LlamaConfig
from levanter.models.lm_model import LmConfig
from levanter.optim import AdamConfig, EmaModelAveragingConfig, OptimizerConfig
from levanter.schedule import BatchSchedule, IntSchedule
from levanter.tracker.wandb import WandbConfig, truncate_wandb_run_name
from levanter.trainer import TrainerConfig
from levanter.utils import fsspec_utils
from levanter.utils.mesh import MeshConfig

from experiments.paloma import paloma_tokenized
from marin.datakit.download.huggingface import DownloadConfig, download_hf
from marin.datakit.download.uncheatable_eval import make_uncheatable_eval_step
from marin.evaluation.evaluation_config import EvalTaskConfig, convert_to_levanter_task_config
from marin.execution.executor import unwrap_versioned_value
from marin.execution.remote import remote
from marin.execution.types import ExecutorStep, InputName, VersionedValue, ensure_versioned, this_output_path, versioned
from marin.processing.tokenize import (
    TokenizeConfig,
    TokenizerStep,
    add_validation_sets_to_mixture,
    lm_data_config,
    lm_mixture_data_config,
)
from marin.processing.tokenize.tokenize import HfDatasetSpec, HfTokenizeConfig, TokenizeConfigBase
from marin.training.training import (
    TrainDpoOnPodConfig,
    TrainLmOnPodConfig,
    run_levanter_train_dpo,
    run_levanter_train_lm,
)


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


@dataclasses.dataclass(frozen=True)
class SimpleTrainConfig:
    resources: ResourceConfig
    train_batch_size: int | IntSchedule
    """
    The batch size for training. If an IntSchedule is provided, the batch size will be
    varied according to the schedule.
    """
    num_train_steps: int
    learning_rate: float
    train_seq_len: int | None = None
    data_seed: int | None = None
    weight_decay: float | None = None
    beta1: float | None = None
    beta2: float | None = None
    epsilon: float | None = None
    max_grad_norm: float | None = None
    warmup: float | None = None
    decay: float | None = None
    rewarmup: float | None = None
    """
    The rewarmup parameter is used to re-warmup the learning rate after a decay cycles
    """
    lr_schedule: str | None = None
    min_lr_ratio: float | None = None
    cycle_length: int | list[int] | None = None
    z_loss_weight: float | None = None
    ema_beta: float | None = None
    """exponential moving average beta"""
    skip_bad_steps: bool = False
    """If True, skips steps where the loss or grad is significantly higher than the historical mean."""

    steps_per_eval: int | None = None
    """how often to run validation losses"""
    steps_per_export: int | None = None
    """How often to keep a permanent checkpoint. None (default) keeps only the final
    checkpoint; rolling temporary checkpoints are still written for resumption."""
    steps_per_task_eval: int | None = None
    """how often to run task evaluations"""
    steps_per_hf_export: int | None = None
    """None means match steps_per_export, -1 disables"""
    hf_generation_eos_token_ids: list[int] | None = None
    """EOS token IDs to write to generation_config.json. None means no generation config."""
    per_device_parallelism: int = -1
    """How many examples to process in parallel on each device. -1 (default) means
    train_batch_size/num_devices (no gradient accumulation). Set to a positive value
    to enable gradient accumulation."""
    per_device_eval_parallelism: int | None = None
    """Number of examples to evaluate in parallel on each device"""
    max_eval_batches: int | None = None
    """Maximum number of batches to evaluate on. None means all batches"""

    initialize_from_checkpoint_path: str | None = None
    """If set, the training will resume from the checkpoint at this path. Otherwise, training will start from scratch."""
    initialize_from_hf: str | None = None
    """If set, the training will start from the hf model at this path. Otherwise, training will start from scratch."""
    reset_data_loader_on_init: bool = True
    """Pairs with initialize_from_checkpoint_path. If True, initialize_from_checkpoint_path will reset the data loader
    so that it starts from step 0. Otherwise, it will resume from the step in the checkpoint."""

    allow_partial_checkpoint: bool = False
    """
    Allow loading partial checkpoints. This is useful for converting training to EMA, e.g.
    """

    int8: bool = False
    """Int8 (quantized) training in Levanter."""

    pad_tokenizer_to_match_model: bool = False
    """If True, pad the tokenizer's vocab to match the model's vocab size by adding dummy tokens.
    Useful when the model checkpoint has a larger vocab than the tokenizer (e.g., Qwen models
    pad their vocab to be divisible by 4 for TPU efficiency)."""

    optimizer_config: OptimizerConfig | None = None
    """Optimizer configuration to use. If not set, Adam will be used."""

    watch: WatchConfig = dataclasses.field(default_factory=WatchConfig)
    """Config for watching gradients, parameters, etc. Default is to log norms of gradients and parameters."""

    profiler: ProfilerConfig = dataclasses.field(default_factory=ProfilerConfig)
    """JAX profiler settings for training."""

    explicit_mesh_axes: bool = False
    """If True, build the device mesh with `AxisType.Explicit` axes.

    Required for models that call `jax.sharding.reshard(..., PartitionSpec(...))`.
    """

    tensor_parallel_size: int = 1
    """Size of the model (tensor parallel) axis. >1 shards model weights and activations
    across multiple devices. Useful when batch_size < num_chips."""

    env_vars: dict[str, str] | None = None
    """Environment variables to pass to the training task."""


# tasks to run (corresponding to lm_eval_harness tasks)
# subset from from page 43 of the DCLM paper: https://arxiv.org/pdf/2406.11794
# TODO: add more once supported in lm-eval-harness and/or tested on our end
CORE_TASKS = (
    EvalTaskConfig("agieval_lsat_ar", 3),  # 3-shot tests in legal domain
    EvalTaskConfig("arc_easy", 10),  # 10-shot, four-way MCQ questions involving grade 3-9 basic science
    EvalTaskConfig("arc_challenge", 10),  # a (harder) version of arc_easy
    EvalTaskConfig("boolq", 10),  # answer yes/no questions based on a passage
    EvalTaskConfig("commonsense_qa", 10),  # 5-way multiple-choice questions based on common-sense, everyday scenarios
    EvalTaskConfig("copa", 0),  # use causal reasoning to predict the correct outcome of a given scenario
    EvalTaskConfig("hellaswag", 0, task_alias="hellaswag_0shot"),  # 4-way multiple choice commonsense reasoning dataset
    EvalTaskConfig("hellaswag", 10, task_alias="hellaswag_10shot"),  # 4-way MCQ commonsense reasoning dataset
    EvalTaskConfig("lambada_openai", 0),  # predict the endings of text passages
    EvalTaskConfig("openbookqa", 0),  # 4-way multiple choice question answering task that requires multi-step reasoning
    EvalTaskConfig("piqa", 10),  # answer questions based on a passage
    # (requires generation which is not supported in Levanter at the moment)
    # EvalTaskConfig("squadv2", 10),  # reading comprehension benchmark
    EvalTaskConfig("wsc273", 0),  # Winograd Schema Challenge
    EvalTaskConfig("winogrande", 0),  # Winograd challenge, extended to more domains
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
                project="marin",
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
            f"= {total_examples * (train_length or 0)} tokens."
        ),
        fn=run_levanter_train_lm,
        resources=train_config.resources,
        config=config,
        override_output_path=override_output_path,
    )


@dataclass(frozen=True)
class SimpleSFTConfig:
    """
    A simplified configuration for Supervised Fine-Tuning (SFT) that works for both
    single dataset and mixture training approaches.
    """

    # Hardware configuration
    resources: ResourceConfig

    # Core training parameters
    train_batch_size: int | IntSchedule = 128
    """
    The batch size for training. If an IntSchedule is provided, the batch size will be
    varied according to the schedule.
    """
    num_train_steps: int = 10000
    """Number of training steps."""

    learning_rate: float = 5e-6
    """Learning rate for the optimizer."""

    # Model configuration
    tokenizer: str | None = None
    """Tokenizer to use for training."""

    initialize_from_hf: str | None = None
    """HF model name or path to initialize from (e.g., 'meta-llama/Llama-3.1-8B').
    Mutually exclusive with initialize_from_checkpoint_path."""

    initialize_from_checkpoint_path: str | None = None
    """Path to a levanter checkpoint to initialize from.
    Mutually exclusive with initialize_from_hf."""

    max_seq_len: int = 4096
    """Maximum sequence length for training."""

    # Optimizer parameters
    weight_decay: float = 0.0
    """Weight decay for the optimizer."""

    beta1: float | None = None
    """AdamW optimizer beta1."""

    beta2: float | None = None
    """AdamW optimizer beta2."""

    warmup: float = 0.03
    """Fraction of training steps to use for learning rate warmup."""

    decay: float = 0.0
    """Fraction of training steps to use for learning rate decay."""

    lr_schedule: str = "linear"
    """Learning rate schedule to use: 'linear', 'cosine', etc."""

    min_lr_ratio: float = 0.0
    """Minimum learning rate as a ratio of the base learning rate."""

    max_grad_norm: float | None = None
    """Maximum gradient norm for gradient clipping."""

    # Checkpointing and evaluation
    steps_per_eval: int = 1000
    """How often to run validation losses."""

    steps_per_checkpoint: int | None = None
    """How often to keep a permanent checkpoint. None (default) keeps only the final
    checkpoint; rolling temporary checkpoints are still written for resumption."""

    steps_per_hf_export: int = 500
    """How often to save HuggingFace checkpoints."""

    hf_generation_eos_token_ids: list[int] | None = None
    """EOS token IDs to write to generation_config.json. None means no generation config.
    For chat models, include the turn-boundary token (e.g. [128001, 128009])."""

    # Mixture-specific parameters
    mixture_block_size: int = 2048
    """Block size for dataset mixing (only used with mixture training)."""

    stop_strategy: str = "restart"
    """
    Strategy for handling dataset completion (only used with mixture training).
    Options: 'restart' or 'exit'.
    """

    # Other parameters
    seed: int = 0
    """Random seed for training."""

    node_count: int = 1
    """Number of TPU slices for training."""

    int8: bool = False
    """Int8 (quantized) training in Levanter."""

    pad_tokenizer_to_match_model: bool = False
    """If True, pad the tokenizer's vocab to match the model's vocab size by adding dummy tokens.
    Useful when the model checkpoint has a larger vocab than the tokenizer (e.g., Qwen models
    pad their vocab to be divisible by 4 for TPU efficiency)."""

    z_loss_weight: float = 0.0

    per_device_parallelism: int = -1
    """How many examples to process in parallel on each device. -1 (default) means
    train_batch_size/num_devices (no gradient accumulation). Set to a positive value
    to enable gradient accumulation. For example, with 8 devices, batch_size=32, and
    per_device_parallelism=1, you get gradient accumulation of 4."""

    reinit_tokens: list[str] | bool = False
    """
    if set, will reinitialize the embeddings for the given tokens. If True, will reinitialize the default tokens
    for llama3's tokenizer
    """


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


@dataclass(frozen=True)
class SimpleDPOConfig:
    """
    A simplified configuration for Direct Preference Optimization (DPO).
    """

    resources: ResourceConfig

    train_batch_size: int | IntSchedule = 128
    num_train_steps: int | None = None
    num_epochs: float = 1.0
    """Approximate number of passes over the DPO train set when num_train_steps is unset."""
    learning_rate: float = 1e-6
    wandb_project: str | None = None

    tokenizer: str | None = None
    model_name_or_path: str | None = None
    initialize_from_checkpoint_path: str | None = None

    adapter: AdaptorConfig = dataclasses.field(default_factory=NoAdaptorConfig)
    reference: DpoReferenceConfig = dataclasses.field(default_factory=SeparateReferenceConfig)
    reference_model_path: str | None = None
    reference_is_hf: bool = True
    beta: float = 0.1
    validation_split_fraction: float | None = 0.1
    reference_eval_cache: ReferenceEvalCacheConfig = dataclasses.field(
        default_factory=lambda: ReferenceEvalCacheConfig(mode="build_or_load")
    )

    train_seq_len: int | None = None
    max_seq_len: int = 4096

    weight_decay: float = 0.0
    warmup: float = 0.0
    cooldown: float | None = None
    lr_schedule: str = "linear"
    min_lr_ratio: float = 0.0
    max_grad_norm: float | None = 1

    steps_per_eval: int | None = None
    """None auto-schedules validation five times: before training, three interior points, and at the end."""
    steps_per_checkpoint: int | None = None
    """How often to keep a permanent checkpoint. None (default) keeps only the final
    checkpoint; rolling temporary checkpoints are still written for resumption."""
    steps_per_hf_export: int = 500
    hf_save_dtype: str | None = None
    hf_generation_eos_token_ids: list[int] | None = None
    """EOS token IDs to write to generation_config.json. None means no generation config.
    For chat models, include the turn-boundary token (e.g. [128001, 128009])."""

    per_device_eval_parallelism: int = -1

    seed: int = 0
    initialize_from_hf: bool | None = None

    profiler: ProfilerConfig = dataclasses.field(default_factory=ProfilerConfig)

    allow_partial_checkpoint: bool = False
    int8: bool = False

    def __post_init__(self):
        if self.num_train_steps is not None and self.num_train_steps <= 0:
            raise ValueError(f"num_train_steps must be positive, got {self.num_train_steps}")
        if self.num_epochs <= 0:
            raise ValueError(f"num_epochs must be positive, got {self.num_epochs}")
        if self.steps_per_eval is not None and self.steps_per_eval <= 0:
            raise ValueError(f"steps_per_eval must be positive, got {self.steps_per_eval}")
        if self.steps_per_checkpoint is not None and self.steps_per_checkpoint <= 0:
            raise ValueError(f"steps_per_checkpoint must be positive, got {self.steps_per_checkpoint}")


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


uncheatable_eval = make_uncheatable_eval_step()


def uncheatable_eval_tokenized(
    *, base_path="tokenized/", tokenizer: str | None = None, uncheatable_eval_raw: ExecutorStep = uncheatable_eval
) -> dict[str, TokenizerStep]:
    uncheatable_eval_steps: dict[str, ExecutorStep[TokenizeConfig]] = {}
    for dataset in ACTIVE_DATASETS:
        path_part = ALL_UNCHEATABLE_EVAL_DATASETS[dataset]
        uncheatable_eval_steps[os.path.join("uncheatable_eval", dataset)] = default_tokenize(
            name=os.path.join("uncheatable_eval", dataset),
            dataset=uncheatable_eval_raw.cd(f"{path_part}"),
            tokenizer=tokenizer,
            is_validation=True,
        )

    return uncheatable_eval_steps


ALL_UNCHEATABLE_EVAL_DATASETS = {
    "wikipedia_arabic": "wikipedia_arabic_*.jsonl.gz",
    "wikipedia_english": "wikipedia_english_*.jsonl.gz",
    "wikipedia_french": "wikipedia_french_*.jsonl.gz",
    "wikipedia_german": "wikipedia_german_*.jsonl.gz",
    "wikipedia_japanese": "wikipedia_japanese_*.jsonl.gz",
    "wikipedia_spanish": "wikipedia_spanish_*.jsonl.gz",
    "github_python": "github_python_*.jsonl.gz",
    "github_cpp": "github_cpp_*.jsonl.gz",
    "bbc_news": "bbc_news_*.jsonl.gz",
    "arxiv_physics": "arxiv_physics_*.jsonl.gz",
    "arxiv_computer_science": "arxiv_computer_science_*.jsonl.gz",
    "ao3_chinese": "ao3_chinese_*.jsonl.gz",
    "ao3_english": "ao3_english_*.jsonl.gz",
}
ACTIVE_DATASETS = [
    "wikipedia_english",
    "github_python",
    "github_cpp",
    "bbc_news",
    "arxiv_physics",
    "arxiv_computer_science",
    "ao3_english",
]


def compute_per_device_parallelism(
    global_batch_size: int,
    microbatch_size: int,
    resources: ResourceConfig,
) -> int:
    """Compute per_device_parallelism for gradient accumulation.

    Args:
        global_batch_size: The effective batch size after gradient accumulation.
        microbatch_size: The batch size that fits in memory (local batch size).
        resources: The ResourceConfig specifying TPU/GPU resources.

    Returns:
        per_device_parallelism: Number of examples each device processes per forward/backward pass.

    Example:
        For v5p-8 (4 chips), global_batch_size=128, microbatch_size=8:
        - per_device_parallelism = 8 / 4 = 2
        - gradient_accumulation = 128 / 8 = 16 steps
    """
    num_devices = resources.chip_count()

    if microbatch_size % num_devices != 0:
        raise ValueError(f"microbatch_size ({microbatch_size}) must be divisible by " f"num_devices ({num_devices})")

    if global_batch_size % microbatch_size != 0:
        raise ValueError(
            f"global_batch_size ({global_batch_size}) must be divisible by " f"microbatch_size ({microbatch_size})"
        )

    per_device_parallelism = microbatch_size // num_devices
    grad_accum_steps = global_batch_size // microbatch_size

    print(
        f"Gradient accumulation config: "
        f"global_batch={global_batch_size}, microbatch={microbatch_size}, "
        f"num_devices={num_devices}, per_device_parallelism={per_device_parallelism}, "
        f"grad_accum_steps={grad_accum_steps}"
    )

    return per_device_parallelism


HF_BUCKET_URI_PREFIX = "hf://buckets/"
HF_BUCKET_PATH_PREFIX = "buckets/"


def _is_hf_bucket_path(path: str) -> bool:
    return path.startswith(HF_BUCKET_URI_PREFIX) or path.startswith(HF_BUCKET_PATH_PREFIX)


def _normalize_hf_bucket_path(path: str) -> str:
    if path.startswith(HF_BUCKET_URI_PREFIX):
        return path.removeprefix("hf://")
    return path


def default_download(
    name: str,
    hf_dataset_id: str,
    revision: str | None = None,
    override_output_path: str | None = None,
    **kwargs: Any,
) -> InputName:
    """
    Download a HuggingFace dataset and upload it to a specified path with default configuration.

    Args:
        name: The name of the Download step. It forms the basis of the output path
            unless override_output_path is explicitly specified.
        hf_dataset_id: Hugging Face source. Either `$ORG/$DATASET` on HF Hub or `hf://buckets/...`.
        revision: The revision of the dataset to download for Hub datasets.
            Optional for bucket paths.
        override_output_path: Optional. The output path for the dataset.
        **kwargs: Additional keyword arguments that are passed to the download config.

    The final output data will reside in '{output_path}/{revision}'.
    """

    download_kwargs = dict(kwargs)
    hf_repo_type_prefix = download_kwargs.pop("hf_repo_type_prefix", None)
    if _is_hf_bucket_path(hf_dataset_id):
        normalized_dataset_id = _normalize_hf_bucket_path(hf_dataset_id)
        description = f"Download {hf_dataset_id}"
        resolved_hf_repo_type_prefix = "" if hf_repo_type_prefix is None else hf_repo_type_prefix
        resolved_revision = "main" if revision is None else revision
    else:
        if revision is None:
            raise ValueError("revision is required for non-bucket Hugging Face dataset downloads.")
        normalized_dataset_id = hf_dataset_id
        description = f"Download {hf_dataset_id} revision {revision}"
        resolved_hf_repo_type_prefix = "datasets" if hf_repo_type_prefix is None else hf_repo_type_prefix
        resolved_revision = revision

    step = ExecutorStep(
        name=name,
        description=description,
        fn=download_hf,
        config=DownloadConfig(
            hf_dataset_id=normalized_dataset_id,
            revision=resolved_revision,
            gcs_output_path=this_output_path(),
            wait_for_completion=True,
            hf_repo_type_prefix=resolved_hf_repo_type_prefix,
            **download_kwargs,
        ),
        override_output_path=override_output_path,
    )

    return step.as_input_name()


def default_tokenize(
    name: str,
    dataset: InputName | ExecutorStep | str | HfDatasetSpec,
    tokenizer: str,
    format: LmDatasetFormatBase = TextLmDatasetFormat(),  # noqa
    *,
    sample_count: int | VersionedValue[int] | None = None,
    is_validation: bool = False,
    levanter_batch_size: int | None = None,
    tags: Sequence[str] = (),
    resources: ResourceConfig | None = None,
    worker_resources: ResourceConfig | None = None,
) -> ExecutorStep:
    """
    Tokenizes a dataset using the specified tokenizer and Levanter's tokenization infrastructure.

    Args:
        name: The name of the tokenized dataset. This is used to form the output path for the executor step.
            `tokenized/` will be prepended to the name.
        dataset:  The dataset to tokenize. This can be an InputName, ExecutorStep, a string as a
            path to the dataset or a HuggingFace dataset ID, or ``HfDatasetSpec`` to specify a
            dataset with a particular subset name.
        tokenizer: string HuggingFace tokenizer name. Should be the same as you intend to use in the tokenizer
            spec for the training run.
        format: The format of the dataset. This is used to determine how to tokenize the data.

            See [Levanter's documentation](https://levanter.readthedocs.io/en/latest/reference/Data-Formats/)
            for more details.
        sample_count: Optional limit on the number of samples to tokenize per shard. If ``None``, tokenize everything.
        is_validation: Whether the dataset is a validation set. Doesn't do anything for HF datasets.
        tags: Tags to attach to the Levanter dataset source for tagged evaluation.
    Returns:
        An ExecutorStep that represents the tokenized dataset.
    """

    # Common kwargs for config constructors
    extra_kwargs: dict = {}
    if worker_resources is not None:
        extra_kwargs["worker_resources"] = worker_resources

    # sniff out if it's a HuggingFace dataset
    if isinstance(dataset, HfDatasetSpec):
        config = HfTokenizeConfig(
            id=dataset.id,
            name=dataset.name,
            cache_path=this_output_path(),
            tokenizer=ensure_versioned(tokenizer),
            format=format,
            sample_count=ensure_versioned(sample_count) if sample_count is not None else None,
            levanter_batch_size=levanter_batch_size,
            tags=[*tags],
            **extra_kwargs,
        )
    elif (
        isinstance(dataset, str)
        and not _is_hf_bucket_path(dataset)
        and dataset.count("/") == 1
        and not fsspec_utils.exists(dataset)
    ):
        config = HfTokenizeConfig(
            id=dataset,
            cache_path=this_output_path(),
            tokenizer=ensure_versioned(tokenizer),
            format=format,
            sample_count=ensure_versioned(sample_count) if sample_count is not None else None,
            levanter_batch_size=levanter_batch_size,
            tags=[*tags],
            **extra_kwargs,
        )
    else:
        config = TokenizeConfig(
            train_paths=[dataset] if not is_validation else [],
            validation_paths=[dataset] if is_validation else [],
            cache_path=this_output_path(),
            tokenizer=ensure_versioned(tokenizer),
            format=format,
            sample_count=ensure_versioned(sample_count) if sample_count is not None else None,
            levanter_batch_size=levanter_batch_size,
            tags=[*tags],
            **extra_kwargs,
        )

    return ExecutorStep(
        name=os.path.join("tokenized", name),
        description=f"Tokenize raw text using the {tokenizer} tokenizer.",
        fn=remote(
            tokenizer,
            resources=resources or ResourceConfig.with_cpu(cpu=4, ram="16g", disk="10g"),
            pip_dependency_groups=["cpu"],
            env_vars={
                "TRANSFORMERS_NO_TORCH": "1",
                "TRANSFORMERS_NO_TORCHVISION": "1",
                "USE_TORCH": "0",
                "TORCH_DISABLE_GLOBAL_DEPS": "1",
            },
        ),
        config=config,
    )
