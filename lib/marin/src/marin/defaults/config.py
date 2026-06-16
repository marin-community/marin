# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses
from dataclasses import dataclass

from fray import ResourceConfig
from levanter.adaptor import AdaptorConfig, NoAdaptorConfig
from levanter.callbacks.profiler import ProfilerConfig
from levanter.callbacks.watch import WatchConfig
from levanter.dpo import ReferenceEvalCacheConfig
from levanter.main.train_dpo import DpoReferenceConfig, SeparateReferenceConfig
from levanter.optim import OptimizerConfig
from levanter.schedule import IntSchedule


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
