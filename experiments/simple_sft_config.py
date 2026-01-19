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

from dataclasses import dataclass

from levanter.schedule import IntSchedule

from fray.cluster import ResourceConfig


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
        raise ValueError(
            f"microbatch_size ({microbatch_size}) must be divisible by "
            f"num_devices ({num_devices})"
        )

    if global_batch_size % microbatch_size != 0:
        raise ValueError(
            f"global_batch_size ({global_batch_size}) must be divisible by "
            f"microbatch_size ({microbatch_size})"
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

    model_name_or_path: str | None = None
    """Path to the pretrained HF model checkpoint to initialize from"""

    initialize_from_checkpoint_path: str | None = None
    """Path to a levanter checkpoint to initialize from."""

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

    steps_per_checkpoint: int = 1000
    """How often to save checkpoints."""

    steps_per_hf_export: int = 500
    """How often to save HuggingFace checkpoints."""

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

    initialize_from_hf: bool | None = None
    """Whether to initialize from HuggingFace model.
    If false, we will load a levanter checkpoint. None defaults to True if
    model_name_or_path is set and initialize_from_checkpoint_path is not set."""

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
    train_batch_size/num_devices (no gradient accumulation). Set to a smaller value
    to enable gradient accumulation. For example, with 8 devices, batch_size=32, and
    per_device_parallelism=1, you get gradient accumulation of 4."""

    reinit_tokens: list[str] | bool = False
    """
    if set, will reinitialize the embeddings for the given tokens. If True, will reinitialize the default tokens
    for llama3's tokenizer
    """
