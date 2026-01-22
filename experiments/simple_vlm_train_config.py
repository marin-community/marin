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

import dataclasses
from dataclasses import dataclass

from fray.cluster import ResourceConfig
from levanter.callbacks.watch import WatchConfig
from levanter.optim import OptimizerConfig
from levanter.schedule import IntSchedule


@dataclass(frozen=True)
class SimpleVlmTrainConfig:
    """
    A simplified configuration for VLM (Vision-Language Model) training.

    This provides a user-friendly interface for distributed VLM training,
    similar to SimpleTrainConfig for LLM training.
    """

    # Hardware configuration
    resources: ResourceConfig
    """Resource configuration for TPU/GPU allocation. Use ResourceConfig.with_tpu('v4-128', slice_count=4)."""

    # Core training parameters
    train_batch_size: int | IntSchedule
    """
    The batch size for training. If an IntSchedule is provided, the batch size will be
    varied according to the schedule.
    """
    num_train_steps: int
    """Total number of training steps."""
    learning_rate: float
    """Peak learning rate for training."""

    per_device_parallelism: int = -1
    """
    How many examples to process in parallel on each device.
    -1 (default) means train_batch_size/num_devices (no gradient accumulation).
    Set to a smaller value to enable gradient accumulation.
    For example, with 64 TPU chips, train_batch_size=256, per_device_parallelism=1:
      - Each device processes 1 example at a time
      - Gradient accumulation steps = 256 / (1 * 64) = 4
    """

    # Sequence and data parameters
    train_seq_len: int | None = None
    """Maximum sequence length for training. If None, uses model's max_seq_len."""
    data_seed: int | None = None
    """Seed for data shuffling. If None, uses trainer seed."""

    # VLM-specific parameters
    processor: str | None = None
    """HuggingFace processor path for image preprocessing (e.g., 'llava-hf/llava-onevision-qwen2-0.5b-ov-hf')."""
    vlm_checkpoint: str | None = None
    """Complete VLM HuggingFace checkpoint path (loads vision encoder + projector + LLM).
    Use this for loading from a previously trained VLM, e.g., for stage 2 training."""
    vision_checkpoint: str | None = None
    """HuggingFace checkpoint for vision encoder only (e.g., 'google/siglip-so400m-patch14-384').
    Use this with llm_checkpoint for loading separate vision and LLM weights."""
    llm_checkpoint: str | None = None
    """HuggingFace checkpoint for language model (e.g., 'Qwen/Qwen3-1.7B')."""
    freeze_vision_encoder: bool = False
    """If True, freeze vision encoder weights (only train projector + LLM)."""
    freeze_llm: bool = False
    """If True, freeze LLM weights (only train projector + vision encoder)."""

    # Optimizer parameters
    weight_decay: float | None = None
    beta1: float | None = None
    beta2: float | None = None
    epsilon: float | None = None
    max_grad_norm: float | None = None
    warmup: float | None = None
    decay: float | None = None
    rewarmup: float | None = None
    """The rewarmup parameter is used to re-warmup the learning rate after decay cycles."""
    lr_schedule: str | None = None
    min_lr_ratio: float | None = None
    cycle_length: int | list[int] | None = None
    z_loss_weight: float | None = None
    skip_bad_steps: bool = False
    """If True, skips steps where the loss or grad is significantly higher than the historical mean."""

    # Evaluation parameters
    steps_per_eval: int | None = None
    """How often to run validation losses."""
    no_eval: bool = False
    """If True, disable evaluation completely to save memory."""
    max_eval_batches: int | None = None
    """Maximum number of batches to evaluate on. None means all batches."""

    # Checkpoint parameters
    steps_per_export: int = 10000
    """How often to save Levanter checkpoints."""
    steps_per_hf_export: int | None = None
    """How often to save HuggingFace checkpoints. None means match steps_per_export, -1 disables."""

    # Initialization parameters
    initialize_from_checkpoint_path: str | None = None
    """If set, training will resume from this Levanter checkpoint path."""
    initialize_from_hf: str | None = None
    """If set, training will start from this HuggingFace model path."""
    use_hf_model_config: bool = False
    """If True, replace model config with HF config from checkpoint."""
    reset_data_loader_on_init: bool = True
    """If True, data loader starts from step 0 when initializing from checkpoint."""
    allow_partial_checkpoint: bool = False
    """Allow loading partial checkpoints (useful for EMA conversion)."""

    # Training control
    epoch: int = 0
    """Number of epochs to train. 0 means train until num_train_steps is reached."""

    # Misc
    mp: str = "p=f32,c=bfloat16"
    """Mixed precision policy string (e.g., 'p=f32,c=bfloat16' for params in f32, compute in bfloat16)."""

    # Streaming mode performance tuning
    streaming_max_buffered_batches: int = 8
    """Maximum buffered batches in streaming mode. Increase for better throughput (default: 8)."""
    streaming_prefetch_size: int = 4
    """Prefetch size in streaming mode. Increase for better throughput (default: 4)."""

    int8: bool = False
    """Int8 (quantized) training in Levanter."""

    optimizer_config: OptimizerConfig | None = None
    """Optimizer configuration. If not set, Adam will be used."""

    watch: WatchConfig = dataclasses.field(default_factory=WatchConfig)
    """Config for watching gradients, parameters, etc."""

    # Profiler
    profiler: bool = False
    """Whether to run the JAX profiler during training."""
    profiler_start_step: int = 5
    """Which step to start profiling."""
    profiler_num_steps: int = 100
    """How many steps to profile for."""

    explicit_mesh_axes: bool = False
    """If True, build device mesh with AxisType.Explicit axes."""
