# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field

from fray.cluster import ResourceConfig
from levanter.callbacks.profiler import ProfilerConfig
from levanter.schedule import IntSchedule

# DPO runs two models (policy + reference) but eval doesn't need gradients/optimizer,
# so we can fit more examples per device during eval than training.
# Keyed by TPU variant string from ResourceConfig.
DPO_EVAL_PARALLELISM: dict[str, int] = {
    "v5p-8": 16,
    "v5p-16": 16,
    "v5p-32": 32,
    "v5p-64": 32,
    "v5p-128": 32,
    "v5p-256": 64,
}


@dataclass(frozen=True)
class SimpleDPOConfig:
    """
    A simplified configuration for Direct Preference Optimization (DPO).
    """

    resources: ResourceConfig

    train_batch_size: int | IntSchedule = 128
    num_train_steps: int = 10000
    learning_rate: float = 1e-6
    wandb_project: str | None = None

    tokenizer: str | None = None
    model_name_or_path: str | None = None
    initialize_from_checkpoint_path: str | None = None

    reference_model_path: str | None = None
    reference_is_hf: bool = True
    beta: float = 0.1
    validation_split_fraction: float | None = 0.1

    train_seq_len: int | None = None
    max_seq_len: int = 4096

    weight_decay: float = 0.0
    warmup: float = 0.0
    cooldown: float | None = None
    lr_schedule: str = "linear"
    min_lr_ratio: float = 0.0
    max_grad_norm: float | None = 1

    steps_per_eval: int = 1000
    steps_per_checkpoint: int = 1000
    steps_per_hf_export: int = 500
    hf_save_dtype: str | None = None
    hf_generation_eos_token_ids: list[int] | None = None
    """EOS token IDs to write to generation_config.json. None means no generation config.
    For chat models, include the turn-boundary token (e.g. [128001, 128009])."""

    per_device_eval_parallelism: int = -1

    seed: int = 0
    initialize_from_hf: bool | None = None

    profiler: ProfilerConfig = field(default_factory=ProfilerConfig)

    allow_partial_checkpoint: bool = False
    int8: bool = False
