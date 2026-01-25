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

from fray.cluster import ResourceConfig
from levanter.schedule import IntSchedule


@dataclass(frozen=True)
class SimpleSimPOConfig:
    """
    A simplified configuration for Simple Preference Optimization (SimPO).
    """

    resources: ResourceConfig

    train_batch_size: int | IntSchedule = 128
    num_train_steps: int = 10000
    learning_rate: float = 6e-7
    wandb_project: str | None = None

    tokenizer: str | None = None
    model_name_or_path: str | None = None
    initialize_from_checkpoint_path: str | None = None

    beta: float = 2.0
    gamma_beta_ratio: float = 0.5
    validation_split_fraction: float | None = 0.1

    train_seq_len: int | None = None
    max_seq_len: int = 4096

    weight_decay: float = 0.0
    warmup: float = 0.03
    cooldown: float | None = None
    lr_schedule: str = "linear"
    min_lr_ratio: float = 0.0
    max_grad_norm: float | None = None

    steps_per_eval: int = 1000
    steps_per_checkpoint: int = 1000
    steps_per_hf_export: int = 500
    hf_save_dtype: str | None = None

    seed: int = 0
    initialize_from_hf: bool | None = None

    allow_partial_checkpoint: bool = False
    int8: bool = False
