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

"""Speedruns using the AdamW optimizer for various Llama model sizes (Chinchilla optimal steps).

Optimizer configs were searched & provided by Kaiyue Wen in https://wandb.ai/marin-community/marin/reports/Fantastic-Optimizers-and-Where-to-Find-Them--VmlldzoxMjgzMzQ2NQ
"""

import dataclasses
import logging

from levanter.optim import AdamConfig

from experiments.llama import llama_1_4b, llama_150m, llama_300m, llama_600m
from experiments.simple_train_config import SimpleTrainConfig
from marin.execution.executor import executor_main
from marin.resources import TpuPodConfig
from marin.speedrun.speedrun import Author, SpeedrunConfig, default_speedrun

AUTHOR = Author(name="William Held", affiliation="Georgia Tech", url="https://WilliamHeld.com")

logger = logging.getLogger("ray")


def get_num_train_steps(param_count, batch_size, max_seq_len):
    """Compute the number of steps for Chinchilla optimal training (20x params tokens)."""
    total_tokens = param_count * 20
    tokens_per_step = batch_size * max_seq_len
    return total_tokens // tokens_per_step


def build_config(size: str) -> tuple[str, SpeedrunConfig]:
    # Parameter counts
    param_counts = {
        "130m": 130_000_000,
        "300m": 300_000_000,
        "520m": 520_000_000,
        "1_2b": 1_200_000_000,
    }

    # Model configs
    model_cfgs = {
        "130m": llama_150m,
        "300m": llama_300m,
        "520m": llama_600m,
        "1_2b": llama_1_4b,
    }

    # Training batch sizes
    batch_sizes = {
        "130m": 128,
        "300m": 128,
        "520m": 256,
        "1_2b": 256,
    }

    # Resource configs
    resource_cfgs = {
        "130m": TpuPodConfig(tpu_type="v5p-32"),
        "300m": TpuPodConfig(tpu_type="v5p-32"),
        "520m": TpuPodConfig(tpu_type="v5p-32"),
        "1_2b": TpuPodConfig(tpu_type="v5p-32"),
    }

    # AdamW optimizer configs for each size
    adam_configs = {
        "130m": AdamConfig(
            learning_rate=0.008,
            weight_decay=0.1,
            min_lr_ratio=0,
            warmup=2000,
            beta1=0.9,
            beta2=0.98,
            epsilon=1e-20,
            max_grad_norm=1,
            nesterov=False,
        ),
        "300m": AdamConfig(
            learning_rate=0.008,
            weight_decay=0.1,
            min_lr_ratio=0,
            warmup=2000,
            beta1=0.9,
            beta2=0.98,
            epsilon=1e-10,
            max_grad_norm=1,
            nesterov=False,
        ),
        "520m": AdamConfig(
            learning_rate=0.004,
            weight_decay=0.2,
            min_lr_ratio=0,
            warmup=1000,
            beta1=0.9,
            beta2=0.98,
            epsilon=1e-10,
            max_grad_norm=1,
            nesterov=False,
        ),
        "1_2b": AdamConfig(
            learning_rate=0.002,
            weight_decay=0.2,
            min_lr_ratio=0,
            warmup=1000,
            beta1=0.9,
            beta2=0.98,
            epsilon=1e-25,
            max_grad_norm=2,
            nesterov=False,
        ),
    }

    # Descriptions
    descriptions = {
        "130m": "130M parameter model trained with the AdamW optimizer.",
        "300m": "300M parameter model trained with the AdamW optimizer.",
        "520m": "520M parameter model trained with the AdamW optimizer.",
        "1_2b": "1.2B parameter model trained with the AdamW optimizer.",
    }

    # Names for the runs
    run_names = {
        "130m": "llama_130m_adamw_4096",
        "300m": "llama_300m_adamw_4096",
        "520m": "llama_520m_adamw_4096",
        "1_2b": "llama_1_2b_adamw_fix_4096",
    }

    # Gather config for the requested size
    if size not in param_counts:
        raise ValueError(f"Unknown size: {size}")

    param_count = param_counts[size]
    batch_size = batch_sizes[size]
    model_config = dataclasses.replace(model_cfgs[size], max_seq_len=4096)
    train_seq_len = model_config.max_seq_len
    resource_config = resource_cfgs[size]
    adam = adam_configs[size]
    description = descriptions[size]
    run_name = run_names[size]

    num_train_steps = get_num_train_steps(param_count, batch_size, train_seq_len)

    train = SimpleTrainConfig(
        resource_config,
        train_batch_size=batch_size,
        num_train_steps=num_train_steps,
        learning_rate=adam.learning_rate,
        optimizer_config=adam,
    )
    cfg = SpeedrunConfig(
        author=AUTHOR,
        description=description,
        model_config=model_config,
        train_config=train,
    )
    return run_name, cfg


if __name__ == "__main__":
    runs = [
        build_config("130m"),
        build_config("300m"),
        build_config("520m"),
        build_config("1_2b"),
    ]

    steps = []
    for name, cfg in runs:
        cfg.print_run_info()
        steps.extend(default_speedrun(name, cfg))

    executor_main(steps=steps, description="AdamW speedruns (Chinchilla optimal)")
