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

"""Speedruns using the AdamH optimizer for various Llama model sizes (Chinchilla optimal steps)."""

import dataclasses
import logging
import os
from levanter.optim import AdamHConfig
from experiments.simple_train_config import SimpleTrainConfig
from marin.execution.executor import executor_main
from marin.resources import TpuPodConfig
from marin.speedrun.speedrun import Author, SpeedrunConfig, default_speedrun
from experiments.speedrun.adamh_llama_scaling.llama_with_hybrid_norm import (
    llama_150m_all_norm,
    llama_300m_all_norm,
    llama_600m_all_norm,
    llama_1_4b_all_norm,
)

AUTHOR = Author(name="Kaiyue Wen", affiliation="Stanford University", url="https://whenwen.github.io")

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
        "130m": llama_150m_all_norm,
        "300m": llama_300m_all_norm,
        "520m": llama_600m_all_norm,
        "1_2b": llama_1_4b_all_norm,
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
        "130m": TpuPodConfig(tpu_type="v5litepod-64"),
        "300m": TpuPodConfig(tpu_type="v5litepod-64"),
        "520m": TpuPodConfig(tpu_type="v5litepod-64"),
        "1_2b": TpuPodConfig(tpu_type="v5litepod-64"),
    }

    # AdamH optimizer configs for each size
    adam_configs = {
        "130m": AdamHConfig(
            learning_rate=0.02,
            adam_lr=0.008,
            min_lr_ratio=0,
            warmup=1000,
            beta1=0.9,
            beta2=0.98,
            epsilon=1e-20,
            max_grad_norm=1,
            nesterov=False,
            power_of_schedule=1.0,
        ),
        "300m": AdamHConfig(
            learning_rate=0.02,
            adam_lr=0.008,
            min_lr_ratio=0,
            warmup=1000,
            beta1=0.9,
            beta2=0.98,
            epsilon=1e-10,
            max_grad_norm=1,
            nesterov=False,
        ),
        "520m": AdamHConfig(
            learning_rate=0.02,
            adam_lr=0.004,
            min_lr_ratio=0,
            warmup=1000,
            beta1=0.9,
            beta2=0.98,
            epsilon=1e-10,
            max_grad_norm=1,
            nesterov=False,
        ),
        "1_2b": AdamHConfig(
            learning_rate=0.015,
            adam_lr=0.0015,
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
    format_str = (
        "{size} parameter model (basically fully scale invariant) "
        "trained with the AdamH optimizer to maintain constant norm."
    )
    descriptions = {
        "130m": format_str.format(size="130M"),
        "300m": format_str.format(size="300M"),
        "520m": format_str.format(size="520M"),
        "1_2b": format_str.format(size="1.2B"),
    }

    # Names for the runs
    run_names = {
        "130m": "llama_130m_adamh_lr0.02_adam_lr0.008_warmup1000_qk",
        "300m": "llama_300m_adamh_lr0.02_adam_lr0.008_warmup1000_qk",
        "520m": "llama_520m_adamh_lr0.02_adam_lr0.004_warmup1000_qk",
        "1_2b": "llama_1_2b_adamh_lr0.015_adam_lr0.0015_warmup1000_qk",
    }

    # Gather config for the requested size
    if size not in param_counts:
        raise ValueError(f"Unknown size: {size}")

    param_count = param_counts[size]
    batch_size = batch_sizes[size]
    model_config = dataclasses.replace(model_cfgs[size], max_seq_len=4096)
    max_seq_len = model_config.max_seq_len
    resource_config = resource_cfgs[size]
    adam = adam_configs[size]
    description = descriptions[size]
    run_name = run_names[size]

    num_train_steps = get_num_train_steps(param_count, batch_size, max_seq_len)

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


def main():
    if os.getenv("CI", None) is not None:
        logger.info("Skipping experiment execution on CI environment, needs HF access.")
        return

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

    executor_main(steps=steps, description="AdamH speedruns (Chinchilla optimal)")


if __name__ == "__main__":
    main()
