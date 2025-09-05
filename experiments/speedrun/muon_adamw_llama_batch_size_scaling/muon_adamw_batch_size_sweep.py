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

"""Speedruns using the AdamW/Muon optimizer for various Llama model sizes (Chinchilla optimal steps) and batch size.

Optimizer configs were searched & provided by Kaiyue Wen in https://wandb.ai/marin-community/marin/reports/Fantastic-Optimizers-and-Where-to-Find-Them--VmlldzoxMjgzMzQ2NQ
"""

import dataclasses
import logging

from levanter.optim import AdamConfig, MuonConfig

from experiments.llama import llama_1_4b, llama_150m, llama_300m, llama_600m
from experiments.simple_train_config import SimpleTrainConfig
from marin.execution.executor import executor_main
from marin.resources import TpuPodConfig
from marin.speedrun.speedrun import Author, SpeedrunConfig, default_speedrun

AUTHOR = Author(name="Franz Cesista", affiliation="", url="https://leloykun.github.io")

logger = logging.getLogger("ray")


def get_num_train_steps(param_count, batch_size, seq_len):
    """Compute the number of steps for Chinchilla optimal training (20x params tokens)."""
    total_tokens = param_count * 20
    tokens_per_step = batch_size * seq_len
    return total_tokens // tokens_per_step


def build_config(optimizer_name: str, size: str, batch_size: int, seq_len: int = 4096) -> tuple[str, SpeedrunConfig]:
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

    # Resource configs
    resource_cfgs = {
        "130m": TpuPodConfig(tpu_type="v5p-32"),
        "300m": TpuPodConfig(tpu_type="v5p-32"),
        "520m": TpuPodConfig(tpu_type="v5p-32"),
        "1_2b": TpuPodConfig(tpu_type="v5p-32"),
    }

    # Optimizer configs for each size
    muon_configs = {
        "130m": MuonConfig(
            learning_rate=0.016,
            adam_lr=0.0032,
            weight_decay=0.1,
            min_lr_ratio=0,
            warmup=0,
            momentum=0.95,
            beta1=0.8,
            beta2=0.98,
            epsilon=1e-15,
            muon_epsilon=1e-5,
            max_grad_norm=1,
            lr_schedule="linear",
            decay=0.8,
        ),
        "300m": MuonConfig(
            learning_rate=0.008,
            adam_lr=0.0024,
            weight_decay=0.1,
            min_lr_ratio=0,
            warmup=0,
            momentum=0.98,
            beta1=0.8,
            beta2=0.98,
            epsilon=1e-15,
            muon_epsilon=1e-5,
            max_grad_norm=1,
            lr_schedule="linear",
            decay=0.8,
        ),
        "520m": MuonConfig(
            learning_rate=0.008,
            adam_lr=0.0024,
            weight_decay=0.1,
            min_lr_ratio=0,
            warmup=0,
            momentum=0.98,
            beta1=0.8,
            beta2=0.98,
            epsilon=1e-25,
            muon_epsilon=1e-5,
            max_grad_norm=1,
            lr_schedule="linear",
            decay=1,
        ),
        "1_2b": MuonConfig(
            learning_rate=0.004,
            adam_lr=0.0012,
            weight_decay=0.1,
            min_lr_ratio=0,
            warmup=0,
            momentum=0.98,
            beta1=0.8,
            beta2=0.98,
            epsilon=1e-15,
            muon_epsilon=1e-5,
            max_grad_norm=2,
            lr_schedule="linear",
            decay=1,
        ),
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
        "130m": (
            f"130M parameter model trained with the {optimizer_name} optimizer with tokens-per-step={seq_len*batch_size}."
        ),
        "300m": (
            f"300M parameter model trained with the {optimizer_name} optimizer with tokens-per-step={seq_len*batch_size}."
        ),
        "520m": (
            f"520M parameter model trained with the {optimizer_name} optimizer with tokens-per-step={seq_len*batch_size}."
        ),
        "1_2b": (
            f"1.2B parameter model trained with the {optimizer_name} optimizer with tokens-per-step={seq_len*batch_size}."
        ),
    }

    # Names for the runs
    run_names = {
        "130m": f"llama_130m_{optimizer_name}_tps{seq_len*batch_size}",
        "300m": f"llama_300m_{optimizer_name}_tps{seq_len*batch_size}",
        "520m": f"llama_520m_{optimizer_name}_tps{seq_len*batch_size}",
        "1_2b": f"llama_1_2b_{optimizer_name}_tps{seq_len*batch_size}",
    }

    # Gather config for the requested size
    if size not in param_counts:
        raise ValueError(f"Unknown size: {size}")

    param_count = param_counts[size]
    model_config = dataclasses.replace(model_cfgs[size], seq_len=seq_len)
    seq_len = model_config.seq_len
    resource_config = resource_cfgs[size]
    if optimizer_name == "muon":
        optimizer_config = muon_configs[size]
    elif optimizer_name == "adamw":
        optimizer_config = adam_configs[size]
    else:
        raise NotImplementedError(f"Optimizer {optimizer_name} not supported yet in this sweep.")
    description = descriptions[size]
    run_name = run_names[size]

    num_train_steps = get_num_train_steps(param_count, batch_size, seq_len)

    # Taken from Simo Ryu's observation that lr ~ sqrt(BS) also holds for Shampoo & Muon: https://x.com/cloneofsimo/status/1907731069878825400
    baseline_batch_size = 128
    learning_rate = optimizer_config.learning_rate * (batch_size / baseline_batch_size)**0.5
    train = SimpleTrainConfig(
        resource_config,
        train_batch_size=batch_size,
        num_train_steps=num_train_steps,
        learning_rate=learning_rate,
        optimizer_config=optimizer_config,
    )
    cfg = SpeedrunConfig(
        author=AUTHOR,
        description=description,
        model_config=model_config,
        train_config=train,
    )
    return run_name, cfg


if __name__ == "__main__":
    runs = []
    for optimizer_name in ["muon", "adamw"]:
        for model_size in ["130m", "300m"]:  # For future sweep, add "520m", "1_2b"
            for batch_size in [64, 128, 256, 512, 1024]:
                runs.append(build_config(optimizer_name, model_size, batch_size))

    steps = []
    for name, cfg in runs:
        cfg.print_run_info()
        steps.extend(default_speedrun(name, cfg))

    executor_main(steps=steps, description="Muon/AdamW speedruns (Chinchilla optimal) | Batch Size Sweep")
