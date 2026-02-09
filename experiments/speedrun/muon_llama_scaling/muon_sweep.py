# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Speedruns using the Muon optimizer for various Llama model sizes (Chinchilla optimal steps).

Optimizer configs were searched & provided by Kaiyue Wen in https://wandb.ai/marin-community/marin/reports/Fantastic-Optimizers-and-Where-to-Find-Them--VmlldzoxMjgzMzQ2NQ
"""

import dataclasses
import logging

from levanter.optim import MuonConfig

from experiments.llama import llama_1_4b, llama_150m, llama_300m, llama_600m
from experiments.simple_train_config import SimpleTrainConfig
from marin.execution.executor import executor_main
from fray.cluster import ResourceConfig
from marin.speedrun.speedrun import Author, SpeedrunConfig, default_speedrun

AUTHOR = Author(name="William Held", affiliation="Georgia Tech", url="https://WilliamHeld.com")

logger = logging.getLogger("ray")


def get_num_train_steps(param_count, batch_size, seq_len):
    """Compute the number of steps for Chinchilla optimal training (20x params tokens)."""
    total_tokens = param_count * 20
    tokens_per_step = batch_size * seq_len
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
        "520m": 128,
        "1_2b": 256,
    }

    # Resource configs
    resource_cfgs = {
        "130m": ResourceConfig.with_tpu("v5p-32"),
        "300m": ResourceConfig.with_tpu("v5p-32"),
        "520m": ResourceConfig.with_tpu("v5p-32"),
        "1_2b": ResourceConfig.with_tpu("v5p-32"),
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

    # Descriptions
    descriptions = {
        "130m": "130M parameter model trained with the Muon optimizer.",
        "300m": "300M parameter model trained with the Muon optimizer.",
        "520m": "520M parameter model trained with the Muon optimizer.",
        "1_2b": "1.2B parameter model trained with the Muon optimizer.",
    }

    # Names for the runs
    run_names = {
        "130m": "llama_130m_muon_4096",
        "300m": "llama_300m_muon_4096",
        "520m": "llama_520m_muon_4096",
        "1_2b": "llama_1_2b_muon_4096",
    }

    # Gather config for the requested size
    if size not in param_counts:
        raise ValueError(f"Unknown size: {size}")

    param_count = param_counts[size]
    batch_size = batch_sizes[size]
    model_config = dataclasses.replace(model_cfgs[size], max_seq_len=4096)
    seq_len = model_config.max_seq_len
    resource_config = resource_cfgs[size]
    muon = muon_configs[size]
    description = descriptions[size]
    run_name = run_names[size]

    num_train_steps = get_num_train_steps(param_count, batch_size, seq_len)

    train = SimpleTrainConfig(
        resource_config,
        train_batch_size=batch_size,
        num_train_steps=num_train_steps,
        learning_rate=muon.learning_rate,
        optimizer_config=muon,
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

    executor_main(steps=steps, description="Muon speedruns (Chinchilla optimal)")
