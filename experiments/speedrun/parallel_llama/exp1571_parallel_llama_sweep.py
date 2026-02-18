# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""
Speedruns using the Parallel Llama architecture for various model sizes.
Based on the parallel computation pattern where attention and MLP are computed simultaneously.
"""

import logging
from levanter.optim.cautious import CautiousConfig

from experiments.llama import llama_75m, llama_150m, llama_300m, llama_600m
from experiments.simple_train_config import SimpleTrainConfig
from experiments.speedrun.parallel_llama.exp1571_parallel_llama import ParallelLlamaConfig
from marin.execution.executor import executor_main
from fray.cluster import ResourceConfig
from marin.speedrun.speedrun import Author, SpeedrunConfig, default_speedrun
import time

AUTHOR = Author(
    name="Harry Shin",
    affiliation="Independent",
    url="https://www.linkedin.com/in/harry-shin-34743216a/",
)

logger = logging.getLogger("ray")


def get_num_train_steps(param_count, batch_size, seq_len):
    """Compute the number of steps for Chinchilla optimal training (20x params tokens)."""
    total_tokens = param_count * 20
    tokens_per_step = batch_size * seq_len
    return total_tokens // tokens_per_step


def _to_parallel_llama_from_llama(llama_cfg, *, seq_len_override=None) -> ParallelLlamaConfig:
    """
    Build a ParallelLlamaConfig with identical sizes to a given LLaMA config.
    """
    parallel_llama = ParallelLlamaConfig(
        max_seq_len=seq_len_override if seq_len_override is not None else llama_cfg.max_seq_len,
        hidden_dim=llama_cfg.hidden_dim,
        intermediate_dim=llama_cfg.intermediate_dim,
        num_layers=llama_cfg.num_layers,
        num_heads=llama_cfg.num_heads,
        num_kv_heads=llama_cfg.num_kv_heads,
        use_bias=False,
        use_layer_norm_weight=True,
        tie_word_embeddings=False,
        use_parallel_blocks=True,
    )
    return parallel_llama


def build_config(size: str) -> tuple[str, SpeedrunConfig]:
    param_counts = {
        "75m": 75_000_000,
        "150m": 150_000_000,
        "300m": 300_000_000,
        "520m": 520_000_000,
    }

    llama_model_cfgs = {
        "75m": llama_75m,
        "150m": llama_150m,
        "300m": llama_300m,
        "520m": llama_600m,
    }

    batch_sizes = {
        "75m": 128,
        "150m": 128,
        "300m": 128,
        "520m": 128,
    }

    # Resource configurations - using GPUs for smaller models, TPUs for larger
    resource_cfgs = {
        "75m": ResourceConfig.with_gpu("H100", count=1),
        "150m": ResourceConfig.with_gpu("H100", count=1),
        "300m": ResourceConfig.with_gpu("H100", count=1),
        "520m": ResourceConfig.with_gpu("H100", count=1),
    }

    # Cautious optimizer configs for each size
    cautious_configs = {
        "75m": CautiousConfig(
            learning_rate=0.008,
            weight_decay=0.1,
            min_lr_ratio=0.0,
            warmup=2000,
            beta1=0.95,
            beta2=0.98,
            epsilon=1e-15,
            max_grad_norm=1,
        ),
        "150m": CautiousConfig(
            learning_rate=0.008,
            weight_decay=0.1,
            min_lr_ratio=0.0,
            warmup=2000,
            beta1=0.95,
            beta2=0.98,
            epsilon=1e-15,
            max_grad_norm=1,
        ),
        "300m": CautiousConfig(
            learning_rate=0.008,
            weight_decay=0.1,
            min_lr_ratio=0.0,
            warmup=2000,
            beta1=0.98,
            beta2=0.98,
            epsilon=1e-25,
            max_grad_norm=2,
        ),
        "520m": CautiousConfig(
            learning_rate=0.008,
            weight_decay=0.1,
            min_lr_ratio=0.0,
            warmup=2000,
            beta1=0.98,
            beta2=0.98,
            epsilon=1e-25,
            max_grad_norm=1,
        ),
    }

    descriptions = {
        "75m": "Parallel Llama ~75M with parallel attention/MLP computation.",
        "150m": "Parallel Llama ~150M with parallel attention/MLP computation.",
        "300m": "Parallel Llama ~300M with parallel attention/MLP computation.",
        "520m": "Parallel Llama ~520m with parallel attention/MLP computation.",
    }

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_names = {
        "75m": f"parallel_llama_75m_{timestamp}",
        "150m": f"parallel_llama_150m_{timestamp}",
        "300m": f"parallel_llama_300m_{timestamp}",
        "520m": f"parallel_llama_520m_{timestamp}",
    }

    if size not in param_counts:
        raise ValueError(f"Unknown size: {size}")

    llama_cfg = llama_model_cfgs[size]
    batch_size = batch_sizes[size]
    resource_config = resource_cfgs[size]
    cautious = cautious_configs[size]
    description = descriptions[size]
    run_name = run_names[size]

    # Convert to ParallelLlamaConfig and keep seq_len from original config
    model_config = _to_parallel_llama_from_llama(llama_cfg)
    seq_len = model_config.max_seq_len

    num_train_steps = get_num_train_steps(param_counts[size], batch_size, seq_len)

    train = SimpleTrainConfig(
        resource_config,
        train_batch_size=batch_size,
        num_train_steps=num_train_steps,
        learning_rate=cautious.learning_rate,
        optimizer_config=cautious,
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
        build_config("75m"),
        build_config("150m"),
        build_config("300m"),
        # build_config("520m"),
    ]

    steps = []
    for name, cfg in runs:
        cfg.print_run_info()
        steps.extend(default_speedrun(name, cfg))

    executor_main(
        steps=steps, description="Parallel Llama speedruns (Chinchilla-optimal tokens, w/ parallel attention/MLP)"
    )
