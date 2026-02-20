# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Sweep configs for grouped-matmul Mixtral MoE runs based on the original 300M setup."""

import logging
from collections.abc import Sequence
from typing import Any

from experiments.speedrun.custom_mixtral import MixtralConfig
from experiments.simple_train_config import SimpleTrainConfig
from fray.cluster import ResourceConfig
from marin.execution.executor import executor_main
from marin.speedrun.speedrun import Author, SpeedrunConfig, default_speedrun

AUTHOR = Author(name="Pranshu Chaturvedi", affiliation="Stanford University", url="https://stanford.edu/~pranshu")
LOGGER = logging.getLogger("ray")

VOCAB_SIZE = 32_000
SEQ_LEN = 1024

MODEL_ORDER: Sequence[str] = (
    "mixtral_300m",
    "mixtral_1b",
    "mixtral_1_5b",
)

BASE_MODEL_ARGS: dict[str, Any] = dict(
    seq_len=SEQ_LEN,
    hidden_dim=768,
    intermediate_dim=768,
    num_layers=12,
    num_heads=12,
    num_kv_heads=12,
    gradient_checkpointing=True,
    scan_layers=True,
    n_routed_experts=32,
    num_experts_per_tok=4,
    use_gmm=True,
    lbl_coef=None,
    rzl_coef=None,
)

MODEL_CONFIGS = {
    "mixtral_300m": MixtralConfig(**BASE_MODEL_ARGS),
    "mixtral_1b": MixtralConfig(**(BASE_MODEL_ARGS | dict(hidden_dim=1536, intermediate_dim=1536, num_layers=16))),
    "mixtral_1_5b": MixtralConfig(**(BASE_MODEL_ARGS | dict(hidden_dim=2048, intermediate_dim=2048, num_layers=20))),
}


def build_speedrun_config(model_name: str, *, train_batch_size: int, lr: float) -> SpeedrunConfig:
    """Build a SpeedrunConfig for a given model size."""
    return SpeedrunConfig(
        author=AUTHOR,
        description=f"Grouped-matmul Mixtral MoE sweep: {model_name}",
        model_config=MODEL_CONFIGS[model_name],
        train_config=SimpleTrainConfig(
            ResourceConfig.with_tpu("v4-8", slice_count=1),
            train_batch_size=train_batch_size,
            num_train_steps=2000,
            learning_rate=lr,
            weight_decay=0.1,
            steps_per_eval=500,
        ),
    )


def main():
    configs = {
        "mixtral_300m": build_speedrun_config("mixtral_300m", train_batch_size=256, lr=5e-4),
        "mixtral_1b": build_speedrun_config("mixtral_1b", train_batch_size=128, lr=3e-4),
        "mixtral_1_5b": build_speedrun_config("mixtral_1_5b", train_batch_size=96, lr=2e-4),
    }
    for name in MODEL_ORDER:
        cfg = configs[name]
        LOGGER.info("Launching %s", name)
        executor_main(steps=default_speedrun(f"pranshu_mixtral_moe_gmm_sweep_{name}", cfg))


if __name__ == "__main__":
    main()
