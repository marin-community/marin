# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# nodryrun

import logging

from experiments.llama import llama_75m
from experiments.simple_train_config import SimpleTrainConfig
from fray.cluster import ResourceConfig
from marin.execution.executor import executor_main
from marin.speedrun.speedrun import Author, SpeedrunConfig, default_speedrun

logger = logging.getLogger("ray")

speedrun_config = SpeedrunConfig(
    author=Author(
        name="Pranshu Chaturvedi",
        affiliation="Stanford University",
        url="https://stanford.edu/~pranshu",
    ),
    description="Training Llama 75M on a TPU v4-8 for the speedrun.",
    model_config=llama_75m,
    train_config=SimpleTrainConfig(
        ResourceConfig.with_tpu("v4-8", slice_count=1),
        train_batch_size=256,
        num_train_steps=4000,
        learning_rate=5e-4,
        weight_decay=0.1,
        steps_per_eval=1000,
    ),
)

if __name__ == "__main__":
    logger.info("Launching Llama 75M speedrun on TPU v4-8.")
    executor_main(steps=default_speedrun("pranshu_llama_75m_speedrun_v4_8", speedrun_config))
