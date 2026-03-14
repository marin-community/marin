# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""
Speedrun code for a 50M parameter model based on the Llama architecture.
It is trained for 10 times the Chinchilla-optimal number of tokens, i.e 10B tokens.
"""

import logging

from experiments.llama import llama_50m
from experiments.simple_train_config import SimpleTrainConfig
from marin.execution.executor import executor_main
from fray.cluster import ResourceConfig
from marin.speedrun.speedrun import Author, SpeedrunConfig, default_speedrun

logger = logging.getLogger("ray")

speedrun_config = SpeedrunConfig(
    author=Author(
        name="Nikil Ravi",
        affiliation="Stanford University",
        url="https://www.linkedin.com/in/nikilravi/",
    ),
    description="50M parameter trained for 10 times Chinchilla-optimal number of tokens, i.e 10B tokens.",
    model_config=llama_50m,
    train_config=SimpleTrainConfig(
        ResourceConfig.with_tpu("v4-128"),
        train_batch_size=512,
        num_train_steps=20000,
        learning_rate=3e-3,
        weight_decay=0.1,
        steps_per_eval=1500,
    ),
)

speedrun_config.print_run_info()

if __name__ == "__main__":
    executor_main(steps=default_speedrun("llama_50m_10xC", speedrun_config))
