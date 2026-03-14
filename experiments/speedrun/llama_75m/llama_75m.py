# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""
Speedrun code for a 75M parameter model based on the Llama architecture.
"""

import logging

from experiments.llama import llama_75m
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
    description="75M parameter model based on Llama architecture.",
    model_config=llama_75m,
    train_config=SimpleTrainConfig(
        resources=ResourceConfig.with_tpu("v4-128"),
        train_batch_size=512,
        num_train_steps=6000,
        learning_rate=3e-3,
        weight_decay=0.1,
        steps_per_eval=2000,
    ),
)

speedrun_config.print_run_info()

if __name__ == "__main__":
    executor_main(steps=default_speedrun("llama_75m", speedrun_config))
