# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""
Speedrun code for a 50M parameter model based on the Llama architecture. The model is trained on the Fineweb-Edu dataset
(the default dataset for speedruns) on one H200.
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
        name="Will Held",
        affiliation="Georgia Institute of Technology",
        url="WilliamHeld.com",
    ),
    description="50M param model based on Llama architecture on H200.",
    model_config=llama_50m,
    train_config=SimpleTrainConfig(
        ResourceConfig.with_gpu("H200", count=1),
        train_batch_size=128,
        num_train_steps=7600,
        learning_rate=3e-3,
        weight_decay=0.1,
        steps_per_eval=500,
    ),
)

speedrun_config.print_run_info()

if __name__ == "__main__":
    executor_main(steps=default_speedrun("llama_50m_gpu_H200_run", speedrun_config))
