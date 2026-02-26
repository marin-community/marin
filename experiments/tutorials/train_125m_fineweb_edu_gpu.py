# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""
This is a tutorial on how to train an approximately 125M parameter model on FineWeb-Edu using a GPU.

This script demonstrates how to:
1. Reuse a pretokenized FineWeb-Edu cache
2. Train a ~125M Llama model on a single GPU
3. Pin outputs to a local Marin prefix for the run
"""

from fray.cluster import ResourceConfig
from marin.execution.executor import executor_main

from experiments.defaults import default_train
from experiments.llama import llama_150m
from experiments.simple_train_config import SimpleTrainConfig
from experiments.speedrun.prebuilt_caches import fineweb_edu_subcache_10M

llama_150m_train_config = SimpleTrainConfig(
    resources=ResourceConfig.with_gpu(count=1),
    train_batch_size=32,
    num_train_steps=1000,
    learning_rate=3e-4,
    weight_decay=0.1,
)

llama_150m_fineweb_edu_model = default_train(
    name="llama-150m-fineweb-edu-gpu",
    tokenized=fineweb_edu_subcache_10M,
    model_config=llama_150m,
    train_config=llama_150m_train_config,
    tags=["llama", "150m", "fineweb-edu", "gpu", "tutorial"],
    eval_harness_tasks=[],
    use_default_validation=False,
)

if __name__ == "__main__":
    executor_main(steps=[llama_150m_fineweb_edu_model])
