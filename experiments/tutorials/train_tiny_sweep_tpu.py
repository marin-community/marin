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

"""
This is a tutorial script demonstrating how to perform a hyperparameter sweep
while training a tiny model on the TinyStories dataset using TPU hardware.
"""
import dataclasses

from fray.cluster import ResourceConfig
from marin.execution.executor import executor_main, versioned

from experiments.defaults import default_train
from experiments.evals.task_configs import CORE_TASKS
from experiments.llama import llama_30m
from experiments.pretraining_datasets.simple import tokenized
from experiments.simple_train_config import SimpleTrainConfig

RESOURCES = ResourceConfig.with_tpu("v4-8")
EVALS = CORE_TASKS

small_train_config = SimpleTrainConfig(
    # Here we define the hardware resources we need.
    resources=RESOURCES,
    train_batch_size=128,
    num_train_steps=10000,
    # set hyperparameters
    learning_rate=6e-4,
    weight_decay=0.1,
)

sweep_configs = [
    dataclasses.replace(
        small_train_config,
        learning_rate=lr,
        weight_decay=wd,
    )
    for lr in [3e-4, 6e-4, 1e-3]
    for wd in [0.0, 0.1, 0.2]
]

# 4. Define an lr sweep
runs = []

for config in sweep_configs:
    # 5. Train the model
    lr, wd = config.learning_rate, config.weight_decay
    run = default_train(
        # Marin will automatically create unique ids for runs b/c the model_config is versioned
        # however, we can give each run a unique name for easier identification
        name=f"tutorial-slimpajama_6b-30m-sweep-lr{lr}-wd{wd}",
        tokenized=tokenized["slimpajama_6b"],
        model_config=versioned(llama_30m),
        train_config=config,
        # wandb tags
        tags=["llama", "30m", "slimpajama_6b", "tutorial", "sweep", "test20251117"],
        eval_harness_tasks=CORE_TASKS,
    )
    runs.append(run)

if __name__ == "__main__":
    executor_main(steps=runs)
