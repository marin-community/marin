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
Speedrun code for a 1.4B parameter model based on the Llama architecture.
"""

import logging

from experiments.simple_train_config import SimpleTrainConfig
from experiments.speedrun.prebuilt_caches import fineweb_edu_subcache_10B
from experiments.tutorials.exp1077_reproduce_dclm_1b1x import BATCH_SIZE, SEQ_LEN, llama_1_4b_dclm
from marin.execution.executor import executor_main, versioned
from marin.resources import GpuConfig
from marin.speedrun.speedrun import Author, SpeedrunConfig, default_speedrun

logger = logging.getLogger("ray")

NUM_TRAIN_TOKENS = int(10e9)
NUM_TRAIN_STEPS = NUM_TRAIN_TOKENS // (BATCH_SIZE * SEQ_LEN)

# Basically DCLM Baseline Reproduction, but
speedrun_config = SpeedrunConfig(
    author=Author(
        name="Will Held",
        affiliation="Georgia Institute of Technology",
        url="WilliamHeld.com",
    ),
    description="1.4B param model based on Llama architecture.",
    model_config=llama_1_4b_dclm,
    train_config=SimpleTrainConfig(
        GpuConfig(gpu_count=8, accelerator_type="H200"),
        train_batch_size=BATCH_SIZE,
        num_train_steps=NUM_TRAIN_STEPS,
        learning_rate=3e-3,
        warmup=versioned(5000),
        z_loss_weight=1e-4,
        weight_decay=0.033,
        steps_per_eval=1000,
    ),
    tokenized_dataset=fineweb_edu_subcache_10B,
)

speedrun_config.print_run_info()

if __name__ == "__main__":
    executor_main(steps=default_speedrun("llama_1_4B_8xH200", config=speedrun_config))
