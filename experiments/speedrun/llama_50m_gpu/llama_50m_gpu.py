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
Speedrun code for a 50M parameter model based on the Llama architecture. The model is trained on the Fineweb-Edu dataset
(the default dataset for speedruns) on one A100.
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
        name="Herumb Shandilya",
        affiliation="Stanford University",
        url="https://x.com/krypticmouse",
    ),
    description="50M param model based on Llama architecture.",
    model_config=llama_50m,
    train_config=SimpleTrainConfig(
        ResourceConfig.with_gpu("A100", count=4),
        train_batch_size=128,
        num_train_steps=7600,
        learning_rate=3e-3,
        weight_decay=0.1,
        steps_per_eval=500,
    ),
)

speedrun_config.print_run_info()

if __name__ == "__main__":
    executor_main(steps=default_speedrun("llama_50m_gpu_4xA100_run", speedrun_config))
