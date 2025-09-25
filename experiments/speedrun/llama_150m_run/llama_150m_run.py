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
Speedrun code for a 150M parameter model based on the Llama architecture.
"""

import logging

from experiments.llama import llama_150m
from experiments.simple_train_config import SimpleTrainConfig
from marin.execution.executor import executor_main
from marin.resources import TpuPodConfig
from marin.speedrun.speedrun import Author, SpeedrunConfig, default_speedrun

logger = logging.getLogger("ray")

speedrun_config = SpeedrunConfig(
    author=Author(
        name="Nikil Ravi",
        affiliation="Stanford University",
        url="https://www.linkedin.com/in/nikilravi/",
    ),
    description="150M param model based on Llama architecture.",
    model_config=llama_150m,
    train_config=SimpleTrainConfig(
        TpuPodConfig(tpu_type="v4-128"),
        train_batch_size=512,
        num_train_steps=3000,
        learning_rate=3e-3,
        weight_decay=0.1,
        steps_per_eval=1000,
    ),
)

speedrun_config.print_run_info()

if __name__ == "__main__":
    executor_main(
        steps=default_speedrun(
            "llama_150m_run", speedrun_config, override_output_path="checkpoints/speedrun/llama_150m_run-37ce8d"
        )
    )
