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
This is a hello-world speedrun that trains a tiny model, and is designed to run on a single **GPU**.
See https://github.com/marin-community/marin/tree/main/docs/how-to-guides/submitting-speedrun.md for
further documentation around submitting speedrun, and for a more detailed walkthrough of the code
and process.
"""

import logging

from experiments.llama import llama_nano
from experiments.simple_train_config import SimpleTrainConfig
from marin.execution.executor import executor_main
from fray.cluster import ResourceConfig
from marin.speedrun.speedrun import Author, SpeedrunConfig, default_speedrun

logger = logging.getLogger("ray")

speedrun_config = SpeedrunConfig(
    author=Author(
        name="Herumb Shandilya",
        affiliation="Stanford University",
        url="https://www.x.com/krypticmouse",
    ),
    description="Nano model based on Llama architecture.",
    model_config=llama_nano,
    train_config=SimpleTrainConfig(
        ResourceConfig.with_gpu("A100", count=1),
        train_batch_size=32,
        num_train_steps=100,
        learning_rate=3e-3,
        weight_decay=0.1,
        steps_per_eval=500,
    ),
    # tokenized_dataset need not be set here; by default it will be set to a FineWeb-Edu 10B token dataset
    # that we have pre-tokenized and shuffled, available at https://huggingface.co/datasets/marin-community/fineweb-edu-pretokenized-10B
    # tokenized_dataset=fineweb_edu_subcache_10B,
)

# Shows your speedrun configuration, model FLOPs, model size and (training) hardware FLOPs- you
# can use this before actually kicking off a run to validate your setup
speedrun_config.print_run_info()

if __name__ == "__main__":
    executor_main(steps=default_speedrun("llama_nano_gpu_speedrun", speedrun_config))
