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
Script to launch a grid of experiments of interest
"""

from experiments.data_efficiency.train import DataEfficiencyConfig, data_efficiency_train_step
from marin.execution.executor import executor_main

# 800 steps ==> 200M tokens

train_steps = [
    data_efficiency_train_step(
        DataEfficiencyConfig(
            data_name="dclm",
            epochs=epochs,
            base_train_steps=base_train_steps,
            train_batch_size=batch_size,
            lr_schedule="cosine",
            lr=lr,
            weight_decay=weight_decay,
            wandb_project_name="suhas-data-efficiency",
            model_name=model_name,
            nametag=f"-bs{batch_size}",
            tpu_type="v4-64",
            # tpu_type="v5litepod-128",
            # per_device_parallelism=2,
        )
    )
    for base_train_steps in [800]
    for epochs in [4]
    for weight_decay in [0.1]
    for batch_size in [64]
    for model_name, lr in [
        ("moedebug", 1e-3),
    ]
]

if __name__ == "__main__":
    executor_main(
        steps=train_steps,
        description="Data scaling baseline",
    )
