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
Octothinker mid-training
"""

from experiments.data_efficiency.train import DataEfficiencyConfig, data_efficiency_train_step
from marin.evaluation.evaluation_config import EvalTaskConfig
from marin.execution.executor import executor_main

# 16000 steps ==> 4B tokens
# 280000 steps ==> 73B tokens

tasks = [
    EvalTaskConfig(name="mathqa", num_fewshot=8),
]

train_steps = [
    data_efficiency_train_step(
        DataEfficiencyConfig(
            data_name="octo",
            epochs=epochs,
            base_train_steps=base_train_steps * (64 / batch_size),
            train_batch_size=batch_size,
            lr_schedule="cosine",
            lr=lr,
            weight_decay=weight_decay,
            wandb_project_name="suhas-cpt-data-efficiency",
            wandb_additional_tags=["octothinker-cpt"],
            model_name="l3b",
            nametag=(f"-seed{seed}" if seed is not None else ""),
            initialize_from_hf=initialize_from_hf,
            eval_harness_tasks=tasks,
            train_seed=seed if seed else 0,
            data_seed=seed if seed else 0,
            tpu_type="v4-64",
            per_device_parallelism=2 if batch_size == 512 else -1,
        )
    )
    for base_train_steps, batch_size, epochs, seed in [
        # Default 4B
        (16_000, 512, 1, 0),
        # Lower batch size
        (16_000, 64, 1, 0),
        # Epoching
        (16_000, 64, 4, 0),
        # Remaining ensemble members
        (16_000, 64, 4, 1),
        (16_000, 64, 4, 2),
        (16_000, 64, 4, 3),
        (16_000, 64, 4, 4),
        (16_000, 64, 4, 5),
        (16_000, 64, 4, 6),
        (16_000, 64, 4, 7),
        # Default 73B
        (280_000, 512, 1, 0),
    ]
    for weight_decay in [0.1]
    for initialize_from_hf in ["meta-llama/Llama-3.2-3B"]
    for lr in [3e-5]
]

if __name__ == "__main__":
    executor_main(
        steps=train_steps,
        description="Octothinker",
    )
