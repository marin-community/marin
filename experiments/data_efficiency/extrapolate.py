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
Script to run the neighborhood of a given hyper-parameter to search for locally-optimal hyper-parameters
"""

from experiments.data_efficiency.train import DataEfficiencyConfig, data_efficiency_train_step
from experiments.data_efficiency.utils import get_bounding_box
from marin.execution.executor import executor_main

train_steps = [
    [
        data_efficiency_train_step(
            DataEfficiencyConfig(
                data_name="dclm",
                epochs=epochs,
                base_train_steps=base_train_steps,
                train_batch_size=64,
                lr_schedule="cosine",
                lr=lr,
                weight_decay=weight_decay,
                wandb_project_name="suhas-data-efficiency",
                model_name=model_name,
                tpu_type="v4-64",
            )
        )
        for base_train_steps, epochs, lr, weight_decay, model_name in get_bounding_box(*candidate_hparams)
    ]
    for candidate_hparams in [
        # Fixed regularized
        # (800, 4, 1e-3, 3.2, "3_2b4k_alln"),
        # (800, 8, 1e-3, 3.2, "3_2b4k_alln"),
        # Fixed unregularized
        (800, 4, 3e-4, 0.1, "3_2b4k_qkn", False),
    ]
]

train_steps = [step for sublist in train_steps for step in sublist]

if __name__ == "__main__":
    executor_main(
        steps=train_steps,
        description="Data scaling baseline",
    )
