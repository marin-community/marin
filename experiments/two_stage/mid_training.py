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
Searching for the best two-stage data schedule (based on two_stage_config.py).
Mid-training setup where we have a single learning rate schedule and a different mixture for each stage.
Hyperparameters are tuned for the baseline of all data at the end of training.
"""

from experiments.two_stage.two_stage_config import TwoStageConfig, two_stage_train_step
from marin.execution.executor import executor_main

if __name__ == "__main__":
    train_steps = [
        two_stage_train_step(
            TwoStageConfig(
                rare_data_name=rare_data_name,
                common_data_name="c4",
                rare_fraction=rare_fraction,
                replay_ratio=replay_ratio,
                rare_stage2_allocation=rare_stage2_allocation,
                rare_data_epochs=rare_data_epochs,
                num_train_steps=1024,
                lr_schedule=lr_schedule,
                lr=3e-3,
                lr_cooldown_duration=lr_cooldown_duration,
                wandb_project_name="suhas-two-stage",
                wandb_additional_tags=["mid-training-data-schedule", f"{rare_data_name}-c4-mid-training-data-schedule"],
                model_name="150m4k",
                nametag="",
            )
        )
        for rare_fraction in [1.0 / 1024.0]
        for replay_ratio in [0.0, 0.25, 0.5, 0.75, 0.875]
        for rare_stage2_allocation in [1.0, 0.5, 0.25, 0.125]
        for rare_data_name, rare_data_epochs, lr_schedule, lr_cooldown_duration in [
            ("finemath", 32, "linear", 0.1),
            ("starcoder", 32, "linear", 0.1),
            ("flan", 32, "linear", 0.1),
        ]
    ]

    executor_main(
        steps=train_steps,
        description="Mid-training data schedule",
    )
