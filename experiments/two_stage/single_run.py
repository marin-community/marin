# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from experiments.two_stage.two_stage_config import TwoStageConfig, two_stage_train_step
from marin.execution.executor import executor_main

if __name__ == "__main__":
    train_steps = [
        two_stage_train_step(
            TwoStageConfig(
                rare_data_name="starcoder",
                common_data_name="c4",
                rare_fraction=0.01,
                replay_ratio=0.8,
                rare_stage2_allocation=0.9,
                num_train_steps=1000,
                lr_schedule=lr_schedule,
                lr=lr,
                lr_cooldown_duration=lr_cooldown_duration,
                wandb_project_name="suhas-two-stage",
                wandb_additional_tags=["debug"],
                model_name="150m4k",
                nametag="",
            )
        )
        for lr, lr_schedule, lr_cooldown_duration in [
            (3e-3, "cosine", 0.4),
        ]
    ]

    executor_main(
        steps=train_steps,
        description="Debugging two-stage training",
    )
