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
Searching for the best replay ratio for fine-tuning (based on two_stage_config.py).
Two stage training: pre-training on only C4, followed by fine-tuning on rare + wC4.
We cooldown/rewarmup the learning rate and reset the optimizer state in between checkpoints.
For a fair comparison, we keep the total number of training steps fixed across replay ratio.
When we increase the replay ratio, the second stage has more steps so we decrease the length of pre-training.
"""

from experiments.two_stage.two_stage_config import TwoStageConfig, two_stage_train_step
from marin.execution.executor import executor_main, output_path_of


def pretraining_for_fixed_steps(steps: int):
    return two_stage_train_step(
        TwoStageConfig(
            rare_data_name="finemath",
            common_data_name="c4",
            rare_fraction=0.0,
            rare_stage2_allocation=1.0,
            stage2_duration=1.0,
            num_train_steps=steps,
            lr_schedule="cosine",
            lr=1e-3,
            wandb_project_name="suhas-two-stage",
            wandb_additional_tags=["pretraining-replay"],
            model_name="150m4k",
            nametag=f"-{int(steps)}",
        )
    )


def finetuning_with_replay(
    rare_data_name: str,
    rare_data_epochs: int,
    num_rare_steps: int,
    replay_multiplier: int,
    num_total_steps: int,
    lr: float,
    nametag: str,
    wandb_additional_tags: list[str],
):
    num_fine_tuning_steps = int(num_rare_steps * rare_data_epochs * replay_multiplier)
    assert num_fine_tuning_steps <= num_total_steps
    num_pretraining_steps = num_total_steps - num_fine_tuning_steps
    pretraining_step = pretraining_for_fixed_steps(num_pretraining_steps)

    return two_stage_train_step(
        TwoStageConfig(
            rare_data_name=rare_data_name,
            common_data_name="c4",
            rare_fraction=float(num_rare_steps) / num_fine_tuning_steps,
            rare_stage2_allocation=1.0,
            stage2_duration=1.0,
            rare_data_epochs=rare_data_epochs,
            num_train_steps=num_fine_tuning_steps,
            lr_schedule="cosine",
            lr=lr,
            wandb_project_name="suhas-two-stage",
            wandb_additional_tags=wandb_additional_tags,
            model_name="150m4k",
            initialize_from_hf=output_path_of(pretraining_step).cd(f"hf/step-{num_pretraining_steps - 1}"),
            nametag=nametag,
        )
    )


if __name__ == "__main__":
    NUM_RARE_STEPS = 1
    TOTAL_STEPS = 1024

    train_steps = [
        finetuning_with_replay(
            rare_data_name=rare_data_name,
            rare_data_epochs=rare_data_epochs,
            replay_multiplier=replay_multiplier,
            num_rare_steps=NUM_RARE_STEPS,
            num_total_steps=TOTAL_STEPS,
            lr=lr,
            nametag="-j",
            wandb_additional_tags=["fine-tuning-replay", f"{rare_data_name}-c4-fine-tuning-replay"],
        )
        for replay_multiplier in [1.125, 1.25, 1.5, 2.0, 3.0, 4.0]
        for lr in [3e-4]
        for rare_data_name in ["finemath", "flan", "starcoder"]
        for rare_data_epochs in [64]
    ]

    executor_main(
        steps=train_steps,
        description="Fine-tuning with replay",
    )
