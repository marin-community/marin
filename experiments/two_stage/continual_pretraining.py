"""
Example of continual pretraining under the two stage framework for Basque data.
Simply load a pretrained model and do a single stage of training with choice of replay data.
"""

from experiments.two_stage.two_stage_config import TwoStageConfig, two_stage_train_step
from marin.evaluation.evaluation_config import EvalTaskConfig
from marin.execution.executor import executor_main

BASQUE_TASKS = (EvalTaskConfig("xcopa_eu", num_fewshot=0, task_alias="xcopa_eu"),)

if __name__ == "__main__":
    NUM_RARE_STEPS = 400.0  # 200M tokens
    train_steps = [
        two_stage_train_step(
            TwoStageConfig(
                rare_data_name=rare_data_name,
                common_data_name="spj",
                rare_fraction=1 - replay_ratio,
                rare_stage2_allocation=1.0,
                stage2_duration=1.0,
                rare_data_epochs=rare_data_epochs,
                num_train_steps=NUM_RARE_STEPS / replay_ratio,
                lr_schedule=lr_schedule,
                lr=lr,
                lr_cooldown_duration=lr_cooldown_duration,
                train_batch_size=128,
                wandb_project_name="suhas-two-stage",
                wandb_additional_tags=["cpt", f"{rare_data_name}-spj-cpt"],
                model_name="l8b",
                initialize_from_hf="meta-llama/Meta-Llama-3.1-8B",
                eval_harness_tasks=BASQUE_TASKS,
            )
        )
        for lr_schedule, lr_cooldown_duration in [
            ("cosine", 1.0),
        ]
        for lr in [1e-5]
        for replay_ratio in [0.0, 0.5, 0.75, 0.9]
        for rare_data_name in ["latxa"]
        for rare_data_epochs in [1]
    ]

    executor_main(
        steps=train_steps,
        description="Sanity check for lr data schedule",
    )
