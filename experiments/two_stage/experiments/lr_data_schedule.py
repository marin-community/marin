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
                num_train_steps=1000,
                lr_schedule=lr_schedule,
                lr=3e-3,
                lr_cooldown_duration=lr_cooldown_duration,
                wandb_project_name="suhas-two-stage",
                wandb_additional_tags=[f"{rare_data_name}-c4-lr-data-schedule"],
                model_name="150m4k",
                nametag="",
            )
        )
        for lr_schedule, lr_cooldown_duration in [
            ("cosine", 1.0),
            ("linear", 0.01),
            ("linear", 0.05),
            ("linear", 0.1),
            ("linear", 0.2),
        ]
        for rare_fraction in [0.01]
        for replay_ratio in [0.99, 0.95, 0.9, 0.6, 0.0]
        for rare_stage2_allocation in [1.0, 0.5, 0.25, 0.1, 0.05]
        for rare_data_name in ["finemath", "starcoder", "flan", "spj"]
    ]

    executor_main(
        steps=train_steps,
        description="Sanity check for lr data schedule",
    )
    