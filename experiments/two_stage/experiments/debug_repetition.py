from experiments.two_stage.two_stage_config import TwoStageConfig, two_stage_train_step
from marin.execution.executor import executor_main

if __name__ == "__main__":
    train_steps = [
        two_stage_train_step(
            TwoStageConfig(
                rare_data_name=rare_data_name,
                common_data_name="c4",
                rare_fraction=0.9 / rare_data_epochs,
                stage2_duration=stage2_duration,
                rare_stage2_allocation=rare_stage2_allocation,
                rare_data_epochs=rare_data_epochs,
                num_train_steps=1000,
                lr_schedule=lr_schedule,
                lr=1e-3,
                lr_cooldown_duration=lr_cooldown_duration,
                wandb_project_name="suhas-two-stage",
                wandb_additional_tags=[f"debug-repetition"],
                model_name="150m4k",
                nametag="-v1",
            )
        )
        for lr_schedule, lr_cooldown_duration in [
            ("cosine", 1.0),
        ]
        for rare_stage2_allocation in [1.0]
        for stage2_duration in [1.0]
        for rare_data_name in ["finemath"]
        for rare_data_epochs in [1, 2, 4, 8, 16, 32]
    ]

    executor_main(
        steps=train_steps,
        description="Debugging repetition",
    )