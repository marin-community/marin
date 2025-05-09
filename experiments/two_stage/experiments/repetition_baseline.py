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
                num_train_steps=1000,
                lr_schedule=lr_schedule,
                lr={"cosine": 1e-3, "linear": 3e-3}[lr_schedule],
                lr_cooldown_duration=lr_cooldown_duration,
                wandb_project_name="suhas-two-stage",
                wandb_additional_tags=[f"repetition-trial-v7", f"{rare_data_name}-c4-repetition-trial-v7"],
                model_name="150m4k",
                nametag="-v7",
            )
        )
        for lr_schedule, lr_cooldown_duration in [
            # ("cosine", 1.0),
            # ("linear", 0.01),
            ("linear", 0.02),
            # ("linear", 0.05),
            ("linear", 0.1),
            # ("linear", 0.2),
            # ("linear", 0.5),
        ]
        for rare_fraction in [1.0/1024.0]
        for replay_ratio in [0.0, 0.5, 0.75, 0.875, 0.9375, 0.96875]
        for rare_stage2_allocation in [1.0]
        for rare_data_name in ["finemath", "flan"]
        for rare_data_epochs in [4, 8, 16]
    ]

    executor_main(
        steps=train_steps,
        description="Sanity check for lr data schedule",
    )
    