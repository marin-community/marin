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
                lr=lr,
                lr_cooldown_duration=lr_cooldown_duration,
                wandb_project_name="suhas-two-stage",
                wandb_additional_tags=[f"finding-repetitions-1.6", f"{rare_data_name}-c4-finding-repetitions-1.6"],
                model_name="150m4k",
                nametag="-r",
            )
        )
        for lr_schedule, lr_cooldown_duration, lr in [
            ("linear", 0.0, 3e-3),
            ("linear", 0.0, 1e-3),
        ]
        for rare_fraction in [16.0/1024.0]
        for replay_ratio in [0.0]
        for rare_stage2_allocation in [1.0]
        for rare_data_name in ["starcoder"]
        for rare_data_epochs in [4, 8, 16, 32]
    ]

    executor_main(
        steps=train_steps,
        description="Finding repetitions",
    )
    