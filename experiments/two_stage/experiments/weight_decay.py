from experiments.two_stage.two_stage_config import TwoStageConfig, two_stage_train_step
from marin.execution.executor import executor_main

if __name__ == "__main__":
    train_steps = [
        two_stage_train_step(
            TwoStageConfig(
                rare_data_name=rare_data_name,
                common_data_name="c4",
                rare_fraction=rare_fraction,
                replay_ratio=0.0,
                rare_stage2_allocation=1.0,
                rare_data_epochs=rare_data_epochs,
                num_train_steps=1024,
                lr_schedule=lr_schedule,
                lr=3e-3,
                lr_cooldown_duration=lr_cooldown_duration,
                wandb_project_name="suhas-two-stage",
                wandb_additional_tags=[f"finding-weight-decay", f"{rare_data_name}-c4-finding-weight-decay"],
                model_name="150m4k",
                nametag=f"-w{str(int(weight_decay * 10))[:-1]}",
                weight_decay=weight_decay,
            )
        )
        for lr_schedule, lr_cooldown_duration in [
            ("linear", 0.1),
        ]
        for rare_fraction in [1.0/1024.0]
        for rare_data_name, rare_data_epochs in [
            ("finemath", 32),
            ("finemath", 64),
        ]
        for weight_decay in [0.0, 0.1, 0.2, 0.4, 0.8, 1.6]
    ]

    executor_main(
        steps=train_steps,
        description="Sanity check for lr data schedule",
    )
    