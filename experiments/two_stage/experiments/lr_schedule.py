from experiments.two_stage.two_stage_config import TwoStageConfig, two_stage_train_step
from marin.execution.executor import executor_main

if __name__ == "__main__":
    train_steps = [
        two_stage_train_step(
            TwoStageConfig(
                rare_data_name=rare_data_name,
                common_data_name="c4",
                rare_fraction=rare_fraction,
                stage2_duration=rare_fraction * rare_data_epochs if all_at_end else 1.0,
                rare_stage2_allocation=rare_stage2_allocation,
                rare_data_epochs=rare_data_epochs,
                num_train_steps=1024,
                lr_schedule=lr_schedule,
                lr=lr,
                lr_cooldown_duration=lr_cooldown_duration,
                wandb_project_name="suhas-two-stage",
                wandb_additional_tags=[f"finding-lr-schedule-v2", f"{rare_data_name}-c4-finding-lr-schedule-v2"],
                model_name="150m4k",
                nametag="-fl",
            )
        )
        for lr_schedule, lr_cooldown_duration in [
            ("linear", 0.0),
            ("linear", 0.05),
            ("linear", 0.1),
            ("linear", 0.2),
            ("linear", 0.5),
            ("linear", 0.99),
        ]
        for lr in [1e-3, 3e-3]
        for rare_fraction in [1.0/1024.0]
        for rare_stage2_allocation in [1.0]
        for all_at_end in [True]
        for rare_data_name, rare_data_epochs in [
            ("finemath", 32)
        ]
    ]

    executor_main(
        steps=train_steps,
        description="Sanity check for lr data schedule",
    )
    