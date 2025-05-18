from experiments.two_stage.two_stage_config import TwoStageConfig, two_stage_train_step
from marin.execution.executor import executor_main

if __name__ == "__main__":
    train_steps = [
        two_stage_train_step(
            TwoStageConfig(
                rare_data_name=rare_data_name,
                common_data_name="c4",
                rare_fraction=rare_fraction,
                stage2_duration=stage2_duration,
                rare_stage2_allocation=rare_stage2_allocation,
                rare_data_epochs=rare_data_epochs,
                num_train_steps=1024,
                lr_schedule=lr_schedule,
                lr=lr,
                wandb_project_name="suhas-two-stage",
                wandb_additional_tags=["efficiency-baselines-v3", f"{rare_data_name}-c4-efficiency-baselines-v3"],
                model_name="150m4k",
                nametag="-eb",
            )
        )
        for lr_schedule in ["cosine"]
        for rare_fraction in [0.25/1024.0, 0.5/1024.0, 1.0/1024.0, 2.0/1024.0, 4.0/1024.0, 8.0/1024.0, 16.0/1024.0]
        for stage2_duration in [1.0]
        for rare_stage2_allocation in [1.0]
        for lr in [3e-3]
        for rare_data_name in [
            "flan",
            "starcoder",
            "finemath",
        ]
        for rare_data_epochs in [32]
    ]

    executor_main(
        steps=train_steps,
        description="Data efficiency baselines",
    )
    