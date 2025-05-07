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
                num_train_steps=1000,
                lr_schedule=lr_schedule,
                lr=lr,
                lr_cooldown_duration=lr_cooldown_duration,
                wandb_project_name="suhas-two-stage",
                wandb_additional_tags=["efficiency-baselines", f"{rare_data_name}-c4-efficiency-baselines"],
                model_name="150m4k",
                nametag="",
            )
        )
        for lr_schedule, lr_cooldown_duration in [
            ("cosine", 1.0),
        ]
        for rare_fraction in [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
        for stage2_duration in [1.0]
        for rare_stage2_allocation in [1.0]
        for lr in [3e-3, 1e-3]
        for rare_data_name in ["finemath", "starcoder", "flan", "spj"]
    ]

    executor_main(
        steps=train_steps,
        description="Data efficiency baselines",
    )
    