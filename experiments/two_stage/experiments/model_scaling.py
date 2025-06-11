from experiments.two_stage.two_stage_config import TwoStageConfig, two_stage_train_step
from marin.execution.executor import executor_main
from experiments.two_stage.models import inverse_width_dict

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
                lr=3e-3 * inverse_width_dict[model_name],
                lr_cooldown_duration=lr_cooldown_duration,
                wandb_project_name="suhas-two-stage",
                wandb_additional_tags=[f"model-scaling-v3", f"{rare_data_name}-c4-model-scaling-v3"],
                model_name=model_name,
                nametag="-m",
            )
        )
        for lr_schedule, lr_cooldown_duration in [
            ("linear", 0.1),
        ]
        for rare_fraction in [1.0/1024.0]
        for replay_ratio in [0.0, 0.25, 0.5, 0.75, 0.875]
        for rare_stage2_allocation in [1.0]
        for rare_data_name in ["finemath"]
        for rare_data_epochs in [32]
        for model_name in ["150m4k", "300m4k", "600m4k", "1_9b4k"]
    ]

    executor_main(
        steps=train_steps,
        description="Sanity check for lr data schedule",
    )
    