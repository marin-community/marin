from marin.execution.executor import output_path_of, executor_main

from experiments.two_stage.two_stage_config import TwoStageConfig, two_stage_train_step
from experiments.two_stage.experiments.pretraining import pretraining_configs, NUM_PRETRAINING_STEPS

if __name__ == "__main__":
    NUM_RARE_STEPS = 10.0
    train_steps = [
        two_stage_train_step(
            TwoStageConfig(
                rare_data_name=pretraining_config.rare_data_name,
                common_data_name="c4",
                rare_fraction=NUM_RARE_STEPS / num_train_steps,
                rare_stage2_allocation=1.0,
                stage2_duration=1.0,
                num_train_steps=num_train_steps,
                lr_schedule=lr_schedule,
                lr=lr,
                lr_cooldown_duration=lr_cooldown_duration,
                wandb_project_name="suhas-two-stage",
                wandb_additional_tags=["fine-tuning-5-6", f"{pretraining_config.rare_data_name}-c4-fine-tuning"],
                model_name="150m4k",
                initialize_from_hf=output_path_of(two_stage_train_step(pretraining_config)).cd(f"hf/step-{NUM_PRETRAINING_STEPS-1}"),
                nametag="-0" if pretraining_config.rare_fraction == 0.0 else "-1",
            )
        )
        for lr_schedule, lr_cooldown_duration in [
            ("cosine", 1.0),
        ]
        for lr in [1e-3]
        for num_train_steps in [10, 20, 50, 100, 200, 1000]
        for pretraining_config in pretraining_configs
    ]

    executor_main(
        steps=train_steps,
        description="Sanity check for lr data schedule",
    )
    