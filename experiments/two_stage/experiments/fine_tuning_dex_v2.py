from marin.execution.executor import output_path_of, executor_main

from experiments.two_stage.two_stage_config import TwoStageConfig, two_stage_train_step

from experiments.two_stage.experiments.joint_ptft import finetuning_with_replay

if __name__ == "__main__":
    TOTAL_STEPS = 1024
    train_steps = [
        finetuning_with_replay(
            rare_data_name=rare_data_name,
            rare_data_epochs=rare_data_epochs,
            replay_multiplier=1.0,
            num_rare_steps=num_rare_steps,
            num_total_steps=TOTAL_STEPS,
            lr=lr,
            nametag="-s3",
            wandb_additional_tags=["fine-tuning-scaling-law-v3", f"{rare_data_name}-c4-fine-tuning-scaling-law-v3"],
        )
        for lr in [1e-4]
        for rare_data_name in [
            "finemath",
            "flan",
            "starcoder",
            # "spj",
        ]
        for rare_data_epochs in [64]
        for num_rare_steps in [0.5]
    ]

    executor_main(
        steps=train_steps,
        description="Sanity check for lr data schedule",
    )
    