from marin.execution.executor import output_path_of, executor_main

from experiments.two_stage.two_stage_config import TwoStageConfig, two_stage_train_step

if __name__ == "__main__":
    train_steps = [
        two_stage_train_step(
            TwoStageConfig(
                rare_data_name=rare_data_name,
                common_data_name="c4",
                rare_fraction=1.0 / rare_data_epochs,
                rare_stage2_allocation=1.0,
                stage2_duration=1.0,
                rare_data_epochs=rare_data_epochs,
                num_train_steps=num_rare_steps * rare_data_epochs,
                lr_schedule=lr_schedule,
                lr=lr,
                lr_cooldown_duration=lr_cooldown_duration,
                wandb_project_name="suhas-two-stage",
                wandb_additional_tags=["fine-tuning-5-11-scaling-law", f"{rare_data_name}-c4-fine-tuning-scaling-law"],
                model_name="150m4k",
                initialize_from_hf=f"gs://marin-us-central2/checkpoints/two_stage/150m4k-4.2B-finemathx0.00x1-c4-rr1.00-rs1.00-cos-0.001-1.00-pt/hf/step-999",
                nametag="-fsl" if rare_data_name != "starcoder" else "-fs"
            )
        )
        for lr, lr_schedule, lr_cooldown_duration in [
            (1e-3, "cosine", 1.0),
        ]
        for rare_data_name, rare_data_epochs in [
            ("finemath", 16),
            ("spj", 16),
            ("flan", 16),
            ("starcoder", 16),
        ]
        for num_rare_steps in [5, 10, 20, 40]
    ]

    executor_main(
        steps=train_steps,
        description="Sanity check for lr data schedule",
    )
    