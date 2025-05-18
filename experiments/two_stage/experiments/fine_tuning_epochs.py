from marin.execution.executor import output_path_of, executor_main

from experiments.two_stage.two_stage_config import TwoStageConfig, two_stage_train_step

if __name__ == "__main__":
    NUM_RARE_STEPS = 1.0
    train_steps = [
        two_stage_train_step(
            TwoStageConfig(
                rare_data_name=rare_data_name,
                common_data_name="c4",
                rare_fraction=1.0 / rare_data_epochs,
                rare_stage2_allocation=1.0,
                stage2_duration=1.0,
                rare_data_epochs=rare_data_epochs,
                num_train_steps=NUM_RARE_STEPS * rare_data_epochs,
                lr_schedule=lr_schedule,
                lr=lr,
                wandb_project_name="suhas-two-stage",
                wandb_additional_tags=["fine-tuning-5-12-epochs_v3", f"{rare_data_name}-c4-fine-tuning-epochs-v3"],
                model_name="150m4k",
                initialize_from_hf=f"gs://marin-us-central2/checkpoints/two_stage/150m4k-4.2B-finemathx0.000x1-c4-rr1.00-rs1.00-cos-0.001-na-pt/hf/step-999",
                nametag="-fe"
            )
        )
        for lr, lr_schedule in [
            (1e-3, "cosine"),
        ]
        for rare_data_epochs in [128, 256]
        for rare_data_name in ["finemath", "spj", "flan", "starcoder"]
        # for rare_data_epochs in [16]
        # for rare_data_name in ["finemath"]
    ]

    executor_main(
        steps=train_steps,
        description="Sanity check for lr data schedule",
    )
    