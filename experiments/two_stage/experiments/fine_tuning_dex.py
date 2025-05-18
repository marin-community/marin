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
                num_train_steps=int(num_rare_steps * rare_data_epochs),
                lr_schedule=lr_schedule,
                lr=lr,
                wandb_project_name="suhas-two-stage",
                wandb_additional_tags=["fine-tuning-scaling-law-v2", f"{rare_data_name}-c4-fine-tuning-scaling-law-v2"],
                model_name="150m4k",
                initialize_from_hf=f"gs://marin-us-central2/checkpoints/two_stage/150m4k-4.2B-finemathx0.000x1-c4-rr1.00-rs1.00-cos-0.001-na-pt/hf/step-999",
                nametag="-s2"
            )
        )
        for lr, lr_schedule in [
            (3e-4, "cosine"),
            (1e-3, "cosine"),
        ]
        for rare_data_name, rare_data_epochs in [
            ("finemath", 64),
            # ("spj", 64),
            ("flan", 64),
            ("starcoder", 64),
        ]
        for num_rare_steps in [1, 2, 4]
    ]

    executor_main(
        steps=train_steps,
        description="Sanity check for lr data schedule",
    )
    