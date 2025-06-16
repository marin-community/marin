"""
Searching for the best two-stage data schedule (based on two_stage_config.py).
Mid-training setup where we have a single learning rate schedule and a different mixture for each stage.
Hyperparameters are tuned for the baseline of all data at the end of training.
"""

from experiments.data_efficiency.train import DataEfficiencyConfig, data_efficiency_train_step
from marin.execution.executor import executor_main

if __name__ == "__main__":
    train_steps = [
        data_efficiency_train_step(
            DataEfficiencyConfig(
                train_seed=seed,
                data_seed=seed,
                data_name="dclm",
                epochs=epochs,
                base_train_steps=base_train_steps * 1024 / batch_size,
                train_batch_size=batch_size,
                lr_schedule="cosine",
                lr=lr,
                weight_decay=weight_decay,
                wandb_project_name="suhas-data-efficiency",
                wandb_additional_tags=["ensemble-members-6-13"],
                model_name=model_name,
                nametag=f"-seed{seed}",
            )
        )
        for base_train_steps, epochs, lr, weight_decay in [
            (50, 16, 3e-3, 1.6),
            (100, 16, 3e-3, 0.8),
            (200, 16, 3e-3, 0.4),
            (400, 32, 3e-3, 0.8),
        ]
        for batch_size in [64]
        for model_name in ["300m4k"]
        for seed in list(range(10))
    ]

    executor_main(
        steps=train_steps,
        description="Data scaling baseline",
    )
