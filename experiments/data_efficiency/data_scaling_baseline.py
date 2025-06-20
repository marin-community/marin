from experiments.data_efficiency.train import DataEfficiencyConfig, data_efficiency_train_step
from marin.execution.executor import executor_main

train_steps = [
    data_efficiency_train_step(
        DataEfficiencyConfig(
            data_name="dclm",
            epochs=epochs,
            base_train_steps=base_train_steps * 1024 / batch_size,
            train_batch_size=batch_size,
            lr_schedule="cosine",
            lr=lr,
            weight_decay=weight_decay,
            wandb_project_name="suhas-data-efficiency",
            wandb_additional_tags=["data-scaling-laws-6-10"],
            model_name=model_name,
            nametag=f"-suTEST1-bs{batch_size}",
        )
    )
    for base_train_steps in [50]
    for epochs in [1]
    for lr in [3e-3]
    for weight_decay in [0.8]
    for batch_size in [64]
    for model_name in ["300m4k"]
]

if __name__ == "__main__":
    executor_main(
        steps=train_steps,
        description="Data scaling baseline",
    )
