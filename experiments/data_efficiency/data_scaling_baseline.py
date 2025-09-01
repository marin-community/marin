from experiments.data_efficiency.train import DataEfficiencyConfig, data_efficiency_train_step
from marin.execution.executor import executor_main

# 800 steps ==> 200M tokens

train_steps = [
    data_efficiency_train_step(
        DataEfficiencyConfig(
            data_name="dclm",
            epochs=epochs,
            base_train_steps=base_train_steps,
            train_batch_size=batch_size,
            lr_schedule="cosine",
            lr=lr,
            weight_decay=weight_decay,
            wandb_project_name="suhas-data-efficiency",
            # wandb_additional_tags=["weight-decay-test-6-22"],
            model_name=model_name,
            nametag=f"-bs{batch_size}",
            tpu_type="v4-64",
            per_device_parallelism=2,
        )
    )
    for base_train_steps in [800]
    for epochs in [16]
    for weight_decay in [0.1]
    for batch_size in [64]
    for model_name, lr, epochs in [
        # ("300m4k", 3e-3),
        # ("300m4k", 1e-3),
        # ("300m4k", 3e-4),
        # ("300moe", 3e-3),
    ]
]

if __name__ == "__main__":
    executor_main(
        steps=train_steps,
        description="Data scaling baseline",
    )
