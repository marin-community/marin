from experiments.data_efficiency.train import DataEfficiencyConfig, data_efficiency_train_step
from marin.execution.executor import executor_main

# 800 steps ==> 200M tokens

# tasks = [

# ]

train_steps = [
    data_efficiency_train_step(
        DataEfficiencyConfig(
            data_name="octo",
            epochs=epochs,
            base_train_steps=base_train_steps,
            train_batch_size=batch_size,
            lr_schedule="cosine",
            lr=lr,
            weight_decay=weight_decay,
            wandb_project_name="suhas-cpt-data-efficiency",
            wandb_additional_tags=["octothinker-cpt"],
            model_name="l3b",
            nametag=f"-bs{batch_size}",
            initialize_from_hf=initialize_from_hf,
        )
    )
    for base_train_steps in [200]
    for weight_decay in [0.1]
    for initialize_from_hf in [
        "meta-llama/Llama-3.2-3B",
    ]
    for lr in [1e-5, 3e-5, 1e-4]
    for epochs in [1]
    for batch_size in [512]
]

if __name__ == "__main__":
    executor_main(
        steps=train_steps,
        description="Octothinker",
    )