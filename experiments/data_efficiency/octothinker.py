from experiments.data_efficiency.train import DataEfficiencyConfig, data_efficiency_train_step
from marin.execution.executor import executor_main

# 800 steps ==> 200M tokens

tasks = [

]

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
            model_name=model_name,
            nametag=f"-bs{batch_size}" + nametag,
            initialize_from_hf=initialize_from_hf,
        )
    )
    for base_train_steps in [1600]
    for weight_decay in [0.1]
    for initialize_from_hf, nametag in [
        ("meta-llama/Llama-3.2-3B", "llama3b"),
    ]
    for lr in [1e-5, 2e-5, 4e-5]
    for epochs in [1]
    for model_name in ["300m4k"]
    for batch_size in [512]
]

if __name__ == "__main__":
    executor_main(
        steps=train_steps,
        description="Octothinker",
    )