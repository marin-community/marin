from experiments.data_efficiency.train import DataEfficiencyConfig, data_efficiency_train_step
from marin.execution.executor import executor_main

train_steps = [
    data_efficiency_train_step(
        DataEfficiencyConfig(
            data_name="dclm",
            teacher_data_name=teacher_data_name,
            teacher_data_weight=teacher_data_weight,
            epochs=epochs,
            base_train_steps=base_train_steps,
            train_batch_size=64,
            lr_schedule="cosine",
            lr=lr,
            weight_decay=weight_decay,
            wandb_project_name="suhas-data-efficiency",
            wandb_additional_tags=["self-distill-7-15" if teacher_data_name == "sd0715" else teacher_data_name],
            model_name=model_name,
            nametag=nametag,
            initialize_from_hf=initialize_from_hf,
        )
    )
    for base_train_steps in [800]
    for weight_decay in [0.1]
    for teacher_data_name, initialize_from_hf, nametag, lr, epochs in [
        ("sd0715", None, "", 3e-3, 16),
        ("ens8x0730", None, "", 3e-3, 16),
    ]
    for teacher_data_weight in [0.75, 0.9]
    for model_name in ["300m4k"]
]

if __name__ == "__main__":
    executor_main(
        steps=train_steps,
        description="Data scaling baseline",
    )
