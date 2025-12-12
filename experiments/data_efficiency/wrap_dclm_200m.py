from experiments.data_efficiency.train import DataEfficiencyConfig, data_efficiency_train_step
from marin.execution.executor import executor_main

train_steps = [
    data_efficiency_train_step(
        DataEfficiencyConfig(
            data_name="dclm_200m",
            val_name="dclm_200m_val",
            teacher_data_name=synthetic_data_name,
            teacher_data_weight=synthetic_data_weight,
            epochs=epochs,
            base_train_steps=base_train_steps,
            train_batch_size=64,
            lr_schedule="cosine",
            lr=lr,
            weight_decay=weight_decay,
            wandb_project_name="suhas-data-efficiency",
            wandb_additional_tags=[synthetic_data_name],
            model_name=model_name,
            nametag=f"-bs64" + nametag,
            initialize_from_hf=initialize_from_hf,
            tpu_type="v4-64"
        )
    )
    for base_train_steps in [750]
    for weight_decay in [0.1, 1.6]
    for synthetic_data_name, initialize_from_hf, nametag, lr, epochs in [
        ("hq_cpr16", None, "", 3e-3, 1),
        ("hq_cpr16", None, "", 3e-3, 2),
        ("hq_cpr16", None, "", 3e-3, 4),
        ("hq_cpr16", None, "", 3e-3, 8),
        ("hq_cpr16", None, "", 3e-3, 16),
    ]
    for synthetic_data_weight in [0.5, 0.75]
    for model_name in ["150m4k"]
]

if __name__ == "__main__":
    executor_main(
        steps=train_steps,
        description="Wrap DCLM 200M"
    )
