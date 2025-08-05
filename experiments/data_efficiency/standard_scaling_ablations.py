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
            wandb_additional_tags=[additional_tag],
            model_name=model_name,
            nametag=f"-bs{batch_size}",
        )
    )
    ## Batch size
    # for base_train_steps, batch_size in [
    #     (800, 64),
    #     (400, 128),
    #     (200, 256),
    #     (100, 512),
    # ]
    # for epochs in [1]
    # for weight_decay in [0.1]
    # for model_name, lr in [("300m4k", 3e-3)]
    # for additional_tag in ["batch-size-test-8-4"]
    ## Epoch overfitting
    # for base_train_steps in [800]
    # for epochs in [1, 2, 4, 8, 16, 32, 64, 128]
    # for weight_decay in [0.1]
    # for batch_size in [64]
    # for model_name, lr in [("300m4k", 3e-3)]
    # for additional_tag in ["epoch-overfitting-8-4"]
    ## Weight decay
    for base_train_steps in [800]
    for batch_size in [64]
    for weight_decay in [0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4, 12.8]
    for epochs, model_name, lr in [
        (1, "300m4k", 3e-3),
        (16, "300m4k", 3e-3),
        (8, "1_4b4k", 1e-3),
    ]
    for additional_tag in ["weight-decay-8-4"]
]

if __name__ == "__main__":
    executor_main(
        steps=train_steps,
        description="Data scaling baseline",
    )
