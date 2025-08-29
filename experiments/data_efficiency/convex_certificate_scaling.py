from experiments.data_efficiency.train import DataEfficiencyConfig, data_efficiency_train_step
from experiments.data_efficiency.utils import get_bounding_box
from marin.execution.executor import executor_main

# 400 steps ==> 100M tokens
# 800 steps ==> 200M tokens
# 1600 steps ==> 400M tokens
# 3200 steps ==> 800M tokens
# 6400 steps ==> 1600M tokens


train_steps = [
    [
        data_efficiency_train_step(
            DataEfficiencyConfig(
                data_name="dclm",
                epochs=epochs,
                base_train_steps=base_train_steps,
                train_batch_size=64,
                lr_schedule="cosine",
                lr=lr,
                weight_decay=weight_decay,
                wandb_project_name="suhas-data-efficiency",
                model_name=model_name,
                nametag="-bs64",
            )
        )
        for base_train_steps, epochs, lr, weight_decay, model_name in get_bounding_box(*candidate_hparams)
    ]
    for candidate_hparams in [
        # (800, 16, 3e-3, 0.8, "150m4k"),
        # (800, 16, 3e-3, 6.4, "150m4k"),
        # (800, 16, 3e-3, 1.6, "300m4k"),
        # (800, 8, 1e-3, 3.2, "600m4k"),
        # (800, 8, 1e-3, 3.2, "1_4b4k"),
        # (1600, 32, 3e-3, 0.8, "150m4k"),
        # (1600, 16, 3e-3, 0.8, "300m4k"),
        # (1600, 8, 1e-3, 1.6, "600m4k"),
        # (1600, 8, 1e-3, 3.2, "1_4b4k"),
        # (3200, 64, 3e-3, 0.4, "150m4k"),
        # (3200, 16, 3e-3, 0.4, "300m4k"),
        # (3200, 16, 3e-3, 0.4, "600m4k"),
        # (3200, 8, 1e-3, 1.6, "1_4b4k"),
        # (6400, 64, 3e-3, 0.1, "150m4k"),
        # (6400, 32, 1e-3, 0.4, "300m4k"),
        # (6400, 16, 1e-3, 0.8, "600m4k"),
        # (6400, 8, 1e-3, 0.8, "1_4b4k"),
        # (800, 8, 3e-3, 0.1, "150m4k"),
        # (800, 8, 1e-3, 0.1, "300m4k"),
        # (800, 4, 1e-3, 0.1, "600m4k"),
        # (800, 4, 3e-4, 0.1, "1_4b4k"),
        # (800, 8, 1e-3, 3.2, "olmoe"),
        # (800, 16, 3e-3, 1.6, "300moe"),
    ]
]

train_steps = [step for sublist in train_steps for step in sublist]

if __name__ == "__main__":
    executor_main(
        steps=train_steps,
        description="Data scaling baseline",
    )
