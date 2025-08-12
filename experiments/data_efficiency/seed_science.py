"""
Searching for the best two-stage data schedule (based on two_stage_config.py).
Mid-training setup where we have a single learning rate schedule and a different mixture for each stage.
Hyperparameters are tuned for the baseline of all data at the end of training.
"""

from experiments.data_efficiency.train import (
    DataEfficiencyConfig,
    data_efficiency_train_step,
    data_efficiency_eval_ensemble,
)
from marin.execution.executor import executor_main

key = "seed-science-8-7"

seed_science_train_steps_dict = {
    (train_seed, data_seed): data_efficiency_train_step(
        DataEfficiencyConfig(
            train_seed=train_seed,
            data_seed=data_seed,
            data_name="dclm",
            epochs=epochs,
            base_train_steps=base_train_steps,
            train_batch_size=batch_size,
            lr_schedule="cosine",
            lr=lr,
            weight_decay=weight_decay,
            wandb_project_name="suhas-data-efficiency",
            wandb_additional_tags=[key],
            model_name=model_name,
            nametag=f"-ts{train_seed}-ds{data_seed}",
            tpu_type="v4-16",
        )
    )
    for base_train_steps in [800]
    for epochs in [16]
    for lr in [3e-3]
    for weight_decay in [1.6]
    for batch_size in [64]
    for model_name in ["300m4k"]
    for train_seed, data_seed in [(0, i) for i in range(5)]
    + [(i, 0) for i in range(1, 5)]
    + [(i, i) for i in range(1, 5)]
}

seed_science_train_steps = list(seed_science_train_steps_dict.values())

ensemble_eval_steps = [
    [
        data_efficiency_eval_ensemble(
            [seed_science_train_steps_dict[(0, i)] for i in range(max_runs - 1, -1, -1)],
            key=key,
            run_prefix="ss",
            name_prefix="",
        ),
        data_efficiency_eval_ensemble(
            [seed_science_train_steps_dict[(i, 0)] for i in range(max_runs - 1, -1, -1)],
            key=key,
            run_prefix="ss",
            name_prefix="",
        ),
        data_efficiency_eval_ensemble(
            [seed_science_train_steps_dict[(i, i)] for i in range(max_runs - 1, -1, -1)],
            key=key,
            run_prefix="ss",
            name_prefix="",
        ),
    ]
    for max_runs in range(1, 6)
]

ensemble_eval_steps = [step for sublist in ensemble_eval_steps for step in sublist]

if __name__ == "__main__":
    executor_main(
        steps=seed_science_train_steps + ensemble_eval_steps,
        description="Seed science",
    )
