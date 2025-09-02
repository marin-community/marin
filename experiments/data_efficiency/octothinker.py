from experiments.data_efficiency.train import DataEfficiencyConfig, data_efficiency_train_step
from marin.evaluation.evaluation_config import EvalTaskConfig
from marin.execution.executor import executor_main

# 4000 steps ==> 1B tokens
# 280000 steps ==> 70B tokens

tasks = [
    EvalTaskConfig(name="mathqa", num_fewshot=8),
]

train_steps = [
    data_efficiency_train_step(
        DataEfficiencyConfig(
            data_name="octo",
            epochs=epochs,
            base_train_steps=base_train_steps * (64 / batch_size),
            train_batch_size=batch_size,
            lr_schedule="cosine",
            lr=lr,
            weight_decay=weight_decay,
            wandb_project_name="suhas-cpt-data-efficiency",
            wandb_additional_tags=["octothinker-cpt"],
            model_name="l3b",
            nametag=f"-bs{batch_size}" + (f"-seed{seed}" if seed is not None else ""),
            initialize_from_hf=initialize_from_hf,
            eval_harness_tasks=tasks,
            train_seed=seed if seed else 0,
            data_seed=seed if seed else 0,
            tpu_type="v4-64",
            # per_device_parallelism=2,
        )
    )
    for base_train_steps in [16_000]
    for weight_decay in [0.1]
    for initialize_from_hf in [
        "meta-llama/Llama-3.2-3B",
    ]
    for lr in [3e-5]
    for epochs in [8]
    for batch_size in [64]
    for seed in [0, 1, 2, 3]
]

if __name__ == "__main__":
    executor_main(
        steps=train_steps,
        description="Octothinker",
    )
