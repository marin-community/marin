from experiments.two_stage.two_stage_config import TwoStageConfig, two_stage_train_step
from marin.execution.executor import executor_main

NUM_PRETRAINING_STEPS = 1000

pretraining_configs = [
    TwoStageConfig(
        rare_data_name=rare_data_name,
        common_data_name="c4",
        rare_fraction=rare_fraction,
        stage2_duration=stage2_duration,
        rare_stage2_allocation=rare_stage2_allocation,
        num_train_steps=NUM_PRETRAINING_STEPS,
        lr_schedule=lr_schedule,
        lr=1e-3,
        lr_cooldown_duration=lr_cooldown_duration,
        wandb_project_name="suhas-two-stage",
        wandb_additional_tags=["efficiency-baselines", f"{rare_data_name}-c4-pretrain-only"],
        model_name="150m4k",
        nametag="-pt",
    )
    for lr_schedule, lr_cooldown_duration in [
        ("cosine", 1.0),
    ]
    for rare_fraction in [0.0]
    for stage2_duration in [1.0]
    for rare_stage2_allocation in [1.0]
    for rare_data_name in ["finemath"]
]

pretraining_steps = [
    two_stage_train_step(config)
    for config in pretraining_configs
]

if __name__ == "__main__":
    executor_main(
        steps=pretraining_steps,
        description="Pretraining",
    )
    