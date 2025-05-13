from marin.execution.executor import output_path_of, executor_main
from experiments.two_stage.two_stage_config import TwoStageConfig, two_stage_train_step

def pretraining_for_fixed_steps(steps: int):
    return two_stage_train_step(
        TwoStageConfig(
            rare_data_name="finemath",
            common_data_name="c4",
            rare_fraction=0.0,
            rare_stage2_allocation=1.0,
            stage2_duration=1.0,
            num_train_steps=steps,
            lr_schedule="cosine",
            lr=1e-3,
            wandb_project_name="suhas-two-stage",
            wandb_additional_tags=["joint-pt-5-12"],
            model_name="150m4k",
            nametag=f"-{int(steps)}"
        )
    )

def finetuning_with_replay(
    rare_data_name: str,
    rare_data_epochs: int,
    num_rare_steps: int,
    replay_multiplier: int,
    num_total_steps: int,
    lr: float,
):
    num_fine_tuning_steps = int(num_rare_steps * rare_data_epochs * replay_multiplier)
    assert num_fine_tuning_steps <= num_total_steps
    num_pretraining_steps = num_total_steps - num_fine_tuning_steps
    pretraining_step = pretraining_for_fixed_steps(num_pretraining_steps)

    return two_stage_train_step(
        TwoStageConfig(
            rare_data_name=rare_data_name,
            common_data_name="c4",
            rare_fraction=float(num_rare_steps) / num_fine_tuning_steps,
            rare_stage2_allocation=1.0,
            stage2_duration=1.0,
            rare_data_epochs=rare_data_epochs,
            num_train_steps=num_fine_tuning_steps,
            lr_schedule="cosine",
            lr=lr,
            wandb_project_name="suhas-two-stage",
            wandb_additional_tags=["joint-ft-5-12", f"{rare_data_name}-c4-fine-tuning-v4"],
            model_name="150m4k",
            initialize_from_hf=output_path_of(pretraining_step).cd(f"hf/step-{num_pretraining_steps - 1}"),
            nametag="-j"
        )
    )

if __name__ == "__main__":
    NUM_RARE_STEPS = 4
    TOTAL_STEPS = 1024
    train_steps = [
        finetuning_with_replay(
            rare_data_name=rare_data_name,
            rare_data_epochs=rare_data_epochs,
            replay_multiplier=replay_multiplier,
            num_rare_steps=NUM_RARE_STEPS,
            num_total_steps=TOTAL_STEPS,
            lr=lr,
        )
        for replay_multiplier in [1, 1.5, 2, 3, 4]
        for lr in [1e-3, 3e-4]
        for rare_data_name, rare_data_epochs in [
            ("finemath", 32),
            ("spj", 32),
            ("flan", 32),
            ("starcoder", 32),
        ]
    ]

    executor_main(
        steps=train_steps,
        description="Sanity check for lr data schedule",
    )

