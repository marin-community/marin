from marin.execution.executor import output_path_of, executor_main
from marin.evaluation.evaluation_config import EvalTaskConfig

from experiments.two_stage.two_stage_config import TwoStageConfig, two_stage_train_step
from experiments.two_stage.experiments.pretraining import pretraining_configs, NUM_PRETRAINING_STEPS


BASQUE_TASKS = (
    EvalTaskConfig("xcopa_eu", num_fewshot=0, task_alias="xcopa_eu"),
)

if __name__ == "__main__":
    NUM_RARE_STEPS = 400.0 # 40M tokens
    train_steps = [
        two_stage_train_step(
            TwoStageConfig(
                rare_data_name=rare_data_name,
                common_data_name="spj",
                rare_fraction=NUM_RARE_STEPS / (base_num_train_steps * rare_data_epochs),
                rare_stage2_allocation=1.0,
                stage2_duration=1.0,
                rare_data_epochs=rare_data_epochs,
                num_train_steps=base_num_train_steps * rare_data_epochs,
                lr_schedule=lr_schedule,
                lr=lr,
                lr_cooldown_duration=lr_cooldown_duration,
                train_batch_size=128,
                wandb_project_name="suhas-two-stage",
                wandb_additional_tags=["cpt", f"{rare_data_name}-spj-cpt"],
                model_name="l8b",
                initialize_from_hf="meta-llama/Meta-Llama-3.1-8B",
                eval_harness_tasks=BASQUE_TASKS,
            )
        )
        for lr_schedule, lr_cooldown_duration in [
            ("cosine", 1.0),
        ]
        for lr in [1e-5]
        for base_num_train_steps in [NUM_RARE_STEPS, NUM_RARE_STEPS * 2, NUM_RARE_STEPS * 4, NUM_RARE_STEPS * 10]
        for rare_data_name in ["latxa"]
        for rare_data_epochs in [1, 2, 4]
    ]

    executor_main(
        steps=train_steps,
        description="Sanity check for lr data schedule",
    )
    