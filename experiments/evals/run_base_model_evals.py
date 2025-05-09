"""
Base model evaluations across multiple LLMs.

This experiment evaluates OLMO Base 8B, LLAMA 3.1 8B, Deeper Starling 8B,
MAP-NEO 7B, and Amber Base 7B models on CORE_TASKS (augmented with GPQA) as well
as dedicated MMLU 0-shot and 5-shot configurations.
"""

from experiments.evals.evals import default_eval
from experiments.evals.resource_configs import SINGLE_TPU_V6E_8
from experiments.evals.task_configs import CORE_TASKS
from experiments.models import amber_base_7b, llama_3_1_8b, map_neo_7b, olmo_2_base_8b
from marin.evaluation.evaluation_config import EvalTaskConfig
from marin.execution.executor import executor_main

# Model path for deeper starling
deeper_starling_path = "gs://marin-us-central2/checkpoints/tootsie-8b-deeper-starling/hf/step-1419999"

# Add GPQA to CORE_TASKS
core_tasks_with_gpqa = CORE_TASKS + (
    EvalTaskConfig(
        "leaderboard_gpqa",
        0,
        task_alias="gpqa_0shot",
    ),
)

# Create reusable EvalTaskConfig objects
mmlu_0shot_config = [EvalTaskConfig("mmlu", 0, task_alias="mmlu_0shot")]
mmlu_5shot_config = [EvalTaskConfig("mmlu", 5, task_alias="mmlu_5shot")]

# Set up evaluations for core tasks (including GPQA)
olmo_base_core = default_eval(step=olmo_2_base_8b, resource_config=SINGLE_TPU_V6E_8, evals=core_tasks_with_gpqa)

llama_core = default_eval(step=llama_3_1_8b, resource_config=SINGLE_TPU_V6E_8, evals=core_tasks_with_gpqa)

deeper_starling_core = default_eval(
    step=deeper_starling_path,
    resource_config=SINGLE_TPU_V6E_8,
    evals=core_tasks_with_gpqa,
)

map_neo_core = default_eval(
    step=map_neo_7b,
    resource_config=SINGLE_TPU_V6E_8,
    evals=core_tasks_with_gpqa,
)

amber_base_core = default_eval(
    step=amber_base_7b,
    resource_config=SINGLE_TPU_V6E_8,
    evals=core_tasks_with_gpqa,
)

# Set up evaluations for MMLU 0-shot (run separately for macro avg calculation)
olmo_base_mmlu_0shot = default_eval(step=olmo_2_base_8b, resource_config=SINGLE_TPU_V6E_8, evals=mmlu_0shot_config)

llama_mmlu_0shot = default_eval(step=llama_3_1_8b, resource_config=SINGLE_TPU_V6E_8, evals=mmlu_0shot_config)

deeper_starling_mmlu_0shot = default_eval(
    step=deeper_starling_path,
    resource_config=SINGLE_TPU_V6E_8,
    evals=mmlu_0shot_config,
)

map_neo_mmlu_0shot = default_eval(
    step=map_neo_7b,
    resource_config=SINGLE_TPU_V6E_8,
    evals=mmlu_0shot_config,
)

amber_base_mmlu_0shot = default_eval(
    step=amber_base_7b,
    resource_config=SINGLE_TPU_V6E_8,
    evals=mmlu_0shot_config,
)

# Set up evaluations for MMLU 5-shot (run separately for macro avg calculation)
olmo_base_mmlu_5shot = default_eval(step=olmo_2_base_8b, resource_config=SINGLE_TPU_V6E_8, evals=mmlu_5shot_config)

llama_mmlu_5shot = default_eval(step=llama_3_1_8b, resource_config=SINGLE_TPU_V6E_8, evals=mmlu_5shot_config)

deeper_starling_mmlu_5shot = default_eval(
    step=deeper_starling_path,
    resource_config=SINGLE_TPU_V6E_8,
    evals=mmlu_5shot_config,
)

map_neo_mmlu_5shot = default_eval(
    step=map_neo_7b,
    resource_config=SINGLE_TPU_V6E_8,
    evals=mmlu_5shot_config,
)

amber_base_mmlu_5shot = default_eval(
    step=amber_base_7b,
    resource_config=SINGLE_TPU_V6E_8,
    evals=mmlu_5shot_config,
)

if __name__ == "__main__":
    # Run all evaluations on all models
    executor_main(
        steps=[
            # Core tasks evaluations
            olmo_base_core,
            llama_core,
            deeper_starling_core,
            map_neo_core,
            amber_base_core,
            # MMLU 0-shot evaluations
            olmo_base_mmlu_0shot,
            llama_mmlu_0shot,
            deeper_starling_mmlu_0shot,
            map_neo_mmlu_0shot,
            amber_base_mmlu_0shot,
            # MMLU 5-shot evaluations
            olmo_base_mmlu_5shot,
            llama_mmlu_5shot,
            deeper_starling_mmlu_5shot,
            map_neo_mmlu_5shot,
            amber_base_mmlu_5shot,
        ]
    )
