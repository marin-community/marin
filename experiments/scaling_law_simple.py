"""
A simple scaling law experiment (to serve as an example or test script) to predict the performance of a 8B model
from 5 smaller models.
"""

from experiments.defaults import default_scaling_law_analysis
from experiments.evals.task_configs import CORE_TASKS
from marin.execution.executor import executor_main

RUNS = [
    "tootsie-scaling-512-81c36c",
    "tootsie-scaling-768-d17a90",
    "tootsie-scaling-1024-f4e4be",
    "tootsie-scaling-1536-e2a6d8",
    "tootsie-scaling-2048-72c648",
]

PRED_RUN = "llama-8b-tootsie-0.001-19ad63"

scaling_law_8b_performance_pred = default_scaling_law_analysis(
    ladder_runs=RUNS,
    pred_run=PRED_RUN,
    intermediate_task_loss="eval/paloma/c4_en/bpb",
    task_accuracies=CORE_TASKS[4:9], # predict 5 metrics
)

if __name__ == "__main__":
    executor_main(
        steps=[
            scaling_law_8b_performance_pred,
        ],
        description="scaling law suite to predict performance of 8B model on DCLM mix",
    )
