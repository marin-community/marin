"""
A simple scaling law experiment (to serve as an example or test script) to predict the performance of a 8B model
from 5 smaller models.
"""

from experiments.defaults import default_scaling_law_pred
from experiments.evals.task_configs import CORE_TASKS
from marin.execution.executor import executor_main

RUNS = [
    "tootsie-scaling-512-81c36c",
    "tootsie-scaling-768-d17a90",
    "tootsie-scaling-1024-f4e4be",
    "tootsie-scaling-1536-e2a6d8",
    "tootsie-scaling-2048-72c648",
]

# PRED_RUN = "llama-70b-tootsie-dummy-testing-986d5d" #llama-8b-tootsie-0.001-19ad63"
PRED_RUN = "llama-8b-tootsie-0.001-19ad63"

TASK_BPB_RUNS = [
    "tootsie-scaling-soft-metrics-512-64dff5",
    "tootsie-scaling-soft-metrics-768-e6d0bb",
    "tootsie-scaling-soft-metrics-1024-ef3791",
    "tootsie-scaling-soft-metrics-1536-b928e4",
    "tootsie-scaling-soft-metrics-2048-c18a5c",
]

TASK_BPBS_TO_PREDICT = [
    "lm_eval/hellaswag_10shot/bpb",
    "lm_eval/hellaswag_0shot/bpb",
    "lm_eval/boolq/bpb",
    "lm_eval/copa/bpb",
]

scaling_law_8b_performance_pred = default_scaling_law_pred(
    ladder_runs=RUNS,
    pred_run=PRED_RUN,
    task_losses=(
        "eval/paloma/c4_en/bpb",
        "eval/bpb",
        "eval/loss",
    ),
    task_accuracies=CORE_TASKS,
)

scaling_law_bpb_projection = default_scaling_law_pred(
    ladder_runs=TASK_BPB_RUNS,
    pred_run=None,
    task_losses=TASK_BPBS_TO_PREDICT,
)

if __name__ == "__main__":
    executor_main(
        steps=[
            # scaling_law_8b_performance_pred,
            scaling_law_bpb_projection,
        ],
        description="scaling law suite to predict performance of 8B model on DCLM mix",
    )
