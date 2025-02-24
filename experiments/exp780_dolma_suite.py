"""
This script runs a suite of scaling laws on the Dolma mix.

Link to issue: https://github.com/stanford-crfm/marin/issues/780
"""

from experiments.dolma.exp442_dolma import dolma_llama3_tokenized
from marin.execution.executor import executor_main
from marin.scaling_laws.create_ladder_suite import scaling_law_suite
from defaults import default_scaling_law_pred
from experiments.evals.task_configs import CORE_TASKS

dolma_suite = scaling_law_suite(
    sweep_name="scaling-law-suite-dolma",
    tokenized=dolma_llama3_tokenized,
    tags=["scaling_laws"],
)


RUNS = [
    "tootsie-scaling-512-81c36c",
    "tootsie-scaling-768-d17a90",
    "tootsie-scaling-1024-f4e4be",
    "tootsie-scaling-1536-e2a6d8",
    "tootsie-scaling-2048-72c648",
]

dolma_suite_scaling_laws_pred = default_scaling_law_pred(
    ladder_runs=dolma_suite[:-1],
    pred_run="scaling-law-suite-dolma-2048-acd8a1",
    #"llama-22b-tootsie-dummy-testing-373d53",#"llama-13b-tootsie-ema","llama-8b-tootsie-0.001-19ad63",
    task_losses=(
        "eval/paloma/c4_en/bpb",
        "eval/bpb",
        "eval/loss",
        # "lm_eval/hellaswag_10shot/bpb",
        # "lm_eval/hellaswag_0shot/bpb",
        # "lm_eval/boolq/bpb",
        # "lm_eval/copa/bpb",
    ),
    #task_accuracies=None
    task_accuracies=None,#CORE_TASKS[4:9],
)


if __name__ == "__main__":
    executor_main(
        steps=[
            #*dolma_suite,
            dolma_suite_scaling_laws_pred,
        ],
        description="suite for scaling laws on Dolma mix",
    )
