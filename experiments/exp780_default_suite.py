"""
This script runs a suite of scaling laws on the DCLM-Baseline+StarCoder+ProofPile mix.
This is the default mix that we use for our experiments/scaling laws, and can be used 
as a reference point to compare other mixes/scaling law suites against.

Link to issue for scaling law experiments: https://github.com/stanford-crfm/marin/issues/780
"""

from experiments.exp600_tootsie import dclm_mixture_config_llama3
from marin.execution.executor import executor_main
from marin.scaling_laws.create_ladder_suite import scaling_law_suite
from defaults import default_scaling_law_pred
from experiments.evals.task_configs import CORE_TASKS, CORE_TASKS_PLUS_MMLU


default_suite = scaling_law_suite(
    sweep_name="scaling-law-suite-default",
    tokenized=dclm_mixture_config_llama3,
    tags=["scaling_laws"],
)

# RUNS = [
#     "tootsie-scaling-512-81c36c",
#     "tootsie-scaling-768-d17a90",
#     "tootsie-scaling-1024-f4e4be",
#     "tootsie-scaling-1536-e2a6d8",
#     "tootsie-scaling-2048-72c648",
# ]

default_suite_scaling_laws_pred = default_scaling_law_pred(
    ladder_runs=default_suite[:-1], # all but last run
    #pred_run="llama-8b-tootsie-0.001-19ad63", # default_suite[:],
    pred_run=default_suite[-1],
    #"llama-22b-tootsie-dummy-testing-373d53",#"llama-13b-tootsie-ema","llama-8b-tootsie-0.001-19ad63",
    task_losses=(
        "eval/paloma/c4_en/bpb",
        "eval/bpb",
        "eval/loss",
        "lm_eval/hellaswag_10shot/bpb",
        "lm_eval/hellaswag_0shot/bpb",
        "lm_eval/boolq/bpb",
        "lm_eval/copa/bpb",
    ),
    #task_accuracies=None
    task_accuracies=CORE_TASKS_PLUS_MMLU,
)



if __name__ == "__main__":
    executor_main(
        steps=[
            #*default_suite,
            default_suite_scaling_laws_pred,
        ],
        description="suite + predictions for scaling laws on DCLM-Baseline+StarCoder+ProofPile mix",
    )
