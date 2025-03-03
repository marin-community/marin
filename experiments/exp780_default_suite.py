"""
This script runs a suite of scaling laws on the DCLM-Baseline+StarCoder+ProofPile mix.
This is the default mix that we use for our experiments/scaling laws, and can be used
as a reference point to compare other mixes/scaling law suites against.

Link to issue for scaling law experiments: https://github.com/stanford-crfm/marin/issues/780
"""

from defaults import default_scaling_law_pred

from experiments.evals.task_configs import CORE_TASKS
from experiments.exp600_tootsie import dclm_mixture_config_llama3
from marin.execution.executor import executor_main
from marin.scaling_laws.create_ladder_suite import scaling_law_suite

default_suite = scaling_law_suite(
    sweep_name="scaling-law-suite-default-v2",
    tokenized=dclm_mixture_config_llama3,
    tags=["scaling_laws"],
)


default_suite_scaling_laws_pred = default_scaling_law_pred(
    ladder_runs=default_suite,
    # TODO: corresponds to llama_8b_tootsie in exp600_tootsie.py; used wandb ID for now out of caution
    # to avoid accidentally re-running the 8B model
    pred_run="llama-8b-tootsie-0.001-19ad63",
    task_losses=(
        "eval/paloma/c4_en/bpb",
        "eval/bpb",
        "eval/loss",
        "eval/paloma/c4_en/loss",
    ),
    task_accuracies=CORE_TASKS,
)


if __name__ == "__main__":
    executor_main(
        steps=[
            *default_suite,
            # default_suite_scaling_laws_pred,
        ],
        description="suite + predictions for scaling laws on DCLM-Baseline+StarCoder+ProofPile mix",
    )
