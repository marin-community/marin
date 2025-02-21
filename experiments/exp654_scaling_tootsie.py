from experiments.defaults import default_scaling_law_pred
from experiments.evals.task_configs import CORE_TASKS
from experiments.exp600_tootsie import dclm_mixture_config_llama3_wrong
from marin.execution.executor import executor_main
from marin.training.scaling_laws import scaling_law_suite

TAG = ["654_scaling_tootsie"]

suite = scaling_law_suite(sweep_name="tootsie-scaling", tokenized=dclm_mixture_config_llama3_wrong, tags=TAG)

# in addition to training the 8B model, fit scaling laws to predict its performance
scaling_law_8b_performance_pred = default_scaling_law_pred(
    ladder_runs=suite,
    pred_run=None,
    task_losses=("eval/paloma/c4_en/bpb", "eval/bpb", "eval/loss"),
    task_accuracies=CORE_TASKS[4:9],  # predict 5 metrics
)

if __name__ == "__main__":
    executor_main(
        steps=[
            *suite,
            scaling_law_8b_performance_pred,
        ],
        description="scaling law suite to predict performance of 8B model on DCLM mix",
    )
