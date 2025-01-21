from experiments.defaults import default_scaling_law_analysis
from experiments.exp600_tootsie import dclm_mixture_config_llama3, llama_8b_tootsie
from marin.execution.executor import executor_main
from marin.training.scaling_laws import scaling_law_suite

TAG = ["654_scaling_tootsie"]

suite = scaling_law_suite(sweep_name="tootsie-scaling-soft-metrics", tokenized=dclm_mixture_config_llama3, tags=TAG)

# in addition to training the 8B model, fit scaling laws to predict its performance
scaling_law_8b_performance_pred = default_scaling_law_analysis(
    ladder_runs=[
        *suite,
    ],
    pred_run=llama_8b_tootsie,
    intermediate_task_loss="eval/paloma/c4_en/bpb",
    task_accuracies=["lm_eval/hellaswag_10shot/acc"],
)

if __name__ == "__main__":
    executor_main(
        steps=[
            *suite,
            scaling_law_8b_performance_pred,
        ],
        description="scaling law suite to predict performance of 8B model on DCLM mix",
    )
