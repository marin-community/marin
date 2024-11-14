from experiments.evals.evals import evaluate_lm_evaluation_harness
from marin.execution.executor import executor_main

evaluate_resiliparse_custom_fork = evaluate_lm_evaluation_harness(
    model_name="fineweb-small-resiliparse-custom-fork-1.4b",
    model_path="gs://marin-us-central2/checkpoints/fineweb-small-resiliparse-custom-fork-1.4b-9518f2/hf/step-9999",
    evals=["mmlu"],
)

if __name__ == "__main__":
    executor_main(steps=[evaluate_resiliparse_custom_fork])