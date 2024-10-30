from experiments.evals.evals import evaluate_helm
from marin.execution.executor import ExecutorMainConfig, executor_main

executor_main_config = ExecutorMainConfig()
steps = [
    evaluate_helm(
        model_name="dclm_7b0828/step-479999",
        model_path="gs://marin-us-central2/checkpoints/dclm_7b2x/hf/dclm_7b0828/dclm_7b0828/step-479999",
        evals=["mmlu"],
    ),
]

if __name__ == "__main__":
    executor_main(executor_main_config, steps=steps)
