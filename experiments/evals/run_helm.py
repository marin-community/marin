from experiments.evals.evals import evaluate_helm
from marin.evaluation.evaluation_config import EvalTaskConfig
from marin.execution.executor import ExecutorMainConfig, executor_main

executor_main_config = ExecutorMainConfig()
steps = [
    evaluate_helm(
        model_name="pf5pe4ut/step-600",
        model_path="gs://marin-us-central2/checkpoints/quickstart_single_script_docker_test_09_18/"
        "pf5pe4ut/hf/pf5pe4ut/step-600",
        evals=EvalTaskConfig(name="mmlu", num_fewshot=0),
    ),
]

if __name__ == "__main__":
    executor_main(executor_main_config, steps=steps)
