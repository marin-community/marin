from experiments.evals.evals import evaluate_alpaca_eval
from experiments.evals.resource_configs import SINGLE_TPU_V6E_8
from marin.execution.executor import ExecutorMainConfig, executor_main

executor_main_config = ExecutorMainConfig()
steps = [
    evaluate_alpaca_eval(
        model_name="pf5pe4ut/step-600",
        model_path="gs://marin-us-central2/checkpoints/quickstart_single_script_docker_test_09_18/"
        "pf5pe4ut/hf/pf5pe4ut/step-600",
        resource_config=SINGLE_TPU_V6E_8,
    ),
]

if __name__ == "__main__":
    executor_main(executor_main_config, steps=steps)
