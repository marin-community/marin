from experiments.evals.evals import evaluate_alpaca_eval
from experiments.evals.resource_configs import SINGLE_TPU_V6E_8
from marin.execution.executor import ExecutorMainConfig, executor_main

executor_main_config = ExecutorMainConfig()
steps = [
    evaluate_alpaca_eval(
        model_name="tulu-3-8b-sft",
        model_path="gs://marin-us-east5/gcsfuse_mount/models/allenai--Llama-3-1-Tulu-3-8B-SFT",
        resource_config=SINGLE_TPU_V6E_8,
    ),
]

if __name__ == "__main__":
    executor_main(executor_main_config, steps=steps)
