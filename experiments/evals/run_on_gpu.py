from experiments.evals.evals import evaluate_lm_evaluation_harness
from marin.execution.executor import ExecutorMainConfig, executor_main

"""
For evals that need to be run on GPUs (e.g. LM Evaluation Harness).
"""

executor_main_config = ExecutorMainConfig()
steps = [
    evaluate_lm_evaluation_harness(
        model_name="pf5pe4ut/step-600",
        model_path="gs://marin-us-central2/checkpoints/quickstart_single_script_docker_test_09_18/"
        "pf5pe4ut/hf/pf5pe4ut/step-600",
        evals=["mmlu"],
    ),
]

if __name__ == "__main__":
    executor_main(executor_main_config, steps=steps)
