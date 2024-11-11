from experiments.evals.evals import evaluate_lm_evaluation_harness
from marin.execution.executor import ExecutorMainConfig, executor_main

"""
For evals that need to be run on GPUs (e.g. LM Evaluation Harness).
"""

executor_main_config = ExecutorMainConfig()

quickstart_eval_step = evaluate_lm_evaluation_harness(
    model_name="pf5pe4ut/step-600",
    model_path="gs://marin-us-central2/checkpoints/quickstart_single_script_docker_test_09_18/"
    "pf5pe4ut/hf/pf5pe4ut/step-600",
    evals=["mmlu"],
)

exp433_dclm_1b_1x_eval = evaluate_lm_evaluation_harness(
    model_name="dclm_1b_1x_replication_oct26-a28b1e",
    model_path="gs://marin-us-central2/checkpoints/dclm_1b_1x_replication_oct26-a28b1e/hf/dclm_1b_1x_replication_oct26-a28b1e/step-54931",
    evals=["mmlu"],
)

steps = [
    quickstart_eval_step,
    exp433_dclm_1b_1x_eval,
]

if __name__ == "__main__":
    executor_main(executor_main_config, steps=steps)
