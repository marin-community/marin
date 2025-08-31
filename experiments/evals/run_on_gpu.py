"""
For evals that need to be run on GPUs (e.g. LM Evaluation Harness).
"""

# nodryrun
from experiments.dclm.exp433_dclm_run import dclm_baseline_only_model
from experiments.evals.evals import default_eval, evaluate_lm_evaluation_harness
from marin.evaluation.evaluation_config import EvalTaskConfig
from marin.execution.executor import executor_main

# example of how to eval a specific checkpoint
quickstart_eval_step = evaluate_lm_evaluation_harness(
    model_name="pf5pe4ut/step-600",
    model_path="gs://marin-us-central2/checkpoints/quickstart_single_script_docker_test_09_18/"
    "pf5pe4ut/hf/pf5pe4ut/step-600",
    evals=EvalTaskConfig("mmlu", num_fewshot=0),
)

# example of how to use default_eval to run CORE_TASKS on a step
exp433_dclm_1b_1x_eval_nov12 = default_eval(dclm_baseline_only_model)

steps = [
    quickstart_eval_step,
    exp433_dclm_1b_1x_eval_nov12,
]

if __name__ == "__main__":
    executor_main(steps=steps)
