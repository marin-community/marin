from marin.execution.executor import ExecutorStep, executor_main, this_output_path

from scripts.evaluation.evaluation_config import EvaluationConfig
from scripts.evaluation.run import evaluate

"""
Canonical set of evals.

How to run:
ray job submit --address http://127.0.0.1:8265 --working-dir . --no-wait -- python3 experiments/evals.py
"""


# TODO: turn this into a library
# Sample model
model_path: str = (
    "gs://marin-us-central2/checkpoints/quickstart_single_script_docker_test_09_18/pf5pe4ut/hf/pf5pe4ut/step-600"
)
model_name: str = "pf5pe4ut/step-600"

############################################################
# HELM MMLU
helm_evaluate_step = ExecutorStep(
    name="evaluation/hello_world_fw-pliang/helm",
    fn=evaluate,
    config=EvaluationConfig(
        evaluator="helm",
        model_name=model_name,
        model_path=model_path,
        evaluation_path=this_output_path(),
        evals=["mmlu"],
    ),
)

############################################################
# AlpacaEval
alpaca_evaluate_step = ExecutorStep(
    name="evaluation/hello_world_fw-pliang/alpaca",
    fn=evaluate,
    config=EvaluationConfig(
        evaluator="alpaca",
        model_name=model_name,
        model_path=model_path,
        evaluation_path=this_output_path(),
        # TODO: what value should this be since this is the quickstart?
        max_eval_instances=1,
    ),
)

############################################################

if __name__ == "__main__":
    executor_main(
        steps=[
            helm_evaluate_step,
            alpaca_evaluate_step,
        ]
    )
