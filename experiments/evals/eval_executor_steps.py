from marin.evaluation.evaluation_config import EvaluationConfig
from marin.evaluation.run import evaluate
from marin.execution.executor import ExecutorStep, this_output_path

"""
Canonical set of evals and their ExecutorSteps.

TODO: to test with model_path: str = "gs://marin-us-central2/checkpoints/quickstart_single_script_docker_test_09_18/pf5pe4ut/hf/pf5pe4ut/step-600"
"""


def create_helm_executor_step(model_name: str, model_path: str, evals: list[str]) -> ExecutorStep:
    """
    Create a step to evaluate the model using HELM.

    Args:
        model_name (str): Name of the model.
        model_path (str): Path to the model.
        evals (list[str]): List of evaluations to run with HELM, e.g, ["mmlu", "lite"].
    """
    return ExecutorStep(
        name="helm-eval",
        fn=evaluate,
        config=EvaluationConfig(
            evaluator="helm",
            model_name=model_name,
            model_path=model_path,
            evaluation_path=this_output_path(),
            evals=evals,
        ),
    )


def create_lm_evaluation_harness_executor_step(model_name: str, model_path: str, evals: list[str]) -> ExecutorStep:
    """
    Create a step to evaluate the model using LM Evaluation Harness.

    Args:
        model_name (str): Name of the model.
        model_path (str): Path to the model.
        evals (list[str]): List of evaluations to run with LM Evaluation Harness, e.g, ["mmlu"].
    """
    return ExecutorStep(
        name="lm-eval",
        fn=evaluate,
        config=EvaluationConfig(
            evaluator="lm_evaluation_harness",
            model_name=model_name,
            model_path=model_path,
            evaluation_path=this_output_path(),
            evals=evals,
            launch_with_ray=False,
        ),
    )


def create_alpaca_eval_executor_step(model_name: str, model_path: str) -> ExecutorStep:
    """
    Create a step to evaluate the model using AlpacaEval.

    Args:
        model_name (str): Name of the model.
        model_path (str): Path to the model.
    """
    return ExecutorStep(
        name="alpaca-eval",
        fn=evaluate,
        config=EvaluationConfig(
            evaluator="alpaca",
            model_name=model_name,
            model_path=model_path,
            evaluation_path=this_output_path(),
        ),
    )
