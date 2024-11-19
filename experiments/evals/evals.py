from marin.evaluation.evaluation_config import EvaluationConfig
from marin.evaluation.run import evaluate
from marin.execution.executor import ExecutorStep, InputName, output_path_of, this_output_path

"""
Canonical set of evals.
"""


def evaluate_helm(model_name: str, model_path: str, evals: list[str]) -> ExecutorStep:
    """
    Create an ExecutorStep to evaluate the model using HELM.

    Args:
        model_name (str): Name of the model.
        model_path (str): Path to the model.
        evals (list[str]): List of evaluations to run with HELM, e.g, ["mmlu", "lite"].
    """
    return ExecutorStep(
        name=f"evaluation/helm/{model_name}",
        fn=evaluate,
        config=EvaluationConfig(
            evaluator="helm",
            model_name=model_name,
            model_path=model_path,
            evaluation_path=this_output_path(),
            evals=evals,
        ),
    )


def evaluate_helm_on_step(
    step: ExecutorStep | InputName, evals: list[str], max_eval_instances: int | None = None
) -> ExecutorStep:
    """
    Create an ExecutorStep to evaluate the model using HELM on a step.

    Args:
        step (ExecutorStep | InputName): Executor Step to evaluate.
        evals (list[str]): List of evaluations to run with HELM, e.g, ["mmlu", "lite"].
    """
    # TODO: support evaluating all checkpoints in a run
    if isinstance(step, ExecutorStep):
        step = output_path_of(step)
    elif isinstance(step, InputName):
        step = step.step

    return ExecutorStep(
        name=f"evaluation/helm/{step.name}",
        fn=evaluate,
        config=EvaluationConfig(
            evaluator="helm",
            model_name=None,
            model_path=step,  # type: ignore
            evaluation_path=this_output_path(),
            evals=evals,
            discover_latest_checkpoint=True,
            max_eval_instances=max_eval_instances,
        ),
    )


def evaluate_lm_evaluation_harness(model_name: str, model_path: str, evals: list[str]) -> ExecutorStep:
    """
    Create an ExecutorStep to evaluate the model using LM Evaluation Harness.

    Args:
        model_name (str): Name of the model.
        model_path (str): Path to the model.
        evals (list[str]): List of evaluations to run with LM Evaluation Harness, e.g, ["mmlu"].
    """
    return ExecutorStep(
        name=f"evaluation/lm_evaluation_harness/{model_name}",
        fn=evaluate,
        config=EvaluationConfig(
            evaluator="lm_evaluation_harness",
            model_name=model_name,
            model_path=model_path,
            evaluation_path=this_output_path(),
            evals=evals,
            launch_with_ray=False,
        ),
        required_device="gpu",
    )


def evaluate_lm_evaluation_harness_on_step(
    step: ExecutorStep | InputName, evals: list[str], max_eval_instances: int | None = None
) -> ExecutorStep:
    """
    Create an ExecutorStep to evaluate the model using LM Evaluation Harness on a step.

    Args:
        step (ExecutorStep | InputName): Executor Step to evaluate.
        evals (list[str]): List of evaluations to run with LM Evaluation Harness, e.g, ["mmlu"].
    """

    if isinstance(step, ExecutorStep):
        step = output_path_of(step)
    elif isinstance(step, InputName):
        step = step.step

    return ExecutorStep(
        name=f"evaluation/lm_evaluation_harness/{step.name}",
        fn=evaluate,
        config=EvaluationConfig(
            evaluator="lm_evaluation_harness",
            model_name=None,
            model_path=step,  # type: ignore
            evaluation_path=this_output_path(),
            evals=evals,
            discover_latest_checkpoint=True,
            max_eval_instances=max_eval_instances,
            launch_with_ray=False,
        ),
        required_device="gpu",
    )


def evaluate_alpaca_eval(model_name: str, model_path: str) -> ExecutorStep:
    """
    Create an ExecutorStep to evaluate the model using AlpacaEval.

    Args:
        model_name (str): Name of the model.
        model_path (str): Path to the model.
    """
    return ExecutorStep(
        name=f"evaluation/alpaca_eval/{model_name}",
        fn=evaluate,
        config=EvaluationConfig(
            evaluator="alpaca",
            model_name=model_name,
            model_path=model_path,
            evaluation_path=this_output_path(),
        ),
    )
