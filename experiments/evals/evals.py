"""
Canonical set of evals.
"""

import logging

from experiments.evals.task_configs import CORE_TASKS
from marin.evaluation.evaluation_config import EvalTaskConfig, EvaluationConfig
from marin.evaluation.run import evaluate
from marin.execution.executor import ExecutorStep, InputName, output_path_of, this_output_path, versioned

logger = logging.getLogger(__name__)


def evaluate_helm(model_name: str, model_path: str, evals: list[EvalTaskConfig]) -> ExecutorStep:
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
    step: ExecutorStep | InputName, evals: list[EvalTaskConfig], max_eval_instances: int | None = None
) -> ExecutorStep:
    """
    Create an ExecutorStep to evaluate the model using HELM on a step.

    Args:
        step (ExecutorStep | InputName): Executor Step to evaluate.
        evals (list[str]): List of evaluations to run with HELM, e.g, ["mmlu", "lite"].
    """
    # TODO: support evaluating all checkpoints in a run
    if isinstance(step, ExecutorStep):
        model_step_path = output_path_of(step)
        executor_step = step
    elif isinstance(step, InputName):
        model_step_path = output_path_of(step.step)
        executor_step = step.step

    return ExecutorStep(
        name=f"evaluation/helm/{executor_step.name}",
        fn=evaluate,
        config=EvaluationConfig(
            evaluator="helm",
            model_name=None,
            model_path=model_step_path,  # type: ignore
            evaluation_path=this_output_path(),
            evals=evals,
            discover_latest_checkpoint=True,
            max_eval_instances=max_eval_instances,
        ),
    )


def evaluate_lm_evaluation_harness(
    model_name: str, model_path: str, evals: list[EvalTaskConfig], max_eval_instances: int | None = None
) -> ExecutorStep:
    """
    Create an ExecutorStep to evaluate the model using LM Evaluation Harness.

    Args:
        model_name (str): Name of the model.
        model_path (str): Path to the model.
        evals (list[EvalTaskConfig]): List of evaluations to run with LM Evaluation Harness.
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
            max_eval_instances=max_eval_instances,
            launch_with_ray=False,
        ),
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


def _infer_model_name_for_path(model_path: str) -> str:
    """
    Infer model name from model path.
    """
    # path names are like gs://marin-us-central2/checkpoints/dclm_7b2x/hf/dclm_7b0828/dclm_7b0828/step-479999/
    # we want something like: dclm_7b0828_step-479999
    if model_path.endswith("/"):
        model_path = model_path[:-1]

    return "_".join(model_path.split("/")[-2:])


def default_eval(
    step: ExecutorStep | InputName | str,
    evals: list[EvalTaskConfig] | None = None,
    max_eval_instances: int | None = None,
) -> ExecutorStep:
    """
    Create an ExecutorStep to evaluate the model using LM Evaluation Harness on a step.

    Args:
        step (ExecutorStep | InputName): step to evaluate.
        evals (list[EvalTaskConfig]): List of evals to run- defaults to a set of CORE_TASKS defined in task_configs.py
        max_eval_instances (int): Maximum number of evaluation instances to run.
    """

    # this logic extracts the `ExecutorStep` corresponding to the training step, and get the model path
    if isinstance(step, ExecutorStep):
        model_step_path = output_path_of(step)
        name = step.name
    elif isinstance(step, InputName):
        model_step_path = output_path_of(step.step)
        name = step.step.name
    elif isinstance(step, str):
        model_step_path = step
        name = _infer_model_name_for_path(step)
    else:
        raise ValueError(f"Invalid step type: {step}")

    logger.info(f"Creating default evaluation step for {name}")

    # Default to CORE_TASKS
    if evals is None:
        evals = CORE_TASKS

    logger.info(f"Running evals on the following tasks: {evals}")

    return ExecutorStep(
        name=f"evaluation/levanter_lm_evaluation_harness/{name}",
        fn=evaluate,
        config=EvaluationConfig(
            evaluator="levanter_lm_evaluation_harness",
            model_name=None,  # imputed automatically
            model_path=versioned(model_step_path),  # type: ignore
            evaluation_path=this_output_path(),
            evals=versioned(evals),
            discover_latest_checkpoint=True,
            max_eval_instances=versioned(max_eval_instances),
        ),
    )
