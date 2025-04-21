"""
Canonical set of evals.
"""

import logging

from experiments.evals.engine_configs import DEFAULT_VLLM_ENGINE_KWARGS
from experiments.evals.resource_configs import SINGLE_TPU_V4_8, ResourceConfig
from experiments.evals.task_configs import CORE_TASKS, KEY_GENERATION_TASKS, KEY_MULTIPLE_CHOICE_TASKS
from marin.evaluation.evaluation_config import EvalTaskConfig, EvaluationConfig
from marin.evaluation.run import evaluate
from marin.execution.executor import (
    ExecutorStep,
    InputName,
    get_executor_step,
    output_path_of,
    this_output_path,
    versioned,
)

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
    executor_step = get_executor_step(step)
    model_step_path = output_path_of(executor_step)

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
    model_name: str,
    model_path: str,
    evals: list[EvalTaskConfig],
    max_eval_instances: int | None = None,
    engine_kwargs: dict | None = None,
    resource_config: ResourceConfig | None = None,
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
            launch_with_ray=True,
            engine_kwargs=engine_kwargs,
            resource_config=resource_config,
        ),
    )


def evaluate_alpaca_eval(
    model_name: str,
    model_path: str,
    resource_config: ResourceConfig,
    temperature: float = 0.7,
    presence_penalty: float = 0.0,
    frequency_penalty: float = 0.0,
    repetition_penalty: float = 1.0,
    top_p: float = 1.0,
    top_k: int = -1,
) -> ExecutorStep:
    """
    Create an ExecutorStep to evaluate the model using AlpacaEval.

    Args:
        model_name (str): Name of the model.
        model_path (str): Path to the model.
        resource_config (ResourceConfig): Resource configuration for the evaluation.
        temperature (float, optional): Float that controls the randomness of the sampling.
            Lower values make the model more deterministic, while higher values make
            the model more random. Zero means greedy sampling. Defaults to 0.7.
        presence_penalty (float, optional): Float that penalizes new tokens based on whether they
            appear in the generated text so far. Values > 0 encourage the model
            to use new tokens, while values < 0 encourage the model to repeat tokens. Defaults to 0.0.
        frequency_penalty (float, optional): Float that penalizes new tokens based on their
            frequency in the generated text so far. Values > 0 encourage the model to
            use new tokens, while values < 0 encourage the model to repeat tokens. Defaults to 0.0.
        repetition_penalty (float, optional): Float that penalizes new tokens based on whether
            they appear in the prompt and the generated text so far. Values > 1
            encourage the model to use new tokens, while values < 1 encourage
            the model to repeat tokens. Defaults to 1.0.
        top_p (float, optional): Float that controls the cumulative probability of the top tokens
            to consider. Must be in (0, 1]. Set to 1 to consider all tokens. Defaults to 1.0.
        top_k (int, optional): Integer that controls the number of top tokens to consider.
            Set to -1 to consider all tokens. Defaults to -1.
    """
    return ExecutorStep(
        name=f"evaluation/alpaca_eval/{model_name}",
        fn=evaluate,
        config=EvaluationConfig(
            evaluator="alpaca",
            model_name=model_name,
            model_path=model_path,
            evaluation_path=this_output_path(),
            resource_config=resource_config,
            engine_kwargs={
                "temperature": temperature,
                "presence_penalty": presence_penalty,
                "frequency_penalty": frequency_penalty,
                "repetition_penalty": repetition_penalty,
                "top_p": top_p,
                "top_k": top_k,
            },
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


def extract_model_name_and_path(step: ExecutorStep | InputName | str) -> tuple[str, str]:
    """
    Extract the model name and path from a step.
    """
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

    return name, model_step_path


def evaluate_levanter_lm_evaluation_harness(
    model_name: str,
    model_path: str,
    evals: list[EvalTaskConfig],
    resource_config: ResourceConfig,
    max_eval_instances: int | None = None,
) -> ExecutorStep:
    """
    Create an ExecutorStep to evaluate the model using Levanter LM Evaluation Harness.
    """
    return ExecutorStep(
        name=f"evaluation/lm_evaluation_harness_levanter/lmeval_debug_{model_name}",
        fn=evaluate,
        config=EvaluationConfig(
            evaluator="levanter_lm_evaluation_harness",
            model_name=None,  # imputed automatically
            model_path=versioned(model_path),  # type: ignore
            evaluation_path=this_output_path(),
            evals=versioned(evals),
            discover_latest_checkpoint=True,
            max_eval_instances=versioned(max_eval_instances),
            resource_config=resource_config,
        ),
    )


def default_eval(
    step: ExecutorStep | InputName | str,
    resource_config: ResourceConfig = SINGLE_TPU_V4_8,
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
    name, model_step_path = extract_model_name_and_path(step)

    logger.info(f"Creating default evaluation step for {name}")

    # Default to CORE_TASKS
    if evals is None:
        evals = CORE_TASKS

    logger.info(f"Running evals on the following tasks: {evals}")

    return evaluate_levanter_lm_evaluation_harness(
        name, model_step_path, evals, resource_config, max_eval_instances=max_eval_instances
    )


def default_key_evals(
    step: ExecutorStep | InputName | str,
    resource_config: ResourceConfig,
    model_name: str | None = None,
    max_eval_instances: int | None = None,
    engine_kwargs: dict | None = DEFAULT_VLLM_ENGINE_KWARGS,
) -> list[ExecutorStep]:
    """
    Create a list of ExecutorSteps to evaluate the model using LM Evaluation Harness on a step.
    """
    name, model_step_path = extract_model_name_and_path(step)

    if model_name is None:
        model_name = name

    return [
        evaluate_lm_evaluation_harness(
            model_name,
            model_step_path,
            KEY_GENERATION_TASKS,
            max_eval_instances=max_eval_instances,
            engine_kwargs=engine_kwargs,
            resource_config=resource_config,
        ),
        evaluate_levanter_lm_evaluation_harness(
            model_name,
            model_step_path,
            KEY_MULTIPLE_CHOICE_TASKS,
            resource_config,
            max_eval_instances=max_eval_instances,
        ),
        evaluate_alpaca_eval(model_name, model_step_path, resource_config),
    ]
