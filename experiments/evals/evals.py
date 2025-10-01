# Copyright 2025 The Marin Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Canonical set of evals.
"""

import logging

from experiments.evals.engine_configs import DEFAULT_LM_EVAL_MODEL_KWARGS
from experiments.evals.resource_configs import SINGLE_TPU_V4_8, SINGLE_TPU_V6E_8, ResourceConfig
from experiments.evals.task_configs import (
    BASE_GENERATION_TASKS,
    CORE_TASKS,
    CORE_TASKS_PLUS_LEADERBOARD,
    KEY_GENERATION_TASKS,
    KEY_MULTIPLE_CHOICE_TASKS,
    MMLU_0_SHOT,
    MMLU_5_SHOT,
    MMLU_PRO_5_SHOT,
    OPEN_LM_LEADERBOARD_GEN,
    OPEN_LM_LEADERBOARD_MCQ,
)
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
    apply_chat_template: bool = False,
    wandb_tags: list[str] | None = None,
    discover_latest_checkpoint: bool = True,
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
            discover_latest_checkpoint=discover_latest_checkpoint,
            engine_kwargs=engine_kwargs,
            resource_config=resource_config,
            apply_chat_template=apply_chat_template,
            wandb_tags=wandb_tags,
        ),
    )


def evaluate_alpaca_eval(
    model_name: str,
    model_path: str,
    resource_config: ResourceConfig,
    engine_kwargs: dict | None = None,
    max_eval_instances: int | None = None,
    temperature: float = 0.7,
    presence_penalty: float = 0.0,
    frequency_penalty: float = 0.0,
    repetition_penalty: float = 1.0,
    top_p: float = 1.0,
    top_k: int = -1,
    stop_token_ids: list | None = None,
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
        stop_token_ids (list, optional): List of integer token ids that controls the token ids
            that vLLM should consider to stop the generation on. Defaults to None which uses
            the tokenizer config's stop token ids
    """
    return ExecutorStep(
        name=f"evaluation/alpaca_eval/{model_name}",
        fn=evaluate,
        config=EvaluationConfig(
            evaluator="alpaca",
            model_name=model_name,
            model_path=model_path,
            evaluation_path=this_output_path(),
            max_eval_instances=max_eval_instances,
            resource_config=resource_config,
            engine_kwargs=engine_kwargs,
            generation_params={
                "temperature": temperature,
                "presence_penalty": presence_penalty,
                "frequency_penalty": frequency_penalty,
                "repetition_penalty": repetition_penalty,
                "top_p": top_p,
                "top_k": top_k,
                "stop_token_ids": stop_token_ids,
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
        model_step_path = output_path_of(step, "hf" if "gcsfuse" not in step.name else "")
        name = step.name
    elif isinstance(step, InputName):
        model_step_path = output_path_of(step.step, "hf" if "gcsfuse" not in step.step.name else "")
        if step.step is None:
            raise ValueError(f"Hardcoded path {step.name} is not part of the pipeline")
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
    apply_chat_template: bool = False,
    discover_latest_checkpoint: bool = True,
) -> ExecutorStep:
    """
    Create an ExecutorStep to evaluate the model using Levanter LM Evaluation Harness.
    """
    logger.info(f"Running evals on the following tasks: {evals}")
    return ExecutorStep(
        name=f"evaluation/lm_evaluation_harness_levanter/lmeval_debug_{model_name}",
        fn=evaluate,
        config=EvaluationConfig(
            evaluator="levanter_lm_evaluation_harness",
            model_name=None,  # imputed automatically
            model_path=model_path,  # type: ignore
            evaluation_path=this_output_path(),
            evals=versioned(evals),
            discover_latest_checkpoint=discover_latest_checkpoint,
            max_eval_instances=versioned(max_eval_instances),
            resource_config=resource_config,
            apply_chat_template=apply_chat_template,
        ),
    )


def default_eval(
    step: ExecutorStep | InputName | str,
    resource_config: ResourceConfig = SINGLE_TPU_V4_8,
    evals: list[EvalTaskConfig] | None = None,
    max_eval_instances: int | None = None,
    apply_chat_template: bool = False,
    discover_latest_checkpoint: bool = True,
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
        name,
        model_step_path,
        evals,
        resource_config,
        max_eval_instances=max_eval_instances,
        apply_chat_template=apply_chat_template,
        discover_latest_checkpoint=discover_latest_checkpoint,
    )


def default_base_eval(
    step: ExecutorStep | InputName | str,
    resource_config: ResourceConfig = SINGLE_TPU_V6E_8,
    max_eval_instances: int | None = None,
    engine_kwargs: dict | None = DEFAULT_LM_EVAL_MODEL_KWARGS,
    run_generation_evals: bool = True,
    discover_latest_checkpoint: bool = True,
):
    # Add GPQA to CORE_TASKS
    # Set up evaluations for core tasks (including GPQA)
    eval_jobs = []
    core_grouped = default_eval(
        step=step,
        resource_config=resource_config,
        evals=CORE_TASKS_PLUS_LEADERBOARD,
        discover_latest_checkpoint=discover_latest_checkpoint,
    )
    eval_jobs.append(core_grouped)

    # Run tasks where we report Macro_Avg separately to make sure the macro avg gets computed correctly.
    mmlu_0shot = default_eval(
        step=step,
        resource_config=resource_config,
        evals=(MMLU_0_SHOT,),
        discover_latest_checkpoint=discover_latest_checkpoint,
    )
    eval_jobs.append(mmlu_0shot)

    mmlu_5shot = default_eval(
        step=step,
        resource_config=resource_config,
        evals=(MMLU_5_SHOT,),
        discover_latest_checkpoint=discover_latest_checkpoint,
    )
    eval_jobs.append(mmlu_5shot)

    mmlu_pro_5shot = default_eval(
        step=step,
        resource_config=resource_config,
        evals=(MMLU_PRO_5_SHOT,),
        discover_latest_checkpoint=discover_latest_checkpoint,
    )
    eval_jobs.append(mmlu_pro_5shot)

    name, model_step_path = extract_model_name_and_path(step)
    if run_generation_evals:
        generation = evaluate_lm_evaluation_harness(
            name,
            model_step_path,
            BASE_GENERATION_TASKS,
            max_eval_instances=max_eval_instances,
            engine_kwargs=engine_kwargs,
            resource_config=resource_config,
            discover_latest_checkpoint=discover_latest_checkpoint,
        )

        eval_jobs.append(generation)
    return eval_jobs


def default_sft_eval(
    step: ExecutorStep | InputName | str,
    resource_config: ResourceConfig = SINGLE_TPU_V6E_8,
    max_eval_instances: int | None = None,
    engine_kwargs: dict | None = DEFAULT_LM_EVAL_MODEL_KWARGS,
    run_generation_evals: bool = True,
    apply_chat_template: bool = True,
):
    # Set up evaluations for core tasks (including GPQA)
    eval_jobs = []
    leaderboard_grouped = default_eval(
        step=step,
        resource_config=resource_config,
        evals=OPEN_LM_LEADERBOARD_MCQ,
        apply_chat_template=apply_chat_template,
    )
    eval_jobs.append(leaderboard_grouped)

    # Run tasks where we report Macro_Avg separately to make sure the macro avg gets computed correctly.

    mmlu_5shot = default_eval(
        step=step, resource_config=resource_config, evals=(MMLU_5_SHOT,), apply_chat_template=apply_chat_template
    )
    eval_jobs.append(mmlu_5shot)

    mmlu_pro_5shot = default_eval(
        step=step, resource_config=resource_config, evals=(MMLU_PRO_5_SHOT,), apply_chat_template=apply_chat_template
    )
    eval_jobs.append(mmlu_pro_5shot)

    name, model_step_path = extract_model_name_and_path(step)
    if run_generation_evals:
        leaderboard_generation = evaluate_lm_evaluation_harness(
            name,
            model_step_path,
            KEY_GENERATION_TASKS,
            max_eval_instances=max_eval_instances,
            engine_kwargs=engine_kwargs,
            resource_config=resource_config,
            apply_chat_template=apply_chat_template,
        )

        eval_jobs.append(leaderboard_generation)

        olmo_generation = evaluate_lm_evaluation_harness(
            name,
            model_step_path,
            OPEN_LM_LEADERBOARD_GEN,
            max_eval_instances=max_eval_instances,
            engine_kwargs=engine_kwargs,
            resource_config=resource_config,
            apply_chat_template=apply_chat_template,
        )
        eval_jobs.append(olmo_generation)
    return eval_jobs


def default_key_evals(
    step: ExecutorStep | InputName | str,
    resource_config: ResourceConfig,
    model_name: str | None = None,
    max_eval_instances: int | None = None,
    engine_kwargs: dict | None = DEFAULT_LM_EVAL_MODEL_KWARGS,
) -> list[ExecutorStep]:
    """
    Create a list of ExecutorSteps to evaluate the model using LM Evaluation Harness on a step.
    """
    name, model_step_path = extract_model_name_and_path(step)

    if model_name is None:
        model_name = name

    stop_token_ids = []
    if "llama3" in model_name:
        stop_token_ids.append(128009)
    elif "olmo" in model_name:
        stop_token_ids.append(100257)

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
        evaluate_alpaca_eval(
            model_name,
            model_step_path,
            resource_config,
            engine_kwargs,
            stop_token_ids=stop_token_ids,
        ),
    ]
