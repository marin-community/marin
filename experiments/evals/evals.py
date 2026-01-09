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

from fray.cluster import ResourceConfig
from marin.evaluation.evaluation_config import EvalTaskConfig, EvaluationConfig
from marin.evaluation.run import evaluate
from marin.execution.executor import (
    ExecutorStep,
    InputName,
    output_path_of,
    this_output_path,
    versioned,
)

from experiments.evals.engine_configs import DEFAULT_LM_EVAL_MODEL_KWARGS
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

logger = logging.getLogger(__name__)


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
    generation_params: dict | None = None,
    depends_on: "ExecutorStep | None" = None,
    seed: int | None = None,
) -> ExecutorStep:
    """
    Create an ExecutorStep to evaluate the model using LM Evaluation Harness.

    Args:
        model_name (str): Name of the model.
        model_path (str): Path to the model.
        evals (list[EvalTaskConfig]): List of evaluations to run with LM Evaluation Harness.
        generation_params (dict | None): Generation parameters for vLLM (temperature, max_tokens, etc.)
        depends_on (ExecutorStep | None): Optional step to depend on for serial execution.
        seed (int | None): Optional seed for reproducibility. If provided, appended to step name after task.
    """
    from marin.execution.executor import InputName

    suffix = '_'.join([e.name for e in evals])

    # Build step name: {model_name}-{task}-seed{N} (seed appended after task if provided)
    step_name = f"evaluation/lm_evaluation_harness/{model_name}-{suffix}"
    if seed is not None:
        step_name = f"{step_name}-seed{seed}"

    # Create dependency InputName if depends_on is specified
    depends_on_input = None
    if depends_on is not None:
        depends_on_input = InputName(step=depends_on, name=None, block_on_step=True)

    return ExecutorStep(
        name=step_name,
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
            generation_params=generation_params,
            depends_on=depends_on_input,
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
    max_length: int | None = None,
    print_every_n: int | None = None,
    discover_latest_checkpoint: bool = True,
    generation_kwargs: dict | None = None,
) -> ExecutorStep:
    """
    Create an ExecutorStep to evaluate the model using Levanter LM Evaluation Harness.
    """
    logger.info(f"Running evals on the following tasks: {evals}")
    # Sanitize model_name for use in step name (replace / with -)
    sanitized_name = model_name.replace("/", "-")
    return ExecutorStep(
        name=f"evaluation/lm_evaluation_harness_levanter/levanter_lmeval_{'-'.join([eval_task.name for eval_task in evals])}_{sanitized_name}",
        fn=evaluate,
        config=EvaluationConfig(
            evaluator="levanter_lm_evaluation_harness",
            model_name=model_name,  # Pass the original model_name with /
            model_path=model_path,  # type: ignore
            evaluation_path=this_output_path(),
            evals=versioned(evals),
            discover_latest_checkpoint=discover_latest_checkpoint,
            max_eval_instances=versioned(max_eval_instances),
            resource_config=resource_config,
            apply_chat_template=apply_chat_template,
            max_length=max_length,
            print_every_n=print_every_n,
            generation_params=generation_kwargs,
            wandb_tags=["lm-eval", f"{model_name}"] + [eval_task.name for eval_task in evals],
        ),
        pip_dependency_groups=["eval", "tpu"],
    )


def default_eval(
    step: ExecutorStep | InputName | str,
    resource_config: ResourceConfig = ResourceConfig.with_tpu("v4-8"),
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
    resource_config: ResourceConfig = ResourceConfig.with_tpu("v6e-8"),
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
    resource_config: ResourceConfig = ResourceConfig.with_tpu("v6e-8"),
    max_eval_instances: int | None = None,
    engine_kwargs: dict | None = DEFAULT_LM_EVAL_MODEL_KWARGS,
    run_generation_evals: bool = True,
    apply_chat_template: bool = True,
    use_levanter_inference: bool = False,
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
        if use_levanter_inference:
            leaderboard_generation = evaluate_levanter_lm_evaluation_harness(
                name,
                model_step_path,
                KEY_GENERATION_TASKS,
                resource_config,
                max_eval_instances=max_eval_instances,
                apply_chat_template=apply_chat_template,
            )
            eval_jobs.append(leaderboard_generation)

            olmo_generation = evaluate_levanter_lm_evaluation_harness(
                name,
                model_step_path,
                OPEN_LM_LEADERBOARD_GEN,
                resource_config,
                max_eval_instances=max_eval_instances,
                apply_chat_template=apply_chat_template,
            )
            eval_jobs.append(olmo_generation)
        else:
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
    ]
