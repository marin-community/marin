# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""
Canonical set of evals.
"""

import logging
import re
from collections.abc import Sequence

from fray.cluster import ResourceConfig
from marin.evaluation.evaluation_config import EvalTaskConfig, EvaluationConfig
from marin.evaluation.run import evaluate
from marin.execution.executor import (
    ExecutorStep,
    InputName,
    OutputName,
    executor_main,
    output_path_of,
    this_output_path,
    versioned,
)

from experiments.evals.engine_configs import DEFAULT_LM_EVAL_MODEL_KWARGS
from experiments.evals.evalchemy_results_compiler import compile_evalchemy_results_fn
from experiments.evals.evalchemy_task_configs import EVALCHEMY_CORE_TASKS
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


def _infer_model_name_for_path(model_path: str) -> str:
    """
    Infer model name from model path.
    """
    # path names are like gs://marin-us-central2/checkpoints/dclm_7b2x/hf/dclm_7b0828/dclm_7b0828/step-479999/
    # we want something like: dclm_7b0828_step-479999
    if model_path.endswith("/"):
        model_path = model_path[:-1]

    return "_".join(model_path.split("/")[-2:])


def extract_model_name_and_path(step: ExecutorStep | InputName | str) -> tuple[str, InputName | str]:
    """
    Extract the model name and path from a step.

    Always appends /hf for ExecutorSteps; run.py's _normalize_model_path handles
    detecting whether the HF files are at root or in /hf at evaluation time.
    """
    if isinstance(step, ExecutorStep):
        model_step_path = output_path_of(step, "hf")
        name = step.name
    elif isinstance(step, InputName):
        # `InputName.hardcoded(...)` has `step.step is None`; treat it as a direct path.
        if step.step is None:
            if step.name is None:
                raise ValueError("Invalid InputName: both `step` and `name` are None.")
            model_step_path = step.name
            name = _infer_model_name_for_path(step.name)
        else:
            # If `name` is already set, the InputName refers to a specific subpath under the step's output.
            # Otherwise default to the HF export directory.
            model_step_path = step if step.name is not None else output_path_of(step.step, "hf")
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


def evaluate_evalchemy(
    model_name: str,
    model_path: str,
    evals: Sequence[EvalTaskConfig],
    max_eval_instances: int | None = None,
    engine_kwargs: dict | None = None,
    generation_params: dict | None = None,
    resource_config: ResourceConfig | None = None,
    apply_chat_template: bool = False,
    wandb_tags: list[str] | None = None,
    discover_latest_checkpoint: bool = True,
    base_eval_run_name: str | None = None,
) -> ExecutorStep:
    """
    Create an ExecutorStep to evaluate the model using Evalchemy.

    Args:
        model_name (str): Name of the model.
        model_path (str): Path to the model.
        evals (Sequence[EvalTaskConfig]): Evaluations to run with Evalchemy.
        max_eval_instances (int | None): Maximum number of evaluation instances to run.
        engine_kwargs (dict | None): Additional engine kwargs for vLLM.
        generation_params (dict | None): Generation parameters including:
            - temperature: float (e.g., 0.7)
            - top_p: float (e.g., 1.0)
            - max_gen_toks: int (e.g., 32768)
            - seeds: list[int] for multiple runs with different seeds
        resource_config (ResourceConfig | None): Resource configuration for the job.
        apply_chat_template (bool): Whether to apply chat template.
        wandb_tags (list[str] | None): Tags to add to the WandB run.
        discover_latest_checkpoint (bool): Whether to discover the latest checkpoint.
    """
    # Include task names and seed in the step name to ensure different runs get different output paths
    task_names = "_".join(sorted(e.name for e in evals))
    seed = generation_params.get("seed") if generation_params else None
    seed_suffix = f"_seed{seed}" if seed is not None else ""
    return ExecutorStep(
        name=f"evaluation/evalchemy/{model_name}/{task_names}{seed_suffix}",
        fn=evaluate,
        config=EvaluationConfig(
            evaluator="evalchemy",
            model_name=model_name,
            model_path=model_path,
            evaluation_path=this_output_path(),
            evals=evals,
            max_eval_instances=max_eval_instances,
            launch_with_ray=True,
            discover_latest_checkpoint=discover_latest_checkpoint,
            engine_kwargs=engine_kwargs,
            generation_params=generation_params,
            resource_config=resource_config,
            apply_chat_template=apply_chat_template,
            wandb_tags=wandb_tags,
            base_eval_run_name=base_eval_run_name,
        ),
    )


def default_evalchemy_eval(
    step: ExecutorStep | InputName | str,
    resource_config: ResourceConfig = ResourceConfig.with_tpu("v5p-8"),
    evals: Sequence[EvalTaskConfig] | None = None,
    max_eval_instances: int | None = None,
    engine_kwargs: dict | None = None,
    generation_params: dict | None = None,
    apply_chat_template: bool = False,
    discover_latest_checkpoint: bool = True,
    base_eval_run_name: str | None = None,
) -> ExecutorStep:
    """
    Create an ExecutorStep to evaluate the model using Evalchemy reasoning benchmarks.

    Args:
        step (ExecutorStep | InputName | str): Step to evaluate.
        resource_config (ResourceConfig): Resource configuration (defaults to v5p-8 TPU).
        evals (list[EvalTaskConfig] | None): List of evals to run. Defaults to EVALCHEMY_CORE_TASKS.
        max_eval_instances (int | None): Maximum number of evaluation instances to run.
        engine_kwargs (dict | None): Additional vLLM engine kwargs (optional for evalchemy).
        generation_params (dict | None): Generation parameters including:
            - temperature: float (e.g., 0.7)
            - top_p: float (e.g., 1.0)
            - max_gen_toks: int (e.g., 32768)
            - seed: int for reproducibility
        apply_chat_template (bool): Whether to apply chat template.
        discover_latest_checkpoint (bool): Whether to discover the latest checkpoint.
    """
    name, model_step_path = extract_model_name_and_path(step)

    # If base_eval_run_name is provided, use it for the output path name
    if base_eval_run_name:
        # When step is a raw string (e.g. a GCS path), search it directly for a step number.
        # Otherwise, use the extracted name which already incorporates the path structure.
        path_str = step if isinstance(step, str) else name
        step_match = re.search(r"step-(\d+)", path_str)
        step_suffix = f"-step{step_match.group(1)}" if step_match else ""
        name = f"{base_eval_run_name}{step_suffix}"

    logger.info(f"Creating Evalchemy evaluation step for {name}")

    if evals is None:
        evals = EVALCHEMY_CORE_TASKS

    logger.info(f"Running Evalchemy evals on the following tasks: {evals}")

    return evaluate_evalchemy(
        name,
        model_step_path,
        evals,
        max_eval_instances=max_eval_instances,
        engine_kwargs=engine_kwargs or {},  # Pass empty dict to avoid warning
        generation_params=generation_params,
        resource_config=resource_config,
        apply_chat_template=apply_chat_template,
        discover_latest_checkpoint=discover_latest_checkpoint,
        base_eval_run_name=base_eval_run_name,
    )


def compile_evalchemy_results(
    steps: list[ExecutorStep],
    seeds: list[int] | None = None,
    base_eval_run_name: str | None = None,
    model_path: str | None = None,
    task_name: str | None = None,
) -> ExecutorStep:
    """
    Compile results from multiple Evalchemy evaluation steps into aggregated metrics.

    Takes a list of ExecutorSteps for evalchemy tasks and compiles the results into a
    single DataFrame, then logs averaged results to wandb.

    Args:
        steps: List of ExecutorSteps from evalchemy evaluations (one per seed).
        seeds: List of seeds used for the evaluations (for wandb config).

    Returns:
        ExecutorStep that compiles and logs aggregated results.
    """
    # Create input paths from steps
    input_paths = [step.cd("results.json") for step in steps]
    output_path = OutputName("compiled_results")

    # Build compile step name matching the individual run hierarchy:
    #   individual: evaluation/evalchemy/{model_id}/AIME24_seed42
    #   compiled:   evaluation/evalchemy/{model_id}/compile_AIME24_avg5seeds
    if base_eval_run_name:
        step_match = re.search(r"step-(\d+)", model_path or "")
        step_suffix = f"-step{step_match.group(1)}" if step_match else ""
        model_id = f"{base_eval_run_name}{step_suffix}"
    elif model_path:
        model_id = _infer_model_name_for_path(model_path)
    else:
        model_id = "unknown"

    num_seeds = len(seeds) if seeds else len(steps)
    task_suffix = f"_{task_name}" if task_name else ""
    compile_step_name = f"evaluation/evalchemy/{model_id}/compile{task_suffix}_avg{num_seeds}seeds"

    return ExecutorStep(
        name=compile_step_name,
        fn=compile_evalchemy_results_fn,
        config={
            "input_paths": input_paths,
            "output_path": output_path,
            "seeds": seeds or [],
            "base_eval_run_name": base_eval_run_name,
            "model_path": model_path,
            "task_name": task_name,
        },
        description="Compile results from multiple evalchemy evaluation steps",
    )


def build_evalchemy_eval_steps(
    checkpoints: dict[str | None, list[str]],
    task_seed_groups: list[tuple[list[EvalTaskConfig], list[int]]],
    base_generation_params: dict,
    resource_config: ResourceConfig,
    engine_kwargs: dict | None = None,
    apply_chat_template: bool = True,
    discover_latest_checkpoint: bool = False,
) -> tuple[list[ExecutorStep], list[ExecutorStep]]:
    """Build evaluation and compilation steps for an evalchemy experiment.

    Creates one evaluation step per (checkpoint, task, seed) combination, plus
    compilation steps that aggregate results across seeds for each (checkpoint, task).

    Args:
        checkpoints: Mapping from base_eval_run_name to list of checkpoint paths.
            Use None as key to auto-generate names from paths.
        task_seed_groups: List of (tasks, seeds) tuples. Each task in a group
            is evaluated with all seeds in that group.
        base_generation_params: Generation parameters (temperature, top_p, max_gen_toks)
            shared across all runs. Per-seed params are generated by adding "seed": N.
        resource_config: TPU/GPU resource configuration for each eval job.
        engine_kwargs: vLLM engine kwargs (tensor_parallel_size, max_num_seqs, etc.).
        apply_chat_template: Whether to apply chat template.
        discover_latest_checkpoint: Whether to auto-discover latest checkpoint.

    Returns:
        Tuple of (eval_steps, compile_steps).
    """
    eval_steps: list[ExecutorStep] = []
    compile_steps: list[ExecutorStep] = []

    for base_eval_run_name, checkpoint_paths in checkpoints.items():
        for checkpoint in checkpoint_paths:
            task_seed_pairs: list[tuple[EvalTaskConfig, list[int]]] = []
            for tasks, seeds in task_seed_groups:
                task_seed_pairs += [(t, seeds) for t in tasks]

            for task, seeds in task_seed_pairs:
                task_steps: list[ExecutorStep] = []
                for seed in seeds:
                    generation_params = {**base_generation_params, "seed": seed}
                    step = default_evalchemy_eval(
                        step=checkpoint,
                        resource_config=resource_config,
                        evals=[task],
                        engine_kwargs=engine_kwargs,
                        generation_params=generation_params,
                        apply_chat_template=apply_chat_template,
                        discover_latest_checkpoint=discover_latest_checkpoint,
                        base_eval_run_name=base_eval_run_name,
                    )
                    task_steps.append(step)
                    eval_steps.append(step)

                if len(seeds) > 1:
                    compile_step = compile_evalchemy_results(
                        task_steps,
                        seeds=seeds,
                        base_eval_run_name=base_eval_run_name,
                        model_path=checkpoint,
                        task_name=task.name,
                    )
                    compile_steps.append(compile_step)

    return eval_steps, compile_steps


def run_evalchemy_experiment(
    checkpoints: dict[str | None, list[str]],
    task_seed_groups: list[tuple[list[EvalTaskConfig], list[int]]],
    base_generation_params: dict,
    resource_config: ResourceConfig,
    engine_kwargs: dict | None = None,
    apply_chat_template: bool = True,
    discover_latest_checkpoint: bool = False,
    max_parallel_jobs: int | None = None,
) -> None:
    """Run a complete evalchemy evaluation experiment.

    Builds eval and compile steps, then executes them via executor_main
    with optional batching for parallel job limits.

    Args:
        checkpoints: Mapping from base_eval_run_name to list of checkpoint paths.
        task_seed_groups: List of (tasks, seeds) tuples.
        base_generation_params: Shared generation parameters.
        resource_config: TPU/GPU resource configuration.
        engine_kwargs: vLLM engine kwargs.
        apply_chat_template: Whether to apply chat template.
        discover_latest_checkpoint: Whether to auto-discover latest checkpoint.
        max_parallel_jobs: Maximum eval jobs to run concurrently. None for no limit.
    """
    eval_steps, compile_steps = build_evalchemy_eval_steps(
        checkpoints=checkpoints,
        task_seed_groups=task_seed_groups,
        base_generation_params=base_generation_params,
        resource_config=resource_config,
        engine_kwargs=engine_kwargs,
        apply_chat_template=apply_chat_template,
        discover_latest_checkpoint=discover_latest_checkpoint,
    )

    # Run eval steps in batches to limit parallelism.
    # Each executor_main call runs up to max_parallel_jobs eval steps concurrently.
    # Already-completed steps are automatically skipped via status files on disk.
    if max_parallel_jobs is not None:
        for i in range(0, len(eval_steps), max_parallel_jobs):
            batch = eval_steps[i : i + max_parallel_jobs]
            executor_main(steps=batch)
    else:
        executor_main(steps=eval_steps)

    # Run compile steps separately. Their eval-step dependencies have already
    # succeeded, so the executor skips them and only runs the compile steps.
    if compile_steps:
        executor_main(steps=compile_steps)
