# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""
Canonical set of evals.
"""

import logging
import re
from collections.abc import Sequence

from fray.cluster import ResourceConfig
from marin.defaults import EVALCHEMY_DEPENDENCY_GROUPS, extract_model_name_and_path
from marin.defaults.evals import _infer_model_name_for_path
from marin.evaluation.evaluation_config import EvalTaskConfig, EvaluationConfig
from marin.evaluation.evaluators.harbor_evaluator import HARBOR_EVAL_ENV_KEYS, env_vars_from_keys
from marin.evaluation.run import evaluate
from marin.execution.executor import executor_main
from marin.execution.remote import remote
from marin.execution.types import ExecutorStep, InputName, OutputName, this_output_path
from marin.inference.vllm_server import validate_vllm_mode_env

from experiments.evals.evalchemy_results_compiler import compile_evalchemy_results_fn
from experiments.evals.evalchemy_task_configs import EVALCHEMY_CORE_TASKS

logger = logging.getLogger(__name__)


def evaluate_harbor(
    model_name: str,
    model_path: str | None,
    dataset: str,
    version: str = "1.0",
    max_eval_instances: int | None = None,
    resource_config: ResourceConfig | None = None,
    apply_chat_template: bool = False,
    wandb_tags: list[str] | None = None,
    generation_params: dict | None = None,
    agent: str = "claude-code",
    n_concurrent: int = 4,
    env: str = "local",
    agent_kwargs: dict | None = None,
) -> ExecutorStep:
    """
    Evaluate on ANY Harbor dataset from the registry.

    No custom adapters needed! Harbor's registry handles all datasets generically.

    Available datasets: https://harborframework.com/registry
    - aime@1.0: 60 math problems (AIME 2024, 2025-I, 2025-II)
    - terminal-bench@2.0: 89 terminal tasks
    - swebench-verified@1.0: 500 software engineering tasks
    - And 40+ more benchmarks!

    Args:
        model_name: Model identifier
        model_path: Path to model (can be None for API models like Claude)
        dataset: Harbor dataset name (e.g., "aime", "terminal-bench", "swebench-verified")
        version: Dataset version (e.g., "1.0", "2.0")
        max_eval_instances: Limit number of tasks to run
        resource_config: Resource configuration for direct Iris execution
        apply_chat_template: Whether to apply chat template (not used by Harbor)
        wandb_tags: Tags for W&B logging
        generation_params: Generation parameters (not used by Harbor)
        agent: Harbor agent type ("claude-code", "terminus-2", etc.)
        n_concurrent: Number of parallel trials
        env: Environment type ("local", "daytona", "e2b", "modal")

    Returns:
        ExecutorStep configured for Harbor evaluation

    Examples:
        # AIME evaluation
        evaluate_harbor("claude-opus-4", None, "aime", "1.0")

        # Terminal-Bench
        evaluate_harbor("qwen2.5-7b", "gs://.../model", "terminal-bench", "2.0")

        # SWE-bench Verified
        evaluate_harbor("claude-opus-4", None, "swebench-verified", "1.0", max_eval_instances=10)
    """

    if model_path is not None:
        validate_vllm_mode_env()

    # Harbor config goes in engine_kwargs
    engine_kwargs = {
        "harbor_config": {
            "dataset": dataset,
            "version": version,
            "agent": agent,
            "n_concurrent": n_concurrent,
            "env": env,
            "agent_kwargs": agent_kwargs or {},
        }
    }

    # When model_path is set, the evaluator launches a colocated vLLM server on
    # the accelerator resources. The outer executor step runs on CPU for API models.
    dispatch_resources = ResourceConfig.with_cpu() if model_path else resource_config
    return ExecutorStep(
        name=f"evaluation/harbor/{model_name}-{dataset}-{version}",
        fn=remote(
            evaluate,
            resources=dispatch_resources,
            env_vars=env_vars_from_keys(HARBOR_EVAL_ENV_KEYS),
            pip_dependency_groups=["harbor"],
        ),
        config=EvaluationConfig(
            evaluator="harbor",
            model_name=model_name,
            model_path=model_path,
            evaluation_path=this_output_path(),
            evals=[],  # Harbor uses dataset directly, not evals
            max_eval_instances=max_eval_instances,
            discover_latest_checkpoint=False,
            engine_kwargs=engine_kwargs,
            resource_config=resource_config,
            apply_chat_template=apply_chat_template,
            wandb_tags=wandb_tags,
            generation_params=generation_params,
        ),
    )


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
        fn=remote(evaluate, resources=resource_config, pip_dependency_groups=EVALCHEMY_DEPENDENCY_GROUPS),
        config=EvaluationConfig(
            evaluator="evalchemy",
            model_name=model_name,
            model_path=model_path,
            evaluation_path=this_output_path(),
            evals=evals,
            max_eval_instances=max_eval_instances,
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

    # Run all eval steps in a single executor_main call, capping concurrent
    # execution at max_parallel_jobs. The executor walks the shared dependency
    # DAG once instead of once per batch.
    executor_main(steps=eval_steps, max_concurrent=max_parallel_jobs)

    # Run compile steps separately. Their eval-step dependencies have already
    # succeeded, so the executor skips them and only runs the compile steps.
    if compile_steps:
        executor_main(steps=compile_steps)
