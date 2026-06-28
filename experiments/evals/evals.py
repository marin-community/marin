# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""
Canonical set of evals.
"""

import logging
import re
from collections.abc import Sequence

from fray.cluster import ResourceConfig
from marin.evaluation.evaluation_config import EvalTaskConfig, EvaluationConfig
from marin.evaluation.evaluators.harbor_evaluator import HARBOR_EVAL_ENV_KEYS, env_vars_from_keys
from marin.evaluation.run import evaluate
from marin.execution.lazy import Artifact, Checkpoint, RunContext, lower
from marin.execution.remote import remote
from marin.execution.step_runner import StepRunner
from marin.experiment.data import derived
from marin.inference.vllm_server import validate_vllm_mode_env

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

EVAL_DEPENDENCY_GROUPS = ["eval", "vllm", "tpu"]
EVALCHEMY_DEPENDENCY_GROUPS = ["evalchemy", "vllm", "tpu"]

logger = logging.getLogger(__name__)


def evaluate_lm_evaluation_harness(
    model_name: str,
    model: Checkpoint | str,
    evals: list[EvalTaskConfig],
    max_eval_instances: int | None = None,
    engine_kwargs: dict | None = None,
    resource_config: ResourceConfig | None = None,
    apply_chat_template: bool = False,
    wandb_tags: list[str] | None = None,
    discover_latest_checkpoint: bool = True,
    env_vars: dict[str, str] | None = None,
) -> Artifact:
    """
    Create an eval artifact for the model using LM Evaluation Harness.

    Args:
        model_name: Name of the model.
        model: Checkpoint handle or path string to the model.
        evals: List of evaluations to run with LM Evaluation Harness.
        env_vars: Extra env vars to set on the child iris worker.
            Needed for vLLM-on-TPU bring-up (e.g. ``VLLM_ENABLE_V1_MULTIPROCESSING=0``)
            and code-eval-dependent tasks like humaneval (``HF_ALLOW_CODE_EVAL=1``).
            The coordinator's own ``os.environ`` does NOT propagate to iris-spawned
            children — these vars must be threaded through ``remote()``.
    """
    deps: tuple[Artifact, ...] = (model,) if isinstance(model, Checkpoint) else ()

    def build_config(ctx: RunContext) -> EvaluationConfig:
        model_path = ctx.path(model) if isinstance(model, Checkpoint) else model
        return EvaluationConfig(
            evaluator="lm_evaluation_harness",
            model_name=model_name,
            model_path=model_path,
            evaluation_path=ctx.out,
            evals=evals,
            max_eval_instances=max_eval_instances,
            discover_latest_checkpoint=discover_latest_checkpoint,
            engine_kwargs=engine_kwargs,
            resource_config=resource_config,
            apply_chat_template=apply_chat_template,
            wandb_tags=wandb_tags,
        )

    return derived(
        f"evaluation/lm_evaluation_harness/{model_name}",
        fn=remote(
            evaluate,
            resources=resource_config,
            pip_dependency_groups=EVAL_DEPENDENCY_GROUPS,
            env_vars=env_vars,
        ),
        build_config=build_config,
        deps=deps,
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


def extract_model_name_and_path(step: Checkpoint | str) -> tuple[str, Checkpoint | str]:
    """
    Extract the model name and path from a step or string.

    Returns the model name and the original step (Checkpoint or str) for use in deps.
    """
    if isinstance(step, Checkpoint):
        return step.name, step
    return _infer_model_name_for_path(step), step


def evaluate_levanter_lm_evaluation_harness(
    model_name: str,
    model: Checkpoint | str,
    evals: list[EvalTaskConfig],
    resource_config: ResourceConfig,
    max_eval_instances: int | None = None,
    apply_chat_template: bool = False,
    discover_latest_checkpoint: bool = True,
) -> Artifact:
    """
    Create an eval artifact for the model using Levanter LM Evaluation Harness.
    """
    logger.info(f"Running evals on the following tasks: {evals}")
    deps: tuple[Artifact, ...] = (model,) if isinstance(model, Checkpoint) else ()

    def build_config(ctx: RunContext) -> EvaluationConfig:
        model_path = ctx.path(model) if isinstance(model, Checkpoint) else model
        return EvaluationConfig(
            evaluator="levanter_lm_evaluation_harness",
            model_name=None,  # imputed automatically
            model_path=model_path,
            evaluation_path=ctx.out,
            evals=evals,
            discover_latest_checkpoint=discover_latest_checkpoint,
            max_eval_instances=max_eval_instances,
            resource_config=resource_config,
            apply_chat_template=apply_chat_template,
        )

    return derived(
        f"evaluation/lm_evaluation_harness_levanter/lmeval_debug_{model_name}",
        fn=remote(evaluate, resources=resource_config, pip_dependency_groups=EVAL_DEPENDENCY_GROUPS),
        build_config=build_config,
        deps=deps,
    )


def default_eval(
    step: Checkpoint | str,
    resource_config: ResourceConfig = ResourceConfig.with_tpu("v4-8"),
    evals: list[EvalTaskConfig] | None = None,
    max_eval_instances: int | None = None,
    apply_chat_template: bool = False,
    discover_latest_checkpoint: bool = True,
) -> Artifact:
    """
    Create an eval artifact for the model using LM Evaluation Harness on a step.

    Args:
        step: Checkpoint handle or path string to evaluate.
        evals: List of evals to run. Defaults to CORE_TASKS.
        max_eval_instances: Maximum number of evaluation instances to run.
    """
    name, model = extract_model_name_and_path(step)

    logger.info(f"Creating default evaluation step for {name}")

    if evals is None:
        evals = CORE_TASKS

    logger.info(f"Running evals on the following tasks: {evals}")

    return evaluate_levanter_lm_evaluation_harness(
        name,
        model,
        evals,
        resource_config,
        max_eval_instances=max_eval_instances,
        apply_chat_template=apply_chat_template,
        discover_latest_checkpoint=discover_latest_checkpoint,
    )


def default_base_eval(
    step: Checkpoint | str,
    resource_config: ResourceConfig = ResourceConfig.with_tpu("v6e-8"),
    max_eval_instances: int | None = None,
    engine_kwargs: dict | None = DEFAULT_LM_EVAL_MODEL_KWARGS,
    run_generation_evals: bool = True,
    discover_latest_checkpoint: bool = True,
) -> list[Artifact]:
    eval_jobs = []
    core_grouped = default_eval(
        step=step,
        resource_config=resource_config,
        evals=CORE_TASKS_PLUS_LEADERBOARD,
        discover_latest_checkpoint=discover_latest_checkpoint,
    )
    eval_jobs.append(core_grouped)

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

    name, model = extract_model_name_and_path(step)
    if run_generation_evals:
        generation = evaluate_lm_evaluation_harness(
            name,
            model,
            BASE_GENERATION_TASKS,
            max_eval_instances=max_eval_instances,
            engine_kwargs=engine_kwargs,
            resource_config=resource_config,
            discover_latest_checkpoint=discover_latest_checkpoint,
        )
        eval_jobs.append(generation)
    return eval_jobs


def default_sft_eval(
    step: Checkpoint | str,
    resource_config: ResourceConfig = ResourceConfig.with_tpu("v6e-8"),
    max_eval_instances: int | None = None,
    engine_kwargs: dict | None = DEFAULT_LM_EVAL_MODEL_KWARGS,
    run_generation_evals: bool = True,
    apply_chat_template: bool = True,
    use_levanter_inference: bool = False,
) -> list[Artifact]:
    eval_jobs = []
    leaderboard_grouped = default_eval(
        step=step,
        resource_config=resource_config,
        evals=OPEN_LM_LEADERBOARD_MCQ,
        apply_chat_template=apply_chat_template,
    )
    eval_jobs.append(leaderboard_grouped)

    mmlu_5shot = default_eval(
        step=step, resource_config=resource_config, evals=(MMLU_5_SHOT,), apply_chat_template=apply_chat_template
    )
    eval_jobs.append(mmlu_5shot)

    mmlu_pro_5shot = default_eval(
        step=step, resource_config=resource_config, evals=(MMLU_PRO_5_SHOT,), apply_chat_template=apply_chat_template
    )
    eval_jobs.append(mmlu_pro_5shot)

    name, model = extract_model_name_and_path(step)
    if run_generation_evals:
        if use_levanter_inference:
            leaderboard_generation = evaluate_levanter_lm_evaluation_harness(
                name,
                model,
                KEY_GENERATION_TASKS,
                resource_config,
                max_eval_instances=max_eval_instances,
                apply_chat_template=apply_chat_template,
            )
            eval_jobs.append(leaderboard_generation)

            olmo_generation = evaluate_levanter_lm_evaluation_harness(
                name,
                model,
                OPEN_LM_LEADERBOARD_GEN,
                resource_config,
                max_eval_instances=max_eval_instances,
                apply_chat_template=apply_chat_template,
            )
            eval_jobs.append(olmo_generation)
        else:
            leaderboard_generation = evaluate_lm_evaluation_harness(
                name,
                model,
                KEY_GENERATION_TASKS,
                max_eval_instances=max_eval_instances,
                engine_kwargs=engine_kwargs,
                resource_config=resource_config,
                apply_chat_template=apply_chat_template,
            )
            eval_jobs.append(leaderboard_generation)

            olmo_generation = evaluate_lm_evaluation_harness(
                name,
                model,
                OPEN_LM_LEADERBOARD_GEN,
                max_eval_instances=max_eval_instances,
                engine_kwargs=engine_kwargs,
                resource_config=resource_config,
                apply_chat_template=apply_chat_template,
            )
            eval_jobs.append(olmo_generation)
    return eval_jobs


def default_key_evals(
    step: Checkpoint | str,
    resource_config: ResourceConfig,
    model_name: str | None = None,
    max_eval_instances: int | None = None,
    engine_kwargs: dict | None = DEFAULT_LM_EVAL_MODEL_KWARGS,
) -> list[Artifact]:
    """
    Create a list of eval artifacts for the model using LM Evaluation Harness.
    """
    name, model = extract_model_name_and_path(step)

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
            model,
            KEY_GENERATION_TASKS,
            max_eval_instances=max_eval_instances,
            engine_kwargs=engine_kwargs,
            resource_config=resource_config,
        ),
        evaluate_levanter_lm_evaluation_harness(
            model_name,
            model,
            KEY_MULTIPLE_CHOICE_TASKS,
            resource_config,
            max_eval_instances=max_eval_instances,
        ),
    ]


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
) -> Artifact:
    """
    Evaluate on ANY Harbor dataset from the registry.

    No custom adapters needed! Harbor's registry handles all datasets generically.

    Available datasets: https://harborframes.com/registry
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

    dispatch_resources = ResourceConfig.with_cpu() if model_path else resource_config

    def build_config(ctx: RunContext) -> EvaluationConfig:
        return EvaluationConfig(
            evaluator="harbor",
            model_name=model_name,
            model_path=model_path,
            evaluation_path=ctx.out,
            evals=[],
            max_eval_instances=max_eval_instances,
            discover_latest_checkpoint=False,
            engine_kwargs=engine_kwargs,
            resource_config=resource_config,
            apply_chat_template=apply_chat_template,
            wandb_tags=wandb_tags,
            generation_params=generation_params,
        )

    return derived(
        f"evaluation/harbor/{model_name}-{dataset}-{version}",
        fn=remote(
            evaluate,
            resources=dispatch_resources,
            env_vars=env_vars_from_keys(HARBOR_EVAL_ENV_KEYS),
            pip_dependency_groups=["harbor"],
        ),
        build_config=build_config,
    )


def evaluate_evalchemy(
    model_name: str,
    model: Checkpoint | str,
    evals: Sequence[EvalTaskConfig],
    max_eval_instances: int | None = None,
    engine_kwargs: dict | None = None,
    generation_params: dict | None = None,
    resource_config: ResourceConfig | None = None,
    apply_chat_template: bool = False,
    wandb_tags: list[str] | None = None,
    discover_latest_checkpoint: bool = True,
    base_eval_run_name: str | None = None,
) -> Artifact:
    """
    Create an eval artifact for the model using Evalchemy.

    Args:
        model_name: Name of the model.
        model: Checkpoint handle or path string to the model.
        evals: Evaluations to run with Evalchemy.
        max_eval_instances: Maximum number of evaluation instances to run.
        engine_kwargs: Additional engine kwargs for vLLM.
        generation_params: Generation parameters including:
            - temperature: float (e.g., 0.7)
            - top_p: float (e.g., 1.0)
            - max_gen_toks: int (e.g., 32768)
            - seeds: list[int] for multiple runs with different seeds
        resource_config: Resource configuration for the job.
        apply_chat_template: Whether to apply chat template.
        wandb_tags: Tags to add to the WandB run.
        discover_latest_checkpoint: Whether to discover the latest checkpoint.
    """
    task_names = "_".join(sorted(e.name for e in evals))
    seed = generation_params.get("seed") if generation_params else None
    seed_suffix = f"_seed{seed}" if seed is not None else ""
    step_name = f"evaluation/evalchemy/{model_name}/{task_names}{seed_suffix}"
    deps: tuple[Artifact, ...] = (model,) if isinstance(model, Checkpoint) else ()

    def build_config(ctx: RunContext) -> EvaluationConfig:
        model_path = ctx.path(model) if isinstance(model, Checkpoint) else model
        return EvaluationConfig(
            evaluator="evalchemy",
            model_name=model_name,
            model_path=model_path,
            evaluation_path=ctx.out,
            evals=evals,
            max_eval_instances=max_eval_instances,
            discover_latest_checkpoint=discover_latest_checkpoint,
            engine_kwargs=engine_kwargs,
            generation_params=generation_params,
            resource_config=resource_config,
            apply_chat_template=apply_chat_template,
            wandb_tags=wandb_tags,
            base_eval_run_name=base_eval_run_name,
        )

    return derived(
        step_name,
        fn=remote(evaluate, resources=resource_config, pip_dependency_groups=EVALCHEMY_DEPENDENCY_GROUPS),
        build_config=build_config,
        deps=deps,
    )


def default_evalchemy_eval(
    step: Checkpoint | str,
    resource_config: ResourceConfig = ResourceConfig.with_tpu("v5p-8"),
    evals: Sequence[EvalTaskConfig] | None = None,
    max_eval_instances: int | None = None,
    engine_kwargs: dict | None = None,
    generation_params: dict | None = None,
    apply_chat_template: bool = False,
    discover_latest_checkpoint: bool = True,
    base_eval_run_name: str | None = None,
) -> Artifact:
    """
    Create an eval artifact for the model using Evalchemy reasoning benchmarks.

    Args:
        step: Checkpoint handle or path string to evaluate.
        resource_config: Resource configuration (defaults to v5p-8 TPU).
        evals: List of evals to run. Defaults to EVALCHEMY_CORE_TASKS.
        max_eval_instances: Maximum number of evaluation instances to run.
        engine_kwargs: Additional vLLM engine kwargs (optional for evalchemy).
        generation_params: Generation parameters including:
            - temperature: float (e.g., 0.7)
            - top_p: float (e.g., 1.0)
            - max_gen_toks: int (e.g., 32768)
            - seed: int for reproducibility
        apply_chat_template: Whether to apply chat template.
        discover_latest_checkpoint: Whether to discover the latest checkpoint.
    """
    name, model = extract_model_name_and_path(step)

    if base_eval_run_name:
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
        model,
        evals,
        max_eval_instances=max_eval_instances,
        engine_kwargs=engine_kwargs or {},
        generation_params=generation_params,
        resource_config=resource_config,
        apply_chat_template=apply_chat_template,
        discover_latest_checkpoint=discover_latest_checkpoint,
        base_eval_run_name=base_eval_run_name,
    )


def compile_evalchemy_results(
    steps: list[Artifact],
    seeds: list[int] | None = None,
    base_eval_run_name: str | None = None,
    model_path: str | None = None,
    task_name: str | None = None,
) -> Artifact:
    """
    Compile results from multiple Evalchemy evaluation artifacts into aggregated metrics.

    Takes a list of Artifacts from evalchemy evaluations and compiles the results into a
    single DataFrame, then logs averaged results to wandb.

    Args:
        steps: List of Artifacts from evalchemy evaluations (one per seed).
        seeds: List of seeds used for the evaluations (for wandb config).

    Returns:
        Artifact that compiles and logs aggregated results.
    """
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

    def build_config(ctx: RunContext) -> dict:
        input_paths = [ctx.path(step) + "/results.json" for step in steps]
        return {
            "input_paths": input_paths,
            "output_path": ctx.out,
            "seeds": seeds or [],
            "base_eval_run_name": base_eval_run_name,
            "model_path": model_path,
            "task_name": task_name,
        }

    return derived(
        compile_step_name,
        fn=compile_evalchemy_results_fn,
        build_config=build_config,
        deps=tuple(steps),
    )


def build_evalchemy_eval_steps(
    checkpoints: dict[str | None, list[str]],
    task_seed_groups: list[tuple[list[EvalTaskConfig], list[int]]],
    base_generation_params: dict,
    resource_config: ResourceConfig,
    engine_kwargs: dict | None = None,
    apply_chat_template: bool = True,
    discover_latest_checkpoint: bool = False,
) -> tuple[list[Artifact], list[Artifact]]:
    """Build evaluation and compilation artifacts for an evalchemy experiment.

    Creates one evaluation artifact per (checkpoint, task, seed) combination, plus
    compilation artifacts that aggregate results across seeds for each (checkpoint, task).

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
        Tuple of (eval_artifacts, compile_artifacts).
    """
    eval_steps: list[Artifact] = []
    compile_steps: list[Artifact] = []

    for base_eval_run_name, checkpoint_paths in checkpoints.items():
        for checkpoint in checkpoint_paths:
            task_seed_pairs: list[tuple[EvalTaskConfig, list[int]]] = []
            for tasks, seeds in task_seed_groups:
                task_seed_pairs += [(t, seeds) for t in tasks]

            for task, seeds in task_seed_pairs:
                task_steps: list[Artifact] = []
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

    Builds eval and compile artifacts, then executes them via StepRunner
    with optional job concurrency limit.

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

    all_artifacts = eval_steps + compile_steps
    StepRunner().run([lower(x) for x in all_artifacts], max_concurrent=max_parallel_jobs)
