# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Canonical set of evals.

Every helper builds a typed `(ModelDeployment, LmEvalRun|HarborRun)` pair and
wraps the executor step with `remote(...)` so the eval runs on a Fray worker.

The runner functions stand up a `VllmLauncher` (or construct a `RunningModel`
directly for external APIs) and hand the `RunningModel` to `run_lm_eval`,
`run_evalchemy`, or `run_harbor`.
"""

from __future__ import annotations

import dataclasses
import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass

from fray.cluster import ResourceConfig

from marin.evaluation.api import HarborRun, LmEvalRun, run_evalchemy, run_harbor, run_lm_eval
from marin.evaluation.evaluation_config import EvalTaskConfig
from marin.evaluation.evaluators.harbor_evaluator import sanitize_hosted_vllm_canonical_name
from marin.evaluation.evaluators.levanter_lm_eval_evaluator import run_levanter_lm_eval
from marin.evaluation.utils import discover_hf_checkpoints
from marin.execution.executor import (
    ExecutorStep,
    InputName,
    OutputName,
    executor_main,
    output_path_of,
    this_output_path,
    versioned,
)
from marin.execution.remote import remote
from marin.inference.model_launcher import (
    LITELLM_PROVIDER_URL,
    ModelDeployment,
    OpenAIEndpoint,
    RunningModel,
)
from marin.inference.vllm_launcher import VllmLauncher

from experiments.evals.engine_configs import DEFAULT_LM_EVAL_EXTRA_MODEL_ARGS, DEFAULT_VLLM_DEPLOYMENT_KWARGS
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


# =============================================================================
# Step-config dataclasses — passed to remote runners
# =============================================================================


@dataclass(frozen=True)
class _LmEvalStepConfig:
    deployment: ModelDeployment
    run: LmEvalRun
    discover_latest_checkpoint: bool = False


@dataclass(frozen=True)
class _EvalchemyStepConfig:
    deployment: ModelDeployment
    run: LmEvalRun
    discover_latest_checkpoint: bool = False


@dataclass(frozen=True)
class _HarborStepConfig:
    """Harbor step. `deployment is None` means external-API mode (Claude, GPT, ...)."""

    deployment: ModelDeployment | None
    run: HarborRun
    harbor_model_name: str
    discover_latest_checkpoint: bool = False


@dataclass(frozen=True)
class _LevanterLmEvalStepConfig:
    name: str
    path: str
    evals: list[EvalTaskConfig]
    output_path: str
    max_eval_instances: int | None = None
    apply_chat_template: bool = False
    wandb_tags: list[str] | None = None
    discover_latest_checkpoint: bool = False


# =============================================================================
# Step runners — execute on the Fray worker
# =============================================================================


def _run_lm_eval_step(config: _LmEvalStepConfig) -> None:
    deployment = _maybe_discover_latest_checkpoint(config.deployment, config.discover_latest_checkpoint)
    with VllmLauncher().launch(deployment) as model:
        run_lm_eval(model, config.run)


def _run_evalchemy_step(config: _EvalchemyStepConfig) -> None:
    deployment = _maybe_discover_latest_checkpoint(config.deployment, config.discover_latest_checkpoint)
    with VllmLauncher().launch(deployment) as model:
        run_evalchemy(model, config.run)


def _run_harbor_step(config: _HarborStepConfig) -> None:
    if config.deployment is None:
        model = RunningModel(
            endpoint=OpenAIEndpoint(url=LITELLM_PROVIDER_URL, model=config.harbor_model_name),
            tokenizer_ref="",
        )
        run_harbor(model, config.run)
        return
    deployment = _maybe_discover_latest_checkpoint(config.deployment, config.discover_latest_checkpoint)
    launcher = VllmLauncher(extra_args=["--served-model-name", config.harbor_model_name])
    with launcher.launch(deployment) as model:
        run_harbor(model, config.run)


def _run_levanter_lm_eval_step(config: _LevanterLmEvalStepConfig) -> None:
    path = _discover_latest_hf_checkpoint(config.path) if config.discover_latest_checkpoint else config.path
    run_levanter_lm_eval(
        name=config.name,
        path=path,
        evals=config.evals,
        output_path=config.output_path,
        max_eval_instances=config.max_eval_instances,
        apply_chat_template=config.apply_chat_template,
        wandb_tags=config.wandb_tags,
    )


def _maybe_discover_latest_checkpoint(deployment: ModelDeployment, discover: bool) -> ModelDeployment:
    if not discover or deployment.path is None:
        return deployment
    resolved = _discover_latest_hf_checkpoint(deployment.path)
    if resolved == deployment.path:
        return deployment
    return dataclasses.replace(deployment, path=resolved)


def _discover_latest_hf_checkpoint(path: str) -> str:
    """Fall back to the caller-provided path if no HF-style checkpoints are found."""
    ckpts = discover_hf_checkpoints(path)
    return ckpts[-1] if ckpts else path


# =============================================================================
# Path / name helpers
# =============================================================================


def _infer_model_name_for_path(model_path: str) -> str:
    """Human-readable model name from a path like `gs://.../dclm_7b0828/step-479999/`."""
    if model_path.endswith("/"):
        model_path = model_path[:-1]
    return "_".join(model_path.split("/")[-2:])


def _append_step_suffix(base: str, path: str) -> str:
    """Append `-step{N}` to `base` if `path` contains `step-{N}`, else return base."""
    match = re.search(r"step-(\d+)", path)
    return f"{base}-step{match.group(1)}" if match else base


def extract_model_name_and_path(step: ExecutorStep | InputName | str) -> tuple[str, InputName | str]:
    """Resolve a step-or-path into `(name, path)`.

    Always appends /hf for ExecutorSteps; `_discover_latest_hf_checkpoint` at
    run time handles either root or /hf layouts.
    """
    if isinstance(step, ExecutorStep):
        return step.name, output_path_of(step, "hf")
    if isinstance(step, InputName):
        # `InputName.hardcoded(...)` has `step.step is None`; treat as a direct path.
        if step.step is None:
            if step.name is None:
                raise ValueError("Invalid InputName: both `step` and `name` are None.")
            return _infer_model_name_for_path(step.name), step.name
        # If `name` is already set, the InputName refers to a specific subpath under the step's output.
        model_step_path = step if step.name is not None else output_path_of(step.step, "hf")
        return step.step.name, model_step_path
    if isinstance(step, str):
        return _infer_model_name_for_path(step), step
    raise ValueError(f"Invalid step type: {step}")


# =============================================================================
# lm_evaluation_harness
# =============================================================================


def evaluate_lm_evaluation_harness(
    model_name: str,
    model_path: str,
    evals: Sequence[EvalTaskConfig],
    *,
    max_eval_instances: int | None = None,
    deployment_kwargs: dict | None = None,
    extra_model_args: tuple[str, ...] = (),
    resource_config: ResourceConfig | None = None,
    apply_chat_template: bool = False,
    wandb_tags: list[str] | None = None,
    discover_latest_checkpoint: bool = True,
) -> ExecutorStep:
    """lm-evaluation-harness against a vLLM-backed OpenAI endpoint we stand up.

    Args:
        deployment_kwargs: vLLM server flags (e.g. `max_model_len`, `dtype`).
        extra_model_args: `k=v` strings appended to lm-eval's `--model_args`
            (client-side knobs, e.g. `("max_gen_toks=4096",)`).
    """
    deployment = ModelDeployment(path=model_path, engine_kwargs=dict(deployment_kwargs or {}))
    run = LmEvalRun(
        evals=list(evals),
        output_path=this_output_path(),
        apply_chat_template=apply_chat_template,
        generation_params={},
        extra_model_args=extra_model_args,
        max_eval_instances=max_eval_instances,
        wandb_tags=wandb_tags,
        base_eval_run_name=model_name,
    )
    config = _LmEvalStepConfig(deployment=deployment, run=run, discover_latest_checkpoint=discover_latest_checkpoint)
    return ExecutorStep(
        name=f"evaluation/lm_evaluation_harness/{model_name}",
        fn=remote(_run_lm_eval_step, resources=resource_config, pip_dependency_groups=["eval", "tpu"]),
        config=config,
    )


def evaluate_levanter_lm_evaluation_harness(
    model_name: str,
    model_path: str,
    evals: Sequence[EvalTaskConfig],
    resource_config: ResourceConfig,
    *,
    max_eval_instances: int | None = None,
    apply_chat_template: bool = False,
    discover_latest_checkpoint: bool = True,
) -> ExecutorStep:
    """Levanter in-process lm-eval harness."""
    logger.info(f"Running evals on the following tasks: {evals}")
    config = _LevanterLmEvalStepConfig(
        name=model_name,
        path=model_path,
        evals=versioned(list(evals)),
        output_path=this_output_path(),
        max_eval_instances=versioned(max_eval_instances),
        apply_chat_template=apply_chat_template,
        discover_latest_checkpoint=discover_latest_checkpoint,
    )
    return ExecutorStep(
        name=f"evaluation/lm_evaluation_harness_levanter/lmeval_debug_{model_name}",
        fn=remote(_run_levanter_lm_eval_step, resources=resource_config, pip_dependency_groups=["eval", "tpu"]),
        config=config,
    )


def default_eval(
    step: ExecutorStep | InputName | str,
    resource_config: ResourceConfig = ResourceConfig.with_tpu("v4-8"),
    evals: list[EvalTaskConfig] | None = None,
    max_eval_instances: int | None = None,
    apply_chat_template: bool = False,
    discover_latest_checkpoint: bool = True,
) -> ExecutorStep:
    """Default eval — runs the Levanter in-process lm-eval harness."""
    name, model_step_path = extract_model_name_and_path(step)
    logger.info(f"Creating default evaluation step for {name}")
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
    deployment_kwargs: dict | None = None,
    extra_model_args: tuple[str, ...] = DEFAULT_LM_EVAL_EXTRA_MODEL_ARGS,
    run_generation_evals: bool = True,
    discover_latest_checkpoint: bool = True,
) -> list[ExecutorStep]:
    if deployment_kwargs is None:
        deployment_kwargs = dict(DEFAULT_VLLM_DEPLOYMENT_KWARGS)
    eval_jobs = [
        default_eval(
            step=step,
            resource_config=resource_config,
            evals=CORE_TASKS_PLUS_LEADERBOARD,
            discover_latest_checkpoint=discover_latest_checkpoint,
        ),
        default_eval(
            step=step,
            resource_config=resource_config,
            evals=(MMLU_0_SHOT,),
            discover_latest_checkpoint=discover_latest_checkpoint,
        ),
        default_eval(
            step=step,
            resource_config=resource_config,
            evals=(MMLU_5_SHOT,),
            discover_latest_checkpoint=discover_latest_checkpoint,
        ),
        default_eval(
            step=step,
            resource_config=resource_config,
            evals=(MMLU_PRO_5_SHOT,),
            discover_latest_checkpoint=discover_latest_checkpoint,
        ),
    ]
    if run_generation_evals:
        name, model_step_path = extract_model_name_and_path(step)
        eval_jobs.append(
            evaluate_lm_evaluation_harness(
                name,
                model_step_path,
                BASE_GENERATION_TASKS,
                max_eval_instances=max_eval_instances,
                deployment_kwargs=deployment_kwargs,
                extra_model_args=extra_model_args,
                resource_config=resource_config,
                discover_latest_checkpoint=discover_latest_checkpoint,
            )
        )
    return eval_jobs


def default_sft_eval(
    step: ExecutorStep | InputName | str,
    resource_config: ResourceConfig = ResourceConfig.with_tpu("v6e-8"),
    max_eval_instances: int | None = None,
    deployment_kwargs: dict | None = None,
    extra_model_args: tuple[str, ...] = DEFAULT_LM_EVAL_EXTRA_MODEL_ARGS,
    run_generation_evals: bool = True,
    apply_chat_template: bool = True,
    use_levanter_inference: bool = False,
) -> list[ExecutorStep]:
    if deployment_kwargs is None:
        deployment_kwargs = dict(DEFAULT_VLLM_DEPLOYMENT_KWARGS)
    eval_jobs = [
        default_eval(
            step=step,
            resource_config=resource_config,
            evals=OPEN_LM_LEADERBOARD_MCQ,
            apply_chat_template=apply_chat_template,
        ),
        default_eval(
            step=step,
            resource_config=resource_config,
            evals=(MMLU_5_SHOT,),
            apply_chat_template=apply_chat_template,
        ),
        default_eval(
            step=step,
            resource_config=resource_config,
            evals=(MMLU_PRO_5_SHOT,),
            apply_chat_template=apply_chat_template,
        ),
    ]
    if not run_generation_evals:
        return eval_jobs

    name, model_step_path = extract_model_name_and_path(step)
    generation_tasks_groups = (KEY_GENERATION_TASKS, OPEN_LM_LEADERBOARD_GEN)
    for tasks in generation_tasks_groups:
        if use_levanter_inference:
            eval_jobs.append(
                evaluate_levanter_lm_evaluation_harness(
                    name,
                    model_step_path,
                    tasks,
                    resource_config,
                    max_eval_instances=max_eval_instances,
                    apply_chat_template=apply_chat_template,
                )
            )
        else:
            eval_jobs.append(
                evaluate_lm_evaluation_harness(
                    name,
                    model_step_path,
                    tasks,
                    max_eval_instances=max_eval_instances,
                    deployment_kwargs=deployment_kwargs,
                    extra_model_args=extra_model_args,
                    resource_config=resource_config,
                    apply_chat_template=apply_chat_template,
                )
            )
    return eval_jobs


def default_key_evals(
    step: ExecutorStep | InputName | str,
    resource_config: ResourceConfig,
    model_name: str | None = None,
    max_eval_instances: int | None = None,
    deployment_kwargs: dict | None = None,
    extra_model_args: tuple[str, ...] = DEFAULT_LM_EVAL_EXTRA_MODEL_ARGS,
) -> list[ExecutorStep]:
    """Key evals (one HTTP generation step, one Levanter MC step).

    The MC step stays on Levanter because lm-eval's `local-completions` client
    does not currently support multiple-choice tasks on TPU+vLLM (vllm issue #8499).
    """
    name, model_step_path = extract_model_name_and_path(step)
    if model_name is None:
        model_name = name
    if deployment_kwargs is None:
        deployment_kwargs = dict(DEFAULT_VLLM_DEPLOYMENT_KWARGS)

    return [
        evaluate_lm_evaluation_harness(
            model_name,
            model_step_path,
            KEY_GENERATION_TASKS,
            max_eval_instances=max_eval_instances,
            deployment_kwargs=deployment_kwargs,
            extra_model_args=extra_model_args,
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


# =============================================================================
# Harbor
# =============================================================================


def evaluate_harbor(
    model_name: str,
    model_path: str | None,
    dataset: str,
    version: str = "1.0",
    *,
    max_eval_instances: int | None = None,
    resource_config: ResourceConfig | None = None,
    wandb_tags: list[str] | None = None,
    agent: str = "claude-code",
    n_concurrent: int = 4,
    env: str = "local",
    agent_kwargs: dict | None = None,
    deployment_kwargs: dict | None = None,
) -> ExecutorStep:
    """Evaluate on any Harbor dataset from the registry.

    When `model_path is None`, runs as an external-API call (Claude, GPT, ...)
    — LiteLLM routes by model name and no vLLM server is launched.

    Otherwise stands up a vLLM sidecar and points Harbor at it via `hosted_vllm/<canonical>`.

    Args:
        model_name: Model identifier.
        model_path: Path to model. None for external APIs like Claude.
        dataset: Harbor dataset name.
        version: Dataset version.
        agent: Harbor agent type ("claude-code", "terminus-2", ...).
        n_concurrent: Number of parallel trials.
        env: Harbor environment ("local", "daytona", "e2b", "modal", ...).
        deployment_kwargs: vLLM engine kwargs (ignored when model_path is None).
    """
    canonical = sanitize_hosted_vllm_canonical_name(model_name)
    harbor_run = HarborRun(
        evals=[],
        output_path=this_output_path(),
        dataset=dataset,
        version=version,
        agent=agent,
        n_concurrent=n_concurrent,
        env=env,
        agent_kwargs=dict(agent_kwargs or {}),
        wandb_tags=wandb_tags,
        max_eval_instances=max_eval_instances,
    )
    deployment = (
        ModelDeployment(path=model_path, engine_kwargs=dict(deployment_kwargs or {})) if model_path is not None else None
    )
    config = _HarborStepConfig(
        deployment=deployment,
        run=harbor_run,
        harbor_model_name=canonical if model_path is not None else model_name,
    )

    # Dispatch on CPU when we're going to launch a vLLM sidecar ourselves (the
    # sidecar gets resource_config); on the caller-provided resources for pure
    # external-API calls.
    dispatch_resources = ResourceConfig.with_cpu() if model_path else resource_config
    return ExecutorStep(
        name=f"evaluation/harbor/{model_name}-{dataset}-{version}",
        fn=remote(_run_harbor_step, resources=dispatch_resources, pip_dependency_groups=["harbor"]),
        config=config,
    )


# =============================================================================
# Evalchemy
# =============================================================================


def evaluate_evalchemy(
    model_name: str,
    model_path: str,
    evals: Sequence[EvalTaskConfig],
    *,
    max_eval_instances: int | None = None,
    deployment_kwargs: dict | None = None,
    extra_model_args: tuple[str, ...] = (),
    generation_params: dict | None = None,
    batch_size: str = "auto",
    resource_config: ResourceConfig | None = None,
    apply_chat_template: bool = False,
    wandb_tags: list[str] | None = None,
    discover_latest_checkpoint: bool = True,
    base_eval_run_name: str | None = None,
) -> ExecutorStep:
    """Evalchemy against a vLLM-backed OpenAI endpoint we stand up.

    Args:
        deployment_kwargs: vLLM server flags (e.g. `tensor_parallel_size`,
            `max_num_seqs`, `max_model_len`).
        extra_model_args: Pre-formatted `k=v` strings for lm-eval's --model_args.
        generation_params: `temperature`, `top_p`, `max_gen_toks`, `seed`.
        batch_size: Passed as `--batch_size` to evalchemy/lm-eval.
        base_eval_run_name: Pre-assembled wandb name (include `-step{N}` suffix
            if desired — the evaluator no longer sees the checkpoint path).
    """
    task_names = "_".join(sorted(e.name for e in evals))
    seed = (generation_params or {}).get("seed")
    seed_suffix = f"_seed{seed}" if seed is not None else ""

    deployment = ModelDeployment(path=model_path, engine_kwargs=dict(deployment_kwargs or {}))
    run = LmEvalRun(
        evals=list(evals),
        output_path=this_output_path(),
        apply_chat_template=apply_chat_template,
        generation_params=dict(generation_params or {}),
        extra_model_args=extra_model_args,
        batch_size=batch_size,
        max_eval_instances=max_eval_instances,
        wandb_tags=wandb_tags,
        base_eval_run_name=base_eval_run_name,
    )
    config = _EvalchemyStepConfig(deployment=deployment, run=run, discover_latest_checkpoint=discover_latest_checkpoint)
    return ExecutorStep(
        name=f"evaluation/evalchemy/{model_name}/{task_names}{seed_suffix}",
        fn=remote(_run_evalchemy_step, resources=resource_config, pip_dependency_groups=["evalchemy", "tpu", "vllm"]),
        config=config,
    )


def default_evalchemy_eval(
    step: ExecutorStep | InputName | str,
    resource_config: ResourceConfig = ResourceConfig.with_tpu("v5p-8"),
    evals: Sequence[EvalTaskConfig] | None = None,
    max_eval_instances: int | None = None,
    deployment_kwargs: dict | None = None,
    extra_model_args: tuple[str, ...] = (),
    generation_params: dict | None = None,
    batch_size: str = "auto",
    apply_chat_template: bool = False,
    discover_latest_checkpoint: bool = True,
    base_eval_run_name: str | None = None,
) -> ExecutorStep:
    """Build the wandb run name here (evaluator no longer sees the checkpoint path)."""
    name, model_step_path = extract_model_name_and_path(step)
    resolved_base_name = base_eval_run_name
    if resolved_base_name is not None:
        # Pre-assemble `-step{N}` so the evaluator can use `base_eval_run_name` verbatim.
        path_str = step if isinstance(step, str) else name
        resolved_base_name = _append_step_suffix(resolved_base_name, path_str)
        name = resolved_base_name

    logger.info(f"Creating Evalchemy evaluation step for {name}")
    if evals is None:
        evals = EVALCHEMY_CORE_TASKS

    return evaluate_evalchemy(
        name,
        model_step_path,
        evals,
        max_eval_instances=max_eval_instances,
        deployment_kwargs=deployment_kwargs,
        extra_model_args=extra_model_args,
        generation_params=generation_params,
        batch_size=batch_size,
        resource_config=resource_config,
        apply_chat_template=apply_chat_template,
        discover_latest_checkpoint=discover_latest_checkpoint,
        base_eval_run_name=resolved_base_name,
    )


# =============================================================================
# Evalchemy: compile + experiment orchestration
# =============================================================================


def compile_evalchemy_results(
    steps: list[ExecutorStep],
    seeds: list[int] | None = None,
    base_eval_run_name: str | None = None,
    model_path: str | None = None,
    task_name: str | None = None,
) -> ExecutorStep:
    """Compile results from multiple Evalchemy evaluation steps into aggregated metrics."""
    input_paths = [step.cd("results.json") for step in steps]
    output_path = OutputName("compiled_results")

    if base_eval_run_name:
        model_id = _append_step_suffix(base_eval_run_name, model_path or "")
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
    deployment_kwargs: dict | None = None,
    extra_model_args: tuple[str, ...] = (),
    batch_size: str = "auto",
    apply_chat_template: bool = True,
    discover_latest_checkpoint: bool = False,
) -> tuple[list[ExecutorStep], list[ExecutorStep]]:
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
                    step = default_evalchemy_eval(
                        step=checkpoint,
                        resource_config=resource_config,
                        evals=[task],
                        deployment_kwargs=deployment_kwargs,
                        extra_model_args=extra_model_args,
                        batch_size=batch_size,
                        generation_params={**base_generation_params, "seed": seed},
                        apply_chat_template=apply_chat_template,
                        discover_latest_checkpoint=discover_latest_checkpoint,
                        base_eval_run_name=base_eval_run_name,
                    )
                    task_steps.append(step)
                    eval_steps.append(step)
                if len(seeds) > 1:
                    compile_steps.append(
                        compile_evalchemy_results(
                            task_steps,
                            seeds=seeds,
                            base_eval_run_name=base_eval_run_name,
                            model_path=checkpoint,
                            task_name=task.name,
                        )
                    )
    return eval_steps, compile_steps


def run_evalchemy_experiment(
    checkpoints: dict[str | None, list[str]],
    task_seed_groups: list[tuple[list[EvalTaskConfig], list[int]]],
    base_generation_params: dict,
    resource_config: ResourceConfig,
    deployment_kwargs: dict | None = None,
    extra_model_args: tuple[str, ...] = (),
    batch_size: str = "auto",
    apply_chat_template: bool = True,
    discover_latest_checkpoint: bool = False,
    max_parallel_jobs: int | None = None,
) -> None:
    eval_steps, compile_steps = build_evalchemy_eval_steps(
        checkpoints=checkpoints,
        task_seed_groups=task_seed_groups,
        base_generation_params=base_generation_params,
        resource_config=resource_config,
        deployment_kwargs=deployment_kwargs,
        extra_model_args=extra_model_args,
        batch_size=batch_size,
        apply_chat_template=apply_chat_template,
        discover_latest_checkpoint=discover_latest_checkpoint,
    )
    # Run eval steps in batches if max_parallel_jobs is set. Already-completed
    # steps are skipped via on-disk status files.
    if max_parallel_jobs is not None:
        for i in range(0, len(eval_steps), max_parallel_jobs):
            executor_main(steps=eval_steps[i : i + max_parallel_jobs])
    else:
        executor_main(steps=eval_steps)
    if compile_steps:
        executor_main(steps=compile_steps)


# Re-exports used by other experiments
__all__ = [
    "build_evalchemy_eval_steps",
    "compile_evalchemy_results",
    "default_base_eval",
    "default_eval",
    "default_evalchemy_eval",
    "default_key_evals",
    "default_sft_eval",
    "evaluate_evalchemy",
    "evaluate_harbor",
    "evaluate_levanter_lm_evaluation_harness",
    "evaluate_lm_evaluation_harness",
    "extract_model_name_and_path",
    "run_evalchemy_experiment",
]
