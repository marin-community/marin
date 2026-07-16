# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Composable eval artifacts.

An eval is a lazy ``ArtifactStep``: one backend run over one :class:`EvalGroup` of tasks produces
one typed :class:`~marin.evaluation.eval_result.EvalResult`, addressed by
``evaluation/{backend}/{model}/{group}``. Because each group carries its own identity, a user
*combines the groups they want* instead of calling a fixed ``default_*`` bundle, and any group is
cached and reused across experiments. :func:`eval_report` aggregates a suite's results into one
:class:`~marin.evaluation.eval_result.EvalReport` a downstream step can pick up.

The task menus (``core_evals`` / ``key_evals`` / ``base_model_evals``) are data — lists of
``EvalGroup`` — drawn from :mod:`experiments.evals.task_configs`, the same task menu the in-loop
``EvalSuite`` on ``train_lm`` uses. This module is the *post-hoc* path: evals on any existing
checkpoint. :func:`evaluate_harbor` is a separate backend for Harbor registry datasets.
"""

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import cast

from fray.cluster import ResourceConfig
from marin.evaluation.eval_result import (
    EvalReport,
    EvalResult,
    LevanterEvalResult,
    LmEvalHarnessResult,
    ReportEntry,
    compile_eval_report,
)
from marin.evaluation.evaluation_config import EvalTaskConfig, EvaluationConfig
from marin.evaluation.evaluators.harbor_evaluator import HARBOR_EVAL_ENV_KEYS, env_vars_from_keys
from marin.evaluation.run import evaluate
from marin.execution.artifact import Artifact, result_type_name
from marin.execution.build_context import resolve_version
from marin.execution.lazy import ArtifactStep, StepContext
from marin.execution.remote import remote
from marin.inference.vllm_server import validate_vllm_mode_env
from marin.training.training import LevanterCheckpoint

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
)

logger = logging.getLogger(__name__)

# Eval steps defer their version to the ambient BuildContext (see marin.experiment.cli): a driver
# supplies it via --version, so nothing here hardcodes a calendar version.

# lm-eval's ``code_eval`` metric (humaneval) executes model-generated code; the harness refuses to run
# it unless this is set on the worker, and the coordinator's env does not propagate to iris children.
HUMANEVAL_ENV = {"HF_ALLOW_CODE_EVAL": "1"}

# Marin optional-dependency extras installed on the eval worker. The vLLM backend is TPU-only (it
# shells out to the ``tpu-inference`` vllm CLI); the Levanter backend is JAX and runs on TPU or GPU,
# so its extras follow the device.
VLLM_EVAL_GROUPS = ["eval", "vllm", "tpu"]
_LEVANTER_EVAL_GROUPS: dict[str, list[str]] = {
    "tpu": ["eval", "tpu"],
    "gpu": ["eval", "gpu"],
    "cpu": ["eval", "cpu"],
}


def _levanter_dependency_groups(resources: ResourceConfig) -> list[str]:
    """The marin extras a Levanter eval worker installs, keyed by the resource's device kind."""
    groups = _LEVANTER_EVAL_GROUPS.get(resources.device.kind)
    if groups is None:
        raise ValueError(f"no Levanter eval dependency group for device kind {resources.device.kind!r}")
    return groups


class Backend(StrEnum):
    """The eval engine. Each value is the evaluator-registry key in ``marin.evaluation.run``."""

    LEVANTER = "levanter_lm_evaluation_harness"
    """MCQ / logprob tasks, run through Levanter; writes a flat ``results.json``."""

    LM_EVAL = "lm_evaluation_harness"
    """Generation tasks, run through vLLM + EleutherAI lm-eval; writes lm-eval's native tree."""


@dataclass(frozen=True)
class EvalGroup:
    """A set of tasks run together on one backend as a single job -> one ``EvalResult`` artifact.

    The composable unit. Granularity is the *group*, not the task: a group is what one backend job
    evaluates together (a single model load, batched), so a suite of groups does not pay per-task
    startup. Users compose groups; each group is addressed and cached on its own.

    ``id`` is the task-group segment of the artifact name (``evaluation/{backend}/{model}/{id}``) — the
    author's explicit statement of the group's identity. Changing the *tasks* under a fixed ``id``
    trips the advisory drift check (bump ``id`` or the step version to fork identity).
    """

    tasks: tuple[EvalTaskConfig, ...]
    backend: Backend
    resources: ResourceConfig
    id: str
    engine_kwargs: dict | None = None
    apply_chat_template: bool = False
    max_eval_instances: int | None = None
    discover_latest_checkpoint: bool = True
    env_vars: dict[str, str] | None = None
    """Extra env vars for the child worker (e.g. ``HF_ALLOW_CODE_EVAL=1`` for humaneval)."""


def evaluate_lm_evaluation_harness(
    model_name: str,
    model: ArtifactStep[LevanterCheckpoint],
    evals: list[EvalTaskConfig],
    task_group_id: str,
    max_eval_instances: int | None = None,
    engine_kwargs: dict | None = None,
    resource_config: ResourceConfig | None = None,
    apply_chat_template: bool = False,
    wandb_tags: list[str] | None = None,
    discover_latest_checkpoint: bool = True,
    env_vars: dict[str, str] | None = None,
    version: str | None = None,
) -> ArtifactStep[LmEvalHarnessResult]:
    """A vLLM lm-eval-harness eval of one task group -> :class:`LmEvalHarnessResult`. TPU-only.

    Args:
        model_name: Name of the model.
        model: LevanterCheckpoint handle to evaluate. Wrap a pre-existing checkpoint path with
            ``ArtifactStep.adopt(name, version, path, kind=LevanterCheckpoint)`` to pass one in.
        evals: Tasks to run with lm-eval.
        task_group_id: Stable identity segment for this task group, so two groups on one model get
            distinct ``name@version`` addresses instead of colliding.
        env_vars: Extra env vars for the child iris worker. Needed for vLLM-on-TPU bring-up (e.g.
            ``VLLM_ENABLE_V1_MULTIPROCESSING=0``) and code-eval tasks (``HF_ALLOW_CODE_EVAL=1``); the
            coordinator's ``os.environ`` does not propagate to iris-spawned children.
        version: Explicit version, or None to defer to the ambient BuildContext.
    """
    if resource_config is not None and resource_config.device.kind == "gpu":
        raise ValueError(
            "the vLLM lm-eval backend is TPU-only (tpu-inference); run generation evals on TPU, or "
            "use the Levanter backend on GPU"
        )
    deps = (model,)
    name = f"evaluation/lm_evaluation_harness/{model_name}/{task_group_id}"

    def build_config(ctx: StepContext) -> EvaluationConfig:
        return EvaluationConfig(
            evaluator=Backend.LM_EVAL,
            model_name=model_name,
            model_path=ctx.artifact_path(model),
            evaluation_path=ctx.output_path,
            evals=evals,
            max_eval_instances=max_eval_instances,
            discover_latest_checkpoint=discover_latest_checkpoint,
            engine_kwargs=engine_kwargs,
            resource_config=resource_config,
            apply_chat_template=apply_chat_template,
            wandb_tags=wandb_tags,
        )

    return ArtifactStep(
        name=name,
        version=resolve_version(name, version),
        artifact_type=LmEvalHarnessResult,
        run=remote(
            evaluate,
            resources=resource_config,
            pip_dependency_groups=VLLM_EVAL_GROUPS,
            env_vars=env_vars,
        ),
        build_config=build_config,
        deps=deps,
    )


def evaluate_levanter_lm_evaluation_harness(
    model_name: str,
    model: ArtifactStep[LevanterCheckpoint],
    evals: list[EvalTaskConfig],
    resource_config: ResourceConfig,
    task_group_id: str,
    max_eval_instances: int | None = None,
    apply_chat_template: bool = False,
    discover_latest_checkpoint: bool = True,
    env_vars: dict[str, str] | None = None,
    version: str | None = None,
) -> ArtifactStep[LevanterEvalResult]:
    """A Levanter lm-eval-harness eval of one task group -> :class:`LevanterEvalResult`.

    The Levanter evaluator writes a single top-level ``results.json``, so ``resolve(...).averages()``
    reads the cross-task scores without touching the directory layout. ``env_vars`` sets env on the
    child iris worker. ``version`` is explicit, or None to defer to the ambient BuildContext.
    """
    logger.info(f"Running levanter evals on the following tasks: {evals}")
    deps = (model,)
    name = f"evaluation/lm_evaluation_harness_levanter/{model_name}/{task_group_id}"

    def build_config(ctx: StepContext) -> EvaluationConfig:
        return EvaluationConfig(
            evaluator=Backend.LEVANTER,
            model_name=None,  # imputed automatically
            model_path=ctx.artifact_path(model),
            evaluation_path=ctx.output_path,
            evals=evals,
            discover_latest_checkpoint=discover_latest_checkpoint,
            max_eval_instances=max_eval_instances,
            resource_config=resource_config,
            apply_chat_template=apply_chat_template,
        )

    return ArtifactStep(
        name=name,
        version=resolve_version(name, version),
        artifact_type=LevanterEvalResult,
        run=remote(
            evaluate,
            resources=resource_config,
            pip_dependency_groups=_levanter_dependency_groups(resource_config),
            env_vars=env_vars,
        ),
        build_config=build_config,
        deps=deps,
    )


def eval_step(
    model: ArtifactStep[LevanterCheckpoint], group: EvalGroup, *, version: str | None = None
) -> ArtifactStep[EvalResult]:
    """One eval group -> one typed ``EvalResult`` artifact, addressed by backend + model + group id.

    ``version`` is explicit, or None to defer to the ambient BuildContext.
    """
    if group.backend == Backend.LEVANTER:
        levanter_step = evaluate_levanter_lm_evaluation_harness(
            model_name=model.name,
            model=model,
            evals=list(group.tasks),
            resource_config=group.resources,
            task_group_id=group.id,
            max_eval_instances=group.max_eval_instances,
            apply_chat_template=group.apply_chat_template,
            discover_latest_checkpoint=group.discover_latest_checkpoint,
            env_vars=group.env_vars,
            version=version,
        )
        # ArtifactStep is invariant in its artifact type, so widen the concrete result to the base.
        return cast("ArtifactStep[EvalResult]", levanter_step)
    if group.backend == Backend.LM_EVAL:
        lm_eval_step = evaluate_lm_evaluation_harness(
            model_name=model.name,
            model=model,
            evals=list(group.tasks),
            task_group_id=group.id,
            engine_kwargs=group.engine_kwargs,
            resource_config=group.resources,
            max_eval_instances=group.max_eval_instances,
            apply_chat_template=group.apply_chat_template,
            discover_latest_checkpoint=group.discover_latest_checkpoint,
            env_vars=group.env_vars,
            version=version,
        )
        return cast("ArtifactStep[EvalResult]", lm_eval_step)
    raise ValueError(f"unknown eval backend {group.backend!r}")


def eval_steps(
    model: ArtifactStep[LevanterCheckpoint], groups: Sequence[EvalGroup], *, version: str | None = None
) -> list[ArtifactStep[EvalResult]]:
    """Build one ``eval_step`` per group. A convenience over a list comprehension, not an abstraction."""
    return [eval_step(model, group, version=version) for group in groups]


def eval_report(
    results: Sequence[ArtifactStep[EvalResult]],
    *,
    name: str,
    version: str | None = None,
) -> ArtifactStep[EvalReport]:
    """Aggregate a suite's ``EvalResult`` artifacts into one :class:`EvalReport`.

    A CPU step depending on every result. It reads each result's metrics through the typed accessor
    (one code path across backends) and writes the merged per-task metrics + averages. ``name`` is the
    report's identity segment (``evaluation/report/{name}``), e.g. ``f"{model.name}/key"``. ``version``
    is explicit, or None to defer to the ambient BuildContext.
    """
    deps = tuple(results)
    step_name = f"evaluation/report/{name}"

    def build_config(ctx: StepContext) -> dict:
        return {
            "entries": [
                ReportEntry(
                    path=ctx.artifact_path(result),
                    result_type=result_type_name(result.artifact_type),
                    label=result.name,
                )
                for result in results
            ],
            "out": ctx.output_path,
        }

    def run(config: dict) -> EvalReport:
        return compile_eval_report(config["entries"], config["out"])

    return ArtifactStep(
        name=step_name,
        version=resolve_version(step_name, version),
        artifact_type=EvalReport,
        run=run,
        build_config=build_config,
        deps=deps,
    )


# --------------------------------------------------------------------------------------------------
# Task menus: lists of EvalGroup drawn from task_configs, the same menu in-loop EvalSuite uses.
# --------------------------------------------------------------------------------------------------


def core_evals(resources: ResourceConfig | None = None) -> list[EvalGroup]:
    """CORE_TASKS through the Levanter harness — the default multiple-choice suite."""
    resources = resources or ResourceConfig.with_tpu("v4-8")
    return [EvalGroup(tasks=CORE_TASKS, backend=Backend.LEVANTER, resources=resources, id="core")]


def key_evals(resources: ResourceConfig | None = None, max_eval_instances: int | None = None) -> list[EvalGroup]:
    """The key-evals bundle: generation tasks (lm-eval) + multiple-choice tasks (Levanter).

    ``max_eval_instances`` caps examples per task — pass a small value for a fast cluster smoke.
    """
    resources = resources or ResourceConfig.with_tpu("v6e-8")
    return [
        EvalGroup(
            tasks=KEY_GENERATION_TASKS,
            backend=Backend.LM_EVAL,
            resources=resources,
            id="key_generation",
            engine_kwargs=DEFAULT_LM_EVAL_MODEL_KWARGS,
            max_eval_instances=max_eval_instances,
            env_vars=HUMANEVAL_ENV,  # KEY_GENERATION_TASKS includes humaneval (code_eval)
        ),
        EvalGroup(
            tasks=KEY_MULTIPLE_CHOICE_TASKS,
            backend=Backend.LEVANTER,
            resources=resources,
            id="key_multiple_choice",
            max_eval_instances=max_eval_instances,
        ),
    ]


def base_model_evals(
    resources: ResourceConfig | None = None,
    engine_kwargs: dict | None = DEFAULT_LM_EVAL_MODEL_KWARGS,
    run_generation_evals: bool = True,
    discover_latest_checkpoint: bool = True,
) -> list[EvalGroup]:
    """Base-model suite: CORE+leaderboard and each MMLU cut as distinct Levanter groups, plus generation.

    Each MMLU cut is its own group with its own ``id``, so all four run (unlike the old bundle, whose
    identical step names collided and silently dropped three of them).
    """
    resources = resources or ResourceConfig.with_tpu("v6e-8")
    groups = [
        EvalGroup(
            CORE_TASKS_PLUS_LEADERBOARD,
            Backend.LEVANTER,
            resources,
            "core_leaderboard",
            discover_latest_checkpoint=discover_latest_checkpoint,
        ),
        EvalGroup(
            (MMLU_0_SHOT,),
            Backend.LEVANTER,
            resources,
            "mmlu_0shot",
            discover_latest_checkpoint=discover_latest_checkpoint,
        ),
        EvalGroup(
            (MMLU_5_SHOT,),
            Backend.LEVANTER,
            resources,
            "mmlu_5shot",
            discover_latest_checkpoint=discover_latest_checkpoint,
        ),
        EvalGroup(
            (MMLU_PRO_5_SHOT,),
            Backend.LEVANTER,
            resources,
            "mmlu_pro_5shot",
            discover_latest_checkpoint=discover_latest_checkpoint,
        ),
    ]
    if run_generation_evals:
        groups.append(
            EvalGroup(
                BASE_GENERATION_TASKS,
                Backend.LM_EVAL,
                resources,
                "base_generation",
                engine_kwargs=engine_kwargs,
                discover_latest_checkpoint=discover_latest_checkpoint,
            )
        )
    return groups


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
    artifact_version: str | None = None,
) -> ArtifactStep[Artifact]:
    """
    Evaluate on ANY Harbor dataset from the registry.

    ``version`` is the Harbor dataset version (part of the artifact name); ``artifact_version`` is the
    eval step's own version (explicit, or None to defer to the ambient BuildContext).

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

    def build_config(ctx: StepContext) -> EvaluationConfig:
        return EvaluationConfig(
            evaluator="harbor",
            model_name=model_name,
            model_path=model_path,
            evaluation_path=ctx.output_path,
            evals=[],
            max_eval_instances=max_eval_instances,
            discover_latest_checkpoint=False,
            engine_kwargs=engine_kwargs,
            resource_config=resource_config,
            apply_chat_template=apply_chat_template,
            wandb_tags=wandb_tags,
            generation_params=generation_params,
        )

    harbor_name = f"evaluation/harbor/{model_name}-{dataset}-{version}"
    return ArtifactStep(
        name=harbor_name,
        version=resolve_version(harbor_name, artifact_version),
        artifact_type=Artifact,
        run=remote(
            evaluate,
            resources=dispatch_resources,
            env_vars=env_vars_from_keys(HARBOR_EVAL_ENV_KEYS),
            pip_dependency_groups=["harbor"],
        ),
        build_config=build_config,
    )
