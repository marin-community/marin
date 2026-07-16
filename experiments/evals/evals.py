# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Composable eval artifacts.

An eval is a lazy ``ArtifactStep``: one :class:`EvalGroup` of tasks, run against a served model,
produces one typed :class:`~marin.evaluation.eval_result.EvalchemyResult`, addressed by
``evaluation/evalchemy/{model}/{group}``. Because each group carries its own identity, a user
*combines the groups they want* instead of calling a fixed ``default_*`` bundle, and any group is
cached and reused across experiments. :func:`eval_report` aggregates a suite's results into one
:class:`~marin.evaluation.eval_result.EvalReport` a downstream step can pick up.

Every group runs through the evalchemy path (:mod:`experiments.evals.evalchemy.serve_and_eval`): the
model is served once as an OpenAI-compatible endpoint (marin-serve: vLLM or Levanter), the evalchemy
fork evaluates the tasks against that URL, and the server is torn down. The eval is decoupled from the
model backend by the URL (issue #4827), so multiple-choice and generation tasks run the same way — no
separate JAX-logprob backend.

The task menus (``core_evals`` / ``key_evals`` / ``base_model_evals``) are data — lists of
``EvalGroup`` — drawn from :mod:`experiments.evals.task_configs`, the same task menu the in-loop
``EvalSuite`` on ``train_lm`` uses. This module is the *post-hoc* path: evals on any existing
checkpoint. :func:`evaluate_harbor` is a separate backend for Harbor registry datasets.
"""

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import cast

from fray.cluster import ResourceConfig
from marin.evaluation.eval_result import (
    EvalchemyResult,
    EvalReport,
    EvalResult,
    ReportEntry,
    compile_eval_report,
)
from marin.evaluation.evaluation_config import EvalTaskConfig, EvaluationConfig
from marin.evaluation.evaluators.harbor_evaluator import HARBOR_EVAL_ENV_KEYS, env_vars_from_keys
from marin.evaluation.run import evaluate
from marin.evaluation.utils import discover_hf_checkpoints
from marin.execution.artifact import Artifact, result_type_name
from marin.execution.build_context import resolve_version
from marin.execution.lazy import ArtifactStep, StepContext
from marin.execution.remote import remote
from marin.inference.vllm_server import validate_vllm_mode_env
from marin.training.training import LevanterCheckpoint

from experiments.evals.evalchemy.serve_and_eval import (
    DEFAULT_NUM_CONCURRENT,
    EvalchemyEvalConfig,
    ServeSpec,
    serve_and_eval,
)
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

# lm-eval's ``code_eval`` metric (humaneval) executes model-generated code; the eval client sets
# HF_ALLOW_CODE_EVAL on its worker (serve_and_eval), so no per-group env is needed for it.

# The orchestrator step is a lightweight CPU job: it submits the serve + eval child jobs and waits.
# The serving slice rides on the group's ServeSpec, not on this resource.
_ORCHESTRATOR_RESOURCES = ResourceConfig.with_cpu(cpu=1)


@dataclass(frozen=True)
class EvalGroup:
    """A set of tasks evaluated together against one served model -> one ``EvalchemyResult`` artifact.

    The composable unit. Granularity is the *group*: a group is served once (one model boot) and its
    tasks are evaluated against that endpoint, so a suite of groups does not pay per-task startup.
    Users compose groups; each group is addressed and cached on its own.

    ``id`` is the task-group segment of the artifact name (``evaluation/evalchemy/{model}/{id}``) — the
    author's explicit statement of the group's identity. Changing the *tasks* under a fixed ``id``
    trips the advisory drift check (bump ``id`` or the step version to fork identity).

    ``serve`` picks the serving backend (vLLM or Levanter) and slice; ``apply_chat_template`` selects
    the chat vs completion OpenAI route. ``tokenizer`` overrides the HF tokenizer the eval client loads
    (defaults to the served checkpoint; set it to a base-model HF id when serving a ``gs://`` path).
    """

    tasks: tuple[EvalTaskConfig, ...]
    id: str
    serve: ServeSpec = field(default_factory=ServeSpec)
    tokenizer: str | None = None
    apply_chat_template: bool = False
    max_gen_toks: int = 2048
    max_eval_instances: int | None = None
    num_concurrent: int = DEFAULT_NUM_CONCURRENT
    discover_latest_checkpoint: bool = True


def evaluate_evalchemy(
    model_name: str,
    model: ArtifactStep[LevanterCheckpoint],
    evals: Sequence[EvalTaskConfig],
    task_group_id: str,
    serve: ServeSpec,
    *,
    tokenizer: str | None = None,
    max_gen_toks: int = 2048,
    apply_chat_template: bool = False,
    max_eval_instances: int | None = None,
    num_concurrent: int = DEFAULT_NUM_CONCURRENT,
    discover_latest_checkpoint: bool = True,
    version: str | None = None,
) -> ArtifactStep[EvalchemyResult]:
    """An evalchemy eval of one task group against a served model -> :class:`EvalchemyResult`.

    A CPU orchestrator step serves ``model`` (marin-serve), evaluates ``evals`` against its OpenAI URL
    with the evalchemy fork, and tears the server down. ``task_group_id`` is the stable identity
    segment so two groups on one model get distinct ``name@version`` addresses. ``tokenizer`` is the HF
    tokenizer the eval client loads (defaults to the served checkpoint, which the eval image cannot
    load from a ``gs://`` path -- set it to the base model's HF id in that case). ``version`` is
    explicit, or None to defer to the ambient BuildContext.
    """
    deps = (model,)
    name = f"evaluation/evalchemy/{model_name}/{task_group_id}"

    def build_config(ctx: StepContext) -> EvalchemyEvalConfig:
        model_path = ctx.artifact_path(model)
        if discover_latest_checkpoint:
            model_path = discover_hf_checkpoints(model_path)[-1]
        return EvalchemyEvalConfig(
            model=model_path,
            tasks=tuple(evals),
            out_path=ctx.output_path,
            serve=serve,
            tokenizer=tokenizer,
            max_gen_toks=max_gen_toks,
            apply_chat_template=apply_chat_template,
            max_eval_instances=max_eval_instances,
            num_concurrent=num_concurrent,
        )

    return ArtifactStep(
        name=name,
        version=resolve_version(name, version),
        artifact_type=EvalchemyResult,
        run=remote(
            serve_and_eval,
            resources=_ORCHESTRATOR_RESOURCES,
            env_vars=env_vars_from_keys(HARBOR_EVAL_ENV_KEYS),
        ),
        build_config=build_config,
        deps=deps,
    )


def eval_step(
    model: ArtifactStep[LevanterCheckpoint], group: EvalGroup, *, version: str | None = None
) -> ArtifactStep[EvalResult]:
    """One eval group -> one typed ``EvalResult`` artifact, addressed by model + group id.

    ``version`` is explicit, or None to defer to the ambient BuildContext.
    """
    step = evaluate_evalchemy(
        model_name=model.name,
        model=model,
        evals=list(group.tasks),
        task_group_id=group.id,
        serve=group.serve,
        tokenizer=group.tokenizer,
        max_gen_toks=group.max_gen_toks,
        apply_chat_template=group.apply_chat_template,
        max_eval_instances=group.max_eval_instances,
        num_concurrent=group.num_concurrent,
        discover_latest_checkpoint=group.discover_latest_checkpoint,
        version=version,
    )
    # ArtifactStep is invariant in its artifact type, so widen the concrete result to the base.
    return cast("ArtifactStep[EvalResult]", step)


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
    (one code path across groups) and writes the merged per-task metrics + averages. ``name`` is the
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


def core_evals(serve: ServeSpec | None = None) -> list[EvalGroup]:
    """CORE_TASKS as one served group — the default multiple-choice suite (logprobs over the OpenAI API)."""
    serve = serve or ServeSpec(tpu_type="v6e-8")
    return [EvalGroup(tasks=CORE_TASKS, id="core", serve=serve)]


def key_evals(serve: ServeSpec | None = None, max_eval_instances: int | None = None) -> list[EvalGroup]:
    """The key-evals bundle: generation tasks + multiple-choice tasks, each as its own served group.

    ``max_eval_instances`` caps examples per task — pass a small value for a fast cluster smoke.
    """
    serve = serve or ServeSpec(tpu_type="v6e-8")
    return [
        EvalGroup(
            tasks=KEY_GENERATION_TASKS,
            id="key_generation",
            serve=serve,
            max_gen_toks=4096,
            max_eval_instances=max_eval_instances,
        ),
        EvalGroup(
            tasks=KEY_MULTIPLE_CHOICE_TASKS,
            id="key_multiple_choice",
            serve=serve,
            max_eval_instances=max_eval_instances,
        ),
    ]


def base_model_evals(
    serve: ServeSpec | None = None,
    run_generation_evals: bool = True,
    discover_latest_checkpoint: bool = True,
) -> list[EvalGroup]:
    """Base-model suite: CORE+leaderboard and each MMLU cut as distinct served groups, plus generation.

    Each MMLU cut is its own group with its own ``id``, so all four run (unlike the old bundle, whose
    identical step names collided and silently dropped three of them).
    """
    serve = serve or ServeSpec(tpu_type="v6e-8")
    discover = discover_latest_checkpoint
    groups = [
        EvalGroup(CORE_TASKS_PLUS_LEADERBOARD, "core_leaderboard", serve=serve, discover_latest_checkpoint=discover),
        EvalGroup((MMLU_0_SHOT,), "mmlu_0shot", serve=serve, discover_latest_checkpoint=discover),
        EvalGroup((MMLU_5_SHOT,), "mmlu_5shot", serve=serve, discover_latest_checkpoint=discover),
        EvalGroup((MMLU_PRO_5_SHOT,), "mmlu_pro_5shot", serve=serve, discover_latest_checkpoint=discover),
    ]
    if run_generation_evals:
        groups.append(
            EvalGroup(
                BASE_GENERATION_TASKS,
                "base_generation",
                serve=serve,
                max_gen_toks=4096,
                discover_latest_checkpoint=discover,
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
