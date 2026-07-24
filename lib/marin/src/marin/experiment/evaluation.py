# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Reusable lazy evaluation steps built on endpoint-oriented mechanisms."""

import logging
import tempfile
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, cast

from fray.cluster import ResourceConfig
from rigging.filesystem import marin_prefix, prefix_join

from marin.evaluation.eval_env import EVAL_ENV_KEYS, env_vars_from_keys
from marin.evaluation.eval_result import (
    EvalchemyResult,
    EvalReport,
    EvalResult,
    ReportEntry,
    compile_eval_report,
)
from marin.evaluation.evalchemy import (
    DEFAULT_NUM_CONCURRENT,
    EndpointMovedError,
    EvalchemyRunConfig,
    EvalchemyRuntimeConfig,
    run_evalchemy,
)
from marin.evaluation.evalchemy_runtime import EVALCHEMY_IMAGE
from marin.evaluation.evaluation_config import EvalTaskConfig, EvaluationConfig
from marin.evaluation.lm_eval import LM_EVAL_UV_PACKAGES, LmEvalResults, LmEvalRun, run_lm_eval
from marin.evaluation.run import evaluate
from marin.evaluation.serving_config import InferenceLaunch, ServeSpec, build_inference_launch
from marin.evaluation.utils import discover_hf_checkpoints
from marin.execution.artifact import Artifact, result_type_name
from marin.execution.build_context import resolve_version
from marin.execution.lazy import ArtifactStep, StepContext
from marin.execution.remote import remote
from marin.inference.iris import remote_inference
from marin.inference.vllm_server import validate_vllm_mode_env
from marin.training.training import LevanterCheckpoint
from rigging.filesystem import StoragePath

logger = logging.getLogger(__name__)

_ORCHESTRATOR_RESOURCES = ResourceConfig.with_cpu(cpu=1)
_ENDPOINT_RESTART_TIMEOUT = 2400.0


@dataclass(frozen=True)
class EvalchemyEvalConfig:
    """One self-contained Evalchemy artifact build."""

    model: str
    tasks: tuple[EvalTaskConfig, ...]
    out_path: str | None = None
    serve: ServeSpec = field(default_factory=ServeSpec)
    tokenizer: str | None = None
    max_gen_toks: int = 2048
    apply_chat_template: bool = False
    max_eval_instances: int | None = None
    num_concurrent: int = DEFAULT_NUM_CONCURRENT
    extra_gen_kwargs: dict[str, str] = field(default_factory=dict)
    eval_image: str = EVALCHEMY_IMAGE
    eval_cpu: float = 8.0
    eval_memory: str = "32g"
    eval_disk: str = "50g"


@dataclass(frozen=True)
class ServedEvalchemyRun:
    """Durable output and child jobs from one self-contained Evalchemy build."""

    out_path: str
    jobs: dict[str, str]


def _durable_output_dir(output_dir: str | None, run_id: str) -> str:
    if output_dir and "://" in output_dir:
        return output_dir.rstrip("/")

    durable = prefix_join(marin_prefix(), f"eval/evalchemy/{run_id}")
    if output_dir:
        logger.warning("routing pod-local Evalchemy output %r to %r", output_dir, durable)
    return durable


def run_served_evalchemy(config: EvalchemyEvalConfig) -> ServedEvalchemyRun:
    """Start inference, run one Evalchemy task group, and tear inference down."""
    if not config.tasks:
        raise ValueError("Evalchemy requires at least one task")
    output_dir = _durable_output_dir(config.out_path, uuid.uuid4().hex[:8])
    tokenizer = config.tokenizer or config.model
    if config.tokenizer is None and "://" in config.model:
        raise ValueError(
            f"model {config.model!r} is an object-store path; set tokenizer to an HF tokenizer id"
        )
    inference = build_inference_launch(config.model, tokenizer, config.serve)
    run_config = EvalchemyRunConfig(
        name="eval",
        tasks=config.tasks,
        apply_chat_template=config.apply_chat_template,
        max_gen_toks=config.max_gen_toks,
        max_eval_instances=config.max_eval_instances,
        num_concurrent=config.num_concurrent,
        extra_gen_kwargs=config.extra_gen_kwargs,
        runtime=EvalchemyRuntimeConfig(
            image=config.eval_image,
            cpu=config.eval_cpu,
            memory=config.eval_memory,
            disk=config.eval_disk,
            region=config.serve.region,
        ),
    )
    with remote_inference(
        inference.model,
        inference.engine,
        inference.iris,
        instances=inference.instances,
        broker=inference.broker,
        endpoint_access=inference.endpoint_access,
    ) as session:
        model = session.resolve_model()
        try:
            outcome = run_evalchemy(model, run_config, output_dir, observer=session)
        except EndpointMovedError:
            model = session.wait_model(_ENDPOINT_RESTART_TIMEOUT)
            outcome = run_evalchemy(model, run_config, output_dir, observer=session)
        jobs = {f"serve-{index}" if index else "serve": str(job.job_id) for index, job in enumerate(session.jobs)}
        jobs |= outcome.jobs
    return ServedEvalchemyRun(out_path=output_dir, jobs=jobs)


@dataclass(frozen=True)
class EvalGroup:
    """A reusable group of Evalchemy tasks and its serving policy."""

    tasks: tuple[EvalTaskConfig, ...]
    id: str
    serve: ServeSpec = field(default_factory=ServeSpec)
    tokenizer: str | None = None
    apply_chat_template: bool = False
    max_gen_toks: int = 2048
    max_eval_instances: int | None = None
    num_concurrent: int = DEFAULT_NUM_CONCURRENT
    extra_gen_kwargs: dict[str, str] = field(default_factory=dict)
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
    extra_gen_kwargs: Mapping[str, str] | None = None,
    discover_latest_checkpoint: bool = True,
    version: str | None = None,
) -> ArtifactStep[EvalchemyResult]:
    """Build one typed Evalchemy result artifact."""
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
            extra_gen_kwargs=dict(extra_gen_kwargs or {}),
        )

    return ArtifactStep(
        name=name,
        version=resolve_version(name, version),
        artifact_type=EvalchemyResult,
        run=remote(
            run_served_evalchemy,
            resources=_ORCHESTRATOR_RESOURCES,
            env_vars=env_vars_from_keys(EVAL_ENV_KEYS),
        ),
        build_config=build_config,
        deps=deps,
    )


def eval_step(
    model: ArtifactStep[LevanterCheckpoint],
    group: EvalGroup,
    *,
    version: str | None = None,
) -> ArtifactStep[EvalResult]:
    """Build one Evalchemy artifact from ``group``."""
    step = evaluate_evalchemy(
        model_name=model.name,
        model=model,
        evals=group.tasks,
        task_group_id=group.id,
        serve=group.serve,
        tokenizer=group.tokenizer,
        max_gen_toks=group.max_gen_toks,
        apply_chat_template=group.apply_chat_template,
        max_eval_instances=group.max_eval_instances,
        num_concurrent=group.num_concurrent,
        extra_gen_kwargs=group.extra_gen_kwargs,
        discover_latest_checkpoint=group.discover_latest_checkpoint,
        version=version,
    )
    return cast("ArtifactStep[EvalResult]", step)


def eval_steps(
    model: ArtifactStep[LevanterCheckpoint],
    groups: Sequence[EvalGroup],
    *,
    version: str | None = None,
) -> list[ArtifactStep[EvalResult]]:
    """Build one lazy result per task group."""
    return [eval_step(model, group, version=version) for group in groups]


def eval_report(
    results: Sequence[ArtifactStep[EvalResult]],
    *,
    name: str,
    version: str | None = None,
) -> ArtifactStep[EvalReport]:
    """Aggregate typed evaluation artifacts into one report."""
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


def run_served_lm_eval(
    inference: InferenceLaunch,
    eval_run: LmEvalRun,
    output_path: str,
) -> None:
    """Serve a model for one lm-eval run and upload its durable result tree."""
    with tempfile.TemporaryDirectory() as local_output:
        with remote_inference(
            inference.model,
            inference.engine,
            inference.iris,
            instances=inference.instances,
            broker=inference.broker,
            endpoint_access=inference.endpoint_access,
        ) as session:
            run_lm_eval(session.resolve_model(), eval_run, local_output)
        StoragePath(output_path).upload_from(local_output + "/", recursive=True)


def lm_eval_step(
    inference: InferenceLaunch,
    eval_run: LmEvalRun,
    *,
    name: str,
    version: str,
    parent_env_vars: Mapping[str, str],
) -> ArtifactStep[LmEvalResults]:
    """Build an lm-eval artifact using the shared remote-inference lifecycle."""
    worker_resources = inference.iris.worker_resources
    if inference.broker is not None:
        eval_run = replace(
            eval_run,
            extra_model_args={
                "num_concurrent": inference.broker.worker.max_in_flight,
                "timeout": int(inference.broker.proxy.request_timeout_seconds),
                **eval_run.extra_model_args,
            },
        )
    results_path = str(StoragePath("mirror://") / name / version)

    def build_config(context: StepContext) -> dict[str, Any]:
        return {
            "inference": replace(
                inference,
                iris=replace(inference.iris, worker_resources=context.runtime_arg("worker_resources")),
            ),
            "lm_eval_uv_packages": LM_EVAL_UV_PACKAGES,
            "eval_run": eval_run,
            "results_path": results_path,
        }

    def run_step(config: dict[str, Any]) -> LmEvalResults:
        if config["lm_eval_uv_packages"] != LM_EVAL_UV_PACKAGES:
            raise ValueError("artifact lm-eval packages must match the pinned runtime packages")
        remote(
            run_served_lm_eval,
            name=name,
            resources=_ORCHESTRATOR_RESOURCES,
            env_vars=dict(parent_env_vars),
        )(config["inference"], config["eval_run"], config["results_path"])
        return LmEvalResults(results_path=config["results_path"])

    return ArtifactStep(
        name=name,
        version=version,
        artifact_type=LmEvalResults,
        run=run_step,
        build_config=build_config,
        deps=(),
        runtime_args={"worker_resources": worker_resources},
    )


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
    """Build the legacy Harbor evaluator artifact."""
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

    name = f"evaluation/harbor/{model_name}-{dataset}-{version}"
    return ArtifactStep(
        name=name,
        version=resolve_version(name, artifact_version),
        artifact_type=Artifact,
        run=remote(
            evaluate,
            resources=dispatch_resources,
            env_vars=env_vars_from_keys(EVAL_ENV_KEYS),
            pip_dependency_groups=["harbor"],
        ),
        build_config=build_config,
    )
