# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Reusable lazy evaluation steps built on endpoint-oriented mechanisms."""

import logging
import tempfile
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import cast

from fray.cluster import ResourceConfig
from rigging.filesystem import StoragePath, marin_prefix, prefix_join

from marin.evaluation.eval_env import EVAL_ENV_KEYS, EVAL_RUNTIME_ENV_KEYS, env_vars_from_keys
from marin.evaluation.eval_result import (
    EvalchemyResult,
    EvalReport,
    EvalResult,
    ReportEntry,
    compile_eval_report,
)
from marin.evaluation.evalchemy import (
    EvalchemyRunConfig,
    run_evalchemy,
)
from marin.evaluation.hardware import AcceleratorChoice, Platform
from marin.evaluation.lm_eval import LM_EVAL_UV_PACKAGES, LmEvalResults, LmEvalRun, run_lm_eval
from marin.evaluation.model_config import ModelConfig, ResourceHint, ServeConfig
from marin.evaluation.serving_config import inference_config_for_model
from marin.evaluation.utils import discover_hf_checkpoints
from marin.execution.artifact import result_type_name
from marin.execution.build_context import resolve_version
from marin.execution.lazy import ArtifactStep, StepContext
from marin.execution.remote import remote
from marin.inference.config import RemoteInferenceConfig
from marin.inference.iris import remote_inference
from marin.training.training import LevanterCheckpoint

logger = logging.getLogger(__name__)


def _orchestrator_resources(accelerator: AcceleratorChoice) -> ResourceConfig:
    regions = [accelerator.region] if accelerator.region else None
    return ResourceConfig.with_cpu(cpu=1, regions=regions)


@dataclass(frozen=True)
class EvalchemyEvalConfig:
    """One self-contained Evalchemy artifact build."""

    model: ModelConfig
    accelerator: AcceleratorChoice
    run: EvalchemyRunConfig
    out_path: str | None = None


@dataclass(frozen=True)
class ServedEvalchemyRun:
    """Durable output and child jobs from one self-contained Evalchemy build."""

    out_path: str
    jobs: dict[str, str]


@dataclass(frozen=True)
class _EvalReportConfig:
    entries: tuple[ReportEntry, ...]
    out_path: str


@dataclass(frozen=True)
class _LmEvalStepConfig:
    inference: RemoteInferenceConfig
    packages: tuple[str, ...]
    eval_run: LmEvalRun
    results_path: str


def _durable_output_dir(output_dir: str | None, run_id: str) -> str:
    if output_dir and "://" in output_dir:
        return str(StoragePath(output_dir))

    durable = prefix_join(marin_prefix(), f"eval/evalchemy/{run_id}")
    if output_dir:
        logger.warning("routing pod-local Evalchemy output %r to %r", output_dir, durable)
    return durable


def run_served_evalchemy(config: EvalchemyEvalConfig) -> ServedEvalchemyRun:
    """Start inference, run one Evalchemy task group, and tear inference down."""
    if not config.run.tasks:
        raise ValueError("Evalchemy requires at least one task")
    output_dir = _durable_output_dir(config.out_path, uuid.uuid4().hex[:8])
    if config.model.tokenizer is None and "://" in config.model.location:
        raise ValueError(f"model {config.model.location!r} is an object-store path; set tokenizer to an HF tokenizer id")
    runtime_env = env_vars_from_keys(EVAL_RUNTIME_ENV_KEYS)
    inference = inference_config_for_model(
        config.model,
        config.accelerator,
        env_vars=runtime_env,
    )
    with remote_inference(inference) as session:
        outcome = run_evalchemy(
            session.model,
            config.run,
            output_dir,
            env_vars=runtime_env,
        )
        jobs = {f"serve-{index}" if index else "serve": str(job.job_id) for index, job in enumerate(session.jobs)}
        jobs |= outcome.jobs
    return ServedEvalchemyRun(out_path=output_dir, jobs=jobs)


@dataclass(frozen=True)
class EvalGroup:
    """A reusable group of Evalchemy tasks and its serving policy."""

    config: EvalchemyRunConfig
    serve: ServeConfig = field(default_factory=ServeConfig)
    resource_hint: ResourceHint = field(default_factory=ResourceHint)
    accelerator: AcceleratorChoice = field(
        default_factory=lambda: AcceleratorChoice(platform=Platform.TPU, tpu_type="v6e-8")
    )
    tokenizer: str | None = None
    discover_latest_checkpoint: bool = True


def evaluate_evalchemy(
    model_name: str,
    model: ArtifactStep[LevanterCheckpoint],
    config: EvalchemyRunConfig,
    serve: ServeConfig,
    resource_hint: ResourceHint,
    accelerator: AcceleratorChoice,
    *,
    tokenizer: str | None = None,
    discover_latest_checkpoint: bool = True,
    version: str | None = None,
) -> ArtifactStep[EvalchemyResult]:
    """Build one typed Evalchemy result artifact."""
    deps = (model,)
    name = f"evaluation/evalchemy/{model_name}/{config.name}"

    def build_config(ctx: StepContext) -> EvalchemyEvalConfig:
        model_path = ctx.artifact_path(model)
        if discover_latest_checkpoint:
            model_path = discover_hf_checkpoints(model_path)[-1]
        return EvalchemyEvalConfig(
            model=ModelConfig(
                name=model_name,
                location=model_path,
                tokenizer=tokenizer,
                resource_hint=resource_hint,
                serve=serve,
            ),
            accelerator=accelerator,
            run=config,
            out_path=ctx.output_path,
        )

    return ArtifactStep(
        name=name,
        version=resolve_version(name, version),
        artifact_type=EvalchemyResult,
        run=remote(
            run_served_evalchemy,
            resources=_orchestrator_resources(accelerator),
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
        config=group.config,
        serve=group.serve,
        resource_hint=group.resource_hint,
        accelerator=group.accelerator,
        tokenizer=group.tokenizer,
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

    def build_config(ctx: StepContext) -> _EvalReportConfig:
        return _EvalReportConfig(
            entries=tuple(
                ReportEntry(
                    path=ctx.artifact_path(result),
                    result_type=result_type_name(result.artifact_type),
                    label=result.name,
                )
                for result in results
            ),
            out_path=ctx.output_path,
        )

    def run(config: _EvalReportConfig) -> EvalReport:
        return compile_eval_report(config.entries, config.out_path)

    return ArtifactStep(
        name=step_name,
        version=resolve_version(step_name, version),
        artifact_type=EvalReport,
        run=run,
        build_config=build_config,
        deps=deps,
    )


def run_served_lm_eval(
    inference: RemoteInferenceConfig,
    eval_run: LmEvalRun,
    output_path: str,
) -> None:
    """Serve a model for one lm-eval run and upload its durable result tree."""
    with tempfile.TemporaryDirectory() as local_output:
        with remote_inference(inference) as session:
            run_lm_eval(session.model, eval_run, local_output)
        StoragePath(output_path).upload_from(local_output + "/", recursive=True)


def lm_eval_step(
    inference: RemoteInferenceConfig,
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

    def build_config(context: StepContext) -> _LmEvalStepConfig:
        return _LmEvalStepConfig(
            inference=replace(
                inference,
                iris=replace(inference.iris, worker_resources=context.runtime_arg("worker_resources")),
            ),
            packages=LM_EVAL_UV_PACKAGES,
            eval_run=eval_run,
            results_path=results_path,
        )

    def run_step(config: _LmEvalStepConfig) -> LmEvalResults:
        if config.packages != LM_EVAL_UV_PACKAGES:
            raise ValueError("artifact lm-eval packages must match the pinned runtime packages")
        remote(
            run_served_lm_eval,
            name=name,
            resources=ResourceConfig.with_cpu(cpu=1, regions=inference.iris.worker_resources.regions),
            env_vars=dict(parent_env_vars),
        )(config.inference, config.eval_run, config.results_path)
        return LmEvalResults(results_path=config.results_path)

    return ArtifactStep(
        name=name,
        version=version,
        artifact_type=LmEvalResults,
        run=run_step,
        build_config=build_config,
        deps=(),
        runtime_args={"worker_resources": worker_resources},
    )
