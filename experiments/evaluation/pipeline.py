# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Eval runs as pipeline steps.

:func:`eval_step` wraps one launcher run (model x eval selection) as an :class:`ArtifactStep`, so
evals compose into ``StepRunner`` pipelines and can be triggered programmatically -- e.g. right
after a training pipeline exports a checkpoint, or fanned out over a model sweep. The step runs the
same orchestration as the CLI (serve the model once, run evalchemy against the served URL, write
``record.json`` + results + per-question parquet) to the shared eval output root. The step's artifact
path holds its cache record and typed launch result: an identical (model, evals, limit, version)
config is a cache hit.

The step submits an Iris orchestrator job and waits for its records. The launcher chooses the shared
GCS or CoreWeave ``evals`` output root, while the artifact path stores the pipeline cache record. The
pipeline itself must run where it can reach Iris::

    uv run iris --cluster marin job run -- python -m experiments.evaluation.pipeline

The demo pipeline below runs the smoke suite for one small model.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from iris.client.client import iris_ctx
from iris.rpc import job_pb2
from marin.evaluation.hardware import default_platform
from marin.evaluation.model_config import ModelConfig
from marin.evaluation.records import read_record, record_path
from marin.execution.artifact import Artifact
from marin.execution.lazy import ArtifactStep, StepContext, artifact_identity
from marin.execution.step_runner import StepRunner
from marin.inference.config import ResolvedModelLocator, SpeculativeMethod, SpeculativeServingConfig
from rigging.filesystem.storage_path import prefix_join

from experiments.evaluation.evals import resolve_eval_keys
from experiments.evaluation.launch import (
    EVALUATION_CONTROLLER_CLUSTER,
    LaunchSpec,
    launch_group,
    prepare_evaluation_batch,
)
from experiments.evaluation.models import models

_ACCELERATOR_RUNTIME_ARG = "accelerator"
_SUBMISSION_CLUSTER_RUNTIME_ARG = "submission_cluster"
_FEDERATED_CLUSTER_RUNTIME_ARG = "federated_cluster"


@dataclass(frozen=True)
class EvalStepConfig:
    """One pipeline eval's model, eval selection, version, and runtime overrides."""

    model: ModelConfig
    evals: str
    limit: int | None
    artifact_path: str
    accelerator: str | None
    submission_cluster: str
    federated_cluster: str | None
    version: str


@dataclass(frozen=True)
class EvalStepFingerprint:
    """Identity summary used before the target and draft value records exist."""

    model: str
    speculative: str | None
    evals: str
    limit: int | None
    version: str


class EvaluationResult(Artifact):
    """Submitted evaluation group and its durable records and rollout archives."""

    group_id: str
    records_prefix: str
    run_ids: tuple[str, ...]
    results_paths: tuple[str, ...]


class TargetModelArtifact(Artifact):
    """Resolved serving metadata for a target model produced by a pipeline step."""

    model: ModelConfig

    def url(self) -> str:
        return self.model.location

    def config(self) -> ModelConfig:
        return self.model


@dataclass(frozen=True)
class TargetModelStepConfig:
    artifact_path: str
    model: ModelConfig


def run_target_model_step(config: TargetModelStepConfig) -> TargetModelArtifact:
    return TargetModelArtifact(path=config.artifact_path, model=config.model)


def catalog_model_step(name: str, *, version: str) -> ArtifactStep[TargetModelArtifact]:
    """Record a checked-in model configuration as a typed pipeline dependency."""
    model = models()[name]
    return ArtifactStep(
        name=f"models/evaluation/{name}",
        version=version,
        artifact_type=TargetModelArtifact,
        run=run_target_model_step,
        build_config=lambda ctx: TargetModelStepConfig(artifact_path=ctx.output_path, model=model),
    )


def target_model_step(
    source: ArtifactStep,
    model: ModelConfig,
    *,
    name: str,
    version: str,
    relative_path: str = "",
) -> ArtifactStep[TargetModelArtifact]:
    """Record a produced or adopted HF model's resolved URI and serving metadata."""

    def build_config(ctx: StepContext) -> TargetModelStepConfig:
        location = ctx.artifact_path(source)
        if relative_path:
            location = prefix_join(location, relative_path)
        return TargetModelStepConfig(
            artifact_path=ctx.output_path,
            model=replace(model, location=location, identity=artifact_identity(source)),
        )

    return ArtifactStep(
        name=name,
        version=version,
        artifact_type=TargetModelArtifact,
        run=run_target_model_step,
        build_config=build_config,
        deps=(source,),
    )


class DraftModelArtifact(Artifact):
    """A resolved draft model that an evaluation step can load from its dependency record."""

    model: ResolvedModelLocator
    method: SpeculativeMethod
    num_speculative_tokens: int

    def url(self) -> str:
        return self.model.uri

    def config(self) -> SpeculativeServingConfig:
        return SpeculativeServingConfig(
            method=self.method,
            model=self.model,
            num_speculative_tokens=self.num_speculative_tokens,
        )


@dataclass(frozen=True)
class DraftModelStepConfig:
    artifact_path: str
    model: ResolvedModelLocator
    method: SpeculativeMethod
    num_speculative_tokens: int


def run_draft_model_step(config: DraftModelStepConfig) -> DraftModelArtifact:
    return DraftModelArtifact(
        path=config.artifact_path,
        model=config.model,
        method=config.method,
        num_speculative_tokens=config.num_speculative_tokens,
    )


def draft_model_step(
    source: ArtifactStep,
    *,
    name: str,
    version: str,
    method: SpeculativeMethod,
    num_speculative_tokens: int,
    relative_path: str = "",
) -> ArtifactStep[DraftModelArtifact]:
    """Adapt a produced or adopted draft source into a typed evaluation dependency."""

    def build_config(ctx: StepContext) -> DraftModelStepConfig:
        uri = ctx.artifact_path(source)
        if relative_path:
            uri = prefix_join(uri, relative_path)
        return DraftModelStepConfig(
            artifact_path=ctx.output_path,
            model=ResolvedModelLocator(uri=uri, identity=artifact_identity(source)),
            method=method,
            num_speculative_tokens=num_speculative_tokens,
        )

    return ArtifactStep(
        name=name,
        version=version,
        artifact_type=DraftModelArtifact,
        run=run_draft_model_step,
        build_config=build_config,
        deps=(source,),
    )


def run_eval_pipeline_step(config: EvalStepConfig) -> EvaluationResult:
    keys = resolve_eval_keys(config.evals)
    spec = LaunchSpec(
        model=config.model,
        evals=keys,
        evalchemy_definitions=(),
        harbor_definitions=(),
        platform=default_platform(config.model),
        accelerator=config.accelerator,
        limit=config.limit,
        records_prefix=None,
        submission_cluster=config.submission_cluster,
        federated_cluster=config.federated_cluster,
        priority_band=job_pb2.PRIORITY_BAND_INHERIT,
        version=config.version,
    )
    submitted = launch_group(prepare_evaluation_batch(spec), iris_ctx().client)
    submitted.job.wait(timeout=float("inf"))
    run_ids = tuple(evaluation.run_id for evaluation in submitted.evaluations)
    return EvaluationResult(
        path=config.artifact_path,
        group_id=submitted.group_id,
        records_prefix=submitted.records_prefix,
        run_ids=run_ids,
        results_paths=tuple(
            read_record(record_path(submitted.records_prefix, run_id)).results_path for run_id in run_ids
        ),
    )


def eval_step(
    model: ArtifactStep[TargetModelArtifact],
    evals: str,
    *,
    model_name: str,
    version: str,
    speculative: ArtifactStep[DraftModelArtifact] | None = None,
    limit: int | None = None,
    accelerator: str | None = None,
    submission_cluster: str = EVALUATION_CONTROLLER_CLUSTER,
    federated_cluster: str | None = None,
) -> ArtifactStep[EvaluationResult]:
    """Evaluate a static or upstream-produced model with Evalchemy and Harbor."""

    deps = (model,) if speculative is None else (model, speculative)

    def build_config(ctx: StepContext) -> EvalStepConfig | EvalStepFingerprint:
        if ctx.is_fingerprint:
            return EvalStepFingerprint(
                model=artifact_identity(model),
                speculative=artifact_identity(speculative) if speculative is not None else None,
                evals=evals,
                limit=limit,
                version=version,
            )
        resolved_model = ctx.resolved(model).config()
        if resolved_model.name != model_name:
            raise ValueError(f"evaluation model name {model_name!r} differs from resolved model {resolved_model.name!r}")
        if speculative is not None:
            resolved_model = replace(
                resolved_model,
                serve=replace(resolved_model.serve, speculative=ctx.resolved(speculative).config()),
            )
        return EvalStepConfig(
            model=resolved_model,
            evals=evals,
            limit=limit,
            artifact_path=ctx.output_path,
            accelerator=ctx.runtime_arg(_ACCELERATOR_RUNTIME_ARG),
            submission_cluster=ctx.runtime_arg(_SUBMISSION_CLUSTER_RUNTIME_ARG),
            federated_cluster=ctx.runtime_arg(_FEDERATED_CLUSTER_RUNTIME_ARG),
            version=version,
        )

    return ArtifactStep(
        name=f"evals/{model_name}/{evals}",
        version=version,
        artifact_type=EvaluationResult,
        run=run_eval_pipeline_step,
        build_config=build_config,
        deps=deps,
        runtime_args={
            _ACCELERATOR_RUNTIME_ARG: accelerator,
            _SUBMISSION_CLUSTER_RUNTIME_ARG: submission_cluster,
            _FEDERATED_CLUSTER_RUNTIME_ARG: federated_cluster,
        },
    )


def main() -> None:
    model = catalog_model_step("qwen3-1.7b", version="2026.07.19")
    step = eval_step(model, "smoke", model_name="qwen3-1.7b", version="2026.07.19")
    StepRunner().run([step.lower()])


if __name__ == "__main__":
    main()
