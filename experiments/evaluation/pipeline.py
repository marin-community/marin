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
from typing import Protocol

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


class EvaluationResult(Artifact):
    """Submitted evaluation group and its durable records and rollout archives."""

    group_id: str
    records_prefix: str
    run_ids: tuple[str, ...]
    results_paths: tuple[str, ...]


class EvaluationModelSource(Protocol):
    """A static or produced HF model consumable by the shared evaluation runner."""

    def deps(self) -> tuple[ArtifactStep, ...]: ...

    def resolve(self, ctx: StepContext) -> ModelConfig: ...


class EvaluationSpeculativeSource(Protocol):
    """An optional draft resolved before the generic evaluation runner starts."""

    def deps(self) -> tuple[ArtifactStep, ...]: ...

    def resolve(self, ctx: StepContext) -> SpeculativeServingConfig: ...


@dataclass(frozen=True)
class CatalogEvaluationModel:
    """A model selected from the checked-in evaluation catalog."""

    name: str

    def deps(self) -> tuple[ArtifactStep, ...]:
        return ()

    def resolve(self, _ctx: StepContext) -> ModelConfig:
        return models()[self.name]


@dataclass(frozen=True)
class ArtifactEvaluationModel:
    """An HF-format model artifact adapted to the shared evaluation contract."""

    step: ArtifactStep
    model: ModelConfig
    relative_path: str = ""

    def deps(self) -> tuple[ArtifactStep, ...]:
        return (self.step,)

    def resolve(self, ctx: StepContext) -> ModelConfig:
        location = ctx.artifact_path(self.step)
        if self.relative_path:
            location = prefix_join(location, self.relative_path)
        return replace(self.model, location=location, identity=artifact_identity(self.step))


@dataclass(frozen=True)
class ArtifactSpeculativeModel:
    """A draft artifact resolved into the generic speculative-serving contract."""

    step: ArtifactStep
    method: SpeculativeMethod
    num_speculative_tokens: int
    relative_path: str = ""

    def deps(self) -> tuple[ArtifactStep, ...]:
        return (self.step,)

    def resolve(self, ctx: StepContext) -> SpeculativeServingConfig:
        uri = ctx.artifact_path(self.step)
        if self.relative_path:
            uri = prefix_join(uri, self.relative_path)
        return SpeculativeServingConfig(
            method=self.method,
            model=ResolvedModelLocator(uri=uri, identity=artifact_identity(self.step)),
            num_speculative_tokens=self.num_speculative_tokens,
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
    model: EvaluationModelSource,
    evals: str,
    *,
    version: str,
    speculative: EvaluationSpeculativeSource | None = None,
    limit: int | None = None,
    accelerator: str | None = None,
    submission_cluster: str = EVALUATION_CONTROLLER_CLUSTER,
    federated_cluster: str | None = None,
) -> ArtifactStep[EvaluationResult]:
    """Evaluate a static or upstream-produced model with Evalchemy and Harbor."""

    deps = tuple(dict.fromkeys((*model.deps(), *(speculative.deps() if speculative is not None else ()))))

    def build_config(ctx: StepContext) -> EvalStepConfig:
        resolved_model = model.resolve(ctx)
        if speculative is not None:
            resolved_model = replace(
                resolved_model,
                serve=replace(resolved_model.serve, speculative=speculative.resolve(ctx)),
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

    model_name = model.resolve(StepContext.for_fingerprint(deps=deps)).name
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
    step = eval_step(CatalogEvaluationModel("qwen3-1.7b"), "smoke", version="2026.07.19")
    StepRunner().run([step.lower()])


if __name__ == "__main__":
    main()
