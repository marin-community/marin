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
from pathlib import Path
from typing import Protocol

from iris.client.client import iris_ctx
from iris.rpc import job_pb2
from marin.evaluation.harbor.runner import canonical_served_name
from marin.evaluation.hardware import default_platform
from marin.evaluation.model_config import ModelConfig
from marin.evaluation.utils import discover_hf_checkpoints
from marin.execution.artifact import Artifact
from marin.execution.lazy import ArtifactStep, StepContext
from marin.execution.step_runner import StepRunner

from experiments.evaluation.evals import EvalchemyDefinition, resolve_eval_keys
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
    evals: str | None
    evalchemy_config_path: str | None
    limit: int | None
    artifact_path: str
    accelerator: str | None
    submission_cluster: str
    federated_cluster: str | None
    version: str


class EvaluationResult(Artifact):
    """Submitted evaluation group and its durable run-record locations."""

    group_id: str
    records_prefix: str
    run_ids: tuple[str, ...]


class EvaluationModelSource(Protocol):
    """A static or produced HF model consumable by the shared evaluation runner."""

    def deps(self) -> tuple[ArtifactStep, ...]: ...

    def resolve(self, ctx: StepContext) -> ModelConfig: ...


@dataclass(frozen=True)
class CatalogEvaluationModel:
    """A model selected from the checked-in evaluation catalog."""

    name: str

    def deps(self) -> tuple[ArtifactStep, ...]:
        return ()

    def resolve(self, _ctx: StepContext) -> ModelConfig:
        return models()[self.name]


@dataclass(frozen=True)
class ProducedEvaluationModel:
    """Evaluate the newest Hugging Face checkpoint produced by an upstream step."""

    step: ArtifactStep[Artifact]
    model: ModelConfig

    def deps(self) -> tuple[ArtifactStep, ...]:
        return (self.step,)

    def resolve(self, ctx: StepContext) -> ModelConfig:
        if ctx.is_fingerprint:
            location = f"artifact://{self.step.name}@{self.step.version}"
        else:
            checkpoints = discover_hf_checkpoints(ctx.artifact_path(self.step))
            if not checkpoints:
                raise FileNotFoundError(f"no HF checkpoint found under {ctx.artifact_path(self.step)}")
            location = checkpoints[-1]
        return replace(self.model, location=location)


def run_eval_pipeline_step(config: EvalStepConfig) -> EvaluationResult:
    keys = resolve_eval_keys(config.evals) if config.evals is not None else ()
    evalchemy_definitions = ()
    if config.evalchemy_config_path is not None:
        path = Path(config.evalchemy_config_path)
        evalchemy_definitions = (EvalchemyDefinition(name=canonical_served_name(path.stem), config_path=path),)
    spec = LaunchSpec(
        model=config.model,
        evals=keys,
        evalchemy_definitions=evalchemy_definitions,
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
    return EvaluationResult(
        path=config.artifact_path,
        group_id=submitted.group_id,
        records_prefix=submitted.records_prefix,
        run_ids=tuple(evaluation.run_id for evaluation in submitted.evaluations),
    )


def eval_step(
    model: EvaluationModelSource,
    evals: str | None = None,
    *,
    evalchemy_config_path: Path | None = None,
    version: str,
    limit: int | None = None,
    accelerator: str | None = None,
    submission_cluster: str = EVALUATION_CONTROLLER_CLUSTER,
    federated_cluster: str | None = None,
) -> ArtifactStep[EvaluationResult]:
    """Evaluate a model with a registered selection or one checked-in Evalchemy config."""

    if (evals is None) == (evalchemy_config_path is None):
        raise ValueError("set exactly one of evals or evalchemy_config_path")
    if evalchemy_config_path is not None and not evalchemy_config_path.is_file():
        raise ValueError(f"Evalchemy config does not exist: {evalchemy_config_path}")

    selection_name = evals
    if evalchemy_config_path is not None:
        selection_name = canonical_served_name(evalchemy_config_path.stem)
    assert selection_name is not None
    deps = model.deps()

    def build_config(ctx: StepContext) -> EvalStepConfig:
        return EvalStepConfig(
            model=model.resolve(ctx),
            evals=evals,
            evalchemy_config_path=None if evalchemy_config_path is None else str(evalchemy_config_path),
            limit=limit,
            artifact_path=ctx.output_path,
            accelerator=ctx.runtime_arg(_ACCELERATOR_RUNTIME_ARG),
            submission_cluster=ctx.runtime_arg(_SUBMISSION_CLUSTER_RUNTIME_ARG),
            federated_cluster=ctx.runtime_arg(_FEDERATED_CLUSTER_RUNTIME_ARG),
            version=version,
        )

    model_name = model.resolve(StepContext.for_fingerprint(deps=deps)).name
    return ArtifactStep(
        name=f"evals/{model_name}/{selection_name}",
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
