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

from collections.abc import Callable
from dataclasses import dataclass

from iris.client.client import iris_ctx
from iris.rpc import job_pb2
from marin.evaluation.hardware import default_platform
from marin.evaluation.model_config import ModelConfig
from marin.evaluation.records import read_record, record_path
from marin.execution.artifact import Artifact
from marin.execution.lazy import ArtifactStep, StepContext, artifact_identity
from marin.execution.step_runner import StepRunner

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
    """Identity summary for a model and its declared producer artifacts."""

    model: ModelConfig
    dependencies: tuple[str, ...]
    evals: str
    limit: int | None
    version: str


class EvaluationResult(Artifact):
    """Submitted evaluation group and its durable records and rollout archives."""

    group_id: str
    records_prefix: str
    run_ids: tuple[str, ...]
    results_paths: tuple[str, ...]


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
    model: ModelConfig,
    evals: str,
    *,
    version: str,
    deps: tuple[ArtifactStep, ...] = (),
    resolve_model: Callable[[StepContext], ModelConfig] | None = None,
    limit: int | None = None,
    accelerator: str | None = None,
    submission_cluster: str = EVALUATION_CONTROLLER_CLUSTER,
    federated_cluster: str | None = None,
) -> ArtifactStep[EvaluationResult]:
    """Evaluate a model with Evalchemy and Harbor.

    Experiments declare producer dependencies and resolve their paths in ``resolve_model``.
    Checked-in models use ``model`` directly without a producer step.
    """

    def build_config(ctx: StepContext) -> EvalStepConfig | EvalStepFingerprint:
        resolved_model = resolve_model(ctx) if resolve_model is not None else model
        if ctx.is_fingerprint:
            return EvalStepFingerprint(
                model=resolved_model,
                dependencies=tuple(artifact_identity(dep) for dep in deps),
                evals=evals,
                limit=limit,
                version=version,
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
        name=f"evals/{model.name}/{evals}",
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
    step = eval_step(models()["qwen3-1.7b"], "smoke", version="2026.07.19")
    StepRunner().run([step.lower()])


if __name__ == "__main__":
    main()
