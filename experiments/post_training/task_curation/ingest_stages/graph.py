# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bind the ordered ingest stages to Marin artifacts and verified resume receipts."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import partial
from typing import cast

from marin.execution.artifact import read_record
from marin.execution.lazy import ArtifactStep, StepContext, materialized_config
from marin.execution.lazy import run as run_handles
from pydantic import BaseModel, ConfigDict, Field
from taskcompendium.identity import canonical_sha256

from experiments.post_training.task_curation.ingest_stages.ingest_config import GIT_REVISION, IngestConfig
from experiments.post_training.task_curation.ingest_stages.receipts import (
    IngestStage,
    StageArtifact,
    StageInputs,
    StageReceipt,
    load_receipt,
    seal_receipt,
    verify_evidence,
)
from experiments.post_training.task_curation.local import artifact_prefix
from experiments.post_training.task_curation.release_stages.manifest import ArtifactRef, read_ref


class IngestExecution(BaseModel):
    """Worker inputs derived from policy, repository state and regional placement."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    policy: IngestConfig
    source_revision: str = Field(pattern=GIT_REVISION)
    artifact_version: str
    artifact_root: str
    worker_argv: tuple[str, ...] = Field(min_length=1)

    def identity(self) -> dict:
        return {
            "policy": self.policy.model_dump(mode="json"),
            "source_revision": self.source_revision,
            "worker_argv": list(self.worker_argv),
        }


@dataclass(frozen=True)
class StageJob:
    """The stage body receives validated policy and pinned upstream evidence."""

    execution: IngestExecution
    inputs: StageInputs
    output_root: str


StageHandler = Callable[[StageJob], dict[str, ArtifactRef]]
StageDependency = tuple[IngestStage, ArtifactStep[StageArtifact]]


@dataclass(frozen=True)
class IngestGraph:
    execution: IngestExecution
    nodes: tuple[ArtifactStep[StageArtifact], ...]


def parent_receipt(execution: IngestExecution, stage: IngestStage, reference: ArtifactRef) -> StageReceipt:
    """Check a dependency's lineage before exposing its outputs to a stage body."""
    receipt = StageReceipt.model_validate(read_ref(reference))
    inputs = receipt.inputs
    if (
        inputs.stage != stage
        or inputs.source_id != execution.policy.source_id
        or inputs.config_sha256 != execution.policy.digest()
        or inputs.source_revision != execution.source_revision
        or inputs.worker_argv_sha256 != canonical_sha256({"argv": list(execution.worker_argv)})
    ):
        raise ValueError("upstream ingest receipt belongs to another stage or execution")
    return load_receipt(reference, inputs)


def stage_config(
    execution: IngestExecution,
    stage: IngestStage,
    dependency: StageDependency | None,
    ctx: StepContext,
) -> StageJob | dict:
    if ctx.is_fingerprint:
        return {
            "execution": execution.identity(),
            "stage": stage.value,
            "dependency": ctx.artifact_path(dependency[1]) if dependency is not None else None,
        }
    dependencies: dict[IngestStage, ArtifactRef] = {}
    artifacts: dict[str, ArtifactRef] = {}
    if dependency is not None:
        previous_stage, previous_step = dependency
        reference = ctx.resolved(previous_step).receipt
        previous = parent_receipt(execution, previous_stage, reference)
        dependencies[previous_stage] = reference
        artifacts = {
            **previous.inputs.artifacts,
            **{f"{previous_stage.value}/{name}": ref for name, ref in previous.outputs.items()},
        }
    inputs = StageInputs(
        stage=stage,
        source_id=execution.policy.source_id,
        config_sha256=execution.policy.digest(),
        source_revision=execution.source_revision,
        worker_argv_sha256=canonical_sha256({"argv": list(execution.worker_argv)}),
        dependencies=dependencies,
        artifacts=artifacts,
    )
    return StageJob(execution=execution, inputs=inputs, output_root=ctx.output_path)


def execute_stage(handler: StageHandler, job: StageJob) -> StageArtifact:
    verify_evidence([*job.inputs.dependencies.values(), *job.inputs.artifacts.values()])
    outputs = handler(job)
    receipt = StageReceipt(inputs=job.inputs, outputs=outputs)
    return StageArtifact(path=job.output_root, receipt=seal_receipt(job.output_root, receipt))


def ingest_graph(execution: IngestExecution, handlers: Mapping[IngestStage, StageHandler]) -> IngestGraph:
    """Bind every required stage; incomplete handler sets refuse before execution."""
    missing = set(IngestStage) - handlers.keys()
    if missing:
        raise ValueError(f"ingest graph requires handlers for: {sorted(stage.value for stage in missing)}")
    identity = canonical_sha256(execution.identity())
    nodes: list[ArtifactStep[StageArtifact]] = []
    dependency: StageDependency | None = None
    for stage in IngestStage:
        step = ArtifactStep(
            name=f"atlas-ingest/{identity}/{stage.value}",
            version=execution.artifact_version,
            artifact_type=StageArtifact,
            run=partial(execute_stage, handlers[stage]),
            build_config=partial(stage_config, execution, stage, dependency),
            deps=(dependency[1],) if dependency is not None else (),
        )
        nodes.append(step)
        dependency = (stage, step)
    return IngestGraph(execution=execution, nodes=tuple(nodes))


def verify_built_stages(graph: IngestGraph) -> None:
    """Validate all cached stages even when terminal pruning would skip ancestors."""
    for step in graph.nodes:
        path = step.path(graph.execution.artifact_root)
        if read_record(path) is None:
            continue
        job = cast(StageJob, materialized_config(step, graph.execution.artifact_root))
        artifact = StageArtifact.raw_load(path)
        load_receipt(artifact.receipt, job.inputs)


def run_ingest_graph(graph: IngestGraph) -> StageReceipt:
    """Resume valid sealed stages and refuse stale or damaged evidence before work."""
    with artifact_prefix(graph.execution.artifact_root):
        verify_built_stages(graph)
        run_handles(graph.nodes[-1], max_concurrent=1)
        verify_built_stages(graph)
        terminal = graph.nodes[-1]
        job = cast(StageJob, materialized_config(terminal, graph.execution.artifact_root))
        return load_receipt(StageArtifact.raw_load(terminal.path()).receipt, job.inputs)
