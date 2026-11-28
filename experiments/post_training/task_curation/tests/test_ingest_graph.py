# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The artifact graph orders stages, resumes sealed work and rejects bad caches."""

from pathlib import Path

import pytest

from experiments.post_training.task_curation.backend_census_manifest import seal_manifest
from experiments.post_training.task_curation.ingest_stages.graph import (
    IngestExecution,
    StageJob,
    ingest_graph,
    run_ingest_graph,
)
from experiments.post_training.task_curation.ingest_stages.ingest_config import load_ingest_config
from experiments.post_training.task_curation.ingest_stages.receipts import IngestStage, StageReceipt
from experiments.post_training.task_curation.release_stages.manifest import ArtifactRef, read_ref

CONFIGS = Path(__file__).parents[1] / "ingest_stages/configs"


def persist_stage_progress(job: StageJob) -> dict[str, ArtifactRef]:
    """A file-producing stage fixture at the worker's external-work boundary."""
    root = Path(job.output_root)
    root.mkdir(parents=True, exist_ok=True)
    evidence = root / "progress.json"
    if evidence.exists():
        raise FileExistsError("a sealed stage must resume without repeating its work")
    visited = []
    for reference in job.inputs.dependencies.values():
        dependency = StageReceipt.model_validate(read_ref(reference))
        visited = read_ref(dependency.outputs["progress"])["visited"]
    return {
        "progress": seal_manifest(
            {"visited": [*visited, job.inputs.stage.value], "policy": job.execution.policy.digest()}, str(evidence)
        )
    }


def persist_after_transient_failure(job: StageJob) -> dict[str, ArtifactRef]:
    marker = Path(job.execution.artifact_root) / "worker-failure-recorded"
    if job.inputs.stage is IngestStage.CENSUS and not marker.exists():
        marker.write_text("temporary worker failure\n")
        raise RuntimeError("temporary worker failure")
    return persist_stage_progress(job)


@pytest.fixture
def execution(tmp_path):
    return IngestExecution(
        policy=load_ingest_config(CONFIGS / "litmus-train.json"),
        source_revision="a" * 40,
        artifact_version="2026.11.28",
        artifact_root=str(tmp_path / "artifacts"),
        worker_argv=("python", "-m", "worker", "--source-revision", "a" * 40),
    )


def test_ingest_graph_orders_all_stages_and_resumes_without_rewriting_evidence(execution):
    graph = ingest_graph(execution, dict.fromkeys(IngestStage, persist_stage_progress))

    receipt = run_ingest_graph(graph)
    evidence = read_ref(receipt.outputs["progress"])
    assert evidence["visited"] == [stage.value for stage in IngestStage]
    original = {path: path.read_bytes() for path in Path(execution.artifact_root).rglob("*.json")}

    resumed = run_ingest_graph(graph)

    assert resumed == receipt
    assert {path: path.read_bytes() for path in original} == original


def test_ingest_graph_refuses_corrupt_upstream_evidence_before_terminal_cache_reuse(execution):
    graph = ingest_graph(execution, dict.fromkeys(IngestStage, persist_stage_progress))
    run_ingest_graph(graph)
    first_evidence = Path(graph.nodes[0].path(execution.artifact_root)) / "progress.json"
    first_evidence.write_text('{"visited": [], "policy": "corrupted"}\n')
    terminal_record = Path(graph.nodes[-1].path(execution.artifact_root)) / ".artifact.json"
    original = terminal_record.read_bytes()

    with pytest.raises(ValueError, match="evidence digest mismatch"):
        run_ingest_graph(graph)

    assert terminal_record.read_bytes() == original


def test_changed_policy_builds_a_separate_population_lineage(execution):
    graph = ingest_graph(execution, dict.fromkeys(IngestStage, persist_stage_progress))
    original = run_ingest_graph(graph)
    changed_policy = execution.policy.model_copy(
        update={"upstream": execution.policy.upstream.model_copy(update={"revision": "b" * 40})}
    )
    changed_execution = execution.model_copy(update={"policy": changed_policy})
    changed_graph = ingest_graph(changed_execution, dict.fromkeys(IngestStage, persist_stage_progress))

    changed = run_ingest_graph(changed_graph)

    assert read_ref(original.outputs["progress"])["policy"] == execution.policy.digest()
    assert read_ref(changed.outputs["progress"])["policy"] == changed_policy.digest()
    assert original.outputs["progress"].uri != changed.outputs["progress"].uri


def test_ingest_graph_retries_interrupted_work_without_repeating_sealed_predecessors(execution):
    graph = ingest_graph(execution, dict.fromkeys(IngestStage, persist_after_transient_failure))

    with pytest.raises(RuntimeError):
        run_ingest_graph(graph)

    assert not (Path(graph.nodes[-1].path(execution.artifact_root)) / ".artifact.json").exists()
    completed = {
        path: path.read_bytes()
        for node in graph.nodes[: list(IngestStage).index(IngestStage.CENSUS)]
        for path in Path(node.path(execution.artifact_root)).rglob("*.json")
    }
    assert completed

    receipt = run_ingest_graph(graph)

    assert read_ref(receipt.outputs["progress"])["visited"] == [stage.value for stage in IngestStage]
    assert {path: path.read_bytes() for path in completed} == completed
