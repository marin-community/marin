# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exercise artifact caching and merged task views through local stage execution."""

import hashlib
import json
import urllib.error
import urllib.request
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import pytest
from click.testing import CliRunner
from fray.current_client import set_current_client
from fray.local_backend import LocalClient
from fray.types import ResourceConfig
from marin.execution.lazy import run
from taskcompendium.models import TaskSpec
from taskcompendium.pipeline.datasets import svamp
from taskcompendium.pipeline.models import (
    Confidence,
    FilterPolicy,
    Quality,
    ReferenceStatus,
    ReviewRecord,
    ReviewRubric,
    ReviewStatus,
    ReviewVerdict,
    SnapshotSource,
)
from taskcompendium.pipeline.zephyr import AuditExecution, ReviewConfig, SourceAcquisition

from experiments.post_training.glm import GLM_BULK_TOKEN_ENV
from experiments.post_training.task_curation_pipeline import SourceBinding, build_workflow, main


def offline_request(*args: Any, **kwargs: Any) -> None:
    raise urllib.error.URLError("No HTTP service is available in the local graph tests")


@pytest.fixture(autouse=True)
def offline_http(monkeypatch):
    monkeypatch.setattr(urllib.request, "urlopen", offline_request)


@dataclass
class FixtureReviewer:
    reviewed_sources: list[str] = field(default_factory=list)
    credential: str = "fixture-private-credential"

    @property
    def identity(self) -> dict[str, Any]:
        return {"reviewer": "local-fixture", "revision": "1"}

    def review(self, tasks: Sequence[TaskSpec], rubric: ReviewRubric, output_path: Path) -> list[ReviewRecord]:
        self.reviewed_sources.extend(task.source.dataset for task in tasks)
        output_path.mkdir(parents=True, exist_ok=True)
        (output_path / "observed.json").write_text(json.dumps([task.id for task in tasks]))
        return [
            ReviewRecord(
                task_id=task.id,
                status=ReviewStatus.REVIEWED,
                verdict=ReviewVerdict(
                    task_id=task.id,
                    quality=Quality.GOOD,
                    confidence=Confidence.MEDIUM,
                    reference_status=ReferenceStatus.CONSISTENT,
                    defects=[],
                    evidence="The public arithmetic question agrees with its reference.",
                ),
                detail="",
            )
            for task in tasks
        ]


@pytest.fixture
def artifact_storage(tmp_path, monkeypatch):
    prefix = tmp_path / "artifacts"
    monkeypatch.setenv("MARIN_PREFIX", str(prefix))
    client = LocalClient(max_threads=8)
    with set_current_client(client):
        yield prefix
    client.shutdown()


@pytest.fixture
def bindings(tmp_path):
    bindings = []
    for name, rows in (
        ("first", [{"Body": "Ada has 5 apples.", "Question": "How many apples?", "Answer": "5"}, {"Body": ""}]),
        ("second", [{"Body": "Bo has 3 oranges.", "Question": "How many oranges?", "Answer": "3"}]),
    ):
        snapshot = tmp_path / f"{name}.jsonl"
        text = "".join(json.dumps(row) + "\n" for row in rows)
        snapshot.write_text(text)
        source = SnapshotSource(f"fixture/{name}", "a" * 40, "default", "train", str(snapshot))
        recipe = replace(svamp.recipe, name=name, source=source)
        bindings.append(
            SourceBinding(
                name,
                "2026.10.01.1",
                recipe,
                SourceAcquisition(source, len(rows), hashlib.sha256(text.encode()).hexdigest()),
                ReviewConfig("fixture", "model-v1"),
            )
        )
    return bindings


def read_view(path: str) -> list[dict[str, Any]]:
    return [row for file in sorted(Path(path, "data").glob("*.parquet")) for row in pq.read_table(file).to_pylist()]


def test_graph_keeps_rejection_evidence_and_refilters_cached_reviews(artifact_storage, bindings):
    reviewer = FixtureReviewer()
    resources = ResourceConfig.with_cpu(cpu=2, ram="2g")
    execution = AuditExecution(max_workers=2, review_batch_size=1, reviewer=reviewer)
    workflow = build_workflow(bindings, execution=execution, resources=resources)
    merged_audit, merged_accepted = run(workflow.audit, workflow.accepted, max_concurrent=2)
    audit = read_view(merged_audit.path)
    accepted = read_view(merged_accepted.path)
    assert len(audit) == 3
    assert len(accepted) == 2
    assert {row["source_dataset"] for row in accepted} == {"fixture/first", "fixture/second"}
    rejected = [row for row in audit if row["filter_status"] == "reject"]
    assert len(rejected) == 1
    assert rejected[0]["normalization_reason"] == "missing_prompt"
    assert "normalize:missing_prompt" in rejected[0]["filter_reasons"]
    assert sorted(reviewer.reviewed_sources) == ["fixture/first", "fixture/second"]
    assert list(artifact_storage.glob("**/evidence/*/review/observed.json"))
    assert all(reviewer.credential not in file.read_text() for file in artifact_storage.glob("**/*.json"))

    stricter = build_workflow(
        bindings,
        execution=execution,
        resources=resources,
        policy=FilterPolicy(minimum_confidence=Confidence.HIGH),
    )
    strict_audit, strict_accepted = run(stricter.audit, stricter.accepted, max_concurrent=2)
    assert not read_view(strict_accepted.path)
    assert len(read_view(strict_audit.path)) == 3
    assert all(row["filter_status"] == "reject" for row in read_view(strict_audit.path))
    assert sorted(reviewer.reviewed_sources) == ["fixture/first", "fixture/second"]
    assert [source.acquired.path() for source in stricter.sources] == [
        source.acquired.path() for source in workflow.sources
    ]
    assert [source.audited.path() for source in stricter.sources] == [
        source.audited.path() for source in workflow.sources
    ]


def test_graph_changes_one_model_binding_without_reacquiring_sources(artifact_storage, bindings):
    reviewer = FixtureReviewer()
    execution = AuditExecution(max_workers=1, review_batch_size=2, reviewer=reviewer)
    workflow = build_workflow(bindings, execution=execution, resources=ResourceConfig.with_cpu(cpu=1, ram="2g"))
    run(workflow.audit, workflow.accepted, max_concurrent=2)

    moved_execution = replace(execution, max_workers=2, review_batch_size=1)
    moved = build_workflow(bindings, execution=moved_execution, resources=ResourceConfig.with_cpu(cpu=2, ram="4g"))
    run(moved.audit, moved.accepted, max_concurrent=2)
    assert sorted(reviewer.reviewed_sources) == ["fixture/first", "fixture/second"]
    assert [source.audited.fingerprint() for source in moved.sources] == [
        source.audited.fingerprint() for source in workflow.sources
    ]

    changed = [replace(bindings[0], review=replace(bindings[0].review, model_revision="model-v2")), bindings[1]]
    revised = build_workflow(changed, execution=moved_execution, resources=ResourceConfig.with_cpu(cpu=2, ram="4g"))
    merged = run(revised.accepted, max_concurrent=2)[0]
    assert len(read_view(merged.path)) == 2
    assert sorted(reviewer.reviewed_sources) == ["fixture/first", "fixture/first", "fixture/second"]
    assert revised.sources[0].acquired.path() == workflow.sources[0].acquired.path()
    assert revised.sources[0].audited.path() != workflow.sources[0].audited.path()
    assert revised.sources[1].accepted.path() == workflow.sources[1].accepted.path()


@pytest.mark.parametrize(
    "source_args",
    [
        ["--recipe", "taskcompendium.pipeline.datasets.svamp"],
        [
            "--snapshot-recipe",
            "taskcompendium.pipeline.datasets.knowledge_mcqa",
            "/unused-snapshot/task-sample.jsonl",
            "b" * 64,
        ],
    ],
)
def test_cli_plans_without_reviewer_credentials_or_source_downloads(tmp_path, monkeypatch, source_args):
    monkeypatch.setenv("MARIN_PREFIX", str(tmp_path / "artifacts"))
    monkeypatch.delenv(GLM_BULK_TOKEN_ENV, raising=False)
    result = CliRunner().invoke(main, [*source_args, "--model-revision", "fixture-deployment"])
    assert result.exit_code == 0, result.output
    assert "task-curation/merged/audit" in result.output
    assert "task-curation/merged/accepted" in result.output
    assert not (tmp_path / "artifacts").exists()


def test_cli_binds_source_manifest_without_reading_the_snapshot(tmp_path, monkeypatch):
    directory = tmp_path / "samples" / "knowledge_mcqa"
    directory.mkdir(parents=True)
    digest = "b" * 64
    (directory / "sample-manifest.json").write_text(json.dumps({"sample_rows": 7, "snapshot_sha256": digest}))
    monkeypatch.setenv("MARIN_PREFIX", str(tmp_path / "artifacts"))
    monkeypatch.delenv(GLM_BULK_TOKEN_ENV, raising=False)
    result = CliRunner().invoke(
        main,
        [
            "--sources-dir",
            str(directory.parent),
            "--source",
            "knowledge_mcqa",
            "--model-revision",
            "fixture-deployment",
        ],
    )
    assert result.exit_code == 0, result.output
    assert str(directory / "sample.jsonl") in result.output
    assert digest in result.output
    assert not (directory / "sample.jsonl").exists()
    assert not (tmp_path / "artifacts").exists()
