# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exercise artifact caching and merged task views through local stage execution."""

import json
import urllib.error
import urllib.request
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import pytest
from click.testing import CliRunner
from fray.current_client import set_current_client
from fray.local_backend import LocalClient
from fray.types import ResourceConfig
from marin.execution.artifact import Artifact
from marin.execution.lazy import ArtifactStep, run
from taskcompendium.models import TaskSpec
from taskcompendium.pipeline.datasets.numeric_answers import RECIPES
from taskcompendium.pipeline.inputs import RecipeInputs, SourceFiles, SourceFormat
from taskcompendium.pipeline.models import (
    Confidence,
    FilterPolicy,
    HFSource,
    Quality,
    ReferenceStatus,
    ReviewRecord,
    ReviewRubric,
    ReviewStatus,
    ReviewVerdict,
)
from taskcompendium.pipeline.rewriting import BatchRewriter
from taskcompendium.pipeline.zephyr import AuditExecution, ReviewConfig

from experiments.post_training.glm import GLM_BULK_TOKEN_ENV
from experiments.post_training.task_curation.pipeline import RewriteSelection, SourceBinding, build_workflow, main


def offline_request(*args: Any, **kwargs: Any) -> None:
    raise urllib.error.URLError("No HTTP service is available in the local graph tests")


@pytest.fixture(autouse=True)
def offline_http(monkeypatch):
    monkeypatch.setattr(urllib.request, "urlopen", offline_request)


@dataclass
class FixtureReviewer:
    reviewed_sources: list[str] = field(default_factory=list)
    reviewed_parent_ids: list[str] = field(default_factory=list)
    credential: str = "fixture-private-credential"

    @property
    def identity(self) -> dict[str, Any]:
        return {"reviewer": "local-fixture", "revision": "1"}

    def review(
        self,
        tasks: Sequence[TaskSpec],
        rubric: ReviewRubric,
        output_path: Path,
        *,
        originals: Mapping[str, TaskSpec] | None = None,
    ) -> list[ReviewRecord]:
        self.reviewed_sources.extend(task.source.dataset for task in tasks)
        if originals is not None:
            self.reviewed_parent_ids.extend(task.id for task in originals.values())
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


@dataclass(frozen=True)
class RewriteSubmission:
    file_id: str
    batch_id: str


@dataclass(frozen=True)
class RewriteOutput:
    output: str
    errors: str | None = None


@dataclass
class GraphRewriteService:
    submissions: dict[str, list[dict[str, Any]]] = field(default_factory=dict)

    def submit(self, requests, filename):
        batch_id = f"batch-{len(self.submissions)}"
        self.submissions[batch_id] = list(requests)
        return RewriteSubmission("file-0", batch_id)

    def wait(self, batch_id, poll_seconds):
        return {"id": batch_id, "status": "completed"}

    def output(self, batch):
        responses = []
        for request in self.submissions[batch["id"]]:
            task_id = request["custom_id"]
            responses.append(
                {
                    "custom_id": task_id,
                    "response": {
                        "status_code": 200,
                        "body": {
                            "choices": [
                                {
                                    "finish_reason": "stop",
                                    "message": {
                                        "role": "assistant",
                                        "content": None,
                                        "tool_calls": [
                                            {
                                                "id": "call-0",
                                                "type": "function",
                                                "function": {
                                                    "name": "propose_rewrite",
                                                    "arguments": json.dumps(
                                                        {
                                                            "task_id": task_id,
                                                            "action": "rewrite",
                                                            "edits": [
                                                                {
                                                                    "old_text": "How many apples?",
                                                                    "replacement": "How many apples are there?",
                                                                }
                                                            ],
                                                            "reason": (
                                                                "Clarify the question without changing its answer."
                                                            ),
                                                        }
                                                    ),
                                                },
                                            }
                                        ],
                                    },
                                }
                            ]
                        },
                    },
                }
            )
        return RewriteOutput("".join(json.dumps(response) + "\n" for response in responses))


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
        directory = tmp_path / name
        directory.mkdir()
        snapshot = directory / "records.jsonl"
        text = "".join(json.dumps(row) + "\n" for row in rows)
        snapshot.write_text(text)
        source = HFSource(f"fixture/{name}", "a" * 40, "default", "train")
        recipe = replace(
            RECIPES["svamp"],
            name=name,
            source=source,
            inputs=RecipeInputs(SourceFiles(("*.jsonl",), SourceFormat.JSONL), ()),
        )
        bindings.append(
            SourceBinding(
                name,
                "2026.10.01.1",
                recipe,
                ArtifactStep.adopt(name=f"fixture/{name}", version="2026.10.02.1", source=str(directory), kind=Artifact),
                ReviewConfig("fixture", "model-v1"),
                10,
            )
        )
    return bindings


def read_view(path: str, view: str) -> list[dict[str, Any]]:
    return [row for file in sorted(Path(path, view).glob("*.parquet")) for row in pq.read_table(file).to_pylist()]


def test_graph_keeps_rejection_evidence_and_refilters_cached_reviews(artifact_storage, bindings):
    reviewer = FixtureReviewer()
    resources = ResourceConfig.with_cpu(cpu=2, ram="2g")
    execution = AuditExecution(max_workers=2, review_batch_size=1, reviewer=reviewer)
    workflow = build_workflow(bindings, execution=execution, resources=resources)
    canonical = run(workflow.canonical, max_concurrent=2)[0]
    audit = read_view(canonical.path, "audit")
    accepted = read_view(canonical.path, "accepted")
    assert len(audit) == 3
    assert len(accepted) == 2
    assert {row["source_dataset"] for row in accepted} == {"fixture/first", "fixture/second"}
    rejected = [row for row in audit if row["filter_status"] == "reject"]
    assert len(rejected) == 1
    assert rejected[0]["normalization_reason"] == "missing_prompt"
    assert "normalize:missing_prompt" in rejected[0]["filter_reasons"]
    assert sorted(reviewer.reviewed_sources) == ["fixture/first", "fixture/second"]
    assert list(artifact_storage.glob("**/evidence/**/review/observed.json"))
    assert all(reviewer.credential not in file.read_text() for file in artifact_storage.glob("**/*.json"))

    stricter = build_workflow(
        bindings,
        execution=execution,
        resources=resources,
        policy=FilterPolicy(minimum_confidence=Confidence.HIGH),
    )
    strict = run(stricter.canonical, max_concurrent=2)[0]
    assert not read_view(strict.path, "accepted")
    assert len(read_view(strict.path, "audit")) == 3
    assert all(row["filter_status"] == "reject" for row in read_view(strict.path, "audit"))
    assert sorted(reviewer.reviewed_sources) == ["fixture/first", "fixture/second"]
    assert [source.downloaded.path() for source in stricter.sources] == [
        source.downloaded.path() for source in workflow.sources
    ]
    assert [source.audited.path() for source in stricter.sources] == [
        source.audited.path() for source in workflow.sources
    ]


def test_graph_changes_one_model_binding_without_reacquiring_sources(artifact_storage, bindings):
    reviewer = FixtureReviewer()
    execution = AuditExecution(max_workers=1, review_batch_size=2, reviewer=reviewer)
    workflow = build_workflow(bindings, execution=execution, resources=ResourceConfig.with_cpu(cpu=1, ram="2g"))
    run(workflow.canonical, max_concurrent=2)

    moved_execution = replace(execution, max_workers=2, review_batch_size=1)
    moved = build_workflow(bindings, execution=moved_execution, resources=ResourceConfig.with_cpu(cpu=2, ram="4g"))
    run(moved.canonical, max_concurrent=2)
    assert sorted(reviewer.reviewed_sources) == ["fixture/first", "fixture/second"]
    assert [source.audited.fingerprint() for source in moved.sources] == [
        source.audited.fingerprint() for source in workflow.sources
    ]

    changed = [replace(bindings[0], review=replace(bindings[0].review, model_revision="model-v2")), bindings[1]]
    revised = build_workflow(changed, execution=moved_execution, resources=ResourceConfig.with_cpu(cpu=2, ram="4g"))
    merged = run(revised.canonical, max_concurrent=2)[0]
    assert len(read_view(merged.path, "accepted")) == 2
    assert sorted(reviewer.reviewed_sources) == ["fixture/first", "fixture/first", "fixture/second"]
    assert revised.sources[0].downloaded.path() == workflow.sources[0].downloaded.path()
    assert revised.sources[0].audited.path() != workflow.sources[0].audited.path()
    assert revised.sources[1].accepted.path() == workflow.sources[1].accepted.path()


@pytest.mark.parametrize("source", ["math500", "aime24", "svamp", "gpqa", "instruction_following", "structured_output"])
def test_cli_plans_download_and_pipeline_without_credentials(tmp_path, monkeypatch, source):
    monkeypatch.setenv("MARIN_PREFIX", str(tmp_path / "artifacts"))
    monkeypatch.delenv(GLM_BULK_TOKEN_ENV, raising=False)
    result = CliRunner().invoke(main, ["--source", source, "--limit", "10", "--model-revision", "fixture"])
    assert result.exit_code == 0, result.output
    assert "task-curation/canonical" in result.output
    assert not (tmp_path / "artifacts").exists()


def test_graph_limit_spans_input_files_and_keeps_original_row_identity(artifact_storage, bindings, tmp_path):
    binding = bindings[0]
    # The first file contains a valid row and a malformed row; both consume input budget.
    (tmp_path / "first" / "second.jsonl").write_text(
        json.dumps({"Body": "Cy has 7 pears.", "Question": "How many pears?", "Answer": "7"}) + "\n"
    )
    reviewer = FixtureReviewer()
    execution = AuditExecution(max_workers=2, review_batch_size=1, reviewer=reviewer)
    resources = ResourceConfig.with_cpu(cpu=2, ram="2g")
    small = build_workflow([replace(binding, limit=2)], execution=execution, resources=resources)
    small_result = run(small.canonical, max_concurrent=2)[0]
    small_rows = read_view(small_result.path, "audit")
    assert len(small_rows) == 2
    assert sum(row["filter_status"] == "reject" for row in small_rows) == 1

    larger = build_workflow([replace(binding, limit=10)], execution=execution, resources=resources)
    large_result = run(larger.canonical, max_concurrent=2)[0]
    large_rows = read_view(large_result.path, "audit")
    assert len(large_rows) == 3
    assert {row["task_id"] for row in small_rows} <= {row["task_id"] for row in large_rows}
    assert small.sources[0].downloaded.path() == larger.sources[0].downloaded.path()


def test_graph_rewrite_rechecks_candidate_and_changes_only_rewrite_artifact(artifact_storage, bindings):
    binding = bindings[0]
    reviewer = FixtureReviewer()
    execution = AuditExecution(max_workers=1, review_batch_size=1, reviewer=reviewer)
    resources = ResourceConfig.with_cpu(cpu=2, ram="2g")
    base = build_workflow([binding], execution=execution, resources=resources)
    run(base.canonical, max_concurrent=2)
    original = read_view(base.sources[0].accepted.path(), "accepted")[0]
    service = GraphRewriteService()
    rewriter = BatchRewriter(service, "fixture", "model-v1")
    selection = RewriteSelection(
        (original["task_id"],),
        ReviewRubric("arithmetic-instruction-repair", "1", ("Keep the arithmetic question and answer.",)),
    )
    rewritten = build_workflow(
        [replace(binding, rewrite=selection)], execution=execution, resources=resources, rewriter=rewriter
    )
    canonical = run(rewritten.canonical, max_concurrent=2)[0]
    audit = read_view(canonical.path, "audit")
    candidate = next(row for row in audit if row["parent_id"] == original["task_id"])
    assert candidate["task_id"] != original["task_id"]
    assert candidate["filter_status"] == "keep"
    assert "How many apples are there?" in candidate["task_json"]
    assert json.loads(candidate["cleanup_lineage_json"])["original_audit"]["task_id"] == original["task_id"]
    assert reviewer.reviewed_parent_ids == [original["task_id"]]
    assert len(service.submissions) == 1
    assert rewritten.sources[0].audited.path() == base.sources[0].audited.path()
    assert rewritten.sources[0].accepted.path() != base.sources[0].accepted.path()

    revised = build_workflow(
        [replace(binding, rewrite=replace(selection, rubric=replace(selection.rubric, version="2")))],
        execution=execution,
        resources=resources,
        rewriter=rewriter,
    )
    assert revised.sources[0].accepted.path() != rewritten.sources[0].accepted.path()
    assert revised.sources[0].audited.path() == rewritten.sources[0].audited.path()
