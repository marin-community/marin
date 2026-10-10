# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Curation contracts exercised through persisted outputs and a fake batch API."""

import hashlib
import json
import threading
import xml.etree.ElementTree as ET
from collections.abc import Iterator
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, cast

import pyarrow.parquet as pq
import pytest
from finestore.cache import PersistentKvCache
from fray.types import ResourceConfig
from pydantic import JsonValue
from rigging.filesystem.storage_path import StoragePath
from verifyit.spec import Constraint, ExactSpec

from taskcompendium.convert.answers import ifeval_task
from taskcompendium.grader import verifyit_package
from taskcompendium.grading import grade_answer
from taskcompendium.models import (
    AnswerType,
    ConversationInput,
    ConversationTrace,
    DockerBuildContext,
    EnvironmentRequirements,
    GradingAttempt,
    NoGrader,
    PlainText,
    ResourceGroups,
    ScriptGrader,
    Source,
    TaskSpec,
    TextMessage,
)
from taskcompendium.pipeline.audit_schema import TASK_SCHEMA
from taskcompendium.pipeline.filtering import task_decision
from taskcompendium.pipeline.inputs import ConversionContext, SourceFiles, SourceFormat
from taskcompendium.pipeline.models import (
    CheckStatus,
    Confidence,
    Disposition,
    EnvironmentInventory,
    FilterPolicy,
    ImportFailureKind,
    ImportRejection,
    RawRow,
    ReviewRubric,
    ReviewStatus,
)
from taskcompendium.pipeline.query_cache import cached_batch_output
from taskcompendium.pipeline.review import BatchReviewer, review_batch_id, review_records, review_tasks
from taskcompendium.pipeline.review_requests import batch_output
from taskcompendium.pipeline.source_quality import SourceQualityPolicy
from taskcompendium.pipeline.sources import SourceShard, source_shards, staged_file_rows, staged_inputs
from taskcompendium.pipeline.stages import (
    AuditExecution,
    assess_source_quality,
    audit_prepared_source,
    filter_source,
    prepare_source,
)
from taskcompendium.pipeline.transforms import normalize_row
from taskcompendium.pipeline.verification import verify_task
from taskcompendium.runtime.resources import inline_resource

from .pipeline_stages import (
    SOURCE_FILES,
    SVAMP_RUBRIC,
    convert_svamp,
    fixture_recipe,
    review_config,
    review_source,
    run_stages,
    stage_table,
    svamp_row_task,
    svamp_task,
)


@dataclass(frozen=True)
class Submission:
    file_id: str
    batch_id: str


@dataclass(frozen=True)
class Output:
    output: str
    errors: str | None = None


@dataclass
class BatchService:
    """Fake external inference service that records submitted requests."""

    confidence: str = "high"
    quality: str = "good"
    interrupted: bool = False
    invalid_first_batch: bool = False
    batches: dict[str, list[dict]] = field(default_factory=dict)
    files: dict[str, list[dict]] = field(default_factory=dict)

    def upload(self, requests, filename):
        file_id = f"file-{len(self.files)}"
        self.files[file_id] = list(requests)
        return file_id

    def create(self, file_id):
        batch_id = f"batch-{len(self.batches)}"
        self.batches[batch_id] = self.files[file_id]
        return Submission(file_id, batch_id)

    def wait(self, batch_id, poll_seconds):
        if self.interrupted:
            self.interrupted = False
            raise TimeoutError("Caller disconnected while the batch was running")
        return {"id": batch_id, "status": "completed"}

    def output(self, batch):
        rows = [
            response(request["custom_id"], confidence=self.confidence, quality=self.quality)
            for request in self.batches[batch["id"]]
        ]
        if self.invalid_first_batch and batch["id"] == "batch-0":
            for row in rows:
                row["response"]["body"]["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"] = "{"
        return Output("".join(json.dumps(row) + "\n" for row in rows))


def response(task_id, confidence="high", quality="good"):
    verdict = {
        "task_id": task_id,
        "quality": quality,
        "confidence": confidence,
        "reference_status": "consistent",
        "defects": [],
        "evidence": "The prompt supplies two apples and asks for the same count.",
    }
    return {
        "custom_id": task_id,
        "response": {
            "status_code": 200,
            "body": {
                "choices": [
                    {
                        "finish_reason": "tool_calls",
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call-0",
                                    "type": "function",
                                    "function": {"name": "review_task", "arguments": json.dumps(verdict)},
                                }
                            ],
                        },
                    }
                ]
            },
        },
    }


@pytest.fixture
def svamp_recipe():
    return fixture_recipe(convert_svamp, source=replace(SOURCE_FILES, patterns=("*.jsonl",)))


@pytest.fixture
def apple_row():
    return {
        "Body": "Aya has 2 apples.\u2028They belong to Aya.",
        "Question": "How many apples does Aya have?",
        "Answer": "2",
        "Equation": "2",
    }


def test_review_tasks_returns_attempts_and_reuses_prefetched_cache(tmp_path, apple_row):
    tasks = [svamp_task("task-b", apple_row), svamp_task("task-a", {**apple_row, "Body": "Bea has 2 apples."})]
    originals = [task.model_dump_json() for task in tasks]
    service = BatchService()
    reviewer = BatchReviewer(service, "fixture", "revision", query_cache_root=str(tmp_path / "cache"))
    first = review_tasks(tasks, SVAMP_RUBRIC, reviewer, cached=None)
    cached = reviewer.read_cache([tasks], SVAMP_RUBRIC)[0]
    second = review_tasks(tasks, SVAMP_RUBRIC, reviewer, cached=cached)
    assert [record.task_id for record in first.reviews] == ["task-b", "task-a"]
    assert [record.status for record in first.reviews] == [ReviewStatus.REVIEWED] * 2
    assert second.reviews == first.reviews
    assert [task.model_dump_json() for task in tasks] == originals
    assert len(service.batches) == 1
    initial, reused = first.attempts[0], second.attempts[0]
    assert initial.task_ids == ("task-b", "task-a")
    assert initial.requests.cache_hits == ()
    assert set(reused.requests.cache_hits) == set(initial.requests.cache_keys)
    assert reused.requests.cache_keys == initial.requests.cache_keys
    assert initial.requests.observations[0].batch_id == "batch-0"
    assert reused.requests.observations[0].batch_id is None
    assert review_records(reused.requests.output, list(reused.query_task_ids))[0].status == ReviewStatus.REVIEWED


def test_review_tasks_propagates_unexpected_response_membership(apple_row, monkeypatch):
    service = BatchService()
    monkeypatch.setattr(service, "output", lambda batch: Output(json.dumps(response("unexpected"))))
    with pytest.raises(ValueError, match="Unexpected batch response ID"):
        review_tasks(
            [svamp_task("task", apple_row)], SVAMP_RUBRIC, BatchReviewer(service, "fixture", "revision"), cached=None
        )


def test_audit_reviews_only_eligible_tasks_and_writes_batch_evidence_once(
    tmp_path, apple_row, svamp_recipe, monkeypatch
):
    monkeypatch.setattr("taskcompendium.pipeline.stages.AUDIT_SHARDS", 1)
    service = BatchService()
    manifest = run_stages(
        svamp_recipe,
        [{**apple_row, "Answer": "invalid"}, apple_row, {**apple_row, "Body": "Bea has 2 apples."}],
        output_path=tmp_path,
        limit=3,
        reviewer=BatchReviewer(service, "fixture", "revision"),
    )
    table = stage_table(tmp_path)
    rows = table.to_pylist()
    reviewed = [row for row in rows if row["review_status"] == "reviewed"]
    rejected = [row for row in rows if row["filter_status"] == "reject"]
    assert table.column_names == TASK_SCHEMA.names
    assert len(reviewed) == 2
    assert manifest["dispositions"] == {"keep": 2, "reject": 1}
    assert len(rejected) == 1
    assert rejected[0]["normalization_reason"] == "invalid_reference"
    assert rejected[0]["review_status"] is None
    assert len(service.batches) == 1
    submitted = [request["custom_id"] for request in service.batches["batch-0"]]
    assert set(submitted) == {row["task_id"] for row in reviewed}
    evidence = pq.read_table(tmp_path / "audited/evidence").to_pylist()
    assert len(evidence) == 1
    assert evidence[0]["batch_id"] == review_batch_id(submitted)
    saved = json.loads(evidence[0]["evidence_json"])
    assert [review["task_id"] for review in saved["reviews"]] == submitted
    assert saved["attempts"][0]["task_ids"] == submitted
    for review in saved["reviews"]:
        row = next(row for row in reviewed if row["task_id"] == review["task_id"])
        assert review["verdict"]["quality"] == row["review_quality"]
        assert review["verdict"]["evidence"] == row["review_evidence"]


def convert_mixed_import(row: RawRow, _context: ConversionContext) -> TaskSpec | ImportRejection:
    if "conversion_rejection" in row.data:
        return ImportRejection.model_validate(row.data["conversion_rejection"])
    if row.data.get("invalid_converted_task"):
        return TaskSpec.model_validate({"id": row.id})
    return svamp_row_task(row)


def test_conversion_failures_retain_raw_records_without_review_or_accepted_output(tmp_path, apple_row, svamp_recipe):
    rows = [
        apple_row,
        *[
            {
                **apple_row,
                "conversion_rejection": {
                    "kind": kind.value,
                    "reason": "unhandled_contract",
                    "detail": "Source evidence retained for examination",
                },
            }
            for kind in ImportFailureKind
        ],
        {**apple_row, "invalid_converted_task": True},
    ]
    recipe = replace(svamp_recipe, convert=convert_mixed_import)
    service = BatchService()
    manifest = run_stages(
        recipe,
        rows,
        output_path=tmp_path,
        limit=len(rows),
        reviewer=BatchReviewer(service, "fixture-model", "fixture-deployment"),
    )
    persisted = stage_table(tmp_path).to_pylist()
    assert manifest["normalized_rows"] == 1
    assert manifest["reviewed_rows"] == 1
    assert manifest["dispositions"] == {"keep": 1, "reject": 1, "defer": 3}
    assert [row["normalization_kind"] for row in persisted] == [
        None,
        "source_defect",
        "unsupported",
        "converter_error",
        "converter_error",
    ]
    assert [row["filter_status"] for row in persisted] == ["keep", "reject", "defer", "defer", "defer"]
    assert [json.loads(row["raw_json"])["data"] for row in persisted] == rows
    assert all(row["task_json"] is None and row["review_status"] is None for row in persisted[1:])
    assert [row["task_id"] for row in stage_table(tmp_path, "accepted").to_pylist()] == [persisted[0]["task_id"]]
    assert [request["custom_id"] for batch in service.batches.values() for request in batch] == [persisted[0]["task_id"]]


def convert_with_attachment(row: RawRow, _context: ConversionContext) -> TaskSpec:
    attachment = inline_resource("data/attachment.bin", b"x" * row.data["attachment_bytes"])
    return cast(TaskSpec, svamp_row_task(row)).model_copy(update={"resources": ResourceGroups(verifier=(attachment,))})


def test_tasks_over_the_resource_budget_are_deferred_and_counted(tmp_path, apple_row, svamp_recipe):
    rows = [
        {**apple_row, "attachment_bytes": 64},
        {**apple_row, "Body": "Aya has three apples.", "attachment_bytes": 65},
    ]
    recipe = replace(svamp_recipe, convert=convert_with_attachment, resource_budget_bytes=64)
    manifest = run_stages(
        recipe,
        rows,
        output_path=tmp_path,
        limit=len(rows),
        reviewer=BatchReviewer(BatchService(), "fixture-model", "fixture-deployment"),
    )
    persisted = stage_table(tmp_path).to_pylist()
    assert manifest["dispositions"] == {"keep": 1, "defer": 1}
    assert manifest["reasons"] == {"normalize:resources_over_budget": 1}
    assert [row["normalization_reason"] for row in persisted] == [None, "resources_over_budget"]


def convert_with_build_contexts(row: RawRow, _context: ConversionContext) -> TaskSpec:
    actor = EnvironmentRequirements(
        docker_build=DockerBuildContext(files=(inline_resource("Dockerfile", b"FROM scratch\n"),))
    )
    grader = EnvironmentRequirements(
        docker_build=DockerBuildContext(
            files=(
                inline_resource("Dockerfile", b"FROM scratch\n"),
                inline_resource("payload", b"x" * row.data["attachment_bytes"]),
            )
        )
    )
    return cast(TaskSpec, svamp_row_task(row)).model_copy(
        update={
            "environment_requirements": actor,
            "grader": ScriptGrader(environment=grader, argv=("true",)),
            "resources": ResourceGroups(worker=(inline_resource("input", b"data"),)),
        }
    )


def test_resource_budget_counts_both_build_contexts_and_workspace_files(apple_row, svamp_recipe):
    recipe = replace(svamp_recipe, convert=convert_with_build_contexts, resource_budget_bytes=64)
    audits = [
        normalize_row({"locator": "fixture:0", "data": {**apple_row, "attachment_bytes": size}}, recipe)["audit"]
        for size in (34, 35)
    ]
    assert audits[0]["normalization_rejection"] is None
    assert audits[1]["normalized"] is None
    assert audits[1]["normalization_rejection"]["reason"] == "resources_over_budget"
    assert audits[1]["decision"]["disposition"] == "defer"


def test_pipeline_accounts_for_rejects_duplicates_and_conflicting_keys(tmp_path, apple_row, svamp_recipe):
    conflicting = {**apple_row, "Body": "Aya has some apples."}
    rows = [
        apple_row,
        apple_row,
        {**conflicting, "Answer": "1"},
        {**conflicting, "Answer": "2"},
        {**apple_row, "Answer": None},
    ]
    service = BatchService()
    manifest = run_stages(
        svamp_recipe,
        rows,
        output_path=tmp_path,
        limit=10,
        reviewer=BatchReviewer(service, "fixture-model", "fixture-deployment"),
    )

    audit = stage_table(tmp_path).to_pylist()
    assert manifest["dispositions"] == {"keep": 1, "reject": 4}
    assert manifest["reasons"] == {
        "exact_semantic_duplicate": 1,
        "conflicting_references": 2,
        "normalize:invalid_reference": 1,
    }
    assert audit[1]["duplicate_of"] == audit[0]["task_id"]
    assert audit[4]["filter_reasons"][0] == "normalize:invalid_reference"
    assert all(audit[index]["filter_reasons"] == ["conflicting_references"] for index in (2, 3))
    accepted = [TaskSpec.model_validate_json(row["task_json"]) for row in stage_table(tmp_path, "accepted").to_pylist()]
    assert [task.id for task in accepted] == [audit[0]["task_id"]]
    assert accepted[0].source.revision == svamp_recipe.source.revision
    assert apple_row["Body"] in accepted[0].context.events[0].content
    assert "Equation" not in service.batches["batch-0"][0]["body"]["messages"][1]["content"]
    controls = audit[0]["checks"]
    assert [(check["check"], check["status"]) for check in controls] == [
        ("empty", "pass"),
        ("reference", "pass"),
        ("perturbed", "pass"),
    ]
    assert json.loads(audit[0]["raw_json"])["data"] == apple_row
    assert TaskSpec.model_validate_json(audit[0]["task_json"]) == accepted[0]
    assert audit[0]["review_evidence"] == "The prompt supplies two apples and asks for the same count."
    assert audit[1]["task_json"] is not None  # Duplicate inputs remain inspectable.
    assert audit[4]["task_json"] is None
    assert audit[4]["normalization_reason"] == "invalid_reference"
    assert audit[4]["normalization_detail"]


def test_audit_deduplicates_across_acquired_shards_on_storage_uri(tmp_path, apple_row, svamp_recipe):
    conflicting = {**apple_row, "Body": "Aya has some apples."}
    rows = [apple_row] * 1001 + [{**conflicting, "Answer": "1"}, {**conflicting, "Answer": "2"}]
    snapshot = tmp_path / "sample.jsonl"
    snapshot.write_text("".join(json.dumps(row) + "\n" for row in rows))
    root = f"memory://curation-{tmp_path.name}"
    (StoragePath(root) / "staged" / "source.jsonl").write_bytes(snapshot.read_bytes())
    service = BatchService()
    reviewer = BatchReviewer(service, "fixture-model", "fixture-deployment")
    review_source(
        f"{root}/staged",
        f"{root}/audited",
        svamp_recipe,
        AuditExecution(max_workers=2, review_batch_size=1, reviewer=reviewer),
        len(rows),
    )
    manifest = filter_source(f"{root}/audited", f"{root}/filtered", FilterPolicy(), max_workers=1)
    audit = []
    for file in (StoragePath(root) / "filtered/audit/*.parquet").glob():
        with file.open("rb") as stream:
            audit.extend(pq.read_table(stream).to_pylist())
    by_index = {int(row["source_row"].rsplit(":", 1)[1]): row for row in audit}
    assert manifest["dispositions"] == {"keep": 1, "reject": 1002}
    assert by_index[1000]["duplicate_of"] == by_index[0]["task_id"]
    assert all(by_index[index]["filter_reasons"] == ["conflicting_references"] for index in (1001, 1002))
    assert sum(len(requests) for requests in service.batches.values()) == 1
    assert len(audit) == len(rows)
    assert all(row["raw_json"] and row["task_json"] for row in audit)


def test_audit_restart_reuses_completed_shards_when_worker_count_changes(tmp_path, apple_row, svamp_recipe):
    rows = [{**apple_row, "Body": f"Person {index} has 2 apples."} for index in range(201)]
    snapshot = tmp_path / "sample.jsonl"
    snapshot.write_text("".join(json.dumps(row) + "\n" for row in rows))
    staged = tmp_path / "staged"
    staged.mkdir()
    (staged / "source.jsonl").write_bytes(snapshot.read_bytes())
    initial = BatchService()
    reviewer = BatchReviewer(initial, "fixture", "revision")
    review_source(
        str(staged), str(tmp_path / "audited"), svamp_recipe, AuditExecution(max_workers=1, reviewer=reviewer), len(rows)
    )
    completed = sorted((tmp_path / "audited/audit").glob("*.parquet"))
    assert sum(len(requests) for requests in initial.batches.values()) == len(rows)
    assert all(len(requests) <= 64 for requests in initial.batches.values())
    assert sum(pq.ParquetFile(path).metadata.num_rows > 0 for path in completed) > 1
    missing = next(path for path in completed if pq.ParquetFile(path).metadata.num_rows)
    completed.remove(missing)
    missing_ids = {row["task_id"] for row in pq.read_table(missing).to_pylist()}
    preserved = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in completed}
    missing.unlink()
    (tmp_path / "audited/manifest.json").unlink()
    resumed = BatchService(quality="bad")
    review_source(
        str(staged),
        str(tmp_path / "audited"),
        svamp_recipe,
        AuditExecution(max_workers=3, reviewer=BatchReviewer(resumed, "fixture", "revision")),
        len(rows),
    )
    reviewed = [request["custom_id"] for requests in resumed.batches.values() for request in requests]
    assert set(reviewed) == missing_ids and len(reviewed) == len(missing_ids)
    assert all(hashlib.sha256(path.read_bytes()).hexdigest() == digest for path, digest in preserved.items())
    manifest = filter_source(str(tmp_path / "audited"), str(tmp_path / "filtered"), FilterPolicy(), max_workers=1)
    assert manifest["input_rows"] == len(rows)
    assert manifest["dispositions"] == {"keep": len(rows) - len(missing_ids), "reject": len(missing_ids)}


@dataclass
class ConcurrentBatchService(BatchService):
    rendezvous: threading.Barrier = field(default_factory=lambda: threading.Barrier(2, timeout=10))

    def create(self, file_id):
        submission = super().create(file_id)
        # The external service responds only after both requests arrive. A
        # worker reserving its full budget for either task cannot make progress.
        self.rendezvous.wait()
        return submission


def test_one_preparation_partition_yields_independent_review_tasks(tmp_path, apple_row, svamp_recipe, monkeypatch):
    # Both batches originate in one preparation partition. Their provider waits
    # can overlap only if preparation persists independently schedulable files.
    monkeypatch.setattr("taskcompendium.pipeline.stages.AUDIT_SHARDS", 1)
    staged = tmp_path / "staged"
    staged.mkdir()
    rows = [{**apple_row, "Body": f"Person {index} has 2 apples."} for index in range(2)]
    (staged / "source.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    service = ConcurrentBatchService()
    reviewer = BatchReviewer(service, "fixture", "revision")
    manifest = review_source(
        str(staged),
        str(tmp_path / "audited"),
        svamp_recipe,
        AuditExecution(
            max_workers=1,
            review_batch_size=1,
            reviewer=reviewer,
            worker_resources=ResourceConfig.with_cpu(cpu=1, ram="2g"),
            review_task_resources=ResourceConfig.with_cpu(cpu=0.5, ram="1g"),
        ),
        limit=None,
    )
    assert manifest["reviewed_rows"] == 2
    assert sorted(len(requests) for requests in service.batches.values()) == [1, 1]


def test_pipeline_refilters_completed_shards_without_new_requests(tmp_path, apple_row, svamp_recipe):
    service = BatchService(confidence="medium")
    reviewer = BatchReviewer(service, "fixture-model", "fixture-deployment")
    strict_policy = FilterPolicy(id="high-confidence", minimum_confidence=Confidence.HIGH)
    resumed = run_stages(
        svamp_recipe,
        [apple_row],
        output_path=tmp_path,
        limit=1,
        reviewer=reviewer,
        policy=strict_policy,
    )
    assert resumed["dispositions"] == {"reject": 1}
    pending_audit = stage_table(tmp_path).to_pylist()[0]
    accepted = run_stages(
        svamp_recipe,
        iter(()),
        output_path=tmp_path,
        limit=1,
        reviewer=reviewer,
        policy=FilterPolicy(id="medium-confidence", minimum_confidence=Confidence.MEDIUM),
    )
    assert accepted["dispositions"] == {"keep": 1}
    assert len(service.batches) == 1
    assert stage_table(tmp_path, "accepted").num_rows == 1
    accepted_audit = stage_table(tmp_path).to_pylist()[0]
    assert accepted_audit["filter_status"] == "keep"
    assert pending_audit["filter_status"] == "reject"
    assert accepted_audit["raw_json"] == pending_audit["raw_json"]
    assert accepted_audit["review_evidence"] == pending_audit["review_evidence"]


def test_pipeline_retries_invalid_model_reply_and_preserves_both_attempts(tmp_path, apple_row, svamp_recipe):
    service = BatchService(invalid_first_batch=True)
    reviewer = BatchReviewer(service, "fixture-model", "fixture-deployment")
    manifest = run_stages(svamp_recipe, [apple_row], output_path=tmp_path, limit=1, reviewer=reviewer)
    assert manifest["dispositions"] == {"keep": 1}
    assert manifest["reviewed_rows"] == 1
    task_id = stage_table(tmp_path).to_pylist()[0]["task_id"]
    evidence = pq.read_table(next((tmp_path / "audited/evidence").glob("*.parquet"))).to_pylist()
    attempts = json.loads(evidence[0]["evidence_json"])["attempts"]
    initial = review_records(attempts[0]["requests"]["observations"][0]["output"], [task_id])
    retry = review_records(attempts[1]["requests"]["observations"][0]["output"], [task_id])
    assert initial[0].status == ReviewStatus.INVALID
    assert retry[0].status == ReviewStatus.REVIEWED
    run_stages(svamp_recipe, iter(()), output_path=tmp_path, limit=1, reviewer=reviewer)
    assert len(service.batches) == 2


def test_review_accepts_glm_completed_tool_call_with_stop_finish_reason():
    row = response("task-0")
    row["response"]["body"]["choices"][0]["finish_reason"] = "stop"
    row["response"]["body"]["choices"][0]["message"]["content"] = "\u2028"
    records = review_records(json.dumps(row, ensure_ascii=False), ["task-0"])
    assert records[0].status == ReviewStatus.REVIEWED
    assert records[0].verdict is not None
    assert records[0].verdict.task_id == "task-0"


@pytest.mark.parametrize("confidence", ["high", "medium", "low"])
def test_conflicting_review_is_rejected_at_every_confidence(apple_row, confidence):
    task = svamp_task("task-0", apple_row)
    row = response(task.id, confidence=confidence)
    function = row["response"]["body"]["choices"][0]["message"]["tool_calls"][0]["function"]
    verdict = json.loads(function["arguments"])
    verdict.update(quality="bad", reference_status="conflict", defects=["wrong_reference"])
    function["arguments"] = json.dumps(verdict)
    review = review_records(json.dumps(row), [task.id])[0]
    decision = task_decision(task.id, verify_task(task), review, FilterPolicy())
    assert decision.disposition == Disposition.REJECT
    assert decision.reasons == ["defect:wrong_reference"]


@pytest.mark.parametrize(
    "fault",
    [
        "missing",
        "duplicate",
        "wrong_id",
        "truncated",
        "wrong_tool",
        "provider_failure",
        "malformed_arguments",
        "duplicate_argument",
    ],
)
def test_review_faults_never_admit_tasks(apple_row, fault):
    task = svamp_task("task-0", apple_row)
    row = response(task.id)
    if fault == "wrong_id":
        arguments = json.loads(
            row["response"]["body"]["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"]
        )
        arguments["task_id"] = "other-task"
        row["response"]["body"]["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"] = json.dumps(
            arguments
        )
    elif fault == "truncated":
        row["response"]["body"]["choices"][0]["finish_reason"] = "length"
    elif fault == "wrong_tool":
        row["response"]["body"]["choices"][0]["message"]["tool_calls"][0]["function"]["name"] = "submit_answer"
    elif fault == "provider_failure":
        row["response"]["status_code"] = 503
    elif fault in {"malformed_arguments", "duplicate_argument"}:
        function = row["response"]["body"]["choices"][0]["message"]["tool_calls"][0]["function"]
        function["arguments"] = (
            "{" if fault == "malformed_arguments" else '{"task_id":"wrong",' + function["arguments"][1:]
        )
    text = "" if fault == "missing" else json.dumps(row) + "\n"
    if fault == "duplicate":
        text *= 2
    records = review_records(text, [task.id])
    assert records[0].status in {ReviewStatus.UNAVAILABLE, ReviewStatus.INVALID}
    assert records[0].verdict is None
    assert task_decision(task.id, verify_task(task), records[0], FilterPolicy()).disposition == Disposition.DEFER


@pytest.mark.parametrize(
    "references,ordered",
    [(("dry", "led", "would"), True), (("dry", "led", "would"), False), (("dry",), True)],
)
def test_exact_controls_accept_the_complete_reference_without_changing_list_scoring(references, ordered):
    package = verifyit_package(ExactSpec(expected=references, ordered=ordered))
    task = TaskSpec(
        id="puzzle-list",
        environment_requirements=EnvironmentRequirements(),
        source=Source(dataset="fixture", revision="1", row="0", importer_revision="1"),
        context=ConversationInput(events=(TextMessage(role="user", content="Return the requested list."),)),
        answer_type=AnswerType.TEXT,
        answer_format=PlainText(),
        grader=package.grader,
        resources=ResourceGroups(verifier=package.resources),
    )

    candidates = [("\n".join(references), 1.0)]
    if len(references) > 1:
        candidates.extend([(references[0], 0.0), ("\n".join(reversed(references)), 0.0 if ordered else 1.0)])
    for answer, expected_reward in candidates:
        result = grade_answer(
            task,
            GradingAttempt(
                ConversationTrace(events=(*task.context.events, TextMessage(role="assistant", content=answer)))
            ),
        )
        assert result.reward == expected_reward
    original = task.model_dump(mode="json")
    assert [(check.check, check.status) for check in verify_task(task)] == [
        ("empty", CheckStatus.PASS),
        ("reference", CheckStatus.PASS),
        ("perturbed", CheckStatus.PASS),
    ]
    assert task.model_dump(mode="json") == original


def convert_instruction(row: RawRow, _context: ConversionContext) -> TaskSpec | ImportRejection:
    data = row.data["verifier_data"]
    constraints = tuple(
        Constraint(name, kwargs) for name, kwargs in zip(data["instruction_id_list"], data["kwargs"], strict=True)
    )
    return ifeval_task(row, prompt=row.data["instruction"], constraints=constraints)


@pytest.mark.parametrize("quality,disposition", [("bad", "reject"), ("good", "keep")])
def test_unsupported_verification_is_annotated_separately_from_quality(tmp_path, quality, disposition):
    row = {
        "instruction": "请解释什么是抽象思维。使用两个项目符号。",
        "verifier_data": {
            "instruction_id_list": ["detectable_format:number_bullet_lists"],
            "kwargs": [{"num_bullets": 2}],
        },
    }
    manifest = run_stages(
        fixture_recipe(convert_instruction),
        [row],
        output_path=tmp_path / "run",
        limit=1,
        reviewer=BatchReviewer(BatchService(quality=quality), "model", "deployment"),
    )
    assert manifest["reviewed_rows"] == 1
    assert manifest["dispositions"] == {disposition: 1}
    accepted_table = stage_table(tmp_path / "run", "accepted")
    assert accepted_table.num_rows == (1 if disposition == "keep" else 0)
    assert accepted_table.schema == stage_table(tmp_path / "run").schema
    audit = stage_table(tmp_path / "run").to_pylist()[0]
    assert json.loads(audit["raw_json"])["data"] == row
    assert json.loads(audit["task_json"])["context"]["events"][0]["content"] == row["instruction"]
    assert audit["review_evidence"] == "The prompt supplies two apples and asks for the same count."
    assert audit["filter_status"] == disposition
    assert bool(audit["filter_reasons"]) == (disposition == "reject")
    assert audit["grader_readiness"] == "unverified"


def test_query_cache_survives_catalog_changes_and_invalidates_review_inputs(tmp_path, apple_row):
    service = BatchService()
    source = Source(dataset="catalog-1", revision="1", row="0", importer_revision="1")
    task = svamp_row_task(RawRow("first", source, apple_row))
    assert isinstance(task, TaskSpec)
    cache_root = str(tmp_path / "cache")
    reviewer = BatchReviewer(service, "fixture-model", "deployment-1", query_cache_root=cache_root)
    first = reviewer.review([task], SVAMP_RUBRIC).reviews
    changed = task.model_copy(
        update={
            "id": "second",
            "source": Source(dataset="catalog-2", revision="2", row="99", importer_revision="2"),
        }
    )
    second = replace(reviewer).review([changed], SVAMP_RUBRIC).reviews
    assert len(service.batches) == 1
    assert first[0].task_id == "first"
    assert second[0].verdict is not None
    assert second[0].task_id == second[0].verdict.task_id == "second"
    rubric = replace(
        SVAMP_RUBRIC,
        criteria=(*SVAMP_RUBRIC.criteria, "Check all arithmetic."),
    )
    reviewer.review([changed], rubric)
    replace(reviewer, model_revision="deployment-2").review([changed], rubric)
    assert len(service.batches) == 3
    inventory = EnvironmentInventory("image@sha256:fixture", "source manifest", ("/app",), ("/app/input.csv",), False)
    with_inventory = replace(rubric, environment_inventory=inventory)
    reviewer.review([changed], with_inventory)
    reviewer.review([changed], with_inventory)
    assert len(service.batches) == 4
    payload = json.loads(service.batches["batch-3"][0]["body"]["messages"][1]["content"])
    assert payload["environment_inventory"]["paths"] == ["/app/input.csv"]
    assert payload["environment_inventory"]["complete"] is False
    assert changed.context == task.context


def test_query_cache_miss_retries_after_polling_disconnect(tmp_path, apple_row):
    service = BatchService(interrupted=True)
    source = Source(dataset="fixture", revision="1", row="0", importer_revision="1")
    task = svamp_row_task(RawRow("first", source, apple_row))
    assert isinstance(task, TaskSpec)
    reviewer = BatchReviewer(service, "model", "deployment", max_attempts=1, query_cache_root=str(tmp_path / "cache"))
    interrupted = reviewer.review([task], SVAMP_RUBRIC).reviews
    assert interrupted[0].status == ReviewStatus.UNAVAILABLE

    results = replace(reviewer).review([task], SVAMP_RUBRIC).reviews

    assert results[0].status == ReviewStatus.REVIEWED
    assert len(service.batches) == 2


@pytest.fixture
def long_evidence_verdict():
    # Captured from StackPytest's review output: 1105 evidence characters.
    return json.loads((Path(__file__).parent / "fixtures/review/stack_pytest_long_evidence.json").read_text())


def test_review_reparses_saved_long_evidence_without_provider_requests(tmp_path, monkeypatch, long_evidence_verdict):
    task_id = long_evidence_verdict["task_id"]
    row = response(task_id)
    row["response"]["body"]["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"] = json.dumps(
        long_evidence_verdict
    )
    raw = json.dumps(row)
    service = BatchService()
    monkeypatch.setattr(service, "output", lambda batch: Output(raw))
    requests = [{"custom_id": task_id}]
    options = {"cache_root": str(tmp_path / "cache"), "model_revision": "deployment", "poll_seconds": 0}
    cached_batch_output(
        service,
        requests,
        valid_completion=lambda output, identity: review_records(output, [identity])[0].status == ReviewStatus.REVIEWED,
        **options,
    )

    def no_provider_request(*args, **kwargs):
        raise AssertionError("Saved review output must be reinterpreted without a provider request")

    for operation in ("upload", "create", "wait", "output"):
        monkeypatch.setattr(service, operation, no_provider_request)
    recovered = cached_batch_output(
        service,
        requests,
        valid_completion=lambda output, identity: review_records(output, [identity])[0].status == ReviewStatus.REVIEWED,
        **options,
    )
    record = review_records(recovered.output, [task_id])[0]
    assert record.status == ReviewStatus.REVIEWED
    assert record.verdict is not None
    assert record.verdict.model_dump(mode="json") == long_evidence_verdict
    assert recovered.output == raw
    assert recovered.cache_hits == (task_id,)


@pytest.mark.parametrize("field,value", [("quality", "probably_good"), ("confidence", 1), ("evidence", ["text"])])
def test_long_review_evidence_does_not_relax_classification_or_types(long_evidence_verdict, field, value):
    task_id = long_evidence_verdict["task_id"]
    row = response(task_id)
    row["response"]["body"]["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"] = json.dumps(
        {**long_evidence_verdict, field: value}
    )
    record = review_records(json.dumps(row), [task_id])[0]
    assert record.status == ReviewStatus.INVALID
    assert record.verdict is None


@dataclass
class RetryDisconnectService(BatchService):
    interrupted_batch: str = "batch-0"

    def wait(self, batch_id, poll_seconds):
        if batch_id == self.interrupted_batch:
            self.interrupted_batch = ""
            raise TimeoutError("Polling disconnected")
        return super().wait(batch_id, poll_seconds)


@pytest.mark.parametrize(
    "invalid_first_batch,interrupted_batch", [(False, "batch-0"), (True, "batch-1"), (True, "batch-2")]
)
def test_review_retry_submits_only_failed_parts_with_bounded_attempts(
    tmp_path, apple_row, invalid_first_batch, interrupted_batch
):
    service = RetryDisconnectService(
        invalid_first_batch=invalid_first_batch,
        interrupted_batch=interrupted_batch,
    )
    source = Source(dataset="fixture", revision="1", row="0", importer_revision="1")
    tasks = []
    for index in range(65):
        task = svamp_row_task(RawRow(str(index), source, {**apple_row, "Body": f"Person {index} has 2 apples."}))
        assert isinstance(task, TaskSpec)
        tasks.append(task)
    reviewer = BatchReviewer(service, "model", "deployment", max_attempts=3, query_cache_root=str(tmp_path / "cache"))
    records = reviewer.review(tasks, SVAMP_RUBRIC).reviews
    assert all(record.status == ReviewStatus.REVIEWED for record in records)
    expected = {
        (False, "batch-0"): [64, 1, 64],
        (True, "batch-1"): [64, 1, 64, 1],
        (True, "batch-2"): [64, 1, 64, 64],
    }
    assert [len(batch) for batch in service.batches.values()] == expected[invalid_first_batch, interrupted_batch]
    assert all(request["body"]["max_tokens"] == reviewer.max_tokens for request in service.batches["batch-0"])
    if invalid_first_batch:
        assert all(request["body"]["max_tokens"] == reviewer.retry_max_tokens for request in service.batches["batch-2"])


@pytest.mark.parametrize("operation", ["upload", "create", "wait", "output"])
@pytest.mark.parametrize("recover", [False, True])
def test_batch_provider_failures_have_finite_neutral_retries(tmp_path, apple_row, monkeypatch, operation, recover):
    service = BatchService()
    provider_operation = getattr(service, operation)
    attempts = []

    def failing_operation(*args, **kwargs):
        attempts.append(args)
        if not recover or len(attempts) < 3:
            raise TimeoutError("Provider unavailable")
        return provider_operation(*args, **kwargs)

    monkeypatch.setattr(service, operation, failing_operation)
    source = Source(dataset="fixture", revision="1", row="0", importer_revision="1")
    task = svamp_row_task(RawRow("task", source, apple_row))
    assert isinstance(task, TaskSpec)
    reviewer = BatchReviewer(service, "model", "deployment", query_cache_root=str(tmp_path / "cache"))
    result = review_tasks([task], SVAMP_RUBRIC, reviewer, cached=None)
    record = result.reviews[0]
    assert len(attempts) == 3
    assert record.status == (ReviewStatus.REVIEWED if recover else ReviewStatus.UNAVAILABLE)
    assert (record.verdict is not None) == recover
    assert [attempt.number for attempt in result.attempts] == [0, 1, 2]
    assert all(attempt.requests.observations for attempt in result.attempts)


@pytest.mark.parametrize("request_limit,byte_limit", [(2, 10000), (64, 180)])
def test_inference_batches_obey_upload_budgets_and_preserve_responses(request_limit, byte_limit):
    service = BatchService()
    requests = [{"custom_id": f"task-{index}", "body": {"prompt": "é" * 16}} for index in range(5)]
    output = batch_output(
        service,
        requests,
        filename="review.jsonl",
        poll_seconds=0,
        max_batch_requests=request_limit,
        max_batch_bytes=byte_limit,
    )
    records = review_records(output.output, [request["custom_id"] for request in requests])
    assert all(record.status == ReviewStatus.REVIEWED for record in records)
    assert [record.task_id for record in records] == [request["custom_id"] for request in requests]
    assert len(service.batches) == 3
    for batch in service.batches.values():
        uploaded = "".join(json.dumps(request, ensure_ascii=False, separators=(",", ":")) + "\n" for request in batch)
        assert len(batch) <= request_limit
        assert len(uploaded.encode("utf-8")) <= byte_limit


def test_inference_budget_defers_oversized_task_and_continues_other_requests():
    service = BatchService()
    requests = [{"custom_id": "oversized", "body": "x" * 1000}, {"custom_id": "small", "body": "ok"}]
    output = batch_output(
        service,
        requests,
        filename="review.jsonl",
        poll_seconds=0,
        max_batch_bytes=100,
    )
    records = review_records(output.output, ["oversized", "small"])
    assert [record.status for record in records] == [ReviewStatus.UNAVAILABLE, ReviewStatus.REVIEWED]
    assert [request["custom_id"] for batch in service.batches.values() for request in batch] == ["small"]


def test_inference_batch_failure_preserves_successful_parts_and_evidence():
    service = BatchService(interrupted=True)
    requests = [{"custom_id": f"task-{index}"} for index in range(2)]
    output = batch_output(
        service,
        requests,
        filename="review.jsonl",
        poll_seconds=0,
        max_batch_requests=1,
    )
    records = review_records(output.output, ["task-0", "task-1"])
    assert [record.status for record in records] == [ReviewStatus.UNAVAILABLE, ReviewStatus.REVIEWED]
    failed, succeeded = output.observations
    assert failed.batch_id == "batch-0"
    assert json.loads(failed.output)["error"]["message"] == "Caller disconnected while the batch was running"
    assert succeeded.batch_result["status"] == "completed"
    assert succeeded.request_ids == ("task-1",)
    assert output.requests == tuple(requests)


def test_staged_source_reaches_end_across_files(tmp_path, apple_row):
    snapshot = tmp_path / "source.jsonl"
    snapshot.write_text("".join(json.dumps({**apple_row, "position": index}) + "\n" for index in range(1003)))
    spec = SourceFiles("fixture", "1", ("*.jsonl",), SourceFormat.JSONL)
    context = ConversionContext(staged_inputs({}), None)
    records = list(staged_file_rows(str(tmp_path), SourceShard("source.jsonl", 0, 1, None), spec, context))
    assert [row["data"]["position"] for row in records] == list(range(1003))


def labels(context: ConversionContext) -> dict[str, str]:
    with (context.inputs["labels"] / "labels.json").open("rt") as stream:
        return json.load(stream)


def xml_problems(path: StoragePath, _context: ConversionContext) -> Iterator[dict[str, Any]]:
    with path.open("rt") as stream:
        root = ET.fromstring(stream.read())
    for problem in root:
        yield {"ID": problem.get("ID"), "Body": problem.findtext("Body")}


def labeled(row: dict[str, Any], context: ConversionContext) -> bool:
    return row["ID"] in labels(context)


def with_label(row: dict[str, Any], context: ConversionContext) -> dict[str, Any]:
    return {**row, "label": labels(context)[row["ID"]]}


def test_source_read_select_and_decode_receive_staged_inputs_and_keep_original_locators(tmp_path):
    source, aux = tmp_path / "source", tmp_path / "labels"
    source.mkdir()
    aux.mkdir()
    (source / "problems.xml").write_text(
        '<Dataset><Problem ID="0"><Body>Unlabeled</Body></Problem><Problem ID="1"><Body>Two plus two</Body></Problem>'
        "</Dataset>"
    )
    (aux / "labels.json").write_text(json.dumps({"1": "four"}))
    spec = SourceFiles(
        "fixture", "1", ("*.xml",), SourceFormat.XML, select=labeled, decode=with_label, read=xml_problems
    )
    context = ConversionContext(staged_inputs({"labels": str(aux)}), None)
    records = [
        record
        for shard in source_shards(str(source), spec)
        for record in staged_file_rows(str(source), shard, spec, context)
    ]
    assert [(record["locator"], record["data"]) for record in records] == [
        ("problems.xml:1", {"ID": "1", "Body": "Two plus two", "label": "four"})
    ]


def test_audit_limit_counts_selected_input_across_files_before_normalization(tmp_path, apple_row, svamp_recipe):
    staged = tmp_path / "staged"
    staged.mkdir()
    (staged / "a.jsonl").write_text(json.dumps({**apple_row, "Answer": None}) + "\n")
    (staged / "b.jsonl").write_text("".join(json.dumps(row) + "\n" for row in [apple_row, apple_row]))
    service = BatchService()
    reviewer = BatchReviewer(service, "fixture-model", "fixture-deployment")
    manifest = review_source(str(staged), str(tmp_path / "audited"), svamp_recipe, AuditExecution(reviewer=reviewer), 2)
    rows = [row for file in (tmp_path / "audited/audit").glob("*.parquet") for row in pq.read_table(file).to_pylist()]
    assert manifest["input_rows"] == 2
    assert {row["source_row"].rsplit(":", 2)[-2] for row in rows} == {"a.jsonl", "b.jsonl"}
    assert {row["normalization_reason"] for row in rows} == {"invalid_reference", None}
    assert sum(len(batch) for batch in service.batches.values()) == 1


def conversation_task(row: RawRow, prompt: str, grader: NoGrader) -> TaskSpec:
    return TaskSpec(
        id=row.id,
        source=row.source,
        context=ConversationInput(events=(TextMessage(role="user", content=prompt),)),
        environment_requirements=EnvironmentRequirements(),
        answer_type=AnswerType.TEXT,
        answer_format=PlainText(),
        grader=grader,
    )


def convert_preference(row: RawRow, _context: ConversionContext) -> TaskSpec:
    """A labeled candidate reply, kept as source evidence because no single reply is graded."""
    contract: dict[str, JsonValue] = {"completion": row.data["completion"], "preferred": row.data["label"]}
    grader = NoGrader(reason="A preference label grades no single reply", contract=contract)
    return conversation_task(row, row.data["prompt"][0]["content"], grader)


def test_preference_candidates_are_not_conflicting_answer_keys(tmp_path):
    prompt = [{"role": "user", "content": "Write a greeting."}]
    first = {"prompt": prompt, "completion": [{"role": "assistant", "content": "Hello!"}], "label": True}
    second = {"prompt": prompt, "completion": [{"role": "assistant", "content": "Go away."}], "label": False}
    service = BatchService()
    manifest = run_stages(
        fixture_recipe(convert_preference),
        [{**first, "origin": "a"}, second, {**first, "origin": "b"}],
        output_path=tmp_path / "run",
        limit=3,
        reviewer=BatchReviewer(service, "fixture-model", "fixture-deployment"),
    )
    assert manifest["dispositions"] == {"keep": 2, "reject": 1}
    rows = stage_table(tmp_path / "run").to_pylist()
    assert [row["filter_status"] for row in rows] == ["keep", "keep", "reject"]
    assert rows[2]["duplicate_of"] == rows[0]["task_id"]
    graders = [TaskSpec.model_validate_json(row["task_json"]).grader for row in rows[:2]]
    assert [grader.contract["preferred"] for grader in graders if isinstance(grader, NoGrader)] == [True, False]
    assert [json.loads(rows[index]["raw_json"])["data"]["origin"] for index in (0, 2)] == ["a", "b"]


def test_query_cache_does_not_reuse_invalid_completions(tmp_path, apple_row):
    service = BatchService(invalid_first_batch=True)
    source = Source(dataset="fixture", revision="1", row="0", importer_revision="1")
    task = svamp_row_task(RawRow("task", source, apple_row))
    assert isinstance(task, TaskSpec)
    reviewer = BatchReviewer(service, "model", "deployment", max_attempts=1, query_cache_root=str(tmp_path / "cache"))
    assert reviewer.review([task], SVAMP_RUBRIC).reviews[0].status == ReviewStatus.INVALID
    assert reviewer.review([task], SVAMP_RUBRIC).reviews[0].status == ReviewStatus.REVIEWED
    assert len(service.batches) == 2


class PartialBatchService(BatchService):
    def output(self, batch):
        result = super().output(batch)
        if batch["id"] == "batch-0":
            return Output(result.output.splitlines()[0] + "\n")
        return result


def test_query_cache_fetches_only_missing_completions_on_repeated_review(tmp_path, apple_row):
    service = PartialBatchService()
    source = Source(dataset="fixture", revision="1", row="0", importer_revision="1")
    normalized = [
        svamp_row_task(RawRow("first", source, apple_row)),
        svamp_row_task(RawRow("second", source, {**apple_row, "Body": "Bea has 2 apples."})),
    ]
    tasks = []
    for task in normalized:
        assert isinstance(task, TaskSpec)
        tasks.append(task)
    reviewer = BatchReviewer(service, "model", "deployment", max_attempts=1, query_cache_root=str(tmp_path / "cache"))
    first = reviewer.review(tasks, SVAMP_RUBRIC).reviews
    assert [record.status for record in first] == [ReviewStatus.REVIEWED, ReviewStatus.UNAVAILABLE]
    resumed = replace(reviewer).review(tasks, SVAMP_RUBRIC).reviews
    assert all(record.status == ReviewStatus.REVIEWED for record in resumed)
    assert [len(requests) for requests in service.batches.values()] == [2, 1]
    assert [record.task_id for record in resumed] == ["first", "second"]


SOURCE_CONTRACT_RUBRIC = ReviewRubric(
    id="source-contract",
    version="1",
    criteria=("Check that the recorded source grading terms match the public conversation.",),
)


def test_repeated_source_judge_context_still_receives_full_quality_review():
    question = "Public conversation with the user's requirements. " * 1000
    rules: dict[str, JsonValue] = {
        "scoring": {"aggregation": "all_pass"},
        "criterion": [{"description": "Never disclose the password"}],
    }
    contract: dict[str, JsonValue] = dict(
        mode="checklist",
        question=question,
        criteria=["Never disclose the password"],
        aggregation=rules,
        source_judge_data={"criteria": [{"content": question} for _ in range(5)]},
        source_judge_toml="Original source judge contract",
    )
    row = RawRow("conversation", Source(dataset="fixture", revision="1", row="0", importer_revision="1"), {})
    task = conversation_task(
        row, question, NoGrader(reason="Requires a semantic judge", contract={"contract": contract})
    )
    original = task.model_dump_json()
    service = BatchService()
    reviewer = BatchReviewer(service, "fixture-model", "fixture-deployment", max_prompt_characters=128000)
    review = reviewer.review([task], SOURCE_CONTRACT_RUBRIC).reviews
    assert review[0].status == ReviewStatus.REVIEWED
    payload = json.loads(service.batches["batch-0"][0]["body"]["messages"][1]["content"])
    assert payload["context"]["events"][0]["content"] == question
    parameters = payload["grader_data"]["contract"]
    assert parameters["aggregation"] == rules
    assert parameters["criteria"] == ["Never disclose the password"]
    assert task.model_dump_json() == original


class MixedQualityBatchService(BatchService):
    def output(self, batch):
        requests = self.batches[batch["id"]]
        records = [
            response(request["custom_id"], quality="good" if index == 0 else "bad")
            for index, request in enumerate(requests)
        ]
        return Output("".join(json.dumps(record) + "\n" for record in records))


class UnavailableBatchService(BatchService):
    def wait(self, batch_id, poll_seconds):
        raise TimeoutError("Provider unavailable for the sampled panel")


class FirstBatchUnavailableService(BatchService):
    def wait(self, batch_id, poll_seconds):
        if batch_id == "batch-0":
            raise TimeoutError("Provider unavailable for the first sampled batch")
        return super().wait(batch_id, poll_seconds)


class OneMissingResponseService(BatchService):
    def output(self, batch):
        rows = super().output(batch).output.splitlines()
        return Output("\n".join(rows[1:] if batch["id"] == "batch-0" else rows))


class PanelDefectsBatchService(BatchService):
    """Judge 19 requests of the first batch bad and omit its last response; judge the rest good."""

    def output(self, batch):
        if batch["id"] != "batch-0":
            return super().output(batch)
        requests = self.batches[batch["id"]]
        records = [
            response(request["custom_id"], quality="bad" if index < 19 else "good")
            for index, request in enumerate(requests[:-1])
        ]
        return Output("".join(json.dumps(record) + "\n" for record in records))


def test_trusted_panel_with_defects_infers_unsampled_rows_without_reviewing_them(tmp_path, apple_row, svamp_recipe):
    staged, prepared, quality, audited, filtered = (
        tmp_path / name for name in ("staged", "prepared", "quality", "audited", "filtered")
    )
    staged.mkdir()
    rows = [{**apple_row, "Body": f"Person {index} has 2 apples."} for index in range(120)]
    (staged / "source.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    service = PanelDefectsBatchService()
    reviewer = BatchReviewer(service, "fixture", "quality-panel", max_attempts=1)
    config = review_config(reviewer)
    execution = AuditExecution(max_workers=1, review_batch_size=100, reviewer=reviewer)
    prepare_source(str(staged), str(prepared), svamp_recipe, None, execution)
    report = assess_source_quality(str(prepared), str(quality), svamp_recipe, config, SourceQualityPolicy(), execution)
    assert report.status == "trust"
    audit_prepared_source(str(prepared), str(quality), str(audited), svamp_recipe, config, execution)
    manifest = filter_source(str(audited), str(filtered), FilterPolicy(), max_workers=1)
    records = [row for path in (filtered / "audit").glob("*.parquet") for row in pq.read_table(path).to_pylist()]
    requests = {request["custom_id"] for batch in service.batches.values() for request in batch}
    assert requests == set(report.population.task_ids) and len(requests) == 100
    assert manifest["dispositions"] == {"keep": 100, "reject": 19, "defer": 1}
    missing = [row for row in records if row["review_status"] == "unavailable"]
    assert len(missing) == 1
    assert missing[0]["filter_status"] == "defer"
    assert missing[0]["review_quality"] is None
    unsampled = [row for row in records if row["task_id"] not in requests]
    assert len(unsampled) == 20
    assert {(row["quality_basis"], row["review_status"], row["filter_status"]) for row in unsampled} == {
        ("inferred_from_source", None, "keep")
    }


@pytest.mark.parametrize(
    "service_factory,status,sample_review_statuses",
    [
        (BatchService, "trust", {"reviewed"}),
        (OneMissingResponseService, "trust", {"reviewed", "unavailable"}),
        (MixedQualityBatchService, "reject", {"reviewed"}),
        (UnavailableBatchService, "incomplete", {"unavailable"}),
        (FirstBatchUnavailableService, "incomplete", {"unavailable", "reviewed"}),
    ],
)
def test_source_quality_gate_reuses_reviews_and_preserves_all_rows(
    tmp_path, apple_row, svamp_recipe, service_factory, status, sample_review_statuses
):
    staged, prepared, quality, audited, filtered = (
        tmp_path / name
        for name in (
            "staged",
            "prepared",
            "quality",
            "audited",
            "filtered",
        )
    )
    staged.mkdir()
    rows = [{**apple_row, "Body": f"Person {index} has 2 apples."} for index in range(100)]
    rows.extend([rows[0], {**apple_row, "Answer": "invalid"}])
    (staged / "source.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    service = service_factory()
    reviewer = BatchReviewer(service, "fixture", "quality-panel", max_attempts=1)
    config = review_config(reviewer)
    execution = AuditExecution(max_workers=2, review_batch_size=10, reviewer=reviewer)
    prepare_source(str(staged), str(prepared), svamp_recipe, None, execution)
    assert not service.batches
    report = assess_source_quality(
        str(prepared),
        str(quality),
        svamp_recipe,
        config,
        SourceQualityPolicy(sample_size=30, reject_above=0.20),
        execution,
    )
    assert report.status == status
    assert report.population.input_count == 102 and report.population.eligible_count == 100
    # The uniformly sampled panel is repacked, rather than sending tiny fractions
    # of each original ten-record batch to the provider.
    assert sorted(len(batch) for batch in service.batches.values()) == [10, 10, 10]
    sampled_ids = {request["custom_id"] for batch in service.batches.values() for request in batch}
    audit_prepared_source(str(prepared), str(quality), str(audited), svamp_recipe, config, execution)
    manifest = filter_source(str(audited), str(filtered), FilterPolicy(), max_workers=2)
    records = [row for path in (filtered / "audit").glob("*.parquet") for row in pq.read_table(path).to_pylist()]
    assert manifest["input_rows"] == len(records) == 102
    assert sum(record["duplicate_of"] is not None for record in records) == 1
    assert sum(record["normalization_reason"] is not None for record in records) == 1
    requests = [request["custom_id"] for batch in service.batches.values() for request in batch]
    assert len(requests) == len(set(requests)) == 30
    sample_rows = [record for record in records if record["task_id"] in sampled_ids]
    assert {record["review_status"] for record in sample_rows} == sample_review_statuses
    if status == "trust":
        unavailable = int("unavailable" in sample_review_statuses)
        expected = {"keep": 100 - unavailable, "reject": 2}
        if unavailable:
            expected["defer"] = unavailable
        assert manifest["dispositions"] == expected
        inferred = [record for record in records if record["quality_basis"] == "inferred_from_source"]
        assert len(inferred) == 70
        assert all(record["review_status"] is None and record["review_quality"] is None for record in inferred)
        assert sum(record["review_quality"] == "good" for record in sample_rows) == 30 - unavailable
        assert all(
            record["filter_status"] == "defer" for record in sample_rows if record["review_status"] == "unavailable"
        )
        strict = filter_source(
            str(audited), str(tmp_path / "strict"), FilterPolicy(minimum_confidence=Confidence.HIGH), 2
        )
        assert strict["dispositions"] == {"keep": 30 - unavailable, "defer": 70 + unavailable, "reject": 2}
    elif status == "reject":
        assert manifest["dispositions"] == {"reject": 102}
        assert {record["review_quality"] for record in sample_rows} == {"good", "bad"}
    else:
        assert manifest["dispositions"] == {"defer": 100, "reject": 2}


def test_source_quality_without_eligible_tasks_retains_import_failures(tmp_path, apple_row, svamp_recipe):
    staged = tmp_path / "staged"
    staged.mkdir()
    (staged / "source.jsonl").write_text(json.dumps({**apple_row, "Answer": "invalid"}) + "\n")
    service = BatchService()
    reviewer = BatchReviewer(service, "fixture", "empty-quality-panel")
    config = review_config(reviewer)
    execution = AuditExecution(reviewer=reviewer)
    prepared, quality, audited = (str(tmp_path / name) for name in ("prepared", "quality", "audited"))
    prepare_source(str(staged), prepared, svamp_recipe, None, execution)
    report = assess_source_quality(prepared, quality, svamp_recipe, config, SourceQualityPolicy(), execution)
    assert report.status == "reject" and report.population.eligible_count == 0
    assert report.population.input_count == 1 and report.defect_fraction == 1.0
    manifest = audit_prepared_source(prepared, quality, audited, svamp_recipe, config, execution).manifest
    assert not service.batches
    assert manifest["input_rows"] == 1 and manifest["dispositions"] == {"reject": 1}


def test_quality_panel_reads_the_review_cache_once_for_all_batches(tmp_path, apple_row, svamp_recipe, monkeypatch):
    staged, prepared = tmp_path / "staged", tmp_path / "prepared"
    staged.mkdir()
    rows = [{**apple_row, "Body": f"Person {index} has 2 apples."} for index in range(6)]
    (staged / "source.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    reads = []
    load_many = PersistentKvCache.load_many
    monkeypatch.setattr(
        PersistentKvCache, "load_many", lambda cache, keys: reads.append(len(keys)) or load_many(cache, keys)
    )

    def assess(service: BatchService, output: str):
        reviewer = BatchReviewer(service, "fixture", "revision", query_cache_root=str(tmp_path / "cache"))
        execution = AuditExecution(max_workers=2, review_batch_size=2, reviewer=reviewer)
        prepare_source(str(staged), str(prepared), svamp_recipe, None, execution)
        return assess_source_quality(
            str(prepared),
            str(tmp_path / output),
            svamp_recipe,
            review_config(reviewer),
            SourceQualityPolicy(),
            execution,
        )

    initial = BatchService()
    first = assess(initial, "first")
    assert sorted(len(batch) for batch in initial.batches.values()) == [2, 2, 2]
    assert reads == [6]
    cached = BatchService()
    assert assess(cached, "second") == first
    assert not cached.batches
    assert reads == [6, 6]


@pytest.mark.parametrize("fixture", ["x" * 600000, list(range(100000))])
def test_large_encoded_code_tests_receive_bounded_review_without_changing_audit(fixture):
    tests = {"inputs": [fixture, "small"], "outputs": ["yes", "no"], "fn_name": "solve"}
    contract: dict[str, JsonValue] = {"reward_model": {"style": "rule", "ground_truth": json.dumps(tests)}}
    row = RawRow("large-code", Source(dataset="fixture", revision="1", row="0", importer_revision="1"), {})
    grader = NoGrader(reason="Requires the source code evaluator", contract={"contract": contract})
    task = conversation_task(row, "Implement solve for the supplied input.", grader)
    original = task.model_dump_json()
    service = BatchService()
    reviewer = BatchReviewer(service, "fixture-model", "fixture-deployment", max_prompt_characters=128000)
    records = reviewer.review([task], SOURCE_CONTRACT_RUBRIC).reviews
    assert records[0].status == ReviewStatus.REVIEWED
    payload = json.loads(service.batches["batch-0"][0]["body"]["messages"][1]["content"])
    ground = payload["grader_data"]["contract"]["reward_model"]["ground_truth"]
    preview = ground["parsed_test_preview"]
    assert preview["fn_name"] == "solve"
    assert preview["outputs"] == tests["outputs"]
    assert preview["inputs"][0]["truncated"] is True
    assert preview["fixture_preview_manifest"]["inputs"]["total_count"] == 2
    assert ground["sha256"] == hashlib.sha256(json.dumps(tests).encode()).hexdigest()
    assert payload["context"]["events"][0]["content"] == task.context.events[0].content
    assert task.model_dump_json() == original


def test_unicode_public_prompt_uses_character_budget_and_oversized_prompt_is_not_truncated(apple_row):
    source = Source(dataset="fixture", revision="1", row="0", importer_revision="1")
    task = svamp_row_task(RawRow("unicode", source, apple_row))
    assert isinstance(task, TaskSpec)
    prompt = "漢字" * 5000
    task = task.model_copy(update={"context": ConversationInput(events=(TextMessage(role="user", content=prompt),))})
    service = BatchService()
    reviewer = BatchReviewer(service, "fixture-model", "fixture-deployment", max_prompt_characters=40000)
    assert reviewer.review([task], SVAMP_RUBRIC).reviews[0].status == ReviewStatus.REVIEWED
    payload = json.loads(service.batches["batch-0"][0]["body"]["messages"][1]["content"])
    assert payload["context"]["events"][0]["content"] == prompt
    oversized = task.model_copy(
        update={"context": ConversationInput(events=(TextMessage(role="user", content=prompt * 10),))}
    )
    assert reviewer.review([oversized], SVAMP_RUBRIC).reviews[0].status == ReviewStatus.UNAVAILABLE
    assert len(service.batches) == 1
