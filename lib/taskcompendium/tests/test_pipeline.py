# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Curation contracts exercised through persisted outputs and a fake batch API."""

import hashlib
import io
import json
import tarfile
import threading
from dataclasses import dataclass, field, replace
from functools import partial
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from fray.types import ResourceConfig
from pydantic import JsonValue
from rigging.filesystem.storage_path import StoragePath
from verifyit.spec import ExactSpec, MathSpec, McqSpec

from taskcompendium.datasets import code_contracts, gpqa, instruction_following, preference_tasks, rubric_tasks
from taskcompendium.datasets.direct_contracts import source_contract_package
from taskcompendium.datasets.math_answers import asdiv_rows, math_controls, normalize_numina_math
from taskcompendium.datasets.numeric_answers import normalize_aime24, normalize_svamp, svamp_policy
from taskcompendium.datasets.source_definitions import tasktrove_files
from taskcompendium.grader import grader_config, grader_package, native_command_package
from taskcompendium.grading import grade_answer
from taskcompendium.grading_contract import GradingAttempt
from taskcompendium.models import (
    AnswerType,
    ConversationInput,
    ConversationTrace,
    EnvironmentRequirements,
    ResourceGroups,
    Source,
    TaskSpec,
    TextMessage,
)
from taskcompendium.native_grader import NativeCommandSpec
from taskcompendium.pipeline.audit_schema import TASK_SCHEMA, audit_columns
from taskcompendium.pipeline.filtering import task_decision
from taskcompendium.pipeline.inputs import RecipeInputs, SourceFiles, SourceFormat
from taskcompendium.pipeline.models import (
    CheckStatus,
    Confidence,
    DatasetRecipe,
    Decision,
    Disposition,
    EnvironmentInventory,
    FilterPolicy,
    HFSource,
    ImportFailureKind,
    ImportRejection,
    IntendedUse,
    Quality,
    RawRow,
    ReferenceStatus,
    ReviewRecord,
    ReviewStatus,
    ReviewVerdict,
    TaskAudit,
)
from taskcompendium.pipeline.query_cache import cached_batch_output
from taskcompendium.pipeline.recorded_review import RecordedReviewer
from taskcompendium.pipeline.review import BatchReviewer, review_records
from taskcompendium.pipeline.review_transport import batch_output
from taskcompendium.pipeline.source_quality import SourceQualityPolicy
from taskcompendium.pipeline.sources import staged_file_rows
from taskcompendium.pipeline.stages import (
    AuditExecution,
    ReviewConfig,
    ReviewTransport,
    assess_source_quality,
    audit_prepared_source,
    audit_source,
    canonicalize_sources,
    filter_source,
    prepare_source,
)
from taskcompendium.pipeline.verification import verify_task, verify_witness
from taskcompendium.runtime.resources import inline_resource, resource_bytes
from taskcompendium.runtime.task_grading import grade_task
from taskcompendium.submission import PlainText

from .pipeline_stages import fixture_recipe, run_stages, stage_table


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
    return DatasetRecipe(
        name="svamp-fixture",
        version="1",
        source=HFSource("fixture/svamp", "1", "default", "train"),
        policy=svamp_policy(),
        intended_use=IntendedUse.TRAIN,
        inputs=RecipeInputs(SourceFiles(("*.jsonl",), SourceFormat.JSONL), ()),
    )


@pytest.fixture
def apple_row():
    return {
        "Body": "Aya has 2 apples.\u2028They belong to Aya.",
        "Question": "How many apples does Aya have?",
        "Answer": "2",
        "Equation": "2",
    }


def normalize_mixed_import(row: RawRow) -> TaskSpec | ImportRejection:
    if "conversion_rejection" in row.data:
        return ImportRejection.model_validate(row.data["conversion_rejection"])
    if row.data.get("invalid_converted_task"):
        return TaskSpec.model_validate({"id": row.id})
    return normalize_svamp(row)


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
    recipe = replace(svamp_recipe, policy=replace(svamp_recipe.policy, normalize=normalize_mixed_import))
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
    audit_source(
        f"{root}/staged",
        f"{root}/audited",
        svamp_recipe,
        ReviewConfig(
            reviewer.model,
            reviewer.model_revision,
            reviewer.max_prompt_characters,
            reviewer.max_tokens,
            transport=ReviewTransport.PROVIDER_BATCH,
        ),
        AuditExecution(max_workers=2, review_batch_size=1, reviewer=reviewer),
        SourceFiles(("source.jsonl",), SourceFormat.JSONL),
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
    config = ReviewConfig(
        reviewer.model,
        reviewer.model_revision,
        reviewer.max_prompt_characters,
        reviewer.max_tokens,
        transport=ReviewTransport.PROVIDER_BATCH,
    )
    audit_source(
        str(staged),
        str(tmp_path / "audited"),
        svamp_recipe,
        config,
        AuditExecution(max_workers=1, reviewer=reviewer),
        SourceFiles(("source.jsonl",), SourceFormat.JSONL),
        len(rows),
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
    audit_source(
        str(staged),
        str(tmp_path / "audited"),
        svamp_recipe,
        config,
        AuditExecution(max_workers=3, reviewer=BatchReviewer(resumed, "fixture", "revision")),
        SourceFiles(("source.jsonl",), SourceFormat.JSONL),
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
    manifest = audit_source(
        str(staged),
        str(tmp_path / "audited"),
        svamp_recipe,
        ReviewConfig(reviewer.model, reviewer.model_revision, transport=ReviewTransport.PROVIDER_BATCH),
        AuditExecution(
            max_workers=1,
            review_batch_size=1,
            reviewer=reviewer,
            worker_resources=ResourceConfig.with_cpu(cpu=1, ram="2g"),
            review_task_resources=ResourceConfig.with_cpu(cpu=0.5, ram="1g"),
        ),
        SourceFiles(("source.jsonl",), SourceFormat.JSONL),
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
    review_path = next((tmp_path / "audited/evidence").glob("*/attempt-*/review"))
    initial = review_records((review_path / "raw-output.jsonl").read_text(), [task_id])
    retry = review_records((review_path / "retry-1/raw-output.jsonl").read_text(), [task_id])
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
    task = normalize_svamp(
        RawRow("task-0", Source(dataset="fixture", revision="1", row="0", importer_revision="1"), apple_row)
    )
    assert isinstance(task, TaskSpec)
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
    raw = RawRow("task-0", Source(dataset="fixture", revision="1", row="0", importer_revision="1"), apple_row)
    task = normalize_svamp(raw)
    assert isinstance(task, TaskSpec)
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
    "normalize,data,expected,private_field",
    [
        (
            normalize_svamp,
            {"Body": "Aya has 2 apples.", "Question": "How many apples?", "Answer": "2", "Equation": "private"},
            "2",
            "Equation",
        ),
        (
            normalize_svamp,
            {
                "Body": "Aya has 9007199254740993 apples.",
                "Question": "How many apples?",
                "Answer": "9007199254740993",
                "Equation": "private",
            },
            "9007199254740993",
            "Equation",
        ),
        (
            normalize_aime24,
            {"problem": "Find 7 + 5.", "answer": "012", "solution": "private"},
            "012",
            "solution",
        ),
        (
            gpqa.normalize,
            {
                "Question": "Which option is correct?",
                "Correct Answer": "right",
                "Incorrect Answer 1": "wrong1",
                "Incorrect Answer 2": "wrong2",
                "Incorrect Answer 3": "wrong3",
                "Explanation": "private",
            },
            None,
            "Explanation",
        ),
    ],
)
def test_recipes_normalize_source_contract_and_keep_supervision_private(normalize, data, expected, private_field):
    raw = RawRow("task-0", Source(dataset="fixture", revision="1", row="0", importer_revision="1"), data)
    task = normalize(raw)
    assert isinstance(task, TaskSpec)
    assert all(result.status.value == "pass" for result in verify_task(task))
    message = task.context.events[0]
    assert isinstance(message, TextMessage)
    prompt = message.content
    assert private_field not in prompt and "private" not in prompt
    parameters = json.loads(task.verifier.parameters_json)
    if expected is not None:
        assert parameters["expected"] == expected
        convention = PlainText(id="plain")
        for answer, reward in ((expected, 1.0), (str(int(expected) - 1), 0.0)):
            conversation = ConversationTrace(
                events=(*task.context.events, TextMessage(role="assistant", content=answer))
            )
            assert grade_task(task, convention, conversation).reward == reward
    else:
        option = f"{parameters['expected']}. right"
        assert option in prompt
        assert "A. " in prompt and "D. " in prompt
        assert normalize(raw) == task


@pytest.mark.parametrize(
    "references,ordered",
    [(("dry", "led", "would"), True), (("dry", "led", "would"), False), (("dry",), True)],
)
def test_exact_controls_accept_the_complete_reference_without_changing_list_scoring(references, ordered):
    package = grader_package(ExactSpec(expected=references, ordered=ordered))
    task = TaskSpec(
        id="puzzle-list",
        environment_requirements=EnvironmentRequirements(),
        source=Source(dataset="fixture", revision="1", row="0", importer_revision="1"),
        context=ConversationInput(events=(TextMessage(role="user", content="Return the requested list."),)),
        answer_type=AnswerType.TEXT,
        verifier=package.verifier,
        resources=ResourceGroups(verifier=package.resources),
    )
    convention = PlainText(id="plain")

    candidates = [("\n".join(references), 1.0)]
    if len(references) > 1:
        candidates.extend([(references[0], 0.0), ("\n".join(reversed(references)), 0.0 if ordered else 1.0)])
    for answer, expected_reward in candidates:
        result = grade_answer(
            task,
            convention,
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


def test_gpqa_rejects_repeated_options_instead_of_choosing_a_key():
    row = RawRow(
        "task",
        Source(dataset="fixture", revision="1", row="0", importer_revision="1"),
        {
            "Question": "Which?",
            "Correct Answer": "same",
            "Incorrect Answer 1": "same ",
            "Incorrect Answer 2": "other",
            "Incorrect Answer 3": "third",
        },
    )
    result = gpqa.normalize(row)
    assert isinstance(result, ImportRejection)
    assert result.reason == "duplicate_options"


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
        fixture_recipe(instruction_following.policy()),
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


def test_explicit_language_conflict_is_rejected_even_when_grader_and_model_pass():
    row = RawRow(
        "language-conflict",
        Source(dataset="fixture", revision="1", row="0", importer_revision="1"),
        {
            "instruction": (
                "Explain empty PostgreSQL strings. Your ENTIRE response should be in Korean language, "
                "no other language is allowed. The last word of your response should be the word charity."
            ),
            "verifier_data": {
                "instruction_id_list": ["language:response_language", "last_word:last_word_answer"],
                "kwargs": [{"language": "ko"}, {"last_word": "charity"}],
            },
        },
    )
    task = instruction_following.normalize(row)
    assert isinstance(task, TaskSpec)
    witness = "빈 문자열을 확인하려면 빈 문자열과 비교하는 조건을 사용하세요 charity"
    formal = verify_witness(task, witness, "hello charity")
    assert all(check.status == CheckStatus.PASS for check in formal)
    report = instruction_following.verification_report(task)
    assert report.checks[0].status == CheckStatus.FAIL
    review = ReviewRecord(
        task_id=task.id,
        status=ReviewStatus.REVIEWED,
        detail="",
        verdict=ReviewVerdict(
            task_id=task.id,
            quality=Quality.GOOD,
            confidence=Confidence.HIGH,
            reference_status=ReferenceStatus.CONSISTENT,
            defects=[],
            evidence="The checker and format appear compatible.",
        ),
    )
    assert task_decision(task.id, report.checks, review, FilterPolicy()).disposition == Disposition.REJECT
    # A question in another language does not itself make a mixed answer contradictory.
    allowed = task.model_copy(
        update={
            "context": task.context.model_copy(
                update={
                    "events": (
                        task.context.events[0].model_copy(
                            update={"content": "请解释如何检查空字符串。The last word must be charity."}
                        ),
                    )
                }
            )
        }
    )
    assert not any(
        check.status == CheckStatus.FAIL for check in instruction_following.verification_report(allowed).checks
    )


def test_query_cache_survives_catalog_changes_and_invalidates_review_inputs(tmp_path, apple_row, svamp_recipe):
    service = BatchService()
    source = Source(dataset="catalog-1", revision="1", row="0", importer_revision="1")
    task = svamp_recipe.policy.normalize(RawRow("first", source, apple_row))
    assert isinstance(task, TaskSpec)
    cache_root = str(tmp_path / "cache")
    reviewer = BatchReviewer(service, "fixture-model", "deployment-1", query_cache_root=cache_root)
    first = reviewer.review([task], svamp_recipe.policy.rubric, tmp_path / "first")
    changed = task.model_copy(
        update={
            "id": "second",
            "source": Source(dataset="catalog-2", revision="2", row="99", importer_revision="2"),
        }
    )
    second = replace(reviewer).review([changed], svamp_recipe.policy.rubric, tmp_path / "second")
    assert len(service.batches) == 1
    assert first[0].task_id == "first"
    assert second[0].verdict is not None
    assert second[0].task_id == second[0].verdict.task_id == "second"
    assert list((tmp_path / "second/query-cache").glob("*.json"))
    rubric = replace(
        svamp_recipe.policy.rubric,
        criteria=(*svamp_recipe.policy.rubric.criteria, "Check all arithmetic."),
    )
    reviewer.review([changed], rubric, tmp_path / "rubric")
    replace(reviewer, model_revision="deployment-2").review([changed], rubric, tmp_path / "rubric")
    assert len(service.batches) == 3
    inventory = EnvironmentInventory("image@sha256:fixture", "source manifest", ("/app",), ("/app/input.csv",), False)
    with_inventory = replace(rubric, environment_inventory=inventory)
    reviewer.review([changed], with_inventory, tmp_path / "inventory")
    reviewer.review([changed], with_inventory, tmp_path / "inventory-again")
    assert len(service.batches) == 4
    payload = json.loads(service.batches["batch-3"][0]["body"]["messages"][1]["content"])
    assert payload["environment_inventory"]["paths"] == ["/app/input.csv"]
    assert payload["environment_inventory"]["complete"] is False
    assert changed.context == task.context


def test_query_cache_miss_retries_after_polling_disconnect(tmp_path, apple_row, svamp_recipe):
    service = BatchService(interrupted=True)
    source = Source(dataset="fixture", revision="1", row="0", importer_revision="1")
    task = svamp_recipe.policy.normalize(RawRow("first", source, apple_row))
    assert isinstance(task, TaskSpec)
    reviewer = BatchReviewer(service, "model", "deployment", max_attempts=1, query_cache_root=str(tmp_path / "cache"))
    interrupted = reviewer.review([task], svamp_recipe.policy.rubric, tmp_path / "interrupted")
    assert interrupted[0].status == ReviewStatus.UNAVAILABLE

    results = replace(reviewer).review([task], svamp_recipe.policy.rubric, tmp_path / "replacement-worker")

    assert results[0].status == ReviewStatus.REVIEWED
    assert len(service.batches) == 2
    evidence = list((tmp_path / "replacement-worker").rglob("batch-submission.json"))
    assert json.loads(evidence[0].read_text())["batch_id"] == "batch-1"


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
        tmp_path / "original",
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
        tmp_path / "recovered",
        valid_completion=lambda output, identity: review_records(output, [identity])[0].status == ReviewStatus.REVIEWED,
        **options,
    )
    record = review_records(recovered, [task_id])[0]
    assert record.status == ReviewStatus.REVIEWED
    assert record.verdict is not None
    assert record.verdict.model_dump(mode="json") == long_evidence_verdict
    assert recovered == raw
    cached = next((tmp_path / "recovered/query-cache").glob("*.json"))
    assert json.loads(cached.read_text())["raw_output"] == raw


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
    tmp_path, apple_row, svamp_recipe, invalid_first_batch, interrupted_batch
):
    service = RetryDisconnectService(
        invalid_first_batch=invalid_first_batch,
        interrupted_batch=interrupted_batch,
    )
    source = Source(dataset="fixture", revision="1", row="0", importer_revision="1")
    tasks = []
    for index in range(65):
        task = svamp_recipe.policy.normalize(
            RawRow(str(index), source, {**apple_row, "Body": f"Person {index} has 2 apples."})
        )
        assert isinstance(task, TaskSpec)
        tasks.append(task)
    reviewer = BatchReviewer(service, "model", "deployment", max_attempts=3, query_cache_root=str(tmp_path / "cache"))
    records = reviewer.review(tasks, svamp_recipe.policy.rubric, tmp_path / "review")
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
def test_batch_provider_failures_have_finite_neutral_retries(
    tmp_path, apple_row, svamp_recipe, monkeypatch, operation, recover
):
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
    task = svamp_recipe.policy.normalize(RawRow("task", source, apple_row))
    assert isinstance(task, TaskSpec)
    reviewer = BatchReviewer(service, "model", "deployment", query_cache_root=str(tmp_path / "cache"))
    record = reviewer.review([task], svamp_recipe.policy.rubric, tmp_path / "review")[0]
    assert len(attempts) == 3
    assert record.status == (ReviewStatus.REVIEWED if recover else ReviewStatus.UNAVAILABLE)
    assert (record.verdict is not None) == recover
    assert len(list((tmp_path / "review").rglob("requests.jsonl"))) == 3


@pytest.mark.parametrize("request_limit,byte_limit", [(2, 10000), (64, 180)])
def test_inference_batches_obey_upload_budgets_and_preserve_responses(tmp_path, request_limit, byte_limit):
    service = BatchService()
    requests = [{"custom_id": f"task-{index}", "body": {"prompt": "é" * 16}} for index in range(5)]
    output = batch_output(
        service,
        requests,
        tmp_path,
        filename="review.jsonl",
        poll_seconds=0,
        max_batch_requests=request_limit,
        max_batch_bytes=byte_limit,
    )
    records = review_records(output, [request["custom_id"] for request in requests])
    assert all(record.status == ReviewStatus.REVIEWED for record in records)
    assert [record.task_id for record in records] == [request["custom_id"] for request in requests]
    assert len(service.batches) == 3
    for batch in service.batches.values():
        uploaded = "".join(json.dumps(request, ensure_ascii=False, separators=(",", ":")) + "\n" for request in batch)
        assert len(batch) <= request_limit
        assert len(uploaded.encode("utf-8")) <= byte_limit


def test_inference_budget_defers_oversized_task_and_continues_other_requests(tmp_path):
    service = BatchService()
    requests = [{"custom_id": "oversized", "body": "x" * 1000}, {"custom_id": "small", "body": "ok"}]
    output = batch_output(
        service,
        requests,
        tmp_path,
        filename="review.jsonl",
        poll_seconds=0,
        max_batch_bytes=100,
    )
    records = review_records(output, ["oversized", "small"])
    assert [record.status for record in records] == [ReviewStatus.UNAVAILABLE, ReviewStatus.REVIEWED]
    assert [request["custom_id"] for batch in service.batches.values() for request in batch] == ["small"]


def test_inference_batch_failure_preserves_successful_parts_and_evidence(tmp_path):
    service = BatchService(interrupted=True)
    requests = [{"custom_id": f"task-{index}"} for index in range(2)]
    output = batch_output(
        service,
        requests,
        tmp_path,
        filename="review.jsonl",
        poll_seconds=0,
        max_batch_requests=1,
    )
    records = review_records(output, ["task-0", "task-1"])
    assert [record.status for record in records] == [ReviewStatus.UNAVAILABLE, ReviewStatus.REVIEWED]
    assert json.loads((tmp_path / "part-00000/batch-submission.json").read_text())["batch_id"] == "batch-0"
    assert (tmp_path / "part-00000/raw-output.jsonl").read_text() == output.splitlines()[0]
    assert json.loads((tmp_path / "part-00001/batch-result.json").read_text())["status"] == "completed"
    assert json.loads((tmp_path / "part-00001/requests.jsonl").read_text()) == requests[1]


def test_staged_source_reaches_end_across_files(tmp_path, apple_row):
    snapshot = tmp_path / "source.jsonl"
    snapshot.write_text("".join(json.dumps({**apple_row, "position": index}) + "\n" for index in range(1003)))
    records = list(staged_file_rows(str(tmp_path), "source.jsonl", SourceFiles(("*.jsonl",), SourceFormat.JSONL)))
    assert [row["data"]["position"] for row in records] == list(range(1003))


def test_family_source_hooks_preserve_tasktrove_archive_and_asdiv_xml_records(tmp_path):
    archive_bytes = io.BytesIO()
    with tarfile.open(fileobj=archive_bytes, mode="w") as archive:
        for name, content in {
            "instruction.md": b"Solve the task",
            "tests/verifier_data.json": b'{"answer": "42"}',
        }.items():
            info = tarfile.TarInfo(name)
            info.size = len(content)
            archive.addfile(info, io.BytesIO(content))
    config_dir = tmp_path / "sample"
    config_dir.mkdir()
    pq.write_table(
        pa.table({"path": ["sample/task-1"], "task_binary": [archive_bytes.getvalue()]}), config_dir / "tasks.parquet"
    )
    tasktrove_row = next(staged_file_rows(str(tmp_path), "sample/tasks.parquet", tasktrove_files("sample")))
    assert tasktrove_row["data"]["instruction"] == "Solve the task"
    assert tasktrove_row["data"]["verifier_data"] == {"answer": "42"}
    assert tasktrove_row["data"]["archive_sha256"] == hashlib.sha256(archive_bytes.getvalue()).hexdigest()

    (tmp_path / "ASDiv.xml").write_text(
        '<Dataset><Problem ID="1"><Body>Two plus two</Body><Question>How many?</Question>'
        "<Answer>4 (things)</Answer></Problem></Dataset>"
    )
    asdiv_row = next(
        staged_file_rows(str(tmp_path), "ASDiv.xml", SourceFiles(("ASDiv.xml",), SourceFormat.XML, reader=asdiv_rows))
    )
    assert asdiv_row["data"] == {"ID": "1", "Body": "Two plus two", "Question": "How many?", "Answer": "4 (things)"}


def test_audit_limit_counts_selected_input_across_files_before_normalization(tmp_path, apple_row, svamp_recipe):
    staged = tmp_path / "staged"
    staged.mkdir()
    (staged / "a.jsonl").write_text(json.dumps({**apple_row, "Answer": None}) + "\n")
    (staged / "b.jsonl").write_text("".join(json.dumps(row) + "\n" for row in [apple_row, apple_row]))
    service = BatchService()
    reviewer = BatchReviewer(service, "fixture-model", "fixture-deployment")
    manifest = audit_source(
        str(staged),
        str(tmp_path / "audited"),
        svamp_recipe,
        ReviewConfig(
            reviewer.model,
            reviewer.model_revision,
            reviewer.max_prompt_characters,
            reviewer.max_tokens,
            transport=ReviewTransport.PROVIDER_BATCH,
        ),
        AuditExecution(reviewer=reviewer),
        SourceFiles(("*.jsonl",), SourceFormat.JSONL),
        2,
    )
    rows = [row for file in (tmp_path / "audited/audit").glob("*.parquet") for row in pq.read_table(file).to_pylist()]
    assert manifest["input_rows"] == 2
    assert {row["source_row"].rsplit(":", 2)[-2] for row in rows} == {"a.jsonl", "b.jsonl"}
    assert {row["normalization_reason"] for row in rows} == {"invalid_reference", None}
    assert sum(len(batch) for batch in service.batches.values()) == 1


def test_preference_candidates_are_not_conflicting_answer_keys(tmp_path):
    prompt = [{"role": "user", "content": "Write a greeting."}]
    first = {"prompt": prompt, "completion": [{"role": "assistant", "content": "Hello!"}], "label": True}
    second = {"prompt": prompt, "completion": [{"role": "assistant", "content": "Go away."}], "label": False}
    recipe = fixture_recipe(preference_tasks.binary_policy(preference_tasks.KTO_MIX_RUBRIC))
    service = BatchService()
    manifest = run_stages(
        recipe,
        [{**first, "origin": "a"}, second, {**first, "origin": "b"}],
        output_path=tmp_path / "run",
        limit=3,
        reviewer=BatchReviewer(service, "fixture-model", "fixture-deployment"),
    )
    assert manifest["dispositions"] == {"keep": 2, "reject": 1}
    rows = stage_table(tmp_path / "run").to_pylist()
    assert [row["filter_status"] for row in rows] == ["keep", "keep", "reject"]
    assert rows[2]["duplicate_of"] == rows[0]["task_id"]
    evidence = [grader_config(TaskSpec.model_validate_json(row["task_json"]))["contract"] for row in rows[:2]]
    assert [item["preferred"] for item in evidence] == [True, False]
    assert [json.loads(rows[index]["raw_json"])["data"]["origin"] for index in (0, 2)] == ["a", "b"]


def test_canonical_merge_keeps_evidence_and_separates_evaluation_overlap(tmp_path, apple_row, svamp_recipe):
    specifications = (
        ("a", "duplicate", "5", IntendedUse.TRAIN, Disposition.KEEP),
        ("b", "duplicate", "5", IntendedUse.TRAIN, Disposition.KEEP),
        ("c", "conflict", "1", IntendedUse.TRAIN, Disposition.KEEP),
        ("d", "conflict", "2", IntendedUse.TRAIN, Disposition.KEEP),
        ("e", "benchmark", "5", IntendedUse.TRAIN, Disposition.KEEP),
        ("f", "benchmark", "5", IntendedUse.EVAL, Disposition.KEEP),
        ("g", "reviewed", "1", IntendedUse.TRAIN, Disposition.REJECT),
        ("h", "reviewed", "5", IntendedUse.TRAIN, Disposition.KEEP),
    )
    rows = []
    for name, prompt, answer, use, disposition in specifications:
        source = Source(dataset=name, revision="a" * 40, row="0", importer_revision="1")
        task = svamp_recipe.policy.normalize(RawRow(name, source, {**apple_row, "Body": prompt, "Answer": answer}))
        assert isinstance(task, TaskSpec)
        audit = TaskAudit(
            task_id=name,
            source=source,
            raw={"evidence": name},
            normalized=task,
            normalization_rejection=None,
            checks=[],
            review=None,
            intended_use=use,
            decision=Decision(
                task_id=name,
                disposition=disposition,
                reasons=[] if disposition == Disposition.KEEP else ["bad_reference"],
            ),
        )
        rows.append(audit_columns(audit))
    merged = tmp_path / "merged"
    (merged / "data").mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist(rows, schema=TASK_SCHEMA), merged / "data/part-0.parquet")
    (merged / "manifest.json").write_text(json.dumps({"input_rows": len(rows)}))
    output = tmp_path / "canonical"
    manifest = canonicalize_sources(str(merged), str(output), max_workers=1)
    audited = {
        row["task_id"]: row for file in (output / "audit").glob("*.parquet") for row in pq.read_table(file).to_pylist()
    }
    assert manifest["input_rows"] == 8
    assert audited["b"]["duplicate_of"] == "a"
    assert (
        audited["c"]["filter_reasons"]
        == audited["d"]["filter_reasons"]
        == ["cross_source_conflicting_verifier_contracts"]
    )
    assert audited["e"]["filter_reasons"] == ["evaluation_overlap"]
    assert audited["g"]["filter_reasons"] == ["bad_reference"]
    assert {row["task_id"] for row in audited.values() if row["filter_status"] == "keep"} == {"a", "f", "h"}
    assert {
        row["task_id"] for file in (output / "train").glob("*.parquet") for row in pq.read_table(file).to_pylist()
    } == {"a", "h"}
    assert {
        row["task_id"] for file in (output / "eval").glob("*.parquet") for row in pq.read_table(file).to_pylist()
    } == {"f"}
    assert {name: json.loads(row["raw_json"]) for name, row in audited.items()} == {
        item[0]: {"evidence": item[0]} for item in specifications
    }


def test_query_cache_does_not_reuse_invalid_completions(tmp_path, apple_row, svamp_recipe):
    service = BatchService(invalid_first_batch=True)
    source = Source(dataset="fixture", revision="1", row="0", importer_revision="1")
    task = svamp_recipe.policy.normalize(RawRow("task", source, apple_row))
    assert isinstance(task, TaskSpec)
    reviewer = BatchReviewer(service, "model", "deployment", max_attempts=1, query_cache_root=str(tmp_path / "cache"))
    assert reviewer.review([task], svamp_recipe.policy.rubric, tmp_path / "first")[0].status == ReviewStatus.INVALID
    assert reviewer.review([task], svamp_recipe.policy.rubric, tmp_path / "second")[0].status == ReviewStatus.REVIEWED
    assert len(service.batches) == 2


class PartialBatchService(BatchService):
    def output(self, batch):
        result = super().output(batch)
        if batch["id"] == "batch-0":
            return Output(result.output.splitlines()[0] + "\n")
        return result


def test_query_cache_fetches_only_missing_completions_in_same_evidence_directory(tmp_path, apple_row, svamp_recipe):
    service = PartialBatchService()
    source = Source(dataset="fixture", revision="1", row="0", importer_revision="1")
    normalized = [
        svamp_recipe.policy.normalize(RawRow("first", source, apple_row)),
        svamp_recipe.policy.normalize(RawRow("second", source, {**apple_row, "Body": "Bea has 2 apples."})),
    ]
    tasks = []
    for task in normalized:
        assert isinstance(task, TaskSpec)
        tasks.append(task)
    reviewer = BatchReviewer(service, "model", "deployment", max_attempts=1, query_cache_root=str(tmp_path / "cache"))
    first = reviewer.review(tasks, svamp_recipe.policy.rubric, tmp_path / "review")
    assert [record.status for record in first] == [ReviewStatus.REVIEWED, ReviewStatus.UNAVAILABLE]
    resumed = replace(reviewer).review(tasks, svamp_recipe.policy.rubric, tmp_path / "review")
    assert all(record.status == ReviewStatus.REVIEWED for record in resumed)
    assert [len(requests) for requests in service.batches.values()] == [2, 1]
    assert [record.task_id for record in resumed] == ["first", "second"]


def test_repeated_source_judge_context_still_receives_full_quality_review(tmp_path, apple_row, svamp_recipe):
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
    package = source_contract_package("rubric", "1", contract, ("semantic judge",))
    source = Source(dataset="fixture", revision="1", row="0", importer_revision="1")
    task = svamp_recipe.policy.normalize(RawRow("conversation", source, apple_row))
    assert isinstance(task, TaskSpec)
    task = task.model_copy(
        update={
            "context": ConversationInput(events=(TextMessage(role="user", content=question),)),
            "verifier": package.verifier,
            "resources": ResourceGroups(verifier=package.resources),
        }
    )
    original = task.model_dump_json()
    service = BatchService()
    reviewer = BatchReviewer(service, "fixture-model", "fixture-deployment", max_prompt_characters=128000)
    review = reviewer.review([task], rubric_tasks.RUBRICS["wizard_orca"], tmp_path)
    assert review[0].status == ReviewStatus.REVIEWED
    payload = json.loads(service.batches["batch-0"][0]["body"]["messages"][1]["content"])
    assert payload["context"]["events"][0]["content"] == question
    parameters = payload["grader_data"]["contract"]
    assert parameters["aggregation"] == rules
    assert parameters["criteria"] == ["Never disclose the password"]
    assert task.model_dump_json() == original


def test_numina_inequality_proof_is_unsupported_instead_of_a_failed_scalar_witness():
    # Numina train-00003-of-00005.parquet:32188 supplies this proof request
    # and boxed conclusion; comparing its answer cannot grade the proof.
    problem = (
        r"Given a positive number \( M \) and an array"
        "\n$$\n"
        r"\begin{array}{l}"
        "\n"
        r"a_{11}, a_{12}, \cdots, a_{1n} \\"
        "\n"
        r"a_{21}, a_{22}, \cdots, a_{2n} \\"
        "\n"
        r"a_{n1}, a_{n2}, \cdots, a_{nn}"
        "\n"
        r"\end{array}"
        "\n$$\n\n"
        r"such that for any \( x_1, x_2, \cdots, x_n \in \{-1, 1\} \),"
        "\n$$\n"
        r"\sum_{k=1}^{n} \left| a_{k1} x_1 + a_{k2} x_2 + \cdots + a_{kn} x_n \right| \leq M,"
        "\n$$\n\n"
        r"prove that \( \left| a_{11} \right| + \left| a_{22} \right| + \cdots + \left| a_{nn} \right| \leq M \)."
    )
    conclusion = r"\left| a_{11} \right| + \left| a_{22} \right| + \cdots + \left| a_{n n} \right| \leqslant M"
    source = Source(dataset="AI-MO/NuminaMath-CoT", revision="fixture", row="32188", importer_revision="1")
    data = {"problem": problem, "solution": rf"\boxed{{{conclusion}}}", "source": "olympiads"}

    result = normalize_numina_math(RawRow("proof", source, data))

    assert isinstance(result, ImportRejection)
    assert result.kind is ImportFailureKind.UNSUPPORTED
    assert result.reason == "unsupported_proof_contract"


def test_numina_numeric_reference_keeps_working_math_controls():
    # Actual Numina train-00004-of-00005.parquet:165579 has numeric answer19.
    problem = (
        r"Let  $ABCD$  be a quadrilateral with an inscribed circle  $\omega$  and let  $P$  be the "
        r"intersection of its diagonals  $AC$  and  $BD$ . Let  $R_1$ ,  $R_2$ ,  $R_3$ ,  $R_4$  be "
        r"the circumradii of triangles  $APB$ ,  $BPC$ ,  $CPD$ ,  $DPA$  respectively. If  $R_1=31$  "
        r"and  $R_2=24$  and  $R_3=12$ , find  $R_4$ ."
    )
    source = Source(dataset="AI-MO/NuminaMath-CoT", revision="fixture", row="165579", importer_revision="1")
    data = {"problem": problem, "solution": r"The final answer is $\boxed{19}$.", "source": "aops_forum"}

    task = normalize_numina_math(RawRow("numeric", source, data))

    assert isinstance(task, TaskSpec)
    assert task.context.events[0].content == problem
    assert {check.check: check.status for check in math_controls(task).checks} == {
        "empty": CheckStatus.PASS,
        "witness": CheckStatus.PASS,
        "negative": CheckStatus.PASS,
    }
    evidence = json.loads(resource_bytes(task.resources.verifier[0]))
    assert evidence == {"solution": data["solution"], "source": data["source"]}


@pytest.mark.parametrize("kind", ["math", "mcq"])
def test_canonical_merge_ignores_private_solution_evidence_but_retains_grader_conflicts(tmp_path, kind):
    references = ["5", "5", "1", "2", "5"] if kind == "math" else ["A", "A", "B", "C", "A"]
    rows = []
    original_resources = {}
    for index, expected in enumerate(references):
        name = chr(97 + index)
        source = Source(dataset=name, revision="a" * 40, row="0", importer_revision="1")
        verifier = (
            grader_package(MathSpec(expected=expected)).verifier
            if kind == "math"
            else grader_package(McqSpec(expected=expected, options=4)).verifier
        )
        resources = ResourceGroups(
            verifier=(inline_resource("reference/source-evidence.json", json.dumps({"solution": name}).encode()),),
            worker=(inline_resource("input/context.txt", b"different public input" if index == 4 else b"public input"),),
        )
        task = TaskSpec(
            id=name,
            source=source,
            environment_requirements=EnvironmentRequirements(),
            answer_type=AnswerType.TEXT,
            context=ConversationInput(
                events=(TextMessage(role="user", content="conflict" if index in (2, 3) else "duplicate"),)
            ),
            resources=resources,
            verifier=verifier,
        )
        original_resources[name] = resources
        audit = TaskAudit(
            task_id=name,
            source=source,
            raw={"solution": name},
            normalized=task,
            normalization_rejection=None,
            checks=[],
            review=None,
            intended_use=IntendedUse.TRAIN,
            decision=Decision(task_id=name, disposition=Disposition.KEEP, reasons=[]),
        )
        rows.append(audit_columns(audit))
    merged = tmp_path / "merged"
    (merged / "data").mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist(rows, schema=TASK_SCHEMA), merged / "data/part-0.parquet")
    (merged / "manifest.json").write_text(json.dumps({"input_rows": len(rows)}))
    output = tmp_path / "canonical"
    canonicalize_sources(str(merged), str(output), max_workers=1)
    audited = {
        row["task_id"]: row for file in (output / "audit").glob("*.parquet") for row in pq.read_table(file).to_pylist()
    }
    assert {name for name, row in audited.items() if row["filter_status"] == "keep"} == {"a", "e"}
    assert audited["b"]["duplicate_of"] == "a"
    assert (
        audited["c"]["filter_reasons"]
        == audited["d"]["filter_reasons"]
        == ["cross_source_conflicting_verifier_contracts"]
    )
    assert {
        name: TaskSpec.model_validate_json(row["task_json"]).resources for name, row in audited.items()
    } == original_resources


@pytest.mark.parametrize("grader_kind", ["source_unavailable", "native_command"])
def test_canonical_merge_preserves_distinct_opaque_contracts_and_deduplicates_exact_copies(tmp_path, grader_kind):
    rows = []
    contracts: dict[str, dict[str, JsonValue]] = {
        "a": {"uuid": "first"},
        "b": {"uuid": "second"},
        "c": {"uuid": "first"},
    }
    for name, contract in contracts.items():
        source = Source(dataset=name, revision="a" * 40, row="0", importer_revision="1")
        package = source_contract_package(
            evaluator="unbound-source-agent",
            source_revision="b" * 40,
            contract=contract,
            runtime_requirements=("source evaluator",),
        )
        if grader_kind == "native_command":
            package = native_command_package(
                NativeCommandSpec(
                    argv=("bash", "/tests/test.sh"),
                    cwd="/",
                    result_format="reward_file",
                    result_path="/logs/verifier/reward.txt",
                    timeout=60,
                ),
                (inline_resource("config.json", json.dumps({"contract": contract}).encode()),),
            )
        task = TaskSpec(
            id=name,
            source=source,
            environment_requirements=EnvironmentRequirements(),
            answer_type=AnswerType.TEXT,
            context=ConversationInput(events=(TextMessage(role="user", content="Shared public question"),)),
            verifier=package.verifier,
            resources=ResourceGroups(verifier=package.resources),
        )
        audit = TaskAudit(
            task_id=name,
            source=source,
            raw={"contract": contract},
            normalized=task,
            normalization_rejection=None,
            checks=[],
            review=None,
            intended_use=IntendedUse.TRAIN,
            decision=Decision(task_id=name, disposition=Disposition.KEEP, reasons=[]),
        )
        rows.append(audit_columns(audit))
    merged = tmp_path / "merged"
    (merged / "data").mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist(rows, schema=TASK_SCHEMA), merged / "data/part-0.parquet")
    (merged / "manifest.json").write_text(json.dumps({"input_rows": len(rows)}))
    output = tmp_path / "canonical"
    canonicalize_sources(str(merged), str(output), max_workers=1)
    audited = {
        row["task_id"]: row for file in (output / "audit").glob("*.parquet") for row in pq.read_table(file).to_pylist()
    }
    assert {name for name, row in audited.items() if row["filter_status"] == "keep"} == {"a", "b"}
    assert audited["c"]["duplicate_of"] == "a"
    assert {
        name: grader_config(TaskSpec.model_validate_json(row["task_json"]))["contract"] for name, row in audited.items()
    } == contracts


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


class CertainMiddleBatchService(BatchService):
    def output(self, batch):
        if batch["id"] != "batch-0":
            return super().output(batch)
        requests = self.batches[batch["id"]]
        records = [
            response(request["custom_id"], quality="bad" if index < 19 else "good")
            for index, request in enumerate(requests[:-1])
        ]
        return Output("".join(json.dumps(record) + "\n" for record in records))


def test_certain_middle_panel_reviews_remaining_tasks_without_imputing_missing_verdict(
    tmp_path, apple_row, svamp_recipe
):
    staged, prepared, quality, audited, filtered = (
        tmp_path / name for name in ("staged", "prepared", "quality", "audited", "filtered")
    )
    staged.mkdir()
    rows = [{**apple_row, "Body": f"Person {index} has 2 apples."} for index in range(120)]
    (staged / "source.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    service = CertainMiddleBatchService()
    reviewer = BatchReviewer(service, "fixture", "quality-panel", max_attempts=1)
    config = ReviewConfig(
        reviewer.model, reviewer.model_revision, max_attempts=1, transport=ReviewTransport.PROVIDER_BATCH
    )
    execution = AuditExecution(max_workers=1, review_batch_size=100, reviewer=reviewer)
    prepare_source(str(staged), str(prepared), svamp_recipe, svamp_recipe.inputs.files, None, execution)
    report = assess_source_quality(str(prepared), str(quality), svamp_recipe, config, SourceQualityPolicy(), execution)
    assert report.status == "full_review"
    audit_prepared_source(str(prepared), str(quality), str(audited), svamp_recipe, config, execution)
    manifest = filter_source(str(audited), str(filtered), FilterPolicy(), max_workers=1)
    records = [row for path in (filtered / "audit").glob("*.parquet") for row in pq.read_table(path).to_pylist()]
    requests = [request["custom_id"] for batch in service.batches.values() for request in batch]
    assert len(requests) == len(set(requests)) == 120
    assert manifest["dispositions"] == {"keep": 100, "reject": 19, "defer": 1}
    missing = [row for row in records if row["review_status"] == "unavailable"]
    assert len(missing) == 1
    assert missing[0]["filter_status"] == "defer"
    assert missing[0]["review_quality"] is None
    assert all(row["quality_basis"] != "inferred_from_source" for row in records)


@pytest.mark.parametrize(
    "service_factory,status,sample_review_statuses",
    [
        (BatchService, "trust", {"reviewed"}),
        (OneMissingResponseService, "trust", {"reviewed", "unavailable"}),
        (MixedQualityBatchService, "reject", {"reviewed"}),
        (partial(BatchService, quality="unknown"), "full_review", {"reviewed"}),
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
    config = ReviewConfig(
        reviewer.model, reviewer.model_revision, max_attempts=1, transport=ReviewTransport.PROVIDER_BATCH
    )
    execution = AuditExecution(max_workers=2, review_batch_size=10, reviewer=reviewer)
    prepare_source(str(staged), str(prepared), svamp_recipe, svamp_recipe.inputs.files, None, execution)
    assert not service.batches
    report = assess_source_quality(
        str(prepared),
        str(quality),
        svamp_recipe,
        config,
        SourceQualityPolicy(sample_size=30, trust_below=0.15, reject_above=0.20),
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
    assert len(requests) == len(set(requests)) == (100 if status == "full_review" else 30)
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
    elif status == "incomplete":
        assert manifest["dispositions"] == {"defer": 100, "reject": 2}
    else:
        assert manifest["reviewed_rows"] == 100
        assert all(record["quality_basis"] != "inferred_from_source" for record in records)


def test_source_quality_without_eligible_tasks_retains_import_failures(tmp_path, apple_row, svamp_recipe):
    staged = tmp_path / "staged"
    staged.mkdir()
    (staged / "source.jsonl").write_text(json.dumps({**apple_row, "Answer": "invalid"}) + "\n")
    service = BatchService()
    reviewer = BatchReviewer(service, "fixture", "empty-quality-panel")
    config = ReviewConfig(reviewer.model, reviewer.model_revision, transport=ReviewTransport.PROVIDER_BATCH)
    execution = AuditExecution(reviewer=reviewer)
    prepared, quality, audited = (str(tmp_path / name) for name in ("prepared", "quality", "audited"))
    prepare_source(str(staged), prepared, svamp_recipe, svamp_recipe.inputs.files, None, execution)
    report = assess_source_quality(prepared, quality, svamp_recipe, config, SourceQualityPolicy(), execution)
    assert report.status == "reject" and report.population.eligible_count == 0
    assert report.population.input_count == 1 and report.defect_fraction == 1.0
    manifest = audit_prepared_source(prepared, quality, audited, svamp_recipe, config, execution)
    assert not service.batches
    assert manifest["input_rows"] == 1 and manifest["dispositions"] == {"reject": 1}


@pytest.mark.parametrize("fixture", ["x" * 600000, list(range(100000))])
def test_large_encoded_code_tests_receive_bounded_review_without_changing_audit(tmp_path, fixture):
    source = Source(dataset="fixture", revision="1", row="0", importer_revision="1")
    tests = {"inputs": [fixture, "small"], "outputs": ["yes", "no"], "fn_name": "solve"}
    task = code_contracts.normalize_eurus2_code(
        RawRow(
            "large-code",
            source,
            {
                "ability": "code",
                "prompt": [{"role": "user", "content": "Implement solve for the supplied input."}],
                "reward_model": {"style": "rule", "ground_truth": json.dumps(tests)},
                "extra_info": {},
                "data_source": "fixture",
            },
        )
    )
    assert isinstance(task, TaskSpec)
    original = task.model_dump_json()
    service = BatchService()
    reviewer = BatchReviewer(service, "fixture-model", "fixture-deployment", max_prompt_characters=128000)
    records = reviewer.review([task], code_contracts.EURUS2_CODE_RUBRIC, tmp_path)
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


def test_unicode_public_prompt_uses_character_budget_and_oversized_prompt_is_not_truncated(
    tmp_path, apple_row, svamp_recipe
):
    source = Source(dataset="fixture", revision="1", row="0", importer_revision="1")
    task = svamp_recipe.policy.normalize(RawRow("unicode", source, apple_row))
    assert isinstance(task, TaskSpec)
    prompt = "漢字" * 5000
    task = task.model_copy(update={"context": ConversationInput(events=(TextMessage(role="user", content=prompt),))})
    service = BatchService()
    reviewer = BatchReviewer(service, "fixture-model", "fixture-deployment", max_prompt_characters=40000)
    assert reviewer.review([task], svamp_recipe.policy.rubric, tmp_path / "unicode")[0].status == ReviewStatus.REVIEWED
    payload = json.loads(service.batches["batch-0"][0]["body"]["messages"][1]["content"])
    assert payload["context"]["events"][0]["content"] == prompt
    oversized = task.model_copy(
        update={"context": ConversationInput(events=(TextMessage(role="user", content=prompt * 10),))}
    )
    assert (
        reviewer.review([oversized], svamp_recipe.policy.rubric, tmp_path / "oversized")[0].status
        == ReviewStatus.UNAVAILABLE
    )
    assert len(service.batches) == 1


def test_recorded_review_does_not_bypass_fallback_model_validation(tmp_path, svamp_recipe):
    service = BatchService()
    fallback = BatchReviewer(service, "actual-model", "actual-revision")
    reviewer = RecordedReviewer(fallback, str(tmp_path / "manual.json"), "a" * 64)
    with pytest.raises(ValueError, match="Review configuration differs"):
        assess_source_quality(
            str(tmp_path / "prepared"),
            str(tmp_path / "quality"),
            svamp_recipe,
            ReviewConfig("different-model", "different-revision", transport=ReviewTransport.PROVIDER_BATCH),
            SourceQualityPolicy(),
            AuditExecution(reviewer=reviewer),
        )
    assert not service.files and not service.batches
