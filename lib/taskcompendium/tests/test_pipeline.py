# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Curation contracts exercised through persisted outputs and a fake batch API."""

import hashlib
import io
import json
import tarfile
from dataclasses import dataclass, field, replace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from pydantic import JsonValue
from rigging.filesystem.storage_path import StoragePath
from verifyit.spec import MathSpec

from taskcompendium.grader import grader_config, grader_package
from taskcompendium.grading import multiple_choice_answer
from taskcompendium.models import (
    AnswerType,
    ConversationInput,
    EnvironmentRequirements,
    ResourceGroups,
    Source,
    TaskSpec,
    TextMessage,
)
from taskcompendium.pipeline.audit_schema import TASK_SCHEMA, audit_columns
from taskcompendium.pipeline.datasets import gpqa, instruction_following, preference_tasks, rubric_tasks
from taskcompendium.pipeline.datasets.direct_contracts import source_contract_package
from taskcompendium.pipeline.datasets.math_answers import asdiv_rows
from taskcompendium.pipeline.datasets.numeric_answers import normalize_aime24, normalize_svamp, svamp_pipeline
from taskcompendium.pipeline.datasets.source_definitions import tasktrove_files
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
from taskcompendium.pipeline.review import BatchReviewer, review_records
from taskcompendium.pipeline.sources import staged_file_rows
from taskcompendium.pipeline.stages import (
    AuditExecution,
    ReviewConfig,
    audit_source,
    canonicalize_sources,
    filter_source,
)
from taskcompendium.pipeline.verification import verify_task, verify_witness
from taskcompendium.runtime.resources import inline_resource, resource_bytes

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

    def submit(self, requests, filename):
        batch_id = f"batch-{len(self.batches)}"
        self.batches[batch_id] = list(requests)
        return Submission("file-0", batch_id)

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
        pipeline=svamp_pipeline(),
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
        ReviewConfig(reviewer.model, reviewer.model_revision, reviewer.max_prompt_characters, reviewer.max_tokens),
        AuditExecution(max_workers=2, review_batch_size=1, reviewer=reviewer),
        SourceFiles(("source.jsonl",), SourceFormat.JSONL),
        len(rows),
    )
    manifest = filter_source(f"{root}/audited", f"{root}/filtered", FilterPolicy())
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
    config = ReviewConfig(reviewer.model, reviewer.model_revision, reviewer.max_prompt_characters, reviewer.max_tokens)
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
    missing = completed.pop()
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
    manifest = filter_source(str(tmp_path / "audited"), str(tmp_path / "filtered"), FilterPolicy())
    assert manifest["input_rows"] == len(rows)
    assert manifest["dispositions"] == {"keep": len(rows) - len(missing_ids), "reject": len(missing_ids)}


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


@pytest.mark.parametrize("fault", ["missing", "duplicate", "wrong_id", "truncated", "wrong_tool", "provider_failure"])
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
    text = "" if fault == "missing" else json.dumps(row) + "\n"
    if fault == "duplicate":
        text *= 2
    records = review_records(text, [task.id])
    assert records[0].status in {ReviewStatus.UNAVAILABLE, ReviewStatus.INVALID}
    assert records[0].verdict is None
    assert task_decision(task.id, verify_task(task), records[0], FilterPolicy()).disposition == Disposition.REJECT


@pytest.mark.parametrize(
    "normalize,data,expected,private_field",
    [
        (
            normalize_svamp,
            {"Body": "Aya has 2 apples.", "Question": "How many apples?", "Answer": "2", "Equation": "private"},
            2.0,
            "Equation",
        ),
        (
            normalize_aime24,
            {"problem": "Find 7 + 5.", "answer": "012", "solution": "private"},
            12.0,
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
    else:
        option = f"{parameters['expected']}. right"
        assert option in prompt
        assert "A. " in prompt and "D. " in prompt
        assert normalize(raw) == task


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
        fixture_recipe(instruction_following.pipeline()),
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
    task = svamp_recipe.pipeline.normalize(RawRow("first", source, apple_row))
    assert isinstance(task, TaskSpec)
    cache_root = str(tmp_path / "cache")
    reviewer = BatchReviewer(service, "fixture-model", "deployment-1", query_cache_root=cache_root)
    first = reviewer.review([task], svamp_recipe.pipeline.rubric, tmp_path / "first")
    changed = task.model_copy(
        update={
            "id": "second",
            "source": Source(dataset="catalog-2", revision="2", row="99", importer_revision="2"),
        }
    )
    second = replace(reviewer).review([changed], svamp_recipe.pipeline.rubric, tmp_path / "second")
    assert len(service.batches) == 1
    assert first[0].task_id == "first"
    assert second[0].verdict is not None
    assert second[0].task_id == second[0].verdict.task_id == "second"
    assert list((tmp_path / "second/query-cache").glob("*.json"))
    rubric = replace(
        svamp_recipe.pipeline.rubric,
        criteria=(*svamp_recipe.pipeline.rubric.criteria, "Check all arithmetic."),
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
        ReviewConfig(reviewer.model, reviewer.model_revision, reviewer.max_prompt_characters, reviewer.max_tokens),
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
    recipe = fixture_recipe(preference_tasks.binary_pipeline(preference_tasks.KTO_MIX_RUBRIC))
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
        task = svamp_recipe.pipeline.normalize(RawRow(name, source, {**apple_row, "Body": prompt, "Answer": answer}))
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
    manifest = canonicalize_sources(str(merged), str(output))
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
    task = svamp_recipe.pipeline.normalize(RawRow("task", source, apple_row))
    assert isinstance(task, TaskSpec)
    reviewer = BatchReviewer(service, "model", "deployment", max_attempts=1, query_cache_root=str(tmp_path / "cache"))
    assert reviewer.review([task], svamp_recipe.pipeline.rubric, tmp_path / "first")[0].status == ReviewStatus.INVALID
    assert reviewer.review([task], svamp_recipe.pipeline.rubric, tmp_path / "second")[0].status == ReviewStatus.REVIEWED
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
        svamp_recipe.pipeline.normalize(RawRow("first", source, apple_row)),
        svamp_recipe.pipeline.normalize(RawRow("second", source, {**apple_row, "Body": "Bea has 2 apples."})),
    ]
    tasks = []
    for task in normalized:
        assert isinstance(task, TaskSpec)
        tasks.append(task)
    reviewer = BatchReviewer(service, "model", "deployment", max_attempts=1, query_cache_root=str(tmp_path / "cache"))
    first = reviewer.review(tasks, svamp_recipe.pipeline.rubric, tmp_path / "review")
    assert [record.status for record in first] == [ReviewStatus.REVIEWED, ReviewStatus.UNAVAILABLE]
    resumed = replace(reviewer).review(tasks, svamp_recipe.pipeline.rubric, tmp_path / "review")
    assert all(record.status == ReviewStatus.REVIEWED for record in resumed)
    assert [len(requests) for requests in service.batches.values()] == [2, 1]
    assert [record.task_id for record in resumed] == ["first", "second"]


def test_repeated_source_judge_context_still_receives_full_quality_review(tmp_path, apple_row, svamp_recipe):
    question = "Public conversation with the user's requirements. " * 1000
    rules: dict[str, JsonValue] = {
        "scoring": {"aggregation": "all_pass"},
        "criterion": [{"description": "Never disclose the password"}],
    }
    contract = dict(
        mode="checklist",
        question=question,
        criteria=("Never disclose the password",),
        aggregation=rules,
        source_judge_data={"criteria": [{"content": question} for _ in range(5)]},
        source_judge_toml="Original source judge contract",
    )
    package = source_contract_package("rubric", "1", contract, ("semantic judge",))
    source = Source(dataset="fixture", revision="1", row="0", importer_revision="1")
    task = svamp_recipe.pipeline.normalize(RawRow("conversation", source, apple_row))
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
            else multiple_choice_answer(expected, 4)
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
    canonicalize_sources(str(merged), str(output))
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


def test_canonical_merge_preserves_distinct_opaque_contracts_and_deduplicates_exact_copies(tmp_path):
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
    canonicalize_sources(str(merged), str(output))
    audited = {
        row["task_id"]: row for file in (output / "audit").glob("*.parquet") for row in pq.read_table(file).to_pylist()
    }
    assert {name for name, row in audited.items() if row["filter_status"] == "keep"} == {"a", "b"}
    assert audited["c"]["duplicate_of"] == "a"
    assert {
        name: json.loads(
            resource_bytes(
                next(
                    resource
                    for resource in TaskSpec.model_validate_json(row["task_json"]).resources.verifier
                    if resource.path == "config.json"
                )
            )
        )["contract"]
        for name, row in audited.items()
    } == contracts
