# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Curation contracts exercised through persisted outputs and a fake batch API."""

import json
from dataclasses import dataclass, field
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from taskcompendium.models import Source, TaskSpec
from taskcompendium.pipeline.datasets import aime24, gpqa, instruction_following, svamp
from taskcompendium.pipeline.filtering import task_decision
from taskcompendium.pipeline.models import (
    CheckStatus,
    Confidence,
    Disposition,
    FilterPolicy,
    ImportRejection,
    Quality,
    RawRow,
    ReferenceStatus,
    ReviewRecord,
    ReviewStatus,
    ReviewVerdict,
)
from taskcompendium.pipeline.review import BatchReviewer, review_records
from taskcompendium.pipeline.runner import run_pipeline
from taskcompendium.pipeline.verification import verify_task, verify_witness


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
    """Fake external inference service; requests and acknowledged batches survive callers."""

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


def read_jsonl(path: Path):
    with path.open() as stream:
        return [json.loads(line) for line in stream]


@pytest.fixture
def apple_row():
    return {
        "Body": "Aya has 2 apples.\u2028They belong to Aya.",
        "Question": "How many apples does Aya have?",
        "Answer": "2",
        "Equation": "2",
    }


def test_pipeline_accounts_for_rejects_duplicates_and_conflicting_keys(tmp_path, apple_row):
    conflicting = {**apple_row, "Body": "Aya has some apples."}
    rows = [
        apple_row,
        apple_row,
        {**conflicting, "Answer": "1"},
        {**conflicting, "Answer": "2"},
        {**apple_row, "Answer": None},
    ]
    service = BatchService()
    manifest = run_pipeline(
        svamp.recipe,
        rows,
        output_path=tmp_path,
        limit=5,
        reviewer=BatchReviewer(service, "fixture-model", "fixture-deployment"),
    )

    decisions = read_jsonl(tmp_path / "decisions.jsonl")
    raw = read_jsonl(tmp_path / "raw.jsonl")
    by_id = {item["task_id"]: item for item in decisions}
    assert manifest["dispositions"] == {"keep": 1, "reject": 4}
    assert by_id[raw[1]["task_id"]]["duplicate_of"] == raw[0]["task_id"]
    assert by_id[raw[4]["task_id"]]["reasons"][0] == "normalize:invalid_reference"
    assert all(by_id[raw[index]["task_id"]]["reasons"] == ["conflicting_references"] for index in (2, 3))
    accepted = [
        TaskSpec.model_validate_json(row["task_json"])
        for row in pq.read_table(tmp_path / "accepted.parquet").to_pylist()
    ]
    assert [task.id for task in accepted] == [raw[0]["task_id"]]
    assert accepted[0].source.revision == svamp.recipe.source.revision
    assert apple_row["Body"] in accepted[0].context.events[0].content
    assert "Equation" not in service.batches["batch-0"][0]["body"]["messages"][1]["content"]
    controls = read_jsonl(tmp_path / "checks.jsonl")[0]["checks"]
    assert [(check["check"], check["status"]) for check in controls] == [
        ("empty", "pass"),
        ("reference", "pass"),
        ("perturbed", "pass"),
    ]
    audit = pq.read_table(tmp_path / "audit.parquet").to_pylist()
    assert [json.loads(row["raw_json"]) for row in audit] == raw
    assert {row["task_id"]: row["filter_status"] for row in audit} == {
        key: value["disposition"] for key, value in by_id.items()
    }
    assert {row["task_id"]: row["filter_reasons"] for row in audit} == {
        key: value["reasons"] for key, value in by_id.items()
    }
    assert TaskSpec.model_validate_json(audit[0]["task_json"]) == accepted[0]
    assert audit[0]["checks"] == controls
    assert audit[0]["review_evidence"] == read_jsonl(tmp_path / "reviews.jsonl")[0]["verdict"]["evidence"]
    assert audit[1]["task_json"] is not None  # Duplicate inputs remain inspectable.
    assert audit[4]["task_json"] is None
    assert audit[4]["normalization_reason"] == "invalid_reference"
    assert audit[4]["normalization_detail"]


def test_pipeline_resumes_acknowledged_batch_and_refilters_without_new_requests(tmp_path, apple_row):
    service = BatchService(confidence="medium", interrupted=True)
    reviewer = BatchReviewer(service, "fixture-model", "fixture-deployment")
    strict_policy = FilterPolicy(id="high-confidence", minimum_confidence=Confidence.HIGH)
    with pytest.raises(TimeoutError):
        run_pipeline(svamp.recipe, [apple_row], output_path=tmp_path, limit=1, reviewer=reviewer, policy=strict_policy)

    assert json.loads((tmp_path / "review/batch-state.json").read_text())["batch_id"] == "batch-0"
    resumed = run_pipeline(
        svamp.recipe, iter(()), output_path=tmp_path, limit=1, reviewer=reviewer, policy=strict_policy
    )
    assert resumed["dispositions"] == {"reject": 1}
    pending_audit = pq.read_table(tmp_path / "audit.parquet").to_pylist()[0]
    accepted = run_pipeline(
        svamp.recipe,
        iter(()),
        output_path=tmp_path,
        limit=1,
        reviewer=reviewer,
        policy=FilterPolicy(id="medium-confidence", minimum_confidence=Confidence.MEDIUM),
    )
    assert accepted["dispositions"] == {"keep": 1}
    assert len(service.batches) == 1
    assert pq.read_table(tmp_path / "accepted.parquet").num_rows == 1
    accepted_audit = pq.read_table(tmp_path / "audit.parquet").to_pylist()[0]
    assert accepted_audit["filter_status"] == "keep"
    assert pending_audit["filter_status"] == "reject"
    assert accepted_audit["raw_json"] == pending_audit["raw_json"]
    assert accepted_audit["review_evidence"] == pending_audit["review_evidence"]


def test_pipeline_retries_invalid_model_reply_and_preserves_both_attempts(tmp_path, apple_row):
    service = BatchService(invalid_first_batch=True)
    reviewer = BatchReviewer(service, "fixture-model", "fixture-deployment")
    manifest = run_pipeline(svamp.recipe, [apple_row], output_path=tmp_path, limit=1, reviewer=reviewer)
    assert manifest["dispositions"] == {"keep": 1}
    assert manifest["reviewed_rows"] == 1
    task_id = read_jsonl(tmp_path / "raw.jsonl")[0]["task_id"]
    initial = review_records((tmp_path / "review/raw-output.jsonl").read_text(), [task_id])
    retry = review_records((tmp_path / "review/retry-1/raw-output.jsonl").read_text(), [task_id])
    assert initial[0].status == ReviewStatus.INVALID
    assert retry[0].status == ReviewStatus.REVIEWED
    run_pipeline(svamp.recipe, iter(()), output_path=tmp_path, limit=1, reviewer=reviewer)
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
    task = svamp.normalize(
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
    task = svamp.normalize(raw)
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
    "module,data,expected,private_field",
    [
        (
            svamp,
            {"Body": "Aya has 2 apples.", "Question": "How many apples?", "Answer": "2", "Equation": "private"},
            2.0,
            "Equation",
        ),
        (aime24, {"problem": "Find 7 + 5.", "answer": "012", "solution": "private"}, 12.0, "solution"),
        (
            gpqa,
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
def test_recipes_normalize_source_contract_and_keep_supervision_private(module, data, expected, private_field):
    raw = RawRow("task-0", Source(dataset="fixture", revision="1", row="0", importer_revision="1"), data)
    task = module.normalize(raw)
    assert isinstance(task, TaskSpec)
    assert all(result.status.value == "pass" for result in verify_task(task))
    prompt = task.context.events[0].content
    assert private_field not in prompt and "private" not in prompt
    parameters = json.loads(task.verifier.parameters_json)
    if expected is not None:
        assert parameters["expected"] == expected
    else:
        option = f"{parameters['expected']}. right"
        assert option in prompt
        assert "A. " in prompt and "D. " in prompt
        assert module.normalize(raw) == task


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
    manifest = run_pipeline(
        instruction_following.recipe(tmp_path / "snapshot.jsonl"),
        [row],
        output_path=tmp_path / "run",
        limit=1,
        reviewer=BatchReviewer(BatchService(quality=quality), "model", "deployment"),
    )
    assert manifest["reviewed_rows"] == 1
    assert manifest["dispositions"] == {disposition: 1}
    accepted_table = pq.read_table(tmp_path / "run/accepted.parquet")
    assert accepted_table.num_rows == (1 if disposition == "keep" else 0)
    assert accepted_table.schema == pq.read_table(tmp_path / "run/audit.parquet").schema
    audit = pq.read_table(tmp_path / "run/audit.parquet").to_pylist()[0]
    assert json.loads(audit["raw_json"])["data"] == row
    assert json.loads(audit["task_json"])["context"]["events"][0]["content"] == row["instruction"]
    assert audit["review_evidence"] == read_jsonl(tmp_path / "run/reviews.jsonl")[0]["verdict"]["evidence"]
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
