# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Instruction repairs retain grader behavior, originals, and auditable lineage."""

import json
from dataclasses import dataclass, field

import pyarrow.parquet as pq
import pytest

from taskcompendium.models import Source, TaskSpec
from taskcompendium.pipeline.datasets import instruction_following, structured_output
from taskcompendium.pipeline.filtering import task_decision
from taskcompendium.pipeline.models import (
    Confidence,
    Decision,
    Disposition,
    FilterPolicy,
    Quality,
    RawRow,
    ReferenceStatus,
    ReviewRecord,
    ReviewRubric,
    ReviewStatus,
    ReviewVerdict,
)
from taskcompendium.pipeline.parquet import write_accepted_parquet
from taskcompendium.pipeline.records import read_jsonl
from taskcompendium.pipeline.rewriting import BatchRewriter, protected_text_checks, rewrite_records, write_rewrite_audit
from taskcompendium.pipeline.verification import verify_witness


@dataclass(frozen=True)
class Submission:
    file_id: str
    batch_id: str


@dataclass(frozen=True)
class Output:
    output: str
    errors: str | None = None


def rewrite_response(task_id, action, edits):
    return {
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
                                                "action": action,
                                                "edits": edits,
                                                "reason": "Clarify the existing generation contract.",
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


@dataclass
class RewriteService:
    """An external batch-service fake with retained submissions."""

    responses: list[dict]
    batches: dict[str, list[dict]] = field(default_factory=dict)
    interrupted: bool = True

    def submit(self, requests, filename):
        batch_id = f"batch-{len(self.batches)}"
        self.batches[batch_id] = list(requests)
        return Submission("file-0", batch_id)

    def wait(self, batch_id, poll_seconds):
        if self.interrupted:
            self.interrupted = False
            raise TimeoutError("Disconnected after submission")
        return {"id": batch_id, "status": "completed"}

    def output(self, batch):
        return Output("".join(json.dumps(row) + "\n" for row in self.responses))


@pytest.fixture
def structured_task():
    schema = {
        "type": "object",
        "required": ["count"],
        "properties": {"count": {"type": "integer", "minimum": 1}},
        "additionalProperties": False,
    }
    instruction = (
        "Produce any instance satisfying this schema. Choose unstated values. "
        "Parse the document and recover all values.\n" + json.dumps(schema)
    )
    task = structured_output.normalize(
        RawRow(
            "source-task",
            Source(dataset="fixture", revision="1", row="0", importer_revision="1"),
            {"instruction": instruction, "verifier_data": {"schema_type": "json", "schema": schema}},
        )
    )
    assert isinstance(task, TaskSpec)
    return task


def test_rewrite_resumes_and_retains_original_grader_and_lineage(tmp_path, structured_task):
    old_text = "Parse the document and recover all values."
    replacement = structured_task.context.events[0].content.replace(old_text, "Generate a schema-valid instance.")
    service = RewriteService(
        [
            rewrite_response(
                structured_task.id,
                "rewrite",
                [{"old_text": old_text, "replacement": "Generate a schema-valid instance."}],
            )
        ]
    )
    rewriter = BatchRewriter(service, "model", "deployment")
    rubric = ReviewRubric("repair", "1", ("Preserve the schema.",))
    with pytest.raises(TimeoutError):
        rewriter.rewrite([structured_task], rubric, tmp_path)
    rewriter.rewrite([structured_task], rubric, tmp_path)
    rewriter.rewrite([structured_task], rubric, tmp_path)
    assert len(service.batches) == 1
    original = TaskSpec.model_validate_json((tmp_path / "originals.jsonl").read_text())
    candidate = TaskSpec.model_validate_json((tmp_path / "candidates.jsonl").read_text())
    assert original == structured_task
    assert candidate.id != original.id
    assert candidate.context.events[0].content == replacement
    assert candidate.model_dump(exclude={"context", "id"}) == original.model_dump(exclude={"context", "id"})
    for task in (original, candidate):
        checks = verify_witness(task, '{"count": 2}', '{"count": 0}')
        assert [check.status.value for check in checks] == ["pass", "pass", "pass"]
    lineage = json.loads((tmp_path / "lineage.jsonl").read_text())
    assert lineage["parent_id"] == original.id and lineage["task_id"] == candidate.id
    assert lineage["parent_sha256"] != lineage["candidate_sha256"]
    audit = pq.read_table(tmp_path / "audit.parquet").to_pylist()[0]
    assert TaskSpec.model_validate_json(audit["original_task_json"]) == original
    assert TaskSpec.model_validate_json(audit["task_json"]) == candidate
    assert audit["cleanup_edits"] == [{"old_text": old_text, "replacement": "Generate a schema-valid instance."}]
    assert audit["cleanup_reason"]
    assert json.loads(audit["cleanup_lineage_json"]) == lineage
    assert audit["filter_status"] is None  # A proposal has not yet passed the final filter.


@pytest.mark.parametrize("action", ["unchanged", "unrepairable"])
def test_final_rewrite_audit_keeps_original_decision_and_rejects_failed_candidate(tmp_path, structured_task, action):
    original = structured_task.model_copy(update={"id": "original-task"})
    edits = [{"old_text": "Parse the document and recover all values.", "replacement": "Generate an instance."}]
    service = RewriteService(
        [rewrite_response(structured_task.id, "rewrite", edits), rewrite_response(original.id, action, [])],
        interrupted=False,
    )
    BatchRewriter(service, "model", "deployment").rewrite(
        [structured_task, original], ReviewRubric("repair", "1", ()), tmp_path
    )
    candidate = TaskSpec.model_validate(read_jsonl(tmp_path / "candidates.jsonl")[0])
    checks = verify_witness(candidate, '{"count": 0}', "__invalid__")
    review = ReviewRecord(task_id=candidate.id, status=ReviewStatus.UNAVAILABLE, verdict=None, detail="No review")
    decision = task_decision(candidate.id, checks, review, FilterPolicy())
    original_checks = verify_witness(original, '{"count": 2}', '{"count": 0}')
    original_review = ReviewRecord(
        task_id=original.id,
        status=ReviewStatus.REVIEWED,
        verdict=ReviewVerdict(
            task_id=original.id,
            quality=Quality.GOOD,
            confidence=Confidence.HIGH,
            reference_status=ReferenceStatus.CONSISTENT,
            defects=[],
            evidence="The generation contract admits any integer count greater than zero.",
        ),
        detail="",
    )
    original_decision = task_decision(original.id, original_checks, original_review, FilterPolicy())
    table = write_rewrite_audit(
        tmp_path,
        checks={candidate.id: checks, original.id: original_checks},
        reviews=[review, original_review],
        decisions=[decision, original_decision],
    )
    write_accepted_parquet(tmp_path / "accepted.parquet", table)
    audit = pq.read_table(tmp_path / "audit.parquet").to_pylist()
    assert [row["parent_id"] for row in audit] == [structured_task.id, original.id]
    assert audit[0]["task_id"] == candidate.id
    assert TaskSpec.model_validate_json(audit[0]["task_json"]) == candidate
    assert audit[0]["cleanup_edits"] == edits
    assert audit[0]["filter_status"] == "reject"
    assert audit[0]["filter_reasons"] == ["check:witness"]
    assert audit[0]["checks"][1]["status"] == "fail"
    assert audit[1]["cleanup_action"] == action
    assert TaskSpec.model_validate_json(audit[1]["task_json"]) == original
    assert audit[1]["filter_status"] == "keep"
    assert audit[1]["review_evidence"] == original_review.verdict.evidence
    assert audit[1]["checks"] == [check.model_dump(mode="json") for check in original_checks]
    accepted = pq.read_table(tmp_path / "accepted.parquet").to_pylist()
    assert [row["task_id"] for row in accepted] == [original.id]


@pytest.mark.parametrize("fault", ["missing", "duplicate"])
def test_final_rewrite_decisions_cannot_omit_or_repeat_an_effective_task(tmp_path, structured_task, fault):
    second = structured_task.model_copy(update={"id": "second-task"})
    service = RewriteService(
        [rewrite_response(task.id, "unchanged", []) for task in (structured_task, second)], interrupted=False
    )
    BatchRewriter(service, "model", "deployment").rewrite(
        [structured_task, second], ReviewRubric("repair", "1", ()), tmp_path
    )
    decision = Decision(task_id=structured_task.id, disposition=Disposition.KEEP, reasons=[])
    decisions = [decision] if fault == "missing" else [decision, decision]
    with pytest.raises(ValueError, match="every effective task exactly once"):
        write_rewrite_audit(tmp_path, checks={}, reviews=(), decisions=decisions)


@pytest.mark.parametrize("action", ["unchanged", "unrepairable"])
def test_rewriter_retains_non_rewritten_tasks_without_candidates(tmp_path, structured_task, action):
    service = RewriteService([rewrite_response(structured_task.id, action, [])], interrupted=False)
    records = BatchRewriter(service, "model", "deployment").rewrite(
        [structured_task], ReviewRubric("repair", "1", ()), tmp_path
    )
    assert records[0].proposal.action.value == action
    assert not (tmp_path / "candidates.jsonl").read_text()
    assert TaskSpec.model_validate_json((tmp_path / "originals.jsonl").read_text()) == structured_task


@pytest.mark.parametrize("fault", ["missing", "duplicate", "truncated", "wrong_id"])
def test_incomplete_rewrite_responses_cannot_create_proposals(fault):
    row = rewrite_response(
        "task", "rewrite", [{"old_text": "Confusing instruction", "replacement": "Clear instruction"}]
    )
    if fault == "truncated":
        row["response"]["body"]["choices"][0]["finish_reason"] = "length"
    if fault == "wrong_id":
        function = row["response"]["body"]["choices"][0]["message"]["tool_calls"][0]["function"]
        value = json.loads(function["arguments"])
        value["task_id"] = "another"
        function["arguments"] = json.dumps(value)
    output = "" if fault == "missing" else json.dumps(row) + "\n"
    if fault == "duplicate":
        output *= 2
    record = rewrite_records(output, ["task"])[0]
    assert record.proposal is None
    assert record.status in {ReviewStatus.INVALID, ReviewStatus.UNAVAILABLE}


def test_ifeval_normalization_keeps_content_and_uses_real_constraint_checks():
    prompt = "如何提高抽象思维能力?请用两个项目符号回答。"
    row = RawRow(
        "task",
        Source(dataset="fixture", revision="1", row="0", importer_revision="1"),
        {
            "instruction": "You are running in a shell-based sandbox. Write /app/answer.txt.\n---\n" + prompt,
            "verifier_data": {
                "instruction_id_list": ["detectable_format:number_bullet_lists"],
                "kwargs": [{"num_bullets": 2}],
            },
        },
    )
    task = instruction_following.normalize(row)
    assert isinstance(task, TaskSpec)
    assert task.context.events[0].content == prompt
    assert [check.status.value for check in verify_witness(task, "* 练习分类\n* 比较不同概念", "只有一个段落")] == [
        "pass",
        "pass",
        "pass",
    ]


def test_rewrite_cannot_pass_public_schema_changes_with_an_unchanged_private_grader(tmp_path, structured_task):
    service = RewriteService(
        [rewrite_response(structured_task.id, "rewrite", [{"old_text": '"minimum": 1', "replacement": '"minimum": 0'}])],
        interrupted=False,
    )
    BatchRewriter(service, "model", "deployment").rewrite([structured_task], ReviewRubric("repair", "1", ()), tmp_path)
    candidate = TaskSpec.model_validate_json((tmp_path / "candidates.jsonl").read_text())
    assert all(check.status.value == "pass" for check in verify_witness(candidate, '{"count": 2}', '{"count": 0}'))
    original_schema = json.loads(structured_task.verifier.parameters_json)["document_schema_json"]
    checks = protected_text_checks(candidate, {"public_schema": original_schema})
    assert [(check.check, check.status.value) for check in checks] == [("preserve:public_schema", "fail")]


@pytest.mark.parametrize(
    "edits",
    [
        [{"old_text": "not present", "replacement": "new text"}],
        [
            {"old_text": "Choose unstated values.", "replacement": "Choose values."},
            {"old_text": "unstated values", "replacement": "values"},
        ],
    ],
)
def test_invalid_literal_edits_remain_recorded_without_candidates(tmp_path, structured_task, edits):
    service = RewriteService([rewrite_response(structured_task.id, "rewrite", edits)], interrupted=False)
    records = BatchRewriter(service, "model", "deployment").rewrite(
        [structured_task], ReviewRubric("repair", "1", ()), tmp_path
    )
    assert records[0].status == ReviewStatus.INVALID
    assert records[0].proposal is None
    assert not (tmp_path / "candidates.jsonl").read_text()


@pytest.mark.parametrize("required,expected_failure", [(["materials"], True), ([], False)])
def test_mandatory_schema_contradictions_do_not_reject_optional_branches(required, expected_failure):
    schema = {
        "type": "object",
        "required": required,
        "properties": {
            "materials": {
                "type": "object",
                "required": ["list", "supplierInfo"],
                "properties": {"list": {"type": "array"}},
                "additionalProperties": False,
            }
        },
        "additionalProperties": False,
    }
    task = structured_output.normalize(
        RawRow(
            "schema-conflict",
            Source(dataset="fixture", revision="1", row="0", importer_revision="1"),
            {
                "instruction": "Generate any schema-valid JSON instance. " + json.dumps(schema),
                "verifier_data": {"schema_type": "json", "schema": schema},
            },
        )
    )
    assert isinstance(task, TaskSpec)
    report = structured_output.verification_report(task)
    assert any(check.status.value == "fail" for check in report.checks) == expected_failure
    if not expected_failure:
        assert all(check.status.value == "pass" for check in verify_witness(task, "{}", "null"))
