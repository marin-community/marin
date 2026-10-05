# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Instruction repairs retain grader behavior, originals, and auditable lineage."""

import json
from dataclasses import dataclass, field

import pyarrow.parquet as pq
import pytest
from zephyr.writers import write_parquet_file

from taskcompendium.models import Source, TaskSpec
from taskcompendium.pipeline.audit_schema import TASK_SCHEMA, audit_columns
from taskcompendium.pipeline.datasets import instruction_following, structured_output
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
    TaskAudit,
)
from taskcompendium.pipeline.rewriting import (
    BatchRewriter,
    protected_text_checks,
    rewrite_records,
)
from taskcompendium.pipeline.stages import rewrite_audit_source
from taskcompendium.pipeline.verification import verify_witness

from .pipeline_stages import fixture_recipe


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


def test_rewrite_retry_resubmits_batch_and_retains_lineage(tmp_path, structured_task):
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
        rewriter.rewrite([structured_task], rubric, tmp_path / "failed-attempt")
    result = rewriter.rewrite([structured_task], rubric, tmp_path)
    assert len(service.batches) == 2
    assert json.loads((tmp_path / "failed-attempt/batch-submission.json").read_text())["batch_id"] == "batch-0"
    assert json.loads((tmp_path / "batch-submission.json").read_text())["batch_id"] == "batch-1"
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
    assert result.candidates == [candidate]
    assert result.lineage[0].model_dump(mode="json", exclude_none=True) == lineage
    assert result.records[0].proposal is not None
    assert result.records[0].proposal.edits[0].old_text == old_text


@pytest.mark.parametrize("action", ["unchanged", "unrepairable"])
def test_rewriter_retains_non_rewritten_tasks_without_candidates(tmp_path, structured_task, action):
    service = RewriteService([rewrite_response(structured_task.id, action, [])], interrupted=False)
    records = BatchRewriter(service, "model", "deployment").rewrite(
        [structured_task], ReviewRubric("repair", "1", ()), tmp_path
    )
    assert records.records[0].proposal is not None
    assert records.records[0].proposal.action.value == action
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
    original_schema = next(
        resource.content.decode() for resource in structured_task.verifier.files if resource.path == "/tests/schema.json"
    )
    checks = protected_text_checks(candidate, {"public_schema": original_schema})
    assert [(check.check, check.status.value) for check in checks] == [("preserve:public_schema", "fail")]


def test_rewrite_stage_rechecks_candidate_and_retains_original_audit(tmp_path, structured_task):
    original = TaskAudit(
        task_id=structured_task.id,
        source=structured_task.source,
        raw={"data": {"instruction": structured_task.context.events[0].content}},
        normalized=structured_task,
        normalization_rejection=None,
        checks=[],
        review=None,
        decision=Decision(task_id=structured_task.id, disposition=Disposition.REJECT, reasons=["review:bad"]),
    )
    source = tmp_path / "filtered"
    (source / "audit").mkdir(parents=True)
    write_parquet_file([audit_columns(original)], str(source / "audit/part-00000.parquet"), schema=TASK_SCHEMA)
    (source / "manifest.json").write_text(json.dumps({"input_rows": 1}))
    service = RewriteService(
        [
            rewrite_response(
                structured_task.id,
                "rewrite",
                [
                    {
                        "old_text": "Parse the document and recover all values.",
                        "replacement": "Generate a schema-valid instance.",
                    }
                ],
            )
        ],
        interrupted=False,
    )

    class CandidateReviewer:
        calls = 0

        @property
        def identity(self):
            return {"model": "fixture"}

        def review(self, tasks, rubric, output_path, *, originals=None):
            self.calls += 1
            if self.calls > 1:
                raise AssertionError("Completed rewrite shard called the reviewer again")
            assert originals is not None
            assert {task.id for task in tasks} == set(originals)
            return [
                ReviewRecord(
                    task_id=task.id,
                    status=ReviewStatus.REVIEWED,
                    verdict=ReviewVerdict(
                        task_id=task.id,
                        quality=Quality.GOOD,
                        confidence=Confidence.HIGH,
                        reference_status=ReferenceStatus.CONSISTENT,
                        defects=[],
                        evidence="The repaired instruction preserves the schema contract.",
                    ),
                    detail="",
                )
                for task in tasks
            ]

    output = tmp_path / "rewritten"
    reviewer = CandidateReviewer()
    manifest = rewrite_audit_source(
        str(source),
        str(output),
        fixture_recipe(structured_output.pipeline()),
        FilterPolicy(),
        ReviewRubric("repair", "1", ("Preserve the schema.",)),
        BatchRewriter(service, "model", "deployment"),
        reviewer,
        (structured_task.id,),
        review_batch_size=1,
        max_workers=1,
    )
    row = pq.read_table(output / "audit/part-00000.parquet").to_pylist()[0]
    assert manifest["dispositions"] == {"keep": 1}
    assert row["parent_id"] == structured_task.id
    assert row["task_id"] != structured_task.id
    assert row["checks"] and all(check["status"] != "fail" for check in row["checks"])
    lineage = json.loads(row["cleanup_lineage_json"])
    assert lineage["original_audit"]["filter_reasons"] == ["review:bad"]
    assert pq.read_table(output / "accepted/part-00000.parquet").num_rows == 1
    submitted = len(service.batches)
    service.interrupted = True
    repeated = rewrite_audit_source(
        str(source),
        str(output),
        fixture_recipe(structured_output.pipeline()),
        FilterPolicy(),
        ReviewRubric("repair", "1", ("Preserve the schema.",)),
        BatchRewriter(service, "model", "deployment"),
        reviewer,
        (structured_task.id,),
        review_batch_size=1,
        max_workers=1,
    )
    assert repeated == manifest
    assert pq.read_table(output / "audit/part-00000.parquet").to_pylist()[0] == row
    assert len(service.batches) == submitted
    assert reviewer.calls == 1


def test_rewrite_stage_rejects_candidate_with_unavailable_review(tmp_path, structured_task):
    original = TaskAudit(
        task_id=structured_task.id,
        source=structured_task.source,
        raw={"data": {"instruction": structured_task.context.events[0].content}},
        normalized=structured_task,
        normalization_rejection=None,
        checks=[],
        review=None,
        decision=Decision(task_id=structured_task.id, disposition=Disposition.REJECT, reasons=["review:bad"]),
    )
    source = tmp_path / "filtered"
    write_parquet_file([audit_columns(original)], str(source / "audit/part-00000.parquet"), schema=TASK_SCHEMA)
    (source / "manifest.json").write_text(json.dumps({"input_rows": 1}))
    service = RewriteService(
        [
            rewrite_response(
                structured_task.id,
                "rewrite",
                [
                    {
                        "old_text": "Parse the document and recover all values.",
                        "replacement": "Generate a schema-valid instance.",
                    }
                ],
            )
        ],
        interrupted=False,
    )

    class UnavailableReviewer:
        @property
        def identity(self):
            return {"model": "fixture"}

        def review(self, tasks, rubric, output_path, *, originals=None):
            assert originals is not None and set(originals) == {task.id for task in tasks}
            return [
                ReviewRecord(task_id=task.id, status=ReviewStatus.UNAVAILABLE, verdict=None, detail="No review")
                for task in tasks
            ]

    output = tmp_path / "rewritten"
    manifest = rewrite_audit_source(
        str(source),
        str(output),
        fixture_recipe(structured_output.pipeline()),
        FilterPolicy(),
        ReviewRubric("repair", "1", ()),
        BatchRewriter(service, "model", "deployment"),
        UnavailableReviewer(),
        (structured_task.id,),
        review_batch_size=1,
        max_workers=1,
    )
    row = pq.read_table(output / "audit/part-00000.parquet").to_pylist()[0]
    assert row["task_id"] != structured_task.id
    assert row["filter_status"] == "reject"
    assert row["review_status"] == "unavailable"
    assert manifest["rewritten_rows"] == 1 and manifest["dispositions"] == {"reject": 1}
    assert json.loads(row["cleanup_lineage_json"])["original_audit"]["filter_reasons"] == ["review:bad"]


@pytest.mark.parametrize("action", ["unchanged", "unrepairable"])
def test_rewrite_stage_preserves_original_decisions_without_a_candidate(tmp_path, structured_task, action):
    selected = TaskAudit(
        task_id=structured_task.id,
        source=structured_task.source,
        raw={"data": {"instruction": structured_task.context.events[0].content}},
        normalized=structured_task,
        normalization_rejection=None,
        checks=[],
        review=None,
        decision=Decision(task_id=structured_task.id, disposition=Disposition.REJECT, reasons=["review:bad"]),
    )
    other_task = structured_task.model_copy(update={"id": "unselected-task"})
    unselected = selected.model_copy(
        update={
            "task_id": other_task.id,
            "normalized": other_task,
            "decision": Decision(task_id=other_task.id, disposition=Disposition.KEEP, reasons=[]),
        }
    )
    source = tmp_path / "filtered"
    write_parquet_file(
        [audit_columns(selected), audit_columns(unselected)],
        str(source / "audit/part-00000.parquet"),
        schema=TASK_SCHEMA,
    )
    (source / "manifest.json").write_text(json.dumps({"input_rows": 2}))
    service = RewriteService([rewrite_response(structured_task.id, action, [])], interrupted=False)

    class UnusedReviewer:
        @property
        def identity(self):
            return {"model": "fixture"}

        def review(self, tasks, rubric, output_path, *, originals=None):
            raise AssertionError("No candidate should be reviewed")

    output = tmp_path / "rewritten"
    manifest = rewrite_audit_source(
        str(source),
        str(output),
        fixture_recipe(structured_output.pipeline()),
        FilterPolicy(),
        ReviewRubric("repair", "1", ()),
        BatchRewriter(service, "model", "deployment"),
        UnusedReviewer(),
        (structured_task.id,),
        review_batch_size=1,
        max_workers=1,
    )
    rows = {row["task_id"]: row for row in pq.read_table(output / "audit/part-00000.parquet").to_pylist()}
    assert manifest["input_rows"] == 2 and manifest["rewritten_rows"] == 0
    assert rows[structured_task.id]["cleanup_action"] == action
    assert rows[structured_task.id]["filter_status"] == "reject"
    assert rows[structured_task.id]["filter_reasons"] == ["review:bad"]
    assert rows[other_task.id]["parent_id"] is None
    assert rows[other_task.id]["filter_status"] == "keep"


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
    assert records.records[0].status == ReviewStatus.INVALID
    assert records.records[0].proposal is None
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
