# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from experiments.post_training.tasktrove.mcqa_routing import (
    ROUTE_MAPPINGS_FILENAME,
    AnswerStatus,
    Confidence,
    Defect,
    EvidenceSource,
    GlmDecision,
    OperationType,
    Route,
    RoutingTask,
    Subject,
    batch_lines,
    final_route,
    parse_batch_output,
    read_batch_output,
    select_tasks,
)
from experiments.post_training.tasktrove.mcqa_routing_pipeline import (
    RoutingArtifactConfig,
    aggregate_worker_outputs,
)


def _task(index: int) -> RoutingTask:
    return RoutingTask(
        task_id=f"task-{index:03d}",
        question=f"What is {index} + 1?",
        options={"A": str(index), "B": str(index + 1)},
        expected="B",
        mechanical_flags=(),
    )


def test_select_tasks_partitions_ledger_and_returns_exact_sample(tmp_path):
    rows = []
    for index in range(40):
        rejected = index in {3, 17}
        rows.append(
            {
                "path": f"task-{index:03d}",
                "status": "reject" if rejected else "keep",
                "reasons": ["duplicate_options"] if rejected else [],
                "flags": [],
                "question": f"What is {index} + 1?",
                "expected": "B",
                "options_json": json.dumps({"A": str(index), "B": str(index + 1)}),
            }
        )
    path = tmp_path / "ledger.parquet"
    pq.write_table(pa.Table.from_pylist(rows), path, row_group_size=5)

    worker_0, summary_0 = select_tasks(
        str(path), source="unused", worker_index=0, worker_count=2, sample_size=5, sample_seed="sample"
    )
    worker_1, summary_1 = select_tasks(
        str(path), source="unused", worker_index=1, worker_count=2, sample_size=5, sample_seed="sample"
    )

    selected = {task.task_id for task in worker_0 + worker_1}
    assert len(selected) == 10
    assert "task-003" not in selected
    assert "task-017" not in selected
    assert summary_0["assigned_row_groups"] == [0, 2, 4, 6]
    assert summary_1["assigned_row_groups"] == [1, 3, 5, 7]
    assert summary_0["source_rows"] + summary_1["source_rows"] == 40
    assert summary_0["mechanical_rejects"] + summary_1["mechanical_rejects"] == 2

    empty, _ = select_tasks(
        str(path), source="unused", worker_index=0, worker_count=2, sample_size=0, sample_seed="sample"
    )
    assert empty == []


def test_batch_lines_pack_requested_number_of_tasks():
    lines, by_custom_id = batch_lines([_task(index) for index in range(41)], batch_size=20, worker_index=2)

    assert [len(by_custom_id[line["custom_id"]]) for line in lines] == [20, 20, 1]
    assert [line["custom_id"] for line in lines] == [
        "worker-002-batch-00000",
        "worker-002-batch-00001",
        "worker-002-batch-00002",
    ]
    first_body = lines[0]["body"]
    function = first_body["tools"][0]["function"]
    results_schema = function["parameters"]["properties"]["results"]
    assert function["strict"] is True
    assert first_body["parallel_tool_calls"] is False
    assert first_body["max_tokens"] == 12000
    assert (results_schema["minItems"], results_schema["maxItems"]) == (20, 20)
    assert results_schema["items"]["properties"]["id"]["enum"] == list(range(20))


def test_parse_batch_output_expands_valid_tool_results():
    tasks = [_task(0), _task(1)]
    results = [
        {
            "id": 0,
            "r": "rl",
            "c": "high",
            "choice": "B",
            "answer": "match",
            "op": "chain",
            "evidence": "prompt",
            "defect": "none",
            "subject": "math",
            "check": "Add one to the supplied integer.",
        },
        {
            "id": 1,
            "r": "sft",
            "c": "medium",
            "choice": "B",
            "answer": "match",
            "op": "single",
            "evidence": "prompt",
            "defect": "none",
            "subject": "math",
            "check": "One direct addition selects B.",
        },
    ]
    body = {
        "choices": [
            {
                "message": {
                    "tool_calls": [
                        {
                            "function": {
                                "name": "submit_routes",
                                "arguments": json.dumps({"results": results}),
                            }
                        }
                    ]
                }
            }
        ]
    }
    output = json.dumps(
        {
            "custom_id": "worker-000-batch-00000",
            "response": {"status_code": 200, "body": body},
            "error": None,
        }
    )

    routed, failed = parse_batch_output(output, {"worker-000-batch-00000": tasks})

    assert failed == []
    assert [(row["task_id"], row["route"]) for row in routed] == [("task-000", "rl"), ("task-001", "sft")]
    assert routed[0]["model_route"] == "rl"
    assert routed[0]["reason_codes"] == [
        "model_route:rl",
        "confidence:high",
        "answer:match",
        "operation:chain",
        "evidence:prompt",
        "defect:none",
    ]


def test_parse_batch_output_falls_back_only_missing_rows():
    tasks = [_task(0), _task(1)]
    result = {
        "id": 0,
        "r": "rl",
        "c": "high",
        "choice": "B",
        "answer": "match",
        "op": "chain",
        "evidence": "prompt",
        "defect": "none",
        "subject": "math",
        "check": "Add one to the supplied integer.",
    }
    body = {
        "choices": [
            {
                "message": {
                    "tool_calls": [
                        {
                            "function": {
                                "name": "submit_routes",
                                "arguments": json.dumps({"results": [result]}),
                            }
                        }
                    ]
                }
            }
        ]
    }
    output = json.dumps(
        {
            "custom_id": "worker-000-batch-00000",
            "response": {"status_code": 200, "body": body},
            "error": None,
        }
    )

    routed, degraded = parse_batch_output(output, {"worker-000-batch-00000": tasks})

    assert degraded == ["worker-000-batch-00000"]
    assert [(row["task_id"], row["route"], row["classification_status"]) for row in routed] == [
        ("task-000", "rl", "classified"),
        ("task-001", "sft", "fallback"),
    ]


def test_parse_batch_output_falls_back_for_missing_request():
    tasks = [_task(0), _task(1)]

    routed, degraded = parse_batch_output("", {"worker-000-batch-00000": tasks})

    assert degraded == ["worker-000-batch-00000"]
    assert [(row["task_id"], row["route"], row["classification_status"]) for row in routed] == [
        ("task-000", "sft", "fallback"),
        ("task-001", "sft", "fallback"),
    ]


def test_read_batch_output_allows_failed_batch_without_output():
    output, errors = read_batch_output("unused", "unused", {"id": "batch-1", "status": "failed"})

    assert output == ""
    assert errors is None


def test_aggregate_worker_outputs_writes_complete_sorted_ledger(tmp_path):
    config = RoutingArtifactConfig(
        input_path="input.parquet",
        output_path=str(tmp_path),
        source="source",
        git_revision="abc123",
        sample_size=2,
        sample_seed="sample",
        request_batch_size=20,
        relay_job="/relay",
        poll_seconds=10,
        worker_count=2,
        worker_cpu=2,
        worker_ram="4g",
        worker_disk="8g",
    )
    rows = [
        {
            "task_id": "task-b",
            "route": "sft",
            "route_source": "best-effort-fallback",
            "policy_version": "v1",
            "reason_codes": ["fallback:invalid_row"],
            "subject": "other",
        },
        {
            "task_id": "task-a",
            "route": "rl",
            "route_source": "glm-5.3",
            "policy_version": "v1",
            "reason_codes": ["model_route:rl"],
            "subject": "math",
        },
    ]
    for worker_index, row in enumerate(rows):
        worker_root = tmp_path / f"worker-{worker_index:03d}"
        worker_root.mkdir()
        (worker_root / "decisions.jsonl").write_text(json.dumps(row) + "\n")
        mapping = {key: row[key] for key in ("task_id", "route", "route_source", "policy_version", "reason_codes")}
        (worker_root / ROUTE_MAPPINGS_FILENAME).write_text(json.dumps(mapping) + "\n")
        (worker_root / "summary.json").write_text(
            json.dumps(
                {
                    "requests": 1,
                    "degraded_requests": int(row["route_source"] == "best-effort-fallback"),
                    "fallback_rows": int(row["route_source"] == "best-effort-fallback"),
                }
            )
        )

    summary = aggregate_worker_outputs(config)

    assert summary["routed_rows"] == 2
    assert summary["route_counts"] == {"sft": 1, "rl": 1}
    assert summary["fallback_rows"] == 1
    assert [json.loads(line)["task_id"] for line in (tmp_path / ROUTE_MAPPINGS_FILENAME).read_text().splitlines()] == [
        "task-a",
        "task-b",
    ]


@pytest.mark.parametrize(
    ("model_route", "confidence", "choice", "answer", "operation", "evidence", "defect", "expected"),
    [
        ("rl", "high", "B", "match", "chain", "prompt", "none", Route.RL),
        ("rl", "medium", "B", "match", "chain", "prompt", "none", Route.SFT),
        ("rl", "high", "A", "match", "chain", "prompt", "none", Route.GARBAGE),
        ("sft", "high", "B", "match", "single", "prompt", "wrong_key", Route.GARBAGE),
        ("garbage", "high", "B", "match", "chain", "prompt", "none", Route.GARBAGE),
    ],
)
def test_final_route_fails_closed(model_route, confidence, choice, answer, operation, evidence, defect, expected):
    decision = GlmDecision(
        row_id=0,
        route=Route(model_route),
        confidence=Confidence(confidence),
        derived_choice=choice,
        answer_status=AnswerStatus(answer),
        operation_type=OperationType(operation),
        evidence_source=EvidenceSource(evidence),
        defect=Defect(defect),
        subject=Subject.MATH,
        check="Check.",
    )

    assert final_route(_task(0), decision) is expected
