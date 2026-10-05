# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import base64
import json
from dataclasses import replace

import pytest

from taskcompendium.grader import grader_config
from taskcompendium.grading import grade_task
from taskcompendium.grading_result import Outcome
from taskcompendium.models import ConversationTrace, Source, TaskSpec, TextMessage
from taskcompendium.pipeline.datasets import calendar_tasks, reasoning_tasks
from taskcompendium.pipeline.models import CheckStatus, RawRow
from taskcompendium.submission import AnswerFormat, SubmissionConvention


@pytest.fixture
def row_source():
    return Source(dataset="test/tasks", revision="1", row="0", importer_revision="1")


def encoded_file(value):
    return base64.b64encode(json.dumps(value).encode()).decode()


def answer_grade(task, answer):
    return grade_task(
        task,
        SubmissionConvention(id="plain", answer_format=AnswerFormat.PLAIN),
        ConversationTrace(events=(*task.context.events, TextMessage(role="assistant", content=answer))),
    )


def test_puzzle_ingestion_preserves_order_and_symbolic_coordinates(row_source):
    ordered = reasoning_tasks.normalize_puzzle(
        RawRow(
            "ordered",
            row_source,
            {
                "instruction": "Sort in ASCII order: chair, Defect, donate, Salt. Return a comma-separated list.",
                "files": {
                    "tests/gold.json": encoded_file(
                        {"gold": "Defect, Salt, chair, donate", "answer_type": "ordered_list"}
                    )
                },
            },
        )
    )
    assert isinstance(ordered, TaskSpec)
    assert answer_grade(ordered, "Defect\nSalt\nchair\ndonate").reward == 1.0
    assert answer_grade(ordered, "chair, Defect, donate, Salt").reward == 0.0
    coordinates = reasoning_tasks.normalize_puzzle(
        RawRow(
            "coords",
            row_source,
            {
                "instruction": "Return the point halfway between (1, 2) and (3, 4).",
                "files": {"tests/gold.json": encoded_file({"gold": "(2, 3)", "answer_type": "coords"})},
            },
        )
    )
    assert isinstance(coordinates, TaskSpec)
    assert answer_grade(coordinates, "(2.0, 3.0)").reward == 1.0
    assert answer_grade(coordinates, "(3, 2)").reward == 0.0


def test_reasoning_ingestion_preserves_upstream_contract_without_local_execution(row_source):
    task = reasoning_tasks.normalize_reasoning(
        RawRow(
            "equation",
            row_source,
            {
                "instruction": "Solve x + 8 = 50. Reply with x.",
                "verifier_data": {"answer": "42", "metadata": {"source_dataset": "simple_equations"}},
            },
        )
    )
    assert isinstance(task, TaskSpec)
    assert grader_config(task)["contract"]["entry"]["answer"] == "42"
    assert answer_grade(task, "42").status == Outcome.INFRA_ERROR
    assert all(check.status == CheckStatus.UNSUPPORTED for check in reasoning_tasks.reasoning_checks(task).checks)


def test_calendar_ingestion_accepts_alternatives_rejects_overlap_and_keeps_missing_witness_pending(row_source):
    expected = {
        "0": {
            "event_name": "Review",
            "duration": 30,
            "min_time": "10:00",
            "max_time": "16:00",
            "constraint": "before 11am",
        },
        "1": {"event_name": "Plan", "duration": 30, "min_time": "10:00", "max_time": "16:00", "constraint": None},
    }
    witness = [
        {"event_id": 0, "event_name": "Review", "duration": 30, "start_time": "10:00"},
        {"event_id": 1, "event_name": "Plan", "duration": 30, "start_time": "11:00"},
    ]
    row = RawRow(
        "calendar",
        row_source,
        {
            "instruction": (
                "Schedule Review for 30 minutes before 11am and Plan for 30 minutes. Working hours: 10am-4pm."
            ),
            "verifier_data": {"expected_events": expected},
            "files": {"solution/answer.json": encoded_file(witness)},
        },
    )
    task = calendar_tasks.normalize(row)
    assert isinstance(task, TaskSpec)
    assert all(check.status == CheckStatus.PASS for check in calendar_tasks.verification_report(task).checks)
    witness[0]["start_time"] = "10:15"
    witness[1]["start_time"] = "12:00"
    assert answer_grade(task, json.dumps(witness)).reward == 1.0
    witness[1]["start_time"] = "10:30"
    grade = answer_grade(task, json.dumps(witness))
    assert grade.status == Outcome.GRADED and grade.reward == 0.0
    missing_witness = calendar_tasks.normalize(replace(row, data={**row.data, "files": {}}))
    assert isinstance(missing_witness, TaskSpec)
    assert [check.status for check in calendar_tasks.verification_report(missing_witness).checks] == [
        CheckStatus.UNSUPPORTED
    ]
