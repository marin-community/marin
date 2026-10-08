# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Answer conversions reject unusable rows and grade replies against the row's reference in process."""

import json

import pytest
from verifyit.spec import SchemaFormat

from taskcompendium.convert.answers import (
    EVIDENCE_PATH,
    exact_answer_task,
    ifeval_task,
    json_schema_task,
    math_answer_task,
    mcq_task,
    numeric_answer_task,
)
from taskcompendium.grading_result import Outcome
from taskcompendium.models import ConversationTrace, GradingAttempt, Source, TaskSpec
from taskcompendium.pipeline.models import ImportFailureKind, ImportRejection, RawRow
from taskcompendium.pipeline.verification import answer_event
from taskcompendium.runtime.resources import resource_bytes
from taskcompendium.runtime.task_grading import grade_task

ROW = RawRow("task", Source(dataset="fixture", revision="1", row="0", importer_revision="1"), {})
DEFECT, UNSUPPORTED = ImportFailureKind.SOURCE_DEFECT, ImportFailureKind.UNSUPPORTED
COUNT_SCHEMA = json.dumps({"type": "object", "required": ["count"], "properties": {"count": {"type": "integer"}}})


def reward(task: TaskSpec | ImportRejection, answer: str) -> float | None:
    assert isinstance(task, TaskSpec)
    result = grade_task(
        task, GradingAttempt(ConversationTrace(events=(*task.context.events, answer_event(task, answer))))
    )
    assert result.status in {Outcome.GRADED, Outcome.SUBMISSION_FAILURE}, result.error
    return result.reward


@pytest.mark.parametrize(
    "converted,kind,reason",
    [
        (math_answer_task(ROW, prompt=" ", answer="1"), DEFECT, "missing_prompt"),
        (math_answer_task(ROW, prompt="Add.", answer=""), DEFECT, "invalid_reference"),
        (
            numeric_answer_task(ROW, prompt="Add.", answer="seven", tolerance_abs=0, tolerance_rel=0),
            DEFECT,
            "invalid_reference",
        ),
        (
            numeric_answer_task(ROW, prompt="Add.", answer=None, tolerance_abs=0, tolerance_rel=0),
            DEFECT,
            "invalid_reference",
        ),
        (mcq_task(ROW, prompt="Pick.", answer="E", options=4), DEFECT, "invalid_reference"),
        (mcq_task(ROW, prompt="Pick.", answer="AB", options=4), DEFECT, "invalid_reference"),
        (exact_answer_task(ROW, prompt="Name it.", answers=(), ignore_case=False), DEFECT, "invalid_reference"),
        (
            exact_answer_task(ROW, prompt="Name it.", answers=("Paris", " "), ignore_case=False),
            DEFECT,
            "invalid_reference",
        ),
        (ifeval_task(ROW, prompt="Write.", constraints=()), UNSUPPORTED, "invalid_constraints"),
        (
            json_schema_task(ROW, prompt="Return JSON.", schema="", schema_format=SchemaFormat.JSON),
            DEFECT,
            "invalid_schema",
        ),
    ],
    ids=[
        "math_blank_prompt",
        "math_blank_reference",
        "numeric_word_reference",
        "numeric_missing_reference",
        "mcq_key_beyond_options",
        "mcq_two_letter_key",
        "exact_no_answers",
        "exact_blank_answer",
        "ifeval_no_constraints",
        "schema_blank",
    ],
)
def test_unusable_rows_are_rejected_with_the_defect_named(converted, kind, reason):
    assert isinstance(converted, ImportRejection)
    assert (converted.kind, converted.reason) == (kind, reason)


@pytest.mark.parametrize(
    "task,answer,expected",
    [
        (math_answer_task(ROW, prompt="Half of one?", answer=r"\boxed{\frac{1}{2}}"), r"\boxed{\frac{1}{2}}", 1.0),
        (math_answer_task(ROW, prompt="Half of one?", answer=r"\boxed{\frac{1}{2}}"), r"\boxed{\frac{1}{3}}", 0.0),
        # An ordered pair stays a tuple, so swapping its components is wrong.
        (math_answer_task(ROW, prompt="Solve.", answer="(1, 2)"), r"\boxed{(1, 2)}", 1.0),
        (math_answer_task(ROW, prompt="Solve.", answer="(1, 2)"), r"\boxed{(2, 1)}", 0.0),
        # Integers beyond float precision compare exactly.
        (
            numeric_answer_task(ROW, prompt="Count.", answer="12345678901234567891", tolerance_abs=0, tolerance_rel=0),
            "12345678901234567891",
            1.0,
        ),
        (
            numeric_answer_task(ROW, prompt="Count.", answer="12345678901234567891", tolerance_abs=0, tolerance_rel=0),
            "12345678901234567890",
            0.0,
        ),
        (numeric_answer_task(ROW, prompt="Count.", answer=7, tolerance_abs=0, tolerance_rel=0), "7", 1.0),
        (mcq_task(ROW, prompt="Which? A. x B. y C. z D. w", answer="c", options=4), "C", 1.0),
        (mcq_task(ROW, prompt="Which? A. x B. y C. z D. w", answer="c", options=4), "B", 0.0),
        (exact_answer_task(ROW, prompt="Capital?", answers=("Paris",), ignore_case=True), "paris", 1.0),
        (exact_answer_task(ROW, prompt="Capital?", answers=("Paris",), ignore_case=False), "paris", 0.0),
        (
            json_schema_task(ROW, prompt="Return a count.", schema=COUNT_SCHEMA, schema_format=SchemaFormat.JSON),
            '{"count": 3}',
            1.0,
        ),
        (
            json_schema_task(ROW, prompt="Return a count.", schema=COUNT_SCHEMA, schema_format=SchemaFormat.JSON),
            '{"count": "three"}',
            0.0,
        ),
    ],
    ids=[
        "math_boxed_reference",
        "math_wrong_value",
        "math_tuple",
        "math_swapped_tuple",
        "numeric_big_integer",
        "numeric_big_integer_off_by_one",
        "numeric_integer_reference",
        "mcq_lowercase_key",
        "mcq_wrong_letter",
        "exact_ignore_case",
        "exact_case_sensitive",
        "schema_valid",
        "schema_invalid",
    ],
)
def test_answer_tasks_grade_replies_against_the_row_reference(task, answer, expected):
    assert reward(task, answer) == expected


def test_evidence_stays_with_the_grader_and_out_of_the_solver_view():
    evidence = {"Equation": "( 3.0 + 4.0 )", "Type": "Addition"}
    task = numeric_answer_task(
        ROW,
        prompt="Aya has 3 apples and finds 4. How many?",
        answer="7",
        tolerance_abs=0,
        tolerance_rel=0,
        evidence=evidence,
    )
    assert isinstance(task, TaskSpec)
    (stored,) = [resource for resource in task.resources.verifier if resource.path == EVIDENCE_PATH]
    assert json.loads(resource_bytes(stored)) == evidence
    assert task.resources.worker == task.resources.oracle == ()
    assert reward(task, "7") == 1.0
