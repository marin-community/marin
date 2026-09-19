# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Contracts for TaskTrove math and GSM8K importers."""

import json
from pathlib import Path

import pytest

from taskcompendium.grading import grade_attempt
from taskcompendium.importers.gsm8k import import_row
from taskcompendium.importers.tasktrove import read_archive
from taskcompendium.importers.tasktrove_math import import_task
from taskcompendium.models import (
    AssistantFinal,
    BoxedLatex,
    FileSubmission,
    JsonPath,
    Outcome,
    Rejected,
    Rendering,
    ResourceRole,
    Source,
)

ROOT = Path(__file__).parent
MATH_FIXTURES = ROOT / "fixtures/tasktrove/math"
GSM8K = json.loads((ROOT / "fixtures/gsm8k.json").read_text())


def _math(row: str):
    return import_task(read_archive((MATH_FIXTURES / f"math-row-{row}.tar.gz").read_bytes(), row, "math-answer"))


def test_tasktrove_math_preserves_problem_and_provenance():
    result = _math("114")

    assert not isinstance(result, Rejected)
    assert result.requirements.capabilities == ()
    assert "orthocenter" in result.steps[0].instructions
    assert result.metadata.source.row == "114"
    assert result.steps[0].verifier.parameters["expected"] == "(-5.167, 4.693)"


@pytest.mark.parametrize("answer", ["", "working only", "#### 1\n#### 2", "#### "])
def test_gsm8k_rejects_missing_or_ambiguous_gold(answer):
    source = Source(GSM8K["dataset"], GSM8K["revision"], "0", "test")

    result = import_row("A question", answer, source)

    assert isinstance(result, Rejected)


def test_gsm8k_keeps_reasoning_private_and_uses_delimited_gold():
    row = GSM8K["rows"][0]["data"]
    source = Source(GSM8K["dataset"], GSM8K["revision"], "0", "test")

    result = import_row(row["question"], row["answer"], source)

    assert not isinstance(result, Rejected)
    assert "####" not in result.steps[0].instructions
    assert result.steps[0].verifier.parameters["expected"] == "72"
    assert result.resources[0].roles == (ResourceRole.ORACLE,)


@pytest.mark.parametrize(
    ("protocol", "response"),
    [
        (Rendering("plain", AssistantFinal()), "246"),
        (Rendering("boxed", AssistantFinal(BoxedLatex())), r"\\boxed{246}"),
        (Rendering("json", AssistantFinal(JsonPath())), '{"answer":"246"}'),
        (Rendering("file", FileSubmission("/app/answer.txt")), None),
    ],
)
def test_math_grading_accepts_explicit_submission_wrappers(protocol, response, tmp_path):
    specification = _math("115")
    assert not isinstance(specification, Rejected)
    if response is None:
        (tmp_path / "answer.txt").write_text("246")

    result = grade_attempt(specification, protocol, response, tmp_path)

    assert result.status is Outcome.GRADED and result.reward == 1.0


@pytest.mark.parametrize("row,bad", [("114", "(0, 0)"), ("115", "247")])
def test_real_math_tasks_reject_wrong_and_empty_submissions(row, bad, tmp_path):
    specification = _math(row)
    assert not isinstance(specification, Rejected)
    protocol = Rendering("plain", AssistantFinal())
    assert grade_attempt(specification, protocol, bad, tmp_path).reward == 0.0
    empty = grade_attempt(specification, protocol, "", tmp_path)
    assert empty.status == Outcome.EXTRACTION_ERROR and empty.reward is None


def test_gsm8k_rejects_gold_before_trailing_solution_text():
    source = Source(GSM8K["dataset"], GSM8K["revision"], "bad", "test")
    result = import_row("Compute 1 + 1.", "#### 2\nActually I changed my answer", source)
    assert isinstance(result, Rejected)


def test_math_does_not_treat_numeric_overlap_as_gold_leak():
    archive = read_archive((MATH_FIXTURES / "math-row-115.tar.gz").read_bytes(), "115", "math-answer")
    instruction = archive.instructions.replace(
        "## Problem Statement", "## Problem Statement\n\nA value given in this problem is 246."
    )
    archive.files["instruction.md"] = instruction.encode()
    result = import_task(archive)
    assert not isinstance(result, Rejected)
    assert "A value given in this problem is 246." in result.steps[0].instructions


def test_math_import_rejects_a_choice_task_with_a_math_verifier():
    archive = read_archive((MATH_FIXTURES / "math-row-115.tar.gz").read_bytes(), "115", "math-answer")
    archive.files["task.toml"] = archive.files["task.toml"].replace(b'answer_type = "number"', b'answer_type = "choice"')
    archive.files["tests/verifier.toml"] = archive.files["tests/verifier.toml"].replace(
        b'expected = "246"', b'expected = "on"'
    )
    result = import_task(archive)
    assert isinstance(result, Rejected)
    assert result.reason.value == "broken_grader"


def test_import_rejects_an_archive_from_a_different_verifier_revision():
    archive = read_archive((MATH_FIXTURES / "math-row-115.tar.gz").read_bytes(), "115", "math-answer")
    archive.files["environment/Dockerfile"] = archive.files["environment/Dockerfile"].replace(
        b"b2b68d8b0a770cdc0ab3903780172c4b3eea81b1", b"b76d03131cd88bd9fc711dba206659027edba3a8"
    )
    result = import_task(archive)
    assert isinstance(result, Rejected)
    assert result.reason.value == "unrecoverable_source"
