# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
from typing import cast

import pytest
from taskcompendium.grader import grader_config
from taskcompendium.models import AnswerType, ScriptGrader
from taskcompendium.runtime.resources import resource_bytes

from experiments.post_training.task_curation.datasets import (
    ai2_qasper,
    ai2_qasper_verifiable,
    ai2_qasper_verifiable_grade,
)
from experiments.post_training.task_curation.pipeline import CurationRecipe
from experiments.post_training.task_curation.tests.conversion import converted_task

RECIPE = cast(CurationRecipe, ai2_qasper_verifiable.sources()[0].config)
ANSWERS = {
    "answer": [
        {
            "unanswerable": False,
            "extractive_spans": [],
            "yes_no": True,
            "free_form_answer": "",
            "evidence": ["The paper reports the result."],
            "highlighted_evidence": ["reports the result"],
        }
    ],
    "annotation_id": ["annotation-1"],
    "worker_id": ["worker-1"],
}
ROW = {
    "paper": {
        "id": "paper-1",
        "title": "A Paper",
        "abstract": "An abstract.",
        "full_text": {"section_name": ["Results"], "paragraphs": [["The paper reports the result."]]},
        "figures_and_tables": {"caption": [], "file": []},
    },
    "question": "Does the paper report the result?",
    "question_id": "question-1",
    "answers": ANSWERS,
    "expected": "yes",
}


def test_converts_agreed_question_to_file_task():
    task = converted_task(RECIPE, ROW)
    paper = json.loads(resource_bytes(task.resources.worker[0]))
    config = grader_config(task)

    assert task.answer_type == AnswerType.FILE
    assert task.output_paths == (ai2_qasper_verifiable.ANSWER_PATH,)
    assert task.environment_requirements.working_directory == "/app"
    assert paper == ROW["paper"]
    assert config["expected"] == "yes"
    assert isinstance(task.grader, ScriptGrader) and task.grader.answer_path is None
    prompt = task.context.events[0].content
    assert isinstance(prompt, str)
    assert "/app/paper.json" in prompt and "/app/answer.json" in prompt
    assert ROW["question"] in prompt


@pytest.mark.parametrize(
    ("annotations", "expected"),
    [
        ([{"unanswerable": False, "yes_no": True}] * 2, "yes"),
        ([{"unanswerable": False, "yes_no": False}] * 2, "no"),
        ([{"unanswerable": True, "yes_no": None}] * 2, "unanswerable"),
        ([{"unanswerable": False, "yes_no": False}, {"unanswerable": True, "yes_no": None}], None),
        ([{"unanswerable": False, "yes_no": None}], None),
    ],
)
def test_requires_exact_annotation_agreement(annotations, expected):
    assert ai2_qasper_verifiable.agreed_answer(annotations) == expected


@pytest.mark.parametrize(
    ("evidence", "expected"),
    [
        (["The result is reported in the text."], False),
        (["FLOAT SELECTED: Table 1: Results."], True),
        (["The text discusses the result.", "FLOAT SELECTED: Figure 1: Results."], True),
    ],
)
def test_detects_unavailable_figure_and_table_evidence(evidence, expected):
    assert ai2_qasper.uses_missing_evidence([{"evidence": evidence}]) is expected


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        ('{"answer": " YES "}', 1),
        ('{"answer": "yes."}', 0),
        ('{"answer": "The answer is yes"}', 0),
        ('{"answer": 1}', 0),
        ("not json", 0),
    ],
)
def test_grader_normalizes_only_case_and_outer_whitespace(tmp_path, payload, expected):
    answer = tmp_path / "answer.json"
    answer.write_text(payload)
    assert ai2_qasper_verifiable_grade.score(answer, "yes") == expected
