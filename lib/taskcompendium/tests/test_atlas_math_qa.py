# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Answer adapters preserve private evidence and the source question's scope."""

import base64
import json

import pytest

from taskcompendium.grading import grade_answer
from taskcompendium.grading_result import Outcome
from taskcompendium.models import ConversationTrace, Source, TaskSpec, TextMessage
from taskcompendium.pipeline.datasets import atlas_math_qa
from taskcompendium.pipeline.models import ImportRejection, RawRow
from taskcompendium.runtime.resources import resource_bytes
from taskcompendium.submission import AnswerFormat, SubmissionConvention


def row(name, instruction, data):
    files = {"tests/verifier_data.json": base64.b64encode(json.dumps(data).encode()).decode()}
    return RawRow(
        "test-task",
        Source(
            dataset="open-thoughts/TaskTrove",
            revision="fixture-v1",
            row=f"{name}:train:0",
            importer_revision="test",
        ),
        {"instruction": instruction, "verifier_data": data, "files": files},
    )


def grade(task, answer):
    return grade_answer(
        task,
        SubmissionConvention(id="plain", answer_format=AnswerFormat.PLAIN),
        ConversationTrace(events=(*task.context.events, TextMessage(role="assistant", content=answer))),
    )


@pytest.mark.parametrize(
    "math_type,reference,equivalent,wrong",
    [
        ("scalar", "1/2", r"\boxed{0.5}", r"\boxed{2}"),
        ("interval", "(2, \\infty)", r"\boxed{x > 2}", r"\boxed{x < 2}"),
        ("list", "[1,2]", r"\boxed{1,2}", r"\boxed{2,1}"),
    ],
)
def test_typed_math_grades_equivalent_expressions_without_losing_answer_type(math_type, reference, equivalent, wrong):
    task = atlas_math_qa.normalize(
        row(
            "math_openreasoning",
            "Show your work and solve this problem.",
            {"answer_type": math_type, "expected_answer": reference},
        ),
        "math_openreasoning",
    )
    assert isinstance(task, TaskSpec)
    assert (grade(task, equivalent).status, grade(task, equivalent).reward) == (Outcome.GRADED, 1.0)
    assert grade(task, wrong).reward == 0.0
    source_evidence = next(
        resource for resource in task.resources.verifier if resource.path == "source/tests/verifier_data.json"
    )
    assert json.loads(resource_bytes(source_evidence)) == {
        "answer_type": math_type,
        "expected_answer": reference,
    }


def test_numeric_scope_and_tolerance_survive_delivery_normalization():
    instruction = (
        "Read the problem and write the final numeric answer (a single number) to `/app/answer.txt`.\n"
        "If several values are requested, write ONLY the value of the LAST one to `/app/answer.txt`.\n"
        "Find cos(8) and cos(1)."
    )
    task = atlas_math_qa.normalize(
        row(
            "advanced_calculations",
            instruction,
            {"expected_value": 0.54030231, "tolerance_abs": 0.0001, "tolerance_rel": 0.0001},
        ),
        "advanced_calculations",
    )
    assert isinstance(task, TaskSpec)
    assert "/app/answer.txt" not in task.context.events[0].content
    assert "Find cos(8) and cos(1)." in task.context.events[0].content
    assert grade(task, "0.54035").reward == 1.0
    assert grade(task, "0.541").reward == 0.0


@pytest.mark.parametrize("name", ["knowledge_mcqa", "web_search_mcqa"])
def test_mcqa_keeps_choices_public_and_key_private_after_replacing_submission_wrapper(name):
    instruction = "Write to a file.\n---\n\nWhich option equals two?\nA: one\nB: two\nC: three"
    task = atlas_math_qa.normalize(
        row(name, instruction, {"expected_answer": "B", "output_regex": atlas_math_qa.MCQ_REGEX}), name
    )
    assert isinstance(task, TaskSpec)
    assert "A: one\nB: two\nC: three" in task.context.events[0].content
    assert grade(task, "B").reward == 1.0
    assert grade(task, "A").reward == 0.0
    assert grade(task, "Answer: B").status == Outcome.EXTRACTION_ERROR
    assert not task.resources.worker and not task.resources.all


def test_abstention_is_zero_while_paraphrases_remain_ungraded():
    task = atlas_math_qa.normalize(
        row(
            "qa_abstention",
            "Name the capital of France.",
            {"expected_answer": "Paris", "question": "Name the capital of France.", "abstention_token": None},
        ),
        "qa_abstention",
    )
    assert isinstance(task, TaskSpec)
    assert grade(task, r"\boxed{PARIS}").reward == 1.0
    assert (grade(task, r"\boxed{[IDK]}").status, grade(task, r"\boxed{[IDK]}").reward) == (Outcome.GRADED, 0.0)
    assert (grade(task, "France's capital is Paris.").status, grade(task, "France's capital is Paris.").reward) == (
        Outcome.INFRA_ERROR,
        None,
    )
    assert grade(task, r"\boxed{\text{Paris}}").status == Outcome.INFRA_ERROR


@pytest.mark.parametrize("wrapper", ["{}", r"\boxed{{{}}}"])
def test_mcqa_roman_statements_do_not_invent_options_or_remove_question_constraints(wrapper):
    question = (
        "Consider statements (I) and (II). Put the value of statement (I) in \\boxed{} when explaining.\n"
        "(I) Every infinite set has a countably infinite subset.\n"
        "(II) A countable union of countable sets is countable.\n"
        "A: I implies II.\nB: II implies I.\nC: Equivalent.\nD: Neither implies the other."
    )
    instruction = (
        "Write to a file.\n---\n\n"
        f"{atlas_math_qa.MCQ_FORMAT_PREFIX}'Answer: {wrapper.format('A/B/C/D')}' "
        f"(e.g. 'Answer: {wrapper.format('B')}').\n\n{question}"
    )
    task = atlas_math_qa.normalize(
        row(
            "knowledge_mcqa",
            instruction,
            {
                "expected_answer": "B",
                "output_regex": atlas_math_qa.MCQ_BOXED_REGEX.replace("\\", "\\\\"),
            },
        ),
        "knowledge_mcqa",
    )
    assert isinstance(task, TaskSpec)
    assert question in task.context.events[0].content
    assert grade(task, "B").reward == 1.0
    assert grade(task, "I").reward == 0.0


def test_unsupported_text_math_retains_explicit_import_rejection():
    result = atlas_math_qa.normalize(
        row(
            "math_openreasoning",
            "Name the mathematical theorem.",
            {"answer_type": "text", "expected_answer": "Bayes theorem"},
        ),
        "math_openreasoning",
    )
    assert isinstance(result, ImportRejection)
    assert result.reason == "unsupported_answer_contract"
