# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Source contracts retain unsupported semantics instead of exact-only acceptance."""

import json
from pathlib import Path

import pytest

from taskcompendium.grading import Outcome
from taskcompendium.models import Source, TaskSpec, TextMessage
from taskcompendium.pipeline.datasets import nemo_actions, qa_tasks
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
from taskcompendium.pipeline.verification import PLAIN
from taskcompendium.verifiers.base import GradingAttempt
from taskcompendium.verifiers.reference_answers import ReferenceAnswersVerifier


@pytest.fixture(params=["knowledge", "science"])
def qa_row(request):
    question = "What is the capital of France? Put your answer inside \\boxed{}."
    references = (
        {"expected_answers": ["**Paris**", "City of Paris"]}
        if request.param == "knowledge"
        else {"reference_answer": "**Paris**"}
    )
    return RawRow(
        "qa-1",
        Source(dataset="open-thoughts/TaskTrove", revision=qa_tasks.REVISION, row="test:0", importer_revision="test-v1"),
        {
            "instruction": (
                "You are answering an open-ended question. " + qa_tasks.SOURCE_DELIVERY + "\n---\n" + question
            ),
            "verifier_data": {
                "instruction": question,
                **references,
                "judge_system_prompt": "Accept substantive equivalence, including paraphrases.",
            },
        },
    )


@pytest.mark.parametrize("answer", ["\\boxed{PARIS}", "The Paris."])
def test_openqa_gate_accepts_source_answer_shapes_and_preserves_private_judge(qa_row, answer):
    task = qa_tasks.normalize(qa_row)
    assert isinstance(task, TaskSpec)
    verifier = ReferenceAnswersVerifier.model_validate_json(task.verifier.parameters_json)
    result = verifier.grade(
        GradingAttempt(PLAIN, (*task.context.events, TextMessage(role="assistant", content=answer)), None)
    )
    assert (result.status, result.reward) == (Outcome.GRADED, 1.0)
    assert "Accept substantive equivalence" not in task.context.events[0].content
    assert verifier.source_judge_data == qa_row.data["verifier_data"]


def test_openqa_static_quality_acceptance_preserves_unbound_answer_grading(qa_row):
    task = qa_tasks.normalize(qa_row)
    assert isinstance(task, TaskSpec)
    verifier = ReferenceAnswersVerifier.model_validate_json(task.verifier.parameters_json)
    result = verifier.grade(
        GradingAttempt(
            PLAIN,
            (*task.context.events, TextMessage(role="assistant", content="France's capital is Paris.")),
            None,
        )
    )
    assert (result.status, result.reward) == (Outcome.INFRA_ERROR, None)
    report = qa_tasks.verification_report(task)
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
            evidence="The capital is supplied by the references and the question is complete.",
        ),
    )
    decision = task_decision(task.id, report.checks, review, FilterPolicy())
    assert decision.disposition == Disposition.KEEP
    assert report.checks[-1].status == CheckStatus.UNSUPPORTED


def test_nemo_reference_schema_violation_is_not_hidden_by_matching_comparator():
    data = json.loads((Path(__file__).parent / "fixtures/nemo/predicted-action.json").read_text())
    arguments = json.loads(data["expected_action"]["arguments"])
    arguments["user_id"] = 123
    data["expected_action"]["arguments"] = json.dumps(arguments)
    row = RawRow(
        "action-1",
        Source(dataset=nemo_actions.DATASET, revision=nemo_actions.REVISION, row="test:0", importer_revision="test-v1"),
        data,
    )
    task = nemo_actions.normalize(row)
    assert isinstance(task, TaskSpec)
    checks = nemo_actions.verification_report(task).checks
    assert any(check.check == "reference_arguments" and check.status == CheckStatus.FAIL for check in checks)
    assert any(check.check == "reference" and check.status == CheckStatus.PASS for check in checks)


def test_nemo_text_target_retains_explicit_unsupported_reason():
    data = json.loads((Path(__file__).parent / "fixtures/nemo/predicted-action.json").read_text())
    data["expected_action"] = {"type": "message", "content": "Please provide your user ID."}
    row = RawRow(
        "action-2",
        Source(dataset=nemo_actions.DATASET, revision=nemo_actions.REVISION, row="test:1", importer_revision="test-v1"),
        data,
    )
    result = nemo_actions.normalize(row)
    assert isinstance(result, ImportRejection)
    assert result.reason == "unsupported_action"
