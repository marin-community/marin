# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import base64
import json

import pytest
from taskcompendium.grading import grade_answer
from taskcompendium.grading_result import Outcome
from taskcompendium.models import (
    ConversationTrace,
    GradingAttempt,
    InlineFile,
    TextMessage,
    VerifyitGrader,
    verifyit_spec,
)
from verifyit.spec import StructuredExactSpec

from experiments.post_training.capability_driven_envs.covid_repertoire_tasks.generate_tasks import (
    REQUIRED_COLUMNS,
    generate,
    read_counts,
)
from experiments.post_training.capability_driven_envs.covid_repertoire_tasks.render_review import render


def fixture_counts() -> bytes:
    lines = [",".join(REQUIRED_COLUMNS)]
    cell_types = ("CD4", "CD8", "Lymph_prolif", "MAIT", "Treg", "gdT")
    for donor_index in range(1, 9):
        for type_index, cell_type in enumerate(cell_types, start=1):
            called = 60 + donor_index + type_index
            trav = min(called, donor_index * type_index)
            lines.append(f"donor_{donor_index:03d},{cell_type},{called + 5},{called},{trav}")
    return ("\n".join(lines) + "\n").encode()


def test_generates_a_balanced_reproducible_25_task_release():
    data = fixture_counts()

    first = generate(data, source_revision="sha256:fixture", count=25)
    second = generate(data, source_revision="sha256:fixture", count=25)

    assert [task.model_dump_json() for task in first] == [task.model_dump_json() for task in second]
    assert len({task.id for task in first}) == 25
    assert len({task.tags[-1] for task in first}) == 7
    for task in first:
        assert isinstance(task.grader, VerifyitGrader)
        spec = verifyit_spec(task.grader)
        assert isinstance(spec, StructuredExactSpec)
        assert task.resources.verifier == ()
        assert len(task.resources.worker) == 3
        data_resource = next(
            resource for resource in task.resources.worker if resource.path.endswith("repertoire_counts.csv")
        )
        resource = data_resource.source
        assert isinstance(resource, InlineFile)
        assert base64.b64decode(resource.content_base64) == data
        json.dumps(spec.expected, allow_nan=False)

    task = first[0]
    spec = verifyit_spec(task.grader)
    assert isinstance(spec, StructuredExactSpec)
    correct = GradingAttempt(
        conversation=ConversationTrace(
            events=(*task.context.events, TextMessage(role="assistant", content=json.dumps(spec.expected)))
        )
    )
    wrong = GradingAttempt(
        conversation=ConversationTrace(events=(*task.context.events, TextMessage(role="assistant", content="{}")))
    )
    correct_result = grade_answer(task, correct)
    wrong_result = grade_answer(task, wrong)
    assert (correct_result.status, correct_result.reward) == (Outcome.GRADED, 1.0)
    assert (wrong_result.status, wrong_result.reward) == (Outcome.GRADED, 0.0)


def test_private_review_sheet_contains_prompts_answers_and_checklist(tmp_path):
    tasks = generate(fixture_counts(), source_revision="sha256:fixture", count=2)
    path = tmp_path / "tasks.jsonl"
    path.write_text("\n".join(task.model_dump_json() for task in tasks) + "\n")

    review = render(path)

    assert tasks[0].id in review
    assert "Hidden expected answer" in review
    assert "Uses biological knowledge or reference material: TODO" in review
    assert "Do not expose this file to solvers" in review


def test_rejects_inconsistent_counts():
    data = (",".join(REQUIRED_COLUMNS) + "\n" + "donor_001,MAIT,10,8,9\n").encode()

    with pytest.raises(ValueError, match="TRAV1-2 <= called <= cells"):
        read_counts(data)


def test_refuses_to_silently_emit_fewer_tasks_than_requested():
    data = fixture_counts()

    with pytest.raises(ValueError, match="supports only"):
        generate(data, source_revision="sha256:fixture", count=10_000)
