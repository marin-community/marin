# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json

import pytest
from taskcompendium.models import (
    AnswerType,
    ConversationInput,
    EnvironmentRequirements,
    NoGrader,
    PlainText,
    Source,
    TaskSpec,
    TextMessage,
)

from experiments.post_training.long_loop.exp4_data import NEMOTRON_REVISION, NEMOTRON_SOURCE, stamped_rows


def test_admitted_task_stamping_preserves_task_semantics_and_refuses_delivery_contracts():
    task = TaskSpec(
        id="fixture:1",
        context=ConversationInput(events=(TextMessage(role="user", content="Answer A"),)),
        environment_requirements=EnvironmentRequirements(),
        answer_type=AnswerType.TEXT,
        answer_format=PlainText(),
        grader=NoGrader(reason="fixture"),
        source=Source(dataset=NEMOTRON_SOURCE, revision=NEMOTRON_REVISION, row="1", importer_revision="1"),
    )
    raw = task.model_dump(mode="json")
    atlas = json.dumps({**raw, "integrity_contract": None})
    rows, normalized = stamped_rows([{"task_spec": atlas, "runtime": "{}"}], nemotron=True)
    assert normalized == 1
    assert rows[0]["teacher_route"] == "nemotron_if"
    assert TaskSpec.model_validate_json(rows[0]["task_spec"]) == task
    with pytest.raises(ValueError, match="delivery contract"):
        stamped_rows([{"task_spec": json.dumps({**raw, "integrity_contract": {"protected": True}})}], nemotron=True)
    with pytest.raises(ValueError, match="runtime"):
        stamped_rows([{"task_spec": atlas, "runtime": '{"required":"runtime"}'}], nemotron=True)
    original = task.model_dump_json()
    carry, normalized = stamped_rows([{"task_spec": original}], nemotron=False)
    assert carry == [{"task_spec": original, "teacher_route": NEMOTRON_SOURCE}]
    assert normalized == 0
