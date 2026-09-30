# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json

from experiments.post_training.taskcompendium.records import catalog_record, public_task_record

from taskcompendium.models import AnswerType, ConversationInput, EnvironmentRequirements, Source, TaskSpec, TextMessage
from taskcompendium.verifiers.multiple_choice import multiple_choice_answer


def test_catalog_and_public_projection_preserve_tags_and_isolate_verifier():
    specification = TaskSpec(
        id="tasktrove-sample",
        context=ConversationInput(events=(TextMessage(role="user", content="Choose an answer."),)),
        environment_requirements=EnvironmentRequirements(),
        answer_type=AnswerType.TEXT,
        verifier=multiple_choice_answer("C", 4),
        source=Source(dataset="tasktrove-clean", revision="2026.09.18.3", row="source:path", importer_revision="test"),
        tags=("qa", "mcq", "nemotron"),
    )
    metadata = {
        "source": "source",
        "path": "path",
        "route": "rl",
        "mode": "mcq",
        "family": "qa-short-answer",
        "converter": "nemotron_mcqa",
        "template_id": "template-1",
        "archive_sha256": "a" * 64,
        "source_metadata_json": '{"source_dataset":"example/source"}',
    }

    private = catalog_record(specification, **metadata)
    public = public_task_record(specification, family=metadata["family"])

    assert private["tags"] == ["qa", "mcq", "nemotron"]
    assert private["archive_sha256"] == "a" * 64
    private_specification = json.loads(private["specification_json"])
    assert json.loads(private_specification["verifier"]["parameters_json"]) == {"expected": "C", "options": 4}
    assert public["tags"] == ["qa", "mcq", "nemotron"]
    assert public["record_version"] == 1
    assert set(public) == {
        "record_version",
        "id",
        "context",
        "environment_requirements",
        "tool_providers",
        "final_tools",
        "answer_type",
        "source",
        "submission_instruction",
        "tags",
        "source_category",
    }
    assert "verifier" not in public
    assert "expected" not in json.dumps(public)
