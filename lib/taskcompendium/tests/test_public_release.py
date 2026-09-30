# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The public dataset projection excludes private grading material."""

import json

from taskcompendium.grading import exact_answer, structured_exact
from taskcompendium.lowering import HarborEnvironmentConfig, ToolBinding
from taskcompendium.models import (
    AnswerType,
    ConversationInput,
    EnvironmentRequirements,
    ProviderRequirement,
    Source,
    TaskSpec,
    TextMessage,
)
from taskcompendium.public_projection import public_task
from taskcompendium.submission import PlainText, ProviderState


def test_public_task_retains_agent_input_and_ordered_tags_without_expected_state():
    specification = TaskSpec(
        id="example",
        context=ConversationInput(events=(TextMessage(role="user", content="What is the answer?"),)),
        environment_requirements=EnvironmentRequirements(),
        answer_type=AnswerType.TEXT,
        verifier=exact_answer("secret-gold-action"),
        source=Source(dataset="source", revision="pinned", row="1", importer_revision="1"),
    )
    record = public_task(
        specification,
        PlainText(id="plain"),
        HarborEnvironmentConfig(),
        tags=("first", "second"),
    )
    payload = json.loads(record.model_dump_json())
    assert payload["context"]["events"][0]["content"] == "What is the answer?"
    assert payload["tags"] == ["first", "second"]
    assert payload["submission_instruction"] == "Give your answer as plain text."
    assert "verifier" not in payload
    assert "secret-gold-action" not in record.model_dump_json()


def test_public_state_task_keeps_requirement_without_runtime_binding_or_expected_state():
    specification = TaskSpec(
        id="workplace-example",
        context=ConversationInput(events=(TextMessage(role="user", content="Update the calendar."),)),
        environment_requirements=EnvironmentRequirements(),
        tool_providers={"workplace": ProviderRequirement(action_interface="workplace.v1", seed_sha256="a" * 64)},
        answer_type=AnswerType.STATE,
        verifier=structured_exact({"private_marker": "secret-expected-state"}),
        source=Source(dataset="workplace", revision="pinned", row="train:0", importer_revision="1"),
    )
    environment = HarborEnvironmentConfig(
        tool_providers={
            "workplace": ToolBinding(
                action_interface="workplace.v1",
                seed_sha256="a" * 64,
                provider="python:internal.secret:NemoWorkplaceProvider",
                provider_revision="1",
                tools=("calendar_update_event",),
                tools_sha256="b" * 64,
            )
        }
    )
    record = public_task(
        specification,
        ProviderState(id="state", provider="workplace"),
        environment,
        tags=(),
        source_category="workplace_assistant_calendar",
    )
    payload = json.loads(record.model_dump_json())
    assert payload["record_version"] == 1
    assert payload["tool_providers"]["workplace"]["action_interface"] == "workplace.v1"
    assert payload["source_category"] == "workplace_assistant_calendar"
    assert payload["submission_instruction"].startswith("Use the available tools")
    assert "secret-expected-state" not in record.model_dump_json()
    assert "internal.secret" not in record.model_dump_json()
