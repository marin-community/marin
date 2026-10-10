# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Blend rows keep the model's conversation and tools public and every other field as grading data."""

import pytest

from taskcompendium.convert.nemotron_ultra import (
    BlendRequest,
    agent_request,
    blend_component,
    blend_request,
    text_request,
)
from taskcompendium.models import (
    AssistantToolCalls,
    ConversationToolCall,
    FunctionDefinition,
    TextMessage,
    ToolResult,
)
from taskcompendium.pipeline.models import ImportFailureKind, ImportRejection

TOOL = {
    "type": "function",
    "name": "lookup",
    "description": "Look up a value.",
    "parameters": {"type": "object", "properties": {"key": {"type": "string"}}},
}
REASONING = {"type": "reasoning", "summary": [{"type": "summary_text", "text": "Look it up first."}]}


def row(**fields) -> dict:
    return {
        "dataset": "component",
        "agent_ref": {"name": "fixture_agent"},
        "responses_create_params": {
            "input": [
                {"role": "system", "content": ""},
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": "Find "}, {"type": "input_text", "text": "x."}],
                },
                REASONING,
                {"type": "function_call", "call_id": "call-1", "name": "lookup", "arguments": '{"key": "x"}'},
                {"type": "function_call_output", "call_id": "call-1", "output": {"value": 3}},
            ],
            "tools": [TOOL],
            "temperature": 1.0,
        },
        "expected_answer": "3",
        "environment": {"x": 3},
        "path": "mopd/component/0",
        **fields,
    }


def test_blend_request_keeps_conversation_tools_and_grading_data_apart():
    request = blend_request(row())
    assert isinstance(request, BlendRequest)
    assert request.agent == "fixture_agent"
    assert request.context.events == (
        TextMessage(role="user", content="Find x."),
        AssistantToolCalls(calls=(ConversationToolCall(call_id="call-1", name="lookup", arguments={"key": "x"}),)),
        ToolResult(call_id="call-1", content='{"value": 3}'),
    )
    assert request.tools == (
        FunctionDefinition(name="lookup", parameters=TOOL["parameters"], description="Look up a value."),
    )
    assert request.contract == {
        "dataset": "component",
        "agent_ref": {"name": "fixture_agent"},
        "expected_answer": "3",
        "environment": {"x": 3},
        "request_options": {"temperature": 1.0},
        "provider_reasoning": [REASONING],
    }
    assert request.state == {"environment": {"x": 3}}
    assert [change.field for change in request.changes] == ["input[0]", "input[2]"]


@pytest.mark.parametrize(
    ("data", "reason"),
    [
        (
            row(_hf_question_placeholder={"dataset": "upstream", "split": "train", "row": 0}),
            "unresolved_external_placeholder",
        ),
        (
            row(responses_create_params={"input": [{"role": "user", "content": [{"type": "input_image"}]}]}),
            "unsupported_request",
        ),
        (row(responses_create_params={"input": [], "tools": [{"type": "web_search"}]}), "unsupported_request"),
    ],
)
def test_blend_request_rejects_rows_it_cannot_pose(data, reason):
    result = blend_request(data)
    assert isinstance(result, ImportRejection)
    assert (result.kind, result.reason) == (ImportFailureKind.UNSUPPORTED, reason)


def test_agent_and_text_requests_admit_only_their_grading_agents():
    assert isinstance(agent_request(row(), ("fixture_agent",)), BlendRequest)
    other = agent_request(row(), ("other_agent",))
    assert isinstance(other, ImportRejection) and other.reason == "unsupported_agent"
    with_tools = text_request(row(), ("fixture_agent",))
    assert isinstance(with_tools, ImportRejection) and with_tools.reason == "unsupported_tool_request"
    without_tools = row(responses_create_params={"input": [{"role": "user", "content": "Say hi."}]})
    assert isinstance(text_request(without_tools, ("fixture_agent",)), BlendRequest)


def test_blend_component_names_agent_keyed_rows_by_agent():
    assert blend_component(row()) == "component"
    assert blend_component({"agent_ref": {"name": "next_action_agent"}}) == "agent:next_action_agent"
