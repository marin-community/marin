# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Decode Nemotron Ultra blend rows into conversations and grading contracts.

A blend row is one NeMo Gym request: ``responses_create_params`` holds the Responses API input and
tools the model sees, ``agent_ref`` names the NeMo Gym agent that grades it, and every other field
is grading data for that agent. Decoding keeps the conversation and tools public and the remaining
fields as the row's grading contract.
"""

import json
from collections.abc import Collection, Mapping
from dataclasses import dataclass
from typing import Any

from pydantic import JsonValue, ValidationError

from taskcompendium.convert.answers import unsupported
from taskcompendium.grader import GraderPackage
from taskcompendium.models import (
    AnswerType,
    AssistantToolCalls,
    ConversationEvent,
    ConversationInput,
    ConversationToolCall,
    EnvironmentRequirements,
    FinalAction,
    FunctionDefinition,
    PlainText,
    ResourceGroups,
    TaskSpec,
    TextMessage,
    ToolResult,
)
from taskcompendium.pipeline.models import ImportRejection, NormalizationChange, NormalizedTask, RawRow

AGENT_COMPONENT_PREFIX = "agent:"
REQUEST_FIELD = "responses_create_params"
PLACEHOLDER_FIELD = "_hf_question_placeholder"
STATE_FIELDS = ("environment", "scenario", "info", "metadata", "exp_cal_state")
"""Row fields that describe an agent's initial environment rather than its answer key."""
TEXT_BLOCKS = frozenset({"input_text", "output_text", "text"})


def blend_component(data: Mapping[str, Any]) -> str:
    """The blend component of a row: its ``dataset``, or ``agent:<name>`` for agent-keyed rows."""
    return data.get("dataset") or AGENT_COMPONENT_PREFIX + data["agent_ref"]["name"]


def message_text(content: Any) -> str:
    """Join text blocks, rejecting media rather than silently dropping their context."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        raise ValueError("Message content must be text or text blocks")
    if any(block.get("type") not in TEXT_BLOCKS for block in content):
        raise ValueError("Unsupported nontext message content")
    return "".join(block["text"] for block in content)


def conversation_events(
    items: list[dict[str, Any]],
) -> tuple[tuple[ConversationEvent, ...], tuple[NormalizationChange, ...]]:
    """Keep role order and tool call IDs; provider reasoning stays in the grading contract."""
    events: list[ConversationEvent] = []
    changes = []
    calls = []
    for index, item in enumerate(items):
        kind = item.get("type", "message")
        if kind == "reasoning":
            changes.append(
                NormalizationChange(
                    field=f"input[{index}]",
                    reason="Provider reasoning state is grading evidence, not a conversation message",
                    original=json.dumps(item, ensure_ascii=False),
                    replacement="Retained in the grading contract",
                )
            )
            continue
        if kind == "function_call":
            arguments = item["arguments"]
            calls.append(
                ConversationToolCall(
                    call_id=item["call_id"],
                    name=item["name"],
                    arguments=json.loads(arguments) if isinstance(arguments, str) else arguments,
                )
            )
            continue
        if calls:
            events.append(AssistantToolCalls(calls=tuple(calls)))
            calls = []
        if kind == "function_call_output":
            output = item["output"]
            events.append(
                ToolResult(
                    call_id=item["call_id"],
                    content=output if isinstance(output, str) else json.dumps(output, ensure_ascii=False),
                )
            )
        elif kind == "message":
            text = message_text(item["content"])
            if not text.strip() and item["role"] in {"system", "assistant"}:
                changes.append(
                    NormalizationChange(
                        field=f"input[{index}]",
                        reason="Empty source message carries no text; tool calls remain separate events",
                        original=json.dumps(item),
                        replacement="",
                    )
                )
                continue
            events.append(TextMessage(role=item["role"], content=text))
        else:
            raise ValueError(f"Unsupported source event: {kind}")
    if calls:
        events.append(AssistantToolCalls(calls=tuple(calls)))
    return tuple(events), tuple(changes)


def functions(tools: list[dict[str, Any]]) -> tuple[FunctionDefinition, ...]:
    """Advertised function tools in either the flat or the nested ``function`` layout."""
    result = []
    for tool in tools:
        if tool["type"] != "function":
            raise ValueError(f"Unsupported provider tool: {tool['type']}")
        function = tool.get("function", tool)
        result.append(
            FunctionDefinition(
                name=function["name"],
                parameters=function["parameters"],
                description=function.get("description"),
                strict=function.get("strict"),
            )
        )
    return tuple(result)


@dataclass(frozen=True)
class BlendRequest:
    """One decoded blend row.

    ``contract`` holds every row field except the request, plus the request's sampling options and
    any provider reasoning; ``state`` is the subset of fields describing the agent's environment.
    """

    agent: str
    context: ConversationInput
    tools: tuple[FunctionDefinition, ...]
    contract: dict[str, JsonValue]
    state: dict[str, JsonValue]
    changes: tuple[NormalizationChange, ...]


def blend_request(data: Mapping[str, Any]) -> BlendRequest | ImportRejection:
    """Decode a row; a question still held by an external placeholder cannot be posed."""
    if data.get(PLACEHOLDER_FIELD):
        return unsupported("unresolved_external_placeholder", json.dumps(data[PLACEHOLDER_FIELD], ensure_ascii=False))
    request = data[REQUEST_FIELD]
    try:
        events, changes = conversation_events(request["input"])
        context = ConversationInput(events=events)
        tools = functions(request.get("tools", []))
    except (ValueError, KeyError, TypeError, ValidationError) as error:
        return unsupported("unsupported_request", str(error))
    contract = {key: value for key, value in data.items() if key not in {REQUEST_FIELD, "path"}}
    contract["request_options"] = {key: value for key, value in request.items() if key not in {"input", "tools"}}
    reasoning = [item for item in request["input"] if item.get("type") == "reasoning"]
    if reasoning:
        contract["provider_reasoning"] = reasoning
    return BlendRequest(
        agent=data["agent_ref"]["name"],
        context=context,
        tools=tools,
        contract=contract,
        state={key: data[key] for key in STATE_FIELDS if key in data},
        changes=changes,
    )


def agent_request(data: Mapping[str, Any], agents: Collection[str]) -> BlendRequest | ImportRejection:
    """Decode a row whose grading agent is one of ``agents``; other agents have no grader here."""
    request = blend_request(data)
    if isinstance(request, ImportRejection):
        return request
    if request.agent not in agents:
        return unsupported("unsupported_agent", f"Only rows graded by {sorted(agents)} are supported")
    return request


def text_request(data: Mapping[str, Any], agents: Collection[str]) -> BlendRequest | ImportRejection:
    """Decode a row whose agent grades the final reply text, so the request may offer no tools."""
    request = agent_request(data, agents)
    if isinstance(request, BlendRequest) and request.tools:
        return unsupported("unsupported_tool_request", "This scorer grades terminal text only")
    return request


def blend_task(
    row: RawRow,
    request: BlendRequest,
    package: GraderPackage,
    answer_type: AnswerType = AnswerType.TEXT,
    changes: tuple[NormalizationChange, ...] = (),
) -> NormalizedTask:
    """A conversation task that ``package`` grades in its own machine.

    The request's tools stay advertised as final actions; the agent never executes them.
    ``changes`` are edits made to the row before decoding.
    """
    return NormalizedTask(
        TaskSpec(
            id=row.id,
            source=row.source,
            context=request.context,
            environment_requirements=EnvironmentRequirements(),
            final_tools=request.tools,
            answer_type=answer_type,
            answer_format=FinalAction() if answer_type == AnswerType.NATIVE_ACTION else PlainText(),
            grader=package.grader,
            resources=ResourceGroups(verifier=package.resources),
        ),
        (*changes, *request.changes),
    )
