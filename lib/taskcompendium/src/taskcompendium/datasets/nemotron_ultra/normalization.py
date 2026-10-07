# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Import original NeMo Gym requests without replacing their agent reward contracts."""

import json
from typing import Any

from pydantic import ValidationError

from taskcompendium.datasets.direct_contracts import source_contract_package
from taskcompendium.datasets.nemotron.placeholders import restore_placeholder
from taskcompendium.models import (
    AnswerType,
    AssistantToolCalls,
    ConversationEvent,
    ConversationInput,
    ConversationToolCall,
    EnvironmentRequirements,
    FunctionDefinition,
    ProviderRequirement,
    ResourceGroups,
    TaskSpec,
    TextMessage,
    ToolResult,
)
from taskcompendium.pipeline.models import (
    ImportFailureKind,
    ImportRejection,
    NormalizationChange,
    NormalizedTask,
    RawRow,
)

VERIFIER_REVISION = "d8b6e8c163def3660e9d3072c1c174226a1709fa"


def message_text(content: Any) -> str:
    """Decode source text blocks, rejecting media rather than dropping their context."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        raise ValueError("Message content must be text or text blocks")
    if any(block.get("type") not in {"input_text", "output_text", "text"} for block in content):
        raise ValueError("Unsupported nontext message content")
    return "".join(block["text"] for block in content)


def conversation_events(
    items: list[dict[str, Any]],
) -> tuple[tuple[ConversationEvent, ...], tuple[NormalizationChange, ...]]:
    """Keep role order and tool call IDs; retain provider reasoning in private raw evidence."""
    events: list[ConversationEvent] = []
    changes = []
    calls = []
    for index, item in enumerate(items):
        kind = item.get("type", "message")
        if kind == "reasoning":
            changes.append(
                NormalizationChange(
                    field=f"input[{index}]",
                    reason="Provider reasoning state is private evidence, not a conversation message",
                    original=json.dumps(item, ensure_ascii=False),
                    replacement="Retained in private source record",
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


def normalize(row: RawRow, selector: str, family: str) -> NormalizedTask | ImportRejection:
    """Preserve the exact request and agent contract for static curation before runtime binding."""
    actual = row.data.get("dataset") or "agent:" + row.data["agent_ref"]["name"]
    if actual != selector:
        return ImportRejection(
            kind=ImportFailureKind.CONVERTER_ERROR,
            reason="component_mismatch",
            detail=f"Expected {selector}; got {actual}",
        )
    data = dict(row.data)
    placeholder_changes = ()
    if data.get("_hf_question_placeholder") and "placeholder_source" in data:
        try:
            data, placeholder_changes = restore_placeholder(data)
        except (ValueError, KeyError, TypeError) as error:
            return ImportRejection(
                kind=ImportFailureKind.UNSUPPORTED, reason="invalid_placeholder_source", detail=str(error)
            )
    if data.get("_hf_question_placeholder"):
        return ImportRejection(
            kind=ImportFailureKind.UNSUPPORTED,
            reason="unresolved_external_placeholder",
            detail=json.dumps(row.data["_hf_question_placeholder"], ensure_ascii=False),
        )
    request = data["responses_create_params"]
    try:
        events, changes = conversation_events(request["input"])
        context = ConversationInput(events=events)
        tools = functions(request.get("tools", []))
    except (ValueError, KeyError, TypeError, ValidationError) as error:
        return ImportRejection(kind=ImportFailureKind.UNSUPPORTED, reason="unsupported_request", detail=str(error))
    requirements = (
        "nemotron-agent:" + row.data["agent_ref"]["name"],
        "upstream-NeMo-Gym-binding:unverified",
    )
    contract = {key: value for key, value in data.items() if key not in {"responses_create_params", "path"}}
    contract["request_options"] = {key: value for key, value in request.items() if key not in {"input", "tools"}}
    reasoning = [item for item in request["input"] if item.get("type") == "reasoning"]
    if reasoning:
        contract["provider_reasoning"] = reasoning
    package = source_contract_package(
        "MarinSkyRL NemotronUltraEnv " + row.data["agent_ref"]["name"],
        VERIFIER_REVISION,
        contract,
        requirements,
    )
    state = {key: data[key] for key in ("environment", "scenario", "info", "metadata", "exp_cal_state") if key in data}
    providers = (
        {"nemotron_agent": ProviderRequirement(action_interface=requirements[0], initial_state=state)}
        if tools or (state and family in {"tool-use", "agentic-safety", "swe-repo"})
        else {}
    )
    task = TaskSpec(
        id=row.id,
        source=row.source,
        context=context,
        environment_requirements=EnvironmentRequirements(capabilities=requirements, tool_providers=providers),
        final_tools=tools,
        interaction_tools=tools,
        answer_type=(
            AnswerType.NATIVE_ACTION
            if data.get("expected_action", {}).get("type") == "function_call" and tools
            else AnswerType.TEXT
        ),
        verifier=package.verifier,
        resources=ResourceGroups(verifier=package.resources),
    )
    return NormalizedTask(task, (*placeholder_changes, *changes))
