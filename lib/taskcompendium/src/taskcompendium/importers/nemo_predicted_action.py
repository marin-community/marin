# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Import pinned NeMo next-action rows as final submissions."""

import hashlib
import json
from typing import Any

from taskcompendium.models import (
    AnswerType,
    AssistantToolCalls,
    ConversationInput,
    ConversationToolCall,
    FunctionCall,
    FunctionDefinition,
    Source,
    TaskRequirements,
    TaskSpec,
    TaskTools,
    TextMessage,
    ToolResult,
)
from taskcompendium.predicted_action_verifier import predicted_action_verifier
from taskcompendium.submission import AnswerFormat, SubmissionConvention

DATASET = "nvidia/Nemotron-RL-Agentic-Conversational-Tool-Use-Pivot-v1"
REVISION = "9643c8103d7bfbc2d7fc4d15991d6739c612ff58"
IMPORTER_REVISION = "taskcompendium-nemo-predicted-action-v1"
FUNCTION_CALL_TYPE = "function_call"


def canonical_sha256(row: dict[str, Any]) -> str:
    document = json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(document.encode()).hexdigest()


def _expected_calls(value: Any) -> tuple[FunctionCall, ...]:
    if not isinstance(value, dict):
        raise ValueError("expected_action must be an object")
    if value.get("type") == "message":
        raise ValueError("message targets have no correctness comparison")
    if (
        value.get("type") == FUNCTION_CALL_TYPE
        and isinstance(value.get("name"), str)
        and isinstance(value.get("arguments"), str)
    ):
        return (FunctionCall(value["name"], value["arguments"]),)
    if value.get("type") == "function_call_batch" and isinstance(value.get("calls"), list) and value["calls"]:
        calls = value["calls"]
        if all(
            isinstance(call, dict)
            and call.get("type") == FUNCTION_CALL_TYPE
            and isinstance(call.get("name"), str)
            and isinstance(call.get("arguments"), str)
            for call in calls
        ):
            return tuple(FunctionCall(call["name"], call["arguments"]) for call in calls)
    raise ValueError("unsupported expected_action")


def _functions(request: dict[str, Any]) -> tuple[FunctionDefinition, ...]:
    tools = request.get("tools")
    if not isinstance(tools, list) or not tools:
        raise ValueError("source request requires advertised functions")
    functions = []
    for tool in tools:
        if not isinstance(tool, dict) or tool.get("type") != "function":
            raise ValueError("only native function definitions are supported")
        name, parameters = tool.get("name"), tool.get("parameters")
        if not isinstance(name, str) or not name or not isinstance(parameters, dict):
            raise ValueError("function definitions require a name and parameter schema")
        description, strict = tool.get("description"), tool.get("strict")
        if description is not None and not isinstance(description, str):
            raise ValueError("function description must be a string")
        if strict is not None and not isinstance(strict, bool):
            raise ValueError("function strict must be a boolean")
        functions.append(FunctionDefinition(name=name, parameters=parameters, description=description, strict=strict))
    if len({function.name for function in functions}) != len(functions):
        raise ValueError("advertised function names must be unique")
    return tuple(functions)


def _events(request: dict[str, Any]) -> tuple[TextMessage | AssistantToolCalls | ToolResult, ...]:
    items = request.get("input")
    if not isinstance(items, list):
        raise ValueError("source input must be a list")
    events: list[TextMessage | AssistantToolCalls | ToolResult] = []
    pending_calls: list[ConversationToolCall] = []
    reasoning_without_visible_result = False
    for item in items:
        if not isinstance(item, dict):
            raise ValueError("unsupported source input item")
        if item.get("type") == "reasoning":
            if item.get("encrypted_content") is not None or not isinstance(item.get("summary"), list):
                raise ValueError("unsupported source reasoning item")
            reasoning_without_visible_result = True
            continue  # Historical reasoning is omitted; its visible result must follow.
        if item.get("type") == FUNCTION_CALL_TYPE:
            if not all(isinstance(item.get(key), str) and item[key] for key in ("call_id", "name", "arguments")):
                raise ValueError("source function calls require call_id, name, and arguments")
            pending_calls.append(
                ConversationToolCall(call_id=item["call_id"], name=item["name"], arguments=item["arguments"])
            )
            reasoning_without_visible_result = False
            continue
        if reasoning_without_visible_result and (item.get("type") != "message" or item.get("role") != "assistant"):
            raise ValueError("source reasoning has no visible assistant result")
        if pending_calls:
            events.append(AssistantToolCalls(calls=tuple(pending_calls)))
            pending_calls = []
        if item.get("type") == "function_call_output":
            if not isinstance(item.get("call_id"), str) or not isinstance(item.get("output"), str):
                raise ValueError("source function results require call_id and string output")
            events.append(ToolResult(call_id=item["call_id"], content=item["output"]))
            continue
        if item.get("type") != "message" or item.get("role") not in {"system", "developer", "user", "assistant"}:
            raise ValueError("unsupported source input item")
        content = item.get("content")
        if isinstance(content, list):
            if not all(
                isinstance(item, dict) and item.get("type") == "output_text" and isinstance(item.get("text"), str)
                for item in content
            ):
                raise ValueError("unsupported message content")
            content = "".join(item["text"] for item in content)
        if not isinstance(content, str) or not content.strip():
            raise ValueError("source messages require text")
        events.append(TextMessage(role=item["role"], content=content))
        reasoning_without_visible_result = False
    if reasoning_without_visible_result:
        raise ValueError("source reasoning has no visible assistant result")
    if pending_calls:
        events.append(AssistantToolCalls(calls=tuple(pending_calls)))
    if not events:
        raise ValueError("source input has no conversation events")
    return tuple(events)


def import_row(row: dict[str, Any], expected_sha256: str) -> tuple[TaskSpec, SubmissionConvention]:
    """Verify row identity and retain the expected action only in private TaskSpec data."""
    if canonical_sha256(row) != expected_sha256:
        raise ValueError("source row does not match its pinned canonical hash")
    request = row.get("responses_create_params")
    if not isinstance(request, dict):
        raise ValueError("responses_create_params must be an object")
    unsupported = [
        key
        for key, value in request.items()
        if key not in {"input", "tools", "tool_choice", "parallel_tool_calls"} and value is not None
    ]
    if unsupported:
        raise ValueError(f"unsupported source request settings: {', '.join(sorted(unsupported))}")
    functions = _functions(request)
    events = _events(request)
    tool_choice = request.get("tool_choice")
    parallel_tool_calls = request.get("parallel_tool_calls")
    if tool_choice is not None and (not isinstance(tool_choice, str) or tool_choice not in {"auto", "none", "required"}):
        raise ValueError("unsupported source tool_choice")
    if parallel_tool_calls is not None and not isinstance(parallel_tool_calls, bool):
        raise ValueError("source parallel_tool_calls must be a boolean")
    expected_calls = _expected_calls(row.get("expected_action"))
    if tool_choice == "none":
        raise ValueError("expected function calls contradict tool_choice=none")
    if parallel_tool_calls is False and len(expected_calls) > 1:
        raise ValueError("multiple expected calls contradict parallel_tool_calls=false")
    advertised = {function.name for function in functions}
    if any(call.name not in advertised for call in expected_calls):
        raise ValueError("expected function call is absent from source tools")
    source = Source(dataset=DATASET, revision=REVISION, row=expected_sha256, importer_revision=IMPORTER_REVISION)
    specification = TaskSpec(
        id=f"nemo-predicted-action-{expected_sha256}",
        context=ConversationInput(events=events),
        requirements=TaskRequirements(),
        tools=TaskTools(
            functions=functions,
            tool_choice=tool_choice,
            parallel_tool_calls=parallel_tool_calls,
        ),
        answer_type=AnswerType.NATIVE_ACTION,
        verifier=predicted_action_verifier(expected_calls),
        source=source,
    )
    convention = SubmissionConvention(
        id="native-final-action",
        answer_format=AnswerFormat.FINAL_ACTION,
    )
    return specification, convention
