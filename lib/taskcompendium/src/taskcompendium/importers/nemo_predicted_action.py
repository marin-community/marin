# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Import pinned NeMo next-action rows as final submissions."""

import hashlib
import json
from typing import Any

from taskcompendium.models import AnswerFormat, FunctionCall, NativeFunction, Source, TaskRequirements, TaskSpec
from taskcompendium.nemo_verifier import predicted_action_verifier
from taskcompendium.rendering import NativeMessage, Rendering, format_native_messages

DATASET = "nvidia/Nemotron-RL-Agentic-Conversational-Tool-Use-Pivot-v1"
REVISION = "9643c8103d7bfbc2d7fc4d15991d6739c612ff58"
IMPORTER_REVISION = "taskcompendium-nemo-predicted-action-v1"


def canonical_sha256(row: dict[str, Any]) -> str:
    document = json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(document.encode()).hexdigest()


def _expected_calls(value: Any) -> tuple[FunctionCall, ...]:
    if not isinstance(value, dict):
        raise ValueError("expected_action must be an object")
    if value.get("type") == "message":
        raise ValueError("message targets have no correctness comparison")
    if (
        value.get("type") == "function_call"
        and isinstance(value.get("name"), str)
        and isinstance(value.get("arguments"), str)
    ):
        return (FunctionCall(value["name"], value["arguments"]),)
    if value.get("type") == "function_call_batch" and isinstance(value.get("calls"), list) and value["calls"]:
        calls = value["calls"]
        if all(
            isinstance(call, dict)
            and call.get("type") == "function_call"
            and isinstance(call.get("name"), str)
            and isinstance(call.get("arguments"), str)
            for call in calls
        ):
            return tuple(FunctionCall(call["name"], call["arguments"]) for call in calls)
    raise ValueError("unsupported expected_action")


def _functions(request: dict[str, Any]) -> tuple[NativeFunction, ...]:
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
        functions.append(NativeFunction(name=name, parameters=parameters, description=description, strict=strict))
    if len({function.name for function in functions}) != len(functions):
        raise ValueError("advertised function names must be unique")
    return tuple(functions)


def _messages(request: dict[str, Any]) -> tuple[NativeMessage, ...]:
    messages = request.get("input")
    if not isinstance(messages, list):
        raise ValueError("source input must be a list")
    turns = []
    for message in messages:
        if not isinstance(message, dict):
            raise ValueError("unsupported source input item")
        if message.get("type") == "reasoning":
            if message.get("encrypted_content") is not None or not isinstance(message.get("summary"), list):
                raise ValueError("unsupported source reasoning item")
            continue  # API reasoning summaries are not conversation messages.
        if message.get("type") != "message" or message.get("role") not in {"system", "user", "assistant"}:
            raise ValueError("unsupported source input item")
        content = message.get("content")
        if isinstance(content, list):
            if not all(
                isinstance(item, dict) and item.get("type") == "output_text" and isinstance(item.get("text"), str)
                for item in content
            ):
                raise ValueError("unsupported message content")
            content = "".join(item["text"] for item in content)
        if not isinstance(content, str) or not content.strip():
            raise ValueError("source messages require text")
        turns.append(NativeMessage(role=message["role"], content=content))
    if not turns:
        raise ValueError("source input has no messages")
    return tuple(turns)


def import_row(row: dict[str, Any], expected_sha256: str) -> tuple[TaskSpec, Rendering]:
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
    messages = _messages(request)
    tool_choice = request.get("tool_choice")
    parallel_tool_calls = request.get("parallel_tool_calls")
    if tool_choice is not None and (not isinstance(tool_choice, str) or tool_choice not in {"auto", "none", "required"}):
        raise ValueError("unsupported source tool_choice")
    if parallel_tool_calls is not None and not isinstance(parallel_tool_calls, bool):
        raise ValueError("source parallel_tool_calls must be a boolean")
    expected_calls = _expected_calls(row.get("expected_action"))
    advertised = {function.name for function in functions}
    if any(call.name not in advertised for call in expected_calls):
        raise ValueError("expected function call is absent from source tools")
    source = Source(dataset=DATASET, revision=REVISION, row=expected_sha256, importer_revision=IMPORTER_REVISION)
    specification = TaskSpec(
        id=f"nemo-predicted-action-{expected_sha256}",
        instructions=format_native_messages(messages),
        verifier=predicted_action_verifier(expected_calls),
        source=source,
        requirements=TaskRequirements(),
        permitted_answer_formats=(AnswerFormat.FINAL_ACTION,),
    )
    rendering = Rendering(
        id="nemo-native-final-action",
        answer_format=AnswerFormat.FINAL_ACTION,
        functions=functions,
        messages=messages,
        tool_choice=tool_choice,
        parallel_tool_calls=parallel_tool_calls,
    )
    return specification, rendering
