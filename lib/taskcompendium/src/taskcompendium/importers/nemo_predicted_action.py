# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Import pinned NeMo single-step next-action rows without dispatching tools."""

import hashlib
import json
from typing import Any

from taskcompendium.models import (
    AnswerRequirements,
    Embedded,
    ExpectedAction,
    ExpectedFunctionCall,
    ExpectedFunctionCallBatch,
    ExpectedMessage,
    FinalActionSubmission,
    FunctionCall,
    NativeFunction,
    PredictedActionVerifier,
    Rejected,
    RejectionReason,
    Rendering,
    Resource,
    ResourceRole,
    Source,
    StepSpecification,
    TaskMetadata,
    TaskRequirements,
    TaskSpec,
    ToolCallComparatorConfig,
)

DATASET = "nvidia/Nemotron-RL-Agentic-Conversational-Tool-Use-Pivot-v1"
REVISION = "9643c8103d7bfbc2d7fc4d15991d6739c612ff58"
IMPORTER_REVISION = "taskcompendium-nemo-predicted-action-v0.6"
GYM_REVISION = "1e668906d2e69a9e8ee9aaafc60050a4025d9688"
GYM_FIXTURE_PATH = "resources_servers/single_step_tool_use_with_argument_comparison/data/example.jsonl"
GYM_FIXTURE_BLOB = "f7dc270084858ac2e7eda3f037a9170357ff6f60"


def canonical_sha256(row: dict[str, Any]) -> str:
    """Match the coverage ledger's canonical source-row digest."""
    document = json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(document.encode()).hexdigest()


def _expected_action(value: Any) -> ExpectedAction:
    if not isinstance(value, dict):
        raise ValueError("expected_action must be an object")
    kind = value.get("type")
    if kind == "message" and isinstance(value.get("content"), str):
        return ExpectedMessage(value["content"])
    if kind == "function_call" and isinstance(value.get("name"), str) and isinstance(value.get("arguments"), str):
        return ExpectedFunctionCall(value["name"], value["arguments"])
    if kind == "function_call_batch" and isinstance(value.get("calls"), list):
        calls = tuple(
            FunctionCall(call["name"], call["arguments"])
            for call in value["calls"]
            if isinstance(call, dict)
            and call.get("type") == "function_call"
            and isinstance(call.get("name"), str)
            and isinstance(call.get("arguments"), str)
        )
        if len(calls) == len(value["calls"]):
            return ExpectedFunctionCallBatch(calls)
    raise ValueError("unsupported expected_action")


def _functions(value: Any) -> tuple[NativeFunction, ...]:
    if not isinstance(value, list):
        raise ValueError("responses_create_params.tools must be a list")
    functions: list[NativeFunction] = []
    for tool in value:
        if not isinstance(tool, dict) or tool.get("type") != "function":
            raise ValueError("only native function tool definitions are supported")
        name, parameters = tool.get("name"), tool.get("parameters")
        if not isinstance(name, str) or not isinstance(parameters, dict):
            raise ValueError("function definitions require name and parameter schema")
        description, strict = tool.get("description"), tool.get("strict")
        if description is not None and not isinstance(description, str):
            raise ValueError("function description must be a string")
        if strict is not None and not isinstance(strict, bool):
            raise ValueError("function strict must be a boolean")
        functions.append(NativeFunction(name, parameters, description, strict))
    return tuple(functions)


def _instructions(messages: Any) -> str:
    if not isinstance(messages, list):
        raise ValueError("responses_create_params.input must be a list")
    turns: list[str] = []
    for message in messages:
        if not isinstance(message, dict):
            raise ValueError("unsupported source input item")
        if message.get("type") == "reasoning":
            if message.get("encrypted_content") is not None or not isinstance(message.get("summary"), list):
                raise ValueError("unsupported source reasoning item")
            # API reasoning records are not materialized conversation turns.
            continue
        if message.get("type") != "message":
            raise ValueError("unsupported source input item")
        role, content = message.get("role"), message.get("content")
        if role not in {"system", "user", "assistant"}:
            raise ValueError("unsupported source message role")
        if isinstance(content, list):
            if not all(
                isinstance(item, dict) and item.get("type") == "output_text" and isinstance(item.get("text"), str)
                for item in content
            ):
                raise ValueError("unsupported source message content item")
            content = "".join(item["text"] for item in content)
        if not isinstance(content, str) or not content.strip():
            raise ValueError("source message content must be nonempty text")
        turns.append(f"{role.title()}:\n{content.strip()}")
    if not turns:
        raise ValueError("source row has no materialized conversation messages")
    return "\n\n".join(turns)


def import_row(row: dict[str, Any], expected_sha256: str) -> TaskSpec | Rejected:
    """Convert one canonical source row while retaining its scorer configuration privately."""
    source = Source(DATASET, REVISION, expected_sha256, IMPORTER_REVISION)
    if canonical_sha256(row) != expected_sha256:
        return Rejected(
            source, RejectionReason.UNRECOVERABLE_SOURCE, "source row does not match its pinned canonical hash"
        )
    try:
        request = row["responses_create_params"]
        if not isinstance(request, dict):
            raise ValueError("responses_create_params must be an object")
        instructions = _instructions(request.get("input"))
        _functions(request.get("tools"))
        expected = _expected_action(row.get("expected_action"))
    except (KeyError, TypeError, ValueError) as error:
        return Rejected(source, RejectionReason.UNRECOVERABLE_SOURCE, str(error))
    comparator = ToolCallComparatorConfig(word_count_similarity_threshold=0.1)
    private_row = json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    source_provenance = json.dumps(
        {
            "repository": "NVIDIA-NeMo/Gym",
            "revision": GYM_REVISION,
            "path": GYM_FIXTURE_PATH,
            "blob": GYM_FIXTURE_BLOB,
            "canonical_json_sha256": expected_sha256,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return TaskSpec(
        id=f"nemo-predicted-action-{canonical_sha256({'responses_create_params': request})}",
        requirements=TaskRequirements(),
        resources=(
            Resource("source-row.json", (ResourceRole.VERIFIER,), Embedded(private_row)),
            Resource("source-provenance.json", (ResourceRole.VERIFIER,), Embedded(source_provenance)),
        ),
        metadata=TaskMetadata(source=source, competencies=("function_calling",), task_shape="predicted_action"),
        steps=(
            StepSpecification(
                instructions=instructions,
                verifier=PredictedActionVerifier(expected, comparator, REVISION),
                answer_requirements=AnswerRequirements("text"),
            ),
        ),
    )


def rendering(row: dict[str, Any], expected_sha256: str) -> Rendering:
    """Build the public native output contract from source-advertised functions only."""
    if canonical_sha256(row) != expected_sha256:
        raise ValueError("source row does not match its pinned canonical hash")
    request = row.get("responses_create_params")
    if not isinstance(request, dict):
        raise ValueError("responses_create_params must be an object")
    return Rendering(
        id="nemo-native-final-action",
        submission=FinalActionSubmission(_functions(request.get("tools"))),
    )


def replay_action(name: str, arguments: str) -> dict[str, Any]:
    """Construct a Harbor replay action in the native Chat Completions wire shape."""
    return {
        "tool_calls": [{"id": "replay-action", "type": "function", "function": {"name": name, "arguments": arguments}}]
    }
