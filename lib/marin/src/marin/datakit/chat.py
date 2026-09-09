# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Datakit chat sources and validation of canonical Harmony conversations."""

import json
import re
from collections import deque
from dataclasses import dataclass
from enum import StrEnum

import pyarrow as pa
from openai_harmony import Message, Role, TextContent

from marin.execution.step_spec import StepSpec

_SAFE_TOOL_IDENTIFIER = re.compile(r"[A-Za-z0-9_.:-]+")


CHAT_MESSAGE_TYPE = pa.struct(
    [
        pa.field("role", pa.string(), nullable=False),
        pa.field("name", pa.string()),
        pa.field("channel", pa.string()),
        pa.field("recipient", pa.string()),
        pa.field(
            "content",
            pa.list_(
                pa.struct(
                    [
                        pa.field("type", pa.string(), nullable=False),
                        pa.field("text", pa.string(), nullable=False),
                    ]
                )
            ),
            nullable=False,
        ),
    ]
)
CHAT_SCHEMA = pa.schema(
    [
        pa.field("id", pa.string(), nullable=False),
        pa.field("messages", pa.list_(CHAT_MESSAGE_TYPE), nullable=False),
        pa.field("source", pa.string()),
        pa.field("source_id", pa.string()),
        pa.field("chat_template_kwargs", pa.string()),
    ]
)


class ChatChannel(StrEnum):
    """Harmony channels emitted by Datakit chat normalization."""

    ANALYSIS = "analysis"
    COMMENTARY = "commentary"
    FINAL = "final"


@dataclass(frozen=True)
class DatakitChatSource:
    """A source whose normalized artifact contains structured Harmony messages."""

    name: str
    normalize_steps: tuple[StepSpec, ...]
    rough_token_count_b: float

    @property
    def normalized(self) -> StepSpec:
        return self.normalize_steps[-1]


def message_text(message: Message) -> str:
    """Read the text-only content supported by Datakit chat artifacts."""
    if not message.content or any(not isinstance(part, TextContent) for part in message.content):
        raise ValueError("Datakit chat messages require text content parts")
    return "".join(part.text for part in message.content)


def validate_chat_messages(messages: list[Message]) -> None:
    """Validate Harmony channels, conversation order, and function handoffs."""
    if not messages:
        raise ValueError("A chat record must contain messages")
    pending: deque[str] = deque()
    previous: Message | None = None
    seen_user = False
    for message in messages:
        role = message.author.role
        text = message_text(message)
        if message.content_type is not None:
            raise ValueError("Datakit text messages do not support content_type")
        if role in {Role.SYSTEM, Role.DEVELOPER}:
            if seen_user:
                raise ValueError("System and developer messages must precede all conversation turns")
            if message.channel is not None or message.recipient is not None:
                raise ValueError("Instruction messages cannot have channels or recipients")
            continue
        if not seen_user:
            if role != Role.USER:
                raise ValueError("The first conversation message must be a user message")
            seen_user = True
        if role == Role.USER:
            if not text.strip():
                raise ValueError("User messages must contain non-empty text")
            if message.channel is not None or message.recipient is not None:
                raise ValueError("User messages cannot have channels or recipients")
            if pending:
                raise ValueError("A user turn cannot replace a pending tool observation")
            if previous is not None and previous.author.role == Role.USER:
                raise ValueError("Consecutive user turns must be merged by the source adapter")
        elif role == Role.ASSISTANT:
            if not text.strip():
                raise ValueError("Assistant messages must contain non-empty text")
            channel = ChatChannel(message.channel)
            if previous is not None and previous.channel == ChatChannel.FINAL:
                raise ValueError("An assistant final answer must be followed by a user turn")
            if message.recipient is not None:
                if channel != ChatChannel.COMMENTARY or not message.recipient.startswith("functions."):
                    raise ValueError("Function calls require commentary and a functions.<name> recipient")
                name = message.recipient.removeprefix("functions.")
                if _SAFE_TOOL_IDENTIFIER.fullmatch(name) is None:
                    raise ValueError("Function calls require a valid tool name")
                if pending and previous is not None and previous.author.role == Role.TOOL:
                    raise ValueError("Every pending tool call must receive an observation before another call")
                if not isinstance(json.loads(text), dict):
                    raise ValueError("Tool-call arguments must encode a JSON object")
                pending.append(message.recipient)
            elif pending:
                raise ValueError("Every tool call must receive an observation before the assistant continues")
        elif role == Role.TOOL:
            if message.channel != ChatChannel.COMMENTARY or message.recipient != Role.ASSISTANT.value:
                raise ValueError("Tool observations require commentary addressed to assistant")
            if not pending or message.author.name != pending.popleft():
                raise ValueError("Tool observations must match pending calls in call order")
        previous = message
    if not seen_user or messages[-1].author.role != Role.ASSISTANT:
        raise ValueError("A chat training record must end with an assistant response")


def inferred_tool_definitions(messages: list[Message]) -> list[dict]:
    """Build minimal function schemas from Harmony call arguments."""
    definitions: dict[str, dict] = {}
    for message in messages:
        if message.author.role != Role.ASSISTANT or message.recipient is None:
            continue
        name = message.recipient.removeprefix("functions.")
        properties = definitions.setdefault(name, {})
        for key, value in json.loads(message_text(message)).items():
            if isinstance(value, bool):
                json_type = "boolean"
            elif isinstance(value, (int, float)):
                json_type = "number"
            elif isinstance(value, list):
                json_type = "array"
            elif isinstance(value, dict):
                json_type = "object"
            else:
                json_type = "string"
            properties[key] = {"type": json_type}
    return [
        {
            "type": "function",
            "name": name,
            "description": f"Execute the {name} tool.",
            "parameters": {"type": "object", "properties": properties},
        }
        for name, properties in definitions.items()
    ]


def validate_tool_definitions(tools: list[dict]) -> None:
    """Check function definition names and parameter object structure."""
    names: set[str] = set()
    for tool in tools:
        if not isinstance(tool, dict):
            raise ValueError("Tool definitions must be JSON objects")
        function = tool.get("function", tool)
        if not isinstance(function, dict):
            raise ValueError("Function definitions must be JSON objects")
        name = function.get("name")
        if not isinstance(name, str) or _SAFE_TOOL_IDENTIFIER.fullmatch(name) is None:
            raise ValueError("Tool definitions require valid function names")
        if name in names:
            raise ValueError(f"Tool definition names must be unique: {name!r}")
        names.add(name)
        parameters = function.get("parameters")
        if parameters is not None and not isinstance(parameters, dict):
            raise ValueError(f"Tool definition {name!r} parameters must be a JSON object")
