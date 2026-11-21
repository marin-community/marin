# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Row conversion and transforms for nanoswe SWE-smith tool-call trajectories (``nanoswe/nanoswe-trajs-261002``).

Every trajectory comes from one teacher, Nemotron 3.5 Lightning, acting through the NanoSWE harness's two
tools, ``bash`` and ``file_editor``. ``messages`` starts with the bare issue as a user turn (the teacher's
system prompt and tool schemas are stripped upstream) and then alternates assistant and user turns. An
assistant turn has empty ``content`` and ``parts`` holding optional ``text`` parts, the teacher's reasoning,
followed by one ``python`` part: the tool call as JSON ``{"name": ..., "arguments": ...}``. The user turn
after it holds the tool result. Every conversation ends on an assistant tool call.

Two renderings are produced. The pretraining text is role-tagged turns in which each tool call's arguments
appear as one JSON object, so multi-line edit strings keep their boundaries. It carries no outcome tag: a tag
derived from the eventual grade would tell a teacher-forced evaluation how the trajectory ends. The
structured chat declares the harness's own tool definitions. Both renderings keep only the ``bash``
argument the harness forwards (``command``) and drop calls whose ``file_editor`` arguments the harness
schema rejects. Both keep ``instance_id`` and ``verified`` as metadata, so a consumer can select resolved
rollouts and hold out tasks; rows without a task id or a boolean grade are dropped and counted.
"""

import json
from enum import StrEnum
from typing import NamedTuple

import pyarrow as pa
from fray.types import ResourceConfig
from rigging.filesystem.storage_path import prefix_join
from zephyr import counters
from zephyr.context import ZephyrContext
from zephyr.dataset import Dataset

from marin.datakit.chat_normalize import CHAT_SCHEMA
from marin.datakit.download.rollout_transforms import (
    checked_openai_chat_document,
    load_parquet_batched,
    render_role_message,
    text_document,
)

SOURCE_NAME = "nanoswe-lightning"
SOURCE_ID = "nanoswe/nanoswe-trajs-261002"
# Keys the processed outputs; bump it with any change to how rows convert. v3 checks every file_editor argument
# against the harness schema; v2 dropped the outcome tag, wrote tool-call arguments as JSON and pinned the harness
# tool definitions.
TRANSFORM_VERSION = "v3"

# Columns carried from the source row into every output document.
METADATA_FIELDS = ("instance_id", "verified", "origin", "exit_status", "traj_hash", "n_toolcalls")
METADATA_SCHEMA_FIELDS = (
    pa.field("instance_id", pa.string()),
    pa.field("verified", pa.bool_()),
    pa.field("origin", pa.string()),
    pa.field("exit_status", pa.string()),
    pa.field("traj_hash", pa.string()),
    pa.field("n_toolcalls", pa.int32()),
)
SOURCE_CHAT_SCHEMA = pa.schema([*CHAT_SCHEMA, *METADATA_SCHEMA_FIELDS])

# Tool definitions copied from the NanoSWE harness, nanosweb/nanoswe@e70bf717b3be3bbbeab543f7a3b948101b19990d,
# eval/minisweagent/models/utils/actions_toolcall.py (MIT).
BASH_TOOL = {
    "type": "function",
    "function": {
        "name": "bash",
        "description": "Run a shell command.",
        "parameters": {
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
        },
    },
}
FILE_EDITOR_TOOL = {
    "type": "function",
    "function": {
        "name": "file_editor",
        "description": (
            "View and edit UTF-8 text files using absolute paths. create refuses existing paths. str_replace "
            "requires old_str to match exactly once; include enough unchanged context to make it unique. "
            "undo_edit reverts the latest edit to path."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "enum": ["view", "create", "str_replace", "insert", "undo_edit"]},
                "path": {"type": "string", "description": "Absolute file or directory path."},
                "file_text": {"type": "string", "description": "Content for create."},
                "old_str": {"type": "string", "description": "Exact unique text for str_replace."},
                "new_str": {"type": "string", "description": "New text for str_replace or insert."},
                "insert_line": {
                    "type": "integer",
                    "minimum": 0,
                    "description": "For insert: add new_str after this line; 0 means file start.",
                },
                "view_range": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "minItems": 2,
                    "maxItems": 2,
                    "description": "For view: 1-based inclusive [start, end]; -1 end means EOF.",
                },
            },
            "required": ["command", "path"],
            "additionalProperties": False,
        },
    },
}
TOOLS = (BASH_TOOL, FILE_EDITOR_TOOL)
_FILE_EDITOR_SCHEMA = FILE_EDITOR_TOOL["function"]["parameters"]
_SHARD_TEMPLATE = "data-{shard:05d}-of-{total:05d}.parquet"


class Split(StrEnum):
    TRAIN = "train"
    VALIDATION = "validation"


SPLIT_GLOBS = {Split.TRAIN: "train-*.parquet", Split.VALIDATION: "val/*.parquet"}


def _metadata(row: dict) -> dict:
    return {field: row.get(field) for field in METADATA_FIELDS}


def _has_task_and_grade(row: dict) -> bool:
    instance_id = row.get("instance_id")
    return isinstance(instance_id, str) and bool(instance_id) and isinstance(row.get("verified"), bool)


class ToolCall(NamedTuple):
    name: str
    arguments: dict
    reasoning: str


def _matches(value: object, schema: dict) -> bool:
    """Whether ``value`` satisfies the subset of JSON Schema the harness tool definitions use."""
    kind = schema.get("type")
    if kind == "string" and not isinstance(value, str):
        return False
    if kind == "integer" and (not isinstance(value, int) or isinstance(value, bool)):
        return False
    if kind == "array":
        if not isinstance(value, list) or not schema.get("minItems", 0) <= len(value) <= schema.get(
            "maxItems", len(value)
        ):
            return False
        if "items" in schema and not all(_matches(item, schema["items"]) for item in value):
            return False
    if "enum" in schema and value not in schema["enum"]:
        return False
    return "minimum" not in schema or value >= schema["minimum"]


def _harness_arguments(name: str, arguments: dict) -> dict | None:
    """Return the arguments the harness would act on, or None if the harness would reject the call."""
    if name == "bash":
        command = arguments.get("command")
        if not isinstance(command, str):
            return None
        if len(arguments) > 1:
            counters.pipeline.update_counter(f"{SOURCE_NAME}/bash_extra_arguments_dropped", 1)
        return {"command": command}
    if name == "file_editor":
        properties = _FILE_EDITOR_SCHEMA["properties"]
        if not set(arguments) <= set(properties) or not set(_FILE_EDITOR_SCHEMA["required"]) <= set(arguments):
            return None
        if not all(_matches(value, properties[key]) for key, value in arguments.items()):
            return None
        return arguments
    return None


def _tool_call(message: dict) -> ToolCall | None:
    """Return an assistant turn's tool call, or None if it is malformed."""
    parts = message.get("parts") or []
    if not parts or parts[-1].get("type") != "python" or any(part.get("type") != "text" for part in parts[:-1]):
        return None
    try:
        call = json.loads(parts[-1].get("text") or "")
    except json.JSONDecodeError:
        return None
    if not isinstance(call, dict):
        return None
    name, arguments = call.get("name"), call.get("arguments")
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            return None
    if not isinstance(name, str) or not isinstance(arguments, dict):
        return None
    arguments = _harness_arguments(name, arguments)
    if arguments is None:
        return None
    reasoning = "\n\n".join(text for text in ((part.get("text") or "").strip() for part in parts[:-1]) if text)
    return ToolCall(name, arguments, reasoning)


def _openai_messages(row: dict) -> list[dict] | None:
    """Convert a source trajectory to OpenAI-style messages, or None if its turn structure is malformed."""
    messages = row.get("messages") or []
    if len(messages) < 2 or messages[0].get("role") != "user" or not messages[0].get("content"):
        return None
    converted: list[dict] = [{"role": "user", "content": messages[0]["content"]}]
    pending: tuple[str, str] | None = None
    for index, message in enumerate(messages[1:], start=1):
        role = message.get("role")
        if role == "assistant" and pending is None:
            call = _tool_call(message)
            if call is None:
                return None
            call_id = f"call_{index}"
            converted.append(
                {
                    "role": "assistant",
                    "content": "",
                    "reasoning_content": call.reasoning,
                    "tool_calls": [
                        {"id": call_id, "type": "function", "function": {"name": call.name, "arguments": call.arguments}}
                    ],
                }
            )
            pending = (call_id, call.name)
        elif role == "user" and pending is not None:
            call_id, name = pending
            converted.append(
                {"role": "tool", "content": message.get("content") or "", "name": name, "tool_call_id": call_id}
            )
            pending = None
        else:
            return None
    if converted[-1]["role"] != "assistant":
        return None
    return converted


def _render_assistant(message: dict) -> str:
    function = message["tool_calls"][0]["function"]
    name = function["name"]
    lines = [message["reasoning_content"]] if message["reasoning_content"] else []
    lines.append(f"<tool_call:{name}>\n{json.dumps(function['arguments'], ensure_ascii=False)}\n</tool_call:{name}>")
    return "<assistant>\n" + "\n".join(lines) + "\n</assistant>"


def _converted_messages(row: dict, counter: str) -> list[dict] | None:
    if not _has_task_and_grade(row):
        counters.pipeline.update_counter(f"{SOURCE_NAME}/missing_task_or_grade_filtered", 1)
        return None
    messages = _openai_messages(row)
    if messages is None:
        counters.pipeline.update_counter(f"{SOURCE_NAME}/{counter}", 1)
    return messages


def row_to_doc(row: dict) -> list[dict]:
    """Render one trajectory as role-tagged pretraining text with JSON tool-call arguments."""
    messages = _converted_messages(row, "malformed_filtered")
    if messages is None:
        return []
    blocks = [
        _render_assistant(message) if message["role"] == "assistant" else render_role_message(message)
        for message in messages
    ]
    counters.pipeline.update_counter(f"{SOURCE_NAME}/kept", 1)
    return [{**text_document("\n\n".join(blocks), SOURCE_ID), **_metadata(row)}]


def row_to_chat_doc(row: dict) -> list[dict]:
    """Convert one trajectory to a Harmony chat document with the harness's tool definitions and metadata."""
    messages = _converted_messages(row, "chat_malformed_filtered")
    if messages is None:
        return []
    return checked_openai_chat_document(
        messages,
        SOURCE_ID,
        counter_prefix=f"{SOURCE_NAME}/chat",
        chat_template_kwargs={"tools": list(TOOLS)},
        **_metadata(row),
    )


def transform(input_path: str, output_path: str, split: Split) -> None:
    pipeline = (
        Dataset.from_files(prefix_join(input_path, SPLIT_GLOBS[split]))
        .flat_map(load_parquet_batched)
        .flat_map(row_to_doc)
        .write_parquet(prefix_join(output_path, _SHARD_TEMPLATE), skip_existing=True)
    )
    ZephyrContext(name=f"{SOURCE_NAME}-{split}-transform", resources=ResourceConfig(cpu=1, ram="16g")).execute(pipeline)


def transform_chat(input_path: str, output_path: str, split: Split) -> None:
    pipeline = (
        Dataset.from_files(prefix_join(input_path, SPLIT_GLOBS[split]))
        .flat_map(load_parquet_batched)
        .flat_map(row_to_chat_doc)
        .write_parquet(prefix_join(output_path, _SHARD_TEMPLATE), schema=SOURCE_CHAT_SCHEMA, skip_existing=True)
    )
    ZephyrContext(name=f"{SOURCE_NAME}-{split}-chat-transform", resources=ResourceConfig(cpu=1, ram="16g")).execute(
        pipeline
    )
