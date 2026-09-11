# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Normalize structured conversations into Datakit's canonical chat artifact."""

import json
import re
from collections import deque
from collections.abc import Callable, Iterator
from enum import StrEnum
from typing import Any

import dupekit
import pyarrow as pa
from fray.types import ResourceConfig
from openai_harmony import Message, Role, TextContent
from rigging.filesystem.storage_path import prefix_join
from zephyr import counters
from zephyr.context import ZephyrContext
from zephyr.dataset import Dataset
from zephyr.readers import load_file

from marin.datakit.normalize import (
    DEFAULT_MAX_WORKERS,
    DedupMode,
    ExactDupSideOutput,
    MainOutput,
    NormalizedData,
    _discover_files,
    _make_split_writer,
)
from marin.execution.step_spec import StepSpec

CHAT_NORMALIZE_VERSION = "2026.09.11.source-provenance"
MAX_REJECTED_RECORD_FRACTION = 0.05


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
        match role:
            case Role.SYSTEM | Role.DEVELOPER:
                if seen_user:
                    raise ValueError("System and developer messages must precede all conversation turns")
                if message.channel is not None or message.recipient is not None:
                    raise ValueError("Instruction messages cannot have channels or recipients")
            case _ if not seen_user and role != Role.USER:
                raise ValueError("The first conversation message must be a user message")
            case Role.USER:
                seen_user = True
                if not text.strip():
                    raise ValueError("User messages must contain non-empty text")
                if message.channel is not None or message.recipient is not None:
                    raise ValueError("User messages cannot have channels or recipients")
                if pending:
                    raise ValueError("A user turn cannot replace a pending tool observation")
                if previous is not None and previous.author.role == Role.USER:
                    raise ValueError("Consecutive user turns must be merged by the source adapter")
            case Role.ASSISTANT:
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
            case Role.TOOL:
                if message.channel != ChatChannel.COMMENTARY or message.recipient != Role.ASSISTANT.value:
                    raise ValueError("Tool observations require commentary addressed to assistant")
                if not pending or message.author.name != pending[0]:
                    raise ValueError("Tool observations must match pending calls in call order")
                pending.popleft()
        previous = message
    if not seen_user or messages[-1].author.role != Role.ASSISTANT:
        raise ValueError("A chat training record must end with an assistant response")


def validate_tool_definitions(tools: list[dict], messages: list[Message]) -> None:
    """Require explicit definitions for calls without rewriting their arguments."""
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
        if not isinstance(parameters, dict):
            raise ValueError(f"Tool definition {name!r} parameters must be a JSON object")
    for message in messages:
        if message.author.role == Role.ASSISTANT and message.recipient is not None:
            name = message.recipient.removeprefix("functions.")
            if name not in names:
                raise ValueError(f"Tool call {name!r} has no explicit definition")


def _normalize_chat_record(record: dict[str, Any], messages_field: str, id_field: str) -> dict[str, Any]:
    messages_value = record[messages_field]
    if not isinstance(messages_value, list):
        raise ValueError(f"{messages_field!r} must be a list")
    # Require the canonical serialized shape, rather than Harmony's string-content shorthand.
    if any(not isinstance(message, dict) or not isinstance(message.get("content"), list) for message in messages_value):
        raise ValueError("Source adapters must emit Harmony messages with text content parts")
    for message in messages_value:
        if "role" not in message or any(
            not isinstance(part, dict) or part.get("type") != "text" or not isinstance(part.get("text"), str)
            for part in message["content"]
        ):
            raise ValueError("Harmony messages require a role and text content parts")
        if any(
            message.get(field) is not None and not isinstance(message[field], str) for field in ("channel", "recipient")
        ):
            raise ValueError("Harmony channels and recipients must be strings")
        if {"tool_calls", "tool_call_id", "reasoning_content", "function_call"} & message.keys():
            raise ValueError("Source adapters must emit Harmony channels and recipients")
    messages = [Message.from_dict(message) for message in messages_value]
    validate_chat_messages(messages)

    raw_kwargs = record.get("chat_template_kwargs") or {}
    if isinstance(raw_kwargs, str):
        raw_kwargs = json.loads(raw_kwargs)
    if not isinstance(raw_kwargs, dict):
        raise ValueError("chat_template_kwargs must be a JSON object")
    kwargs = dict(raw_kwargs)
    tools = kwargs.get("tools", [])
    if not isinstance(tools, list):
        raise ValueError("tools must be a list of function definitions")
    validate_tool_definitions(tools, messages)
    serialized_messages = [message.to_dict() for message in messages]

    source_id = record.get("source_id")
    if source_id is None:
        source_id = record.get(id_field)
    out = {key: value for key, value in record.items() if key not in {id_field, messages_field, "chat_template_kwargs"}}
    identity = json.dumps(
        {"messages": serialized_messages, "chat_template_kwargs": kwargs},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    out["id"] = format(dupekit.hash_xxh3_128(identity), "032x")
    out["messages"] = serialized_messages
    if kwargs:
        out["chat_template_kwargs"] = json.dumps(kwargs, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    if source_id is not None:
        out["source_id"] = source_id
    return out


def _build_chat_pipeline(
    files: list[str],
    output_dir: str,
    num_shards: int,
    messages_field: str,
    id_field: str,
    dedup_mode: DedupMode,
    output_schema: pa.Schema,
) -> Dataset:
    def normalize_record(record: dict[str, Any]) -> list[dict[str, Any]]:
        try:
            normalized = _normalize_chat_record(record, messages_field, id_field)
        except (UnicodeError, ValueError) as error:
            counters.pipeline.update_counter("normalize_chat/records_quarantined", 1)
            counters.pipeline.update_counter(f"normalize_chat/quarantined/{type(error).__name__}", 1)
            return []
        counters.pipeline.update_counter("normalize_chat/records_validated", 1)
        return [normalized]

    def dedup(_key: str, items: Iterator[dict[str, Any]]) -> Iterator[MainOutput | ExactDupSideOutput]:
        previous_id: str | None = None
        for record in items:
            if record["id"] != previous_id:
                previous_id = record["id"]
                yield MainOutput(data=record)
            else:
                yield ExactDupSideOutput(data=record)

    def passthrough(_key: str, items: Iterator[dict[str, Any]]) -> Iterator[MainOutput]:
        yield from (MainOutput(data=item) for item in items)

    def has_messages(record: dict[str, Any]) -> bool:
        messages = record.get(messages_field)
        if not isinstance(messages, list) or not messages:
            counters.pipeline.update_counter("normalize_chat/empty_messages_filtered", 1)
            return False
        return True

    reducers: dict[DedupMode, Callable] = {DedupMode.EXACT: dedup, DedupMode.NONE: passthrough}
    return (
        Dataset.from_list(files)
        .flat_map(load_file)
        .filter(has_messages)
        .flat_map(normalize_record)
        .group_by(
            key=lambda record: record["id"],
            reducer=reducers[dedup_mode],
            sort_by=lambda record: record["id"],
            num_output_shards=num_shards,
        )
        .map_shard(_make_split_writer(output_dir, output_schema=output_schema))
    )


def normalize_chat_to_parquet(
    *,
    input_path: str,
    output_path: str,
    messages_field: str = "messages",
    id_field: str = "id",
    target_partition_bytes: int = 256 * 1024 * 1024,
    worker_resources: ResourceConfig | None = None,
    max_workers: int = DEFAULT_MAX_WORKERS,
    file_extensions: tuple[str, ...] | None = None,
    dedup_mode: DedupMode = DedupMode.EXACT,
    output_schema: pa.Schema = CHAT_SCHEMA,
) -> NormalizedData:
    """Normalize source conversations into deduplicated Harmony-message Parquet."""
    resources = worker_resources or ResourceConfig(cpu=2, ram="32g", disk="10g")
    file_sizes = _discover_files(input_path, file_extensions=file_extensions)
    if not file_sizes:
        raise FileNotFoundError(f"No data files found under {input_path}")
    num_shards = max(1, sum(file_sizes.values()) // target_partition_bytes)
    pipeline = _build_chat_pipeline(
        list(file_sizes), output_path, num_shards, messages_field, id_field, dedup_mode, output_schema
    )
    outcome = ZephyrContext(name="normalize-chat", resources=resources, max_workers=max_workers).execute(pipeline)
    counters_dict = dict(outcome.counters)
    total_in = counters_dict.get("zephyr/records_in", 0)
    if total_in > 0 and counters_dict.get("normalize_chat/empty_messages_filtered", 0) == total_in:
        raise ValueError(f"All {total_in} records were filtered because {messages_field!r} was empty or missing")
    if not counters_dict.get("normalize_chat/records_validated", 0):
        raise ValueError(f"Chat source {input_path} contained no valid records")
    rejected = counters_dict.get("normalize_chat/empty_messages_filtered", 0) + counters_dict.get(
        "normalize_chat/records_quarantined", 0
    )
    if total_in and rejected / total_in > MAX_REJECTED_RECORD_FRACTION:
        raise ValueError(
            f"Chat source {input_path} rejected {rejected}/{total_in} records, above the "
            f"{MAX_REJECTED_RECORD_FRACTION:.0%} health limit"
        )
    return NormalizedData(
        main_output_dir=prefix_join(output_path, "outputs/main"),
        dup_output_dir=prefix_join(output_path, "outputs/dups"),
        counters=counters_dict,
    )


def normalize_chat_step(
    *,
    name: str,
    download: StepSpec,
    messages_field: str = "messages",
    id_field: str = "id",
    target_partition_bytes: int = 256 * 1024 * 1024,
    worker_resources: ResourceConfig | None = None,
    max_workers: int = DEFAULT_MAX_WORKERS,
    file_extensions: tuple[str, ...] | None = None,
    dedup_mode: DedupMode = DedupMode.EXACT,
    output_schema: pa.Schema = CHAT_SCHEMA,
) -> StepSpec:
    """Create a versioned Harmony-message normalization step."""
    hash_attrs = {
        "version": CHAT_NORMALIZE_VERSION,
        "messages_field": messages_field,
        "id_field": id_field,
        "target_partition_bytes": target_partition_bytes,
        "file_extensions": file_extensions,
        "dedup_mode": dedup_mode,
        "output_schema": str(output_schema),
    }
    return StepSpec(
        name=name,
        fn=lambda output_path: normalize_chat_to_parquet(
            input_path=download.output_path,
            output_path=output_path,
            messages_field=messages_field,
            id_field=id_field,
            target_partition_bytes=target_partition_bytes,
            worker_resources=worker_resources,
            max_workers=max_workers,
            file_extensions=file_extensions,
            dedup_mode=dedup_mode,
            output_schema=output_schema,
        ),
        deps=[download],
        hash_attrs=hash_attrs,
    )
