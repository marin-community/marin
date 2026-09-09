# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Normalize structured conversations into Datakit's canonical chat artifact."""

import json
import re
from collections.abc import Callable, Iterator
from typing import Any

import dupekit
from fray.types import ResourceConfig
from rigging.filesystem.storage_path import prefix_join
from zephyr import counters
from zephyr.context import ZephyrContext
from zephyr.dataset import Dataset
from zephyr.readers import load_file

from marin.datakit.chat import to_harmony_messages
from marin.datakit.download.rollout_transforms import (
    CHAT_CONTROL_TOKEN,
    REASONING_TOKEN,
    canonical_chat_messages,
    inferred_tool_definitions,
)
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

CHAT_NORMALIZE_VERSION = "2026.09.08.harmony"
MAX_REJECTED_RECORD_FRACTION = 0.05
_INLINE_TOOL_SYNTAX = re.compile(r"<tool_call(?::[^>]*)?>", re.IGNORECASE)
_TOOL_RESPONSE_SYNTAX = re.compile(r"</?tool_response(?:\s|>)", re.IGNORECASE)
_RAW_REASONING_TOKEN = re.compile(r"</?think>", re.IGNORECASE)
_SAFE_TOOL_IDENTIFIER = re.compile(r"[A-Za-z0-9_.:-]+")


def _contains_unsafe_markup(value: object) -> bool:
    if isinstance(value, str):
        return bool(
            CHAT_CONTROL_TOKEN.search(value) or REASONING_TOKEN.search(value) or _RAW_REASONING_TOKEN.search(value)
        )
    if isinstance(value, dict):
        return any(_contains_unsafe_markup(key) or _contains_unsafe_markup(item) for key, item in value.items())
    if isinstance(value, list):
        return any(_contains_unsafe_markup(item) for item in value)
    return False


def _tool_name(tool: dict) -> str | None:
    name = tool.get("name")
    if isinstance(name, str):
        return name
    function = tool.get("function")
    return function.get("name") if isinstance(function, dict) and isinstance(function.get("name"), str) else None


def _validate_tools(tools: list[dict]) -> set[str]:
    names: set[str] = set()
    for tool in tools:
        if not isinstance(tool, dict):
            raise ValueError("Tool definitions must be JSON objects")
        if _contains_unsafe_markup(tool):
            raise ValueError("Tool definitions must not contain chat control or reasoning tokens")
        name = _tool_name(tool)
        if not name:
            raise ValueError("Every tool definition must have a non-empty name")
        if _SAFE_TOOL_IDENTIFIER.fullmatch(name) is None:
            raise ValueError(f"Tool definition names contain unsafe characters: {name!r}")
        if name in names:
            raise ValueError(f"Tool definition names must be unique: {name!r}")
        names.add(name)
        parameters = tool.get("parameters")
        if parameters is None and isinstance(tool.get("function"), dict):
            parameters = tool["function"].get("parameters")
        if parameters is not None and not isinstance(parameters, dict):
            raise ValueError(f"Tool definition {name!r} parameters must be a JSON object")
    return names


def _validate_reasoning(content: str) -> None:
    start = "<|start_think|>"
    end = "<|end_think|>"
    if start not in content and end not in content:
        return
    if content.count(start) != 1 or content.count(end) != 1:
        raise ValueError("Assistant reasoning must contain exactly one balanced delimiter pair")
    if not content.startswith(start) or content.index(end) <= len(start):
        raise ValueError("Assistant reasoning must be a non-empty prefix of the reply")


def validate_chat_messages(messages: list[dict], tools: list[dict]) -> None:
    """Validate turn order, reasoning spans, and tool-call linkage."""
    if not messages:
        raise ValueError("A chat record must contain messages")

    tool_names = _validate_tools(tools)
    pending_calls: dict[str, str] = {}
    seen_call_ids: set[str] = set()
    seen_non_system = False
    previous_role: str | None = None
    for message in messages:
        role = message["role"]
        content = message.get("content")
        if role not in {"assistant", "developer", "system", "tool", "user"}:
            raise ValueError(f"Unsupported canonical chat role {role!r}")
        if isinstance(content, str) and CHAT_CONTROL_TOKEN.search(content):
            raise ValueError("Message content must not contain tokenizer control tokens")
        if isinstance(content, str) and _RAW_REASONING_TOKEN.search(content):
            raise ValueError("Raw reasoning tags must be normalized before chat validation")
        if role != "assistant" and isinstance(content, str) and REASONING_TOKEN.search(content):
            raise ValueError("Reasoning delimiters are only valid in assistant messages")
        reasoning = message.get("reasoning_content")
        if reasoning is not None:
            if role != "assistant" or not isinstance(reasoning, str):
                raise ValueError("reasoning_content is only valid as assistant text")
            if _contains_unsafe_markup(reasoning):
                raise ValueError("reasoning_content must contain plain text without control or reasoning tokens")
        if role in {"system", "developer"}:
            if seen_non_system:
                raise ValueError("System and developer messages must precede all conversation turns")
            continue
        if not seen_non_system:
            if role != "user":
                raise ValueError("The first non-system message must be a user message")
            seen_non_system = True

        if role == "user":
            if not isinstance(content, str) or not content.strip():
                raise ValueError("User messages must contain non-empty text")
            if pending_calls:
                raise ValueError("A user turn cannot replace a pending tool observation")
            if previous_role == "user":
                raise ValueError("Consecutive user turns must be merged by the source adapter")
        elif role == "assistant":
            if pending_calls:
                raise ValueError("Every tool call must be followed by its observations before another assistant turn")
            if previous_role == "assistant":
                raise ValueError("Consecutive assistant turns must be merged by the source adapter")
            if not isinstance(content, str) and content is not None:
                raise ValueError("Assistant content must be text or null")
            if isinstance(content, str):
                _validate_reasoning(content)
                if _INLINE_TOOL_SYNTAX.search(content):
                    raise ValueError("Inline tool-call syntax must be split out by the source adapter")
            calls = message.get("tool_calls") or []
            if _contains_unsafe_markup(calls):
                raise ValueError("Tool calls must not contain chat control or reasoning tokens")
            if (
                not calls
                and not (reasoning and reasoning.strip())
                and (not isinstance(content, str) or not content.strip())
            ):
                raise ValueError("Assistant turns must contain text, reasoning, or a tool call")
            for call in calls:
                call_id = call.get("id")
                function = call.get("function") or {}
                name = function.get("name")
                if not isinstance(call_id, str) or call_id in seen_call_ids:
                    raise ValueError("Tool-call IDs must be present and unique")
                if _SAFE_TOOL_IDENTIFIER.fullmatch(call_id) is None:
                    raise ValueError("Tool-call IDs contain unsafe characters")
                seen_call_ids.add(call_id)
                arguments = function.get("arguments")
                if isinstance(arguments, str):
                    arguments = json.loads(arguments)
                if not isinstance(arguments, dict):
                    raise ValueError("Tool-call arguments must encode a JSON object")
                if not isinstance(name, str):
                    raise ValueError("Tool calls must name a function")
                if _SAFE_TOOL_IDENTIFIER.fullmatch(name) is None:
                    raise ValueError("Tool-call names contain unsafe characters")
                if name not in tool_names:
                    raise ValueError(f"Tool call {name!r} has no matching tool definition")
                pending_calls[call_id] = name
        elif role == "tool":
            if not isinstance(content, str):
                raise ValueError("Tool observations must contain text")
            if _INLINE_TOOL_SYNTAX.search(content) or _TOOL_RESPONSE_SYNTAX.search(content):
                raise ValueError("Tool observations must not contain chat protocol wrappers")
            call_id = message.get("tool_call_id")
            if call_id not in pending_calls:
                raise ValueError("Tool observations must reference a pending call")
            if message.get("name") not in (None, pending_calls[call_id]):
                raise ValueError("Tool observation name must match its call")
            del pending_calls[call_id]
        previous_role = role

    if not seen_non_system or messages[-1]["role"] != "assistant":
        raise ValueError("A chat training record must end with an assistant response")


def _normalize_chat_record(record: dict[str, Any], messages_field: str, id_field: str) -> dict[str, Any]:
    messages_value = record[messages_field]
    if not isinstance(messages_value, list):
        raise ValueError(f"{messages_field!r} must be a list")
    messages = canonical_chat_messages(messages_value)

    raw_kwargs = record.get("chat_template_kwargs") or {}
    if isinstance(raw_kwargs, str):
        raw_kwargs = json.loads(raw_kwargs)
    if not isinstance(raw_kwargs, dict):
        raise ValueError("chat_template_kwargs must be a JSON object")
    kwargs = dict(raw_kwargs)
    if _contains_unsafe_markup({key: value for key, value in kwargs.items() if key != "tools"}):
        raise ValueError("Chat template arguments must not contain chat control or reasoning tokens")
    tools = list(kwargs.get("tools") or [])
    existing_names = {_tool_name(tool) for tool in tools if isinstance(tool, dict)}
    tools.extend(tool for tool in inferred_tool_definitions(messages) if tool["name"] not in existing_names)
    if tools:
        kwargs["tools"] = tools
    validate_chat_messages(messages, tools)

    messages = to_harmony_messages(messages)

    source_id = record.get(id_field)
    out = {key: value for key, value in record.items() if key not in {id_field, messages_field, "chat_template_kwargs"}}
    identity = json.dumps(
        {"messages": messages, "chat_template_kwargs": kwargs},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    out["id"] = format(dupekit.hash_xxh3_128(identity), "032x")
    out["messages"] = messages
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
        .map_shard(_make_split_writer(output_dir))
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
) -> NormalizedData:
    """Normalize source conversations into deduplicated Harmony-message Parquet."""
    resources = worker_resources or ResourceConfig(cpu=2, ram="32g", disk="10g")
    file_sizes = _discover_files(input_path, file_extensions=file_extensions)
    if not file_sizes:
        raise FileNotFoundError(f"No data files found under {input_path}")
    num_shards = max(1, sum(file_sizes.values()) // target_partition_bytes)
    pipeline = _build_chat_pipeline(list(file_sizes), output_path, num_shards, messages_field, id_field, dedup_mode)
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
) -> StepSpec:
    """Create a versioned Harmony-message normalization step."""
    hash_attrs = {
        "version": CHAT_NORMALIZE_VERSION,
        "messages_field": messages_field,
        "id_field": id_field,
        "target_partition_bytes": target_partition_bytes,
        "file_extensions": file_extensions,
        "dedup_mode": dedup_mode,
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
        ),
        deps=[download],
        hash_attrs=hash_attrs,
    )
