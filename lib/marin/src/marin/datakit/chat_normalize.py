# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Normalize structured conversations into Datakit's canonical chat artifact."""

import json
from collections.abc import Callable, Iterator
from typing import Any

import dupekit
import pyarrow as pa
from fray.types import ResourceConfig
from openai_harmony import Message
from rigging.filesystem.storage_path import prefix_join
from zephyr import counters
from zephyr.context import ZephyrContext
from zephyr.dataset import Dataset
from zephyr.readers import load_file

from marin.datakit.chat import CHAT_SCHEMA, validate_chat_messages, validate_tool_definitions
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

CHAT_NORMALIZE_VERSION = "2026.09.09.explicit-tools"
MAX_REJECTED_RECORD_FRACTION = 0.05


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
