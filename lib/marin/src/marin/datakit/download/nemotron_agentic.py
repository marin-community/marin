# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""nvidia/Nemotron-SFT-Agentic-v2 trajectory download and transform."""

import hashlib
import json
from collections.abc import Iterator
from enum import StrEnum
from html import escape
from typing import Any

import msgspec
from fray import ResourceConfig
from zephyr import Dataset, ZephyrContext, counters
from zephyr.readers import open_file

from marin.datakit.download.huggingface import download_hf_step
from marin.datakit.normalize import normalize_step
from marin.execution.step_spec import StepSpec

HF_DATASET_ID = "nvidia/Nemotron-SFT-Agentic-v2"
HF_REVISION = "49e79a3"
LICENSES = ("cc-by-4.0", "apache-2.0", "mit")
TRANSFORM_VERSION = "v1"


class NemotronAgenticSubset(StrEnum):
    TOOL_CALLING = "tool_calling"
    INTERACTIVE_AGENT = "interactive_agent"
    SEARCH = "search"


HF_FILE_BY_SUBSET = {
    NemotronAgenticSubset.TOOL_CALLING: "data/tool_calling.jsonl",
    NemotronAgenticSubset.INTERACTIVE_AGENT: "data/interactive_agent.jsonl",
    NemotronAgenticSubset.SEARCH: "data/search.jsonl",
}
SUPPORTED_SUBSETS = frozenset({NemotronAgenticSubset.TOOL_CALLING})
KNOWN_MALFORMED_JSONL_LINES_BY_SUBSET = {
    NemotronAgenticSubset.TOOL_CALLING: frozenset({1095}),
}


def stable_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _require_supported_subset(subset: NemotronAgenticSubset) -> None:
    if subset not in SUPPORTED_SUBSETS:
        raise ValueError(f"{subset.value} is not implemented yet; PR 1 supports only tool_calling")


def load_nemotron_agentic_jsonl(source: str, subset: NemotronAgenticSubset) -> Iterator[dict[str, Any]]:
    """Load a pinned Nemotron Agentic JSONL file, skipping known bad upstream rows."""
    skipped_lines = KNOWN_MALFORMED_JSONL_LINES_BY_SUBSET.get(subset, frozenset())
    decoder = msgspec.json.Decoder()

    with open_file(source, "rt") as f:
        for line_number, line in enumerate(f, start=1):
            if line_number in skipped_lines:
                counters.increment(f"nemotron_agentic/{subset.value}/malformed_source_line")
                continue

            line = line.strip()
            if not line:
                continue

            record = decoder.decode(line)
            if not isinstance(record, dict):
                raise ValueError(f"Line {line_number} in {source} must decode to a JSON object")
            counters.increment("zephyr/records_in")
            yield record


def _source_context(row: dict[str, Any]) -> str:
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    for key in ("uuid", "alt_id"):
        value = metadata.get(key)
        if isinstance(value, str) and value:
            return value
    value = row.get("uuid")
    if isinstance(value, str) and value:
        return value
    return "unknown"


def _render_value(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return stable_json(value)


def normalize_tool_call(call: dict[str, Any], *, source_id: str) -> dict[str, Any]:
    function = call.get("function")
    if not isinstance(function, dict):
        raise ValueError(f"Tool call in {source_id} is missing a function object")

    name = function.get("name")
    if not isinstance(name, str) or not name:
        raise ValueError(f"Tool call in {source_id} is missing function.name")

    arguments = function.get("arguments", {})
    if isinstance(arguments, dict):
        normalized_arguments = stable_json(arguments)
    elif isinstance(arguments, str):
        try:
            normalized_arguments = stable_json(json.loads(arguments))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Tool call in {source_id} has invalid JSON arguments") from exc
    else:
        raise ValueError(f"Tool call in {source_id} has unsupported arguments type {type(arguments).__name__}")

    normalized: dict[str, Any] = {
        "type": call.get("type", "function"),
        "function": {
            "name": name,
            "arguments": normalized_arguments,
        },
    }
    if call.get("id") is not None:
        normalized["id"] = call["id"]
    return normalized


def normalize_messages(row: dict[str, Any], subset: NemotronAgenticSubset) -> list[dict[str, Any]]:
    _require_supported_subset(subset)
    messages_raw = row.get("messages")
    if not messages_raw:
        return []
    if not isinstance(messages_raw, list):
        raise ValueError(f"Messages for {_source_context(row)} must be a list")

    source_id = _source_context(row)
    messages: list[dict[str, Any]] = []
    for index, raw_message in enumerate(messages_raw):
        if not isinstance(raw_message, dict):
            raise ValueError(f"Message {index} in {source_id} must be an object")

        role = raw_message.get("role")
        if not isinstance(role, str) or not role:
            raise ValueError(f"Message {index} in {source_id} is missing role")

        message: dict[str, Any] = {"role": role, "content": raw_message.get("content") or ""}
        if raw_message.get("tool_call_id") is not None:
            message["tool_call_id"] = raw_message["tool_call_id"]
        if raw_message.get("name") is not None:
            message["name"] = raw_message["name"]
        if raw_message.get("reasoning_content"):
            message["thinking"] = raw_message["reasoning_content"]
        if raw_message.get("tool_calls"):
            tool_calls = raw_message["tool_calls"]
            if not isinstance(tool_calls, list):
                raise ValueError(f"tool_calls in {source_id} must be a list")
            for tool_call_index, tool_call in enumerate(tool_calls):
                if not isinstance(tool_call, dict):
                    raise ValueError(f"Tool call {tool_call_index} in {source_id} must be an object")
            message["tool_calls"] = [normalize_tool_call(tool_call, source_id=source_id) for tool_call in tool_calls]
        messages.append(message)

    return messages


def normalize_tools(row: dict[str, Any], subset: NemotronAgenticSubset) -> list[dict[str, Any]]:
    _require_supported_subset(subset)
    tools_raw = row.get("tools") or []
    if not isinstance(tools_raw, list):
        raise ValueError(f"Tools for {_source_context(row)} must be a list")

    source_id = _source_context(row)
    tools: list[dict[str, Any]] = []
    for index, raw_tool in enumerate(tools_raw):
        if not isinstance(raw_tool, dict):
            raise ValueError(f"Tool {index} in {source_id} must be an object")
        function = raw_tool.get("function")
        if not isinstance(function, dict):
            raise ValueError(f"Tool {index} in {source_id} is missing function")

        name = function.get("name")
        parameters = function.get("parameters")
        if not isinstance(name, str) or not name:
            raise ValueError(f"Tool {index} in {source_id} is missing function.name")
        if not isinstance(parameters, dict):
            raise ValueError(f"Tool {name} in {source_id} must have dict parameters")

        tools.append(
            {
                "type": raw_tool.get("type", "function"),
                "function": {
                    "name": name,
                    "description": function.get("description") or "",
                    "parameters": parameters,
                    "strict": function.get("strict"),
                },
            }
        )
    return tools


def source_id_from_row(row: dict[str, Any], messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> str:
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    for value in (metadata.get("uuid"), metadata.get("alt_id"), row.get("uuid")):
        if isinstance(value, str) and value:
            return value

    fingerprint = stable_json({"messages": messages, "tools": tools})
    return hashlib.sha256(fingerprint.encode("utf-8")).hexdigest()


def _render_attrs(attrs: dict[str, object]) -> str:
    rendered = []
    for key, value in attrs.items():
        if value is None:
            continue
        rendered.append(f'{key}="{escape(_render_value(value), quote=True)}"')
    return (" " + " ".join(rendered)) if rendered else ""


def _render_tool_call(tool_call: dict[str, Any]) -> str:
    function = tool_call["function"]
    attrs = _render_attrs({"id": tool_call.get("id"), "name": function["name"]})
    return "\n".join(
        [
            f"<tool_call{attrs}>",
            function["arguments"],
            "</tool_call>",
        ]
    )


def _render_message(message: dict[str, Any]) -> str:
    role = message["role"]
    content = _render_value(message.get("content"))

    if role == "assistant":
        parts = ["<assistant>"]
        thinking = _render_value(message.get("thinking"))
        if thinking:
            parts.extend(["<thinking>", thinking, "</thinking>"])
        if content:
            parts.extend(["<content>", content, "</content>"])
        for tool_call in message.get("tool_calls") or []:
            parts.append(_render_tool_call(tool_call))
        parts.append("</assistant>")
        return "\n".join(parts)

    if role == "tool":
        attrs = _render_attrs({"tool_call_id": message.get("tool_call_id"), "name": message.get("name")})
        return "\n".join([f"<tool{attrs}>", content, "</tool>"])

    return "\n".join([f"<{role}>", content, f"</{role}>"])


def render_trajectory(
    *,
    subset: NemotronAgenticSubset,
    source_id: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
) -> str:
    parts = [
        f'<trajectory source="{HF_DATASET_ID}" subset="{subset.value}" id="{source_id}">',
        "<tools>",
        stable_json(tools),
        "</tools>",
    ]
    parts.extend(_render_message(message) for message in messages)
    parts.append("</trajectory>")
    return "\n\n".join(parts)


def _token_count(metadata: dict[str, Any], key: str) -> int:
    all_turns = metadata.get("all_turns_token_count")
    if not isinstance(all_turns, dict):
        return 0
    value = all_turns.get(key)
    return value if isinstance(value, int) else 0


def row_to_doc(row: dict[str, Any], subset: NemotronAgenticSubset) -> list[dict[str, Any]]:
    _require_supported_subset(subset)
    messages = normalize_messages(row, subset)
    if not messages:
        counters.increment(f"nemotron_agentic/{subset.value}/dropped")
        return []

    tools = normalize_tools(row, subset)
    source_id = source_id_from_row(row, messages, tools)
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    processing_info = row.get("processing_info") if isinstance(row.get("processing_info"), dict) else {}
    chat_template_kwargs = row.get("chat_template_kwargs") if isinstance(row.get("chat_template_kwargs"), dict) else {}
    tool_names = [tool["function"]["name"] for tool in tools]
    tool_call_count = sum(len(message.get("tool_calls") or []) for message in messages)
    text = render_trajectory(subset=subset, source_id=source_id, messages=messages, tools=tools)

    counters.increment(f"nemotron_agentic/{subset.value}/kept")
    return [
        {
            "id": source_id,
            "text": text,
            "source": HF_DATASET_ID,
            "source_dataset": HF_DATASET_ID,
            "source_subset": subset.value,
            "trajectory_id": source_id,
            "hf_revision": HF_REVISION,
            "transform_version": TRANSFORM_VERSION,
            "licenses_json": stable_json(list(LICENSES)),
            "model": row.get("model") or "",
            "domain": row.get("domain") or "",
            "temperature": row.get("temperature"),
            "parallel_tool_calls": row.get("parallel_tool_calls"),
            "chat_template_kwargs_json": stable_json(chat_template_kwargs),
            "messages_json": stable_json(messages),
            "tools_json": stable_json(tools),
            "metadata_json": stable_json(metadata),
            "processing_info_json": stable_json(processing_info),
            "tool_names_json": stable_json(tool_names),
            "tool_call_count": tool_call_count,
            "turn_count": len(messages),
            "reasoning_token_count": _token_count(metadata, "reasoning"),
            "content_token_count": _token_count(metadata, "content"),
            "all_token_count": _token_count(metadata, "all"),
        }
    ]


def transform(input_path: str, output_path: str, subset: NemotronAgenticSubset) -> None:
    _require_supported_subset(subset)
    input_file = f"{input_path}/{HF_FILE_BY_SUBSET[subset]}"
    pipeline = (
        Dataset.from_files(input_file)
        .flat_map(lambda source: load_nemotron_agentic_jsonl(source, subset))
        .flat_map(lambda row: row_to_doc(row, subset))
        .write_parquet(f"{output_path}/data-{{shard:05d}}-of-{{total:05d}}.parquet", skip_existing=True)
    )
    ctx = ZephyrContext(
        name=f"nemotron-agentic-{subset.value}-transform",
        resources=ResourceConfig(cpu=1, ram="16g"),
    )
    ctx.execute(pipeline)


def download_nemotron_agentic_raw_step(subset: NemotronAgenticSubset) -> StepSpec:
    """Download one Nemotron Agentic v2 split from Hugging Face."""
    _require_supported_subset(subset)
    return download_hf_step(
        f"raw/nemotron-sft-agentic-v2/{subset.value}",
        hf_dataset_id=HF_DATASET_ID,
        revision=HF_REVISION,
        hf_urls_glob=[HF_FILE_BY_SUBSET[subset]],
    )


def download_nemotron_agentic_step(subset: NemotronAgenticSubset) -> StepSpec:
    """Download and transform one Nemotron Agentic v2 split."""
    dl = download_nemotron_agentic_raw_step(subset)
    return StepSpec(
        name=f"processed/nemotron-sft-agentic-v2/{subset.value}",
        deps=[dl],
        fn=lambda output_path: transform(input_path=dl.output_path, output_path=output_path, subset=subset),
        hash_attrs={"subset": subset.value, "version": TRANSFORM_VERSION},
    )


def nemotron_agentic_normalize_steps(subset: NemotronAgenticSubset) -> tuple[StepSpec, ...]:
    """Return the full ``(download+transform, normalize)`` chain for one split."""
    processed = download_nemotron_agentic_step(subset)
    return (
        processed,
        normalize_step(
            name=f"normalized/nemotron-sft-agentic-v2/{subset.value}",
            download=processed,
            text_field="text",
            id_field="id",
        ),
    )
