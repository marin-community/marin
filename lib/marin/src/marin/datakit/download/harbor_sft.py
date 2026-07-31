# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Convert durable Harbor trajectories into structured, tools-aware SFT records.

Harbor evaluation and synthetic-data generation share trial execution. They
diverge after a trial finishes: evaluation aggregates verifier metrics, while
synthetic-data generation accepts training rows and preserves the trajectory's
structured supervision. This module implements that completion transform as a
Datakit step.

Co-located harnesses can export structured messages directly. Installed
harnesses such as OpenCode run inside the sandbox and call a remote inference
service, so their ``conversations`` projection omits the served system prompt,
tool definitions, and structured calls. For those rows, the literal prompt and
completion token columns are the source of truth.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from enum import StrEnum
from functools import cache, partial
from pathlib import Path
from typing import Protocol

import pyarrow as pa
from fray.types import ResourceConfig
from rigging.filesystem import StoragePath, prefix_join
from transformers import AutoTokenizer
from zephyr import counters
from zephyr.dataset import Dataset
from zephyr.execution import ZephyrContext

from marin.datakit.download.huggingface import download_hf_step
from marin.datakit.download.rollout_transforms import load_parquet_batched
from marin.execution.step_spec import StepSpec

TOKENIZER_PROVENANCE_FILE = "tokenizer_provenance.json"

_TURN_RE = re.compile(r"<\|im_start\|>(\w+)\n(.*?)<\|im_end\|>", re.DOTALL)
_SPECIAL_TAIL_RE = re.compile(r"(<\|im_end\|>|<\|endoftext\|>)\s*$")
_TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
_FUNCTION_RE = re.compile(r"<function=([^>]+)>(.*)</function>", re.DOTALL)
_PARAMETER_RE = re.compile(r"<parameter=([^>]+)>\n?(.*?)\n?</parameter>", re.DOTALL)
_TOOL_CALL_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)

_TOOL_CALL_SCHEMA = pa.struct(
    [
        pa.field("type", pa.string()),
        pa.field(
            "function",
            pa.struct(
                [
                    pa.field("name", pa.string()),
                    pa.field("arguments", pa.string()),
                ]
            ),
        ),
    ]
)
HARBOR_SFT_SCHEMA = pa.schema(
    [
        pa.field(
            "messages",
            pa.list_(
                pa.struct(
                    [
                        pa.field("role", pa.string()),
                        pa.field("content", pa.string()),
                        pa.field("tool_calls", pa.list_(_TOOL_CALL_SCHEMA)),
                    ]
                )
            ),
        ),
        pa.field("tools", pa.string()),
        pa.field("task", pa.string()),
        pa.field("num_turns", pa.int32()),
        pa.field("num_tool_calls", pa.int32()),
    ]
)


class TokenDecoder(Protocol):
    """The tokenizer surface needed to decode recorded literal token IDs."""

    def decode(self, token_ids, *, skip_special_tokens: bool) -> str: ...


class HarborSftHarness(StrEnum):
    """How a Harbor row recorded the model interaction."""

    STRUCTURED = "structured"
    OPENCODE_LITERALS = "opencode_literals"


class RejectionReason(StrEnum):
    """Why a Harbor row cannot safely become an SFT example."""

    MISSING_LITERALS = "missing_literals"
    ASSISTANT_COMPLETION_MISMATCH = "assistant_completion_mismatch"
    MISSING_SYSTEM = "missing_system"
    MISSING_TOOLS = "missing_tools"
    MISSING_TASK = "missing_task"
    INVALID_MESSAGES = "invalid_messages"


@dataclass(frozen=True)
class HarborSftConversion:
    """One accepted record or an explicit rejection."""

    record: dict | None = None
    rejection: RejectionReason | None = None

    def __post_init__(self) -> None:
        if (self.record is None) == (self.rejection is None):
            raise ValueError("conversion must contain exactly one of record or rejection")


@dataclass(frozen=True)
class HarborSftSource:
    """A pinned Hugging Face Harbor-trace source for Datakit conversion."""

    name: str
    hf_dataset_id: str
    revision: str
    harness: HarborSftHarness
    teacher_tokenizer: str | None = None
    teacher_tokenizer_revision: str | None = None
    expected_rows: int | None = None
    hf_urls_glob: tuple[str, ...] = ("**/*.parquet", TOKENIZER_PROVENANCE_FILE)


@dataclass(frozen=True)
class HarborSftManifest:
    """A reproducible set of pinned Harbor SFT sources."""

    name: str
    sources: tuple[HarborSftSource, ...]


def _parse_tools(prompt_text: str) -> list[dict]:
    match = re.search(r"<tools>\s*(.*?)\s*</tools>", prompt_text, re.DOTALL)
    if not match:
        return []
    tools: list[dict] = []
    for line in match.group(1).splitlines():
        try:
            tool = json.loads(line.strip())
        except json.JSONDecodeError:
            continue
        if isinstance(tool, dict):
            tools.append(tool)
    return tools


def _recover_system_content(teacher_system: str) -> str:
    important_end = teacher_system.find("</IMPORTANT>")
    if important_end != -1:
        return teacher_system[important_end + len("</IMPORTANT>") :].lstrip()
    opencode_start = teacher_system.find("You are opencode")
    if opencode_start != -1:
        return teacher_system[opencode_start:].lstrip()
    tools_end = teacher_system.find("</tools>")
    if tools_end != -1:
        return teacher_system[tools_end + len("</tools>") :].lstrip()
    return teacher_system.strip()


def _leading_turns(prompt_text: str) -> list[dict]:
    return [{"role": role, "content": content.strip()} for role, content in _TURN_RE.findall(prompt_text)]


def _fix_orphan_think(text: str) -> str:
    if "</think>" in text and "<think>" not in text:
        return f"<think>\n{text}"
    return text


def _coerce_argument(tool_name: str, argument_name: str, value: str, type_map: dict) -> object:
    declared_type = type_map.get(tool_name, {}).get(argument_name)
    if declared_type == "string":
        return value
    if declared_type in {"integer", "number", "boolean", "object", "array"}:
        try:
            return json.loads(value)
        except (json.JSONDecodeError, ValueError):
            return value
    try:
        parsed = json.loads(value)
    except (json.JSONDecodeError, ValueError):
        return value
    return parsed if not isinstance(parsed, str) else value


def _normalize_tool_call(name: str, arguments) -> dict:
    if not isinstance(arguments, str):
        arguments = json.dumps(arguments or {}, ensure_ascii=False)
    return {
        "type": "function",
        "function": {
            "name": name,
            "arguments": arguments,
        },
    }


def _parse_tool_calls(assistant_text: str, type_map: dict) -> list[dict]:
    calls: list[dict] = []
    for block in _TOOL_CALL_RE.findall(assistant_text):
        function_match = _FUNCTION_RE.search(block)
        if function_match:
            name = function_match.group(1).strip()
            arguments = {
                key.strip(): _coerce_argument(name, key.strip(), value, type_map)
                for key, value in _PARAMETER_RE.findall(function_match.group(2))
            }
            calls.append(_normalize_tool_call(name, arguments))
            continue

        json_match = _TOOL_CALL_JSON_RE.search(block)
        if not json_match:
            continue
        try:
            call = json.loads(json_match.group(0))
        except json.JSONDecodeError:
            continue
        if isinstance(call, dict) and call.get("name"):
            calls.append(_normalize_tool_call(call["name"], call.get("arguments", {})))
    return calls


def _strip_tool_calls(assistant_text: str) -> str:
    content = _TOOL_CALL_RE.sub("", assistant_text)
    return _SPECIAL_TAIL_RE.sub("", content).strip()


def _tool_type_map(tools: list[dict]) -> dict[str, dict[str, str | None]]:
    type_map: dict[str, dict[str, str | None]] = {}
    for tool in tools:
        function = tool.get("function", {})
        properties = (function.get("parameters", {}) or {}).get("properties", {}) or {}
        type_map[function.get("name")] = {key: (value or {}).get("type") for key, value in properties.items()}
    return type_map


def _reject(reason: RejectionReason) -> HarborSftConversion:
    return HarborSftConversion(rejection=reason)


def _convert_opencode_literals(row: dict, tokenizer: TokenDecoder | None) -> HarborSftConversion:
    if tokenizer is None:
        raise ValueError("OpenCode literal conversion requires a teacher tokenizer")

    completions = row.get("completion_token_ids")
    prompts = row.get("prompt_token_ids")
    conversations = row.get("conversations") or []
    if not completions or not any(completions) or not prompts or not prompts[0]:
        return _reject(RejectionReason.MISSING_LITERALS)

    assistant_messages = [message for message in conversations if message.get("role") == "assistant"]
    user_messages = [message for message in conversations if message.get("role") == "user"]
    if len(assistant_messages) != len(completions):
        return _reject(RejectionReason.ASSISTANT_COMPLETION_MISMATCH)

    prompt_text = tokenizer.decode(prompts[0], skip_special_tokens=False)
    leading = _leading_turns(prompt_text)
    if not leading or leading[0]["role"] != "system":
        return _reject(RejectionReason.MISSING_SYSTEM)

    tools = _parse_tools(prompt_text)
    if not tools:
        return _reject(RejectionReason.MISSING_TOOLS)

    task_prompt = next(
        (message["content"] for message in leading[1:] if message["role"] == "user"),
        None,
    )
    if not task_prompt:
        return _reject(RejectionReason.MISSING_TASK)

    messages = [
        {
            "role": "system",
            "content": _recover_system_content(leading[0]["content"]),
            "tool_calls": [],
        },
        {"role": "user", "content": task_prompt, "tool_calls": []},
    ]
    num_tool_calls = 0
    type_map = _tool_type_map(tools)
    for index, completion in enumerate(completions):
        assistant_text = _fix_orphan_think(tokenizer.decode(completion, skip_special_tokens=False))
        tool_calls = _parse_tool_calls(assistant_text, type_map)
        messages.append(
            {
                "role": "assistant",
                "content": _strip_tool_calls(assistant_text),
                "tool_calls": tool_calls,
            }
        )
        num_tool_calls += len(tool_calls)
        if index >= len(completions) - 1 or index + 1 >= len(user_messages):
            continue
        observation = (user_messages[index + 1].get("content") or "").strip()
        observation = re.sub(
            r"^<tool_response>\s*|\s*</tool_response>$",
            "",
            observation,
        ).strip()
        if observation:
            messages.append(
                {
                    "role": "tool",
                    "content": observation,
                    "tool_calls": [],
                }
            )

    return HarborSftConversion(
        record={
            "messages": messages,
            "tools": json.dumps(tools, ensure_ascii=False),
            "task": row.get("task"),
            "num_turns": len(messages),
            "num_tool_calls": num_tool_calls,
        }
    )


def _serialized_tools(value) -> str | None:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return None
    if value is None:
        value = []
    if not isinstance(value, list) or not all(isinstance(tool, dict) for tool in value):
        return None
    return json.dumps(value, ensure_ascii=False)


def _convert_structured(row: dict) -> HarborSftConversion:
    source_messages = row.get("messages") or row.get("conversations")
    if not isinstance(source_messages, list) or not source_messages:
        return _reject(RejectionReason.INVALID_MESSAGES)

    messages: list[dict] = []
    num_tool_calls = 0
    has_task_prompt = False
    has_assistant = False
    for source_message in source_messages:
        if not isinstance(source_message, dict):
            return _reject(RejectionReason.INVALID_MESSAGES)
        role = source_message.get("role")
        content = source_message.get("content")
        if role not in {"system", "user", "assistant", "tool"} or not isinstance(content, str):
            return _reject(RejectionReason.INVALID_MESSAGES)
        if role == "user" and content.strip():
            has_task_prompt = True
        if role == "assistant":
            has_assistant = True

        tool_calls: list[dict] = []
        for tool_call in source_message.get("tool_calls") or []:
            if not isinstance(tool_call, dict):
                return _reject(RejectionReason.INVALID_MESSAGES)
            function = tool_call.get("function") or tool_call
            name = function.get("name") if isinstance(function, dict) else None
            if not name:
                return _reject(RejectionReason.INVALID_MESSAGES)
            tool_calls.append(_normalize_tool_call(name, function.get("arguments", {})))
        num_tool_calls += len(tool_calls)
        messages.append(
            {
                "role": role,
                "content": content,
                "tool_calls": tool_calls,
            }
        )

    if not has_task_prompt:
        return _reject(RejectionReason.MISSING_TASK)
    if not has_assistant:
        return _reject(RejectionReason.INVALID_MESSAGES)

    tools = _serialized_tools(row.get("tools"))
    if tools is None:
        return _reject(RejectionReason.MISSING_TOOLS)

    return HarborSftConversion(
        record={
            "messages": messages,
            "tools": tools,
            "task": row.get("task") or row.get("task_id") or row.get("id"),
            "num_turns": len(messages),
            "num_tool_calls": num_tool_calls,
        }
    )


def convert_harbor_row(
    row: dict,
    harness: HarborSftHarness,
    tokenizer: TokenDecoder | None,
) -> HarborSftConversion:
    """Convert one Harbor row without silently using a lossy fallback."""
    if harness is HarborSftHarness.OPENCODE_LITERALS:
        return _convert_opencode_literals(row, tokenizer)
    if harness is HarborSftHarness.STRUCTURED:
        return _convert_structured(row)
    raise ValueError(f"unsupported Harbor SFT harness: {harness}")


@cache
def _load_tokenizer(tokenizer_ref: str, tokenizer_revision: str | None):
    return AutoTokenizer.from_pretrained(
        tokenizer_ref,
        revision=tokenizer_revision,
        trust_remote_code=True,
    )


def _pipeline_convert(
    row: dict,
    harness: HarborSftHarness,
    tokenizer_ref: str | None,
    tokenizer_revision: str | None,
) -> list[dict]:
    tokenizer = _load_tokenizer(tokenizer_ref, tokenizer_revision) if tokenizer_ref else None
    result = convert_harbor_row(row, harness, tokenizer)
    if result.record is not None:
        counters.pipeline.update_counter("harbor_sft/accepted", 1)
        return [result.record]
    counters.pipeline.update_counter(f"harbor_sft/rejected/{result.rejection.value}", 1)
    return []


def resolve_teacher_tokenizer(
    input_path: str,
    harness: HarborSftHarness,
    override: str | None,
    override_revision: str | None,
) -> tuple[str | None, str | None]:
    """Resolve the exact tokenizer that produced literal IDs, failing closed."""
    if harness is HarborSftHarness.STRUCTURED:
        return None, None
    if override:
        if not override_revision:
            raise ValueError("teacher_tokenizer_revision is required for literal conversion")
        return override, override_revision
    provenance_path = StoragePath(prefix_join(input_path, TOKENIZER_PROVENANCE_FILE))
    if not provenance_path.exists():
        raise ValueError(
            f"{TOKENIZER_PROVENANCE_FILE} is required for literal conversion unless teacher_tokenizer is set explicitly"
        )
    provenance = json.loads(provenance_path.read_text())
    served_model = provenance.get("served_model")
    if not isinstance(served_model, str) or not served_model:
        raise ValueError(f"{provenance_path} does not contain a non-empty served_model")
    served_model_revision = provenance.get("served_model_revision")
    if not isinstance(served_model_revision, str) or not served_model_revision:
        raise ValueError(
            f"{provenance_path} does not pin served_model_revision; set teacher_tokenizer and "
            "teacher_tokenizer_revision explicitly"
        )
    return served_model, served_model_revision


def transform_harbor_sft(
    input_path: str,
    output_path: str,
    *,
    harness: HarborSftHarness,
    teacher_tokenizer: str | None = None,
    teacher_tokenizer_revision: str | None = None,
    expected_rows: int | None = None,
) -> None:
    """Stream downloaded Harbor parquet through the selected SFT adapter."""
    tokenizer_ref, tokenizer_revision = resolve_teacher_tokenizer(
        input_path,
        harness,
        teacher_tokenizer,
        teacher_tokenizer_revision,
    )
    pipeline = (
        Dataset.from_files(prefix_join(input_path, "**/*.parquet"))
        .flat_map(load_parquet_batched)
        .flat_map(
            partial(
                _pipeline_convert,
                harness=harness,
                tokenizer_ref=tokenizer_ref,
                tokenizer_revision=tokenizer_revision,
            )
        )
        .write_parquet(
            prefix_join(output_path, "data-{shard:05d}-of-{total:05d}.parquet"),
            schema=HARBOR_SFT_SCHEMA,
            skip_existing=True,
        )
    )
    context = ZephyrContext(
        name=f"harbor-sft-{harness.value}",
        resources=ResourceConfig(cpu=1, ram="32g"),
    )
    outcome = context.execute(pipeline)
    accepted = outcome.counters.get("harbor_sft/accepted", 0)
    if expected_rows is not None and accepted != expected_rows:
        raise ValueError(f"Harbor SFT row-count mismatch: expected {expected_rows}, accepted {accepted}")


def harbor_sft_steps(source: HarborSftSource) -> tuple[StepSpec, StepSpec]:
    """Return Datakit download and SFT conversion steps for one pinned source."""
    download = download_hf_step(
        f"raw/harbor-sft/{source.name}",
        hf_dataset_id=source.hf_dataset_id,
        revision=source.revision,
        hf_urls_glob=list(source.hf_urls_glob),
    )
    processed = StepSpec(
        name=f"processed/harbor-sft/{source.name}",
        deps=[download],
        fn=lambda output_path: transform_harbor_sft(
            input_path=download.output_path,
            output_path=output_path,
            harness=source.harness,
            teacher_tokenizer=source.teacher_tokenizer,
            teacher_tokenizer_revision=source.teacher_tokenizer_revision,
            expected_rows=source.expected_rows,
        ),
        hash_attrs={
            "version": "v1",
            "hf_dataset_id": source.hf_dataset_id,
            "revision": source.revision,
            "harness": source.harness.value,
            "teacher_tokenizer": source.teacher_tokenizer,
            "teacher_tokenizer_revision": source.teacher_tokenizer_revision,
            "expected_rows": source.expected_rows,
        },
    )
    return download, processed


def load_harbor_sft_manifest(path: str | Path) -> HarborSftManifest:
    """Load a pinned source manifest with optional top-level adapter defaults."""
    manifest_path = Path(path)
    data = json.loads(manifest_path.read_text())
    name = data.get("name")
    raw_sources = data.get("sources")
    if not isinstance(name, str) or not name or not isinstance(raw_sources, list):
        raise ValueError(f"invalid Harbor SFT manifest: {manifest_path}")

    default_harness = data.get("harness")
    default_tokenizer = data.get("teacher_tokenizer")
    default_tokenizer_revision = data.get("teacher_tokenizer_revision")
    sources: list[HarborSftSource] = []
    for raw_source in raw_sources:
        if not isinstance(raw_source, dict):
            raise ValueError(f"invalid Harbor SFT source in {manifest_path}: {raw_source!r}")
        harness_value = raw_source.get("harness", default_harness)
        try:
            harness = HarborSftHarness(harness_value)
        except ValueError as exc:
            raise ValueError(f"invalid Harbor SFT harness for {raw_source.get('name')!r}: {harness_value!r}") from exc
        sources.append(
            HarborSftSource(
                name=raw_source["name"],
                hf_dataset_id=raw_source["hf_dataset_id"],
                revision=raw_source["revision"],
                harness=harness,
                teacher_tokenizer=raw_source.get("teacher_tokenizer", default_tokenizer),
                teacher_tokenizer_revision=raw_source.get(
                    "teacher_tokenizer_revision",
                    default_tokenizer_revision,
                ),
                expected_rows=raw_source.get("expected_rows"),
            )
        )

    source_names = [source.name for source in sources]
    if len(source_names) != len(set(source_names)):
        raise ValueError(f"duplicate Harbor SFT source names in {manifest_path}")
    if not sources:
        raise ValueError(f"Harbor SFT manifest contains no sources: {manifest_path}")
    return HarborSftManifest(name=name, sources=tuple(sources))
