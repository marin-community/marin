# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Structured-chat Datakit sources and lowering to ordinary text sources."""

import json
import logging
import re
from dataclasses import asdict, dataclass

from fray.types import ResourceConfig
from levanter.data.text.formats import ChatLmDatasetFormat
from levanter.tokenizers import MarinTokenizer, load_tokenizer
from rigging.filesystem.storage_path import prefix_join
from zephyr import counters
from zephyr.context import ZephyrContext
from zephyr.dataset import Dataset
from zephyr.readers import load_parquet

from marin.datakit.download.rollout_transforms import text_document
from marin.datakit.normalize import NormalizedData, normalize_step
from marin.datakit.sources import DatakitSource
from marin.execution.artifact import read_artifact
from marin.execution.step_spec import StepSpec

_CHAT_BLOCK = re.compile(
    r"<\|start_header_id\|>(?P<role>[^<]+)<\|end_header_id\|>\n(?P<body>.*?)<\|eot_id\|>",
    re.DOTALL,
)
_TOOL_CALL = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
_LONG_ASSISTANT_TOKEN_THRESHOLD = 10_000

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DatakitChatSource:
    """A source whose terminal normalized artifact contains canonical messages."""

    name: str
    normalize_steps: tuple[StepSpec, ...]
    format: ChatLmDatasetFormat
    rough_token_count_b: float

    @property
    def normalized(self) -> StepSpec:
        return self.normalize_steps[-1]


def _render_messages(record: dict, tokenizer: MarinTokenizer, chat_format: ChatLmDatasetFormat) -> list[dict]:
    messages = _messages_for_template(record[chat_format.messages_field])
    for message in messages:
        content = message.get("content")
        if (
            message.get("role") == "assistant"
            and isinstance(content, str)
            and "<|start_think|>" not in content
            and len(content.encode("utf-8")) > _LONG_ASSISTANT_TOKEN_THRESHOLD
            and len(tokenizer.encode(content, add_special_tokens=False)) > _LONG_ASSISTANT_TOKEN_THRESHOLD
        ):
            counters.pipeline.update_counter("render_chat/long_assistant_without_reasoning", 1)
    raw_kwargs = record.get(chat_format.chat_template_kwargs) if chat_format.chat_template_kwargs else None
    if isinstance(raw_kwargs, str):
        raw_kwargs = json.loads(raw_kwargs)
    if raw_kwargs is not None and not isinstance(raw_kwargs, dict):
        raise ValueError("chat_template_kwargs must be a JSON object")
    kwargs = dict(raw_kwargs or {})
    kwargs.setdefault("enable_thinking", _has_reasoning(messages))
    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
        **kwargs,
    )
    assert isinstance(rendered, str)
    if tokenizer.bos_token and rendered.startswith(tokenizer.bos_token):
        rendered = rendered.removeprefix(tokenizer.bos_token)
    validate_rendered_chat(messages, kwargs, rendered)
    counters.pipeline.update_counter("render_chat/records_validated", 1)
    return [text_document(rendered, record.get("source", ""))]


def _has_reasoning(messages: list[dict]) -> bool:
    return any(
        message.get("role") == "assistant"
        and isinstance(message.get("content"), str)
        and "<|start_think|>" in message["content"]
        for message in messages
    )


def validate_rendered_chat(messages: list[dict], kwargs: dict, rendered: str) -> None:
    """Verify the canonical messages survive rendering as one valid Marin transcript."""
    seen_non_system = False
    pending_calls: dict[str, str] = {}
    for message in messages:
        role = message.get("role")
        content = message.get("content")
        if role == "system":
            if seen_non_system:
                raise ValueError("System messages must precede conversation turns")
            continue
        if not seen_non_system:
            if role != "user":
                raise ValueError("The first non-system message must be a user message")
            seen_non_system = True
        if role == "user":
            if not isinstance(content, str) or not content.strip():
                raise ValueError("User messages must contain non-empty text")
            if pending_calls:
                raise ValueError("Tool observations must use the tool role, not the user role")
        elif role == "assistant":
            calls = message.get("tool_calls") or []
            for call in calls:
                call_id = call.get("id")
                function = call.get("function") or {}
                if not isinstance(call_id, str) or not isinstance(function.get("name"), str):
                    raise ValueError("Canonical tool calls require string IDs and names")
                pending_calls[call_id] = function["name"]
        elif role == "tool":
            call_id = message.get("tool_call_id")
            if call_id not in pending_calls:
                raise ValueError("Tool responses must reference a pending tool-call ID")
            if message.get("name") not in (None, pending_calls[call_id]):
                raise ValueError("Tool response name does not match its tool call")
            del pending_calls[call_id]

    if not rendered or "<think>" in rendered or "</think>" in rendered:
        raise ValueError("Rendered chat must be non-empty and use atomic reasoning delimiters")

    blocks = [(match.group("role"), match.group("body")) for match in _CHAT_BLOCK.finditer(rendered)]
    expected_roles = [message["role"] for message in messages]
    if kwargs.get("enable_thinking") is not None or kwargs.get("tools"):
        expected_roles.insert(0, "system")
    if [role for role, _ in blocks] != expected_roles:
        raise ValueError("Rendered role blocks do not match canonical messages")

    tools = kwargs.get("tools") or []
    tool_names = {
        tool.get("name") or (tool.get("function") or {}).get("name") for tool in tools if isinstance(tool, dict)
    }
    message_blocks = blocks[-len(messages) :]
    for message, (role, body) in zip(messages, message_blocks, strict=True):
        content = message.get("content")
        if isinstance(content, str) and content.strip() and content.strip() not in body:
            raise ValueError(f"Rendered {role} turn lost its text or reasoning")
        if role == "assistant":
            calls = message.get("tool_calls") or []
            rendered_calls = _TOOL_CALL.findall(body)
            if len(rendered_calls) != len(calls):
                raise ValueError("Rendered assistant tool calls do not match canonical messages")
            for call, encoded in zip(calls, rendered_calls, strict=True):
                payload = json.loads(encoded)
                function = call["function"]
                if payload.get("name") != function["name"] or not isinstance(payload.get("arguments"), dict):
                    raise ValueError("Rendered tool call must contain the canonical name and JSON-object arguments")
                if function["name"] not in tool_names:
                    raise ValueError(f"Rendered tool call {function['name']!r} has no matching tool definition")
            if not calls:
                try:
                    payload = json.loads(body)
                except json.JSONDecodeError:
                    payload = None
                if isinstance(payload, dict) and ({"analysis", "commands"} & payload.keys()):
                    raise ValueError("Provider-specific JSON action protocols must be adapted before rendering")
        elif role == "tool":
            if "<tool_response" not in body or "</tool_response>" not in body:
                raise ValueError("Rendered tool message must use a tool_response block")
            call_id = message["tool_call_id"]
            if f'id="{call_id}"' not in body:
                raise ValueError("Rendered tool response lost its tool-call ID")

    depth = 0
    for match in re.finditer(r"<\|(start|end)_think\|>", rendered):
        depth += 1 if match.group(1) == "start" else -1
        if depth not in (0, 1):
            raise ValueError("Rendered reasoning delimiters must be balanced and cannot nest")
    if depth:
        raise ValueError("Rendered reasoning delimiters must be balanced and cannot nest")


def _messages_for_template(messages: list[dict]) -> list[dict]:
    """Decode canonical JSON argument strings for chat-template serialization."""
    rendered_messages = []
    for message in messages:
        rendered_message = dict(message)
        tool_calls = message.get("tool_calls")
        if tool_calls:
            rendered_tool_calls = []
            for tool_call in tool_calls:
                rendered_tool_call = dict(tool_call)
                function = dict(tool_call["function"])
                arguments = function["arguments"]
                if isinstance(arguments, str):
                    arguments = json.loads(arguments)
                if not isinstance(arguments, dict):
                    raise ValueError("Tool-call arguments must decode to a JSON object")
                function["arguments"] = arguments
                rendered_tool_call["function"] = function
                rendered_tool_calls.append(rendered_tool_call)
            rendered_message["tool_calls"] = rendered_tool_calls
        rendered_messages.append(rendered_message)
    return rendered_messages


def _render_chat_artifact(
    input_path: str,
    output_path: str,
    tokenizer_name: str,
    chat_format: ChatLmDatasetFormat,
) -> None:
    normalized = read_artifact(input_path, NormalizedData)
    tokenizer = load_tokenizer(tokenizer_name)
    if chat_format.chat_template is not None:
        tokenizer = tokenizer.with_chat_template(chat_format.chat_template)
    pipeline = (
        Dataset.from_files(prefix_join(normalized.main_output_dir, "*.parquet"))
        .flat_map(load_parquet)
        .flat_map(lambda record: _render_messages(record, tokenizer, chat_format))
        .write_parquet(prefix_join(output_path, "data-{shard:05d}-of-{total:05d}.parquet"), skip_existing=True)
    )
    outcome = ZephyrContext(name="render-chat", resources=ResourceConfig(cpu=1, ram="8g")).execute(pipeline)
    if not outcome.counters.get("render_chat/records_validated", 0):
        raise ValueError(f"Chat source {input_path} rendered no records")
    long_without_reasoning = outcome.counters.get("render_chat/long_assistant_without_reasoning", 0)
    if long_without_reasoning:
        logger.warning(
            "%s rendered assistant replies exceed %d tokens without reasoning delimiters",
            long_without_reasoning,
            _LONG_ASSISTANT_TOKEN_THRESHOLD,
        )


def render_chat_source(source: DatakitChatSource, *, tokenizer: str) -> DatakitSource:
    """Render a structured-chat source and return an ordinary text Datakit source."""
    rendered = StepSpec(
        name=f"rendered-chat/{source.name}",
        deps=[source.normalized],
        fn=lambda output_path: _render_chat_artifact(
            source.normalized.output_path, output_path, tokenizer, source.format
        ),
        hash_attrs={"tokenizer": tokenizer, "format": asdict(source.format)},
    )
    normalized = normalize_step(name=f"normalized-rendered-chat/{source.name}", download=rendered)
    return DatakitSource(
        name=source.name,
        normalize_steps=(*source.normalize_steps, rendered, normalized),
        rough_token_count_b=source.rough_token_count_b,
    )
