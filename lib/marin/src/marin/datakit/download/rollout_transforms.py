# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared transform helpers for datakit download pipelines."""

import hashlib
import json
import logging
import re
from collections.abc import Iterator
from types import MappingProxyType

import pyarrow.parquet as pq
from rigging.filesystem.factory import open_url
from zephyr import counters

logger = logging.getLogger(__name__)

CANONICAL_CHAT_ROLES = frozenset({"assistant", "system", "tool", "user"})
CHAT_CONTROL_TOKEN = re.compile(
    r"<\|(?:begin_of_text|end_of_text|finetune_right_pad_id|start_header_id|end_header_id|"
    r"eom_id|eot_id|python_tag|reserved_special_token_\d+)\|>"
)
REASONING_TOKEN = re.compile(r"<\|(?:start|end)_think\|>")
CHAT_ROLE_ALIASES = MappingProxyType(
    {
        "bot": "assistant",
        "function": "tool",
        "gpt": "assistant",
        "human": "user",
        "model": "assistant",
    }
)


class ReasoningFormatError(ValueError):
    """Raised when an assistant reasoning span cannot be normalized safely."""


def load_parquet_batched(path: str) -> Iterator[dict]:
    """Read parquet via iter_batches to avoid OOM on large nested-struct columns."""
    with open_url(path, "rb") as f:
        pf = pq.ParquetFile(f)
        for batch in pf.iter_batches(batch_size=16):
            try:
                rows = batch.to_pydict()
            except UnicodeDecodeError as e:
                counters.pipeline.update_counter("load_parquet_batched/utf8_skip_batch", 1)
                logger.warning("Skipping batch from %s due to invalid UTF-8: %s", path, e)
                continue
            n = len(next(iter(rows.values())))
            for i in range(n):
                yield {k: rows[k][i] for k in rows}


def strip_think_tags(text: str) -> str:
    return text.replace("<think>", "").replace("</think>", "").strip()


def normalize_reasoning_tokens(text: str) -> str:
    """Normalize balanced reasoning tags to the tokenizer's atomic delimiters."""
    text = text.replace("<think>", "<|start_think|>").replace("</think>", "<|end_think|>")
    text = re.sub(r"<\|start_think\|>\s*<\|end_think\|>\s*", "", text)
    depth = 0
    for match in re.finditer(r"<\|(start|end)_think\|>", text):
        if match.group(1) == "start":
            depth += 1
        else:
            depth -= 1
        if depth not in (0, 1):
            raise ReasoningFormatError("Assistant reasoning delimiters must be balanced and cannot nest")
    if depth != 0:
        raise ReasoningFormatError("Assistant reasoning delimiters must be balanced and cannot nest")
    return text


def text_document(text: str, source: str) -> dict:
    """Build a datakit document with a content-addressed ``id`` derived from ``text``.

    The ``id`` is the SHA-256 hex digest of the UTF-8-encoded text, so byte-identical
    documents share an id and collapse during exact-dedup normalization.
    """
    return {
        "id": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "text": text,
        "source": source,
    }


def _canonical_tool_calls(tool_calls: object) -> list[dict]:
    if not isinstance(tool_calls, list):
        raise ValueError("Chat message tool_calls must be a list")

    canonical: list[dict[str, object]] = []
    for tool_call in tool_calls:
        if not isinstance(tool_call, dict):
            raise ValueError("Each chat tool call must be an object")
        function = tool_call.get("function")
        if not isinstance(function, dict) or not isinstance(function.get("name"), str):
            raise ValueError("Each chat tool call requires a function name")

        arguments = function.get("arguments")
        if not isinstance(arguments, str):
            arguments = json.dumps(arguments, ensure_ascii=False, sort_keys=True, separators=(",", ":"))

        normalized: dict[str, object] = {}
        for key in ("id", "type"):
            value = tool_call.get(key)
            if value is not None:
                if not isinstance(value, str):
                    raise ValueError(f"Chat tool call {key} must be a string or null")
                normalized[key] = value
        normalized["function"] = {"name": function["name"], "arguments": arguments}
        canonical.append(normalized)
    return canonical


def canonical_chat_messages(messages: list[dict]) -> list[dict]:
    """Normalize roles, tool calls, and parallel calls into the canonical message contract."""
    canonical: list[dict[str, object]] = []
    for message in messages:
        role_value = message.get("role", message.get("from"))
        if not isinstance(role_value, str):
            raise ValueError("Chat messages require a string 'role' or 'from' field")
        role = CHAT_ROLE_ALIASES.get(role_value.lower(), role_value.lower())
        if role not in CANONICAL_CHAT_ROLES:
            raise ValueError(f"Unsupported chat role {role_value!r}")

        content = message.get("content", message.get("value"))
        if content is not None and not isinstance(content, str):
            raise ValueError(f"Chat message content must be a string or null, got {type(content).__name__}")
        if role == "assistant" and content is not None:
            content = normalize_reasoning_tokens(content)
        has_tool_call = bool(message.get("tool_calls") or message.get("function_call"))
        if content is None and (role != "assistant" or not has_tool_call):
            raise ValueError("Only assistant tool-call messages may have null content")

        normalized: dict[str, object] = {"role": role, "content": content}
        for key in ("name", "tool_call_id"):
            value = message.get(key)
            if value is not None:
                if not isinstance(value, str):
                    raise ValueError(f"Chat message {key} must be a string or null")
                normalized[key] = value

        tool_calls_value = message.get("tool_calls")
        legacy_function_call = message.get("function_call")
        if not tool_calls_value and legacy_function_call:
            if isinstance(legacy_function_call, str):
                legacy_function_call = json.loads(legacy_function_call)
            tool_calls_value = [{"function": legacy_function_call}]
        if tool_calls_value:
            normalized["tool_calls"] = _canonical_tool_calls(tool_calls_value)
        canonical.append(normalized)
    if not canonical:
        raise ValueError("A conversation must contain at least one message")
    return _link_tool_messages(canonical)


def _link_tool_messages(messages: list[dict[str, object]]) -> list[dict]:
    linked: list[dict] = []
    pending: dict[str, str] = {}
    for message_index, message in enumerate(messages):
        message = dict(message)
        calls = message.get("tool_calls") or []
        if calls:
            linked_calls = []
            for call_index, call_value in enumerate(calls):
                call = dict(call_value)
                call_id = call.get("id") or f"call_{message_index}_{call_index}"
                call["id"] = call_id
                function = call["function"]
                pending[call_id] = function["name"]
                linked_calls.append(call)
            message["tool_calls"] = linked_calls
        if message["role"] == "tool":
            call_id = message.get("tool_call_id")
            if call_id is None and len(pending) == 1:
                call_id = next(iter(pending))
                message["tool_call_id"] = call_id
            if call_id not in pending:
                raise ValueError("Tool messages must reference a pending tool call")
            message.setdefault("name", pending.pop(call_id))
        elif message["role"] == "user" and pending:
            raise ValueError("Tool observations must use the tool role, not the user role")
        linked.append(message)
    return linked


def inferred_tool_definitions(messages: list[dict]) -> list[dict]:
    """Build minimal JSON schemas for tools called by a canonical conversation."""
    definitions: dict[str, dict] = {}
    for message in messages:
        for call in message.get("tool_calls") or []:
            function = call["function"]
            arguments = function["arguments"]
            if isinstance(arguments, str):
                arguments = json.loads(arguments)
            if not isinstance(arguments, dict):
                raise ValueError("Tool-call arguments must be JSON objects")
            properties = definitions.setdefault(function["name"], {})
            for key, value in arguments.items():
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


def chat_document(messages: list[dict], source: str, **metadata: object) -> dict:
    """Build a canonical structured-chat document with a content-derived ID."""
    messages = canonical_chat_messages(messages)
    encoded = json.dumps(messages, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    chat_template_kwargs = metadata.get("chat_template_kwargs")
    if isinstance(chat_template_kwargs, dict):
        metadata["chat_template_kwargs"] = json.dumps(
            chat_template_kwargs, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
    return {
        "id": hashlib.sha256(encoded).hexdigest(),
        "messages": messages,
        "source": source,
        **metadata,
    }


def render_role_message(msg: dict) -> str:
    """Render a single chat message as ``<role>\\ncontent\\n</role>``.

    Missing or null content renders as an empty body.
    """
    role = msg.get("role", "unknown")
    content = msg.get("content") or ""
    return f"<{role}>\n{content}\n</{role}>"


# Outcome tags prepended to agent-rollout transcripts so the model can
# distinguish successful, failed, and unverified attempts. Agent trajectory
# sources derive the outcome differently but render the same tag text.
TRAJECTORY_SOLVED_TAG = "This trajectory solved the task successfully."
TRAJECTORY_FAILED_TAG = "This trajectory failed to solve the task."
TRAJECTORY_UNVERIFIED_TAG = "This trajectory ended before the task was verified."


def render_tool_call(tool_call: dict) -> str:
    """Render an OpenAI-style tool call as a ``<tool_call:name>`` … ``</tool_call:name>`` block.

    ``arguments`` may be a JSON string or an already-decoded value; a mapping renders one
    indented ``key: value`` line per argument, and any other non-null value renders on a
    single indented line. Malformed JSON arguments are kept as their raw string rather than
    raising, so a single bad tool call does not abort the whole transform.
    """
    func = tool_call.get("function") or {}
    name = func.get("name") or "unknown"
    args = func.get("arguments")
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            pass
    parts = [f"<tool_call:{name}>"]
    if isinstance(args, dict):
        for key, value in args.items():
            parts.append(f"  {key}: {value}")
    elif args is not None:
        parts.append(f"  {args}")
    parts.append(f"</tool_call:{name}>")
    return "\n".join(parts)


def render_tool_message(msg: dict) -> str:
    """Render a chat message that may carry ``tool_calls`` as a role-tagged block.

    Like :func:`render_role_message`, but appends any ``tool_calls`` after the text content
    and omits the content line entirely when the message has no text.
    """
    role = msg.get("role") or "unknown"
    content = msg.get("content") or ""
    tool_calls = msg.get("tool_calls")
    parts = [f"<{role}>"]
    if content:
        parts.append(content)
    if tool_calls:
        parts.extend(render_tool_call(tc) for tc in tool_calls)
    parts.append(f"</{role}>")
    return "\n".join(parts)
