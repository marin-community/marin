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
from openai_harmony import Author, Message, Role
from rigging.filesystem.factory import open_url
from zephyr import counters

from marin.datakit.chat_normalize import ChatChannel

logger = logging.getLogger(__name__)

CHAT_CONTROL_TOKEN = re.compile(
    r"<\|(?:begin_of_text|end_of_text|finetune_right_pad_id|start_header_id|end_header_id|"
    r"eom_id|eot_id|python_tag|reserved_special_token_\d+)\|>"
)
REASONING_START = "<|start_think|>"
REASONING_END = "<|end_think|>"
REASONING_TOKEN = re.compile(r"<\|(?:start|end)_think\|>")
CHAT_ROLE_ALIASES = MappingProxyType(
    {
        "bot": Role.ASSISTANT,
        "function": Role.TOOL,
        "gpt": Role.ASSISTANT,
        "human": Role.USER,
        "model": Role.ASSISTANT,
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


def _check_source_markup(value: object) -> None:
    if isinstance(value, str):
        if (
            CHAT_CONTROL_TOKEN.search(value)
            or REASONING_TOKEN.search(value)
            or re.search(r"</?think>", value, re.IGNORECASE)
        ):
            raise ValueError(f"Source data contains unexpected control or reasoning tokens: {value!r}")
    elif isinstance(value, dict):
        for key, item in value.items():
            _check_source_markup(key)
            _check_source_markup(item)
    elif isinstance(value, list):
        for item in value:
            _check_source_markup(item)


def normalize_reasoning_tokens(text: str) -> str:
    """Normalize balanced reasoning tags to the canonical source delimiters."""
    text = re.sub(r"<think>", REASONING_START, text, flags=re.IGNORECASE)
    text = re.sub(r"</think>", REASONING_END, text, flags=re.IGNORECASE)
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
    text = text.strip()
    return text


def _assistant_messages(
    message: dict, author: Author, content: str | None, reasoning: str, index: int
) -> tuple[list[Message], dict[str, str]]:
    output: list[Message] = []
    pending: dict[str, str] = {}
    content = content or ""
    content = normalize_reasoning_tokens(content)
    if REASONING_TOKEN.search(content):
        match = re.fullmatch(r"<\|start_think\|>(.*?)<\|end_think\|>(.*)", content, re.DOTALL)
        if match is None or not match[1].strip():
            raise ReasoningFormatError("Assistant reasoning must be a non-empty prefix of the reply")
        if reasoning:
            raise ValueError("Assistant reasoning is present in both content and reasoning_content")
        reasoning, content = match.groups()
    _check_source_markup(reasoning)
    _check_source_markup(content)
    if re.search(r"<tool_call(?::[^>]*)?>", content, re.IGNORECASE):
        raise ValueError("Inline tool-call syntax must be split out by the source adapter")
    calls = message.get("tool_calls") or []
    if not calls and message.get("function_call"):
        function = message["function_call"]
        if isinstance(function, str):
            function = json.loads(function)
        calls = [{"function": function}]
    if not isinstance(calls, list):
        raise ValueError("Source tool_calls must be a list")
    if not content.strip() and not reasoning.strip() and not calls:
        raise ValueError("Assistant turns must contain text, reasoning, or a tool call")
    if reasoning.strip():
        output.append(Message.from_author_and_content(author, reasoning.strip()).with_channel(ChatChannel.ANALYSIS))
    if content.strip():
        output.append(
            Message.from_author_and_content(author, content.strip()).with_channel(
                ChatChannel.COMMENTARY if calls else ChatChannel.FINAL
            )
        )
    for call_index, call in enumerate(calls):
        if not isinstance(call, dict) or not isinstance(call.get("function"), dict):
            raise ValueError("Each tool call requires a function object")
        function = call["function"]
        tool_name = function.get("name")
        if not isinstance(tool_name, str) or re.fullmatch(r"[A-Za-z0-9_.:-]+", tool_name) is None:
            raise ValueError("Each tool call requires a valid function name")
        call_id = call.get("id") or f"call_{index}_{call_index}"
        if not isinstance(call_id, str) or call_id in pending:
            raise ValueError("Source tool-call IDs must be unique strings")
        arguments = function.get("arguments")
        if isinstance(arguments, str):
            arguments = json.loads(arguments)
        if not isinstance(arguments, dict):
            raise ValueError("Tool-call arguments must be JSON objects")
        _check_source_markup(arguments)
        output.append(
            Message.from_author_and_content(
                author, json.dumps(arguments, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            )
            .with_channel(ChatChannel.COMMENTARY)
            .with_recipient(f"functions.{tool_name}")
        )
        pending[call_id] = tool_name
    return output, pending


def openai_chat_messages(messages: list[dict]) -> list[Message]:
    """Normalize OpenAI-style source turns directly into Harmony messages.

    Interpret source role aliases, reasoning tags, and function calls here.
    Source IDs link observations before being discarded; parallel observations
    are emitted in call order, including repeated calls to the same function.
    """
    output: list[Message] = []
    pending: dict[str, str] = {}
    observations: dict[str, str] = {}
    seen_call_ids: set[str] = set()
    for index, message in enumerate(messages):
        role_value = message.get("role", message.get("from"))
        if not isinstance(role_value, str):
            raise ValueError(f"Chat messages require a string 'role' or 'from' field, got {role_value!r}")
        role = Role(CHAT_ROLE_ALIASES.get(role_value.lower(), role_value.lower()))
        content = message.get("content", message.get("value"))
        if content is not None and not isinstance(content, str):
            raise ValueError(f"Source message content must be a string or null, got {content!r}")
        name = message.get("name")
        if name is not None and not isinstance(name, str):
            raise ValueError(f"Source message names must be strings, got {name!r}")
        author = Author.new(role, name)
        reasoning = message.get("reasoning_content") or ""
        if not isinstance(reasoning, str):
            raise ValueError(f"reasoning_content must be a string, got {reasoning!r}")
        if reasoning and role != Role.ASSISTANT:
            raise ValueError(f"reasoning_content is only valid for assistant messages, got role {role.value!r}")
        _check_source_markup(reasoning)
        match role:
            case Role.TOOL:
                if content is None:
                    raise ValueError("Tool observations must contain text")
                _check_source_markup(content)
                if re.search(r"</?tool_(?:call|response)(?:[: >])", content, re.IGNORECASE):
                    raise ValueError("Tool observations must not contain chat protocol wrappers")
                call_id = message.get("tool_call_id")
                unanswered = pending.keys() - observations.keys()
                if call_id is None and len(unanswered) == 1:
                    call_id = next(iter(unanswered))
                if not isinstance(call_id, str) or call_id not in unanswered:
                    raise ValueError("Tool messages must reference a pending tool call")
                if name is not None and name != pending[call_id]:
                    raise ValueError("Tool observation name must match its call")
                observations[call_id] = content
                if len(observations) == len(pending):
                    for call_id, tool_name in pending.items():
                        output.append(
                            Message.from_author_and_content(
                                Author.new(Role.TOOL, f"functions.{tool_name}"), observations[call_id]
                            )
                            .with_channel(ChatChannel.COMMENTARY)
                            .with_recipient(Role.ASSISTANT.value)
                        )
                    pending.clear()
                    observations.clear()
            case _ if pending:
                raise ValueError("Every source tool call must receive an observation before another turn")
            case Role.SYSTEM | Role.DEVELOPER | Role.USER:
                if content is None:
                    raise ValueError("Only assistant reasoning or tool-call messages may have null content")
                _check_source_markup(content)
                output.append(Message.from_author_and_content(author, content))
            case Role.ASSISTANT:
                assistant_messages, calls = _assistant_messages(message, author, content, reasoning, index)
                if seen_call_ids.intersection(calls):
                    raise ValueError("Source tool-call IDs must be unique strings")
                seen_call_ids.update(calls)
                pending.update(calls)
                output.extend(assistant_messages)
    if observations:
        raise ValueError("A parallel tool-call batch is missing observations")
    return output


def chat_document(messages: list[Message], source: str, **metadata: object) -> dict:
    """Serialize canonical Harmony messages into a source artifact."""
    serialized = [message.to_dict() for message in messages]
    encoded = json.dumps(serialized, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    chat_template_kwargs = metadata.get("chat_template_kwargs")
    if isinstance(chat_template_kwargs, dict):
        metadata["chat_template_kwargs"] = json.dumps(
            chat_template_kwargs, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
    return {"id": hashlib.sha256(encoded).hexdigest(), "messages": serialized, "source": source, **metadata}


def openai_chat_document(messages: list[dict], source: str, **metadata: object) -> dict:
    """Build a Harmony artifact from an OpenAI-style source conversation."""
    _check_source_markup(metadata)
    return chat_document(openai_chat_messages(messages), source, **metadata)


def checked_openai_chat_document(
    messages: list[dict], source: str, *, counter_prefix: str, **metadata: object
) -> list[dict]:
    """Normalize source turns to Harmony, quarantining malformed source rows."""
    try:
        return [openai_chat_document(messages, source, **metadata)]
    except (UnicodeError, ValueError) as error:
        counters.pipeline.update_counter(f"{counter_prefix}/quarantined", 1)
        counters.pipeline.update_counter(f"{counter_prefix}/quarantined/{type(error).__name__}", 1)
        return []


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
