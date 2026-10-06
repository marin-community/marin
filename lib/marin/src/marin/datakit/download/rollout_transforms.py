# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared transform helpers for datakit download pipelines."""

import hashlib
import json
import logging
import re
from collections.abc import Iterator, Sequence
from enum import StrEnum
from types import MappingProxyType

import pyarrow.parquet as pq
from openai_harmony import Author, Message, Role
from rigging.filesystem.factory import open_url
from zephyr import counters

from marin.datakit.chat_normalize import ChatChannel, InvalidToolCallPolicy, message_text
from marin.datakit.chat_render import render_marin_chat

logger = logging.getLogger(__name__)

CHAT_CONTROL_TOKEN = re.compile(
    r"<\|(?:begin_of_text|end_of_text|finetune_right_pad_id|start_header_id|end_header_id|"
    r"eom_id|eot_id|python_tag|reserved_special_token_\d+)\|>"
)
TOOL_WRAPPER = re.compile(r"</?tool_(?:call|response)(?:[: >])", re.IGNORECASE)
INLINE_TOOL_CALL = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
QWEN_FUNCTION_CALL = re.compile(r"<function=([^>]+)>.*?</function>", re.DOTALL)
REASONING_START = "<|start_think|>"
REASONING_END = "<|end_think|>"
REASONING_TOKEN = re.compile(r"<\|(?:start|end)_think\|>")
REASONING_SPAN = re.compile(r"<\|start_think\|>.*?<\|end_think\|>", re.DOTALL)
ASSISTANT_HEADER = "<|start_header_id|>assistant<|end_header_id|>\n"
TURN_END_TOKEN = "<|eot_id|>"
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


class LiteralToolCallFormat(StrEnum):
    HERMES = "hermes"
    QWEN3_CODER = "qwen3_coder"


class ToolObservationMarkupPolicy(StrEnum):
    REJECT_PROTOCOL_WRAPPERS = "reject_protocol_wrappers"
    PRESERVE_LITERAL = "preserve_literal"


class ToolCallLiteralFormatError(ValueError):
    """Captured tool-call syntax cannot be matched to the calls actually executed."""


def normalize_tool_call_literals(text: str, tool_calls: Sequence[dict], *, source_format: LiteralToolCallFormat) -> str:
    """Translate executed Qwen calls through Harmony while retaining surrounding assistant text."""
    if source_format is LiteralToolCallFormat.HERMES or not tool_calls:
        return text
    reasoning = list(REASONING_SPAN.finditer(text))
    blocks = [
        block
        for block in INLINE_TOOL_CALL.finditer(text)
        if not any(span.start() <= block.start() < span.end() for span in reasoning)
    ]
    functions = [list(QWEN_FUNCTION_CALL.finditer(block.group(1))) for block in blocks]
    if sum(len(group) for group in functions) != len(tool_calls):
        raise ToolCallLiteralFormatError("Qwen literal function spans differ from executed tool calls")
    output = []
    start = 0
    call_index = 0
    for block, group in zip(blocks, functions, strict=True):
        if not group:
            continue
        calls = list(tool_calls[call_index : call_index + len(group)])
        if [match.group(1).strip() for match in group] != [call["function"]["name"] for call in calls]:
            raise ToolCallLiteralFormatError("Qwen literal function names differ from executed tool calls")
        rendered = render_marin_chat(
            openai_chat_messages(
                [{"role": "assistant", "tool_calls": calls}],
                invalid_tool_call_policy=InvalidToolCallPolicy.RETAIN,
            ),
            bos_token="",
        )
        if not rendered.startswith(ASSISTANT_HEADER) or not rendered.endswith(TURN_END_TOKEN):
            raise ToolCallLiteralFormatError("Shared Marin template did not render one assistant tool-call turn")
        output.extend((text[start : block.start()], rendered[len(ASSISTANT_HEADER) : -len(TURN_END_TOKEN)]))
        start = block.end()
        call_index += len(group)
    output.append(text[start:])
    return "".join(output)


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
        if CHAT_CONTROL_TOKEN.search(value) or REASONING_TOKEN.search(value):
            raise ValueError(f"Source data contains unexpected control or reasoning tokens: {value!r}")
    elif isinstance(value, dict):
        for key, item in value.items():
            _check_source_markup(key)
            _check_source_markup(item)
    elif isinstance(value, list):
        for item in value:
            _check_source_markup(item)


def normalize_reasoning_delimiters(text: str) -> str:
    """Translate source reasoning tags without repairing or removing sampled text."""
    text = re.sub(r"<think>", REASONING_START, text, flags=re.IGNORECASE)
    return re.sub(r"</think>", REASONING_END, text, flags=re.IGNORECASE)


def normalize_reasoning_tokens(text: str) -> str:
    """Normalize balanced reasoning tags to the canonical source delimiters."""
    text = normalize_reasoning_delimiters(text)
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
    message: dict,
    author: Author,
    content: str | None,
    reasoning: str,
    index: int,
    assistant_prefill: str,
    invalid_tool_call_policy: InvalidToolCallPolicy,
) -> tuple[list[Message], dict[str, str]]:
    output: list[Message] = []
    pending: dict[str, str] = {}
    if "unparsed_content" in message:
        literal = message["unparsed_content"]
        if (
            invalid_tool_call_policy != InvalidToolCallPolicy.RETAIN
            or content
            or reasoning
            or message.get("tool_calls")
            or message.get("function_call")
            or not isinstance(literal, str)
            or not literal.strip()
        ):
            raise ValueError("Retained unparsed assistant text requires an otherwise empty parsed message")
        _check_source_markup(literal)
        return [Message.from_author_and_content(author, literal).with_channel(ChatChannel.FINAL)], pending
    content = content or ""
    if (
        assistant_prefill
        and re.search(r"</think>|<\|end_think\|>", content)
        and not re.search(r"<think>|<\|start_think\|>", content)
    ):
        content = assistant_prefill + content
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
        if (
            not isinstance(tool_name, str)
            or not tool_name
            or (
                invalid_tool_call_policy == InvalidToolCallPolicy.REJECT
                and re.fullmatch(r"[A-Za-z0-9_.:-]+", tool_name) is None
            )
        ):
            raise ValueError("Each tool call requires a valid function name")
        _check_source_markup(tool_name)
        call_id = call.get("id") or f"call_{index}_{call_index}"
        if not isinstance(call_id, str) or call_id in pending:
            raise ValueError("Source tool-call IDs must be unique strings")
        arguments = function.get("arguments")
        if isinstance(arguments, str):
            try:
                decoded = json.loads(arguments)
            except json.JSONDecodeError:
                if invalid_tool_call_policy == InvalidToolCallPolicy.REJECT:
                    raise
            else:
                if isinstance(decoded, dict) or invalid_tool_call_policy == InvalidToolCallPolicy.REJECT:
                    arguments = decoded
        if not isinstance(arguments, dict) and not (
            invalid_tool_call_policy == InvalidToolCallPolicy.RETAIN and isinstance(arguments, str)
        ):
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


def openai_chat_messages(
    messages: list[dict],
    *,
    assistant_prefill: str = "",
    invalid_tool_call_policy: InvalidToolCallPolicy = InvalidToolCallPolicy.REJECT,
    tool_observation_markup_policy: ToolObservationMarkupPolicy = (ToolObservationMarkupPolicy.REJECT_PROTOCOL_WRAPPERS),
) -> list[Message]:
    """Normalize OpenAI-style source turns directly into Harmony messages.

    Interpret source role aliases, reasoning tags, and function calls here.
    Source reasoning XML is interpreted only in assistant content; other roles
    retain it as literal text, while target chat control tokens remain rejected.
    Source IDs link observations before being discarded; parallel observations
    are emitted in call order, including repeated calls to the same function.
    An explicit template prefill restores a reasoning opener absent from the
    sampled completion; malformed reasoning remains rejected.
    RETAIN preserves malformed argument strings and undeclared call names for
    offline preferences. Unparsed assistant text must be supplied explicitly.
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
        if isinstance(content, list):
            if any(
                not isinstance(part, dict) or part.get("type") != "text" or not isinstance(part.get("text"), str)
                for part in content
            ):
                raise ValueError("Source messages require text content blocks")
            content = "".join(part["text"] for part in content)
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
                if (
                    tool_observation_markup_policy is ToolObservationMarkupPolicy.REJECT_PROTOCOL_WRAPPERS
                    and TOOL_WRAPPER.search(content)
                ):
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
            case _ if pending and (role != Role.ASSISTANT or observations):
                raise ValueError("Every source tool call must receive an observation before another turn")
            case Role.SYSTEM | Role.DEVELOPER | Role.USER:
                if content is None:
                    raise ValueError("Only assistant reasoning or tool-call messages may have null content")
                _check_source_markup(content)
                if role == Role.USER and output and output[-1].author.role == Role.USER:
                    output[-1] = Message.from_author_and_content(author, message_text(output[-1]) + "\n\n" + content)
                else:
                    output.append(Message.from_author_and_content(author, content))
            case Role.ASSISTANT:
                if (
                    pending
                    and not content
                    and not reasoning
                    and not message.get("tool_calls")
                    and not message.get("function_call")
                    and "unparsed_content" not in message
                ):
                    # Responses can include an empty text item after a call.
                    # It has no sampled text and leaves the tool handoff pending.
                    continue
                assistant_messages, calls = _assistant_messages(
                    message, author, content, reasoning, index, assistant_prefill, invalid_tool_call_policy
                )
                if seen_call_ids.intersection(calls):
                    raise ValueError("Source tool-call IDs must be unique strings")
                seen_call_ids.update(calls)
                if pending:
                    # Responses can serialize text after calls within the same assistant turn.
                    text_messages = assistant_messages[: -len(calls)] if calls else assistant_messages
                    for text_message in text_messages:
                        if text_message.channel == ChatChannel.FINAL:
                            text_message.with_channel(ChatChannel.COMMENTARY)
                    output[len(output) - len(pending) : len(output) - len(pending)] = text_messages
                    assistant_messages = assistant_messages[-len(calls) :] if calls else []
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


def openai_chat_document(
    messages: list[dict],
    source: str,
    *,
    assistant_prefill: str = "",
    invalid_tool_call_policy: InvalidToolCallPolicy = InvalidToolCallPolicy.REJECT,
    tool_observation_markup_policy: ToolObservationMarkupPolicy = (ToolObservationMarkupPolicy.REJECT_PROTOCOL_WRAPPERS),
    **metadata: object,
) -> dict:
    """Build a Harmony artifact from an OpenAI-style source conversation."""
    _check_source_markup(metadata)
    return chat_document(
        openai_chat_messages(
            messages,
            assistant_prefill=assistant_prefill,
            invalid_tool_call_policy=invalid_tool_call_policy,
            tool_observation_markup_policy=tool_observation_markup_policy,
        ),
        source,
        **metadata,
    )


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
