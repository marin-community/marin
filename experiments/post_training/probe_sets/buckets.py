# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Token-level KL buckets over recorded conversations.

A recorded conversation (a P_agent trajectory or a reference continuation) renders
once through the shared chat template with the architecture's tokenizer. Every
token gets an assistant mask, its assistant-turn index, and one bucket:

- ``reasoning``: ``reasoning_content``, plus Terminus-2's ``analysis`` and ``plan``
  fields, which carry its deliberation inside the JSON reply;
- ``tool_name`` and ``tool_arguments``: a structured tool call's name and arguments,
  or a Terminus-2 command's ``keystrokes``;
- ``final_message``: the last assistant turn's remaining content;
- ``assistant_text``: earlier assistant content outside the buckets above, including
  Terminus-2's JSON scaffolding;
- ``delimiter``: generated structural tokens (think open/close, the closing turn token);
- ``context``: everything unscored (system, user and tool turns, headers).

An assistant turn's reasoning comes either from ``reasoning_content`` (the template
renders the think markers) or from think markers left verbatim in a sampled reply's
``content``. Sampled replies do not always pair their markers: a reply can open a think
block it never closes (the reasoning runs to the end of the reply) or close one it never
opened (the generation prompt carries no open marker, so the reasoning starts at the
reply's start). Passing the raw reply as ``content`` keeps the rendered tokens equal to
what the model sampled; splitting an unpaired reply into ``reasoning_content`` would
make the template add a marker the model never produced.

Tool observations are fixed context, so KL is scored only where the mask is set.
Reasoning tokens also carry their relative-position decile within the reasoning span,
for P_think's thinking-depth buckets.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from experiments.post_training.chat_template import FAMILY_TOKEN_MAPS, shared_template


class Bucket(StrEnum):
    REASONING = "reasoning"
    TOOL_NAME = "tool_name"
    TOOL_ARGUMENTS = "tool_arguments"
    FINAL_MESSAGE = "final_message"
    ASSISTANT_TEXT = "assistant_text"
    DELIMITER = "delimiter"
    CONTEXT = "context"


_TERMINUS_REASONING_FIELDS = ("analysis", "plan")
_TERMINUS_COMMAND_FIELD = "keystrokes"


@dataclass(frozen=True)
class TaggedSequence:
    input_ids: list[int]
    assistant_mask: list[int]
    assistant_turn: list[int]
    buckets: list[Bucket]
    reasoning_decile: list[int]


def _json_string_values(content: str, key: str) -> list[tuple[int, int]]:
    """Character spans of every ``"key": "<value>"`` string value in ``content``, quotes excluded."""
    spans = []
    for match in re.finditer(rf'"{re.escape(key)}"\s*:\s*"', content):
        start = index = match.end()
        while index < len(content) and content[index] != '"':
            index += 2 if content[index] == "\\" else 1
        if index >= len(content):
            raise ValueError(f"unterminated JSON string for {key!r}")
        spans.append((start, index))
    return spans


def _content_spans(content: str) -> list[tuple[int, int, Bucket]]:
    """Terminus-2 spans inside an assistant reply's JSON, relative to the content."""
    spans = [
        (start, end, Bucket.REASONING)
        for key in _TERMINUS_REASONING_FIELDS
        for start, end in _json_string_values(content, key)
    ]
    spans += [
        (start, end, Bucket.TOOL_ARGUMENTS) for start, end in _json_string_values(content, _TERMINUS_COMMAND_FIELD)
    ]
    return sorted(spans)


def _think_marker_spans(content: str, think_open: str, think_close: str) -> list[tuple[int, int, Bucket]]:
    """Reasoning and delimiter spans from think markers in a reply's raw content.

    An open marker starts a reasoning span (a repeated open inside a span is only a
    delimiter); a close ends it. A close before any marker ends a span that started at
    the reply's start; a later close with no open is only a delimiter. A span still open
    at the end runs to the end of the content. Reasoning spans come first so the
    delimiters paint over any marker inside one.
    """
    reasoning: list[tuple[int, int, Bucket]] = []
    delimiters: list[tuple[int, int, Bucket]] = []
    reasoning_start: int | None = None
    seen_marker = False
    for match in re.finditer(f"{re.escape(think_open)}|{re.escape(think_close)}", content):
        delimiters.append((match.start(), match.end(), Bucket.DELIMITER))
        if match.group() == think_open:
            if reasoning_start is None:
                reasoning_start = match.end()
        elif reasoning_start is not None:
            reasoning.append((reasoning_start, match.start(), Bucket.REASONING))
            reasoning_start = None
        elif not seen_marker:
            reasoning.append((0, match.start(), Bucket.REASONING))
        seen_marker = True
    if reasoning_start is not None:
        reasoning.append((reasoning_start, len(content), Bucket.REASONING))
    return reasoning + delimiters


def _find(text: str, piece: str, cursor: int) -> int:
    position = text.find(piece, cursor)
    if position < 0:
        raise ValueError(f"rendered conversation is missing {piece[:40]!r} after offset {cursor}")
    return position


def _char_buckets(text: str, messages: list[dict[str, Any]], family: str) -> tuple[list[Bucket], list[int]]:
    """Per-character bucket and assistant-turn index over the rendered conversation."""
    tokens = FAMILY_TOKEN_MAPS[family]
    buckets = [Bucket.CONTEXT] * len(text)
    turns = [-1] * len(text)
    header = f"{tokens.turn_open_prefix}assistant{tokens.turn_open_suffix}\n"
    assistant_indices = [index for index, message in enumerate(messages) if message["role"] == "assistant"]
    cursor = 0

    def paint(start: int, end: int, bucket: Bucket) -> None:
        buckets[start:end] = [bucket] * (end - start)

    for index, message in enumerate(messages):
        if message["role"] != "assistant":
            if message.get("content"):
                cursor = _find(text, message["content"], cursor) + len(message["content"])
            continue
        turn = assistant_indices.index(index)
        start = _find(text, header, cursor) + len(header)
        content_bucket = Bucket.FINAL_MESSAGE if index == assistant_indices[-1] else Bucket.ASSISTANT_TEXT
        cursor = start
        if message.get("reasoning_content"):
            think_open = _find(text, tokens.think_open, cursor)
            paint(think_open, think_open + len(tokens.think_open), Bucket.DELIMITER)
            cursor = _find(text, message["reasoning_content"], think_open + len(tokens.think_open))
            paint(cursor, cursor + len(message["reasoning_content"]), Bucket.REASONING)
            cursor += len(message["reasoning_content"])
            think_close = _find(text, tokens.think_close, cursor)
            paint(think_close, think_close + len(tokens.think_close), Bucket.DELIMITER)
            cursor = think_close + len(tokens.think_close)
        if message.get("content"):
            content_start = _find(text, message["content"], cursor)
            paint(content_start, content_start + len(message["content"]), content_bucket)
            spans = _content_spans(message["content"])
            spans += _think_marker_spans(message["content"], tokens.think_open, tokens.think_close)
            for span_start, span_end, bucket in spans:
                paint(content_start + span_start, content_start + span_end, bucket)
            cursor = content_start + len(message["content"])
        for call in message.get("tool_calls") or []:
            function = call.get("function", call)
            name = json.dumps(function["name"])
            arguments = (
                function["arguments"] if isinstance(function["arguments"], str) else json.dumps(function["arguments"])
            )
            name_start = _find(text, name, cursor)
            paint(name_start, name_start + len(name), Bucket.TOOL_NAME)
            arguments_start = _find(text, arguments, name_start + len(name))
            paint(arguments_start, arguments_start + len(arguments), Bucket.TOOL_ARGUMENTS)
            cursor = arguments_start + len(arguments)
        close = _find(text, tokens.turn_close, cursor)
        for position in range(start, close):
            if buckets[position] is Bucket.CONTEXT:
                buckets[position] = content_bucket
        paint(close, close + len(tokens.turn_close), Bucket.DELIMITER)
        turns[start : close + len(tokens.turn_close)] = [turn] * (close + len(tokens.turn_close) - start)
        cursor = close + len(tokens.turn_close)
    return buckets, turns


def tag_conversation(tokenizer, family: str, messages: list[dict[str, Any]]) -> TaggedSequence:
    """Render ``messages`` through the shared template and tag every token.

    A token takes the bucket of its first character. ``tokenizer`` must be a fast
    tokenizer, since tagging reads its offset mapping.
    """
    text = tokenizer.apply_chat_template(
        messages, chat_template=shared_template(FAMILY_TOKEN_MAPS[family]), tokenize=False
    )
    char_buckets, char_turns = _char_buckets(text, messages, family)
    encoding = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    buckets: list[Bucket] = []
    turns: list[int] = []
    for start, end in encoding["offset_mapping"]:
        position = start if end > start else max(start - 1, 0)
        buckets.append(char_buckets[position])
        turns.append(char_turns[position])
    mask = [int(bucket is not Bucket.CONTEXT) for bucket in buckets]
    return TaggedSequence(
        input_ids=list(encoding["input_ids"]),
        assistant_mask=mask,
        assistant_turn=turns,
        buckets=buckets,
        reasoning_decile=_reasoning_deciles(buckets, turns),
    )


def _reasoning_deciles(buckets: list[Bucket], turns: list[int]) -> list[int]:
    """Each reasoning token's decile (0-9) of position within its turn's reasoning tokens; -1 elsewhere."""
    positions: dict[int, list[int]] = {}
    for index, bucket in enumerate(buckets):
        if bucket is Bucket.REASONING:
            positions.setdefault(turns[index], []).append(index)
    deciles = [-1] * len(buckets)
    for members in positions.values():
        for rank, index in enumerate(members):
            deciles[index] = min(9, rank * 10 // len(members))
    return deciles
