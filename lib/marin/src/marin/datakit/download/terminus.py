# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Parse the Terminus JSON-command conversation protocol."""

import json
from enum import StrEnum
from typing import NamedTuple

from marin.datakit.download.rollout_transforms import TOOL_WRAPPER

_CLOSING_SUFFIXES = ("", "}", "]}", "}]}")


class TerminusParseMode(StrEnum):
    """Select complete source syntax or supported harness repairs."""

    STRICT = "strict"
    REPAIR = "repair"


class ThinkTokens(NamedTuple):
    """The leading reasoning delimiters used by a source's Terminus responses."""

    start: str
    end: str


def _json_command_payload_and_prefix(
    content: str, think_tokens: ThinkTokens | None, mode: TerminusParseMode
) -> tuple[dict, str] | None:
    decoder = json.JSONDecoder()
    if think_tokens is not None and content.lstrip().startswith(think_tokens.start):
        stripped = content.strip()
        if stripped.endswith(think_tokens.end):
            wrapped = stripped[len(think_tokens.start) : -len(think_tokens.end)].strip()
            try:
                payload = decoder.decode(wrapped)
            except json.JSONDecodeError:
                pass
            else:
                if isinstance(payload, dict) and isinstance(payload.get("commands"), list):
                    return payload, ""
    suffixes = _CLOSING_SUFFIXES if mode is TerminusParseMode.REPAIR else ("",)
    depth = 0
    index = 0
    while index < len(content):
        if think_tokens is not None:
            if content.startswith(think_tokens.start, index):
                depth += 1
                index += len(think_tokens.start)
                continue
            if depth and content.startswith(think_tokens.end, index):
                depth -= 1
                index += len(think_tokens.end)
                continue
        if depth or content[index] != "{":
            index += 1
            continue
        body = content[index:].rstrip()
        for suffix in suffixes:
            try:
                payload, _ = decoder.raw_decode(body + suffix)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict) and isinstance(payload.get("commands"), list):
                prefix = content[:index].strip()
                if mode is TerminusParseMode.REPAIR and think_tokens is not None and think_tokens.start in prefix:
                    reasoning = prefix.replace(think_tokens.start, "").replace(think_tokens.end, "").strip()
                    prefix = f"{think_tokens.start}{reasoning}{think_tokens.end}" if reasoning else ""
                return payload, prefix
            break
        index += 1
    return None


def terminus_protocol_messages(
    conversations: list[dict], think_tokens: ThinkTokens | None, *, mode: TerminusParseMode
) -> list[dict] | None:
    """Extract Terminus commands and optional leading reasoning from a conversation.

    The source must supply its reasoning delimiters, or None if it
    has none. A span containing only command JSON is treated as the response.
    Otherwise, only a nonempty leading reasoning span is retained; incidental
    prose before the first command JSON is discarded. Repair mode merges closed
    reasoning spans and supplies supported missing JSON closing delimiters.
    Commands inside unfinished reasoning are rejected in both modes.
    """
    messages: list[dict] = []
    pending_observation = False
    for message in conversations:
        role = message.get("role")
        content = message.get("content")
        if not isinstance(content, str):
            return None
        if role == "user":
            if not content.strip() or (pending_observation and TOOL_WRAPPER.search(content)):
                return None
            messages.append({"role": "user", "content": content})
            pending_observation = False
            continue
        if role != "assistant":
            messages.append(dict(message))
            continue

        parsed = _json_command_payload_and_prefix(content, think_tokens, mode)
        if parsed is None:
            return None
        payload, prefix = parsed
        response = {"role": "assistant", "content": json.dumps(payload, ensure_ascii=False)}
        if think_tokens is not None and prefix.startswith(think_tokens.start):
            reasoning, separator, _ = prefix.removeprefix(think_tokens.start).partition(think_tokens.end)
            if (
                not separator
                or not reasoning.strip()
                or prefix.count(think_tokens.start) != 1
                or prefix.count(think_tokens.end) != 1
            ):
                return None
            response["reasoning_content"] = reasoning.strip()
        messages.append(response)
        pending_observation = bool(payload["commands"]) or not payload.get("task_complete")
    if not messages or messages[-1]["role"] != "assistant":
        return None
    return messages
