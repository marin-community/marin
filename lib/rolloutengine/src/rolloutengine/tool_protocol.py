# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Score completed Hermes tool-call continuations without executing tools."""

import json
import re
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

TOOL_OPEN = "<tool_call>"
TOOL_CLOSE = "</tool_call>"
THINK_OPEN = "<|start_think|>"
THINK_CLOSE = "<|end_think|>"
_MARKER = re.compile("|".join(re.escape(marker) for marker in (TOOL_OPEN, THINK_OPEN, THINK_CLOSE)))


class ToolCallStatus(StrEnum):
    VALID = "valid"
    REPEATED_OPENER = "repeated_opener"
    MALFORMED_JSON = "malformed_json"
    INVALID_ENVELOPE = "invalid_envelope"
    MISSING_CLOSER = "missing_closer"


@dataclass(frozen=True)
class ToolCallRegion:
    start: int
    payload_start: int
    end: int
    status: ToolCallStatus


def _json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> Any:
    raise ValueError(f"Non-JSON constant: {value}")


_JSON = json.JSONDecoder(object_pairs_hook=_json_object, parse_constant=_reject_constant)


def _call_region(text: str, start: int) -> ToolCallRegion:
    payload_start = start + len(TOOL_OPEN)
    position = payload_start
    while position < len(text) and text[position] in " \t\r\n":
        position += 1
    if text.startswith(TOOL_OPEN, position):
        return ToolCallRegion(start, payload_start, position + len(TOOL_OPEN), ToolCallStatus.REPEATED_OPENER)
    try:
        payload, end = _JSON.raw_decode(text, position)
    except ValueError:
        return ToolCallRegion(start, payload_start, len(text), ToolCallStatus.MALFORMED_JSON)
    if (
        not isinstance(payload, dict)
        or not isinstance(payload.get("name"), str)
        or not payload["name"]
        or not isinstance(payload.get("arguments"), dict)
    ):
        return ToolCallRegion(start, payload_start, end, ToolCallStatus.INVALID_ENVELOPE)
    while end < len(text) and text[end] in " \t\r\n":
        end += 1
    if not text.startswith(TOOL_CLOSE, end):
        return ToolCallRegion(start, payload_start, end, ToolCallStatus.MISSING_CLOSER)
    return ToolCallRegion(start, payload_start, end + len(TOOL_CLOSE), ToolCallStatus.VALID)


def tool_call_regions(text: str, *, initial_thinking: bool) -> tuple[ToolCallRegion, ...]:
    """Find calls outside thinking, stopping at the first malformed call.

    Offsets refer to the supplied decoded assistant span. JSON string contents
    are consumed as part of the payload, so quoted protocol markers are not
    interpreted as calls or thinking boundaries. After a malformed call the
    parser cannot safely recover its state, and leaves the remaining span
    unclassified. Callers must supply the thinking state at the span start.
    """
    regions = []
    position = 0
    thinking = initial_thinking
    while marker := _MARKER.search(text, position):
        position = marker.end()
        match marker.group():
            case marker_text if marker_text == THINK_OPEN:
                thinking = True
            case marker_text if marker_text == THINK_CLOSE:
                thinking = False
            case _:
                if thinking:
                    continue
                region = _call_region(text, marker.start())
                regions.append(region)
                if region.status != ToolCallStatus.VALID:
                    break
                position = region.end
    return tuple(regions)


def score_tool_call_continuation(text: str) -> ToolCallStatus:
    """Score one completed continuation of an outside-thinking opening delimiter.

    The fixed prompt ends with ``<tool_call>``. The generated text must contain
    a nonempty function name, object arguments, and the closing delimiter.
    This checks the first call's syntax only, not its tool schema, execution,
    task answer, or any text after that call. Invoke only after generation ends;
    an unfinished stream is not a scored failure. Infrastructure failures must
    be reported separately rather than passed as empty generated text.
    """
    return _call_region(TOOL_OPEN + text, 0).status
