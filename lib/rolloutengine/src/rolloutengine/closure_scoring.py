# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Content-preserving scoring for explicitly required structure completion."""

import json
from dataclasses import dataclass
from enum import StrEnum

from rolloutengine.structure_closure import (
    ClosureState,
    PrefixStatus,
    delimited_json_closure_state,
    json_closure_state,
    paired_marker_closure_state,
    required_code_fence_state,
)
from rolloutengine.tool_protocol import JSON_WHITESPACE, THINK_CLOSE, THINK_OPEN, TOOL_CLOSE, TOOL_OPEN

SNOWBALL_TURN_END_MARKERS = ("<|eot_id|>", "<|end_of_text|>")
CLOSURE_REGION_SCORER = "closure_region_v1"


def closure_response_content(text: str, *, stop_reason: str) -> str:
    """Remove one terminal Snowball control marker from a normally stopped turn.

    Interior markers, repeated terminal markers, and truncated generations remain
    subject to structural validation. Tool and reasoning closers are content.
    """
    if stop_reason == "stop":
        for marker in SNOWBALL_TURN_END_MARKERS:
            if text.endswith(marker):
                return text[: -len(marker)]
    return text


class ClosureFormat(StrEnum):
    """Supported wire formats; tool and reasoning use Hermes and Snowball markers."""

    JSON = "json"
    TOOL = "tool"
    REASONING = "reasoning"
    CODE_FENCE = "code_fence"


@dataclass(frozen=True)
class ClosureScore:
    syntax_complete: bool
    content_preserved: bool
    success: bool


@dataclass(frozen=True)
class ClosureRegionScore:
    syntax_complete: bool
    content_preserved: bool
    success: bool
    trailing_content: bool
    whole_response_success: bool


def closure_state(text: str, structure_format: ClosureFormat) -> ClosureState:
    if structure_format == ClosureFormat.JSON:
        return json_closure_state(text)
    if structure_format == ClosureFormat.TOOL:
        return delimited_json_closure_state(text, opener=TOOL_OPEN, closer=TOOL_CLOSE)
    if structure_format == ClosureFormat.REASONING:
        return paired_marker_closure_state(text, opener=THINK_OPEN, closer=THINK_CLOSE)
    if structure_format == ClosureFormat.CODE_FENCE:
        return required_code_fence_state(text)
    raise ValueError(f"Unsupported closure format: {structure_format}")


def _json_content(text: str, structure_format: ClosureFormat) -> str:
    if structure_format == ClosureFormat.TOOL:
        text = text[len(TOOL_OPEN) :].rstrip(JSON_WHITESPACE)[: -len(TOOL_CLOSE)]
    # Canonical serialization ignores object order and whitespace, but preserves
    # distinctions such as true versus 1 that Python container equality loses.
    return json.dumps(json.loads(text), sort_keys=True, ensure_ascii=True, allow_nan=False)


def score_closure(text: str, expected: str, structure_format: ClosureFormat) -> ClosureScore:
    """Check syntax and known content; never infer what unfinished content meant.

    JSON permits key-order and whitespace changes. Numeric spellings canonicalize
    through Python's JSON decoder; booleans remain distinct from numbers. Opaque
    reasoning and fence content must match except for trailing whitespace.
    """
    complete = closure_state(text, structure_format).status == PrefixStatus.COMPLETE
    if not complete:
        return ClosureScore(False, False, False)
    if structure_format in (ClosureFormat.JSON, ClosureFormat.TOOL):
        reference = _json_content(expected, structure_format)
        try:
            candidate = _json_content(text, structure_format)
        except ValueError:
            # A syntactically valid exponent can overflow the JSON decoder's
            # float representation. It cannot preserve a finite reference value.
            return ClosureScore(True, False, False)
        preserved = candidate == reference
    else:
        preserved = text.rstrip(JSON_WHITESPACE) == expected.rstrip(JSON_WHITESPACE)
    return ClosureScore(True, preserved, preserved)


def score_closure_region(text: str, expected: str, structure_format: ClosureFormat) -> ClosureRegionScore:
    """Score the first outer structure; report output after its boundary separately.

    Text includes the supplied structure prefix and generated continuation.
    Never skip illegal leading syntax to search for a later valid structure.
    Expected content still rejects premature completion of continue controls.
    """
    state = closure_state(text, structure_format)
    if state.completion_offset is None:
        return ClosureRegionScore(False, False, False, False, False)
    score = score_closure(text[: state.completion_offset], expected, structure_format)
    trailing = bool(text[state.completion_offset :].strip(JSON_WHITESPACE))
    return ClosureRegionScore(
        score.syntax_complete, score.content_preserved, score.success, trailing, score.success and not trailing
    )
