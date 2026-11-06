# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Content-preserving scoring for explicitly required structure completion."""

import json
import re
from dataclasses import dataclass
from enum import StrEnum

from verifyit.json_comparison import JsonValue, NumericTypePolicy, json_values_equal
from verifyit.structure_closure import (
    ClosureState,
    PrefixStatus,
    delimited_json_closure_state,
    json_closure_state,
    paired_marker_closure_state,
    required_code_fence_state,
)
from verifyit.tool_call_markup import JSON_WHITESPACE, THINK_CLOSE, THINK_OPEN, TOOL_CLOSE, TOOL_OPEN

CLOSURE_REGION_SCORER = "closure_region_v1"
SNOWBALL_TURN_END_MARKERS = ("<|eot_id|>", "<|end_of_text|>")
_STRUCTURAL_CLOSERS = frozenset((TOOL_CLOSE, THINK_CLOSE, "}", "]", '"'))


def closure_response_content(text: str, *, stop_reason: str, terminal_markers: tuple[str, ...]) -> str:
    """Remove one caller-defined terminal marker from a normally stopped turn.

    Interior markers, repeated terminal markers, and truncated generations remain
    subject to structural validation. Structural closers remain content.
    """
    if stop_reason == "stop":
        for marker in terminal_markers:
            if not marker or marker.strip() in _STRUCTURAL_CLOSERS:
                continue
            if re.fullmatch(r"(?:`{3,}|~{3,})", marker.strip()):
                continue
            if text.endswith(marker):
                return text[: -len(marker)]
    return text


class ClosureFormat(StrEnum):
    """Supported wire formats; tool and reasoning use Hermes and Snowball markers."""

    JSON = "json"
    TOOL = "tool"
    REASONING = "reasoning"
    CODE_FENCE = "code_fence"


class ContinuationOutcome(StrEnum):
    REFERENCE_COMPLETED = "reference_completed"
    REFERENCE_PREFIX_CLOSED_EARLY = "reference_prefix_closed_early"
    OTHER_CONTENT_CHANGE = "other_content_change"
    UNCLOSED_OR_INVALID = "unclosed_or_invalid"


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


def _decoded_json(text: str, structure_format: ClosureFormat) -> JsonValue:
    if structure_format == ClosureFormat.TOOL:
        text = text[len(TOOL_OPEN) :].rstrip(JSON_WHITESPACE)[: -len(TOOL_CLOSE)]
    value = json.loads(text)
    json.dumps(value, allow_nan=False)
    return value


def score_closure(text: str, expected: str, structure_format: ClosureFormat) -> ClosureScore:
    """Check syntax and known content; never infer what unfinished content meant.

    JSON permits key-order and whitespace changes. Numeric spellings are decoded
    before comparison; booleans and integer versus float values stay distinct. Opaque
    reasoning and fence content must match except for trailing whitespace.
    """
    complete = closure_state(text, structure_format).status == PrefixStatus.COMPLETE
    if not complete:
        return ClosureScore(False, False, False)
    if structure_format in (ClosureFormat.JSON, ClosureFormat.TOOL):
        reference = _decoded_json(expected, structure_format)
        try:
            candidate = _decoded_json(text, structure_format)
        except ValueError:
            # A syntactically valid exponent can overflow the JSON decoder's
            # float representation. It cannot preserve a finite reference value.
            return ClosureScore(True, False, False)
        preserved = json_values_equal(reference, candidate, numeric_types=NumericTypePolicy.STRICT)
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


def continuation_outcome(text: str, expected: str, structure_format: ClosureFormat) -> ContinuationOutcome:
    """Identify closing-only truncation of a known reference, without inferring intent.

    Early closure requires an exact strict reference prefix followed only by a verified
    syntactic repair. Other content changes include ambiguous truncations with
    spelling or whitespace changes; this is a conservative diagnostic, not a
    complete detector of semantic premature closure. Trailing output is ignored.
    """
    state = closure_state(text, structure_format)
    if state.completion_offset is None:
        return ContinuationOutcome.UNCLOSED_OR_INVALID
    region = text[: state.completion_offset]
    score = score_closure(region, expected, structure_format)
    if score.content_preserved:
        return ContinuationOutcome.REFERENCE_COMPLETED
    boundary = 0
    for actual, reference in zip(region, expected, strict=False):
        if actual != reference:
            break
        boundary += 1
    shared = region[:boundary]
    suffix = closure_state(shared, structure_format).closing_suffix
    if suffix is not None and shared + suffix == region:
        return ContinuationOutcome.REFERENCE_PREFIX_CLOSED_EARLY
    return ContinuationOutcome.OTHER_CONTENT_CHANGE
