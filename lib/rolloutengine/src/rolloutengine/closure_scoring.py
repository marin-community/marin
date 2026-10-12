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


class ClosureFormat(StrEnum):
    JSON = "json"
    TOOL = "tool"
    REASONING = "reasoning"
    CODE_FENCE = "code_fence"


@dataclass(frozen=True)
class ClosureScore:
    syntax_complete: bool
    content_preserved: bool
    success: bool


def closure_state(text: str, structure_format: ClosureFormat) -> ClosureState:
    if structure_format == ClosureFormat.JSON:
        return json_closure_state(text)
    if structure_format == ClosureFormat.TOOL:
        return delimited_json_closure_state(text, opener="<tool_call>", closer="</tool_call>")
    if structure_format == ClosureFormat.REASONING:
        return paired_marker_closure_state(text, opener="<|start_think|>", closer="<|end_think|>")
    if structure_format == ClosureFormat.CODE_FENCE:
        return required_code_fence_state(text)
    raise ValueError(f"Unsupported closure format: {structure_format}")


def _json_content(text: str, structure_format: ClosureFormat) -> str:
    if structure_format == ClosureFormat.TOOL:
        text = text[len("<tool_call>") :].rstrip(" \t\r\n")[: -len("</tool_call>")]
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
        preserved = text.rstrip(" \t\r\n") == expected.rstrip(" \t\r\n")
    return ClosureScore(True, preserved, preserved)
