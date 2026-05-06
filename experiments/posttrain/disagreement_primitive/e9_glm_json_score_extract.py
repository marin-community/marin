# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Score-with-partial-reasoning extractor for GLM phase_4 records that the
core repair pass (`e9_glm_json_repair.repair_glm_json`) cannot recover.

WHY THIS EXISTS
---------------
Running `repair_glm_json` on the 315 unparseable phase_4 GLM records on disk
recovers only 121 (38%) via `truncated_close`. The remaining 153 share a
distinct failure mode that the existing strategies don't catch:

GLM emits an unescaped quote mid-`reasoning` value (commonly when quoting a
spec or rubric phrase like ``the spec says ", "the assistant should..."``)
and then the model's logits collapse into a long structured-but-meaningless
tail of `","`, `": "`, tab/space padding, etc. The original repair walks
left from `err.pos` once and tries to escape a single quote — but the tail
is so corrupted that no quote-escape recovers a clean parse.

For κ-by-condition diagnostics the only strictly-required field is `score`;
for `e8_rationale_grounding.py` the only additional required field is
`reasoning` (line 308-309: `if not reasoning or "error" in j: continue`).
All 153 unrepairable records have a clean `{"score": N, "reasoning": "..."`
prefix on disk — the corruption is always to the right of a syntactically
valid initial value pair.

So the extractor truncates at the first unescaped quote (walking left
from `err.pos`), appends ``}`` to close the object, and accepts the
first parse that yields a dict with both `score` and `reasoning`. This
discards any spec_quotes / rubric_quotes / rubric_spec_tension / etc.
that may have been emitted further right — these are typically also
corrupted in the unrepairable failures, so dropping them is honest.

DESIGN
------
`score_and_reasoning_partial(raw_text)` returns either a
`PartialExtract(data, ok=True)` (with `data` carrying at minimum
`score:int` and `reasoning:str`) or `PartialExtract(data=None, ok=False)`.

The extractor refuses to fabricate. If the prefix doesn't begin with a
recognizable `{"score": N, ...` shape, or if no truncation point yields
both score and reasoning, it returns `ok=False`.

Conservative bounds:
  - Up to 80 candidate truncation positions (working back from err.pos).
  - The recovered `reasoning` must be ≥ 20 characters (under that, the
    extracted prefix didn't cover meaningful judge text — likely a
    fabricated parse from very-early structure).
  - The recovered `score` must be in 1..5 (the phase_4 schema range).

OPT-IN ONLY
-----------
This module is not imported by any production script. It is consumed by
`e9_repair_glm_phase4_v2.py` (a forward-looking offline retry CLI) and
optional notebook / one-off analyses. The original `e9_glm_json_repair.py`
strategies and tests are unchanged.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

# Schema-required minimum fields for a record to count as recovered.
_MIN_REASONING_CHARS = 20
_MAX_TRUNCATION_CANDIDATES = 80

# The phase_4 (rubric_plus_spec) schema (see e8_phase4_rubric_plus_spec.py).
# A truncation that yields a dict containing keys outside this set is a
# spurious parse: the corruption tail was interpreted as extra k/v pairs.
# We reject such candidates and try a different truncation point.
_PHASE_4_KEYS = frozenset(
    {
        "score",
        "reasoning",
        "spec_quotes",
        "example_refs",
        "rubric_quotes",
        "rubric_spec_tension",
        "tension_description",
    }
)

# The raw text must begin with this prefix for the extractor to fire. If
# it doesn't, we're staring at output that's malformed in a way the
# extractor was never designed to handle (e.g., the response_format=
# json_object contract was violated more thoroughly than just an
# unescaped quote).
_SCORE_PREFIX_RE = re.compile(r"^\s*\{\s*\"score\"\s*:\s*([1-5])\b")


@dataclass(frozen=True)
class PartialExtract:
    """Outcome of `score_and_reasoning_partial`. `ok=False` means we kept
    out — caller should fall back to whatever it would have done before."""

    data: dict[str, Any] | None
    ok: bool
    truncated_at: int | None = None  # char position where we cut


def _is_unescaped_quote(text: str, i: int) -> bool:
    """True iff text[i] == '"' and the preceding backslashes are even."""
    if i < 0 or i >= len(text) or text[i] != '"':
        return False
    j = i - 1
    backslashes = 0
    while j >= 0 and text[j] == "\\":
        backslashes += 1
        j -= 1
    return backslashes % 2 == 0


def _initial_error_pos(text: str) -> int | None:
    """Returns json's error position, or None if the text parses cleanly.

    We need this to start the leftward walk somewhere meaningful.
    """
    try:
        json.loads(text)
    except json.JSONDecodeError as e:
        return e.pos
    return None


def score_and_reasoning_partial(raw_text: str) -> PartialExtract:
    """Recover ``{score, reasoning}`` from a corrupted phase_4 GLM blob.

    Algorithm (conservative):
      1. Reject if the text doesn't begin with ``{"score": N`` for N in 1..5.
      2. Find json's parse-error position; walk leftward, collecting up to
         `_MAX_TRUNCATION_CANDIDATES` unescaped-quote positions.
      3. For each candidate position `q`, try parsing
         ``text[:q+1] + "}"`` (treat that quote as the close of the
         current string, then close the root object).
      4. The first candidate whose parse yields a dict with `score`
         (an int 1..5) and `reasoning` (a string of length ≥
         `_MIN_REASONING_CHARS`) wins. The resulting dict is returned.

    If no candidate works, returns `PartialExtract(data=None, ok=False)`.
    """
    if not raw_text:
        return PartialExtract(data=None, ok=False)

    if not _SCORE_PREFIX_RE.match(raw_text):
        return PartialExtract(data=None, ok=False)

    err_pos = _initial_error_pos(raw_text)
    if err_pos is None:
        # Cleanly parseable — the caller shouldn't have invoked us. Stay out.
        return PartialExtract(data=None, ok=False)

    # Build candidate truncation positions in two passes:
    #   Pass 1 (backward from err_pos): catches the common case where
    #     err.pos is near the corruption point and we want the closest
    #     pre-error close-quote.
    #   Pass 2 (forward from end of `"reasoning": "`): catches cases where
    #     the corruption tail is so long that backward-scanning from
    #     err.pos can't reach the first legitimate close-quote within
    #     `_MAX_TRUNCATION_CANDIDATES` steps. The forward scan finds the
    #     earliest unescaped quote inside reasoning, which is typically
    #     the spurious-close that broke the parse.
    candidates: list[int] = []
    seen: set[int] = set()
    i = min(err_pos, len(raw_text) - 1)
    while i > 0 and len(candidates) < _MAX_TRUNCATION_CANDIDATES:
        if _is_unescaped_quote(raw_text, i):
            candidates.append(i)
            seen.add(i)
        i -= 1

    # Forward pass: skip past `"reasoning"\s*:\s*"` and walk until we
    # have collected up to `_MAX_TRUNCATION_CANDIDATES` more positions.
    reasoning_start_re = re.compile(r"\"reasoning\"\s*:\s*\"")
    m = reasoning_start_re.search(raw_text)
    if m is not None:
        i = m.end()
        added = 0
        while i < len(raw_text) and added < _MAX_TRUNCATION_CANDIDATES:
            if _is_unescaped_quote(raw_text, i) and i not in seen:
                candidates.append(i)
                added += 1
            i += 1

    for q in candidates:
        # Truncate AT the candidate quote (inclusive), then close the object.
        candidate_text = raw_text[: q + 1] + "}"
        try:
            obj = json.loads(candidate_text)
        except (json.JSONDecodeError, ValueError):
            continue
        if not isinstance(obj, dict):
            continue
        # Reject truncations that introduce spurious sibling keys (the
        # corruption tail being read as extra k/v pairs).
        if not set(obj.keys()).issubset(_PHASE_4_KEYS):
            continue
        score = obj.get("score")
        reasoning = obj.get("reasoning")
        # Tighten: schema-valid score, non-trivial reasoning prefix.
        if not isinstance(score, int) or score not in (1, 2, 3, 4, 5):
            continue
        if not isinstance(reasoning, str) or len(reasoning) < _MIN_REASONING_CHARS:
            continue
        return PartialExtract(data=obj, ok=True, truncated_at=q)

    return PartialExtract(data=None, ok=False)
