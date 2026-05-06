# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""GLM-5.1 JSON repair pass for the phase_4 / rubric_plus_spec judging schema.

WHY THIS EXISTS
---------------
Of 2,758 phase_4 rubric_plus_spec calls dispatched to GLM-5.1 (Together-hosted),
~315 (≈11%) returned content that fails `json.loads`, even though the request
sets `response_format={"type": "json_object"}`. The GPT and Gemini judges
returning the same schema do not exhibit this rate.

The sub-agent that summarized GLM phase_4 (`claude_subagents/lm_judge_rubric_plus_spec/glm.md`)
documented 79 specific failures across the high-disagreement-tuple slice (690 rows).
Their distribution:

| pattern                                              | count | typical cause                  |
|------------------------------------------------------|------:|--------------------------------|
| `Expecting ':' delimiter` at line 3, col N           |    27 | unescaped quote in str value   |
| `Expecting ',' delimiter` at line 3, col N           |    22 | unescaped quote / missing sep  |
| `Expecting value: line 1 column 1`                   |    15 | empty body (length-truncated)  |
| `Unterminated string starting at` (line 3 or 4)      |    12 | hit max_tokens mid-string      |
| `Expecting property name enclosed in double quotes`  |     3 | smart-quote on a key           |

The "line 3 column N" pattern is consistent with GLM's pretty-printed output
shape (top-level `{` on line 1, `\n` after first sibling, then a long line 3
containing the bulk of the JSON content).

DESIGN
------
`repair_glm_json(raw_text)` tries `json.loads` first; if it succeeds it
returns the parsed dict and a no-op strategy tag, leaving valid output
untouched (the load-bearing requirement of this pass — must not change
behavior on any other judge or condition).

On `JSONDecodeError`, it attempts each strategy below in order. Each
strategy returns the parsed dict on success or `None` on failure; the
first hit wins. Strategies are conservative — when an attempted fix
would produce an ambiguous parse, the strategy returns `None` rather
than fabricate a score.

Strategies (in order):
    1. `smart_quote_keys`: replaces left/right curly double quotes with
       straight `"` ONLY when adjacent to JSON structural tokens
       (`{`, `,`, whitespace before a key). Targets the rare smart-quote-
       on-key error.
    2. `escape_unescaped_quote_at_error`: inspects the `pos` of the
       parse error and escapes a stray `"` inside a string value.
    3. `truncated_close`: if the error suggests we ran out of tokens
       mid-structure (Unterminated string, very-late `:` or `,`, or
       error pos > 80% of input length), close any open string,
       array, and object brackets and re-parse. Marks result with
       `_partial_parse: true`.
    4. `empty_body`: if the input is empty / whitespace, return None
       (truly unrecoverable; nothing to repair).

If every strategy fails, the function returns `None` and the caller
should treat the record as a parse failure (matching today's behavior).

IMPORTANT
---------
The repair preserves the original `response_format=json_object` contract;
it does NOT loosen the schema. Downstream callers still see a `dict` with
keys matching `e8_phase2_cross_model.JUDGE_*` schemas.

The repair is OPT-IN. Existing parsing in `e8_phase2_cross_model.py:call_glm_json`
and `e8_paired_indirection.py:parse_json` is unchanged. To use this repair,
call `repair_glm_json(raw_text)` from a new code path, or set the env var
`MARIN_E9_GLM_REPAIR=1` before importing `parse_json_with_glm_repair`.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any

# -------------------- Result types --------------------


@dataclass(frozen=True)
class RepairResult:
    """Outcome of `repair_glm_json`. `data` is None when repair failed."""

    data: dict[str, Any] | None
    strategy: str  # one of: "valid", "smart_quote_keys",
    # "escape_unescaped_quote_at_error", "truncated_close", "empty_body",
    # or "unrepairable"
    partial: bool = False  # True when truncated_close fired


# -------------------- Strategy implementations --------------------


_SMART_QUOTES = "“”„″"  # left, right, low, double-prime


def _try_loads(text: str) -> dict[str, Any] | None:
    """Try json.loads, returning the dict on success and None on JSONDecodeError.

    Also returns None if the parsed object is not a dict — phase_4 schema is
    always an object.
    """
    try:
        obj = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return None
    if isinstance(obj, dict):
        return obj
    return None


def _smart_quote_keys(text: str) -> dict[str, Any] | None:
    """Replace curly double-quotes that appear in key positions with straight quotes.

    Matches the rare ``Expecting property name enclosed in double quotes``
    pattern (3 of 79 documented failures). Only rewrites quotes adjacent to
    JSON structural punctuation (``{`` or ``,``), so a curly quote inside a
    string VALUE (e.g., GLM quoting the user's typographic input) is left
    alone. Also normalises curly quotes that immediately precede a colon
    (the closing quote of a key).
    """
    if not any(q in text for q in _SMART_QUOTES):
        return None
    # Replace opening curly quote in key position: after `{` or `,` and
    # optional whitespace.
    fixed = re.sub(r"([{,]\s*)[“”„″]", r'\1"', text)
    # Replace closing curly quote on a key: immediately before whitespace
    # and a colon.
    fixed = re.sub(r"[“”„″](\s*:)", r'"\1', fixed)
    if fixed == text:
        return None
    return _try_loads(fixed)


_MAX_QUOTE_ESCAPE_PASSES = 8


def _is_unescaped_quote(text: str, i: int) -> bool:
    """Return True if `text[i] == '"'` and the preceding backslashes count is even."""
    if i < 0 or i >= len(text) or text[i] != '"':
        return False
    j = i - 1
    backslashes = 0
    while j >= 0 and text[j] == "\\":
        backslashes += 1
        j -= 1
    return backslashes % 2 == 0


def _escape_unescaped_quote_at_error(text: str) -> dict[str, Any] | None:
    """When a parse error reports a delimiter error mid-string, walk back from
    `err.pos` to find the unescaped ``"`` that prematurely closed the string,
    escape it, and retry. Supports multiple unescaped quotes in the same
    string value by iterating up to `_MAX_QUOTE_ESCAPE_PASSES` times.

    Targets ``Expecting ':' delimiter`` (27 cases) and ``Expecting ',' delimiter``
    (22 cases) at line 3 column N. The diagnostic in those cases is that
    GLM emitted a string value containing a ``"`` it failed to escape, so
    the parser saw `"...broken"` then expected the next colon/comma at a
    position deep inside what was supposed to be the same string.

    Real-world inputs often have a *pair* of unescaped quotes (an open and
    a close, e.g. `the user said "ok" then`). One escape isn't enough; we
    must escape both. The iterative loop handles that: each pass escapes
    one quote, then re-runs `json.loads` to see if more remain.

    The scan is bounded:
      - up to 5 candidate quote positions tried at each pass;
      - up to `_MAX_QUOTE_ESCAPE_PASSES` passes total.

    Returns None if no candidate-set yields a clean parse within the
    bound, or if the error message indicates a non-delimiter problem
    (which means a different strategy should run).
    """
    current = text
    for _pass in range(_MAX_QUOTE_ESCAPE_PASSES):
        try:
            obj = json.loads(current)
        except json.JSONDecodeError as err:
            pos = err.pos
            msg = err.msg
        else:
            return obj if isinstance(obj, dict) else None

        # Only fire on delimiter-mid-string errors. Other shapes belong to
        # other strategies.
        if not (
            "Expecting ',' delimiter" in msg
            or "Expecting ':' delimiter" in msg
            or "Expecting value" in msg  # also fires when the string content
            # starts with non-JSON tokens after the spurious close
        ):
            return None

        if pos <= 1 or pos > len(current):
            return None

        # Walk left from pos to find candidate quote positions.
        candidates: list[int] = []
        i = pos
        while i > 0 and len(candidates) < 5:
            i -= 1
            if _is_unescaped_quote(current, i):
                candidates.append(i)

        if not candidates:
            return None

        repaired_any = False
        for cand in candidates:
            patched = current[:cand] + '\\"' + current[cand + 1 :]
            # Quick check: did the parse position advance? If yes, accept
            # this patch and continue the loop to handle remaining issues.
            try:
                json.loads(patched)
            except json.JSONDecodeError as err2:
                if err2.pos > pos:
                    current = patched
                    repaired_any = True
                    break
            else:
                # Full parse succeeded. Done.
                obj = json.loads(patched)
                return obj if isinstance(obj, dict) else None
        if not repaired_any:
            return None

    return None


def _truncated_close(text: str) -> dict[str, Any] | None:
    """If the input was truncated mid-output, close open string/array/object
    brackets and try to parse the partial result.

    Targets ``Unterminated string starting at`` (12 cases) and the long-tail
    ``Expecting ':'``/``Expecting ','`` cases where the error column is far
    into the document (e.g., col 10217 / 15759 — well past where any
    legitimate phase_4 JSON would still be parsing).

    Strategy:
      1. If we are inside an unterminated string (last `"` opens, no
         matching `"` yet), append `"` first.
      2. Walk the text counting unmatched `[`, `]`, `{`, `}` while
         respecting strings, and append the closers in reverse stack order.
      3. Try to parse. On success, return the dict (caller adds `_partial_parse`).
         On failure, return None.

    Conservative: only fires when the structure is plausibly recoverable
    (i.e., we have an open root object). Refuses to repair if no `{` was
    ever seen.
    """
    if not text or "{" not in text:
        return None

    in_string = False
    escape = False
    stack: list[str] = []
    for ch in text:
        if escape:
            escape = False
            continue
        if ch == "\\" and in_string:
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            stack.append("}")
        elif ch == "[":
            stack.append("]")
        elif ch == "}" or ch == "]":
            if stack and stack[-1] == ch:
                stack.pop()

    if not stack and not in_string:
        return None  # nothing to close, but original failed — not our case

    repaired = text
    # Step 1: close any open string. The truncation often left the string
    # missing its closing `"`; appending one is the minimal fix.
    if in_string:
        repaired = repaired + '"'
    # Step 2: also strip any trailing comma we'd be emitting before closers.
    repaired = re.sub(r",\s*$", "", repaired)
    # Step 3: append unclosed bracket / brace closers in reverse stack order.
    repaired = repaired + "".join(reversed(stack))

    return _try_loads(repaired)


# -------------------- Public API --------------------


def repair_glm_json(raw_text: str) -> RepairResult:
    """Parse `raw_text` as JSON, applying GLM-specific repairs on failure.

    Order of strategies:
      0. Try `json.loads` directly. If it succeeds and the result is a
         dict, return immediately with strategy="valid".
      1. Empty / whitespace-only body → unrepairable (strategy="empty_body",
         data=None).
      2. Smart-quote-on-key fix.
      3. Unescaped-quote-mid-value fix.
      4. Truncated-structure close.
      5. Otherwise return strategy="unrepairable", data=None.

    Strategy 4 sets `partial=True` to flag that the returned dict may have
    fewer fields than the schema requires; callers can choose to gate or
    not on this.
    """
    if raw_text is None:
        return RepairResult(data=None, strategy="empty_body")
    text = raw_text.strip()
    if not text:
        return RepairResult(data=None, strategy="empty_body")

    direct = _try_loads(text)
    if direct is not None:
        return RepairResult(data=direct, strategy="valid")

    # Strategy 1: smart-quoted key.
    candidate = _smart_quote_keys(text)
    if candidate is not None:
        return RepairResult(data=candidate, strategy="smart_quote_keys")

    # Strategy 2: unescaped quote inside a string value.
    candidate = _escape_unescaped_quote_at_error(text)
    if candidate is not None:
        return RepairResult(data=candidate, strategy="escape_unescaped_quote_at_error")

    # Strategy 3: truncated close.
    candidate = _truncated_close(text)
    if candidate is not None:
        return RepairResult(data=candidate, strategy="truncated_close", partial=True)

    return RepairResult(data=None, strategy="unrepairable")


# -------------------- Opt-in wrapper for parse_json --------------------

_REPAIR_FLAG_ENV = "MARIN_E9_GLM_REPAIR"


def parse_json_with_glm_repair(raw_text: str, *, enabled: bool | None = None) -> dict[str, Any]:
    """Parse JSON using `parse_json` semantics (strips ``` fences) but with
    optional GLM repair.

    `enabled=None` reads the env var ``MARIN_E9_GLM_REPAIR`` (truthy means
    repair is on). Pass `enabled=True/False` to force.

    On repaired success, mutates the returned dict to include the metadata
    keys ``_repair_strategy`` and (if applicable) ``_partial_parse``, so a
    downstream consumer can audit which records went through repair without
    needing to re-derive it from the raw transcript.
    """
    if enabled is None:
        enabled = bool(os.environ.get(_REPAIR_FLAG_ENV))

    cleaned = _strip_code_fences(raw_text)

    if not enabled:
        return json.loads(cleaned)

    result = repair_glm_json(cleaned)
    if result.data is None:
        # Replicate the original failure mode so callers behave exactly as
        # before when repair is on but every strategy failed.
        json.loads(cleaned)
        # `json.loads` of unparseable text raises before we get here; this
        # branch only runs if cleaned was empty (returns dict() — but that
        # would also have caused `_try_loads` to return None earlier).
        # Defensive raise to be explicit.
        raise json.JSONDecodeError("repair_glm_json: unrepairable", cleaned, 0)
    out = dict(result.data)
    if result.strategy != "valid":
        out["_repair_strategy"] = result.strategy
    if result.partial:
        out["_partial_parse"] = True
    return out


def _strip_code_fences(raw: str) -> str:
    cleaned = (raw or "").strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
    return cleaned.strip()
