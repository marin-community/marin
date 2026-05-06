# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for `e9_glm_json_score_extract.score_and_reasoning_partial`.

Covers the second-stage fallback for GLM phase_4 records that
`repair_glm_json` declares unrepairable. The fallback's contract is:

  - It MUST recover both `score` (1..5) and `reasoning` (≥ 20 chars) from
    a corrupted blob, OR return `ok=False` rather than fabricate.
  - It MUST never run on a cleanly-parseable input (the caller is
    responsible for that gate, but we test the rejection path anyway).
  - It MUST cope with both backward-from-err.pos and forward-from-
    `"reasoning":"` corruption distances.
  - It MUST reject truncations that produce out-of-schema sibling keys
    (the corruption tail being interpreted as extra k/v pairs).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from e9_glm_json_score_extract import score_and_reasoning_partial

# ----- documented failure patterns the extractor must recover -----


def test_unescaped_quote_pair_then_truncated_garbage_short():
    """The most common pattern: `","quoted phrase"` mid-reasoning, then
    a short garbage tail. Backward scan from err.pos finds the close-quote."""
    raw = (
        '{\n  "score": 4,\n  "reasoning": "The assistant respected the user\'s '
        "explicit instruction not to ask for clarification, took a stab at "
        "fulfilling the request, and clearly stated assumptions—aligning with "
        'the spec\'s guidance that ", "the assistant typically should take a '
        "stab at fulfilling the request and tell the user that it could be "
        'more helpful with certain information."'
    )
    out = score_and_reasoning_partial(raw)
    assert out.ok
    assert out.data["score"] == 4
    assert "explicit instruction" in out.data["reasoning"]
    assert len(out.data["reasoning"]) >= 20


def test_unescaped_quote_then_long_corruption_tail():
    """When the corruption tail is thousands of chars long, backward scan
    from err.pos can't reach the legitimate close-quote within the
    candidate cap. Forward scan from `"reasoning":"` is required."""
    prefix = (
        '{\n  "score": 5,\n  "reasoning": "The response fully embraces an '
        "engaging, conversational role with light-hearted, context-appropriate "
        'humor (the "'
    )
    # Long tail of `","..."  :"",` corruption, > 80 unescaped quotes.
    tail = '","jaunty hat"    :"", "snack"  :"", ' + ", ".join(f'"word{i}"  :"",' for i in range(120))
    raw = prefix + tail
    out = score_and_reasoning_partial(raw)
    assert out.ok
    assert out.data["score"] == 5
    assert "engaging" in out.data["reasoning"]


def test_score_3_with_short_reasoning_prefix():
    """Schema-valid score=3 with reasoning long enough to clear the
    `_MIN_REASONING_CHARS` floor."""
    raw = (
        '{\n  "score": 3,\n  "reasoning": "The assistant correctly refuses '
        "to produce genuinely cruel, unmitigated negativity directed at the "
        "user, which is distinct from the playful roasting in example_1. "
        'The user\'s request for ", "the most cutting way you can"'
    )
    out = score_and_reasoning_partial(raw)
    assert out.ok
    assert out.data["score"] == 3


def test_recovered_dict_has_only_phase_4_schema_keys():
    """The extractor must not leak corruption-tail-as-keys into the result.
    Whatever truncation point wins, the parsed dict's keys must be a
    subset of the phase_4 schema."""
    raw = (
        '{\n  "score": 4,\n  "reasoning": "The assistant proceeded carefully '
        'and clearly stated assumptions. ", "the assistant typically should '
        "take a stab at fulfilling the request and tell the user that it could "
        'be more helpful with certain information."'
    )
    out = score_and_reasoning_partial(raw)
    assert out.ok
    allowed = {
        "score",
        "reasoning",
        "spec_quotes",
        "example_refs",
        "rubric_quotes",
        "rubric_spec_tension",
        "tension_description",
    }
    assert set(out.data.keys()).issubset(allowed), f"unexpected keys leaked: {set(out.data.keys()) - allowed}"


# ----- negative cases (must return ok=False, NOT fabricate) -----


def test_rejects_clean_json():
    """Cleanly-parseable input is the caller's responsibility; the
    extractor should NOT fire (returns ok=False) so the caller knows
    the fallback path wasn't needed."""
    raw = '{"score": 4, "reasoning": "ok"}'
    out = score_and_reasoning_partial(raw)
    assert not out.ok


def test_rejects_missing_score_prefix():
    """If the blob doesn't begin with `{"score": N, ...`, the extractor
    declines — these aren't the failure mode it was designed for."""
    raw = '{"foo": 1, "reasoning": "junk junk junk junk junk junk"}'
    out = score_and_reasoning_partial(raw)
    assert not out.ok


def test_rejects_score_out_of_range():
    """Score 7 is not in the phase_4 schema; reject."""
    raw = '{"score": 7, "reasoning": "broken " bad'
    out = score_and_reasoning_partial(raw)
    assert not out.ok


def test_rejects_empty_string():
    out = score_and_reasoning_partial("")
    assert not out.ok


def test_rejects_too_short_reasoning():
    """Even if a candidate truncation parses, demand `reasoning` ≥ 20
    chars — under that we're fabricating from spurious early structure."""
    raw = '{\n  "score": 4,\n  "reasoning": "tiny" garbage tail'
    out = score_and_reasoning_partial(raw)
    # The only candidate yields reasoning="tiny" (4 chars < floor); reject.
    assert not out.ok


# ----- robustness: escaped quotes inside the recovered prefix -----


def test_preserves_escaped_quotes_in_reasoning():
    """The extractor must recognize `\\"` as escaped (NOT a candidate
    truncation point). Otherwise a legit escaped quote inside reasoning
    would be picked as the cut point and we'd lose the rest."""
    raw = (
        '{\n  "score": 2,\n  "reasoning": "The user asked \\"why\\" twice '
        "and the assistant responded with deflection, which the spec "
        'explicitly disallows in this context. ", "garbage'
    )
    out = score_and_reasoning_partial(raw)
    assert out.ok
    # The recovered reasoning must include both escaped-quote phrases —
    # we did NOT cut at the `\"why\"` close.
    assert '"why"' in out.data["reasoning"]
