# Copyright 2026 Baixin Guo (Max)
# SPDX-License-Identifier: Apache-2.0
"""Tests for the trace -> preference screening analysis.  Experiment-specific, so they
live beside the experiment, not in the root ``tests/**``.  The real-data test skips
unless the user supplies the pinned sample.
"""

import os
from pathlib import Path

import pytest

import parse

ACTION = "<function=bash>\n<parameter=command>\nls -la\n</parameter>\n</function>"
TOOLS = [{"type": "function",
           "function": {"name": "bash", "parameters": {"properties": {"command": {}}}}}]


class FakeTokenizer:
    """Character tokenizer; listed bigrams merge into one id, and ids in ``weird``
    decode to text that re-encodes differently (round-trip break)."""

    def __init__(self, merges=(), weird=frozenset()):
        self.merges, self.weird = list(merges), weird

    def encode(self, text):
        out, i = [], 0
        while i < len(text):
            pair = (text[i], text[i + 1]) if i + 1 < len(text) else None
            if pair in self.merges:
                out.append(1000 + self.merges.index(pair))
                i += 2
            else:
                out.append(ord(text[i]))
                i += 1
        return tuple(out)

    def decode(self, ids):
        return "".join("Z" if i in self.weird
                       else "".join(self.merges[i - 1000]) if i >= 1000 else chr(i)
                       for i in ids)


# --- syntax parser: every status is decided against the literal text --------

CASES = [
    (ACTION, parse.STATUS_VALID),
    (ACTION.replace("</function>", "", 1), parse.STATUS_MISSING_CLOSER),
    (ACTION.replace("</parameter>", "", 1), parse.STATUS_UNCLOSED_PARAMETER),
    (ACTION.replace("<function=bash>", "<function=rm>", 1), parse.STATUS_UNKNOWN_FUNCTION),
    (ACTION.replace("<parameter=command>", "<parameter=nope>", 1), parse.STATUS_UNDECLARED_PARAMETER),
    (ACTION + ACTION, parse.STATUS_REPEATED_OPENER),
    ("plain prose", parse.STATUS_MISSING_OPENER),
]


@pytest.mark.parametrize("text,expected", CASES)
def test_function_block_status(text, expected):
    assert parse.function_block_status(text, TOOLS) == expected


# --- span proof gates ------------------------------------------------------


def test_roundtrip_gate():
    proof = parse.prove_action_span(FakeTokenizer(weird=frozenset({2000})), (2000,))
    assert not proof.roundtrip and not proof.verified
    assert proof.skip_reason == parse.REASON_ROUNDTRIP


def test_no_block_gate():
    t = FakeTokenizer()
    proof = parse.prove_action_span(t, t.encode("just some prose"))
    assert proof.roundtrip and not proof.verified
    assert proof.skip_reason == parse.REASON_NO_BLOCK


def test_multiple_blocks_gate():
    t = FakeTokenizer()
    proof = parse.prove_action_span(t, t.encode(ACTION + ACTION))
    assert not proof.verified
    assert proof.skip_reason == parse.REASON_MULTIPLE_BLOCKS


def test_segmentation_gate_catches_cross_boundary_token():
    """A token spanning either action boundary must fail the span proof."""
    t = FakeTokenizer(merges=[("a", "<"), (">", "b")])
    proof = parse.prove_action_span(t, t.encode("a" + ACTION + "b"))
    assert not proof.verified
    assert proof.skip_reason == parse.REASON_SEGMENTATION


def test_clean_span_is_located_and_bounded():
    t = FakeTokenizer()
    proof = parse.prove_action_span(t, t.encode(ACTION))
    assert proof.verified
    assert proof.span == (0, len(ACTION))
    assert proof.skip_reason is None


# --- real data: skipped unless the user supplies the pinned sample ---------
#
#   export TRACE_PREFS_SAMPLE=.../pf_curriculum_hard.parquet
#   export TRACE_PREFS_TOKENIZER=.../tokenizer.json

_SAMPLE = Path(os.environ.get("TRACE_PREFS_SAMPLE", "examples/real/pf_curriculum_hard.parquet"))
_TOKENIZER = Path(os.environ.get("TRACE_PREFS_TOKENIZER", "examples/real/qwen35_tokenizer.json"))
_NEEDS_SAMPLE = pytest.mark.skipif(
    not (_SAMPLE.exists() and _TOKENIZER.exists()),
    reason="pinned sample / tokenizer not present (set TRACE_PREFS_SAMPLE / "
           "TRACE_PREFS_TOKENIZER)",
)


@_NEEDS_SAMPLE
def test_real_sample_reproduces_the_readme_numbers():
    """The sample is pinned by SHA-256, so the README's counts can be pinned too.

    This checks the README against the data, not the data against a copy of itself.
    """
    pytest.importorskip("pyarrow.parquet")
    pytest.importorskip("tokenizers")
    import analyze_traces as A
    import pyarrow.parquet as pq

    res = A.scan_turns(parse.NativeTokenizer(str(_TOKENIZER)), pq.ParquetFile(str(_SAMPLE)))
    c, classes = res[0], res[1]
    assert (c["rows"], c["turns"], c["schema"]) == (22, 156, 22)
    assert (c["row_mismatch"], c["unmapped"], c["disagree"]) == (0, 0, 0)
    assert c["agree"] == c["turns"]   # native <function=NAME> vs JSON tool-call names
    assert c["misaligned"] == 0      # len(logprobs[t]) == len(ids[t])
    assert dict(classes) == {"clean-parsable-action": 91, "action-exceeds-max-tokens": 32,
                             "text-turn-no-call": 22, "multiple-function-blocks": 8,
                             "tokenizer-roundtrip": 3}
