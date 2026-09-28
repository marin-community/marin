# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""BPB scoring primitives: byte denominator, aggregation weighting, and the
OLMo-Eval tokenization boundary (join-and-slice, BOS, trailing-space move).

Guards these silent-mismatch classes: wrong byte denominator (token count vs
UTF-8 bytes), example- vs byte-weighted aggregation, prompt tokens included in
the scored span, and BOS/EOS drift.
"""

from __future__ import annotations

import math

import pytest
from marin.evaluation.olmo_base_eval.bpb import (
    LN2,
    bits_per_byte,
    continuation_num_bytes,
    encode_context_continuation,
    task_bpb,
)


class FakeMergeTokenizer:
    """Greedy longest-match tokenizer with explicit multi-char merges.

    A merge that straddles the context/continuation boundary makes
    ``encode(a + b) != encode(a) + encode(b)``, which is exactly the case the
    join-and-slice boundary rule must handle. Char/byte values are stable ids.
    """

    def __init__(self, merges: tuple[str, ...] = ("XY",)):
        self.merges = tuple(sorted(merges, key=len, reverse=True))
        self._vocab: dict[str, int] = {}

    def _token_id(self, token: str) -> int:
        return self._vocab.setdefault(token, 1000 + len(self._vocab))

    def encode(self, text: str) -> list[int]:
        ids: list[int] = []
        i = 0
        while i < len(text):
            for merge in self.merges:
                if text.startswith(merge, i):
                    ids.append(self._token_id(merge))
                    i += len(merge)
                    break
            else:
                ids.append(self._token_id(text[i]))
                i += 1
        return ids


# --- byte denominator -------------------------------------------------------


@pytest.mark.parametrize(
    "continuation, expected_bytes",
    [
        (" Paris", 6),  # leading space counts
        ("Paris", 5),
        ("café", 5),  # 'é' is 2 UTF-8 bytes -> 5 bytes for 4 chars
        ("🙂", 4),  # emoji is 4 UTF-8 bytes
        ("naïve résumé", 15),  # 12 chars, three 2-byte codepoints (ï, é, é) -> 12 + 3 bytes
    ],
)
def test_continuation_num_bytes_counts_utf8_bytes_not_chars(continuation, expected_bytes):
    assert continuation_num_bytes(continuation) == expected_bytes
    # And it is bytes, not characters, whenever there is a multibyte codepoint.
    if any(ord(ch) > 127 for ch in continuation):
        assert continuation_num_bytes(continuation) > len(continuation)


def test_bits_per_byte_matches_reference_formula():
    sum_logprob = -12.3456
    num_bytes = 7
    assert bits_per_byte(sum_logprob, num_bytes) == pytest.approx(-sum_logprob / (num_bytes * math.log(2)))


def test_bits_per_byte_zero_bytes_returns_zero():
    # Matches OLMo-Eval's guard; an empty continuation contributes 0.0, not inf/nan.
    assert bits_per_byte(-1.0, 0) == 0.0


# --- aggregation weighting --------------------------------------------------


def test_task_bpb_is_unweighted_instance_mean_not_byte_weighted():
    # Two instances with equal summed logprob but very different byte lengths.
    logprobs = [-10.0, -10.0]
    num_bytes = [5, 20]
    unweighted = (bits_per_byte(-10.0, 5) + bits_per_byte(-10.0, 20)) / 2
    byte_weighted = -(10.0 + 10.0) / ((5 + 20) * LN2)
    assert task_bpb(logprobs, num_bytes) == pytest.approx(unweighted)
    # The two aggregations genuinely differ on unequal byte lengths, so this test
    # would fail if the implementation switched to byte-weighting.
    assert unweighted != pytest.approx(byte_weighted)


def test_task_bpb_requires_matching_lengths():
    with pytest.raises(ValueError):
        task_bpb([-1.0, -2.0], [3])


# --- tokenization boundary (join-and-slice, BOS, trailing-space) ------------


def test_continuation_uses_join_and_slice_not_separate_encode():
    tok = FakeMergeTokenizer(merges=("XY",))
    # context ends in 'X', continuation starts with 'Y' -> 'XY' merges only in the join.
    enc = encode_context_continuation(tok.encode, "aX", "Yb", bos_token_id=None)
    # join: encode("aXYb") = [a, XY, b]; context encode("aX") = [a, X]; slice -> [b].
    assert enc.num_continuation_tokens == 1
    assert enc.tokens[enc.prompt_length :] == (tok._token_id("b"),)
    # separate-encode would have produced [Y, b] (2 tokens) — the contrast is the point.
    assert len(tok.encode("Yb")) == 2


def test_prompt_length_separates_context_from_continuation():
    tok = FakeMergeTokenizer()
    enc = encode_context_continuation(tok.encode, "hello", " world", bos_token_id=None)
    context_ids = tuple(tok.encode("hello"))
    assert enc.tokens[: enc.prompt_length] == context_ids
    # No prompt token leaks into the scored continuation span.
    assert enc.prompt_length == len(context_ids)
    assert enc.tokens[enc.prompt_length :] == tuple(tok.encode("hello world"))[len(context_ids) :]


def test_bos_prepended_only_when_bos_token_id_given():
    tok = FakeMergeTokenizer()
    without = encode_context_continuation(tok.encode, "hi", " there", bos_token_id=None)
    with_bos = encode_context_continuation(tok.encode, "hi", " there", bos_token_id=128000)
    assert with_bos.tokens[0] == 128000
    assert with_bos.prompt_length == without.prompt_length + 1
    assert with_bos.tokens[1:] == without.tokens  # BOS only at the front of the context
    # BOS changes neither the continuation tokens nor the byte denominator.
    assert with_bos.num_continuation_tokens == without.num_continuation_tokens
    assert with_bos.num_bytes == without.num_bytes


def test_no_eos_is_appended_to_the_continuation():
    tok = FakeMergeTokenizer()
    enc = encode_context_continuation(tok.encode, "hi", " there", bos_token_id=128000)
    # Last token is the final continuation char, not an appended EOS/sentinel.
    assert enc.tokens[-1] == tok._token_id("e")
    assert enc.tokens[-1] != tok._token_id("hi")


def test_trailing_space_moves_boundary_but_byte_denominator_uses_original():
    tok = FakeMergeTokenizer()
    # context has a trailing space; continuation is "cd". The space moves into the
    # tokenized continuation, but num_bytes is len("cd") == 2, not len(" cd").
    enc = encode_context_continuation(tok.encode, "ab ", "cd", bos_token_id=None)
    assert enc.num_bytes == 2
    # context tokenized as "ab" (space stripped) -> prompt_length 2; continuation
    # tokens are the space + c + d.
    assert enc.prompt_length == 2
    assert enc.tokens[enc.prompt_length :] == tuple(tok.encode(" cd"))
