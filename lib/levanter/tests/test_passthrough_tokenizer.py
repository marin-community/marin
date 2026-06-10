# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for `PassthroughTokenizer`.

In particular, `encode` must accept non-integer probe strings (e.g. ".") so
that callers like `levanter.utils.hf_utils.byte_length_of_token` — and by
extension `_calculate_bytes_per_token_type` / `cb_tagged_lm_evaluate` — can
initialize against pre-tokenized integer corpora without crashing.
"""

import pytest

from levanter.data.passthrough_tokenizer import PassthroughTokenizer
from levanter.utils.hf_utils import byte_length_of_token


@pytest.fixture
def tokenizer() -> PassthroughTokenizer:
    return PassthroughTokenizer(_vocab_size=256)


def test_encode_integer_string_roundtrip(tokenizer):
    assert tokenizer.encode("1 2 3") == [1, 2, 3]


def test_encode_empty_string(tokenizer):
    assert tokenizer.encode("") == []
    assert tokenizer.encode("   ") == []


def test_encode_non_integer_returns_placeholder(tokenizer):
    # Non-integer probe strings used by Levanter's bytes-per-token estimator
    # must not crash; they fall back to vocab id 0.
    assert tokenizer.encode(".") == [0]
    assert tokenizer.encode("hello") == [0]


def test_encode_mixed_falls_back_per_token(tokenizer):
    assert tokenizer.encode("1 . 2") == [1, 0, 2]


def test_byte_length_of_token_does_not_crash(tokenizer):
    # The probe inside `byte_length_of_token` calls `encode(".")[0]`, which
    # used to raise `ValueError: invalid literal for int(): '.'`. After the
    # fix, the helper returns a finite byte length for every vocab id.
    for idx in (0, 1, 42, tokenizer.vocab_size - 1):
        length = byte_length_of_token(tokenizer, idx)
        assert isinstance(length, int)
        assert length >= 0
