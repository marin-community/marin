# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for `PassthroughTokenizer`.

`encode` stays strict — a non-integer token is a corpus-format error and must
raise, not silently map to a token id. The byte-length probe in
`levanter.utils.hf_utils.byte_length_of_token` (which feeds
`_calculate_bytes_per_token_type` / `cb_tagged_lm_evaluate`) is handled at its
own source so it can run against pre-tokenized integer corpora without
weakening `encode`.
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


@pytest.mark.parametrize("text", [".", "hello", "1 . 2"])
def test_encode_non_integer_raises(tokenizer, text):
    # A non-integer token is a corpus-format error; encode must fail fast
    # rather than mask it as a valid token id.
    with pytest.raises(ValueError):
        tokenizer.encode(text)


@pytest.mark.parametrize("idx, expected", [(0, 1), (5, 1), (42, 2), (255, 3)])
def test_byte_length_of_token_is_digit_count(tokenizer, idx, expected):
    # The "." prefix probe in byte_length_of_token can't be encoded by an
    # integer-only tokenizer; the helper falls back to decoding the token
    # directly, so the byte length is exactly the integer's digit count.
    assert byte_length_of_token(tokenizer, idx) == expected
