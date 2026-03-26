# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for levanter.utils.hf_export — generation config validation and normalization."""

import pytest

from levanter.compat.hf_checkpoints import build_generation_config


class _FakeTokenizer:
    """Minimal tokenizer stub for testing build_generation_config."""

    def __init__(self, vocab_size: int = 200, eos_token_id: int | None = 2, bos_token_id: int | None = 1):
        self._vocab_size = vocab_size
        self.eos_token_id = eos_token_id
        self.bos_token_id = bos_token_id

    def __len__(self):
        return self._vocab_size

    def convert_ids_to_tokens(self, tid: int) -> str | None:
        if 0 <= tid < self._vocab_size:
            return f"<tok_{tid}>"
        return None


class TestBuildGenerationConfig:
    def test_none_returns_none(self):
        tok = _FakeTokenizer()
        assert build_generation_config(tok, None) is None

    def test_valid_ids(self):
        tok = _FakeTokenizer(vocab_size=200, eos_token_id=2, bos_token_id=1)
        result = build_generation_config(tok, [50])
        assert result is not None
        assert result["eos_token_id"] == [2, 50]
        assert result["bos_token_id"] == 1

    def test_deduplication(self):
        tok = _FakeTokenizer(vocab_size=200, eos_token_id=2)
        result = build_generation_config(tok, [50, 50, 2])
        assert result is not None
        assert result["eos_token_id"] == [2, 50]

    def test_sorted_output(self):
        tok = _FakeTokenizer(vocab_size=200, eos_token_id=2)
        result = build_generation_config(tok, [100, 50, 75])
        assert result is not None
        assert result["eos_token_id"] == [2, 50, 75, 100]

    def test_deterministic(self):
        tok = _FakeTokenizer(vocab_size=200, eos_token_id=2)
        r1 = build_generation_config(tok, [128, 64, 128])
        r2 = build_generation_config(tok, [64, 128, 64])
        assert r1 == r2

    def test_auto_adds_tokenizer_eos(self):
        tok = _FakeTokenizer(vocab_size=200, eos_token_id=2)
        result = build_generation_config(tok, [50])
        assert result is not None
        assert 2 in result["eos_token_id"]

    def test_eos_already_included(self):
        tok = _FakeTokenizer(vocab_size=200, eos_token_id=2)
        result = build_generation_config(tok, [2, 50])
        assert result is not None
        assert result["eos_token_id"] == [2, 50]

    def test_tokenizer_eos_none(self):
        tok = _FakeTokenizer(vocab_size=200, eos_token_id=None)
        result = build_generation_config(tok, [50])
        assert result is not None
        assert result["eos_token_id"] == [50]

    def test_bos_included_when_present(self):
        tok = _FakeTokenizer(vocab_size=200, eos_token_id=2, bos_token_id=1)
        result = build_generation_config(tok, [50])
        assert result is not None
        assert result["bos_token_id"] == 1

    def test_bos_omitted_when_none(self):
        tok = _FakeTokenizer(vocab_size=200, eos_token_id=2, bos_token_id=None)
        result = build_generation_config(tok, [50])
        assert result is not None
        assert "bos_token_id" not in result

    def test_empty_list_raises(self):
        tok = _FakeTokenizer()
        with pytest.raises(ValueError, match="non-empty"):
            build_generation_config(tok, [])

    def test_non_int_raises(self):
        tok = _FakeTokenizer()
        with pytest.raises(ValueError, match="non-int"):
            build_generation_config(tok, [1, "two"])  # type: ignore[list-item]

    def test_out_of_range_raises(self):
        tok = _FakeTokenizer(vocab_size=100)
        with pytest.raises(ValueError, match="out of range"):
            build_generation_config(tok, [999])

    def test_negative_id_raises(self):
        tok = _FakeTokenizer(vocab_size=100)
        with pytest.raises(ValueError, match="out of range"):
            build_generation_config(tok, [-1])
