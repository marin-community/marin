# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
from transformers import AutoTokenizer

from levanter.data.text import DNABatchTokenizer


def _skip_if_tokenizer_unavailable(tokenizer_name: str):
    def try_load(name):
        try:
            AutoTokenizer.from_pretrained(name)
        except Exception:
            return False
        return True

    return pytest.mark.skipif(not try_load(tokenizer_name), reason=f"Tokenizer {tokenizer_name} not accessible")


NO_BOS_EOS_TOKENIZER = "songlab/tokenizer-dna-clm"
BOS_EOS_TOKENIZER = "bolinas-dna/tokenizer-char"


@_skip_if_tokenizer_unavailable(NO_BOS_EOS_TOKENIZER)
def test_no_special_tokens():
    """With a tokenizer that has no BOS/EOS, output matches input length."""
    tokenizer = AutoTokenizer.from_pretrained(NO_BOS_EOS_TOKENIZER)
    bt = DNABatchTokenizer(tokenizer, soft_mask_weight=0.01)

    assert bt.num_special_tokens == 0

    batch = [{"seq": "ACGTacgt"}, {"seq": "TTTTaaaa"}]
    results = bt(batch)

    assert len(results) == 2
    for r in results:
        assert r["input_ids"].shape == (8,)
        assert r["loss_weight"].shape == (8,)
        assert r["input_ids"].dtype == np.int32
        assert r["loss_weight"].dtype == np.float32


@_skip_if_tokenizer_unavailable(NO_BOS_EOS_TOKENIZER)
def test_soft_mask_weights_no_special_tokens():
    """Uppercase gets weight 1.0, lowercase gets soft_mask_weight."""
    tokenizer = AutoTokenizer.from_pretrained(NO_BOS_EOS_TOKENIZER)
    bt = DNABatchTokenizer(tokenizer, soft_mask_weight=0.1)

    batch = [{"seq": "ACgt"}]
    result = bt(batch)[0]

    np.testing.assert_allclose(result["loss_weight"], [1.0, 1.0, 0.1, 0.1])


@_skip_if_tokenizer_unavailable(BOS_EOS_TOKENIZER)
def test_bos_eos_tokens_added():
    """With a tokenizer that has BOS/EOS, they are prepended/appended."""
    tokenizer = AutoTokenizer.from_pretrained(BOS_EOS_TOKENIZER)
    bt = DNABatchTokenizer(tokenizer, soft_mask_weight=0.01)

    assert bt.num_special_tokens == 2

    seq = "ACGT"
    batch = [{"seq": seq}]
    result = bt(batch)[0]

    assert result["input_ids"].shape == (len(seq) + 2,)
    assert result["loss_weight"].shape == (len(seq) + 2,)

    assert result["input_ids"][0] == tokenizer.bos_token_id
    assert result["input_ids"][-1] == tokenizer.eos_token_id

    # Interior tokens should match tokenizing without special tokens
    plain_ids = tokenizer(seq, add_special_tokens=False)["input_ids"]
    np.testing.assert_array_equal(result["input_ids"][1:-1], plain_ids)


@_skip_if_tokenizer_unavailable(BOS_EOS_TOKENIZER)
def test_loss_weights_with_bos_eos():
    """BOS/EOS positions get weight 1.0; interior follows soft-masking."""
    tokenizer = AutoTokenizer.from_pretrained(BOS_EOS_TOKENIZER)
    bt = DNABatchTokenizer(tokenizer, soft_mask_weight=0.1)

    batch = [{"seq": "ACgt"}]
    result = bt(batch)[0]

    # BOS weight, uppercase A, uppercase C, lowercase g, lowercase t, EOS weight
    expected = np.array([1.0, 1.0, 1.0, 0.1, 0.1, 1.0], dtype=np.float32)
    np.testing.assert_allclose(result["loss_weight"], expected)


@_skip_if_tokenizer_unavailable(BOS_EOS_TOKENIZER)
def test_batch_consistency_with_bos_eos():
    """All sequences in a batch get BOS/EOS and have the same length."""
    tokenizer = AutoTokenizer.from_pretrained(BOS_EOS_TOKENIZER)
    bt = DNABatchTokenizer(tokenizer, soft_mask_weight=0.01)

    batch = [{"seq": "AAAA"}, {"seq": "CCCC"}, {"seq": "TTTT"}]
    results = bt(batch)

    lengths = {r["input_ids"].shape[0] for r in results}
    assert len(lengths) == 1
    assert lengths.pop() == 4 + 2  # seq_len + BOS + EOS


@_skip_if_tokenizer_unavailable(NO_BOS_EOS_TOKENIZER)
def test_metadata_no_special_tokens():
    tokenizer = AutoTokenizer.from_pretrained(NO_BOS_EOS_TOKENIZER)
    bt = DNABatchTokenizer(tokenizer, soft_mask_weight=0.5)
    meta = bt.metadata

    assert meta["has_bos"] is False
    assert meta["has_eos"] is False
    assert meta["soft_mask_weight"] == 0.5


@_skip_if_tokenizer_unavailable(BOS_EOS_TOKENIZER)
def test_metadata_with_special_tokens():
    tokenizer = AutoTokenizer.from_pretrained(BOS_EOS_TOKENIZER)
    bt = DNABatchTokenizer(tokenizer, soft_mask_weight=0.01)
    meta = bt.metadata

    assert meta["has_bos"] is True
    assert meta["has_eos"] is True


@_skip_if_tokenizer_unavailable(BOS_EOS_TOKENIZER)
def test_bos_only():
    """When only BOS is defined, only BOS is prepended (no EOS)."""
    tokenizer = AutoTokenizer.from_pretrained(BOS_EOS_TOKENIZER)
    # Patch out EOS so only BOS is active
    tokenizer.eos_token_id = None
    bt = DNABatchTokenizer(tokenizer, soft_mask_weight=0.1)

    assert bt.num_special_tokens == 1

    result = bt([{"seq": "ACgt"}])[0]
    assert result["input_ids"].shape == (5,)  # BOS + 4 chars
    assert result["input_ids"][0] == tokenizer.bos_token_id
    np.testing.assert_allclose(result["loss_weight"], [1.0, 1.0, 1.0, 0.1, 0.1])


@_skip_if_tokenizer_unavailable(BOS_EOS_TOKENIZER)
def test_eos_only():
    """When only EOS is defined, only EOS is appended (no BOS)."""
    tokenizer = AutoTokenizer.from_pretrained(BOS_EOS_TOKENIZER)
    eos_id = tokenizer.eos_token_id
    # Patch out BOS so only EOS is active
    tokenizer.bos_token_id = None
    bt = DNABatchTokenizer(tokenizer, soft_mask_weight=0.1)

    assert bt.num_special_tokens == 1

    result = bt([{"seq": "ACgt"}])[0]
    assert result["input_ids"].shape == (5,)  # 4 chars + EOS
    assert result["input_ids"][-1] == eos_id
    np.testing.assert_allclose(result["loss_weight"], [1.0, 1.0, 0.1, 0.1, 1.0])
