# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses

import numpy as np
import pytest

from levanter.data.text import DNABatchTokenizer
from levanter.tokenizers import load_tokenizer


def _skip_if_tokenizer_unavailable(tokenizer_name: str):
    def try_load(name):
        try:
            load_tokenizer(name)
        except Exception:
            return False
        return True

    return pytest.mark.skipif(not try_load(tokenizer_name), reason=f"Tokenizer {tokenizer_name} not accessible")


NO_BOS_EOS_TOKENIZER = "songlab/tokenizer-dna-clm"
BOS_EOS_TOKENIZER = "bolinas-dna/tokenizer-char"


@_skip_if_tokenizer_unavailable(NO_BOS_EOS_TOKENIZER)
def test_no_special_tokens():
    """With a tokenizer that has no BOS/EOS, output matches input length."""
    tokenizer = load_tokenizer(NO_BOS_EOS_TOKENIZER)
    bt = DNABatchTokenizer(tokenizer, lowercase_weight=0.01)

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
def test_weights_target_aligned_no_special_tokens():
    """Weights are target-aligned: loss_weight[i] reflects the case of input_ids[i+1]."""
    tokenizer = load_tokenizer(NO_BOS_EOS_TOKENIZER)
    bt = DNABatchTokenizer(tokenizer, lowercase_weight=0.1)

    # Sequence: A  C  g  t
    # Targets:  C  g  t  (last position placeholder)
    # Weights:  1.0  0.1  0.1  0.0
    batch = [{"seq": "ACgt"}]
    result = bt(batch)[0]

    np.testing.assert_allclose(result["loss_weight"], [1.0, 0.1, 0.1, 0.0])


@_skip_if_tokenizer_unavailable(BOS_EOS_TOKENIZER)
def test_bos_eos_tokens_added():
    """With a tokenizer that has BOS/EOS, they are prepended/appended."""
    tokenizer = load_tokenizer(BOS_EOS_TOKENIZER)
    bt = DNABatchTokenizer(tokenizer, lowercase_weight=0.01)

    assert bt.num_special_tokens == 2

    seq = "ACGT"
    batch = [{"seq": seq}]
    result = bt(batch)[0]

    assert result["input_ids"].shape == (len(seq) + 2,)
    assert result["loss_weight"].shape == (len(seq) + 2,)

    assert result["input_ids"][0] == tokenizer.bos_token_id
    assert result["input_ids"][-1] == tokenizer.eos_token_id

    # Interior tokens should match tokenizing without special tokens
    plain_ids = tokenizer.encode(seq, add_special_tokens=False)
    np.testing.assert_array_equal(result["input_ids"][1:-1], plain_ids)


@_skip_if_tokenizer_unavailable(BOS_EOS_TOKENIZER)
def test_loss_weights_with_bos_eos():
    """Weights are target-aligned with BOS/EOS."""
    tokenizer = load_tokenizer(BOS_EOS_TOKENIZER)
    bt = DNABatchTokenizer(tokenizer, lowercase_weight=0.1)

    batch = [{"seq": "ACgt"}]
    result = bt(batch)[0]

    # Tokens:  BOS  A    C    g    t    EOS
    # Targets: A    C    g    t    EOS  (last masked by loss fn)
    # Weights: 1.0  1.0  0.1  0.1  1.0  1.0
    expected = np.array([1.0, 1.0, 0.1, 0.1, 1.0, 1.0], dtype=np.float32)
    np.testing.assert_allclose(result["loss_weight"], expected)


@_skip_if_tokenizer_unavailable(BOS_EOS_TOKENIZER)
def test_batch_consistency_with_bos_eos():
    """All sequences in a batch get BOS/EOS and have the same length."""
    tokenizer = load_tokenizer(BOS_EOS_TOKENIZER)
    bt = DNABatchTokenizer(tokenizer, lowercase_weight=0.01)

    batch = [{"seq": "AAAA"}, {"seq": "CCCC"}, {"seq": "TTTT"}]
    results = bt(batch)

    lengths = {r["input_ids"].shape[0] for r in results}
    assert len(lengths) == 1
    assert lengths.pop() == 4 + 2  # seq_len + BOS + EOS


@_skip_if_tokenizer_unavailable(NO_BOS_EOS_TOKENIZER)
def test_metadata_no_special_tokens():
    tokenizer = load_tokenizer(NO_BOS_EOS_TOKENIZER)
    bt = DNABatchTokenizer(tokenizer, lowercase_weight=0.5)
    meta = bt.metadata

    assert meta["has_bos"] is False
    assert meta["has_eos"] is False
    assert meta["uppercase_weight"] == 1.0
    assert meta["lowercase_weight"] == 0.5


@_skip_if_tokenizer_unavailable(BOS_EOS_TOKENIZER)
def test_metadata_with_special_tokens():
    tokenizer = load_tokenizer(BOS_EOS_TOKENIZER)
    bt = DNABatchTokenizer(tokenizer, lowercase_weight=0.01)
    meta = bt.metadata

    assert meta["has_bos"] is True
    assert meta["has_eos"] is True


@_skip_if_tokenizer_unavailable(BOS_EOS_TOKENIZER)
def test_bos_only():
    """When only BOS is defined, only BOS is prepended (no EOS)."""
    tokenizer = load_tokenizer(BOS_EOS_TOKENIZER)
    # Patch out EOS so only BOS is active
    tokenizer = dataclasses.replace(tokenizer, _eos_id=None)
    bt = DNABatchTokenizer(tokenizer, lowercase_weight=0.1)

    assert bt.num_special_tokens == 1

    result = bt([{"seq": "ACgt"}])[0]
    assert result["input_ids"].shape == (5,)  # BOS + 4 chars
    assert result["input_ids"][0] == tokenizer.bos_token_id
    # Tokens:  BOS  A    C    g    t
    # Targets: A    C    g    t    (last masked)
    # Weights: 1.0  1.0  0.1  0.1  0.0
    np.testing.assert_allclose(result["loss_weight"], [1.0, 1.0, 0.1, 0.1, 0.0])


@_skip_if_tokenizer_unavailable(NO_BOS_EOS_TOKENIZER)
def test_uppercase_weight_zero():
    """Setting uppercase_weight=0 zeroes out loss for predicting uppercase targets."""
    tokenizer = load_tokenizer(NO_BOS_EOS_TOKENIZER)
    bt = DNABatchTokenizer(tokenizer, uppercase_weight=0.0, lowercase_weight=1.0)

    # Sequence: A    C    g    t
    # Targets:  C    g    t    (last)
    # Weights:  0.0  1.0  1.0  0.0
    batch = [{"seq": "ACgt"}]
    result = bt(batch)[0]

    np.testing.assert_allclose(result["loss_weight"], [0.0, 1.0, 1.0, 0.0])


@_skip_if_tokenizer_unavailable(BOS_EOS_TOKENIZER)
def test_eos_only():
    """When only EOS is defined, only EOS is appended (no BOS)."""
    tokenizer = load_tokenizer(BOS_EOS_TOKENIZER)
    eos_id = tokenizer.eos_token_id
    # Patch out BOS so only EOS is active
    tokenizer = dataclasses.replace(tokenizer, _bos_id=None)
    bt = DNABatchTokenizer(tokenizer, lowercase_weight=0.1)

    assert bt.num_special_tokens == 1

    result = bt([{"seq": "ACgt"}])[0]
    assert result["input_ids"].shape == (5,)  # 4 chars + EOS
    assert result["input_ids"][-1] == eos_id
    # Tokens:  A    C    g    t    EOS
    # Targets: C    g    t    EOS  (last masked)
    # Weights: 1.0  0.1  0.1  1.0  1.0
    np.testing.assert_allclose(result["loss_weight"], [1.0, 0.1, 0.1, 1.0, 1.0])
