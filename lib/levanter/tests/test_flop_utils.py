# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from levanter.utils.flop_utils import lm_attention_flops_per_token, lm_flops_per_token

SHAPE = dict(hidden_dim=64, intermediate_dim=128, num_layers=8, num_kv_heads=2, num_heads=4, vocab_size=1000, glu=True)
ATTENTION_SHAPE = {key: SHAPE[key] for key in ("hidden_dim", "num_heads", "num_layers")}


def test_explicit_full_attention_count_overrides_the_stride():
    """A schedule of every 4th layer plus the last has one more full layer than the stride alone; the difference
    is exactly one layer's attention over the context beyond the window."""
    window, seq_len = 64, 1024
    by_stride = lm_flops_per_token(**SHAPE, seq_len=seq_len, sliding_window=window, global_every=4)
    explicit = lm_flops_per_token(**SHAPE, seq_len=seq_len, sliding_window=window, num_full_attention_layers=3)
    one_full = lm_flops_per_token(**SHAPE, seq_len=seq_len, sliding_window=window, num_full_attention_layers=1)
    no_full = lm_flops_per_token(**SHAPE, seq_len=seq_len, sliding_window=window, num_full_attention_layers=0)
    assert explicit - by_stride == pytest.approx(one_full - no_full)
    assert explicit > by_stride


def _causal_pairs(seq_len: int, window: int) -> int:
    query = np.arange(seq_len)[:, None]
    key = np.arange(seq_len)[None, :]
    return int(np.sum((key <= query) & (key > query - window)))


def test_attention_flops_match_full_and_sliding_window_masks():
    seq_len, window, num_full = 64, 16, 3
    num_layers = ATTENTION_SHAPE["num_layers"]
    hidden = ATTENTION_SHAPE["hidden_dim"]
    heads = ATTENTION_SHAPE["num_heads"]
    per_pair = 4 * hidden + 3 * heads
    full = lm_attention_flops_per_token(**ATTENTION_SHAPE, seq_len=seq_len)
    hybrid = lm_attention_flops_per_token(
        **ATTENTION_SHAPE, seq_len=seq_len, sliding_window=window, num_full_attention_layers=num_full
    )

    assert full.noncausal == pytest.approx(per_pair * num_layers * seq_len)
    assert full.causal == pytest.approx(per_pair * num_layers * _causal_pairs(seq_len, seq_len) / seq_len)
    assert hybrid.noncausal == pytest.approx(per_pair * (num_full * seq_len + (num_layers - num_full) * window))
    assert hybrid.causal == pytest.approx(
        per_pair
        * (num_full * _causal_pairs(seq_len, seq_len) + (num_layers - num_full) * _causal_pairs(seq_len, window))
        / seq_len
    )


def test_model_minus_attention_is_independent_of_context_and_window():
    rest = [
        lm_flops_per_token(**SHAPE, seq_len=seq_len, **schedule)
        - lm_attention_flops_per_token(**ATTENTION_SHAPE, seq_len=seq_len, **schedule).noncausal
        for schedule in ({}, {"sliding_window": 64, "global_every": 4})
        for seq_len in (512, 1024)
    ]
    assert rest == pytest.approx([rest[0]] * len(rest))
