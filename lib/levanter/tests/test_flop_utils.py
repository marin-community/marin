# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import pytest

from levanter.utils.flop_utils import lm_flops_per_token

SHAPE = dict(hidden_dim=64, intermediate_dim=128, num_layers=8, num_kv_heads=2, num_heads=4, vocab_size=1000, glu=True)


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
