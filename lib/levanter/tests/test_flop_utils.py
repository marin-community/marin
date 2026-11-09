# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from levanter.utils.flop_utils import lm_attention_flops_per_token, lm_flops_per_token

SHAPE = dict(hidden_dim=64, intermediate_dim=128, num_layers=8, num_kv_heads=2, num_heads=4, vocab_size=1000, glu=True)
ATTENTION_SHAPE = dict(hidden_dim=64, num_heads=4, num_layers=8)


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
    """Query-key pairs a causal mask keeps when each query sees at most the last ``window`` keys, itself included."""
    query = np.arange(seq_len)[:, None]
    key = np.arange(seq_len)[None, :]
    return int(np.sum((key <= query) & (key > query - window)))


def test_attention_flops_charge_the_query_key_pairs_each_mask_keeps():
    """Non-causally every layer charges all of its span for every query; causally only the pairs the mask keeps."""
    seq_len, window, num_full = 64, 16, 3
    num_layers = ATTENTION_SHAPE["num_layers"]
    full = lm_attention_flops_per_token(**ATTENTION_SHAPE, seq_len=seq_len)
    hybrid = lm_attention_flops_per_token(
        **ATTENTION_SHAPE, seq_len=seq_len, sliding_window=window, num_full_attention_layers=num_full
    )
    per_pair = full.noncausal * seq_len / (num_layers * seq_len * seq_len)

    assert full.causal == pytest.approx(per_pair * num_layers * _causal_pairs(seq_len, seq_len) / seq_len)
    assert hybrid.noncausal == pytest.approx(per_pair * (num_full * seq_len + (num_layers - num_full) * window))
    assert hybrid.causal == pytest.approx(
        per_pair
        * (num_full * _causal_pairs(seq_len, seq_len) + (num_layers - num_full) * _causal_pairs(seq_len, window))
        / seq_len
    )


def test_noncausal_attention_flops_are_what_the_model_count_charges():
    """Only the attention maps depend on the window, so the model count moves exactly as the attention count does."""
    seq_len, window = 1024, 64
    full_model = lm_flops_per_token(**SHAPE, seq_len=seq_len)
    hybrid_model = lm_flops_per_token(**SHAPE, seq_len=seq_len, sliding_window=window, global_every=4)
    full = lm_attention_flops_per_token(**ATTENTION_SHAPE, seq_len=seq_len)
    hybrid = lm_attention_flops_per_token(**ATTENTION_SHAPE, seq_len=seq_len, sliding_window=window, global_every=4)
    assert full_model - hybrid_model == pytest.approx(full.noncausal - hybrid.noncausal)


def test_full_attention_maps_follow_the_megatron_estimate():
    """Per layer and token: 2 * seq_len * hidden for the logits, 3 * seq_len * heads for the mask and softmax, and
    2 * seq_len * hidden to mix the values."""
    seq_len = 1024
    hidden, heads, layers = (ATTENTION_SHAPE[k] for k in ("hidden_dim", "num_heads", "num_layers"))
    full = lm_attention_flops_per_token(**ATTENTION_SHAPE, seq_len=seq_len)
    assert full.noncausal == pytest.approx(layers * seq_len * (4 * hidden + 3 * heads))


@pytest.mark.parametrize("schedule", [{}, {"sliding_window": 64, "global_every": 4}])
def test_noncausal_attention_flops_are_all_of_the_model_count_that_grows_with_context(schedule):
    """Without the attention maps, the model count per token is the same at every context length."""
    rest = [
        lm_flops_per_token(**SHAPE, seq_len=seq_len, **schedule)
        - lm_attention_flops_per_token(**ATTENTION_SHAPE, seq_len=seq_len, **schedule).noncausal
        for seq_len in (512, 1024, 4096)
    ]
    assert rest == pytest.approx([rest[0]] * len(rest))
