# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import jax.numpy as jnp
from levanter.grug.attention import AttentionMask, ThdSegmentMetadata

from experiments.grug.moe.model import _layer_attention_masks


def test_layer_attention_masks_use_direct_short_window_and_full_long_mask():
    metadata = ThdSegmentMetadata(
        segment_lengths=jnp.array([[4, 4]], dtype=jnp.int32),
        num_segments=jnp.array([2], dtype=jnp.int32),
    )
    mask = AttentionMask(is_causal=True, thd_segment_metadata=metadata)

    short_mask, long_mask = _layer_attention_masks(mask, sliding_window=8)

    assert short_mask.sliding_window == 8
    assert long_mask.sliding_window is None
    assert short_mask.thd_segment_metadata is metadata
    assert long_mask.thd_segment_metadata is metadata
    assert short_mask.segment_ids is None
    assert long_mask.segment_ids is None
