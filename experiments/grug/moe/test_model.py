# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import jax.numpy as jnp
from levanter.grug.attention import AttentionMask, ThdSegmentMetadata

from experiments.grug.moe.model import GrugModelConfig, _model_sliding_attention_masks


def test_model_sliding_attention_masks_preserve_thd_metadata():
    cfg = GrugModelConfig(
        vocab_size=16,
        hidden_dim=8,
        num_layers=2,
        num_heads=2,
        num_kv_heads=1,
        max_seq_len=16,
        sliding_window=8,
    )
    metadata = ThdSegmentMetadata(
        segment_lengths=jnp.array([[4, 4]], dtype=jnp.int32),
        num_segments=jnp.array([2], dtype=jnp.int32),
    )
    mask = AttentionMask(is_causal=True, thd_segment_metadata=metadata)

    short_mask, long_mask = _model_sliding_attention_masks(mask, cfg)

    assert short_mask.sliding_window == 8
    assert long_mask.sliding_window == 16
    assert short_mask.thd_segment_metadata is metadata
    assert long_mask.thd_segment_metadata is metadata
    assert short_mask.segment_ids is None
    assert long_mask.segment_ids is None
