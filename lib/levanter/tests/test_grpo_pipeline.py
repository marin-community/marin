# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import haliax as hax
import jax.numpy as jnp
import numpy as np
import pytest

from levanter.grpo_model import GrpoExample
from levanter.grpo_pipeline import packed_pipeline_batch, pipeline_batch


def test_packing_preserves_response_objectives_positions_and_segment_boundaries():
    Batch, Position, Response = hax.Axis("batch", 4), hax.Axis("position", 8), hax.Axis("response", 4)
    tokens = jnp.array(
        [
            [0, 0, 1, 2, 3, 4, 0, 0],
            [0, 5, 6, 7, 8, 0, 0, 0],
            [0, 0, 0, 9, 10, 11, 12, 0],
            [0, 13, 14, 15, 16, 17, 0, 0],
        ]
    )
    attention = (tokens != 0).astype(jnp.int32)
    response_mask = attention[:, 4:].astype(jnp.float32)
    policy_weights = response_mask / (response_mask.sum(axis=1, keepdims=True) * 4)
    positions = jnp.maximum(attention.cumsum(axis=1) - 1, 0)
    positions = positions + jnp.arange(4)[:, None] * attention

    def seq(value):
        return hax.named(value, (Batch, Position))

    def resp(value):
        return hax.named(value, (Batch, Response))

    example = GrpoExample(
        seq(tokens),
        seq(attention),
        seq(positions),
        resp(response_mask),
        resp(response_mask * jnp.array([1.0, -0.6, 0.3, -1.0])[:, None]),
        resp(policy_weights),
        resp(policy_weights * 0.7),
        resp(jnp.full((4, 4), -3.2)),
        resp(jnp.full((4, 4), -3.1)),
    )
    unpacked = pipeline_batch(example, microbatches=2)
    packed = packed_pipeline_batch(
        example, sequence_length=10, rows_per_microbatch=1, pad_token_id=0, minimum_microbatches=3
    )
    assert packed.tokens.shape == (3, 1, 10)
    assert np.all(np.asarray(packed.segment_ids[-1]) == -1)
    for name in ("advantages", "policy_weights", "kl_weights", "old_logprobs", "reference_logprobs"):
        before = np.asarray(getattr(unpacked, name))[np.asarray(unpacked.policy_weights) > 0]
        after = np.asarray(getattr(packed, name))[np.asarray(packed.policy_weights) > 0]
        np.testing.assert_array_equal(after, before)
    for document in range(4):
        locations = np.asarray(packed.segment_ids) == document
        np.testing.assert_array_equal(
            np.asarray(packed.tokens)[locations], np.asarray(tokens[document])[attention[document] > 0]
        )
        np.testing.assert_array_equal(
            np.asarray(packed.position_ids)[locations], np.asarray(positions[document])[attention[document] > 0]
        )
    segments = np.asarray(packed.segment_ids)
    boundary = (segments[..., 1:] != segments[..., :-1]) | (segments[..., 1:] < 0)
    assert np.all(np.asarray(packed.policy_weights)[boundary] == 0)
    assert np.all(np.asarray(packed.kl_weights)[boundary] == 0)
    with pytest.raises(ValueError, match="exceeds"):
        packed_pipeline_batch(example, sequence_length=4, rows_per_microbatch=1, pad_token_id=0)
