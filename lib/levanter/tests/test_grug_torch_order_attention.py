# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from levanter.grug.attention import AttentionMask, attention
from levanter.main.train_lm import preserve_hf_runtime_settings
from levanter.models.snowball import SnowballConfig

S, HQ, HKV, D = 9, 4, 2, 12


def _bf16(key: int, shape: tuple[int, ...]) -> jax.Array:
    return jax.random.normal(jax.random.PRNGKey(key), shape).astype(jnp.bfloat16)


def _qkv(batch: int = 1) -> tuple[jax.Array, jax.Array, jax.Array]:
    return _bf16(0, (batch, S, HQ, D)), _bf16(1, (batch, S, HKV, D)), _bf16(2, (batch, S, HKV, D))


# Outputs captured on CPU with PyTorch 2.11.0 eager einsum and softmax.
# Repeat each KV head for its two query heads, compute (q.float() / sqrt(3)) @ k.float().T,
# fill masked scores with -1e9, then multiply BF16 softmax probabilities by BF16 v.
# The fixture distinguishes casting the probabilities before the value matmul from casting only its result.
_TORCH_EAGER_OUTPUTS = {
    2: (
        (0.5, 0.5, 0.5, 0.5, 0.5, 0.5),
        (-1.2265625, 1.3046875, 0.154296875, 0.2333984375, 0.625, 0.447265625),
        (0.65625, -0.26953125, -0.26171875, 0.345703125, -0.016845703125, -0.203125),
        (1.203125, -0.83984375, -0.73828125, 1.2265625, -0.8046875, -0.5859375),
    ),
    None: (
        (0.5, 0.5, 0.5, 0.5, 0.5, 0.5),
        (-1.2265625, 1.3046875, 0.154296875, 0.2333984375, 0.625, 0.447265625),
        (0.6015625, 0.003173828125, 0.007568359375, 0.474609375, 0.41015625, 0.37890625),
        (0.98046875, -0.44921875, -0.390625, 0.89453125, -0.412109375, -0.349609375),
    ),
}


@pytest.mark.parametrize("window", [2, None])
def test_torch_order_matches_torch_eager_rounding(window):
    rng = np.random.default_rng(5)
    q, k, v = (
        jnp.asarray(rng.integers(-12, 13, shape) / 8, dtype=jnp.bfloat16)
        for shape in ((1, 4, 2, 3), (1, 4, 1, 3), (1, 4, 1, 3))
    )
    mask = AttentionMask(is_causal=True, sliding_window=window)
    ours = attention(q, k, v, mask, implementation="torch_order")
    expected = np.asarray(_TORCH_EAGER_OUTPUTS[window], dtype=np.float32).reshape(q.shape)
    assert ours.dtype == jnp.bfloat16
    np.testing.assert_array_equal(np.asarray(ours, dtype=np.float32), expected)

    # A dense mask selects the reference backend by default on CPU, GPU, and TPU.
    dense_mask = mask.materialize_mask(q.shape[1], k.shape[1])
    default = attention(q, k, v, dense_mask)
    reference = attention(q, k, v, dense_mask, implementation="reference")
    np.testing.assert_array_equal(np.asarray(default), np.asarray(reference))
    assert not np.array_equal(np.asarray(ours), np.asarray(default))


@pytest.mark.parametrize("window", [3, None])
def test_torch_order_keeps_packed_segments_apart(window):
    """A packed two-segment row: each segment's output equals that segment run alone, and changing one segment's
    keys and values leaves the other's output unchanged."""
    q, k, v = _qkv()
    split = 4
    segments = jnp.asarray([[0] * split + [1] * (S - split)])
    packed = AttentionMask(is_causal=True, sliding_window=window, segment_ids=(segments, segments))
    alone = AttentionMask(is_causal=True, sliding_window=window)
    out = attention(q, k, v, packed, implementation="torch_order")
    for lo, hi in ((0, split), (split, S)):
        single = attention(q[:, lo:hi], k[:, lo:hi], v[:, lo:hi], alone, implementation="torch_order")
        np.testing.assert_array_equal(np.asarray(out[:, lo:hi]), np.asarray(single))
    moved = attention(q, k.at[:, :split].set(3.0), v.at[:, :split].set(3.0), packed, implementation="torch_order")
    np.testing.assert_array_equal(np.asarray(out[:, split:]), np.asarray(moved[:, split:]))


@pytest.mark.parametrize("window", [3, None])
def test_torch_order_trailing_padding_does_not_change_document(window):
    q, k, v = _qkv()
    real = 6
    segments = jnp.asarray([[0] * real + [-1] * (S - real)])
    mask = AttentionMask(is_causal=True, sliding_window=window, segment_ids=(segments, segments))
    out = attention(q, k, v, mask, implementation="torch_order")
    unpadded = AttentionMask(is_causal=True, sliding_window=window)
    expected = attention(q[:, :real], k[:, :real], v[:, :real], unpadded, implementation="torch_order")
    np.testing.assert_array_equal(np.asarray(out[:, :real]), np.asarray(expected))


def _tiny(**overrides) -> SnowballConfig:
    base = dict(
        vocab_size=48,
        hidden_dim=32,
        intermediate_dim=64,
        shared_expert_intermediate_dim=48,
        num_experts=8,
        num_experts_per_token=2,
        num_layers=5,
        num_heads=HQ,
        num_kv_heads=HKV,
        head_dim=D,
        max_seq_len=32,
        sliding_window=4,
        qk_mult=1.75,
        layer_norm_eps=1e-5,
        initializer_std=0.2,
    )
    return SnowballConfig(**(base | overrides))


def test_hf_reload_keeps_the_attention_implementation():
    requested = _tiny(attention_implementation="torch_order")
    reloaded = SnowballConfig.from_hf_config(requested.to_hf_config(requested.vocab_size))
    assert reloaded.attention_implementation is None
    restored = preserve_hf_runtime_settings(reloaded, requested)
    assert restored.attention_implementation == "torch_order"
