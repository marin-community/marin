# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The June model's half-RoPE, computed without splitting or concatenating head_dim, matches the split/concat form."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from experiments.june_tpu_67b_a2b.moe.model import RotaryConfig, _apply_half_rope

SEQ, HEADS, HEAD_DIM = 16, 3, 32


def _split_concat_half_rope(x, positions, theta):
    half = HEAD_DIM // 2
    frequencies = 1.0 / theta ** (jnp.arange(half // 2, dtype=jnp.float32) / (half // 2))
    angles = positions.astype(jnp.float32)[..., None] * frequencies
    cos, sin = jnp.cos(angles)[:, :, None, :], jnp.sin(angles)[:, :, None, :]
    x1, x2 = jnp.split(x[..., :half], 2, axis=-1)
    rotated = jnp.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1).astype(x.dtype)
    return jnp.concatenate([rotated, x[..., half:]], axis=-1)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
@pytest.mark.parametrize("explicit_positions", [False, True])
def test_half_rope_matches_split_concat(dtype, explicit_positions):
    rope = RotaryConfig()
    keys = jax.random.split(jax.random.key(0))
    q, k = (jax.random.normal(key, (2, SEQ, HEADS, HEAD_DIM)).astype(dtype) for key in keys)
    positions = jax.random.randint(jax.random.key(1), (2, SEQ), 0, 4096) if explicit_positions else None
    reference_positions = positions if explicit_positions else jnp.broadcast_to(jnp.arange(SEQ)[None], (1, SEQ))

    def new(q, k):
        return _apply_half_rope(q, k, seq_len=SEQ, head_dim=HEAD_DIM, rope=rope, position_ids=positions)

    def old(q, k):
        return tuple(_split_concat_half_rope(x, reference_positions, rope.theta) for x in (q, k))

    def loss(fn):
        return lambda q, k: sum(jnp.sum(jnp.sin(x.astype(jnp.float32))) for x in fn(q, k))

    # Op by op the arithmetic is the same, so values and gradients are bitwise equal.
    with jax.disable_jit():
        pairs = [*zip(new(q, k), old(q, k), strict=True)]
        pairs += zip(jax.grad(loss(new), (0, 1))(q, k), jax.grad(loss(old), (0, 1))(q, k), strict=True)
    for got, want in pairs:
        assert got.dtype == dtype
        np.testing.assert_array_equal(np.asarray(got, np.float32), np.asarray(want, np.float32))
    # XLA may fuse multiply-adds, so compiled values need a float32 rounding allowance.
    atol = 2 * np.finfo(np.float32).eps * float(jnp.max(jnp.abs(jnp.stack([q, k]).astype(jnp.float32))))
    for got, want in zip(jax.jit(new)(q, k), jax.jit(old)(q, k), strict=True):
        np.testing.assert_allclose(np.asarray(got, np.float32), np.asarray(want, np.float32), rtol=0, atol=atol)
