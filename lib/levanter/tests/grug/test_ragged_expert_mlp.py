# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""The ragged EP expert MLP on ``ragged_dot`` matches a per-expert dense reference, gradients included."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from levanter.grug._moe.ep_ragged_all_to_all import _expert_mlp

_HIDDEN = 16
_INTERMEDIATE = 24
_EXPERTS = 3
# Receiver rows: the groups fill 20 of them, and the remaining 8 are the zeroed padding that the
# backend charges to the last expert.
_ROWS = 28
_GROUP_SIZES = np.array([7, 9, 4])


def _reference(x, w13, w2, sizes):
    """Each expert's rows through a dense SwiGLU MLP with that expert's weights."""
    out = []
    start = 0
    for expert, size in enumerate(sizes):
        rows = x[start : start + size]
        gate, up = jnp.split(rows @ w13[expert], [_INTERMEDIATE], axis=-1)
        out.append((jax.nn.silu(gate) * up) @ w2[expert])
        start += size
    return jnp.concatenate(out + [jnp.zeros((x.shape[0] - start, x.shape[1]), x.dtype)])


@pytest.fixture
def operands():
    key = jax.random.key(0)
    kx, k13, k2 = jax.random.split(key, 3)
    active = int(_GROUP_SIZES.sum())
    x = jax.random.normal(kx, (_ROWS, _HIDDEN), jnp.float32)
    x = jnp.where(jnp.arange(_ROWS)[:, None] < active, x, 0.0)
    w13 = jax.random.normal(k13, (_EXPERTS, _HIDDEN, 2 * _INTERMEDIATE), jnp.float32) / np.sqrt(_HIDDEN)
    w2 = jax.random.normal(k2, (_EXPERTS, _INTERMEDIATE, _HIDDEN), jnp.float32) / np.sqrt(_INTERMEDIATE)
    return x, w13, w2


def _physical_group_sizes():
    """The backend's view: trailing padding is charged to the last expert."""
    return jnp.asarray(_GROUP_SIZES, jnp.int32).at[-1].add(_ROWS - int(_GROUP_SIZES.sum()))


def test_forward_matches_the_per_expert_reference(operands):
    x, w13, w2 = operands
    got = _expert_mlp(x, w13, w2, _physical_group_sizes(), jax.nn.silu)
    want = _reference(x, w13, w2, _GROUP_SIZES)
    np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-5)


def test_gradients_match_the_per_expert_reference(operands):
    x, w13, w2 = operands
    cotangent = jax.random.normal(jax.random.key(1), (_ROWS, _HIDDEN), jnp.float32)
    # Padding rows never reach the return transport, so their cotangent is zero in production.
    cotangent = jnp.where(jnp.arange(_ROWS)[:, None] < int(_GROUP_SIZES.sum()), cotangent, 0.0)

    sizes = _physical_group_sizes()
    got = jax.vjp(lambda a, b, c: _expert_mlp(a, b, c, sizes, jax.nn.silu), x, w13, w2)[1](cotangent)
    want = jax.vjp(lambda a, b, c: _reference(a, b, c, _GROUP_SIZES), x, w13, w2)[1](cotangent)
    for name, g, r in zip(("dx", "dw13", "dw2"), got, want, strict=True):
        np.testing.assert_allclose(g, r, rtol=1e-5, atol=1e-5, err_msg=name)
