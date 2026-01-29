# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import numpy as np

import jax
import jax.numpy as jnp
from jax.sharding import AxisType, Mesh

from levanter.grug.loss import linear_softmax_cross_entropy_loss_and_logz


def _make_grug_mesh() -> Mesh:
    devices = jax.devices()
    if not devices:
        raise RuntimeError("No JAX devices available")
    # We only require a mesh context here so the loss can provide `out_sharding=...`.
    mesh_devices = np.array(devices).reshape(len(devices), 1)
    return Mesh(
        mesh_devices,
        axis_names=("data", "model"),
        axis_types=(AxisType.Explicit, AxisType.Explicit),
    )


def _full_loss_and_logz(
    hidden: jax.Array, lm_head: jax.Array, labels: jax.Array, *, precision: jax.lax.PrecisionLike = None
) -> tuple[jax.Array, jax.Array]:
    logits = jax.lax.dot_general(
        hidden,
        lm_head,
        dimension_numbers=(((hidden.ndim - 1,), (0,)), ((), ())),
        precision=precision,
    )
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    logz = jax.scipy.special.logsumexp(logits, axis=-1)
    nll = -jnp.take_along_axis(log_probs, labels[..., None], axis=-1)[..., 0]
    return nll, logz


def test_linear_softmax_cross_entropy_matches_full_logits():
    key = jax.random.key(0)
    b, s, h, v = len(jax.devices()), 5, 8, 17
    hidden = jax.random.normal(key, (b, s, h), dtype=jnp.float32)
    lm_head = jax.random.normal(jax.random.key(1), (h, v), dtype=jnp.float32)
    labels = jax.random.randint(jax.random.key(2), (b, s), 0, v, dtype=jnp.int32)

    mesh = _make_grug_mesh()
    with jax.set_mesh(mesh):
        loss_full, logz_full = jax.jit(_full_loss_and_logz, static_argnames=("precision",))(
            hidden, lm_head, labels, precision=jax.lax.Precision.HIGHEST
        )
        loss_blk, logz_blk = jax.jit(
            lambda x, w, y: linear_softmax_cross_entropy_loss_and_logz(
                x, w, y, block_size=6, precision=jax.lax.Precision.HIGHEST
            )
        )(hidden, lm_head, labels)

    assert loss_blk.shape == loss_full.shape
    assert logz_blk.shape == logz_full.shape
    # On TPU, the streaming logsumexp (blockwise) can differ from the full logsumexp due to
    # different associativity/rounding behavior. We use HIGHEST matmul precision above to
    # keep this fairly tight.
    if jax.default_backend() == "tpu":
        assert jnp.allclose(loss_blk, loss_full, atol=5e-3, rtol=5e-3)
        assert jnp.allclose(logz_blk, logz_full, atol=5e-3, rtol=5e-3)
    else:
        assert jnp.allclose(loss_blk, loss_full, atol=1e-4, rtol=1e-4)
        assert jnp.allclose(logz_blk, logz_full, atol=1e-4, rtol=1e-4)


def test_linear_softmax_cross_entropy_jittable():
    key = jax.random.key(0)
    b, s, h, v = len(jax.devices()), 3, 8, 11
    hidden = jax.random.normal(key, (b, s, h), dtype=jnp.float32)
    lm_head = jax.random.normal(jax.random.key(1), (h, v), dtype=jnp.float32)
    labels = jax.random.randint(jax.random.key(2), (b, s), 0, v, dtype=jnp.int32)

    mesh = _make_grug_mesh()
    with jax.set_mesh(mesh):
        fn = jax.jit(
            lambda x, w, y: linear_softmax_cross_entropy_loss_and_logz(
                x, w, y, block_size=4, precision=jax.lax.Precision.HIGHEST
            )
        )
        loss, logz = fn(hidden, lm_head, labels)
    assert loss.shape == (b, s)
    assert logz.shape == (b, s)


def test_linear_softmax_cross_entropy_grad_matches_full():
    key = jax.random.key(0)
    b, s, h, v = len(jax.devices()), 3, 8, 13
    hidden = jax.random.normal(key, (b, s, h), dtype=jnp.float32)
    lm_head = jax.random.normal(jax.random.key(1), (h, v), dtype=jnp.float32)
    labels = jax.random.randint(jax.random.key(2), (b, s), 0, v, dtype=jnp.int32)

    def loss_full_fn(x):
        loss, _ = _full_loss_and_logz(x, lm_head, labels, precision=jax.lax.Precision.HIGHEST)
        return jnp.mean(loss)

    def loss_blk_fn(x):
        loss, _ = linear_softmax_cross_entropy_loss_and_logz(
            x, lm_head, labels, block_size=5, precision=jax.lax.Precision.HIGHEST
        )
        return jnp.mean(loss)

    mesh = _make_grug_mesh()
    with jax.set_mesh(mesh):
        g_full = jax.grad(loss_full_fn)(hidden)
        g_blk = jax.grad(loss_blk_fn)(hidden)
    if jax.default_backend() == "tpu":
        assert jnp.allclose(g_blk, g_full, atol=5e-4, rtol=5e-3)
    else:
        assert jnp.allclose(g_blk, g_full, atol=1e-4, rtol=1e-4)
