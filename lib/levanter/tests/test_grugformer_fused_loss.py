# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import jax
import jax.numpy as jnp

from levanter.grug.loss import (
    linear_softmax_cross_entropy_loss_and_logz,
    next_token_linear_softmax_cross_entropy,
)


def _full_loss_and_logz(hidden: jax.Array, lm_head: jax.Array, labels: jax.Array) -> tuple[jax.Array, jax.Array]:
    logits = hidden @ lm_head
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    logz = jax.scipy.special.logsumexp(logits, axis=-1)
    nll = -jnp.take_along_axis(log_probs, labels[..., None], axis=-1)[..., 0]
    return nll, logz


def test_linear_softmax_cross_entropy_matches_full_logits():
    key = jax.random.key(0)
    b, s, h, v = 2, 5, 8, 17
    hidden = jax.random.normal(key, (b, s, h), dtype=jnp.float32)
    lm_head = jax.random.normal(jax.random.key(1), (h, v), dtype=jnp.float32)
    labels = jax.random.randint(jax.random.key(2), (b, s), 0, v, dtype=jnp.int32)

    loss_full, logz_full = _full_loss_and_logz(hidden, lm_head, labels)
    loss_blk, logz_blk = linear_softmax_cross_entropy_loss_and_logz(hidden, lm_head, labels, block_size=6)

    assert loss_blk.shape == loss_full.shape
    assert logz_blk.shape == logz_full.shape
    assert jnp.allclose(loss_blk, loss_full, atol=1e-4, rtol=1e-4)
    assert jnp.allclose(logz_blk, logz_full, atol=1e-4, rtol=1e-4)


def test_linear_softmax_cross_entropy_jittable():
    key = jax.random.key(0)
    b, s, h, v = 2, 3, 8, 11
    hidden = jax.random.normal(key, (b, s, h), dtype=jnp.float32)
    lm_head = jax.random.normal(jax.random.key(1), (h, v), dtype=jnp.float32)
    labels = jax.random.randint(jax.random.key(2), (b, s), 0, v, dtype=jnp.int32)

    fn = jax.jit(lambda x, w, y: linear_softmax_cross_entropy_loss_and_logz(x, w, y, block_size=4))
    loss, logz = fn(hidden, lm_head, labels)
    assert loss.shape == (b, s)
    assert logz.shape == (b, s)


def test_linear_softmax_cross_entropy_grad_matches_full():
    key = jax.random.key(0)
    b, s, h, v = 2, 3, 8, 13
    hidden = jax.random.normal(key, (b, s, h), dtype=jnp.float32)
    lm_head = jax.random.normal(jax.random.key(1), (h, v), dtype=jnp.float32)
    labels = jax.random.randint(jax.random.key(2), (b, s), 0, v, dtype=jnp.int32)

    def loss_full_fn(x):
        loss, _ = _full_loss_and_logz(x, lm_head, labels)
        return jnp.mean(loss)

    def loss_blk_fn(x):
        loss, _ = linear_softmax_cross_entropy_loss_and_logz(x, lm_head, labels, block_size=5)
        return jnp.mean(loss)

    g_full = jax.grad(loss_full_fn)(hidden)
    g_blk = jax.grad(loss_blk_fn)(hidden)
    assert jnp.allclose(g_blk, g_full, atol=1e-4, rtol=1e-4)


def test_next_token_loss_ignores_last_position():
    key = jax.random.key(0)
    b, s, h, v = 2, 6, 8, 19
    token_ids = jax.random.randint(key, (b, s), 0, v, dtype=jnp.int32)
    hidden = jax.random.normal(jax.random.key(1), (b, s, h), dtype=jnp.float32)
    lm_head = jax.random.normal(jax.random.key(2), (h, v), dtype=jnp.float32)

    # reference next-token: labels are shifted left, last ignored
    labels = jnp.concatenate([token_ids[:, 1:], jnp.zeros((b, 1), dtype=jnp.int32)], axis=-1)
    loss_full, _ = _full_loss_and_logz(hidden, lm_head, labels)
    loss_full = loss_full[:, :-1]

    loss_blk = next_token_linear_softmax_cross_entropy(
        token_ids, hidden, lm_head, block_size=7, reduction="none", dtype=jnp.float32
    )
    assert loss_blk.shape == (b, s)
    assert jnp.allclose(loss_blk[:, :-1], loss_full, atol=1e-4, rtol=1e-4)
    assert jnp.allclose(loss_blk[:, -1], jnp.zeros((b,), dtype=jnp.float32))
