# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import jax
import jax.numpy as jnp
import pytest

from levanter.kernels.pallas.fused_cross_entropy_loss import api as fused_api
from levanter.kernels.pallas.fused_cross_entropy_loss import pallas_tpu
from levanter.kernels.pallas.fused_cross_entropy_loss.reference import (
    linear_softmax_cross_entropy_loss_reference,
)


def _make_toy_inputs():
    key = jax.random.PRNGKey(0)
    key_x, key_w, key_y = jax.random.split(key, 3)

    x = jax.random.normal(key_x, (2, 3, 4), dtype=jnp.float32)
    w = jax.random.normal(key_w, (4, 5), dtype=jnp.float32)
    y = jax.random.randint(key_y, (2, 3), 0, 5, dtype=jnp.int32)
    return x, w, y


def test_fused_cross_entropy_xla_matches_reference():
    x, w, y = _make_toy_inputs()

    loss = fused_api.fused_cross_entropy_loss_and_logsumexp_penalty(
        x.reshape(6, 4),
        y.reshape(6),
        w,
        reduction=None,
        logsumexp_weight=0.0,
        implementation="xla",
    )

    loss_ref, _ = linear_softmax_cross_entropy_loss_reference(
        x.reshape(6, 4),
        y.reshape(6),
        w,
    )

    assert jnp.allclose(loss, loss_ref, atol=1e-5, rtol=1e-5)


def test_fused_cross_entropy_reduction_and_penalty():
    x, w, y = _make_toy_inputs()
    weight = jnp.array([1.0, 0.0, 1.0, 1.0, 1.0, 0.0], dtype=jnp.float32)
    logsumexp_weight = 0.2
    logit_soft_cap = 1.7

    loss = fused_api.fused_cross_entropy_loss_and_logsumexp_penalty(
        x.reshape(6, 4),
        y.reshape(6),
        w,
        reduction="mean",
        weight=weight,
        logsumexp_weight=logsumexp_weight,
        logit_soft_cap=logit_soft_cap,
        implementation="xla",
    )

    loss_ref, lse_ref = linear_softmax_cross_entropy_loss_reference(
        x.reshape(6, 4),
        y.reshape(6),
        w,
        logit_soft_cap=logit_soft_cap,
    )
    loss_ref = loss_ref + logsumexp_weight * (lse_ref**2)
    loss_ref = (loss_ref * weight).sum() / weight.sum()

    assert jnp.allclose(loss, loss_ref, atol=1e-5, rtol=1e-5)


def test_fused_cross_entropy_grad_matches_reference():
    x, w, y = _make_toy_inputs()
    logsumexp_weight = 0.3

    def loss_api(x_raw, w_raw):
        return fused_api.fused_cross_entropy_loss_and_logsumexp_penalty(
            x_raw.reshape(6, 4),
            y.reshape(6),
            w_raw,
            reduction="mean",
            logsumexp_weight=logsumexp_weight,
            implementation="xla",
        )

    def loss_ref(x_raw, w_raw):
        loss_ref, lse_ref = linear_softmax_cross_entropy_loss_reference(
            x_raw.reshape(6, 4),
            y.reshape(6),
            w_raw,
        )
        loss_ref = loss_ref + logsumexp_weight * (lse_ref**2)
        return loss_ref.mean()

    gx_api, gw_api = jax.grad(loss_api, argnums=(0, 1))(x, w)
    gx_ref, gw_ref = jax.grad(loss_ref, argnums=(0, 1))(x, w)

    assert jnp.allclose(gx_api, gx_ref, atol=1e-5, rtol=1e-5)
    assert jnp.allclose(gw_api, gw_ref, atol=1e-5, rtol=1e-5)


def test_fused_cross_entropy_pallas_requires_tpu():
    if jax.default_backend() == "tpu":
        pytest.skip("requires non-TPU backend")

    x = jnp.zeros((128, 128), dtype=jnp.float32)
    w = jnp.zeros((128, 128), dtype=jnp.float32)
    y = jnp.zeros((128,), dtype=jnp.int32)

    with pytest.raises(pallas_tpu.PallasUnsupportedError):
        fused_api.fused_cross_entropy_loss_and_logsumexp_penalty(
            x,
            y,
            w,
            reduction=None,
            implementation="pallas_tpu",
        )


def test_fused_cross_entropy_default_matches_reference():
    backend = jax.default_backend()
    if backend == "tpu":
        batch, hidden, vocab = 128, 128, 256
        block_sizes = fused_api.BlockSizes(
            b_block_size=128,
            h_block_size=128,
            v_block_size=128,
        )
    else:
        batch, hidden, vocab = 32, 64, 128
        block_sizes = None

    key = jax.random.PRNGKey(0)
    key_x, key_w, key_y = jax.random.split(key, 3)

    x = jax.random.normal(key_x, (batch, hidden), dtype=jnp.float32)
    w = jax.random.normal(key_w, (hidden, vocab), dtype=jnp.float32)
    y = jax.random.randint(key_y, (batch,), 0, vocab, dtype=jnp.int32)

    logsumexp_weight = 0.2
    logit_soft_cap = 2.0

    loss = fused_api.fused_cross_entropy_loss_and_logsumexp_penalty(
        x,
        y,
        w,
        reduction=None,
        logsumexp_weight=logsumexp_weight,
        block_sizes=block_sizes,
        dtype=jnp.float32,
        logit_soft_cap=logit_soft_cap,
    )

    loss_ref, lse_ref = linear_softmax_cross_entropy_loss_reference(
        x,
        y,
        w,
        dtype=jnp.float32,
        logit_soft_cap=logit_soft_cap,
    )
    loss_ref = loss_ref + logsumexp_weight * (lse_ref**2)

    assert jnp.allclose(loss, loss_ref, atol=1e-4, rtol=1e-4)


def test_fused_cross_entropy_default_grad_matches_reference():
    backend = jax.default_backend()
    if backend == "tpu":
        batch, hidden, vocab = 128, 128, 256
        block_sizes = fused_api.BlockSizes(
            b_block_size=128,
            h_block_size=128,
            v_block_size=128,
        )
    else:
        batch, hidden, vocab = 16, 32, 64
        block_sizes = None

    key = jax.random.PRNGKey(0)
    key_x, key_w, key_y = jax.random.split(key, 3)

    x = jax.random.normal(key_x, (batch, hidden), dtype=jnp.float32)
    w = jax.random.normal(key_w, (hidden, vocab), dtype=jnp.float32)
    y = jax.random.randint(key_y, (batch,), 0, vocab, dtype=jnp.int32)

    logsumexp_weight = 0.2
    logit_soft_cap = 1.5

    def loss_default(x_raw, w_raw):
        return fused_api.fused_cross_entropy_loss_and_logsumexp_penalty(
            x_raw,
            y,
            w_raw,
            reduction="mean",
            logsumexp_weight=logsumexp_weight,
            block_sizes=block_sizes,
            dtype=jnp.float32,
            logit_soft_cap=logit_soft_cap,
        )

    def loss_ref(x_raw, w_raw):
        loss_val, lse_val = linear_softmax_cross_entropy_loss_reference(
            x_raw,
            y,
            w_raw,
            dtype=jnp.float32,
            logit_soft_cap=logit_soft_cap,
        )
        loss_val = loss_val + logsumexp_weight * (lse_val**2)
        return loss_val.mean()

    gx_default, gw_default = jax.grad(loss_default, argnums=(0, 1))(x, w)
    gx_ref, gw_ref = jax.grad(loss_ref, argnums=(0, 1))(x, w)

    assert jnp.allclose(gx_default, gx_ref, atol=1e-4, rtol=1e-4)
    assert jnp.allclose(gw_default, gw_ref, atol=1e-4, rtol=1e-4)
