# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import jax
import jax.numpy as jnp

from levanter.kernels.pallas.template_kernel import reference_impl_batched, template_op


def test_template_kernel_matches_reference():
    x = jnp.linspace(-3.0, 3.0, 1024, dtype=jnp.float32)
    y_ref = reference_impl_batched(x[None, :])[0]
    y_fast = template_op(x)
    assert jnp.allclose(y_ref, y_fast, atol=0.0, rtol=0.0)

    xb = x.reshape(32, 32)
    yb_ref = reference_impl_batched(xb)
    yb_fast = template_op(xb)
    assert jnp.allclose(yb_ref, yb_fast, atol=0.0, rtol=0.0)


def test_template_kernel_grad_matches_reference():
    x = jnp.linspace(-1.0, 1.0, 128, dtype=jnp.float32)

    def loss_ref(v):
        return jnp.sum(reference_impl_batched(v[None, :])[0])

    def loss_fast(v):
        return jnp.sum(template_op(v))

    g_ref = jax.grad(loss_ref)(x)
    g_fast = jax.grad(loss_fast)(x)
    assert jnp.allclose(g_ref, g_fast, atol=0.0, rtol=0.0)

    xb = x.reshape(8, 16)

    def loss_ref_batched(v):
        return jnp.sum(reference_impl_batched(v))

    def loss_fast_batched(v):
        return jnp.sum(template_op(v))

    gb_ref = jax.grad(loss_ref_batched)(xb)
    gb_fast = jax.grad(loss_fast_batched)(xb)
    assert jnp.allclose(gb_ref, gb_fast, atol=0.0, rtol=0.0)
