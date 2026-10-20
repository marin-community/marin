# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""XLA versions of the restructured backward's two ``[T, D]`` elementwise steps.

``rstd`` is the per-row RMS scale, ``w`` the RMSNorm weight, ``g`` the gate logits and
``sigma = sigmoid(g)``. Every function computes in fp32 and returns its ``[T, D]`` output in the
cotangent's dtype.
"""

import jax
import jax.numpy as jnp


def dg_xla(x, dy, g, rstd, rms_weight):
    """Returns ``dg = dy * n * sigma * (1 - sigma)`` and the row sums of ``dy * sigma * n``."""
    gate = jax.nn.sigmoid(g.astype(jnp.float32))
    n = x.astype(jnp.float32) * rstd[:, None] * rms_weight.astype(jnp.float32)
    dy_gate_n = dy.astype(jnp.float32) * gate * n
    return (dy_gate_n * (1.0 - gate)).astype(dy.dtype), jnp.sum(dy_gate_n, axis=-1)


def dx_xla(x, dy, g, dn_bottleneck, rstd, c, rms_weight):
    """Returns ``dx`` and the column sums of ``dn * xhat``, with ``dn = dy * sigma + dn_bottleneck``.

    ``c`` is the RMSNorm backward's row sum ``sum_d dn_d * n_d``.
    """
    gate = jax.nn.sigmoid(g.astype(jnp.float32))
    xhat = x.astype(jnp.float32) * rstd[:, None]
    dn = dy.astype(jnp.float32) * gate + dn_bottleneck.astype(jnp.float32)
    dx = rstd[:, None] * (dn * rms_weight.astype(jnp.float32) - xhat * (c[:, None] / x.shape[-1]))
    return dx.astype(dy.dtype), jnp.sum(dn * xhat, axis=0)
