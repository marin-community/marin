# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0
"""Inkling (Thinking Machines) relative-position bias helpers for FA4/CuTe attention.

The bias is a per-head, content-dependent additive term on the pre-softmax logits:
``bias[b,h,i,j] = A[b,h,i, i-j]`` for ``0 <= i-j < rel_extent`` (else 0), where ``A`` is the compact
``[B, H, S, rel_extent]`` tensor produced by ``r_proj @ proj`` in the model.

Forward: the fused kernel adds ``A`` (gathered by distance) to the score tile.
Backward: dQ/dK/dV come from the flash backward with the same bias added to its S-recompute; the bias
gradient ``dA`` is a *banded gather of dScore*:

    dA[b,h,i,delta] = P[b,h,i,i-delta] * (dO[b,h,i] . V[b,h,i-delta] - O[b,h,i] . dO[b,h,i])

for ``0 <= delta < rel_extent`` and ``i-delta`` in-mask (else 0). ``P`` is reconstructed from the
softmax LSE the forward kernel already returns, so no ``[S,S]`` matrix is materialized. This identity
is verified against autodiff in the tests.
"""

import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int


def rel_bias_backward(
    q: Float[Array, "B H S D"],
    k: Float[Array, "B H S D"],
    v: Float[Array, "B H S Dv"],
    out: Float[Array, "B H S Dv"],
    d_out: Float[Array, "B H S Dv"],
    lse: Float[Array, "B H S"],
    rel_bias: Float[Array, "B H S L"],
    lower_bounds: Int[Array, "B S"],
    valid: Bool[Array, "B S"],
    *,
    softmax_scale: float,
) -> Float[Array, "B H S L"]:
    """dA including the bias term in the reconstructed P: P[i,i-delta] = exp(qk*scale + A[i,delta] - lse)."""
    s = q.shape[2]
    rel_extent = rel_bias.shape[-1]
    delta_i = jnp.sum(out.astype(jnp.float32) * d_out.astype(jnp.float32), axis=-1)  # [B,H,S]
    qf, kf, vf, dof = (t.astype(jnp.float32) for t in (q, k, v, d_out))
    af = rel_bias.astype(jnp.float32)  # [B,H,S,L]
    i_idx = jnp.arange(s)

    def one_delta(_carry, delta):
        j = i_idx - delta
        in_range = j >= 0
        jc = jnp.clip(j, 0, s - 1)
        k_sh = jnp.take_along_axis(kf, jnp.broadcast_to(jc[None, None, :, None], kf.shape), axis=2)
        v_sh = jnp.take_along_axis(vf, jnp.broadcast_to(jc[None, None, :, None], vf.shape), axis=2)
        qk = jnp.sum(qf * k_sh, axis=-1) * softmax_scale  # [B,H,S]
        a_delta = af[:, :, :, delta]  # [B,H,S] = A[b,h,i,delta]
        key_ok = (jc >= lower_bounds) & (j[None, :] <= i_idx[None, :]) & valid & in_range[None, :]
        score = jnp.where(key_ok[:, None, :], qk + a_delta, -jnp.inf)
        p = jnp.exp(score - lse)
        dov = jnp.sum(dof * v_sh, axis=-1)
        da = jnp.where(key_ok[:, None, :], p * (dov - delta_i), 0.0)
        return _carry, da.astype(jnp.float32)

    _, da_stack = jax.lax.scan(one_delta, None, jnp.arange(rel_extent))
    return jnp.transpose(da_stack, (1, 2, 3, 0))  # [B,H,S,L]
