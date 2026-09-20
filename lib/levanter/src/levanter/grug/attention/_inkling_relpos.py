# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0
"""Inkling (Thinking Machines) relative-position bias helpers for FA4/CuTe attention.

The bias is a per-head, content-dependent additive term on the pre-softmax logits:
``bias[b,h,i,j] = A[b,h,i, i-j]`` for ``0 <= i-j < rel_extent`` (else 0), where ``A`` is the compact
``[B, H, S, rel_extent]`` tensor produced by ``r_proj @ proj`` in the model.

Forward: the fused kernel adds ``A`` (gathered by distance) to the score tile.
Backward: dQ/dK/dV come from the fused flash backward with the same bias added to its S-recompute; the
bias gradient ``dA`` is a *banded gather of dScore*:

    dA[b,h,i,delta] = P[b,h,i,i-delta] * (dO[b,h,i] . V[b,h,i-delta] - O[b,h,i] . dO[b,h,i])

for ``0 <= delta < rel_extent`` and ``i-delta`` in-mask (else 0). ``P`` is reconstructed from the
softmax LSE the forward kernel returns, so no ``[S,S]`` matrix is materialized. ``rel_bias_backward``
computes this in ``block x block`` tiles over the band only (``rel_extent/block`` key-offset tiles),
which is far faster than a per-distance scan and never forms an ``[S,S]`` matrix. Verified against
autodiff in the tests.
"""

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
    block: int = 128,
) -> Float[Array, "B H S L"]:
    """dA (bias gradient), computed blockwise over the banded region only.

    P[i,i-delta] = exp(qk*scale + A[i,delta] - lse). Tiles are ``[block, block]``; the band spans
    ``rel_extent/block`` key-offset tiles, so cost ~ (rel_extent/S) of a full attention pass.
    """
    b, h, s, d = q.shape
    rel_extent = rel_bias.shape[-1]
    if s % block != 0 or rel_extent % block != 0:
        raise ValueError(f"S={s} and rel_extent={rel_extent} must be multiples of block={block}")
    nqb = s // block
    # +1: a distance up to rel_extent-1 at a block's first row reaches one extra previous key-block.
    noff = rel_extent // block + 1

    qf, kf, vf, dof, af = (t.astype(jnp.float32) for t in (q, k, v, d_out, rel_bias))
    delta_i = jnp.sum(out.astype(jnp.float32) * dof, axis=-1)  # [B,H,S]

    qb = qf.reshape(b, h, nqb, block, d)
    dob = dof.reshape(b, h, nqb, block, d)
    kblk = kf.reshape(b, h, nqb, block, d)
    vblk = vf.reshape(b, h, nqb, block, d)
    ablk = af.reshape(b, h, nqb, block, rel_extent)  # A[b,h, qb*block+x, delta]
    lse_b = lse.reshape(b, h, nqb, block)
    di_b = delta_i.reshape(b, h, nqb, block)
    lb_b = lower_bounds.reshape(b, nqb, block)  # [B,nqb,block]
    val_b = valid.reshape(b, nqb, block)

    x = jnp.arange(block)
    y = jnp.arange(block)
    delta_xy = block * 0 + x[:, None] - y[None, :]  # base; offset added per o -> [block,block]

    da_blk = jnp.zeros((b, h, nqb, block, rel_extent), jnp.float32)
    b_i = jnp.arange(b)[:, None, None, None, None]
    h_i = jnp.arange(h)[None, :, None, None, None]
    qb_i = jnp.arange(nqb)[None, None, :, None, None]
    x_i = x[None, None, None, :, None]

    for o in range(noff):
        # key/value blocks shifted so query-block qb reads key-block (qb - o); qb < o is invalid.
        k_o = jnp.roll(kblk, o, axis=2)
        v_o = jnp.roll(vblk, o, axis=2)
        score = jnp.einsum("bhqxd,bhqyd->bhqxy", qb, k_o) * softmax_scale  # [B,H,nqb,block,block]
        delta = o * block + delta_xy  # [block,block]
        in_band = (delta >= 0) & (delta < rel_extent)
        gdelta = jnp.clip(delta, 0, rel_extent - 1)
        a_g = jnp.take_along_axis(
            ablk, jnp.broadcast_to(gdelta[None, None, None, :, :], (b, h, nqb, block, block)), axis=-1
        )
        score = score + jnp.where(in_band, a_g, 0.0)

        k_glob = (jnp.arange(nqb)[:, None] - o) * block + y[None, :]  # [nqb,block]
        qb_ok = (jnp.arange(nqb) >= o)[None, :, None, None]  # [1,nqb,1,1]
        # key j valid: lb_q <= k <= q, query valid; broadcast over heads.
        key_ok = (
            in_band[None, None]  # [1,1,block,block]
            & qb_ok  # [1,nqb,1,1]
            & val_b[:, None, :, :, None]  # [B,1,nqb,block,1]
            & (k_glob[None, None, :, None, :] >= lb_b[:, None, :, :, None])  # k>=lb_q
        )  # -> [B,1(H),nqb,block,block]
        key_ok = jnp.broadcast_to(key_ok, (b, h, nqb, block, block))

        score = jnp.where(key_ok, score, -jnp.inf)
        p = jnp.exp(score - lse_b[..., None])
        dov = jnp.einsum("bhqxd,bhqyd->bhqxy", dob, v_o)
        dscore = jnp.where(key_ok, p * (dov - di_b[..., None]), 0.0)  # [B,H,nqb,block,block]

        da_blk = da_blk.at[b_i, h_i, qb_i, x_i, gdelta[None, None, None, :, :]].add(dscore.astype(da_blk.dtype))

    return da_blk.reshape(b, h, s, rel_extent)
