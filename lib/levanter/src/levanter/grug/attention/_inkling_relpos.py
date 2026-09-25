# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0
"""Inkling (Thinking Machines) relative-position bias helpers for FA4/CuTe attention.

The bias is a per-head, content-dependent additive term on the pre-softmax logits:
``bias[b,h,i,j] = R[b,h,i,:] . proj[:, i-j]`` for ``0 <= i-j < rel_extent`` (else 0).

Attention consumes it in a *banded, key-aligned* layout ``[B, H, S, W]`` with
``W = rel_extent + REL_BIAS_BLOCK``: queries are grouped in blocks of ``REL_BIAS_BLOCK`` rows, and
row ``i`` (block start ``i0 = i // REL_BIAS_BLOCK * REL_BIAS_BLOCK``) stores the bias for keys
``j = i0 - rel_extent + x`` at column ``x``::

    bias[b,h,i,j] = band[b,h,i, j - i0 + rel_extent]   if 0 <= j - i0 + rel_extent < W, else 0

Every query row of a block shares the same key window, so a (query tile, key tile) pair reads a
dense, aligned sub-tile of ``band`` -- no per-element distance gather or band test in the kernels.
Distances outside ``[0, rel_extent)`` are stored as zeros by :func:`inkling_rel_bias`.

The compact form ``A[b,h,i,delta]`` (``delta = i - j``) is kept for the non-SM90 fallback backward:
:func:`rel_bias_backward` computes ``dA`` as a banded gather of dScore:

    dA[b,h,i,delta] = P[b,h,i,i-delta] * (dO[b,h,i] . V[b,h,i-delta] - O[b,h,i] . dO[b,h,i])

for ``0 <= delta < rel_extent`` and ``i-delta`` in-mask (else 0). ``P`` is reconstructed from the
softmax LSE the forward kernel returns, so no ``[S,S]`` matrix is materialized.
"""

import jax.numpy as jnp
from jax.sharding import PartitionSpec
from jaxtyping import Array, Bool, Float, Int

# Query-block granularity of the banded bias layout. Kernel query tiles must divide it, and kernel
# key tiles must divide both it and rel_extent, so every visited (query tile, key tile) pair maps to
# a whole aligned sub-tile of the band.
REL_BIAS_BLOCK = 128


def rel_bias_band_width(rel_extent: int) -> int:
    """Width ``W`` of the banded bias layout for a given ``rel_extent``."""
    return rel_extent + REL_BIAS_BLOCK


def rel_extent_of_band(band: Float[Array, "B H S W"]) -> int:
    """Inverse of :func:`rel_bias_band_width` for a banded bias tensor."""
    rel_extent = band.shape[-1] - REL_BIAS_BLOCK
    if rel_extent <= 0:
        raise ValueError(f"banded bias width {band.shape[-1]} must exceed REL_BIAS_BLOCK={REL_BIAS_BLOCK}")
    return rel_extent


def _band_distance(rel_extent: int) -> tuple[Int[Array, "T W"], Bool[Array, "T W"]]:
    """Distance ``delta = i - j`` at (row phase p = i - i0, band column x), and whether it is in band."""
    phase = jnp.arange(REL_BIAS_BLOCK)[:, None]
    column = jnp.arange(rel_bias_band_width(rel_extent))[None, :]
    delta = phase + rel_extent - column
    return delta, (delta >= 0) & (delta < rel_extent)


def _skewed_bank(proj: Float[Array, "R L"]) -> Float[Array, "R T W"]:
    """Per-phase bank ``bank[r, p, x] = proj[r, p + L - x]`` for in-band distances, else 0.

    Built with reverse/pad/reshape/slice only (row ``p`` is the reversed bank shifted right by
    ``p + 1``), so its transpose is also pure data movement -- a gather here would make the bank
    gradient a heavily contended scatter-add.
    """
    rank, rel_extent = proj.shape
    width = rel_bias_band_width(rel_extent)
    # Reversed bank, zero-padded to width W + 1: z[r, j] = proj[r, L - 1 - j] for j < L.
    z = jnp.pad(proj[:, ::-1], ((0, 0), (0, width + 1 - rel_extent)))
    # Row p of the flattened [T, W + 1] tiling, read with row stride W and a one-element lead-in, is z
    # shifted right by p + 1; the wrap-around reads land in z's zero padding.
    flat = jnp.broadcast_to(z[:, None, :], (rank, REL_BIAS_BLOCK, width + 1)).reshape(rank, -1)
    flat = jnp.pad(flat, ((0, 0), (1, 0)))[:, : REL_BIAS_BLOCK * width]
    return flat.reshape(rank, REL_BIAS_BLOCK, width)


def inkling_rel_bias(
    relative_states: Float[Array, "B H S R"],
    proj: Float[Array, "R L"],
    *,
    out_sharding: PartitionSpec | None = None,
) -> Float[Array, "B H S W"]:
    """Banded Inkling bias ``band[b,h,i,x] = R[b,h,i,:] . proj[:, i - j(x)]`` (0 outside the band).

    Computed directly in the banded layout by one batched contraction against a per-phase skewed
    copy of the (tiny) bank, so the ``[B,H,S,W]`` tensor is written exactly once.
    """
    b, h, s, r = relative_states.shape
    rel_extent = proj.shape[1]
    if s % REL_BIAS_BLOCK != 0:
        raise ValueError(f"sequence length {s} must be a multiple of REL_BIAS_BLOCK={REL_BIAS_BLOCK}")
    bank = _skewed_bank(proj)
    blocked = relative_states.reshape(b, h, s // REL_BIAS_BLOCK, REL_BIAS_BLOCK, r)
    out_sharding_5d = None if out_sharding is None else PartitionSpec(*out_sharding[:3], None, out_sharding[3])
    band = jnp.einsum("bhtpr,rpw->bhtpw", blocked, bank, out_sharding=out_sharding_5d)
    return band.reshape(b, h, s, rel_bias_band_width(rel_extent))


def dense_rel_bias(band: Float[Array, "B H Q W"], q_len: int, k_len: int) -> Float[Array, "B H Q K"]:
    """Expand a banded bias to the dense ``[B,H,Q,K]`` additive logit term (reference path)."""
    rel_extent = rel_extent_of_band(band)
    i_idx = jnp.arange(q_len)[:, None]
    j_idx = jnp.arange(k_len)[None, :]
    column = j_idx - (i_idx // REL_BIAS_BLOCK) * REL_BIAS_BLOCK + rel_extent  # [Q, K]
    valid = (column >= 0) & (column < band.shape[-1])
    gather_idx = jnp.broadcast_to(jnp.clip(column, 0, band.shape[-1] - 1), band.shape[:2] + (q_len, k_len))
    gathered = jnp.take_along_axis(band, gather_idx, axis=-1)
    return jnp.where(valid, gathered, jnp.zeros((), dtype=band.dtype))


def band_to_compact(band: Float[Array, "B H S W"]) -> Float[Array, "B H S L"]:
    """Compact distance-indexed ``A[b,h,i,delta]`` from a banded bias (in-band entries only)."""
    b, h, s, w = band.shape
    rel_extent = rel_extent_of_band(band)
    phase = jnp.arange(s)[:, None] % REL_BIAS_BLOCK
    column = phase + rel_extent - jnp.arange(rel_extent)[None, :]  # [S, L], always in [1, W)
    return jnp.take_along_axis(band, jnp.broadcast_to(column, (b, h, s, rel_extent)), axis=-1)


def compact_to_band(compact: Float[Array, "B H S L"]) -> Float[Array, "B H S W"]:
    """Banded layout from a compact ``A[b,h,i,delta]`` (zeros outside the band)."""
    b, h, s, rel_extent = compact.shape
    delta, in_band = _band_distance(rel_extent)
    phase = jnp.arange(s) % REL_BIAS_BLOCK
    idx = jnp.broadcast_to(jnp.clip(delta, 0, rel_extent - 1)[phase], (b, h, s, rel_bias_band_width(rel_extent)))
    gathered = jnp.take_along_axis(compact, idx, axis=-1)
    return jnp.where(in_band[phase], gathered, jnp.zeros((), dtype=compact.dtype))


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
    # GQA: k/v carry Hkv <= Hq heads; expand to Hq so each query head uses its shared kv head.
    hkv = kf.shape[1]
    if hkv != h:
        if h % hkv != 0:
            raise ValueError(f"num q heads {h} must be a multiple of kv heads {hkv}")
        kf = jnp.repeat(kf, h // hkv, axis=1)
        vf = jnp.repeat(vf, h // hkv, axis=1)
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
