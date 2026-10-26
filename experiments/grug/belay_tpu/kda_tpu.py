# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Ckpt 1 KDA with selectable preparation and a TPU state kernel.

Inputs are fully device-local heads-first arrays. The model owns the enclosing
shard_map, so batch/heads are partitioned and the sequence remains local.
Backward reads saved FP32 states and corrected values. Fused preparation also
saves prefixes and solve factors; it rebuilds elementwise factors in VMEM.
"""

import functools

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int

from experiments.grug.belay_tpu.kda_chunk_prep import chunk_prep_midpoint
from experiments.grug.belay_tpu.kda_fused_prep_tpu import chunk_prep_fused
from experiments.grug.belay_tpu.kda_state_tiled_tpu import state_pass_tiled
from experiments.grug.belay_tpu.kda_state_tpu import state_pass
from experiments.grug.belay_tpu.kda_triangular_tpu import chunk_prep_solve
from experiments.grug.moe.kda import _chunk_prep_xla, _prepare_qk, _segment_masks, doc_starts


def chunk_kda_tpu(
    q: Float[Array, "B H L D"],
    k: Float[Array, "B H L D"],
    v: Float[Array, "B H L D"],
    g: Float[Array, "B H L D"],
    beta: Float[Array, "B H L"],
    *,
    segment_ids: Int[Array, "B H L"] | None = None,
    erase: Float[Array, "B H L D"] | None = None,
    initial_state: jax.Array | None = None,
    matmul_dtype: jnp.dtype = jnp.bfloat16,
    state_tile: int = 1,
    chunk_size: int = 16,
    solve_groups: int = 0,
    fused_prep: bool = False,
    state_precision: jax.lax.Precision = jax.lax.Precision.HIGHEST,
) -> tuple[jax.Array, jax.Array]:
    """Evaluate packed KDA with a 128-wide saved state and bounded decay tiles."""
    if q.ndim != 4 or q.shape[-1] != 128 or any(x.shape != q.shape for x in (k, v, g)):
        raise ValueError("Ckpt 1 TPU KDA requires matching [B,H,L,128] q/k/v/g")
    if matmul_dtype not in (jnp.float32, jnp.bfloat16):
        raise ValueError("KDA intra-chunk dtype must be FP32 or BF16")
    q, k = _prepare_qk(q, k, True)
    v, g, beta = (x.astype(jnp.float32) for x in (v, g, beta))
    b, h, length, d = q.shape
    if chunk_size not in (16, 32, 64, 128) or (chunk_size != 16 and state_tile != 1):
        raise ValueError("Larger KDA chunks require state_tile=1 and size 32, 64 or 128")
    c = chunk_size
    pad = (-length) % c

    def padded(x):
        return jnp.pad(x, ((0, 0), (0, 0), (0, pad), (0, 0))) if pad else x

    q, k, v, g = (padded(x) for x in (q, k, v, g))
    beta = jnp.pad(beta, ((0, 0), (0, 0), (0, pad))) if pad else beta
    n = q.shape[-2] // c
    masks = None
    if segment_ids is not None:
        ids = jnp.broadcast_to(segment_ids, (b, h, length))
        if pad:
            ids = jnp.pad(ids, ((0, 0), (0, 0), (0, pad)), mode="edge")
        masks = _segment_masks(doc_starts(ids).reshape(b, h, n, c))

    def chunks(x):
        return x.reshape(b, h, n, c, d)

    def mm(spec, left, right):
        return jnp.einsum(spec, left.astype(matmul_dtype), right.astype(matmul_dtype)).astype(jnp.float32)

    qc, kc, vc, gc = (chunks(x) for x in (q, k, v, g))
    ec = kc if erase is None else kc * chunks(padded(erase.astype(jnp.float32)))
    prepare = _chunk_prep_xla if c == 16 else chunk_prep_midpoint
    if fused_prep and not solve_groups:
        raise ValueError("Fused preparation requires one/four solve groups")
    if solve_groups:
        prepare = functools.partial(chunk_prep_fused if fused_prep else chunk_prep_solve, groups=solve_groups)
    prep = prepare(qc, kc, ec, vc, gc, beta.reshape(b, h, n, c), mm, matmul_dtype, c, masks)
    initial = jnp.zeros((b * h, d, d), jnp.float32)
    if initial_state is not None:
        initial = initial_state.astype(jnp.float32).reshape(b * h, d, d)

    def flatten(x):
        return x.reshape(b * h, n, *x.shape[-2:])

    state_kernel = state_pass if state_tile == 1 else functools.partial(state_pass_tiled, tile=state_tile)
    out, final = state_kernel(
        flatten(prep.q_inflate),
        flatten(prep.attn),
        flatten(prep.kw),
        flatten(prep.k_cumdecay),
        flatten(prep.v_pseudo),
        prep.decay.reshape(b * h, n, 1, d),
        initial,
        precision=state_precision,
        matmul_dtype=matmul_dtype,
    )
    return out.reshape(b, h, n * c, d)[..., :length, :], final.reshape(b, h, d, d)
