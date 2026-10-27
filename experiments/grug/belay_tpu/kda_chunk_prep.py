# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Prepare larger KDA chunks using bounded, document-local decay references."""

import jax.numpy as jnp
from jax import lax

from experiments.grug.moe.kda import _unit_lower_triangular_inverse
from experiments.grug.moe.kda_prep_pallas import DEFLATE_EXP_CAP, ChunkPrep


def chunk_prep_midpoint(qc, kc, ec, vc, gc, bc, mm, mm_dtype, c, masks):
    """Build chunk transforms with the source GPU kernel's 32-token decay tiles.

    Within each tile, reference cumulative decay at its middle token. Documents
    that end before that point use the previous tile boundary or their own start.
    A log-decay in [-5, 0] then needs at most exp(80), independent of chunk size.
    This changes reduction order while preserving the KDA recurrence and masks.
    """
    if c not in (32, 64, 128):
        raise ValueError("Midpoint preparation requires chunks of 32, 64 or 128")
    lower = jnp.tril(jnp.ones((c, c), dtype=bool))
    same = None if masks is None else masks.same
    if same is None:
        cumulative = jnp.cumsum(gc, axis=-2)
    else:
        cumulative = jnp.einsum(
            "...tj,...jd->...td", (lower & same).astype(jnp.float32), gc, precision=lax.Precision.HIGHEST
        )
    beta_keys = ec * bc[..., None]
    a_blocks, attention_blocks = [], []
    for start in range(0, c, 32):
        stop, middle = start + 32, start + 15
        reference = jnp.broadcast_to(cumulative[..., middle : middle + 1, :], cumulative.shape)
        if same is not None:
            reference = jnp.where(same[..., :, middle, None], reference, 0.0)
            if start:
                carried = same[..., :, start - 1] & ~same[..., :, middle]
                reference = jnp.where(carried[..., None], cumulative[..., start - 1 : start, :], reference)
        row_factor = jnp.exp(
            jnp.minimum(cumulative[..., start:stop, :] - reference[..., start:stop, :], DEFLATE_EXP_CAP)
        )
        column_keys = kc * jnp.exp(jnp.minimum(reference - cumulative, DEFLATE_EXP_CAP))
        a_blocks.append(-mm("...rd,...id->...ri", beta_keys[..., start:stop, :] * row_factor, column_keys))
        attention_blocks.append(mm("...rd,...id->...ri", qc[..., start:stop, :] * row_factor, column_keys))

    strict = jnp.tril(jnp.ones((c, c), dtype=bool), k=-1)
    if same is not None:
        strict, lower = strict & same, lower & same
    a = jnp.where(strict, jnp.concatenate(a_blocks, axis=-2), 0.0)
    inverse = _unit_lower_triangular_inverse(a.reshape(-1, c, c), c, matmul_dtype=mm_dtype).reshape(a.shape)
    attention = jnp.where(lower, jnp.concatenate(attention_blocks, axis=-2), 0.0)
    inflate = jnp.exp(cumulative)
    qi, state_keys = qc * inflate, beta_keys * inflate
    tail = cumulative[..., -1, :]
    kw = kc * jnp.exp(jnp.minimum(tail[..., None, :] - cumulative, 0.0))
    decay = jnp.exp(tail)
    if masks is not None:
        qi = qi * masks.carry_in[..., None]
        state_keys = state_keys * masks.carry_in[..., None]
        kw = kw * masks.last_doc[..., None]
        decay = decay * masks.carry_through[..., None]
    return ChunkPrep(
        qi,
        mm("...rj,...jd->...rd", inverse, state_keys),
        mm("...rj,...jd->...rd", inverse, vc * bc[..., None]),
        attention,
        kw,
        decay,
    )
