# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Reference Parallax attention (arXiv 2605.29157), the fp32 oracle of the official ``parallax/reference.py``.

With ``p_ij`` the causal softmax weights of ``q_i . k_j * scale`` and ``s_ij = r_i . k_j`` (unscaled):
``o_i = sum_j p_ij v_j (1 + E_p[s_i]) - sum_j p_ij s_ij v_j``, softmax attention minus the attention-weighted
key-value covariance probed by ``r_i``.
"""

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int


def parallax_reference(
    q: Float[Array, "B L Hq D"],
    r: Float[Array, "B L Hq D"],
    k: Float[Array, "B L Hkv D"],
    v: Float[Array, "B L Hkv D"],
    segment_ids: Int[Array, "B L"] | None = None,
    *,
    sm_scale: float,
) -> Float[Array, "B L Hq D"]:
    """Causal (and same-segment) Parallax in fp32; ``Hq`` must be a multiple of ``Hkv`` (GQA)."""
    reps = q.shape[2] // k.shape[2]
    if q.shape[2] != reps * k.shape[2]:
        raise ValueError(f"query heads {q.shape[2]} must be a multiple of kv heads {k.shape[2]}")
    k = jnp.repeat(k, reps, axis=2).astype(jnp.float32)
    v = jnp.repeat(v, reps, axis=2).astype(jnp.float32)
    q, r = q.astype(jnp.float32), r.astype(jnp.float32)
    seq = q.shape[1]
    mask = jnp.tril(jnp.ones((seq, seq), bool))[None, None]
    if segment_ids is not None:
        mask = mask & (segment_ids[:, None, :, None] == segment_ids[:, None, None, :])
    s1 = jnp.einsum("bqhd,bkhd->bhqk", q, k) * sm_scale
    p = jax.nn.softmax(jnp.where(mask, s1, -jnp.inf), axis=-1)
    ps = p * jnp.einsum("bqhd,bkhd->bhqk", r, k)
    o1 = jnp.einsum("bhqk,bkhd->bqhd", p, v)
    o2 = jnp.einsum("bhqk,bkhd->bqhd", ps, v)
    mean_s = jnp.sum(ps, axis=-1).transpose(0, 2, 1)[..., None]
    return o1 * (1.0 + mean_s) - o2
