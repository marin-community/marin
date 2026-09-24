# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Kimi Delta Attention (KDA) kernels: the gated delta rule with a per-channel
forget gate, in the chunked-parallel form that keeps the linear-attention layers
matmul-bound (good MFU) instead of a token-by-token recurrence.

Two entry points, both operating on plain JAX arrays with arbitrary leading batch
dims ``(..., L, d)`` (in grug the leading dims are ``(batch, heads)``):

  - ``recurrent_kda``: the sequential reference (one token at a time). Correct but
    O(L) sequential steps -- used as the numerical oracle and for decode.
  - ``chunk_kda``: the chunkwise-parallel kernel used for training/prefill. Splits
    the sequence into chunks of ``chunk_size``; within a chunk everything is dense
    matmuls plus one unit-lower-triangular inverse (log-depth Neumann product),
    across chunks a single recurrent state ``S`` is carried. This is the
    MFU-friendly form.

State ``S ∈ R^{d_k x d_v}`` maps keys to values; the read is ``o_t = S_t^T q_t`` and
the per-channel decay ``alpha_t = exp(g_t) ∈ (0,1]^{d_k}`` left-multiplies ``S`` on
the key axis. KDA is the per-channel generalization of scalar Gated DeltaNet
(``levanter.layers.gated_deltanet``); in the scalar limit (``g`` broadcast over
``d_k``) the two agree. All decay math is done in fp32.

References: Kimi Linear tech report (arXiv 2510.26692), Gated Delta Networks
(arXiv 2412.06464), Parallelizing Linear Transformers with the Delta Rule
(arXiv 2406.06484), and flash-linear-attention ``fla/ops/kda/naive.py``.
"""

import os

import jax
import jax.numpy as jnp
from jax import lax
from jaxtyping import Array, Float

# Benchmark-only toggle: KDA_NAIVE_INVERSE=1 selects the old C-step serial forward
# substitution for the intra-chunk inverse instead of the log-depth Neumann product,
# so a live A/B isolates the kernel change with everything else held fixed.
_KDA_NAIVE_INVERSE = os.environ.get("KDA_NAIVE_INVERSE") == "1"

_L2NORM_EPS = 1e-6
# Cap the (single-sided) deflation exponent exp(-cumdecay) so extreme per-channel
# decay cannot overflow fp32. The pairwise decay exp(b_r - b_i) it participates in
# is <= 1, so the cap only bites where a channel has already decayed to ~0 (a
# negligible contribution); for realistic trained gates it never triggers.
_DEFLATE_EXP_CAP = 30.0


def _l2norm(x: jax.Array, eps: float = _L2NORM_EPS) -> jax.Array:
    """L2-normalize along the last axis (in fp32)."""
    x32 = x.astype(jnp.float32)
    inv = lax.rsqrt(jnp.sum(x32 * x32, axis=-1, keepdims=True) + eps)
    return x32 * inv


def _prepare_qk(q: jax.Array, k: jax.Array, use_qk_l2norm: bool) -> tuple[jax.Array, jax.Array]:
    """Optionally L2-normalize q,k, then scale q by 1/sqrt(d_k)."""
    q = q.astype(jnp.float32)
    k = k.astype(jnp.float32)
    if use_qk_l2norm:
        q = _l2norm(q)
        k = _l2norm(k)
    q = q * (q.shape[-1] ** -0.5)
    return q, k


def recurrent_kda(
    q: Float[Array, "... L Dk"],
    k: Float[Array, "... L Dk"],
    v: Float[Array, "... L Dv"],
    g: Float[Array, "... L Dk"],
    beta: Float[Array, "... L"],
    *,
    initial_state: jax.Array | None = None,
    use_qk_l2norm: bool = True,
) -> tuple[Float[Array, "... L Dv"], jax.Array]:
    """Sequential per-channel gated delta rule (reference / decode kernel).

    Args:
        q, k: ``(..., L, d_k)`` queries/keys.
        v: ``(..., L, d_v)`` values.
        g: ``(..., L, d_k)`` per-channel log-decay (``alpha = exp(g)``, ``g <= 0``).
        beta: ``(..., L)`` per-head write strength in ``(0, 1)``.
        initial_state: optional ``(..., d_k, d_v)`` starting state.
        use_qk_l2norm: L2-normalize q,k inside the kernel (and scale q by 1/sqrt(d_k)).

    Returns:
        ``(outputs (..., L, d_v), final_state (..., d_k, d_v))``.
    """
    q, k = _prepare_qk(q, k, use_qk_l2norm)
    v = v.astype(jnp.float32)
    g = g.astype(jnp.float32)
    beta = beta.astype(jnp.float32)

    lead = q.shape[:-2]
    dk, dv = q.shape[-1], v.shape[-1]
    if initial_state is None:
        state = jnp.zeros((*lead, dk, dv), dtype=jnp.float32)
    else:
        state = initial_state.astype(jnp.float32)

    # Move the length axis to the front for lax.scan.
    q_t = jnp.moveaxis(q, -2, 0)  # (L, ..., d_k)
    k_t = jnp.moveaxis(k, -2, 0)
    v_t = jnp.moveaxis(v, -2, 0)
    g_t = jnp.moveaxis(g, -2, 0)
    b_t = jnp.moveaxis(beta, -1, 0)  # (L, ...)

    def step(s_prev: jax.Array, inp):
        q_i, k_i, v_i, g_i, b_i = inp
        s_prev = s_prev * jnp.exp(g_i)[..., :, None]  # Diag(alpha) on the key axis
        kv = jnp.sum(k_i[..., :, None] * s_prev, axis=-2)  # S^T k  -> (..., d_v)
        delta = (v_i - kv) * b_i[..., None]
        s_new = s_prev + k_i[..., :, None] * delta[..., None, :]
        o_i = jnp.sum(q_i[..., :, None] * s_new, axis=-2)  # S^T q -> (..., d_v)
        return s_new, o_i

    state, out_t = lax.scan(step, state, (q_t, k_t, v_t, g_t, b_t))
    out = jnp.moveaxis(out_t, 0, -2)  # (..., L, d_v)
    return out, state


def _unit_lower_triangular_inverse(
    a_strict_lower: jax.Array, chunk_size: int, matmul_dtype: jnp.dtype | None = None
) -> jax.Array:
    """Invert a unit-lower-triangular ``(I - A)`` given strictly-lower ``A``.

    Returns ``T = (I - A)^{-1}`` (unit lower-triangular). ``a_strict_lower`` has shape
    ``(..., C, C)`` and is strictly lower (zero diagonal/upper), hence nilpotent
    (``A^C = 0``). The inverse is the finite Neumann series, evaluated as the log-depth
    product ``(I + A)(I + A^2)(I + A^4) ... (I + A^{2^{m-1}})`` with ``2^m >= C``:
    ``2*(ceil(log2 C) - 1)`` batched ``C x C`` matmuls, all parallel over the batch. This
    is the matmul-bound (WY/UT-transform) form used by flash-linear-attention -- both
    forward and (reverse-mode) backward are a handful of dense matmuls, unlike the
    ``C``-step serial forward substitution, whose backward is especially expensive.

    ``matmul_dtype`` (e.g. ``bfloat16``) runs the products in that dtype (Hopper
    tensor cores) while carrying the running series in fp32; ``None`` keeps fp32.
    """
    out_dtype = a_strict_lower.dtype

    def mm(x: jax.Array, y: jax.Array) -> jax.Array:
        if matmul_dtype is None:
            return x @ y
        return (x.astype(matmul_dtype) @ y.astype(matmul_dtype)).astype(out_dtype)

    eye = jnp.eye(chunk_size, dtype=out_dtype)
    power = a_strict_lower  # A^(2^0)
    inv = eye + power  # I + A
    k = 1
    while (1 << k) < chunk_size:
        power = mm(power, power)  # A^(2^k)
        inv = mm(inv, eye + power)
        k += 1
    return inv


def _forward_substitution_naive(a_strict_lower: jax.Array, chunk_size: int) -> jax.Array:
    """Serial forward-substitution inverse (benchmark A/B baseline for the Neumann form).

    ``C`` sequential ``lax.fori_loop`` steps; correct but latency-bound, and its
    reverse-mode backward is especially expensive. Selected via ``KDA_NAIVE_INVERSE=1``.
    """
    eye = jnp.eye(chunk_size, dtype=a_strict_lower.dtype)

    def body(i, attn):
        row_i = lax.dynamic_slice_in_dim(attn, i, 1, axis=-2)
        row_i = jnp.squeeze(row_i, axis=-2)
        idx = jnp.arange(chunk_size, dtype=attn.dtype)
        m1 = (idx < i).astype(attn.dtype)
        m2 = ((idx[:, None] < i) & (idx[None, :] < i)).astype(attn.dtype)
        incr = jnp.sum((row_i * m1)[..., :, None] * (attn * m2), axis=-2)
        new_row = jnp.expand_dims(row_i + incr, axis=-2)
        return lax.dynamic_update_slice_in_dim(attn, new_row, i, axis=-2)

    return lax.fori_loop(1, chunk_size, body, a_strict_lower) + eye


def chunk_kda(
    q: Float[Array, "... L Dk"],
    k: Float[Array, "... L Dk"],
    v: Float[Array, "... L Dv"],
    g: Float[Array, "... L Dk"],
    beta: Float[Array, "... L"],
    *,
    chunk_size: int = 64,
    initial_state: jax.Array | None = None,
    use_qk_l2norm: bool = True,
    matmul_dtype: jnp.dtype | None = jnp.bfloat16,
) -> tuple[Float[Array, "... L Dv"], jax.Array]:
    """Chunkwise-parallel per-channel gated delta rule (KDA train/prefill kernel).

    Does the intra-chunk work as dense matmuls plus one unit-lower-triangular inverse,
    and carries a single recurrent state ``S`` across chunks -- the matmul-bound form
    that keeps the layer efficient. See module docstring for the equations.

    Args mirror :func:`recurrent_kda`. ``chunk_size`` is the intra-chunk length C
    (64 is the H100 sweet spot). ``matmul_dtype`` selects the dtype of the
    *intra-chunk* GEMM operands (delta-correction matrix, its Neumann inverse, the
    pseudo-value/decayed-key products, and the intra-chunk attention); the fp32 cross-
    chunk state recurrence and all decay/cumsum math are always fp32. ``bfloat16``
    (the default) runs the intra-chunk GEMMs on Hopper/Blackwell tensor cores and is
    ~8% (fwd) / ~13% (fwd+bwd) faster on H100 at ~0.5% relative error vs the fp32
    reference; pass ``jnp.float32`` for the exact-fp32 path used as the test oracle.
    """
    mm_dtype = matmul_dtype

    def mm(spec: str, a: jax.Array, b: jax.Array) -> jax.Array:
        """Einsum with operands cast to ``mm_dtype`` (fp32-accumulated), else fp32."""
        if mm_dtype is None:
            return jnp.einsum(spec, a, b)
        return jnp.einsum(spec, a.astype(mm_dtype), b.astype(mm_dtype)).astype(jnp.float32)

    q, k = _prepare_qk(q, k, use_qk_l2norm)
    v = v.astype(jnp.float32)
    g = g.astype(jnp.float32)
    beta = beta.astype(jnp.float32)

    lead = q.shape[:-2]
    orig_len = q.shape[-2]
    dk, dv = q.shape[-1], v.shape[-1]
    pad = (-orig_len) % chunk_size
    if pad:
        q = jnp.pad(q, [(0, 0)] * (q.ndim - 2) + [(0, pad), (0, 0)])
        k = jnp.pad(k, [(0, 0)] * (k.ndim - 2) + [(0, pad), (0, 0)])
        v = jnp.pad(v, [(0, 0)] * (v.ndim - 2) + [(0, pad), (0, 0)])
        g = jnp.pad(g, [(0, 0)] * (g.ndim - 2) + [(0, pad), (0, 0)])
        beta = jnp.pad(beta, [(0, 0)] * (beta.ndim - 1) + [(0, pad)])

    n_chunks = q.shape[-2] // chunk_size
    c = chunk_size

    def to_chunks(x: jax.Array) -> jax.Array:
        # (..., n_chunks, C, d) with chunks on the front for lax.scan later.
        return x.reshape(*lead, n_chunks, c, x.shape[-1])

    qc = to_chunks(q)
    kc = to_chunks(k)
    vc = to_chunks(v)
    gc = to_chunks(g)
    bc = beta.reshape(*lead, n_chunks, c)

    g_cum = jnp.cumsum(gc, axis=-2)  # (..., n, C, d_k) cumulative log-decay per channel
    exp_g = jnp.exp(g_cum)  # <= 1 (g_cum <= 0): inflation factor, always safe
    exp_ng = jnp.exp(jnp.minimum(-g_cum, _DEFLATE_EXP_CAP))  # deflation factor, capped

    v_beta = vc * bc[..., None]
    k_beta = kc * bc[..., None]

    # Delta-correction matrix A[r,i] = -beta_r (k_r . k_i) exp(g_cum_r - g_cum_i),
    # folded via inflate/deflate so it is a plain matmul. Strictly lower triangular.
    k_beta_inflate = k_beta * exp_g
    k_deflate = kc * exp_ng
    a_raw = -mm("...rd,...id->...ri", k_beta_inflate, k_deflate)
    strict_lower = jnp.tril(jnp.ones((c, c), dtype=bool), k=-1)
    a_raw = jnp.where(strict_lower, a_raw, 0.0)

    a_bcc = a_raw.reshape(-1, c, c)
    if _KDA_NAIVE_INVERSE:
        t_mat = _forward_substitution_naive(a_bcc, c).reshape(*lead, n_chunks, c, c)
    else:
        t_mat = _unit_lower_triangular_inverse(a_bcc, c, matmul_dtype=mm_dtype).reshape(*lead, n_chunks, c, c)

    v_pseudo = mm("...rj,...jd->...rd", t_mat, v_beta)
    k_cumdecay = mm("...rj,...jd->...rd", t_mat, k_beta * exp_g)

    q_inflate = qc * exp_g

    if initial_state is None:
        state = jnp.zeros((*lead, dk, dv), dtype=jnp.float32)
    else:
        state = initial_state.astype(jnp.float32)

    strict_upper = jnp.triu(jnp.ones((c, c), dtype=bool), k=1)

    # Scan over chunks (chunks on the leading axis).
    def move_chunk_front(x: jax.Array) -> jax.Array:
        return jnp.moveaxis(x, -3, 0)

    scan_inputs = (
        move_chunk_front(q_inflate),
        move_chunk_front(k_deflate),
        move_chunk_front(kc),
        move_chunk_front(g_cum),
        move_chunk_front(v_pseudo),
        move_chunk_front(k_cumdecay),
    )

    def chunk_step(s_prev: jax.Array, inp):
        q_inf, k_def, k_i, gcum_i, v_ps, k_cd = inp
        # Intra-chunk attention over corrected values (strictly-lower + diagonal).
        attn = mm("...rd,...jd->...rj", q_inf, k_def)
        attn = jnp.where(strict_upper, 0.0, attn)
        # State-touching GEMMs stay fp32 (S carries the whole prefix -- precision matters).
        v_prime = jnp.einsum("...rd,...dm->...rm", k_cd, s_prev)
        v_new = v_ps - v_prime
        inter = jnp.einsum("...rd,...dm->...rm", q_inf, s_prev)
        out_i = inter + mm("...rj,...jm->...rm", attn, v_new)
        # Carry state to the chunk boundary and write the innovations.
        g_tail = gcum_i[..., -1, :]  # (..., d_k)
        decay_tail = jnp.exp(g_tail)
        decay_weights = jnp.exp(g_tail[..., None, :] - gcum_i)  # (..., C, d_k)
        add = jnp.einsum("...rd,...rm->...dm", k_i * decay_weights, v_new)
        s_new = s_prev * decay_tail[..., :, None] + add
        return s_new, out_i

    state, out_chunks = lax.scan(chunk_step, state, scan_inputs)
    out = jnp.moveaxis(out_chunks, 0, -3)  # (..., n, C, d_v)
    out = out.reshape(*lead, n_chunks * c, dv)
    if pad:
        out = out[..., :orig_len, :]
    return out, state
