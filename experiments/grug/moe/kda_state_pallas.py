# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Inter-chunk KDA state pass as a Pallas (Triton) kernel.

Given the per-chunk prep (``kda_prep_pallas``), the state entering chunk ``n`` obeys

    v_new_n = V_pseudo_n - K_cumdecay_n S_{n-1}
    S_n     = Diag(decay_n) S_{n-1} + Kw_n^T v_new_n

    out_n   = Q_inflate_n S_{n-1} + attn_n v_new_n

This kernel runs that recurrence sequentially over chunks with ``S`` resident on-chip,
on a grid ``(G, d_v / block_v)`` (each program owns a ``d_k x block_v`` column slab of
``S``; the slabs are independent), and emits each chunk's output slab as it goes. Per
step it only streams the chunk's C x d_k / C x C operands and a C x block_v slice of
``V_pseudo``, so the whole pass reads each prep tensor ~once -- versus the associative
scan, which multiplies d_k x d_k transition matrices (O(n d_k^3) work plus their HBM
traffic). The state entering every chunk (``h``) and ``v_new`` are saved for the backward.

Outputs are written straight into the model's ``(B, L, H, d_v)`` activation layout.

The backward is the matching reverse-time pass (``dS`` resident, the output cotangent
folded in); it emits the per-chunk ``dS`` and ``dV_pseudo``, from which the cotangents
that contract over d_v (split across programs) are batched GEMMs outside the kernel.
"""

import functools
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plt

from experiments.grug.moe.kda_prep_pallas import dot_f32

f32 = jnp.float32


class StateConfig(NamedTuple):
    num_heads: int
    block_v: int
    num_warps: int
    num_stages: int
    interpret: bool
    save_states: bool


def _state_fwd_kernel(qi_ref, attn_ref, kw_ref, kcd_ref, vp_ref, decay_ref, s0_ref, out_ref, h_ref, vnew_ref, st_ref):
    n = kw_ref.shape[0]

    # The C x d_k / C x C operands may arrive in bf16 (halving the streamed bytes); they
    # are upcast so the GEMMs against S keep an fp32 (TF32) state operand -- S itself is
    # never rounded (only its stored copy ``h`` is, for the backward).
    c = kw_ref.shape[1]

    def body(i, s):
        h_ref[i] = s.astype(h_ref.dtype)
        vn = vp_ref[i] - dot_f32(kcd_ref[i].astype(f32), s)
        vnew_ref[i] = vn
        out = dot_f32(qi_ref[i].astype(f32), s) + dot_f32(attn_ref[i].astype(f32), vn)
        out_ref[pl.ds(pl.multiple_of(i * c, c), c), :] = out.astype(out_ref.dtype)
        return decay_ref[i][:, None] * s + dot_f32(kw_ref[i].astype(f32), vn, trans_a=True)

    st_ref[...] = lax.fori_loop(0, n, body, s0_ref[...])


def _state_bwd_kernel(qi_ref, attn_ref, kw_ref, kcd_ref, decay_ref, dout_ref, dst_ref, ds_ref, dvp_ref, ds0_ref):
    n, c = kw_ref.shape[:2]

    def body(j, ds):
        # ds is the cotangent of S_i (the state leaving chunk i).
        i = n - 1 - j
        ds_ref[i] = ds
        dout = dout_ref[pl.ds(pl.multiple_of(i * c, c), c), :].astype(f32)
        dvn = dot_f32(attn_ref[i].astype(f32), dout, trans_a=True) + dot_f32(kw_ref[i].astype(f32), ds)
        dvp_ref[i] = dvn
        ds_in = decay_ref[i][:, None] * ds - dot_f32(kcd_ref[i].astype(f32), dvn, trans_a=True)
        return ds_in + dot_f32(qi_ref[i].astype(f32), dout, trans_a=True)

    ds0_ref[...] = lax.fori_loop(0, n, body, dst_ref[...])


def _full(n, *dims):
    """Whole-sequence block for one g, full trailing dims."""
    return pl.BlockSpec((None, n, *dims), lambda gi, vi: (gi,) + (0,) * (len(dims) + 1))


def _vslab(n, c, bv):
    return pl.BlockSpec((None, n, c, bv), lambda gi, vi: (gi, 0, 0, vi))


def _seq_slab(length, heads, bv):
    """The (length x block_v) slab of a model-layout ``(B, L, H, d_v)`` tensor for program
    ``(g = b*H + h, v)``: outputs go straight to the activation layout, no transposes."""
    return pl.BlockSpec((None, length, None, bv), lambda gi, vi: (gi // heads, 0, gi % heads, vi))


def _state_slab(dk, bv):
    return pl.BlockSpec((None, dk, bv), lambda gi, vi: (gi, 0, vi))


def _state_fwd_call(qi, attn, kw, kcd, vp, decay, s0, out_dtype, cfg: StateConfig):
    gb, n, c, dk = kw.shape
    dv = vp.shape[-1]
    bv = min(cfg.block_v, dv)
    heads = cfg.num_heads
    return pl.pallas_call(
        _state_fwd_kernel,
        grid=(gb, dv // bv),
        in_specs=[
            _full(n, c, dk),
            _full(n, c, c),
            _full(n, c, dk),
            _full(n, c, dk),
            _vslab(n, c, bv),
            _full(n, dk),
            _state_slab(dk, bv),
        ],
        out_specs=[_seq_slab(n * c, heads, bv), _vslab(n, dk, bv), _vslab(n, c, bv), _state_slab(dk, bv)],
        out_shape=[
            jax.ShapeDtypeStruct((gb // heads, n * c, heads, dv), out_dtype),
            jax.ShapeDtypeStruct((gb, n, dk, dv), kw.dtype),
            jax.ShapeDtypeStruct((gb, n, c, dv), f32),
            jax.ShapeDtypeStruct((gb, dk, dv), f32),
        ],
        compiler_params=plt.CompilerParams(num_warps=cfg.num_warps, num_stages=cfg.num_stages),
        interpret=cfg.interpret,
        name="kda_state_fwd",
    )(qi, attn, kw, kcd, vp, decay, s0)


def _state_bwd_call(qi, attn, kw, kcd, decay, h, vnew, dout, dst, cfg: StateConfig):
    gb, n, c, dk = kw.shape
    dv = vnew.shape[-1]
    bv = min(cfg.block_v, dv)
    heads = cfg.num_heads
    ds, dvp, ds0 = pl.pallas_call(
        _state_bwd_kernel,
        grid=(gb, dv // bv),
        in_specs=[
            _full(n, c, dk),
            _full(n, c, c),
            _full(n, c, dk),
            _full(n, c, dk),
            _full(n, dk),
            _seq_slab(n * c, heads, bv),
            _state_slab(dk, bv),
        ],
        out_specs=[_vslab(n, dk, bv), _vslab(n, c, bv), _state_slab(dk, bv)],
        out_shape=[
            jax.ShapeDtypeStruct((gb, n, dk, dv), f32),
            jax.ShapeDtypeStruct((gb, n, c, dv), f32),
            jax.ShapeDtypeStruct((gb, dk, dv), f32),
        ],
        compiler_params=plt.CompilerParams(num_warps=cfg.num_warps, num_stages=cfg.num_stages),
        interpret=cfg.interpret,
        name="kda_state_bwd",
    )(qi, attn, kw, kcd, decay, dout, dst)
    # The C x d_k and C x C cotangents contract over d_v, which the kernel splits across
    # programs; given the per-chunk dS they are plain batched GEMMs, cheapest done by XLA.
    b = gb // heads
    dout_g = dout.reshape(b, n, c, heads, dv)
    h_g = h.reshape(b, heads, n, dk, dv)
    vnew_g = vnew.reshape(b, heads, n, c, dv)
    dqi = jnp.einsum("bnchv,bhndv->bhncd", dout_g, h_g, preferred_element_type=f32)
    dattn = jnp.einsum("bnrhv,bhnjv->bhnrj", dout_g, vnew_g, preferred_element_type=f32)
    dqi = dqi.reshape(gb, n, c, dk).astype(qi.dtype)
    dattn = dattn.reshape(gb, n, c, c).astype(attn.dtype)
    dkw = jnp.einsum("gncv,gndv->gncd", vnew, ds).astype(kw.dtype)
    dkcd = -jnp.einsum("gncv,gndv->gncd", dvp, h, preferred_element_type=f32).astype(kcd.dtype)
    ddecay = jnp.sum(h.astype(f32) * ds, axis=-1)
    return dqi, dattn, dkw, dkcd, dvp, ddecay, ds0


@functools.partial(jax.custom_vjp, nondiff_argnums=(7, 8))
def _state_pass(qi, attn, kw, kcd, vp, decay, s0, out_dtype, cfg: StateConfig):
    out, _, _, st = _state_fwd_call(qi, attn, kw, kcd, vp, decay, s0, out_dtype, cfg)
    return out, st


def _state_pass_fwd(qi, attn, kw, kcd, vp, decay, s0, out_dtype, cfg):
    out, h, vnew, st = _state_fwd_call(qi, attn, kw, kcd, vp, decay, s0, out_dtype, cfg)
    # Unless ``save_states``, the per-chunk states h ((G, n, d_k, d_v), one d_k x d_v matrix per
    # chunk) are NOT saved: at 16-token chunks they are ~8x the size of the layer's activations,
    # and saved across the layers they OOM'd d1024. The backward then re-runs the forward state
    # pass to rebuild them (the same deterministic kernel, so the same values).
    saved = (h, vnew) if cfg.save_states else None
    return (out, st), (qi, attn, kw, kcd, vp, decay, s0, saved)


def _state_pass_bwd(out_dtype, cfg, res, cts):
    qi, attn, kw, kcd, vp, decay, s0, saved = res
    if saved is None:
        _, h, vnew, _ = _state_fwd_call(qi, attn, kw, kcd, vp, decay, s0, out_dtype, cfg)
    else:
        h, vnew = saved
    dout, dst = cts  # dout is consumed in its own dtype (upcast on-chip)
    return _state_bwd_call(qi, attn, kw, kcd, decay, h, vnew, dout, dst.astype(f32), cfg)


_state_pass.defvjp(_state_pass_fwd, _state_pass_bwd)


def chunk_state_pass(
    q_inflate: jax.Array,
    attn: jax.Array,
    kw: jax.Array,
    k_cumdecay: jax.Array,
    v_pseudo: jax.Array,
    decay: jax.Array,
    initial_state: jax.Array,
    *,
    num_heads: int,
    out_dtype: jnp.dtype = f32,
    block_v: int = 32,
    num_warps: int = 4,
    num_stages: int = 2,
    interpret: bool = False,
    save_states: bool = False,
) -> tuple[jax.Array, jax.Array]:
    """Sequential inter-chunk KDA state pass with the chunk outputs fused in (differentiable).

    Args:
        q_inflate, kw, k_cumdecay: ``(G, n, C, d_k)``; attn: ``(G, n, C, C)`` lower-
            triangular intra-chunk attention (``G = B * num_heads``, b-major, as produced
            by ``fused_chunk_prep``). fp32 or bf16 (upcast on-chip; the state is fp32).
        v_pseudo: ``(G, n, C, d_v)`` fp32.
        decay: ``(G, n, d_k)`` fp32 per-chunk total decay ``exp(g_tail)``.
        initial_state: ``(G, d_k, d_v)`` fp32.
        out_dtype: dtype of the emitted outputs.
        block_v: ``d_v`` slab width per program (parallelism vs. re-reading the d_k operands).
        save_states: keep the per-chunk states for the backward instead of re-running the
            forward pass there (same values; costs ``G * n * d_k * d_v`` stored elements).

    Returns:
        ``(out, final_state)``: ``out_n = q_inflate_n S_{n-1} + attn_n v_new_n`` in model
        layout ``(B, L = n*C, num_heads, d_v)`` and the ``(G, d_k, d_v)`` final state.
    """
    dv = v_pseudo.shape[-1]
    if dv % min(block_v, dv):
        raise ValueError(f"d_v={dv} must be a multiple of block_v={block_v}")
    cfg = StateConfig(num_heads, block_v, num_warps, num_stages, interpret, save_states)
    return _state_pass(q_inflate, attn, kw, k_cumdecay, v_pseudo, decay, initial_state, jnp.dtype(out_dtype), cfg)
