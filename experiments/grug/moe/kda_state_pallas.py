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
from enum import StrEnum
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plt

from experiments.grug.moe.kda_prep_pallas import dot_f32

f32 = jnp.float32
# d_v column block and warps of the fused cotangent kernel (``StateGrads.PALLAS``), tuned on H100
# at C=32, d=128 (0.76 ms vs 0.82-1.44 for other tiles; the XLA einsums take ~0.88).
_GRADS_BLOCK_V = 32
_GRADS_NUM_WARPS = 4


class StateGrads(StrEnum):
    """Where the backward's per-chunk cotangents that contract over d_v are computed."""

    XLA = "xla"  # batched XLA einsums over the saved per-chunk states and their cotangents
    PALLAS = "pallas"  # one parallel kernel per (g, chunk) reading every per-chunk tensor once


class StateLayout(StrEnum):
    """On-chip orientation of the state ``S`` (``d_k x d_v``) in the sequential passes."""

    # S held as (d_k, block_v): every per-step GEMM takes the state (or v_new / dS) as its right operand,
    # which goes through shared memory each step.
    DK_ROWS = "dk_rows"
    # S^T held as (block_v, d_k): the state, v_new and dS are the left operands of the per-step GEMMs (the
    # flash-attention P @ V pattern, register-resident); per-chunk h / v_new / dS are saved transposed.
    DV_ROWS = "dv_rows"


class StateConfig(NamedTuple):
    num_heads: int
    block_v: int
    num_warps: int
    num_stages: int
    interpret: bool
    save_states: bool
    grads: StateGrads
    # d_v tile of the primal (non-differentiated) forward, which then skips the h / v_new stores; 0 keeps
    # the primal on the saving kernel (``block_v``).
    primal_block_v: int
    primal_num_warps: int
    layout: StateLayout
    grads_block_v: int
    grads_num_warps: int


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


def _state_fwd_t_kernel(qi_ref, attn_ref, kw_ref, kcd_ref, vp_ref, decay_ref, s0_ref, out_ref, *refs, save: bool):
    """``_state_fwd_kernel`` on the transposed state ``S^T`` (``StateLayout.DV_ROWS``): with ``vt = v_new^T``,
    ``vt = Vp^T - S^T Kcd^T``, ``out^T = S^T Qi^T + vt attn^T`` and ``S^T <- S^T Diag(decay) + vt Kw``, so the
    state and v_new are always left GEMM operands. With ``save``, h and v_new are stored transposed,
    ``(block_v, d_k)`` / ``(block_v, C)`` per chunk."""
    h_ref, vnew_ref, st_ref = refs if save else (None, None, refs[0])
    n, c = kw_ref.shape[:2]

    def body(i, st):
        if save:
            h_ref[i] = st.astype(h_ref.dtype)
        vt = vp_ref[i].T - dot_f32(st, kcd_ref[i].astype(f32), trans_b=True)
        if save:
            vnew_ref[i] = vt
        out_t = dot_f32(st, qi_ref[i].astype(f32), trans_b=True) + dot_f32(vt, attn_ref[i].astype(f32), trans_b=True)
        out_ref[pl.ds(pl.multiple_of(i * c, c), c), :] = out_t.T.astype(out_ref.dtype)
        return st * decay_ref[i][None, :] + dot_f32(vt, kw_ref[i].astype(f32))

    st_ref[...] = lax.fori_loop(0, n, body, s0_ref[...].T).T


def _state_bwd_t_kernel(qi_ref, attn_ref, kw_ref, kcd_ref, decay_ref, dout_ref, dst_ref, ds_ref, dvp_ref, ds0_ref):
    """``_state_bwd_kernel`` on the transposed cotangent ``dS^T`` (``StateLayout.DV_ROWS``):
    ``dvt = dout^T attn + dS^T Kw^T`` and ``dS^T <- dS^T Diag(decay) - dvt Kcd + dout^T Qi``. Per-chunk dS is
    stored transposed; dV_pseudo in its own layout."""
    n, c = kw_ref.shape[:2]

    def body(j, dst):
        i = n - 1 - j  # dst is the cotangent of S_i^T (the state leaving chunk i)
        ds_ref[i] = dst
        dout_t = dout_ref[pl.ds(pl.multiple_of(i * c, c), c), :].astype(f32).T
        dvt = dot_f32(dout_t, attn_ref[i].astype(f32)) + dot_f32(dst, kw_ref[i].astype(f32), trans_b=True)
        dvp_ref[i] = dvt.T
        ds_in = dst * decay_ref[i][None, :] - dot_f32(dvt, kcd_ref[i].astype(f32))
        return ds_in + dot_f32(dout_t, qi_ref[i].astype(f32))

    ds0_ref[...] = lax.fori_loop(0, n, body, dst_ref[...].T).T


def _state_grads_t_kernel(
    h_ref, ds_ref, vnew_ref, dvp_ref, dout_ref, dqi_ref, dattn_ref, dkw_ref, dkcd_ref, ddecay_ref, *, block_v: int
):
    """``_state_grads_kernel`` on the transposed per-chunk h^T, dS^T, v_new^T (``StateLayout.DV_ROWS``):
    dQi = dout h^T, dattn = dout v_new^T, dKw = v_new dS^T, dKcd = -dVp h^T, ddecay = colsum(h^T * dS^T)."""
    dv, dk = h_ref.shape
    c = vnew_ref.shape[1]
    bv = block_v
    dqi = jnp.zeros((c, dk), f32)
    dattn = jnp.zeros((c, c), f32)
    dkw = jnp.zeros((c, dk), f32)
    dkcd = jnp.zeros((c, dk), f32)
    ddecay = jnp.zeros((dk,), f32)
    for j in range(dv // bv):
        rows = pl.ds(j * bv, bv)
        h = h_ref[rows, :].astype(f32)
        ds = ds_ref[rows, :]
        vt = vnew_ref[rows, :]
        dout = dout_ref[:, rows].astype(f32)
        dqi = dqi + dot_f32(dout, h)
        dattn = dattn + dot_f32(dout, vt)
        dkw = dkw + dot_f32(vt, ds, trans_a=True)
        dkcd = dkcd - dot_f32(dvp_ref[:, rows], h)
        ddecay = ddecay + jnp.sum(h * ds, axis=0)
    dqi_ref[...] = dqi.astype(dqi_ref.dtype)
    dattn_ref[...] = dattn.astype(dattn_ref.dtype)
    dkw_ref[...] = dkw.astype(dkw_ref.dtype)
    dkcd_ref[...] = dkcd.astype(dkcd_ref.dtype)
    ddecay_ref[...] = ddecay


def _state_fwd_lean_kernel(qi_ref, attn_ref, kw_ref, kcd_ref, vp_ref, decay_ref, s0_ref, out_ref, st_ref):
    """``_state_fwd_kernel`` without the per-chunk h / v_new stores (the primal forward needs only ``out``)."""
    n, c = kw_ref.shape[:2]

    def body(i, s):
        vn = vp_ref[i] - dot_f32(kcd_ref[i].astype(f32), s)
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


def _state_grads_kernel(
    h_ref, ds_ref, vnew_ref, dvp_ref, dout_ref, dqi_ref, dattn_ref, dkw_ref, dkcd_ref, ddecay_ref, *, block_v: int
):
    """One chunk's d_v-contracting cotangents from its entering state ``h``, the cotangent ``ds``
    of the state it leaves, ``v_new``, ``dV_pseudo`` and the output cotangent, walking d_v in
    column blocks: dQi = dout h^T, dattn = dout v_new^T, dKw = v_new dS^T, dKcd = -dVp h^T and
    ddecay = rowsum(h * dS) (the einsums of the XLA path, with each tensor read once)."""
    dk, dv = h_ref.shape
    c = vnew_ref.shape[0]
    dqi = jnp.zeros((c, dk), f32)
    dattn = jnp.zeros((c, c), f32)
    dkw = jnp.zeros((c, dk), f32)
    dkcd = jnp.zeros((c, dk), f32)
    ddecay = jnp.zeros((dk,), f32)
    for j in range(dv // block_v):
        cols = pl.ds(j * block_v, block_v)
        h = h_ref[:, cols].astype(f32)
        ds = ds_ref[:, cols]
        vnew = vnew_ref[:, cols]
        dout = dout_ref[:, cols].astype(f32)
        dqi = dqi + dot_f32(dout, h, trans_b=True)
        dattn = dattn + dot_f32(dout, vnew, trans_b=True)
        dkw = dkw + dot_f32(vnew, ds, trans_b=True)
        dkcd = dkcd - dot_f32(dvp_ref[:, cols], h, trans_b=True)
        ddecay = ddecay + jnp.sum(h * ds, axis=1)
    dqi_ref[...] = dqi.astype(dqi_ref.dtype)
    dattn_ref[...] = dattn.astype(dattn_ref.dtype)
    dkw_ref[...] = dkw.astype(dkw_ref.dtype)
    dkcd_ref[...] = dkcd.astype(dkcd_ref.dtype)
    ddecay_ref[...] = ddecay


def _state_grads_call(qi, attn, kw, kcd, h, vnew, dout, ds, dvp, cfg: StateConfig):
    """The per-chunk d_v-contracting cotangents; h / v_new / dS are transposed under ``StateLayout.DV_ROWS``."""
    gb, n, c, dk = kw.shape
    dv = dvp.shape[-1]
    heads = cfg.num_heads
    transposed = cfg.layout == StateLayout.DV_ROWS

    def chunk(*dims):
        return pl.BlockSpec((None, None, *dims), lambda gi, ni: (gi, ni) + (0,) * len(dims))

    dout_spec = pl.BlockSpec((None, c, None, dv), lambda gi, ni: (gi // heads, ni, gi % heads, 0))
    if transposed:
        kernel, saved_specs = _state_grads_t_kernel, [chunk(dv, dk), chunk(dv, dk), chunk(dv, c)]
    else:
        kernel, saved_specs = _state_grads_kernel, [chunk(dk, dv), chunk(dk, dv), chunk(c, dv)]
    return pl.pallas_call(
        functools.partial(kernel, block_v=min(cfg.grads_block_v, dv)),
        grid=(gb, n),
        in_specs=[*saved_specs, chunk(c, dv), dout_spec],
        out_specs=[chunk(c, dk), chunk(c, c), chunk(c, dk), chunk(c, dk), chunk(dk)],
        out_shape=[
            jax.ShapeDtypeStruct(qi.shape, qi.dtype),
            jax.ShapeDtypeStruct(attn.shape, attn.dtype),
            jax.ShapeDtypeStruct(kw.shape, kw.dtype),
            jax.ShapeDtypeStruct(kcd.shape, kcd.dtype),
            jax.ShapeDtypeStruct((gb, n, dk), f32),
        ],
        compiler_params=plt.CompilerParams(num_warps=cfg.grads_num_warps, num_stages=1),
        interpret=cfg.interpret,
        name="kda_state_bwd_grads",
    )(h, ds, vnew, dvp, dout)


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


def _tslab(n, bv, d):
    """Program (g, v)'s (block_v, d) rows of a transposed per-chunk ``(G, n, d_v, d)`` tensor."""
    return pl.BlockSpec((None, n, bv, d), lambda gi, vi: (gi, 0, vi, 0))


def _state_fwd_call(qi, attn, kw, kcd, vp, decay, s0, out_dtype, cfg: StateConfig):
    """Forward state pass that also stores the per-chunk h (entering state) and v_new for the backward."""
    gb, n, c, dk = kw.shape
    dv = vp.shape[-1]
    bv = min(cfg.block_v, dv)
    heads = cfg.num_heads
    out_spec, out_sds = _seq_slab(n * c, heads, bv), jax.ShapeDtypeStruct((gb // heads, n * c, heads, dv), out_dtype)
    st_sds = jax.ShapeDtypeStruct((gb, dk, dv), f32)
    if cfg.layout == StateLayout.DV_ROWS:
        kernel = functools.partial(_state_fwd_t_kernel, save=True)
        saved_specs = [_tslab(n, bv, dk), _tslab(n, bv, c)]
        saved_sds = [jax.ShapeDtypeStruct((gb, n, dv, dk), kw.dtype), jax.ShapeDtypeStruct((gb, n, dv, c), f32)]
    else:
        kernel = _state_fwd_kernel
        saved_specs = [_vslab(n, dk, bv), _vslab(n, c, bv)]
        saved_sds = [jax.ShapeDtypeStruct((gb, n, dk, dv), kw.dtype), jax.ShapeDtypeStruct((gb, n, c, dv), f32)]
    return pl.pallas_call(
        kernel,
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
        out_specs=[out_spec, *saved_specs, _state_slab(dk, bv)],
        out_shape=[out_sds, *saved_sds, st_sds],
        compiler_params=plt.CompilerParams(num_warps=cfg.num_warps, num_stages=cfg.num_stages),
        interpret=cfg.interpret,
        name="kda_state_fwd",
    )(qi, attn, kw, kcd, vp, decay, s0)


def _state_fwd_lean_call(qi, attn, kw, kcd, vp, decay, s0, out_dtype, cfg: StateConfig):
    """Forward state pass emitting only the outputs and the final state (the primal, ``primal_block_v``)."""
    gb, n, c, dk = kw.shape
    dv = vp.shape[-1]
    bv = min(cfg.primal_block_v, dv)
    heads = cfg.num_heads
    if cfg.layout == StateLayout.DV_ROWS:
        kernel = functools.partial(_state_fwd_t_kernel, save=False)
    else:
        kernel = _state_fwd_lean_kernel
    return pl.pallas_call(
        kernel,
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
        out_specs=[_seq_slab(n * c, heads, bv), _state_slab(dk, bv)],
        out_shape=[
            jax.ShapeDtypeStruct((gb // heads, n * c, heads, dv), out_dtype),
            jax.ShapeDtypeStruct((gb, dk, dv), f32),
        ],
        compiler_params=plt.CompilerParams(num_warps=cfg.primal_num_warps, num_stages=cfg.num_stages),
        interpret=cfg.interpret,
        name="kda_state_fwd_lean",
    )(qi, attn, kw, kcd, vp, decay, s0)


def _state_bwd_call(qi, attn, kw, kcd, decay, h, vnew, dout, dst, cfg: StateConfig):
    gb, n, c, dk = kw.shape
    dv = dout.shape[-1]
    bv = min(cfg.block_v, dv)
    heads = cfg.num_heads
    transposed = cfg.layout == StateLayout.DV_ROWS
    ds, dvp, ds0 = pl.pallas_call(
        _state_bwd_t_kernel if transposed else _state_bwd_kernel,
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
        out_specs=[_tslab(n, bv, dk) if transposed else _vslab(n, dk, bv), _vslab(n, c, bv), _state_slab(dk, bv)],
        out_shape=[
            jax.ShapeDtypeStruct((gb, n, dv, dk) if transposed else (gb, n, dk, dv), f32),
            jax.ShapeDtypeStruct((gb, n, c, dv), f32),
            jax.ShapeDtypeStruct((gb, dk, dv), f32),
        ],
        compiler_params=plt.CompilerParams(num_warps=cfg.num_warps, num_stages=cfg.num_stages),
        interpret=cfg.interpret,
        name="kda_state_bwd",
    )(qi, attn, kw, kcd, decay, dout, dst)
    # The C x d_k and C x C cotangents contract over d_v, which the kernel splits across
    # programs; given the per-chunk dS they are plain batched GEMMs (XLA einsums, or one
    # parallel kernel that reads each per-chunk tensor once).
    if cfg.grads == StateGrads.PALLAS:
        dqi, dattn, dkw, dkcd, ddecay = _state_grads_call(qi, attn, kw, kcd, h, vnew, dout, ds, dvp, cfg)
        return dqi, dattn, dkw, dkcd, dvp, ddecay, ds0
    if transposed:
        raise ValueError("StateLayout.DV_ROWS needs StateGrads.PALLAS")
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
    if cfg.primal_block_v:
        out, st = _state_fwd_lean_call(qi, attn, kw, kcd, vp, decay, s0, out_dtype, cfg)
        return out, st
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
    grads: StateGrads = StateGrads.XLA,
    primal_block_v: int = 0,
    primal_num_warps: int = 4,
    layout: StateLayout = StateLayout.DK_ROWS,
    grads_block_v: int = _GRADS_BLOCK_V,
    grads_num_warps: int = _GRADS_NUM_WARPS,
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
        grads: where the backward forms the cotangents that contract over d_v (``StateGrads``).
        primal_block_v: d_v tile of the primal (non-differentiated) forward, which then runs a kernel that
            skips the per-chunk h / v_new stores only the backward reads; 0 keeps the saving kernel there.
        primal_num_warps: Triton warps of that lean primal kernel.
        layout: on-chip orientation of the state in the sequential passes (``StateLayout``; ``DV_ROWS``
            needs ``grads=StateGrads.PALLAS``).
        grads_block_v, grads_num_warps: d_v block / warps of the ``StateGrads.PALLAS`` cotangent kernel.

    Returns:
        ``(out, final_state)``: ``out_n = q_inflate_n S_{n-1} + attn_n v_new_n`` in model
        layout ``(B, L = n*C, num_heads, d_v)`` and the ``(G, d_k, d_v)`` final state.
    """
    dv = v_pseudo.shape[-1]
    for name, bv in (("block_v", block_v), ("primal_block_v", primal_block_v or block_v)):
        if dv % min(bv, dv):
            raise ValueError(f"d_v={dv} must be a multiple of {name}={bv}")
    cfg = StateConfig(
        num_heads,
        block_v,
        num_warps,
        num_stages,
        interpret,
        save_states,
        StateGrads(grads),
        primal_block_v,
        primal_num_warps,
        StateLayout(layout),
        grads_block_v,
        grads_num_warps,
    )
    return _state_pass(q_inflate, attn, kw, k_cumdecay, v_pseudo, decay, initial_state, jnp.dtype(out_dtype), cfg)
