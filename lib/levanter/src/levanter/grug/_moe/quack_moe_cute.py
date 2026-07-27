#!/usr/bin/env python3
# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Vendored CuTeDSL→JAX shim for QuACK's SM100 gated grouped GEMM (SonicMoE kernel).

Mirrors David's `_fa4_cute_backend.py` pattern: build a thin ``@cute.jit`` launcher
with the JAX ``cutlass_call`` signature ``(stream, *inputs, *outputs, **scalars)``
that reuses QuACK's ``GemmGatedSm100`` kernel, then wrap it with
``cutlass.jax.cutlass_call``. Forward-only for now (grouped SwiGLU): tokens are
pre-sorted by expert (varlen_m via ``cu_seqlens_m``; no A_idx gather).

A[M,K] @ B[E,K,2N]  -> (per expert group) -> SwiGLU -> PostAct[M,N]
"""
from __future__ import annotations


import jax
import jax.numpy as jnp

import cutlass
import cutlass.cute as cute
import cutlass.jax as cjax
from quack.gemm_act import GemmActMixin, GemmGatedSm100, act_fn_map, gate_fn_map
from quack.gemm_act import get_max_active_clusters
from quack.gemm_tvm_ffi_utils import make_scheduler_args, make_varlen_args

_ACC = cutlass.Float32
_JAX_TO_CUTE = {
    jnp.dtype(jnp.bfloat16): cutlass.BFloat16,
    jnp.dtype(jnp.float16): cutlass.Float16,
    jnp.dtype(jnp.float32): cutlass.Float32,
}


def _cute_dtype(dt):
    return _JAX_TO_CUTE[jnp.dtype(dt)]


def _build_launcher(*, a_dtype, tile_mn, cluster_mnk, activation, max_active_clusters, max_swizzle):
    """Return a ``@cute.jit`` launcher with the cutlass_call signature.

    Signature: (stream, mA, mB, mCuSeqlens, mD, mPostAct)
      mA:[M,K] tokens (k-major)  mB:[E,K,2N] weights  mCuSeqlens:[E+1] int32
      mD:[M,2N] preact out (n-major)  mPostAct:[M,N] swiglu out (n-major)
    """
    act = gate_fn_map[activation] if activation in gate_fn_map else act_fn_map[activation]

    @cute.jit
    def launcher(stream, mA, mB, mCuSeqlens, mD, mPostAct):
        gemm = GemmGatedSm100(
            _ACC,
            a_dtype,
            tile_mn,
            cluster_mnk,
            gather_A=False,
            use_clc_persistence=False,
        )
        epi_args = GemmActMixin.EpilogueArguments(
            mPostAct,
            act,  # SwiGLU activation function (Constexpr)
            mRowVecBroadcast=None,
            mColVecBroadcast=None,
        )
        scheduler_args = make_scheduler_args(max_active_clusters, max_swizzle, None)
        varlen_args = make_varlen_args(mCuSeqlens, None, None)
        gemm(mA, mB, mD, None, epi_args, scheduler_args, varlen_args, stream)

    return launcher


def quack_gated_grouped_gemm(
    x_sort,
    w_gate_up,
    cu_seqlens,
    *,
    activation="swiglu",
    tile_mn=(256, 128),
    cluster_mnk=(2, 1, 1),
    max_swizzle=8,
    return_preact=False,
):
    """Grouped SwiGLU expert GEMM via QuACK's SM100 kernel.

    x_sort: [M, K] tokens sorted by expert. w_gate_up: [E, K, 2N]. cu_seqlens: [E+1] int32.
    Returns postact [M, N].
    """
    M, K = x_sort.shape
    N2 = w_gate_up.shape[2]
    N = N2 // 2
    a_dtype = _cute_dtype(x_sort.dtype)
    try:
        max_active_clusters = get_max_active_clusters(cluster_mnk[0] * cluster_mnk[1])
    except Exception:
        max_active_clusters = 148  # B200 SM-count fallback for host-side lowering tests
    launcher = _build_launcher(
        a_dtype=a_dtype,
        tile_mn=tile_mn,
        cluster_mnk=cluster_mnk,
        activation=activation,
        max_active_clusters=max_active_clusters,
        max_swizzle=max_swizzle,
    )
    ts = cjax.TensorSpec
    # divisibility is in physical-dim order (contiguous dim gets the vector width).
    # B is physically [E,K,2N] but the kernel wants it as [K,2N,E] (expert = trailing
    # batch/L mode); express that with mode=(1,2,0) rather than a physical transpose.
    a_spec = ts(divisibility=(1, 8), static=False)  # [M,K] k-major
    # B is physically [E,K,2N]; kernel wants n-major logical [2N,K,E] (leading_dim 0).
    b_spec = ts(mode=(2, 1, 0), divisibility=(1, 1, 8), static=False)
    cu_spec = ts(static=False)  # [E+1] int32
    d_spec = ts(divisibility=(1, 8), static=False)  # [M,2N] n-major
    p_spec = ts(divisibility=(1, 8), static=False)  # [M,N]  n-major
    call = cjax.cutlass_call(
        launcher,
        output_shape_dtype=(
            jax.ShapeDtypeStruct((M, N2), x_sort.dtype),
            jax.ShapeDtypeStruct((M, N), x_sort.dtype),
        ),
        input_spec=(a_spec, b_spec, cu_spec),
        output_spec=(d_spec, p_spec),
        use_static_tensors=False,
    )
    preact, postact = call(x_sort, w_gate_up, cu_seqlens.astype(jnp.int32))
    return (preact, postact) if return_preact else postact


# ---- plain grouped GEMM (non-gated), for down projection + backward matmuls ----
from quack.gemm_default_epi import GemmDefaultSm100, GemmDefaultEpiMixin  # noqa: E402


def _build_plain_launcher(*, a_dtype, tile_mn, cluster_mnk, max_active_clusters, max_swizzle):
    @cute.jit
    def launcher(stream, mA, mB, mCuSeqlens, mD):
        gemm = GemmDefaultSm100(_ACC, a_dtype, tile_mn, cluster_mnk, gather_A=False, use_clc_persistence=False)
        epi_args = GemmDefaultEpiMixin.EpilogueArguments()
        scheduler_args = make_scheduler_args(max_active_clusters, max_swizzle, None)
        varlen_args = make_varlen_args(mCuSeqlens, None, None)
        gemm(mA, mB, mD, None, epi_args, scheduler_args, varlen_args, stream)

    return launcher


def quack_grouped_gemm(a, w, cu_seqlens, *, b_major="n", tile_mn=(256, 128), cluster_mnk=(2, 1, 1), max_swizzle=8):
    """Plain grouped GEMM a[M,K] @ w -> [M,N], grouped by cu_seqlens (varlen_m).

    b_major='n': w is [E,K,N] (n-major, mode (2,1,0)).  b_major='k': w is [E,N,K]
    (k-major, mode (1,2,0)) for transposed/backward contractions."""
    M, K = a.shape
    N = w.shape[2] if b_major == "n" else w.shape[1]
    _bmode = (2, 1, 0) if b_major == "n" else (1, 2, 0)
    a_dtype = _cute_dtype(a.dtype)
    try:
        mac = get_max_active_clusters(cluster_mnk[0] * cluster_mnk[1])
    except Exception:
        mac = 148
    launcher = _build_plain_launcher(
        a_dtype=a_dtype, tile_mn=tile_mn, cluster_mnk=cluster_mnk, max_active_clusters=mac, max_swizzle=max_swizzle
    )
    ts = cjax.TensorSpec
    a_spec = ts(divisibility=(1, 8), static=False)
    b_spec = ts(mode=_bmode, divisibility=(1, 1, 8), static=False)
    cu_spec = ts(static=False)
    d_spec = ts(divisibility=(1, 8), static=False)
    call = cjax.cutlass_call(
        launcher,
        output_shape_dtype=jax.ShapeDtypeStruct((M, N), a.dtype),
        input_spec=(a_spec, b_spec, cu_spec),
        output_spec=(d_spec,),
        use_static_tensors=False,
    )
    return call(a, w, cu_seqlens.astype(jnp.int32))


def _build_grouped_wgrad_launcher(*, a_dtype, tile_mn, cluster_mnk, max_active_clusters, max_swizzle):
    @cute.jit
    def launcher(stream, mA, mB, mCuSeqlens, mD):
        gemm = GemmDefaultSm100(_ACC, a_dtype, tile_mn, cluster_mnk, gather_A=False, use_clc_persistence=False)
        epi_args = GemmDefaultEpiMixin.EpilogueArguments()
        scheduler_args = make_scheduler_args(max_active_clusters, max_swizzle, None)
        varlen_args = make_varlen_args(None, mCuSeqlens, None)
        gemm(mA, mB, mD, None, epi_args, scheduler_args, varlen_args, stream)

    return launcher


def quack_grouped_wgrad(
    lhs,
    rhs,
    cu_seqlens,
    *,
    tile_mn=(256, 128),
    cluster_mnk=(2, 1, 1),
    max_swizzle=8,
):
    """Compute per-group ``lhs.T @ rhs`` with QuACK's SM100 variable-K GEMM.

    ``lhs`` and ``rhs`` share a token-major leading dimension and are grouped by
    ``cu_seqlens``. The result has shape ``[num_groups, lhs_width, rhs_width]``.
    Physical token-major inputs are exposed to CuTe as logical M-major/N-major
    operands without materializing transposes.
    """
    total_k, m = lhs.shape
    rhs_total_k, n = rhs.shape
    if rhs_total_k != total_k:
        raise ValueError(f"lhs rows={total_k} must equal rhs rows={rhs_total_k}")
    if lhs.dtype != rhs.dtype:
        raise ValueError(f"lhs dtype={lhs.dtype} must equal rhs dtype={rhs.dtype}")
    num_groups = cu_seqlens.shape[0] - 1
    a_dtype = _cute_dtype(lhs.dtype)
    try:
        max_active_clusters = get_max_active_clusters(cluster_mnk[0] * cluster_mnk[1])
    except Exception:
        max_active_clusters = 148
    launcher = _build_grouped_wgrad_launcher(
        a_dtype=a_dtype,
        tile_mn=tile_mn,
        cluster_mnk=cluster_mnk,
        max_active_clusters=max_active_clusters,
        max_swizzle=max_swizzle,
    )
    ts = cjax.TensorSpec
    # lhs/rhs are physically [total_k, width]. mode=(1, 0) exposes them as
    # logical [width, total_k] with unit stride along M/N as varlen-K requires.
    a_spec = ts(mode=(1, 0), divisibility=(1, 8), static=False)
    b_spec = ts(mode=(1, 0), divisibility=(1, 8), static=False)
    cu_spec = ts(static=False)
    # Physical [group, M, N] is logical [M, N, group] for the GEMM kernel.
    d_spec = ts(mode=(1, 2, 0), divisibility=(1, 1, 8), static=False)
    call = cjax.cutlass_call(
        launcher,
        output_shape_dtype=jax.ShapeDtypeStruct((num_groups, m, n), lhs.dtype),
        input_spec=(a_spec, b_spec, cu_spec),
        output_spec=(d_spec,),
        use_static_tensors=False,
    )
    return call(lhs, rhs, cu_seqlens.astype(jnp.int32))
