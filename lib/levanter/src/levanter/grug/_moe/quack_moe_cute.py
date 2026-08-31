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
from quack.gemm_default_epi import GemmDefaultEpiMixin, GemmDefaultSm100
from levanter.cutlass_kernel_cache import cute_launcher_factory, cutlass_call
from quack.gemm_tvm_ffi_utils import make_scheduler_args, make_varlen_args

_ACC = cutlass.Float32
_FALLBACK_MAX_ACTIVE_CLUSTERS = 148
# Vector width the tensor specs declare to the kernel for the non-grouped (feature) dimensions.
# This is a fixed requirement of the TMA loads, not a tuning knob: the tuned values are the tile,
# cluster, and CLC settings, which callers pass in (see `_QUACK_WGRAD_KW` in `sonic_cute`).
_FEATURE_ALIGNMENT = 8
_JAX_TO_CUTE = {
    jnp.dtype(jnp.bfloat16): cutlass.BFloat16,
    jnp.dtype(jnp.float16): cutlass.Float16,
    jnp.dtype(jnp.float32): cutlass.Float32,
}


def _cute_dtype(dt):
    return _JAX_TO_CUTE[jnp.dtype(dt)]


def _max_active_clusters(cluster_mnk) -> int:
    """Clusters the device can hold, or a fixed stand-in when there is no device to ask.

    `get_max_active_clusters` needs a GPU. CPU tracing must still build the launcher, so it gets
    a constant: the value only sizes the tile scheduler and never changes the computed function.
    """
    if jax.default_backend() == "cpu":
        return _FALLBACK_MAX_ACTIVE_CLUSTERS
    return get_max_active_clusters(cluster_mnk[0] * cluster_mnk[1])


@cute_launcher_factory
def _build_launcher(
    *, a_dtype, tile_mn, cluster_mnk, activation, max_active_clusters, max_swizzle, use_clc_persistence=False
):
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
            use_clc_persistence=use_clc_persistence,
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
    use_clc_persistence=False,
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
    max_active_clusters = _max_active_clusters(cluster_mnk)
    launcher = _build_launcher(
        a_dtype=a_dtype,
        tile_mn=tile_mn,
        cluster_mnk=cluster_mnk,
        activation=activation,
        max_active_clusters=max_active_clusters,
        max_swizzle=max_swizzle,
        use_clc_persistence=use_clc_persistence,
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
    call = cutlass_call(
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


@cute_launcher_factory
def _build_plain_launcher(
    *, a_dtype, tile_mn, cluster_mnk, max_active_clusters, max_swizzle, use_clc_persistence=False, ragged_axis
):
    """Return a ``@cute.jit`` plain grouped-GEMM launcher.

    ``ragged_axis`` picks which of the kernel's two grouping modes ``cu_seqlens`` drives: "m"
    groups over rows, "k" over the contraction dimension. It is the only difference between the
    activation-path GEMMs and the weight gradients, and it is required rather than defaulted
    because the launcher cache builds its identity from the keywords a caller actually passes.
    """
    if ragged_axis not in ("m", "k"):
        raise ValueError(f"ragged_axis must be 'm' or 'k', got {ragged_axis!r}")
    # Pick the slot while building the trace. An in-body branch would have to be wrapped in
    # `cutlass.const_expr` to survive preprocessing, and closing over the choice keeps the
    # launcher body free of grouping-mode logic entirely.
    build_varlen_args = (
        (lambda cu: make_varlen_args(cu, None, None))
        if ragged_axis == "m"
        else (lambda cu: make_varlen_args(None, cu, None))
    )

    @cute.jit
    def launcher(stream, mA, mB, mCuSeqlens, mD):
        gemm = GemmDefaultSm100(
            _ACC, a_dtype, tile_mn, cluster_mnk, gather_A=False, use_clc_persistence=use_clc_persistence
        )
        epi_args = GemmDefaultEpiMixin.EpilogueArguments()
        scheduler_args = make_scheduler_args(max_active_clusters, max_swizzle, None)
        gemm(mA, mB, mD, None, epi_args, scheduler_args, build_varlen_args(mCuSeqlens), stream)

    return launcher


def _grouped_gemm_call(a, b, cu_seqlens, *, ragged_axis, a_spec, b_spec, d_spec, out, **config):
    """Build the launcher and run the `cutlass_call` that both grouping modes share.

    The two modes differ only in `ragged_axis`, in the tensor specs that describe each operand's
    layout, and in the output shape. Everything between those is identical, so it lives here.
    """
    launcher = _build_plain_launcher(
        ragged_axis=ragged_axis,
        a_dtype=_cute_dtype(a.dtype),
        max_active_clusters=_max_active_clusters(config["cluster_mnk"]),
        **config,
    )
    call = cutlass_call(
        launcher,
        output_shape_dtype=out,
        input_spec=(a_spec, b_spec, cjax.TensorSpec(static=False)),
        output_spec=(d_spec,),
        use_static_tensors=False,
    )
    return call(a, b, cu_seqlens.astype(jnp.int32))


def quack_grouped_gemm(
    a,
    w,
    cu_seqlens,
    *,
    b_major="n",
    tile_mn=(256, 128),
    cluster_mnk=(2, 1, 1),
    max_swizzle=8,
    use_clc_persistence=False,
):
    """Plain grouped GEMM a[M,K] @ w -> [M,N], grouped by cu_seqlens (varlen_m).

    b_major='n': w is [E,K,N] (n-major, mode (2,1,0)).  b_major='k': w is [E,N,K]
    (k-major, mode (1,2,0)) for transposed/backward contractions."""
    M = a.shape[0]
    N = w.shape[2] if b_major == "n" else w.shape[1]
    ts = cjax.TensorSpec
    return _grouped_gemm_call(
        a,
        w,
        cu_seqlens,
        ragged_axis="m",
        a_spec=ts(divisibility=(1, _FEATURE_ALIGNMENT), static=False),
        b_spec=ts(
            mode=(2, 1, 0) if b_major == "n" else (1, 2, 0), divisibility=(1, 1, _FEATURE_ALIGNMENT), static=False
        ),
        d_spec=ts(divisibility=(1, _FEATURE_ALIGNMENT), static=False),
        out=jax.ShapeDtypeStruct((M, N), a.dtype),
        tile_mn=tile_mn,
        cluster_mnk=cluster_mnk,
        max_swizzle=max_swizzle,
        use_clc_persistence=use_clc_persistence,
    )


def quack_grouped_wgrad(
    lhs,
    rhs,
    cu_seqlens,
    *,
    tile_mn=(128, 128),
    cluster_mnk=(2, 1, 1),
    max_swizzle=8,
    use_clc_persistence=False,
):
    """Per-expert ``lhs.T @ rhs`` over contiguous ragged row groups, on QuACK's varlen-k GEMM.

    ``lhs`` is [total_rows, M], ``rhs`` is [total_rows, N], and ``cu_seqlens`` [E+1] splits the
    rows into expert groups; the result is [E, M, N]. The rows are the contraction dimension, so
    this is the varlen-k grouping (``cu_seqlens_k``) rather than the varlen-m one the activation
    -path GEMMs use.

    The kernel reads each group through a coordinate offset into a descriptor spanning the whole
    buffer, so the group boundaries need no alignment and rows past ``cu_seqlens[-1]`` are never
    read. An empty group clears its accumulator and yields a zero gradient.
    """
    if lhs.ndim != 2 or rhs.ndim != 2:
        raise ValueError(f"lhs and rhs must be rank 2, got lhs={lhs.shape}, rhs={rhs.shape}")
    if lhs.shape[0] != rhs.shape[0]:
        raise ValueError(f"lhs and rhs row counts must match, got lhs={lhs.shape}, rhs={rhs.shape}")
    if lhs.dtype != rhs.dtype:
        raise ValueError(f"lhs and rhs dtypes must match, got lhs={lhs.dtype}, rhs={rhs.dtype}")
    # float32 operands would be accepted and then quietly demoted: the kernel builds its TMA atoms
    # with `internal_type=TFloat32` for an fp32 element type, so the weight gradient would come
    # back with a 10-bit mantissa and no diagnostic. Refuse instead.
    if lhs.dtype.itemsize != 2:
        raise ValueError(f"grouped Wgrad requires a 16-bit float, got {lhs.dtype}")
    # The tensor specs below promise the kernel this much vectorisation on the feature dims. The
    # group boundaries need no alignment, but these do.
    if lhs.shape[1] % _FEATURE_ALIGNMENT or rhs.shape[1] % _FEATURE_ALIGNMENT:
        raise ValueError(f"feature dimensions must divide {_FEATURE_ALIGNMENT}, got lhs={lhs.shape}, rhs={rhs.shape}")
    # The group count comes from this array's static shape, so an empty or higher-rank one would
    # otherwise reach the launcher as a zero-group or nonsense launch.
    if cu_seqlens.ndim != 1 or cu_seqlens.shape[0] < 2:
        raise ValueError(f"cu_seqlens must be a rank-1 array of at least two offsets, got {cu_seqlens.shape}")

    ts = cjax.TensorSpec
    # Logical order is (M, K) for A, (N, K) for B, (M, N, L) for D; `mode` maps each logical axis
    # to the physical one it comes from, and `divisibility` stays in physical order. varlen_k wants
    # A m-major and B n-major, which is what [rows, M] and [rows, N] already are.
    return _grouped_gemm_call(
        lhs,
        rhs,
        cu_seqlens,
        ragged_axis="k",
        a_spec=ts(mode=(1, 0), divisibility=(1, _FEATURE_ALIGNMENT), static=False),
        b_spec=ts(mode=(1, 0), divisibility=(1, _FEATURE_ALIGNMENT), static=False),
        d_spec=ts(mode=(1, 2, 0), divisibility=(1, 1, _FEATURE_ALIGNMENT), static=False),
        out=jax.ShapeDtypeStruct((cu_seqlens.shape[0] - 1, lhs.shape[1], rhs.shape[1]), lhs.dtype),
        tile_mn=tile_mn,
        cluster_mnk=cluster_mnk,
        max_swizzle=max_swizzle,
        use_clc_persistence=use_clc_persistence,
    )
