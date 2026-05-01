# Copyright The Levanter Authors
#
# SPDX-License-Identifier: Apache-2.0

import functools
import logging
import os
import warnings
from typing import Literal, TypeAlias

import jax
import jax.numpy as jnp

from ..partitioning import ResourceAxis

logger = logging.getLogger(__name__)

# Guard TPU-only megablox import; unavailable on GPU/CPU installs.
_gmm_megablox = None
try:
    from jax.experimental.pallas.ops.tpu.megablox import gmm as _gmm_megablox  # type: ignore[assignment]
except (ImportError, ModuleNotFoundError):
    pass

# Guard Pallas Triton import; unavailable on TPU/CPU installs.
_has_pallas_triton = False
try:
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import triton as plgpu

    _has_pallas_triton = True
except (ImportError, ModuleNotFoundError):
    pass

Implementation: TypeAlias = Literal["auto", "megablox", "triton", "xla"]
_AUTO_FALLBACK_EXCEPTIONS = (NotImplementedError, RuntimeError)
_HAS_WARNED_AUTO_FALLBACK = False


def _ragged_dot_megablox_impl(lhs: jax.Array, rhs: jax.Array, group_sizes: jax.Array) -> jax.Array:
    if _gmm_megablox is None:
        raise NotImplementedError("megablox GMM is not available (TPU-only)")
    tile_size = (512, 1024, 1024)  # (m, k, n)
    m, k, n = lhs.shape[0], lhs.shape[1], rhs.shape[2]
    return _gmm_megablox(
        lhs,
        rhs,
        group_sizes,
        preferred_element_type=lhs.dtype,
        tiling=(min(m, tile_size[0]), min(k, tile_size[1]), min(n, tile_size[2])),
        interpret=jax.default_backend() == "cpu",
    )


def _triton_ragged_dot_kernel(
    a_ref,
    b_ref,
    lo_ref,
    hi_ref,
    out_ref,
    *,
    block_m: int,
    block_k: int,
):
    """Pallas-Triton ragged dot kernel (no quantization)."""
    lo = lo_ref[()]
    hi = hi_ref[()]
    start_m = lo + pl.program_id(0) * block_m

    @pl.when(start_m < hi)
    def _compute():
        span_m = pl.ds(start_m, block_m)
        acc = jnp.zeros((block_m, out_ref.shape[1]), dtype=jnp.float32)
        k = a_ref.shape[1]

        def body(i, acc):
            start_k = i * block_k
            span_k = pl.ds(start_k, block_k)
            a = plgpu.load(a_ref.at[span_m, span_k])
            b = plgpu.load(b_ref.at[span_k, pl.ds(0, b_ref.shape[1])])
            dtype = jnp.result_type(a, b)
            return acc + pl.dot(a.astype(dtype), b.astype(dtype))

        num_k_blocks = pl.cdiv(k, block_k)
        acc = jax.lax.fori_loop(0, num_k_blocks, body, acc)
        mask = (start_m + jnp.arange(block_m)) < hi
        plgpu.store(out_ref.at[span_m, pl.ds(0, out_ref.shape[1])], acc.astype(out_ref.dtype), mask=mask[:, None])


def _triton_default_pallas_call(lhs: jax.Array, rhs: jax.Array, group_sizes: jax.Array) -> jax.Array:
    """Raw Pallas-Triton grouped matmul for the default ragged-dot layout."""
    m, k = lhs.shape
    num_groups, _, n = rhs.shape

    block_m = min(128, int(pl.next_power_of_2(m)))
    block_n = min(128, int(pl.next_power_of_2(n)))
    block_k = min(32, int(pl.next_power_of_2(k)))

    cum_rows = jnp.cumulative_sum(group_sizes, include_initial=True)

    return pl.pallas_call(
        lambda a, b, lo, hi, out: _triton_ragged_dot_kernel(a, b, lo, hi, out, block_m=block_m, block_k=block_k),
        out_shape=jax.ShapeDtypeStruct((m, n), lhs.dtype),
        in_specs=[
            pl.no_block_spec,
            pl.BlockSpec((None, k, block_n), lambda _, j, e: (e, 0, j)),
            pl.BlockSpec((None,), lambda _, __, e: (e,)),
            pl.BlockSpec((None,), lambda _, __, e: (e,)),
        ],
        out_specs=pl.BlockSpec((m, block_n), lambda _, j, __: (0, j)),
        grid=(pl.cdiv(m, block_m), pl.cdiv(n, block_n), num_groups),
        compiler_params=plgpu.CompilerParams(num_warps=4, num_stages=4),
    )(lhs, rhs, cum_rows[:-1], cum_rows[1:])


def _triton_ragged_contracting_dim_dot_kernel(
    a_ref,
    b_ref,
    lo_ref,
    hi_ref,
    out_ref,
    *,
    block_m: int,
    block_k: int,
):
    """Pallas-Triton ragged dot where the ragged dimension is also contracting."""
    lo = lo_ref[()]
    hi = hi_ref[()]

    def body(i, acc, mask_k=False):
        start_k = lo + i * block_k
        span_k = pl.ds(start_k, block_k)
        mask = None
        other = None
        if mask_k:
            mask = (jnp.arange(block_k) < hi - start_k)[:, None]
            other = 0.0
        a = plgpu.load(a_ref.at[span_k], mask=mask, other=other)
        b = plgpu.load(b_ref.at[span_k], mask=mask, other=other)
        dtype = jnp.result_type(a, b)
        return acc + pl.dot(a.astype(dtype).T, b.astype(dtype))

    num_k_blocks = jnp.maximum(pl.cdiv(jnp.int32(hi - lo), block_k), 1)
    acc = jnp.zeros((block_m, out_ref.shape[1]), dtype=jnp.float32)
    acc = jax.lax.fori_loop(0, num_k_blocks - 1, body, acc)
    acc = body(num_k_blocks - 1, acc, mask_k=True)
    plgpu.store(out_ref, acc.astype(out_ref.dtype))


def _triton_ragged_contracting_dim_pallas_call(
    lhs: jax.Array,
    rhs: jax.Array,
    group_sizes: jax.Array,
) -> jax.Array:
    """Raw Pallas-Triton grouped matmul for drhs-style ragged contraction."""
    k, m = lhs.shape
    _, n = rhs.shape

    block_m = min(128, int(pl.next_power_of_2(m)))
    block_n = min(128, int(pl.next_power_of_2(n)))
    block_k = min(32, int(pl.next_power_of_2(k)))

    cum_rows = jnp.cumulative_sum(group_sizes, include_initial=True)

    def one_group(lhs, rhs, lo, hi):
        return pl.pallas_call(
            lambda a, b, lo, hi, out: _triton_ragged_contracting_dim_dot_kernel(
                a,
                b,
                lo,
                hi,
                out,
                block_m=block_m,
                block_k=block_k,
            ),
            out_shape=jax.ShapeDtypeStruct((m, n), lhs.dtype),
            in_specs=[
                pl.BlockSpec((k, block_m), lambda i, j: (0, i)),
                pl.BlockSpec((k, block_n), lambda i, j: (0, j)),
                pl.no_block_spec,
                pl.no_block_spec,
            ],
            out_specs=pl.BlockSpec((block_m, block_n), lambda i, j: (i, j)),
            grid=(pl.cdiv(m, block_m), pl.cdiv(n, block_n)),
            compiler_params=plgpu.CompilerParams(num_warps=4, num_stages=4),
        )(lhs, rhs, lo, hi)

    return jax.vmap(one_group, in_axes=(None, None, 0, 0))(lhs, rhs, cum_rows[:-1], cum_rows[1:])


_DEFAULT_DIM_NUMS = jax.lax.RaggedDotDimensionNumbers(
    dot_dimension_numbers=(((1,), (1,)), ((), ())),
    lhs_ragged_dimensions=(0,),
    rhs_group_dimensions=(0,),
)

# Dimension numbers for the dlhs backward pass: dout[M,N] @ rhs[G,K,N]^T → dlhs[M,K]
# Contracts over N (dout dim 1 with rhs dim 2), groups on rhs dim 0.
_DLHS_DIM_NUMS = jax.lax.RaggedDotDimensionNumbers(
    dot_dimension_numbers=(((1,), (2,)), ((), ())),
    lhs_ragged_dimensions=(0,),
    rhs_group_dimensions=(0,),
)

# Dimension numbers for the drhs backward pass: lhs[M,K]^T @ dout[M,N] → drhs[G,K,N]
# Contracts over M (lhs dim 0 with dout dim 0), ragged on lhs dim 0, no group dim.
_DRHS_DIM_NUMS = jax.lax.RaggedDotDimensionNumbers(
    dot_dimension_numbers=(((0,), (0,)), ((), ())),
    lhs_ragged_dimensions=(0,),
    rhs_group_dimensions=[],
)


def _triton_pallas_call(
    lhs: jax.Array,
    rhs: jax.Array,
    group_sizes: jax.Array,
    ragged_dot_dimension_numbers: jax.lax.RaggedDotDimensionNumbers = _DEFAULT_DIM_NUMS,
) -> jax.Array:
    """Raw Pallas-Triton grouped matmul for supported ragged-dot layouts."""
    if ragged_dot_dimension_numbers == _DEFAULT_DIM_NUMS:
        return _triton_default_pallas_call(lhs, rhs, group_sizes)
    if ragged_dot_dimension_numbers == _DLHS_DIM_NUMS:
        return _triton_default_pallas_call(lhs, rhs.mT, group_sizes)
    if ragged_dot_dimension_numbers == _DRHS_DIM_NUMS:
        return _triton_ragged_contracting_dim_pallas_call(lhs, rhs, group_sizes)
    raise NotImplementedError(f"Unsupported ragged dot dimension numbers for Triton: {ragged_dot_dimension_numbers}")


@functools.partial(jax.custom_vjp, nondiff_argnums=())
def _ragged_dot_triton_impl(lhs: jax.Array, rhs: jax.Array, group_sizes: jax.Array) -> jax.Array:
    """Pallas-Triton grouped matmul with explicit backward pass.

    Uses custom_vjp so JAX never tries to autodiff directly through pallas_call.
    Direct autodiff still fails for this kernel on JAX 0.9.2, while the explicit
    VJP can use the Triton kernels for each ragged-dot contraction layout.
    """
    if not _has_pallas_triton:
        raise NotImplementedError("Pallas Triton backend is not available")
    return _triton_pallas_call(lhs, rhs, group_sizes)


def _ragged_dot_triton_fwd(lhs, rhs, group_sizes):
    out = _triton_pallas_call(lhs, rhs, group_sizes)
    return out, (lhs, rhs, group_sizes)


def _ragged_dot_triton_bwd(residuals, dout):
    lhs, rhs, group_sizes = residuals

    # dlhs[M,K] = dout[M,N] @ rhs[G,K,N]^T
    dlhs = _triton_pallas_call(dout, rhs, group_sizes, _DLHS_DIM_NUMS)

    # drhs[G,K,N] = lhs[M,K]^T @ dout[M,N]
    drhs = _triton_pallas_call(lhs, dout, group_sizes, _DRHS_DIM_NUMS)

    return dlhs, drhs, None  # None for group_sizes (integer, no gradient)


_ragged_dot_triton_impl.defvjp(_ragged_dot_triton_fwd, _ragged_dot_triton_bwd)


def _ragged_dot_xla_impl(lhs: jax.Array, rhs: jax.Array, group_sizes: jax.Array) -> jax.Array:
    return jax.lax.ragged_dot_general(
        lhs=lhs,
        rhs=rhs,
        group_sizes=group_sizes,
        ragged_dot_dimension_numbers=jax.lax.RaggedDotDimensionNumbers(
            dot_dimension_numbers=(((1,), (1,)), ((), ())),
            lhs_ragged_dimensions=(0,),
            rhs_group_dimensions=(0,),
        ),
    )


def _preferred_implementations(implementation: Implementation) -> tuple[Implementation, ...]:
    # Allow override via env var for A/B benchmarking:
    #   RAGGED_DOT_IMPL=xla     → force XLA
    #   RAGGED_DOT_IMPL=triton  → force Triton
    env_override = os.environ.get("RAGGED_DOT_IMPL")
    if env_override is not None:
        return (env_override,)  # type: ignore[return-value]

    if implementation != "auto":
        return (implementation,)

    if jax.default_backend() == "tpu":
        return ("megablox", "xla")

    if jax.default_backend() == "gpu" and _has_pallas_triton:
        return ("triton", "xla")

    return ("xla",)


def _run_impl(name: Implementation, lhs: jax.Array, rhs: jax.Array, group_sizes: jax.Array) -> jax.Array:
    if name == "megablox":
        return _ragged_dot_megablox_impl(lhs, rhs, group_sizes)
    if name == "triton":
        return _ragged_dot_triton_impl(lhs, rhs, group_sizes)
    if name == "xla":
        return _ragged_dot_xla_impl(lhs, rhs, group_sizes)
    raise ValueError(f"Unknown ragged_dot implementation: {name}")


def ragged_dot(
    lhs_: jax.Array,
    rhs_: jax.Array,
    group_sizes_: jax.Array,
    ar: bool = False,
    implementation: Implementation = "auto",
) -> jax.Array:
    """Grouped matrix multiply with backend-dispatched ragged dot implementations.

    Args:
        lhs_: [tokens, in] input matrix.
        rhs_: [experts, in, out] expert weights.
        group_sizes_: [experts] number of tokens per expert.
        ar: Whether to perform an all-reduce over the model axis on the output.
        implementation: Backend selection. ``"auto"`` selects per-platform default.
            ``"triton"`` forces GPU Pallas Triton kernel. ``"megablox"`` forces
            TPU megablox. ``"xla"`` forces ``jax.lax.ragged_dot_general``.

    Returns:
        A [tokens, out] array.
    """
    hs_shape = lhs_.shape
    if hs_shape[0] % 512:
        pad_length = 512 - hs_shape[0] % 512
        lhs_ = jax.lax.pad(lhs_, jnp.zeros((), dtype=lhs_.dtype), [(0, pad_length, 0), (0, 0, 0)])

    out = None

    for impl in _preferred_implementations(implementation):
        try:
            out = _run_impl(impl, lhs_, rhs_, group_sizes_)
            break
        except _AUTO_FALLBACK_EXCEPTIONS as exc:
            if implementation == "auto" and impl != "xla":
                global _HAS_WARNED_AUTO_FALLBACK
                if not _HAS_WARNED_AUTO_FALLBACK:
                    warnings.warn(
                        f"ragged_dot auto fallback: {impl} failed ({type(exc).__name__}), trying next.",
                        RuntimeWarning,
                    )
                    _HAS_WARNED_AUTO_FALLBACK = True
                continue
            raise

    if out is None:
        raise RuntimeError("No ragged_dot implementation was selected")

    if ar:
        out = jax.lax.psum(out, ResourceAxis.MODEL)

    if hs_shape[0] % 512:
        out = out[: hs_shape[0]]

    return out


__all__ = ["Implementation", "ragged_dot"]
