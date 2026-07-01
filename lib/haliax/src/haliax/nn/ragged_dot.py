# Copyright The Levanter Authors
#
# SPDX-License-Identifier: Apache-2.0

import functools
import logging
import os
import warnings
from collections.abc import Callable
from typing import Literal, TypeAlias

import jax
import jax.numpy as jnp
from jax.typing import DTypeLike

from haliax._src.fp8 import dequantize, in_q, out_dq, quantize, update_fp8_meta
from haliax.nn._fp8_mosaic_ragged import (
    mosaic_ragged_dot,
    mosaic_transposed_ragged_dot,
)
from haliax.partitioning import ResourceAxis

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

# Guard native Triton import; unavailable on CPU-only installs.
_has_jax_triton = False
try:
    import jax_triton as jt
    import triton
    import triton.language as tl

    _has_jax_triton = True
except (ImportError, ModuleNotFoundError):
    jt = None
    triton = None
    tl = None

# Adapted from openxla/tokamax@ad75b704:
# tokamax/_src/ops/ragged_dot/pallas_triton.py. In particular:
# _ragged_dot_kernel/_ragged_dot for the default layout,
# _ragged_contracting_dim_dot_kernel/_ragged_contracting_dim_dot for drhs, and
# PallasTritonRaggedDot._fwd for the VJP layout dispatch.
Implementation: TypeAlias = Literal["auto", "megablox", "mosaic", "padded_dense", "triton", "triton_native", "xla"]
_AUTO_FALLBACK_EXCEPTIONS = (NotImplementedError, RuntimeError)
_HAS_WARNED_AUTO_FALLBACK = False
_TRITON_DEFAULT_BLOCK_N = 128
_TRITON_BLACKWELL_BLOCK_N = 256


def _is_blackwell_gpu_backend() -> bool:
    if jax.default_backend() != "gpu":
        return False
    try:
        devices = jax.devices("gpu")
    except RuntimeError:
        return False
    if not devices:
        return False
    device = devices[0]
    compute_capability = getattr(device, "compute_capability", None)
    if compute_capability is not None:
        try:
            return float(compute_capability) >= 10.0
        except (TypeError, ValueError):
            pass
    device_kind = getattr(device, "device_kind", "")
    return any(name in device_kind for name in ("B200", "B300", "GB200", "GB300"))


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
            return acc + pl.dot(a, b)

        num_k_blocks = pl.cdiv(k, block_k)
        acc = jax.lax.fori_loop(0, num_k_blocks, body, acc)
        mask = (start_m + jnp.arange(block_m)) < hi
        plgpu.store(out_ref.at[span_m, pl.ds(0, out_ref.shape[1])], acc.astype(out_ref.dtype), mask=mask[:, None])


def _triton_default_block_sizes(m: int, k: int, n: int) -> tuple[int, int, int]:
    block_m = min(128, int(pl.next_power_of_2(m)))
    max_block_n = _TRITON_BLACKWELL_BLOCK_N if _is_blackwell_gpu_backend() else _TRITON_DEFAULT_BLOCK_N
    block_n = min(max_block_n, int(pl.next_power_of_2(n)))
    block_k = min(32, int(pl.next_power_of_2(k)))
    return block_m, block_n, block_k


def _triton_default_pallas_call(
    lhs: jax.Array,
    rhs: jax.Array,
    group_sizes: jax.Array,
    *,
    out_dtype: DTypeLike | None = None,
    max_group_size: int | None = None,
    block_m_override: int | None = None,
    block_n_override: int | None = None,
    block_k_override: int | None = None,
) -> jax.Array:
    """Raw Pallas-Triton grouped matmul for the default ragged-dot layout."""
    m, k = lhs.shape
    num_groups, _, n = rhs.shape

    default_block_m, default_block_n, default_block_k = _triton_default_block_sizes(m, k, n)
    block_m = default_block_m if block_m_override is None else block_m_override
    block_n = default_block_n if block_n_override is None else block_n_override
    block_k = default_block_k if block_k_override is None else block_k_override

    cum_rows = jnp.cumulative_sum(group_sizes, include_initial=True)
    out_dtype = lhs.dtype if out_dtype is None else out_dtype
    grid_m = m if max_group_size is None else max_group_size

    return pl.pallas_call(
        lambda a, b, lo, hi, out: _triton_ragged_dot_kernel(a, b, lo, hi, out, block_m=block_m, block_k=block_k),
        out_shape=jax.ShapeDtypeStruct((m, n), out_dtype),
        in_specs=[
            pl.no_block_spec,
            pl.BlockSpec((None, k, block_n), lambda _, j, e: (e, 0, j)),
            pl.BlockSpec((None,), lambda _, __, e: (e,)),
            pl.BlockSpec((None,), lambda _, __, e: (e,)),
        ],
        out_specs=pl.BlockSpec((m, block_n), lambda _, j, __: (0, j)),
        grid=(pl.cdiv(grid_m, block_m), pl.cdiv(n, block_n), num_groups),
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
        return acc + pl.dot(a.T, b)

    num_k_blocks = jnp.maximum(pl.cdiv(jnp.int32(hi - lo), jnp.int32(block_k)), jnp.int32(1))
    acc = jnp.zeros((block_m, out_ref.shape[1]), dtype=jnp.float32)
    acc = jax.lax.fori_loop(0, num_k_blocks - 1, body, acc)
    acc = body(num_k_blocks - 1, acc, mask_k=True)
    plgpu.store(out_ref, acc.astype(out_ref.dtype))


def _triton_ragged_contracting_dim_pallas_call(
    lhs: jax.Array,
    rhs: jax.Array,
    group_sizes: jax.Array,
    *,
    out_dtype: DTypeLike | None = None,
    block_m_override: int | None = None,
    block_n_override: int | None = None,
    block_k_override: int | None = None,
) -> jax.Array:
    """Raw Pallas-Triton grouped matmul for drhs-style ragged contraction."""
    k, m = lhs.shape
    _, n = rhs.shape

    block_m = min(128, int(pl.next_power_of_2(m))) if block_m_override is None else block_m_override
    block_n = min(128, int(pl.next_power_of_2(n))) if block_n_override is None else block_n_override
    block_k = min(32, int(pl.next_power_of_2(k))) if block_k_override is None else block_k_override
    block_m = min(block_m, m)
    block_n = min(block_n, n)

    cum_rows = jnp.cumulative_sum(group_sizes, include_initial=True)
    out_dtype = lhs.dtype if out_dtype is None else out_dtype

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
            out_shape=jax.ShapeDtypeStruct((m, n), out_dtype),
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


if triton is not None and tl is not None:

    @triton.jit
    def _native_ragged_dot_kernel(
        a_ptr,
        b_ptr,
        lo_ptr,
        hi_ptr,
        out_ptr,
        k: tl.constexpr,
        n: tl.constexpr,
        block_m: tl.constexpr,
        block_n: tl.constexpr,
        block_k: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        expert = tl.program_id(2)
        lo = tl.load(lo_ptr + expert)
        hi = tl.load(hi_ptr + expert)

        offs_m = lo + pid_m * block_m + tl.arange(0, block_m)
        offs_n = pid_n * block_n + tl.arange(0, block_n)
        offs_k = tl.arange(0, block_k)
        acc = tl.zeros((block_m, block_n), dtype=tl.float32)

        for k0 in range(0, k, block_k):
            k_idxs = k0 + offs_k
            a = tl.load(
                a_ptr + offs_m[:, None] * k + k_idxs[None, :],
                mask=(offs_m[:, None] < hi) & (k_idxs[None, :] < k),
                other=0.0,
            )
            b = tl.load(
                b_ptr + expert * k * n + k_idxs[:, None] * n + offs_n[None, :],
                mask=(k_idxs[:, None] < k) & (offs_n[None, :] < n),
                other=0.0,
            )
            acc += tl.dot(a, b)

        tl.store(
            out_ptr + offs_m[:, None] * n + offs_n[None, :],
            acc,
            mask=(offs_m[:, None] < hi) & (offs_n[None, :] < n),
        )

    @triton.jit
    def _native_ragged_dlhs_kernel(
        g_ptr,
        b_ptr,
        lo_ptr,
        hi_ptr,
        out_ptr,
        k: tl.constexpr,
        n: tl.constexpr,
        block_m: tl.constexpr,
        block_n: tl.constexpr,
        block_k: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_k = tl.program_id(1)
        expert = tl.program_id(2)
        lo = tl.load(lo_ptr + expert)
        hi = tl.load(hi_ptr + expert)

        offs_m = lo + pid_m * block_m + tl.arange(0, block_m)
        offs_k = pid_k * block_k + tl.arange(0, block_k)
        offs_n = tl.arange(0, block_n)
        acc = tl.zeros((block_m, block_k), dtype=tl.float32)

        for n0 in range(0, n, block_n):
            n_idxs = n0 + offs_n
            g = tl.load(
                g_ptr + offs_m[:, None] * n + n_idxs[None, :],
                mask=(offs_m[:, None] < hi) & (n_idxs[None, :] < n),
                other=0.0,
            )
            b = tl.load(
                b_ptr + expert * k * n + offs_k[None, :] * n + n_idxs[:, None],
                mask=(offs_k[None, :] < k) & (n_idxs[:, None] < n),
                other=0.0,
            )
            acc += tl.dot(g, b)

        tl.store(
            out_ptr + offs_m[:, None] * k + offs_k[None, :],
            acc,
            mask=(offs_m[:, None] < hi) & (offs_k[None, :] < k),
        )

    @triton.jit
    def _native_ragged_drhs_kernel(
        a_ptr,
        g_ptr,
        lo_ptr,
        hi_ptr,
        out_ptr,
        k: tl.constexpr,
        n: tl.constexpr,
        block_m: tl.constexpr,
        block_n: tl.constexpr,
        block_k: tl.constexpr,
    ):
        pid_k = tl.program_id(0)
        pid_n = tl.program_id(1)
        expert = tl.program_id(2)
        lo = tl.load(lo_ptr + expert)
        hi = tl.load(hi_ptr + expert)

        offs_k = pid_k * block_k + tl.arange(0, block_k)
        offs_n = pid_n * block_n + tl.arange(0, block_n)
        offs_m = tl.arange(0, block_m)
        acc = tl.zeros((block_k, block_n), dtype=tl.float32)

        for block in tl.range(0, tl.cdiv(hi - lo, block_m)):
            m0 = block * block_m
            m_idxs = lo + m0 + offs_m
            a = tl.load(
                a_ptr + m_idxs[None, :] * k + offs_k[:, None],
                mask=(m_idxs[None, :] < hi) & (offs_k[:, None] < k),
                other=0.0,
            )
            g = tl.load(
                g_ptr + m_idxs[:, None] * n + offs_n[None, :],
                mask=(m_idxs[:, None] < hi) & (offs_n[None, :] < n),
                other=0.0,
            )
            acc += tl.dot(a, g)

        tl.store(
            out_ptr + expert * k * n + offs_k[:, None] * n + offs_n[None, :],
            acc,
            mask=(offs_k[:, None] < k) & (offs_n[None, :] < n),
        )

else:
    _native_ragged_dot_kernel = None
    _native_ragged_dlhs_kernel = None
    _native_ragged_drhs_kernel = None


def _triton_native_pallas_call(
    lhs: jax.Array,
    rhs: jax.Array,
    group_sizes: jax.Array,
    ragged_dot_dimension_numbers: jax.lax.RaggedDotDimensionNumbers = _DEFAULT_DIM_NUMS,
    *,
    out_dtype: DTypeLike,
    max_group_size: int | None,
    block_m: int | None,
    block_n: int | None,
    block_k: int | None,
) -> jax.Array:
    if not _has_jax_triton or jt is None:
        raise NotImplementedError("jax_triton native Triton backend is not available")

    block_m = 128 if block_m is None else block_m
    block_n = 128 if block_n is None else block_n
    block_k = 64 if block_k is None else block_k
    cum_rows = jnp.cumulative_sum(group_sizes, include_initial=True)

    if ragged_dot_dimension_numbers == _DEFAULT_DIM_NUMS:
        m, k = lhs.shape
        experts, _, n = rhs.shape
        grid_m = m if max_group_size is None else max_group_size
        return jt.triton_call(
            lhs,
            rhs,
            cum_rows[:-1],
            cum_rows[1:],
            kernel=_native_ragged_dot_kernel,
            out_shape=jax.ShapeDtypeStruct((m, n), out_dtype),
            grid=(triton.cdiv(grid_m, block_m), triton.cdiv(n, block_n), experts),
            num_warps=4,
            num_stages=4,
            k=k,
            n=n,
            block_m=block_m,
            block_n=block_n,
            block_k=block_k,
        )

    if ragged_dot_dimension_numbers == _DLHS_DIM_NUMS:
        m, n = lhs.shape
        experts, k, _ = rhs.shape
        grid_m = m if max_group_size is None else max_group_size
        return jt.triton_call(
            lhs,
            rhs,
            cum_rows[:-1],
            cum_rows[1:],
            kernel=_native_ragged_dlhs_kernel,
            out_shape=jax.ShapeDtypeStruct((m, k), out_dtype),
            grid=(triton.cdiv(grid_m, block_m), triton.cdiv(k, block_k), experts),
            num_warps=4,
            num_stages=4,
            k=k,
            n=n,
            block_m=block_m,
            block_n=block_n,
            block_k=block_k,
        )

    if ragged_dot_dimension_numbers == _DRHS_DIM_NUMS:
        m, k = lhs.shape
        _, n = rhs.shape
        experts = group_sizes.shape[0]
        return jt.triton_call(
            lhs,
            rhs,
            cum_rows[:-1],
            cum_rows[1:],
            kernel=_native_ragged_drhs_kernel,
            out_shape=jax.ShapeDtypeStruct((experts, k, n), out_dtype),
            grid=(triton.cdiv(k, block_k), triton.cdiv(n, block_n), experts),
            num_warps=4,
            num_stages=4,
            k=k,
            n=n,
            block_m=block_m,
            block_n=block_n,
            block_k=block_k,
        )

    raise NotImplementedError(f"Unsupported ragged dot dimension numbers for native Triton: {ragged_dot_dimension_numbers}")


def _triton_pallas_call(
    lhs: jax.Array,
    rhs: jax.Array,
    group_sizes: jax.Array,
    ragged_dot_dimension_numbers: jax.lax.RaggedDotDimensionNumbers = _DEFAULT_DIM_NUMS,
    *,
    out_dtype: DTypeLike | None = None,
    max_group_size: int | None = None,
    block_m: int | None = None,
    block_n: int | None = None,
    block_k: int | None = None,
) -> jax.Array:
    """Raw Pallas-Triton grouped matmul for supported ragged-dot layouts."""
    if ragged_dot_dimension_numbers == _DEFAULT_DIM_NUMS:
        return _triton_default_pallas_call(
            lhs,
            rhs,
            group_sizes,
            out_dtype=out_dtype,
            max_group_size=max_group_size,
            block_m_override=block_m,
            block_n_override=block_n,
            block_k_override=block_k,
        )
    if ragged_dot_dimension_numbers == _DLHS_DIM_NUMS:
        return _triton_default_pallas_call(
            lhs,
            rhs.mT,
            group_sizes,
            out_dtype=out_dtype,
            max_group_size=max_group_size,
            block_m_override=block_m,
            block_n_override=block_n,
            block_k_override=block_k,
        )
    if ragged_dot_dimension_numbers == _DRHS_DIM_NUMS:
        return _triton_ragged_contracting_dim_pallas_call(
            lhs,
            rhs,
            group_sizes,
            out_dtype=out_dtype,
            block_m_override=block_m,
            block_n_override=block_n,
            block_k_override=block_k,
        )
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


def _pack_ragged_rows(x: jax.Array, group_sizes: jax.Array, max_group_size: int) -> jax.Array:
    starts = jnp.cumulative_sum(group_sizes, include_initial=True)[:-1]
    padded = jnp.pad(x, ((0, max_group_size), (0, 0)))

    def pack_one(start, size):
        rows = jax.lax.dynamic_slice(padded, (start, 0), (max_group_size, x.shape[1]))
        mask = jnp.arange(max_group_size, dtype=size.dtype) < size
        return jnp.where(mask[:, None], rows, jnp.zeros((), dtype=x.dtype))

    return jax.vmap(pack_one)(starts, group_sizes)


def _scatter_packed_rows(packed: jax.Array, group_sizes: jax.Array, tokens: int) -> jax.Array:
    starts = jnp.cumulative_sum(group_sizes, include_initial=True)[:-1]
    max_group_size = packed.shape[1]
    offsets = jnp.arange(max_group_size, dtype=group_sizes.dtype)
    indices = starts[:, None] + offsets[None, :]
    mask = offsets[None, :] < group_sizes[:, None]
    values = jnp.where(mask[:, :, None], packed, jnp.zeros((), dtype=packed.dtype))
    indices = jnp.where(mask, indices, jnp.zeros((), dtype=indices.dtype))
    return jnp.zeros((tokens, packed.shape[2]), dtype=packed.dtype).at[indices].add(values)


def _block_pad_ragged_token_stream(
    lhs: jax.Array,
    rhs: jax.Array,
    group_sizes: jax.Array,
    block_k: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    tokens = lhs.shape[0]
    groups = group_sizes.shape[0]
    padded_bound = tokens + groups * (block_k - 1)
    group_ends = jnp.cumulative_sum(group_sizes)
    group_starts = jnp.concatenate([jnp.zeros((1,), dtype=group_sizes.dtype), group_ends[:-1]])
    padded_group_sizes = ((group_sizes + block_k - 1) // block_k) * block_k
    padded_starts = jnp.cumulative_sum(padded_group_sizes, include_initial=True)[:-1]

    token_ids = jnp.arange(tokens, dtype=group_sizes.dtype)
    group_ids = jnp.searchsorted(group_ends, token_ids, side="right").astype(group_sizes.dtype)
    token_offsets = token_ids - group_starts[group_ids]
    padded_token_ids = padded_starts[group_ids] + token_offsets

    padded_lhs = jnp.zeros((padded_bound, lhs.shape[1]), dtype=lhs.dtype).at[padded_token_ids].set(lhs)
    padded_rhs = jnp.zeros((padded_bound, rhs.shape[1]), dtype=rhs.dtype).at[padded_token_ids].set(rhs)
    return padded_lhs, padded_rhs, padded_group_sizes


def _padded_dense_ragged_dot_impl(
    lhs: jax.Array,
    rhs: jax.Array,
    group_sizes: jax.Array,
    ragged_dot_dimension_numbers: jax.lax.RaggedDotDimensionNumbers,
    out_dtype: DTypeLike,
    max_group_size: int | None,
) -> jax.Array:
    """FP8 ragged contractions via dynamic ragged pack + padded batched GEMM."""
    if max_group_size is None:
        raise ValueError("max_group_size is required for the padded_dense FP8 ragged_dot path")

    if ragged_dot_dimension_numbers == _DEFAULT_DIM_NUMS:
        packed_lhs = _pack_ragged_rows(lhs, group_sizes, max_group_size)
        packed_out = jax.lax.dot_general(
            packed_lhs,
            rhs,
            (((2,), (1,)), ((0,), (0,))),
            preferred_element_type=out_dtype,
        )
        return _scatter_packed_rows(packed_out.astype(out_dtype), group_sizes, lhs.shape[0])

    if ragged_dot_dimension_numbers == _DLHS_DIM_NUMS:
        packed_lhs = _pack_ragged_rows(lhs, group_sizes, max_group_size)
        packed_out = jax.lax.dot_general(
            packed_lhs,
            rhs,
            (((2,), (2,)), ((0,), (0,))),
            preferred_element_type=out_dtype,
        )
        return _scatter_packed_rows(packed_out.astype(out_dtype), group_sizes, lhs.shape[0])

    if ragged_dot_dimension_numbers == _DRHS_DIM_NUMS:
        packed_lhs = _pack_ragged_rows(lhs, group_sizes, max_group_size)
        packed_rhs = _pack_ragged_rows(rhs, group_sizes, max_group_size)
        return jax.lax.dot_general(
            packed_lhs,
            packed_rhs,
            (((1,), (1,)), ((0,), (0,))),
            preferred_element_type=out_dtype,
        ).astype(out_dtype)

    raise NotImplementedError(f"Unsupported ragged dot dimension numbers for padded dense: {ragged_dot_dimension_numbers}")


def _mosaic_fp8_ragged_dot_impl(
    lhs: jax.Array,
    rhs: jax.Array,
    group_sizes: jax.Array,
    ragged_dot_dimension_numbers: jax.lax.RaggedDotDimensionNumbers,
    out_dtype: DTypeLike,
    max_group_size: int | None,
    block_m: int | None,
    block_n: int | None,
    block_k: int | None,
) -> jax.Array:
    block_m = 128 if block_m is None else block_m
    block_n = 128 if block_n is None else block_n
    block_k = 64 if block_k is None else block_k
    max_concurrent_steps = int(os.environ.get("FP8_MOSAIC_MAX_CONCURRENT_STEPS", "4"))
    grid_block_n = int(os.environ.get("FP8_MOSAIC_GRID_BLOCK_N", "8"))
    mosaic_out_dtype = lhs.dtype if os.environ.get("FP8_MOSAIC_OUTPUT_FP8", "0") == "1" else out_dtype

    if ragged_dot_dimension_numbers == _DEFAULT_DIM_NUMS:
        return mosaic_ragged_dot(
            lhs,
            jnp.swapaxes(rhs, 1, 2),
            group_sizes=group_sizes,
            out_dtype=mosaic_out_dtype,
            block_m=block_m,
            block_n=block_n,
            block_k=block_k,
            max_concurrent_steps=max_concurrent_steps,
            grid_block_n=grid_block_n,
        )

    if ragged_dot_dimension_numbers == _DLHS_DIM_NUMS:
        return mosaic_ragged_dot(
            lhs,
            rhs,
            group_sizes=group_sizes,
            out_dtype=mosaic_out_dtype,
            block_m=block_m,
            block_n=block_k,
            block_k=block_n,
            max_concurrent_steps=max_concurrent_steps,
            grid_block_n=grid_block_n,
        )

    if ragged_dot_dimension_numbers == _DRHS_DIM_NUMS:
        wgrad_impl = os.environ.get("FP8_MOSAIC_WGRAD", "triton")
        if wgrad_impl == "triton":
            return _triton_ragged_contracting_dim_pallas_call(
                lhs,
                rhs,
                group_sizes,
                out_dtype=out_dtype,
                block_m_override=128,
                block_n_override=block_n,
                block_k_override=block_k,
            )
        token_block = block_m if block_m <= 128 else 128
        if wgrad_impl == "mosaic_block_pad":
            padded_lhs, padded_rhs, padded_group_sizes = _block_pad_ragged_token_stream(lhs, rhs, group_sizes, token_block)
            padded_lhs_t = padded_lhs.T + jnp.zeros((padded_lhs.shape[1], padded_lhs.shape[0]), dtype=padded_lhs.dtype)
            padded_rhs_t = padded_rhs.T + jnp.zeros((padded_rhs.shape[1], padded_rhs.shape[0]), dtype=padded_rhs.dtype)
            return mosaic_transposed_ragged_dot(
                padded_lhs_t,
                padded_rhs_t,
                group_sizes=padded_group_sizes,
                out_dtype=out_dtype,
                block_m=block_k,
                block_n=block_n,
                block_k=token_block,
                max_concurrent_steps=max_concurrent_steps,
                grid_block_n=grid_block_n,
                mask_boundaries=False,
            )
        return mosaic_transposed_ragged_dot(
            lhs.T + jnp.zeros((lhs.shape[1], lhs.shape[0]), dtype=lhs.dtype),
            rhs.T + jnp.zeros((rhs.shape[1], rhs.shape[0]), dtype=rhs.dtype),
            group_sizes=group_sizes,
            out_dtype=out_dtype,
            block_m=block_k,
            block_n=block_n,
            block_k=token_block,
            max_concurrent_steps=max_concurrent_steps,
            grid_block_n=grid_block_n,
            mask_boundaries=wgrad_impl != "mosaic_nomask",
        )

    raise NotImplementedError(f"Unsupported ragged dot dimension numbers for Mosaic FP8: {ragged_dot_dimension_numbers}")


def _raw_ragged_dot_impl(
    lhs: jax.Array,
    rhs: jax.Array,
    group_sizes: jax.Array,
    ragged_dot_dimension_numbers: jax.lax.RaggedDotDimensionNumbers,
    implementation: Implementation,
    out_dtype: DTypeLike,
    max_group_size: int | None = None,
    block_m: int | None = None,
    block_n: int | None = None,
    block_k: int | None = None,
) -> jax.Array:
    if implementation == "mosaic":
        return _mosaic_fp8_ragged_dot_impl(
            lhs,
            rhs,
            group_sizes,
            ragged_dot_dimension_numbers,
            out_dtype,
            max_group_size,
            block_m,
            block_n,
            block_k,
        )
    if implementation == "padded_dense":
        return _padded_dense_ragged_dot_impl(
            lhs,
            rhs,
            group_sizes,
            ragged_dot_dimension_numbers,
            out_dtype,
            max_group_size,
        )
    if implementation == "triton":
        return _triton_pallas_call(
            lhs,
            rhs,
            group_sizes,
            ragged_dot_dimension_numbers=ragged_dot_dimension_numbers,
            out_dtype=out_dtype,
            max_group_size=max_group_size,
            block_m=block_m,
            block_n=block_n,
            block_k=block_k,
        )
    if implementation == "triton_native":
        return _triton_native_pallas_call(
            lhs,
            rhs,
            group_sizes,
            ragged_dot_dimension_numbers=ragged_dot_dimension_numbers,
            out_dtype=out_dtype,
            max_group_size=max_group_size,
            block_m=block_m,
            block_n=block_n,
            block_k=block_k,
        )
    if implementation == "xla":
        return jax.lax.ragged_dot_general(
            lhs=lhs,
            rhs=rhs,
            group_sizes=group_sizes,
            ragged_dot_dimension_numbers=ragged_dot_dimension_numbers,
        ).astype(out_dtype)
    if implementation == "auto":
        if jax.default_backend() == "gpu" and _has_pallas_triton:
            return _raw_ragged_dot_impl(
                lhs,
                rhs,
                group_sizes,
                ragged_dot_dimension_numbers,
                "triton",
                out_dtype,
                max_group_size,
                block_m,
                block_n,
                block_k,
            )
        return _raw_ragged_dot_impl(lhs, rhs, group_sizes, ragged_dot_dimension_numbers, "xla", out_dtype)
    raise ValueError(f"Unknown ragged_dot implementation for FP8 path: {implementation}")


@functools.partial(jax.custom_vjp, nondiff_argnums=(9, 10, 11, 12, 13, 14, 15))
def _quantized_ragged_dot(
    lhs,
    q_lhs,
    lhs_scale,
    rhs,
    q_rhs,
    rhs_scale,
    out_grad_scale,
    out_grad_amax_history,
    group_sizes,
    implementation,
    preferred_element_type,
    rev_dtype,
    max_group_size,
    block_m,
    block_n,
    block_k,
):
    del lhs, rhs, out_grad_scale, out_grad_amax_history, rev_dtype
    return _raw_ragged_dot_impl(
        q_lhs,
        q_rhs,
        group_sizes,
        _DEFAULT_DIM_NUMS,
        implementation,
        preferred_element_type,
        max_group_size,
        block_m,
        block_n,
        block_k,
    )


def _quantized_ragged_dot_fwd(
    lhs,
    q_lhs,
    lhs_scale,
    rhs,
    q_rhs,
    rhs_scale,
    out_grad_scale,
    out_grad_amax_history,
    group_sizes,
    implementation,
    preferred_element_type,
    rev_dtype,
    max_group_size,
    block_m,
    block_n,
    block_k,
):
    out = _raw_ragged_dot_impl(
        q_lhs,
        q_rhs,
        group_sizes,
        _DEFAULT_DIM_NUMS,
        implementation,
        preferred_element_type,
        max_group_size,
        block_m,
        block_n,
        block_k,
    )
    res = (lhs, q_lhs, lhs_scale, rhs, q_rhs, rhs_scale, out_grad_scale, out_grad_amax_history, group_sizes)
    return out, res


def _quantized_ragged_dot_bwd(
    implementation,
    preferred_element_type,
    rev_dtype,
    max_group_size,
    block_m,
    block_n,
    block_k,
    res,
    g,
):
    lhs, q_lhs, lhs_scale, rhs, q_rhs, rhs_scale, out_grad_scale, out_grad_amax_history, group_sizes = res
    del rhs
    new_out_grad_scale, new_out_grad_amax_history = update_fp8_meta(
        g,
        rev_dtype,
        out_grad_scale,
        out_grad_amax_history,
    )
    q_g = quantize(g, rev_dtype, new_out_grad_scale, preferred_element_type)
    if implementation == "mosaic":
        wgrad_impl = os.environ.get("FP8_MOSAIC_WGRAD", "bf16_triton")
        q_lhs_for_grad = q_lhs if wgrad_impl in ("mosaic", "mosaic_nomask", "mosaic_block_pad") else q_lhs.astype(rev_dtype)
        q_rhs_for_grad = q_rhs
    else:
        q_lhs_for_grad = q_lhs.astype(rev_dtype)
        q_rhs_for_grad = q_rhs.astype(rev_dtype)

    grad_lhs_block_k = block_k
    if implementation == "mosaic":
        grad_lhs_block_k = int(os.environ.get("FP8_MOSAIC_DGRAD_BLOCK_K", "64"))

    grad_lhs = _raw_ragged_dot_impl(
        q_g,
        q_rhs_for_grad,
        group_sizes,
        _DLHS_DIM_NUMS,
        implementation,
        preferred_element_type,
        max_group_size,
        block_m,
        block_n,
        grad_lhs_block_k,
    )
    grad_lhs = dequantize(grad_lhs, preferred_element_type, rhs_scale * new_out_grad_scale)

    if implementation == "mosaic" and os.environ.get("FP8_MOSAIC_WGRAD", "bf16_triton") == "bf16_triton":
        grad_rhs = _raw_ragged_dot_impl(
            lhs.astype(preferred_element_type),
            g.astype(preferred_element_type),
            group_sizes,
            _DRHS_DIM_NUMS,
            "triton",
            preferred_element_type,
            max_group_size,
            128,
            block_n,
            block_k,
        )
    else:
        grad_rhs = _raw_ragged_dot_impl(
            q_lhs_for_grad,
            q_g,
            group_sizes,
            _DRHS_DIM_NUMS,
            implementation,
            preferred_element_type,
            max_group_size,
            block_m,
            block_n,
            block_k,
        )
        grad_rhs = dequantize(grad_rhs, preferred_element_type, lhs_scale * new_out_grad_scale)

    return (
        grad_lhs,
        None,
        None,
        grad_rhs,
        None,
        None,
        new_out_grad_scale,
        new_out_grad_amax_history,
        None,
    )


_quantized_ragged_dot.defvjp(_quantized_ragged_dot_fwd, _quantized_ragged_dot_bwd)


def fp8_scaled_ragged_dot(
    lhs: jax.Array,
    rhs: jax.Array,
    group_sizes: jax.Array,
    *,
    implementation: Implementation = "auto",
    preferred_element_type: DTypeLike,
    max_group_size: int | None = None,
    block_m: int | None = None,
    block_n: int | None = None,
    block_k: int | None = None,
    lhs_scale: jax.Array,
    rhs_scale: jax.Array,
    grad_scale: jax.Array,
    lhs_amax_history: jax.Array,
    rhs_amax_history: jax.Array,
    grad_amax_history: jax.Array,
    quantize_compute_type: DTypeLike = jnp.float32,
    fwd_dtype: DTypeLike = jnp.float8_e4m3fn,
    rev_dtype: DTypeLike = jnp.float8_e5m2,
) -> jax.Array:
    """Ragged grouped matmul with delayed-scaling FP8 operands and custom VJP."""
    q_lhs, new_lhs_scale = in_q(quantize_compute_type, fwd_dtype, lhs, lhs_scale, lhs_amax_history)
    q_rhs, new_rhs_scale = in_q(quantize_compute_type, fwd_dtype, rhs, rhs_scale, rhs_amax_history)
    y = _quantized_ragged_dot(
        lhs,
        q_lhs,
        new_lhs_scale,
        rhs,
        q_rhs,
        new_rhs_scale,
        grad_scale,
        grad_amax_history,
        group_sizes,
        implementation,
        preferred_element_type,
        rev_dtype,
        max_group_size,
        block_m,
        block_n,
        block_k,
    )
    return out_dq(preferred_element_type, new_lhs_scale, new_rhs_scale, y)


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
    if name == "mosaic":
        raise NotImplementedError("mosaic is currently implemented only for the opt-in FP8 ragged_dot path")
    if name == "padded_dense":
        raise NotImplementedError("padded_dense is currently implemented only for the opt-in FP8 ragged_dot path")
    if name == "triton":
        return _ragged_dot_triton_impl(lhs, rhs, group_sizes)
    if name == "triton_native":
        raise NotImplementedError("triton_native is currently implemented only for the opt-in FP8 ragged_dot path")
    if name == "xla":
        return _ragged_dot_xla_impl(lhs, rhs, group_sizes)
    raise ValueError(f"Unknown ragged_dot implementation: {name}")


def ragged_dot(
    lhs_: jax.Array,
    rhs_: jax.Array,
    group_sizes_: jax.Array,
    ar: bool = False,
    implementation: Implementation = "auto",
    fp8_dot: Callable[..., jax.Array] | None = None,
    max_group_size: int | None = None,
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
        fp8_dot: Optional stateful FP8 ragged-dot op. When provided, the grouped
            matmul is computed with that op and the default bf16 path is unchanged.
        max_group_size: Static upper bound for tokens in any expert group. The
            FP8 kernels use this to reduce row-block over-launch for genuinely
            ragged batches; group sizes may still be non-uniform and dynamic.

    Returns:
        A [tokens, out] array.
    """
    hs_shape = lhs_.shape
    if hs_shape[0] % 512:
        pad_length = 512 - hs_shape[0] % 512
        lhs_ = jax.lax.pad(lhs_, jnp.zeros((), dtype=lhs_.dtype), [(0, pad_length, 0), (0, 0, 0)])

    if fp8_dot is not None:
        out = fp8_dot(lhs_, rhs_, group_sizes_, implementation=implementation, max_group_size=max_group_size)
    else:
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
