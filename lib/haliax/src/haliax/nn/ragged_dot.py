# Copyright The Levanter Authors
#
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import functools
import logging
import math
import os
import warnings
from enum import StrEnum
from typing import Callable, Literal, TypeAlias

import jax
import jax.numpy as jnp

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

# Two families of Pallas-Triton kernels, chosen per GPU family by _triton_kernel_family:
#
# - Group grid, adapted from openxla/tokamax@ad75b704:
#   tokamax/_src/ops/ragged_dot/pallas_triton.py. In particular:
#   _ragged_dot_kernel/_ragged_dot for the default layout,
#   _ragged_contracting_dim_dot_kernel/_ragged_contracting_dim_dot for drhs, and
#   PallasTritonRaggedDot._fwd for the VJP layout dispatch.
# - Tile map: forward and dlhs launch one program per (group, row tile, column tile) from a
#   device-side tile map, as megablox's make_group_metadata, instead of a full row grid per group;
#   dlhs reads the weights transposed in place, and drhs uses one flat grid over all groups.
Implementation: TypeAlias = Literal["auto", "megablox", "triton", "xla"]
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


# Megablox GMM tiling (m, k, n). The ``k`` (contraction) dim is the tightest VMEM
# constraint: TPU v4 has less VMEM and only fits k=512; other generations use k=1024.
_MEGABLOX_TILE_DEFAULT: tuple[int, int, int] = (512, 1024, 1024)
_MEGABLOX_TILE_V4: tuple[int, int, int] = (512, 512, 1024)  # halved k for v4's smaller VMEM


def _megablox_tile_size() -> tuple[int, int, int]:
    """Device-dependent (m, k, n) tiling for the megablox GMM.

    Defaults to k=1024; TPU v4 downsizes to k=512 to fit its smaller VMEM.
    """
    try:
        device_kind = jax.devices()[0].device_kind.lower()
    except (RuntimeError, IndexError):
        return _MEGABLOX_TILE_DEFAULT
    # device_kind is e.g. "TPU v4", "TPU v5 lite", "TPU v5p", "TPU v6e".
    if "v4" in device_kind:
        return _MEGABLOX_TILE_V4
    return _MEGABLOX_TILE_DEFAULT


def _megablox_tiling(m: int, k: int, n: int) -> tuple[int, int, int]:
    """Choose a Megablox tile for the shape of each forward or VJP GMM."""
    tile_size = _megablox_tile_size()
    return min(m, tile_size[0]), min(k, tile_size[1]), min(n, tile_size[2])


def _ragged_dot_megablox_impl(lhs: jax.Array, rhs: jax.Array, group_sizes: jax.Array) -> jax.Array:
    if _gmm_megablox is None:
        raise NotImplementedError("megablox GMM is not available (TPU-only)")
    return _gmm_megablox(
        lhs,
        rhs,
        group_sizes,
        preferred_element_type=lhs.dtype,
        tiling=_megablox_tiling,
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
    n: int,
):
    """Pallas-Triton ragged dot kernel (no quantization)."""
    lo = lo_ref[()]
    hi = hi_ref[()]
    start_m = lo + pl.program_id(0) * block_m

    @pl.when(start_m < hi)
    def _compute():
        span_m = pl.ds(start_m, block_m)
        start_n = pl.program_id(1) * out_ref.shape[1]
        acc = jnp.zeros((block_m, out_ref.shape[1]), dtype=jnp.float32)
        k = a_ref.shape[1]

        def body(i, acc):
            start_k = i * block_k
            span_k = pl.ds(start_k, block_k)
            if k % block_k:
                contraction_mask = start_k + jnp.arange(block_k) < k
                a = plgpu.load(a_ref.at[span_m, span_k], mask=contraction_mask[None, :], other=0.0)
                b = plgpu.load(b_ref.at[span_k, pl.ds(0, b_ref.shape[1])], mask=contraction_mask[:, None], other=0.0)
            else:
                a = plgpu.load(a_ref.at[span_m, span_k])
                b = plgpu.load(b_ref.at[span_k, pl.ds(0, b_ref.shape[1])])
            dtype = jnp.result_type(a, b)
            return acc + pl.dot(a.astype(dtype), b.astype(dtype))

        num_k_blocks = pl.cdiv(k, block_k)
        acc = jax.lax.fori_loop(0, num_k_blocks, body, acc)
        # Tokamax's BlockRef masks logical output edges; raw Pallas refs do not.
        store_mask = (start_m + jnp.arange(block_m) < hi)[:, None]
        if n % out_ref.shape[1]:
            store_mask &= (start_n + jnp.arange(out_ref.shape[1]) < n)[None, :]
        plgpu.store(
            out_ref.at[span_m, pl.ds(0, out_ref.shape[1])],
            acc.astype(out_ref.dtype),
            mask=store_mask,
        )


def _triton_default_block_sizes(m: int, k: int, n: int) -> tuple[int, int, int]:
    block_m = min(128, int(pl.next_power_of_2(m)))
    max_block_n = _TRITON_BLACKWELL_BLOCK_N if _is_blackwell_gpu_backend() else _TRITON_DEFAULT_BLOCK_N
    block_n = min(max_block_n, int(pl.next_power_of_2(n)))
    block_k = min(32, int(pl.next_power_of_2(k)))
    return block_m, block_n, block_k


@functools.lru_cache(maxsize=None)
def _triton_default_matmul(m: int, k: int, n: int, num_groups: int, dtype) -> Callable[..., jax.Array]:
    """Build the default-layout kernel for one static shape.

    ``pl.pallas_call`` returns a fresh ``jax.jit`` wrapper per call and JAX's
    tracing cache is keyed on function identity, so building it inline re-traces
    the kernel and its index maps at every call site.
    """
    block_m, block_n, block_k = _triton_default_block_sizes(m, k, n)
    return pl.pallas_call(
        lambda a, b, lo, hi, out: _triton_ragged_dot_kernel(
            a,
            b,
            lo,
            hi,
            out,
            block_m=block_m,
            block_k=block_k,
            n=n,
        ),
        out_shape=jax.ShapeDtypeStruct((m, n), dtype),
        in_specs=[
            pl.no_block_spec,
            pl.BlockSpec((None, k, block_n), lambda _, j, e: (e, 0, j)),
            pl.BlockSpec((None,), lambda _, __, e: (e,)),
            pl.BlockSpec((None,), lambda _, __, e: (e,)),
        ],
        out_specs=pl.BlockSpec((m, block_n), lambda _, j, __: (0, j)),
        grid=(pl.cdiv(m, block_m), pl.cdiv(n, block_n), num_groups),
        compiler_params=plgpu.CompilerParams(num_warps=4, num_stages=4),
    )


def _triton_default_pallas_call(lhs: jax.Array, rhs: jax.Array, group_sizes: jax.Array) -> jax.Array:
    """Raw Pallas-Triton grouped matmul for the default ragged-dot layout."""
    m, k = lhs.shape
    num_groups, _, n = rhs.shape
    cum_rows = jnp.cumulative_sum(group_sizes, include_initial=True)
    matmul = _triton_default_matmul(m, k, n, num_groups, lhs.dtype)
    return matmul(lhs, rhs, cum_rows[:-1], cum_rows[1:])


def _triton_ragged_contracting_dim_dot_kernel(
    a_ref,
    b_ref,
    lo_ref,
    hi_ref,
    out_ref,
    *,
    block_m: int,
    block_k: int,
    m: int,
    n: int,
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

    num_k_blocks = jnp.maximum(pl.cdiv(jnp.int32(hi - lo), jnp.int32(block_k)), jnp.int32(1))
    acc = jnp.zeros((block_m, out_ref.shape[1]), dtype=jnp.float32)
    acc = jax.lax.fori_loop(0, num_k_blocks - 1, body, acc)
    acc = body(num_k_blocks - 1, acc, mask_k=True)
    store_mask = None
    if m % block_m:
        store_mask = (pl.program_id(0) * block_m + jnp.arange(block_m) < m)[:, None]
    if n % out_ref.shape[1]:
        column_mask = (pl.program_id(1) * out_ref.shape[1] + jnp.arange(out_ref.shape[1]) < n)[None, :]
        store_mask = column_mask if store_mask is None else store_mask & column_mask
    plgpu.store(out_ref, acc.astype(out_ref.dtype), mask=store_mask)


@functools.lru_cache(maxsize=None)
def _triton_ragged_contracting_dim_matmul(k: int, m: int, n: int, dtype) -> Callable[..., jax.Array]:
    """Build the drhs-layout kernel for one static shape, vmapped over groups."""
    block_m = min(128, int(pl.next_power_of_2(m)))
    block_n = min(128, int(pl.next_power_of_2(n)))
    block_k = min(32, int(pl.next_power_of_2(k)))
    one_group = pl.pallas_call(
        lambda a, b, lo, hi, out: _triton_ragged_contracting_dim_dot_kernel(
            a,
            b,
            lo,
            hi,
            out,
            block_m=block_m,
            block_k=block_k,
            m=m,
            n=n,
        ),
        out_shape=jax.ShapeDtypeStruct((m, n), dtype),
        in_specs=[
            pl.BlockSpec((k, block_m), lambda i, j: (0, i)),
            pl.BlockSpec((k, block_n), lambda i, j: (0, j)),
            pl.no_block_spec,
            pl.no_block_spec,
        ],
        out_specs=pl.BlockSpec((block_m, block_n), lambda i, j: (i, j)),
        grid=(pl.cdiv(m, block_m), pl.cdiv(n, block_n)),
        compiler_params=plgpu.CompilerParams(num_warps=4, num_stages=4),
    )
    return jax.vmap(one_group, in_axes=(None, None, 0, 0))


def _triton_ragged_contracting_dim_pallas_call(
    lhs: jax.Array,
    rhs: jax.Array,
    group_sizes: jax.Array,
) -> jax.Array:
    """Raw Pallas-Triton grouped matmul for drhs-style ragged contraction."""
    k, m = lhs.shape
    _, n = rhs.shape
    cum_rows = jnp.cumulative_sum(group_sizes, include_initial=True)
    matmul = _triton_ragged_contracting_dim_matmul(k, m, n, lhs.dtype)
    return matmul(lhs, rhs, cum_rows[:-1], cum_rows[1:])


class RaggedLayout(StrEnum):
    """Ragged-dot contraction layouts with a Triton kernel.

    With ``G`` groups and ``M`` ragged rows:

    - ``FWD``: lhs ``[M, K]`` x rhs ``[G, K, N]`` -> ``[M, N]``.
    - ``DLHS``: lhs ``[M, N]`` x rhs ``[G, K, N]`` contracted over ``N`` -> ``[M, K]``.
    - ``DRHS``: lhs ``[M, K]`` x rhs ``[M, N]`` contracted over the ragged ``M`` -> ``[G, K, N]``.
    """

    FWD = "fwd"
    DLHS = "dlhs"
    DRHS = "drhs"


def _group_grid_pallas_call(lhs: jax.Array, rhs: jax.Array, group_sizes: jax.Array, layout: RaggedLayout) -> jax.Array:
    """Group-grid kernels: one full row grid per group, with early exit outside the group's rows."""
    if layout == RaggedLayout.FWD:
        return _triton_default_pallas_call(lhs, rhs, group_sizes)
    if layout == RaggedLayout.DLHS:
        return _triton_default_pallas_call(lhs, rhs.mT, group_sizes)
    return _triton_ragged_contracting_dim_pallas_call(lhs, rhs, group_sizes)


@dataclasses.dataclass(frozen=True)
class TritonBlockConfig:
    """Tile shape and launch parameters of one tile-map ragged-dot kernel.

    ``block_m`` and ``block_n`` tile the output, and ``block_k`` tiles the contraction. For
    ``DRHS`` the output rows are ``K`` and the contraction is the ragged ``M``.

    ``num_xcds`` is the number of chiplets the hardware deals consecutive programs to round-robin
    (8 on MI300X and MI350X). With more than one, the kernel renumbers programs so each chiplet
    runs a contiguous block of tiles that share operands in its L2. Use 1 to keep launch order.

    ``group_m`` orders output tiles in bands of ``group_m`` tile rows, walking down a band before
    moving to the next tile column, so programs that run together share row and column panels.
    Use 1 for plain row-major order.
    """

    block_m: int
    block_n: int
    block_k: int
    num_warps: int
    num_stages: int
    num_xcds: int
    group_m: int

    def fit(self, m: int, k: int, n: int) -> "TritonBlockConfig":
        """Shrink each block to the problem; Triton's dot needs every tile dim to be at least 16."""
        return dataclasses.replace(
            self,
            block_m=min(self.block_m, max(16, int(pl.next_power_of_2(m)))),
            block_n=min(self.block_n, max(16, int(pl.next_power_of_2(n)))),
            block_k=min(self.block_k, max(16, int(pl.next_power_of_2(k)))),
        )


class _GpuFamily(StrEnum):
    AMD_INSTINCT = "amd_instinct"
    NVIDIA_BLACKWELL = "nvidia_blackwell"
    NVIDIA = "nvidia"
    # Any other GPU, or no GPU (the CPU interpreter).
    OTHER = "other"


def _gpu_family() -> _GpuFamily:
    if jax.default_backend() != "gpu":
        return _GpuFamily.OTHER
    device_kind = jax.devices("gpu")[0].device_kind
    if "Instinct" in device_kind:
        return _GpuFamily.AMD_INSTINCT
    if _is_blackwell_gpu_backend():
        return _GpuFamily.NVIDIA_BLACKWELL
    if "NVIDIA" in device_kind:
        return _GpuFamily.NVIDIA
    return _GpuFamily.OTHER


# The group-grid blocks, for layouts and GPU families without a tuned entry below.
_TILE_MAP_GENERIC_CONFIG = TritonBlockConfig(
    block_m=128, block_n=_TRITON_DEFAULT_BLOCK_N, block_k=32, num_warps=4, num_stages=4, num_xcds=1, group_m=1
)
_TILE_MAP_BLACKWELL_ROW_CONFIG = dataclasses.replace(_TILE_MAP_GENERIC_CONFIG, block_n=_TRITON_BLACKWELL_BLOCK_N)
_TILE_MAP_CONFIGS: dict[_GpuFamily, dict[RaggedLayout, TritonBlockConfig]] = {
    # Swept on MI350X (gfx950) at the June (G=32, K and N of 1280-2560) and Mixtral-like (G=8, 4096x14336)
    # expert shapes.
    _GpuFamily.AMD_INSTINCT: {
        RaggedLayout.FWD: TritonBlockConfig(
            block_m=256, block_n=256, block_k=64, num_warps=8, num_stages=2, num_xcds=1, group_m=2
        ),
        RaggedLayout.DLHS: TritonBlockConfig(
            block_m=256, block_n=256, block_k=64, num_warps=8, num_stages=2, num_xcds=1, group_m=4
        ),
        RaggedLayout.DRHS: TritonBlockConfig(
            block_m=256, block_n=128, block_k=64, num_warps=4, num_stages=1, num_xcds=8, group_m=8
        ),
    },
    _GpuFamily.NVIDIA_BLACKWELL: {
        RaggedLayout.FWD: _TILE_MAP_BLACKWELL_ROW_CONFIG,
        RaggedLayout.DLHS: _TILE_MAP_BLACKWELL_ROW_CONFIG,
    },
}


def _tile_map_block_config(layout: RaggedLayout, m: int, k: int, n: int) -> TritonBlockConfig:
    """Device-dependent block config for one layout, fitted to ``m x k x n`` (rows x contraction x columns)."""
    config = _TILE_MAP_CONFIGS.get(_gpu_family(), {}).get(layout, _TILE_MAP_GENERIC_CONFIG)
    return config.fit(m, k, n)


def _matmul_cost(m: int, k: int, n: int, *arrays: jax.ShapeDtypeStruct) -> "pl.CostEstimate":
    return pl.CostEstimate(
        flops=2 * m * k * n,
        transcendentals=0,
        bytes_accessed=sum(math.prod(a.shape) * jnp.dtype(a.dtype).itemsize for a in arrays),
    )


def _and_masks(*masks: jax.Array | None) -> jax.Array | None:
    present = [mask for mask in masks if mask is not None]
    if not present:
        return None
    return functools.reduce(jnp.logical_and, present)


def _masked_load(ref, mask: jax.Array | None) -> jax.Array:
    if mask is None:
        return plgpu.load(ref)
    return plgpu.load(ref, mask=mask, other=0.0)


def _program_index(num_programs: int, num_xcds: int) -> tuple[jax.Array, jax.Array | bool]:
    """This program's logical index and whether it has one.

    The launch grid is padded to a multiple of ``num_xcds``. Launch index ``i`` runs on chiplet
    ``i % num_xcds``; it is mapped to logical index ``(i % num_xcds) * per_xcd + i // num_xcds`` so
    that each chiplet gets ``per_xcd`` consecutive logical programs.
    """
    pid = pl.program_id(0)
    if num_xcds == 1:
        return pid, True
    per_xcd = pl.cdiv(num_programs, num_xcds)
    pid = (pid % num_xcds) * per_xcd + pid // num_xcds
    return pid, pid < num_programs


def _tile_coordinates(pid: jax.Array, num_m: int, num_n: int, group_m: int) -> tuple[jax.Array, jax.Array]:
    """Map a logical program index onto ``(tile_row, tile_col)`` of a ``num_m x num_n`` tile grid."""
    if group_m == 1:
        return pid // num_n, pid % num_n
    per_band = group_m * num_n
    first_row = (pid // per_band) * group_m
    band_rows = jnp.minimum(num_m - first_row, group_m)
    within = pid % per_band
    return first_row + within % band_rows, within // band_rows


def _launch_grid(num_programs: int, num_xcds: int) -> tuple[int]:
    return (pl.cdiv(num_programs, num_xcds) * num_xcds,)


def _row_tile_metadata(group_sizes: jax.Array, m: int, block_m: int) -> jax.Array:
    """Map each (group, row tile) pair that holds rows to a program, as megablox's ``make_group_metadata``.

    Rows past ``sum(group_sizes)`` form one extra pseudo-group that the kernel fills with zeros. A
    row tile that straddles a group boundary appears once per group it touches, so there are at
    most ``cdiv(m, block_m) + num_groups`` tiles. Unused trailing tiles get an empty row range.

    Returns:
        ``int32[4, cdiv(m, block_m) + num_groups]`` holding each tile's group, row-tile index, and
        first and one-past-last row.
    """
    num_groups = group_sizes.shape[0]
    group_sizes = group_sizes.astype(jnp.int32)
    sizes = jnp.concatenate([group_sizes, (m - jnp.sum(group_sizes))[None]])
    ends = jnp.cumsum(sizes)
    starts = ends - sizes
    first_tile = starts // block_m
    tiles_per_group = jnp.where(sizes > 0, (ends - 1) // block_m - first_tile + 1, 0)
    tile_ends = jnp.cumsum(tiles_per_group)

    tile = jnp.arange(pl.cdiv(m, block_m) + num_groups, dtype=jnp.int32)
    # compare_all is one fused kernel; the default scan method launches several and cost ~0.1 ms per call.
    raw_group = jnp.searchsorted(tile_ends, tile, side="right", method="compare_all").astype(jnp.int32)
    group = jnp.minimum(raw_group, num_groups)
    row_tile = first_tile[group] + tile - (tile_ends[group] - tiles_per_group[group])
    lo = jnp.maximum(starts[group], row_tile * block_m)
    hi = jnp.minimum(ends[group], (row_tile + 1) * block_m)
    hi = jnp.where(raw_group > num_groups, lo, hi)
    return jnp.stack([group, row_tile, lo, hi]).astype(jnp.int32)


def _tile_map_row_kernel(
    a_ref,
    b_ref,
    tiles_ref,
    out_ref,
    *,
    config: TritonBlockConfig,
    m: int,
    k: int,
    n: int,
    num_groups: int,
    num_tiles: int,
    transpose_rhs: bool,
):
    """One ``block_m x block_n`` output tile of a row-ragged grouped matmul (FWD and DLHS)."""
    block_m, block_n, block_k = config.block_m, config.block_n, config.block_k
    num_n_tiles = pl.cdiv(n, block_n)
    pid, has_work = _program_index(num_tiles * num_n_tiles, config.num_xcds)
    tile, n_tile = _tile_coordinates(pid, num_tiles, num_n_tiles, config.group_m)
    tile = jnp.minimum(tile, num_tiles - 1)
    start_n = n_tile * block_n
    group = tiles_ref[0, tile]
    start_m = tiles_ref[1, tile] * block_m
    lo = tiles_ref[2, tile]
    hi = jnp.where(has_work, tiles_ref[3, tile], lo)

    rows = start_m + jnp.arange(block_m)
    row_span = pl.ds(start_m, block_m)
    col_span = pl.ds(start_n, block_n)
    row_mask = (rows < m)[:, None] if m % block_m else None
    col_mask = start_n + jnp.arange(block_n) < n if n % block_n else None
    store_mask = _and_masks(((rows >= lo) & (rows < hi))[:, None], None if col_mask is None else col_mask[None, :])

    @pl.when((lo < hi) & (group == num_groups))
    def _zero_fill():
        zeros = jnp.zeros((block_m, block_n), out_ref.dtype)
        plgpu.store(out_ref.at[row_span, col_span], zeros, mask=store_mask)

    @pl.when((lo < hi) & (group < num_groups))
    def _compute():
        def body(i, acc):
            start_k = i * block_k
            k_span = pl.ds(start_k, block_k)
            k_mask = start_k + jnp.arange(block_k) < k if k % block_k else None
            a = _masked_load(
                a_ref.at[row_span, k_span], _and_masks(row_mask, None if k_mask is None else k_mask[None, :])
            )
            if transpose_rhs:
                b_mask = _and_masks(
                    None if col_mask is None else col_mask[:, None], None if k_mask is None else k_mask[None, :]
                )
                b = _masked_load(b_ref.at[group, col_span, k_span], b_mask)
                return acc + pl.dot(a, b, trans_b=True)
            b_mask = _and_masks(
                None if k_mask is None else k_mask[:, None], None if col_mask is None else col_mask[None, :]
            )
            b = _masked_load(b_ref.at[group, k_span, col_span], b_mask)
            return acc + pl.dot(a, b)

        acc = jax.lax.fori_loop(0, pl.cdiv(k, block_k), body, jnp.zeros((block_m, block_n), jnp.float32))
        plgpu.store(out_ref.at[row_span, col_span], acc.astype(out_ref.dtype), mask=store_mask)


@functools.lru_cache(maxsize=None)
def _tile_map_row_matmul(
    m: int, k: int, n: int, num_groups: int, dtype, layout: RaggedLayout, config: TritonBlockConfig
) -> Callable[..., jax.Array]:
    """Build the FWD or DLHS kernel for one static shape.

    ``pl.pallas_call`` returns a fresh ``jax.jit`` wrapper per call and JAX's tracing cache is
    keyed on function identity, so building it inline re-traces the kernel at every call site.
    The grid is flat: ``cdiv(m, block_m) + num_groups`` row tiles, each split into column tiles.
    """
    num_tiles = pl.cdiv(m, config.block_m) + num_groups
    rhs_shape = (num_groups, k, n) if layout == RaggedLayout.FWD else (num_groups, n, k)
    return pl.pallas_call(
        functools.partial(
            _tile_map_row_kernel,
            config=config,
            m=m,
            k=k,
            n=n,
            num_groups=num_groups,
            num_tiles=num_tiles,
            transpose_rhs=layout == RaggedLayout.DLHS,
        ),
        out_shape=jax.ShapeDtypeStruct((m, n), dtype),
        in_specs=[pl.no_block_spec, pl.no_block_spec, pl.no_block_spec],
        out_specs=pl.no_block_spec,
        grid=_launch_grid(num_tiles * pl.cdiv(n, config.block_n), config.num_xcds),
        compiler_params=plgpu.CompilerParams(num_warps=config.num_warps, num_stages=config.num_stages),
        cost_estimate=_matmul_cost(
            m,
            k,
            n,
            jax.ShapeDtypeStruct((m, k), dtype),
            jax.ShapeDtypeStruct(rhs_shape, dtype),
            jax.ShapeDtypeStruct((m, n), dtype),
        ),
        interpret=jax.default_backend() == "cpu",
    )


def _tile_map_contracting_kernel(
    a_ref,
    b_ref,
    bounds_ref,
    out_ref,
    *,
    config: TritonBlockConfig,
    m: int,
    n: int,
    num_groups: int,
):
    """One ``block_m x block_n`` tile of one group's ``a[lo:hi].T @ b[lo:hi]`` (DRHS)."""
    block_m, block_n, block_k = config.block_m, config.block_n, config.block_k
    num_m_tiles = pl.cdiv(m, block_m)
    num_n_tiles = pl.cdiv(n, block_n)
    pid, has_work = _program_index(num_groups * num_m_tiles * num_n_tiles, config.num_xcds)
    pid = jnp.minimum(pid, num_groups * num_m_tiles * num_n_tiles - 1)
    group = pid // (num_m_tiles * num_n_tiles)
    m_tile, n_tile = _tile_coordinates(pid % (num_m_tiles * num_n_tiles), num_m_tiles, num_n_tiles, config.group_m)
    start_m = m_tile * block_m
    start_n = n_tile * block_n
    lo = bounds_ref[group]
    hi = jnp.where(has_work, bounds_ref[group + 1], lo)

    m_span = pl.ds(start_m, block_m)
    n_span = pl.ds(start_n, block_n)
    m_mask = start_m + jnp.arange(block_m) < m if m % block_m else None
    n_mask = start_n + jnp.arange(block_n) < n if n % block_n else None

    def body(i, acc, mask_rows=False):
        start_r = lo + i * block_k
        r_span = pl.ds(start_r, block_k)
        r_mask = (start_r + jnp.arange(block_k) < hi)[:, None] if mask_rows else None
        a = _masked_load(a_ref.at[r_span, m_span], _and_masks(r_mask, None if m_mask is None else m_mask[None, :]))
        b = _masked_load(b_ref.at[r_span, n_span], _and_masks(r_mask, None if n_mask is None else n_mask[None, :]))
        return acc + pl.dot(a, b, trans_a=True)

    # Every row block but the last is full. An empty group runs one fully masked block and stores zeros.
    num_r_blocks = jnp.maximum(pl.cdiv(hi - lo, block_k), 1)
    acc = jax.lax.fori_loop(0, num_r_blocks - 1, body, jnp.zeros((block_m, block_n), jnp.float32))
    acc = body(num_r_blocks - 1, acc, mask_rows=True)
    store_mask = _and_masks(None if m_mask is None else m_mask[:, None], None if n_mask is None else n_mask[None, :])

    @pl.when(has_work)
    def _store():
        plgpu.store(out_ref.at[group, m_span, n_span], acc.astype(out_ref.dtype), mask=store_mask)


@functools.lru_cache(maxsize=None)
def _tile_map_contracting_matmul(
    rows: int, m: int, n: int, num_groups: int, dtype, config: TritonBlockConfig
) -> Callable[..., jax.Array]:
    """Build the DRHS kernel for one static shape: one program per (group, output tile)."""
    return pl.pallas_call(
        functools.partial(_tile_map_contracting_kernel, config=config, m=m, n=n, num_groups=num_groups),
        out_shape=jax.ShapeDtypeStruct((num_groups, m, n), dtype),
        in_specs=[pl.no_block_spec, pl.no_block_spec, pl.no_block_spec],
        out_specs=pl.no_block_spec,
        grid=_launch_grid(num_groups * pl.cdiv(m, config.block_m) * pl.cdiv(n, config.block_n), config.num_xcds),
        compiler_params=plgpu.CompilerParams(num_warps=config.num_warps, num_stages=config.num_stages),
        cost_estimate=_matmul_cost(
            m,
            rows,
            n,
            jax.ShapeDtypeStruct((rows, m), dtype),
            jax.ShapeDtypeStruct((rows, n), dtype),
            jax.ShapeDtypeStruct((num_groups, m, n), dtype),
        ),
        interpret=jax.default_backend() == "cpu",
    )


def _tile_map_pallas_call(
    lhs: jax.Array,
    rhs: jax.Array,
    group_sizes: jax.Array,
    layout: RaggedLayout,
    config: TritonBlockConfig | None = None,
) -> jax.Array:
    """Tile-map kernels. ``config``, when given, replaces the device-dependent block config."""
    if layout == RaggedLayout.DRHS:
        rows, m = lhs.shape
        n = rhs.shape[1]
        num_groups = group_sizes.shape[0]
        config = (config or _tile_map_block_config(layout, m, rows, n)).fit(m, rows, n)
        bounds = jnp.cumulative_sum(group_sizes.astype(jnp.int32), include_initial=True)
        return _tile_map_contracting_matmul(rows, m, n, num_groups, lhs.dtype, config)(lhs, rhs, bounds)

    m, k = lhs.shape
    num_groups = rhs.shape[0]
    n = rhs.shape[2] if layout == RaggedLayout.FWD else rhs.shape[1]
    config = (config or _tile_map_block_config(layout, m, k, n)).fit(m, k, n)
    tiles = _row_tile_metadata(group_sizes, m, config.block_m)
    return _tile_map_row_matmul(m, k, n, num_groups, lhs.dtype, layout, config)(lhs, rhs, tiles)


class TritonKernelFamily(StrEnum):
    """The two Pallas-Triton kernel families behind ``implementation="triton"``."""

    GROUP_GRID = "group_grid"
    TILE_MAP = "tile_map"


_TRITON_KERNELS: dict[TritonKernelFamily, Callable[[jax.Array, jax.Array, jax.Array, RaggedLayout], jax.Array]] = {
    TritonKernelFamily.GROUP_GRID: _group_grid_pallas_call,
    TritonKernelFamily.TILE_MAP: _tile_map_pallas_call,
}
# GPU families that run the tile-map kernels; every other GPU, NVIDIA included, runs the group-grid kernels.
_TILE_MAP_GPU_FAMILIES = frozenset({_GpuFamily.AMD_INSTINCT})


def _triton_kernel_family() -> TritonKernelFamily:
    if _gpu_family() in _TILE_MAP_GPU_FAMILIES:
        return TritonKernelFamily.TILE_MAP
    return TritonKernelFamily.GROUP_GRID


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

_LAYOUT_DIM_NUMS: dict[RaggedLayout, jax.lax.RaggedDotDimensionNumbers] = {
    RaggedLayout.FWD: _DEFAULT_DIM_NUMS,
    RaggedLayout.DLHS: _DLHS_DIM_NUMS,
    RaggedLayout.DRHS: _DRHS_DIM_NUMS,
}


def _triton_pallas_call(
    lhs: jax.Array,
    rhs: jax.Array,
    group_sizes: jax.Array,
    ragged_dot_dimension_numbers: jax.lax.RaggedDotDimensionNumbers = _DEFAULT_DIM_NUMS,
) -> jax.Array:
    """Raw Pallas-Triton grouped matmul for supported ragged-dot layouts, with this GPU's kernel family."""
    for layout, dim_nums in _LAYOUT_DIM_NUMS.items():
        if ragged_dot_dimension_numbers == dim_nums:
            return _TRITON_KERNELS[_triton_kernel_family()](lhs, rhs, group_sizes, layout)
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
