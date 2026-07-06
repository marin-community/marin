# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Benchmark local Blackwell fused W13/SwiGLU for staged source-push MoE."""

from __future__ import annotations

import argparse
import functools
import itertools
import json
import math
import os
import time
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import mosaic_gpu as plgpu
from jax.experimental.pallas.ops.gpu import blackwell_matmul_mgpu, blackwell_ragged_dot_mgpu, ragged_dot_mgpu
from levanter.grug._moe.source_push_inbox_blackwell import BLACKWELL_TARGET_W13_TUNING_CONFIG


TFLOPS = 1.0e12
TARGET_M = 65_536
TARGET_HIDDEN_DIM = 3_072
TARGET_INTERMEDIATE_DIM = 3_072
TARGET_EP_SIZE = 256
TARGET_TOPK = 4
DEFAULT_TUNING_CONFIG = BLACKWELL_TARGET_W13_TUNING_CONFIG
DEFAULT_GRID_MINOR_DIM = blackwell_matmul_mgpu.MatmulDimension[DEFAULT_TUNING_CONFIG.grid_minor_dim.value]


@dataclass(frozen=True)
class W13SwiGLUShape:
    m: int = TARGET_M
    hidden_dim: int = TARGET_HIDDEN_DIM
    intermediate_dim: int = TARGET_INTERMEDIATE_DIM
    num_groups: int = 1
    dtype: str = "bfloat16"

    def validate(self) -> None:
        if self.m <= 0:
            raise ValueError(f"m must be positive, got {self.m}")
        if self.hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {self.hidden_dim}")
        if self.intermediate_dim <= 0:
            raise ValueError(f"intermediate_dim must be positive, got {self.intermediate_dim}")
        if self.num_groups <= 0:
            raise ValueError(f"num_groups must be positive, got {self.num_groups}")
        if self.dtype not in ("bfloat16", "float16"):
            raise ValueError(f"dtype must be 'bfloat16' or 'float16', got {self.dtype!r}")


@dataclass(frozen=True)
class W13SwiGLURunSettings:
    warmup: int = 1
    steps: int = 5
    check: bool = False
    debug_exceptions: bool = False
    implementation: str = "fused"

    def validate(self) -> None:
        if self.warmup < 0:
            raise ValueError(f"warmup must be non-negative, got {self.warmup}")
        if self.steps <= 0:
            raise ValueError(f"steps must be positive, got {self.steps}")


def _parse_int_csv(value: str) -> tuple[int, ...]:
    values = tuple(int(part) for part in value.split(",") if part)
    if not values:
        raise argparse.ArgumentTypeError("expected a comma-separated list of integers")
    return values


def _parse_bool_csv(value: str) -> tuple[bool, ...]:
    values = []
    for part in value.split(","):
        if not part:
            continue
        normalized = part.lower()
        if normalized in ("1", "true", "yes"):
            values.append(True)
        elif normalized in ("0", "false", "no"):
            values.append(False)
        else:
            raise argparse.ArgumentTypeError(f"expected booleans, got {part!r}")
    if not values:
        raise argparse.ArgumentTypeError("expected a comma-separated list of booleans")
    return tuple(values)


def _parse_minor_dim_csv(value: str) -> tuple[blackwell_matmul_mgpu.MatmulDimension, ...]:
    dims = []
    for part in value.split(","):
        if not part:
            continue
        try:
            dims.append(blackwell_matmul_mgpu.MatmulDimension[part.upper()])
        except KeyError as exc:
            raise argparse.ArgumentTypeError("grid minor dimensions must be M or N") from exc
    if not dims:
        raise argparse.ArgumentTypeError("expected a comma-separated list of M/N values")
    return tuple(dims)


def _require_blackwell_gpu() -> str:
    if jax.default_backend() != "gpu":
        raise RuntimeError(f"Blackwell W13/SwiGLU benchmark requires a GPU backend, got {jax.default_backend()!r}")
    devices = jax.devices("gpu")
    if not devices:
        raise RuntimeError("Blackwell W13/SwiGLU benchmark requires visible GPU devices")
    device = devices[0]
    device_kind = getattr(device, "device_kind", "")
    compute_capability = getattr(device, "compute_capability", None)
    if compute_capability is not None:
        try:
            if float(compute_capability) >= 10.0:
                return device_kind
        except (TypeError, ValueError):
            pass
    if any(name in device_kind for name in ("B200", "B300", "GB200", "GB300")):
        return device_kind
    raise RuntimeError(f"Blackwell W13/SwiGLU benchmark requires Blackwell GPUs, got {device_kind!r}")


def _dtype(name: str) -> Any:
    return {
        "bfloat16": jnp.bfloat16,
        "float16": jnp.float16,
    }[name]


def _balanced_group_sizes(num_groups: int, m: int) -> jax.Array:
    base = m // num_groups
    remainder = m % num_groups
    sizes = np.full((num_groups,), base, dtype=np.int32)
    sizes[:remainder] += 1
    return jnp.asarray(sizes)


def _make_inputs(shape: W13SwiGLUShape) -> tuple[jax.Array, jax.Array, jax.Array]:
    dtype = _dtype(shape.dtype)
    lhs_key, rhs_key = jax.random.split(jax.random.key(0))
    lhs = jax.random.normal(lhs_key, (shape.m, shape.hidden_dim), dtype=dtype)
    rhs = jax.random.normal(rhs_key, (shape.num_groups, shape.hidden_dim, 2 * shape.intermediate_dim), dtype=dtype)
    return lhs, rhs, _balanced_group_sizes(shape.num_groups, shape.m)


def _silu(x: jax.Array) -> jax.Array:
    return x * jax.nn.sigmoid(x)


def _useful_w13_tflops(shape: W13SwiGLUShape, seconds: float) -> float:
    flops = 2 * shape.m * shape.hidden_dim * (2 * shape.intermediate_dim)
    return flops / seconds / TFLOPS


def _candidate_configs(args: argparse.Namespace) -> Iterable[blackwell_ragged_dot_mgpu.TuningConfig]:
    if args.preset == "target":
        tile_m_values = (128,)
        tile_n_values = (128,)
        tile_k_values = (64, 128)
        max_concurrent_steps_values = (3, 4, 6, 8)
        collective_values = (True,)
        grid_tile_width_values = (1, 4, 8, 12, 16)
        grid_minor_dim_values = tuple(blackwell_matmul_mgpu.MatmulDimension)
        epilogue_tile_n_values = (64, 128)
    else:
        tile_m_values = args.tile_m
        tile_n_values = args.tile_n
        tile_k_values = args.tile_k
        max_concurrent_steps_values = args.max_concurrent_steps
        collective_values = args.collective
        grid_tile_width_values = args.grid_tile_width
        grid_minor_dim_values = args.grid_minor_dim
        epilogue_tile_n_values = args.epilogue_tile_n

    for (
        tile_m,
        tile_n,
        tile_k,
        max_concurrent_steps,
        collective,
        grid_tile_width,
        grid_minor_dim,
        epilogue_tile_n,
    ) in itertools.product(
        tile_m_values,
        tile_n_values,
        tile_k_values,
        max_concurrent_steps_values,
        collective_values,
        grid_tile_width_values,
        grid_minor_dim_values,
        epilogue_tile_n_values,
    ):
        yield blackwell_ragged_dot_mgpu.TuningConfig(
            tile_m=tile_m,
            tile_n=tile_n,
            tile_k=tile_k,
            max_concurrent_steps=max_concurrent_steps,
            collective=collective,
            grid_tile_width=grid_tile_width,
            grid_minor_dim=grid_minor_dim,
            epilogue_tile_n=epilogue_tile_n,
        )


def _config_row(config: blackwell_ragged_dot_mgpu.TuningConfig) -> dict[str, Any]:
    row = asdict(config)
    row["grid_minor_dim"] = config.grid_minor_dim.name
    return row


def _clamp(min_value, x, max_value):
    return lax.max(lax.min(x, max_value), min_value)


def _do_w13_swiglu_tile(
    a_gmem,
    b_gmem,
    out_gmem,
    grid_indices: Sequence[jax.Array],
    wg_axis: str,
    collective_axes: tuple[str, ...],
    local_index: jax.Array | int,
    config: blackwell_ragged_dot_mgpu.TuningConfig,
    group_info: ragged_dot_mgpu.GroupInfo,
    a_smem,
    w13_smem,
    acc_tmem,
    acc_smem,
    a_tma_barrier,
    w13_tma_barrier,
    store_done_barrier,
    mma_done_barrier,
    consumed_barrier,
) -> None:
    dtype = out_gmem.dtype
    m, k = a_gmem.shape
    _, two_i = b_gmem.shape
    collective = config.collective
    tile_m, tile_n, tile_k = (config.tile_m, config.tile_n, config.tile_k)
    epilogue_tile_n = config.epilogue_tile_n
    max_concurrent_steps = config.max_concurrent_steps
    block_tile_m = tile_m
    block_tile_n = tile_n
    if collective:
        tile_m *= 2
        tile_n *= 2
    k_iters = k // tile_k

    if collective:
        m_index, n_index, cluster_idx = grid_indices
        block_m_index = m_index * 2 + cluster_idx
        is_lead_block = cluster_idx == 0
    else:
        m_index, n_index = grid_indices
        cluster_idx = 0
        block_m_index = m_index
        is_lead_block = True
    wg_idx = lax.axis_index(wg_axis)
    collective_axis = collective_axes[0] if collective else None

    tma_warp = 0
    mma_warp = 1
    compute_wg = 0
    store_wg = 1

    block_slice_m = pl.ds(block_m_index * block_tile_m, block_tile_m)
    slice_m = pl.ds(m_index * tile_m, tile_m)
    slice_n = pl.ds(n_index * tile_n, tile_n)
    acc_slot = jnp.int32(0)
    regs_layout = plgpu.Layout.TCGEN05

    @pl.when(wg_idx == compute_wg)
    @jax.named_scope("compute_wg")
    def _compute_wg() -> None:
        @pl.core_map(plgpu.WarpMesh(axis_name="warp"))
        def _per_warp() -> None:
            warp_id = lax.axis_index("warp")

            @pl.when(warp_id == tma_warp)
            def _memory() -> None:
                def _loop_body(ki, _) -> None:
                    slice_k = pl.ds(ki * tile_k, tile_k)
                    slot = lax.rem(ki, max_concurrent_steps)

                    @pl.when(jnp.logical_or(ki >= max_concurrent_steps, local_index > 0))
                    def _wait_consumed() -> None:
                        plgpu.barrier_wait(consumed_barrier.at[slot])

                    plgpu.copy_gmem_to_smem(
                        a_gmem.at[slice_m, slice_k],
                        a_smem.at[slot],
                        a_tma_barrier.at[slot],
                        leader_tracked=plgpu.CopyPartition.PARTITIONED(0) if collective else None,
                        collective_axes=collective_axis,
                    )
                    plgpu.copy_gmem_to_smem(
                        b_gmem.at[slice_k, pl.ds(n_index * tile_n, tile_n * 2)],
                        w13_smem.at[slot],
                        w13_tma_barrier.at[slot],
                        leader_tracked=plgpu.CopyPartition.PARTITIONED(1) if collective else None,
                        collective_axes=collective_axis,
                    )

                lax.fori_loop(0, k_iters, _loop_body, None)

            @pl.when(jnp.logical_and(warp_id == mma_warp, local_index > 1))
            def _wait_store() -> None:
                plgpu.barrier_wait(store_done_barrier.at[acc_slot])

            @pl.when(jnp.logical_and(warp_id == mma_warp, is_lead_block))
            def _compute() -> None:
                def _loop_body(ki, _) -> None:
                    slot = lax.rem(ki, max_concurrent_steps)
                    plgpu.barrier_wait(a_tma_barrier.at[slot])
                    plgpu.barrier_wait(w13_tma_barrier.at[slot])
                    is_last_iter = ki >= k_iters - 1
                    gate_tmem_slice = acc_tmem.at[:, pl.ds(0, tile_n)]
                    up_tmem_slice = acc_tmem.at[:, pl.ds(tile_n, tile_n)]
                    gate_smem_slice = w13_smem.at[slot, :, pl.ds(0, block_tile_n)]
                    up_smem_slice = w13_smem.at[slot, :, pl.ds(block_tile_n, block_tile_n)]
                    plgpu.tcgen05_mma(
                        gate_tmem_slice,
                        a_smem.at[slot],
                        gate_smem_slice,
                        accumulate=(ki > 0),
                        collective_axis=collective_axis,
                    )
                    plgpu.tcgen05_mma(
                        up_tmem_slice,
                        a_smem.at[slot],
                        up_smem_slice,
                        consumed_barrier.at[slot],
                        accumulate=(ki > 0),
                        collective_axis=collective_axis,
                    )

                    @pl.when(is_last_iter)
                    def _arrive_done() -> None:
                        plgpu.tcgen05_commit_arrive(mma_done_barrier.at[acc_slot], collective_axis=collective_axis)

                lax.fori_loop(0, k_iters, _loop_body, None)

    @pl.when(wg_idx == store_wg)
    @jax.named_scope("store_wg")
    def _store_wg() -> None:
        plgpu.barrier_wait(mma_done_barrier.at[acc_slot])
        gate_tmem_slot = acc_tmem.at[:, pl.ds(0, tile_n)]
        up_tmem_slot = acc_tmem.at[:, pl.ds(tile_n, tile_n)]
        step_out_gmem = out_gmem.at[block_slice_m, slice_n]
        smem_start = group_info.start_within_block - cluster_idx * block_tile_m
        smem_start = lax.max(smem_start, jnp.int32(0))
        block0_copy_size = _clamp(jnp.int32(0), block_tile_m - group_info.start_within_block, group_info.actual_size)
        block_local_size = lax.select(is_lead_block, block0_copy_size, group_info.actual_size - block0_copy_size)
        for ni in range(tile_n // epilogue_tile_n):
            gate = plgpu.async_load_tmem(
                gate_tmem_slot.at[:, pl.ds(ni * epilogue_tile_n, epilogue_tile_n)],
                layout=regs_layout,
            )
            up = plgpu.async_load_tmem(
                up_tmem_slot.at[:, pl.ds(ni * epilogue_tile_n, epilogue_tile_n)],
                layout=regs_layout,
            )
            acc_smem[...] = (_silu(gate) * up).astype(dtype)
            plgpu.commit_smem()
            cur_smem_idx = smem_start
            remaining_rows = min(block_tile_m, m)
            while remaining_rows > 0:
                const_rows_len = 1 << int(math.log2(remaining_rows))
                remaining_rows //= 2

                @pl.when(block_local_size & const_rows_len != 0)
                def _copy_rows() -> None:
                    o_smem_slice = acc_smem.at[pl.ds(cur_smem_idx, const_rows_len)]
                    o_gref_slice = step_out_gmem.at[
                        pl.ds(cur_smem_idx, const_rows_len),
                        pl.ds(ni * epilogue_tile_n, epilogue_tile_n),
                    ]
                    plgpu.copy_smem_to_gmem(o_smem_slice, o_gref_slice)

                cur_smem_idx += block_local_size & const_rows_len
            plgpu.wait_smem_to_gmem(0, wait_read_only=True)
        plgpu.wait_load_tmem()
        plgpu.barrier_arrive(store_done_barrier.at[acc_slot])


def w13_swiglu_kernel(
    a: jax.Array, b: jax.Array, group_sizes: jax.Array, config: blackwell_ragged_dot_mgpu.TuningConfig
):
    dtype = a.dtype
    if a.dtype != b.dtype:
        raise ValueError(f"Matmul LHS and RHS have incompatible dtypes {a.dtype} vs {b.dtype}")
    m, k = a.shape
    num_groups, k2, two_i = b.shape
    if two_i % 2:
        raise ValueError(f"W13 RHS last dimension must be even, got {two_i}")
    intermediate_dim = two_i // 2
    if num_groups != group_sizes.shape[0]:
        raise ValueError("RHS and group_sizes have incompatible shapes.")
    if k != k2:
        raise ValueError(f"Matmul LHS and RHS have incompatible shapes {a.shape} vs {b.shape[1:]}")

    collective = config.collective
    tile_m, tile_n, tile_k = (config.tile_m, config.tile_n, config.tile_k)
    block_tile_m = tile_m
    block_tile_n = tile_n
    if collective:
        tile_m *= 2
        tile_n *= 2
    m_iters = m // tile_m
    n_iters = intermediate_dim // tile_n
    max_concurrent_steps = config.max_concurrent_steps
    epilogue_tile_n = config.epilogue_tile_n
    if tile_n % epilogue_tile_n != 0:
        raise ValueError(f"{tile_n=} must be divisible by {epilogue_tile_n=}")
    if m % tile_m != 0:
        raise ValueError(f"{m=} must be divisible by {tile_m=}")
    if intermediate_dim % tile_n != 0:
        raise ValueError(f"{intermediate_dim=} must be divisible by {tile_n=}")
    if k % tile_k != 0:
        raise ValueError(f"{k=} must be divisible by {tile_k=}")

    swizzle = plgpu.find_swizzle(tile_k * jnp.dtype(dtype).itemsize * 8)
    swizzle_elems = swizzle // jnp.dtype(dtype).itemsize
    transforms = (
        plgpu.TilingTransform((8, swizzle_elems)),
        plgpu.SwizzleTransform(swizzle),
    )

    def kernel(a_gmem, b_gmem, group_sizes_gmem, out_gmem) -> None:
        linear_grid = (m_iters + num_groups - 1) * n_iters
        group_sizes_regs = [group_sizes_gmem[i] for i in range(num_groups)]
        cluster_idx = lax.axis_index("x") if collective else jnp.int32(0)

        @functools.partial(
            pl.run_scoped,
            a_smem=plgpu.SMEM((max_concurrent_steps, block_tile_m, tile_k), dtype, transforms=transforms),
            w13_smem=plgpu.SMEM((max_concurrent_steps, tile_k, block_tile_n * 2), dtype, transforms=transforms),
            acc_smem=plgpu.SMEM((block_tile_m, epilogue_tile_n), dtype),
            a_tma_barrier=plgpu.Barrier(num_arrivals=1, num_barriers=max_concurrent_steps),
            w13_tma_barrier=plgpu.Barrier(num_arrivals=1, num_barriers=max_concurrent_steps),
            store_done_barrier=plgpu.Barrier(num_arrivals=1, orders_tensor_core=True),
            mma_done_barrier=plgpu.Barrier(num_arrivals=1, orders_tensor_core=True),
            consumed_barrier=plgpu.Barrier(
                num_arrivals=1,
                num_barriers=max_concurrent_steps,
                orders_tensor_core=True,
            ),
            acc_tmem=plgpu.TMEM((block_tile_m, tile_n * 2), jnp.float32, collective=collective),
            collective_axes=("wg",),
        )
        def _scoped(**ref_kwargs) -> None:
            @plgpu.nd_loop(grid=(linear_grid,), collective_axes="sm")
            def mn_loop(loop_info: plgpu.NDLoopInfo) -> None:
                (linear_idx,) = loop_info.index
                local_index = loop_info.local_index
                m_index, n_index = plgpu.planar_snake(
                    linear_idx,
                    (m_iters + num_groups - 1, n_iters),
                    config.grid_minor_dim,
                    config.grid_tile_width,
                )
                group_info = ragged_dot_mgpu.GroupInfo.create(group_sizes_regs, tile_m, m_index)
                if collective:
                    grid_indices = (group_info.block, n_index, cluster_idx)
                    collective_axes = ("x",)
                else:
                    grid_indices = (group_info.block, n_index)
                    collective_axes = ()
                _do_w13_swiglu_tile(
                    a_gmem,
                    b_gmem.at[group_info.group_id],
                    out_gmem,
                    grid_indices=grid_indices,
                    wg_axis="wg",
                    collective_axes=collective_axes,
                    local_index=local_index,
                    config=config,
                    group_info=group_info,
                    **ref_kwargs,
                )

    num_sms = jax.local_devices()[0].core_count
    compiler_params = plgpu.CompilerParams(lowering_semantics=plgpu.LoweringSemantics.Warpgroup)
    f = plgpu.kernel(
        kernel,
        compiler_params=compiler_params,
        kernel_name=f"w13_swiglu_kernel_{str(config)}",
        out_shape=jax.ShapeDtypeStruct((m, intermediate_dim), dtype),
        grid=(num_sms // 2,) if collective else (num_sms,),
        grid_names=("sm",),
        num_threads=2,
        thread_name="wg",
        cluster_names=("x",) if collective else (),
        cluster=(2,) if collective else (),
    )
    return f(a, b, group_sizes)


def _reference_w13_swiglu(lhs: jax.Array, rhs: jax.Array, group_sizes: jax.Array) -> jax.Array:
    h = jax.lax.ragged_dot(lhs, rhs, group_sizes, preferred_element_type=jnp.float32)
    gate, up = jnp.split(h, 2, axis=-1)
    return (_silu(gate) * up).astype(lhs.dtype)


def _benchmark_config(
    shape: W13SwiGLUShape,
    settings: W13SwiGLURunSettings,
    lhs: jax.Array,
    rhs: jax.Array,
    group_sizes: jax.Array,
    config: blackwell_ragged_dot_mgpu.TuningConfig,
) -> dict[str, Any]:
    if settings.implementation == "fused":
        fn = jax.jit(functools.partial(w13_swiglu_kernel, config=config))
    elif settings.implementation == "materialized":
        fn = jax.jit(functools.partial(_materialized_w13_swiglu, config=config))
    else:
        raise ValueError(f"unknown implementation {settings.implementation!r}")
    row: dict[str, Any] = {
        "kernel": "blackwell_w13_swiglu",
        "implementation": f"local_blackwell_w13_swiglu:{settings.implementation}",
        "target_ep_size": TARGET_EP_SIZE,
        "target_topk": TARGET_TOPK,
        **asdict(shape),
        **_config_row(config),
    }
    try:
        compile_start = time.perf_counter()
        out = fn(lhs, rhs, group_sizes)
        jax.block_until_ready(out)
        row["compile_time"] = time.perf_counter() - compile_start

        if settings.check:
            expected = _reference_w13_swiglu(lhs, rhs, group_sizes)
            diff = out.astype(jnp.float32) - expected.astype(jnp.float32)
            row["max_abs_diff"] = float(jnp.max(jnp.abs(diff)))
            row["mean_abs_diff"] = float(jnp.mean(jnp.abs(diff)))

        for _ in range(settings.warmup):
            out = fn(lhs, rhs, group_sizes)
            jax.block_until_ready(out)

        step_times = []
        for _ in range(settings.steps):
            start = time.perf_counter()
            out = fn(lhs, rhs, group_sizes)
            jax.block_until_ready(out)
            step_times.append(time.perf_counter() - start)

        median = float(np.median(step_times))
        row.update(
            {
                "ok": True,
                "steady_state_median": median,
                "steady_state_min": float(np.min(step_times)),
                "steady_state_max": float(np.max(step_times)),
                "useful_w13_tflops_per_rank": _useful_w13_tflops(shape, median),
            }
        )
    except Exception as exc:
        if settings.debug_exceptions:
            raise
        row.update({"ok": False, "error_type": type(exc).__name__, "error": str(exc)})
    return row


def _materialized_w13_swiglu(
    lhs: jax.Array,
    rhs: jax.Array,
    group_sizes: jax.Array,
    config: blackwell_ragged_dot_mgpu.TuningConfig,
) -> jax.Array:
    h = blackwell_ragged_dot_mgpu.ragged_dot_kernel(lhs, rhs, group_sizes, config=config)
    gate, up = jnp.split(h, 2, axis=-1)
    return (_silu(gate) * up).astype(lhs.dtype)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=("quick", "target"), default="quick")
    parser.add_argument("--implementation", choices=("fused", "materialized"), default="fused")
    parser.add_argument("--m", type=int, default=TARGET_M)
    parser.add_argument("--hidden-dim", type=int, default=TARGET_HIDDEN_DIM)
    parser.add_argument("--intermediate-dim", type=int, default=TARGET_INTERMEDIATE_DIM)
    parser.add_argument("--num-groups", type=int, default=1)
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--tile-m", type=_parse_int_csv, default=(DEFAULT_TUNING_CONFIG.tile_m,))
    parser.add_argument("--tile-n", type=_parse_int_csv, default=(DEFAULT_TUNING_CONFIG.tile_n,))
    parser.add_argument("--tile-k", type=_parse_int_csv, default=(DEFAULT_TUNING_CONFIG.tile_k,))
    parser.add_argument(
        "--max-concurrent-steps", type=_parse_int_csv, default=(DEFAULT_TUNING_CONFIG.max_concurrent_steps,)
    )
    parser.add_argument("--collective", type=_parse_bool_csv, default=(DEFAULT_TUNING_CONFIG.collective,))
    parser.add_argument("--grid-tile-width", type=_parse_int_csv, default=(DEFAULT_TUNING_CONFIG.grid_tile_width,))
    parser.add_argument("--grid-minor-dim", type=_parse_minor_dim_csv, default=(DEFAULT_GRID_MINOR_DIM,))
    parser.add_argument("--epilogue-tile-n", type=_parse_int_csv, default=(DEFAULT_TUNING_CONFIG.epilogue_tile_n,))
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--check", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--debug-exceptions", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--jsonl", type=str, default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    device_kind = _require_blackwell_gpu()
    shape = W13SwiGLUShape(
        m=args.m,
        hidden_dim=args.hidden_dim,
        intermediate_dim=args.intermediate_dim,
        num_groups=args.num_groups,
        dtype=args.dtype,
    )
    settings = W13SwiGLURunSettings(
        warmup=args.warmup,
        steps=args.steps,
        check=args.check,
        debug_exceptions=args.debug_exceptions,
        implementation=args.implementation,
    )
    shape.validate()
    settings.validate()
    lhs, rhs, group_sizes = _make_inputs(shape)

    if args.jsonl:
        jsonl_dir = os.path.dirname(args.jsonl)
        if jsonl_dir:
            os.makedirs(jsonl_dir, exist_ok=True)

    for config in _candidate_configs(args):
        row = _benchmark_config(shape, settings, lhs, rhs, group_sizes, config)
        row["device_kind"] = device_kind
        row["jax_version"] = jax.__version__
        line = json.dumps(row, sort_keys=True)
        print(line, flush=True)
        if args.jsonl:
            with open(args.jsonl, "a", encoding="utf-8") as f:
                print(line, file=f, flush=True)


if __name__ == "__main__":
    main()
