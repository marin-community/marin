# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Probe SMEM-mediated Mosaic GPU block transport for MoE dispatch-up."""

import argparse
import functools
import time
from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax, shard_map
from jax.experimental import pallas as pl
from jax.experimental.pallas import mosaic_gpu as plgpu
from jax.sharding import AxisType, Mesh, NamedSharding, PartitionSpec as P


def _time_block(label: str, fn: Callable[[], jax.Array]) -> tuple[jax.Array, float]:
    start = time.perf_counter()
    result = fn()
    jax.block_until_ready(result)
    elapsed = time.perf_counter() - start
    print(f"{label}: {elapsed * 1e3:.3f} ms")
    return result, elapsed


def _measure_steady_state(label: str, fn: Callable[[], jax.Array], *, warmup_steps: int, bench_iters: int) -> float:
    for _ in range(warmup_steps):
        jax.block_until_ready(fn())

    times = []
    for _ in range(bench_iters):
        start = time.perf_counter()
        jax.block_until_ready(fn())
        times.append(time.perf_counter() - start)
    times_ms = np.asarray(times) * 1e3
    print(
        f"{label}/steady: "
        f"mean={float(np.mean(times_ms)):.3f} ms "
        f"min={float(np.min(times_ms)):.3f} ms "
        f"max={float(np.max(times_ms)):.3f} ms "
        f"iters={bench_iters}"
    )
    return float(np.mean(times_ms))


def _remote_block_copy_local(
    send_x_by_dst: jax.Array,
    *,
    axis_name: str,
    block_rows: int,
    block_cols: int,
    use_scratch_output: bool,
    loop_tiles: bool,
) -> jax.Array:
    if send_x_by_dst.ndim != 3:
        raise ValueError(f"send_x_by_dst must have shape [EP, R, H], got {send_x_by_dst.shape}")
    if block_rows < 1:
        raise ValueError(f"block_rows must be positive, got {block_rows}")
    if block_cols < 1:
        raise ValueError(f"block_cols must be positive, got {block_cols}")
    ep_size, rows, hidden = send_x_by_dst.shape
    if rows % block_rows != 0:
        raise ValueError(f"rows={rows} must be divisible by block_rows={block_rows}")
    if hidden % block_cols != 0:
        raise ValueError(f"hidden={hidden} must be divisible by block_cols={block_cols}")
    row_blocks = rows // block_rows
    col_blocks = hidden // block_cols
    tiles = row_blocks * col_blocks

    def copy_body(send_ref, recv_ref):
        recv_sem = pl.get_global(plgpu.SemaphoreType.REGULAR)
        src_rank = lax.axis_index(axis_name)
        dst_rank = pl.program_id(0)
        remote_recv_ref = plgpu.remote_ref(recv_ref, jnp.int32(dst_rank))

        def copy_tile(tile):
            row_block = tile // col_blocks
            col_block = tile - row_block * col_blocks
            row_start = row_block * block_rows
            col_start = col_block * block_cols

            @functools.partial(
                pl.run_scoped,
                tile_smem=plgpu.SMEM((block_rows, block_cols), dtype=send_ref.dtype),
                barrier=plgpu.Barrier(),
            )
            def copy_scope(
                tile_smem,
                barrier,
            ):
                src_slice = send_ref.at[dst_rank, pl.ds(row_start, block_rows), pl.ds(col_start, block_cols)]
                plgpu.copy_gmem_to_smem(src_slice, tile_smem, barrier)
                plgpu.barrier_wait(barrier)
                dst_slice = remote_recv_ref.at[src_rank, pl.ds(row_start, block_rows), pl.ds(col_start, block_cols)]
                plgpu.copy_smem_to_gmem(tile_smem, dst_slice)
                plgpu.wait_smem_to_gmem(0, wait_read_only=False)

        if loop_tiles:

            @pl.loop(0, tiles)
            def _copy_loop(tile):
                copy_tile(tile)

            pl.semaphore_signal(recv_sem, device_id=jnp.int32(dst_rank))
            pl.semaphore_wait(recv_sem, value=ep_size, decrement=False)
            return

        copy_tile(pl.program_id(1))

        pl.semaphore_signal(recv_sem, device_id=jnp.int32(dst_rank))
        pl.semaphore_wait(recv_sem, value=ep_size * row_blocks * col_blocks, decrement=False)

    grid = (ep_size,) if loop_tiles else (ep_size, tiles)
    grid_names = ("dst",) if loop_tiles else ("dst", "tile")
    if not use_scratch_output:
        return plgpu.kernel(
            copy_body,
            out_shape=jax.ShapeDtypeStruct((ep_size, rows, hidden), send_x_by_dst.dtype),
            grid=grid,
            grid_names=grid_names,
        )(send_x_by_dst)

    def scratch_kernel_body(send_ref, dummy_ref, scratch_ref):
        del dummy_ref
        copy_body(send_ref, scratch_ref)

    _, scratch = plgpu.kernel(
        scratch_kernel_body,
        out_shape=[
            jax.ShapeDtypeStruct((), send_x_by_dst.dtype),
            jax.ShapeDtypeStruct((ep_size, rows, hidden), send_x_by_dst.dtype),
        ],
        grid=grid,
        grid_names=grid_names,
    )(send_x_by_dst)
    return scratch


def _block_copy_fn(
    mesh: Mesh,
    *,
    block_rows: int,
    block_cols: int,
    use_scratch_output: bool,
    loop_tiles: bool,
) -> Callable[[jax.Array], jax.Array]:
    def local_copy(send_x):
        recv = _remote_block_copy_local(
            jnp.squeeze(send_x, axis=0),
            axis_name="expert",
            block_rows=block_rows,
            block_cols=block_cols,
            use_scratch_output=use_scratch_output,
            loop_tiles=loop_tiles,
        )
        return recv[None, ...]

    return jax.jit(
        shard_map(
            local_copy,
            mesh=mesh,
            in_specs=P("expert", None, None, None),
            out_specs=P("expert", None, None, None),
            check_vma=False,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ep-size", type=int, default=8)
    parser.add_argument("--rows", type=int, default=32)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--block-rows", type=int, default=16)
    parser.add_argument("--block-cols", type=int, default=128)
    parser.add_argument("--dtype", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--warmup-steps", type=int, default=1)
    parser.add_argument("--bench-iters", type=int, default=1)
    parser.add_argument("--skip-reference-checks", action="store_true")
    parser.add_argument("--use-scratch-output", action="store_true")
    parser.add_argument("--loop-tiles", action="store_true")
    args = parser.parse_args()

    dtype = jnp.bfloat16 if args.dtype == "bf16" else jnp.float32
    devices = jax.local_devices()
    print(f"devices: {len(devices)} {[device.platform for device in devices]}")
    print(
        f"shape: EP={args.ep_size} rows={args.rows} H={args.hidden} "
        f"block_rows={args.block_rows} block_cols={args.block_cols} dtype={args.dtype}"
    )
    print(f"use_scratch_output={args.use_scratch_output}")
    print(f"loop_tiles={args.loop_tiles}")
    if len(devices) < args.ep_size:
        raise RuntimeError(f"Need at least {args.ep_size} local devices, found {len(devices)}")

    mesh = Mesh(np.array(devices[: args.ep_size]), ("expert",), axis_types=(AxisType.Explicit,))
    send_sharding = NamedSharding(mesh, P("expert", None, None, None))
    with jax.set_mesh(mesh):
        send_x = jax.random.normal(
            jax.random.key(6597),
            (args.ep_size, args.ep_size, args.rows, args.hidden),
            dtype=dtype,
            out_sharding=send_sharding,
        )
        expected_host = None
        if not args.skip_reference_checks:
            expected_host = np.asarray(send_x).transpose((1, 0, 2, 3))

        copy_fn = _block_copy_fn(
            mesh,
            block_rows=args.block_rows,
            block_cols=args.block_cols,
            use_scratch_output=args.use_scratch_output,
            loop_tiles=args.loop_tiles,
        )

        def run_copy():
            return copy_fn(send_x)

        actual, _ = _time_block("transport/block_copy", run_copy)
        if args.bench_iters > 0:
            steady_ms = _measure_steady_state(
                "transport/block_copy",
                run_copy,
                warmup_steps=args.warmup_steps,
                bench_iters=args.bench_iters,
            )
            dtype_bytes = 2 if dtype == jnp.bfloat16 else 4
            payload_bytes = args.ep_size * args.ep_size * args.rows * args.hidden * dtype_bytes
            print(f"transport_payload={payload_bytes / 1024 / 1024:.3f} MiB")
            print(f"transport_bandwidth={payload_bytes / (steady_ms / 1e3) / 1e9:.6f} GB/s")

        if expected_host is not None:
            actual_host = np.asarray(actual)
            max_abs_float = float(np.max(np.abs(actual_host.astype(np.float32) - expected_host.astype(np.float32))))
            print(f"transport_max_abs_error: {max_abs_float:.6g}")
            if max_abs_float != 0.0:
                raise AssertionError(f"transport max abs error {max_abs_float:.6g}")


if __name__ == "__main__":
    main()
