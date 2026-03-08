# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Standalone SparseCore scatter repro for TPU runtime experiments.

This script intentionally avoids importing Levanter modules so it can be run in
an isolated environment, for example with a newer temporary JAX install.
"""

from __future__ import annotations

import argparse
from functools import partial
import time

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.pallas import tpu_sc as plsc


def _parse_dtype(name: str) -> jnp.dtype:
    mapping = {
        "bf16": jnp.bfloat16,
        "f32": jnp.float32,
        "i32": jnp.int32,
    }
    try:
        return mapping[name]
    except KeyError as exc:
        raise ValueError(f"Unsupported dtype {name!r}; expected one of {sorted(mapping)}") from exc


def _sparsecore_lane_width() -> int:
    if jax.default_backend() != "tpu":
        raise RuntimeError("SparseCore repro requires TPU backend")
    device_kind = jax.devices()[0].device_kind
    if device_kind in ("TPU v5", "TPU v5p", "TPU v6", "TPU v6 lite"):
        return 8
    if device_kind == "TPU7x":
        return 16
    raise NotImplementedError(f"Unsupported device kind {device_kind!r}")


def _xla_scatter_set(updates: jax.Array, row_indices: jax.Array, *, num_rows: int) -> jax.Array:
    out = jnp.zeros((num_rows, updates.shape[1]), dtype=updates.dtype)
    return out.at[row_indices].set(updates, mode="drop")


def _sc_scatter_set_unique(updates: jax.Array, row_indices: jax.Array, *, num_rows: int) -> jax.Array:
    if hasattr(pl, "kernel"):
        return _sc_scatter_set_unique_kernel_api(updates, row_indices, num_rows=num_rows)
    return _sc_scatter_set_unique_pallas_call(updates, row_indices, num_rows=num_rows)


def _sc_scatter_set_unique_kernel_api(updates: jax.Array, row_indices: jax.Array, *, num_rows: int) -> jax.Array:
    if row_indices.dtype != jnp.int32:
        row_indices = row_indices.astype(jnp.int32)
    window = 128
    if updates.shape[0] % window != 0:
        raise ValueError(f"updates.shape[0]={updates.shape[0]} must be divisible by window={window}")
    indices = row_indices.reshape((1, updates.shape[0]))
    vector_mesh = plsc.VectorSubcoreMesh(core_axis_name="core", subcore_axis_name="subcore")

    @jax.jit
    def scatter(x, idx):
        @pl.kernel(
            out_shape=jax.ShapeDtypeStruct((num_rows, x.shape[1]), x.dtype),
            mesh=vector_mesh,
            scratch_shapes=[],
        )
        def kernel(x_hbm, i_hbm, o_hbm):
            def body(x_vmem, i_vmem):
                pltpu.sync_copy(x_vmem, o_hbm.at[i_vmem.at[0]])

            pltpu.emit_pipeline(
                body,
                grid=(x.shape[0] // window,),
                in_specs=[
                    pl.BlockSpec((window, x.shape[1]), index_map=lambda i: (i, 0)),
                    pl.BlockSpec((1, window), index_map=lambda i: (0, i)),
                ],
                out_specs=[],
                dimension_semantics=(pltpu.PARALLEL,),
            )(x_hbm, i_hbm)

        return kernel(x, idx)

    return scatter(updates, indices)


def _sc_scatter_set_unique_pallas_call(updates: jax.Array, row_indices: jax.Array, *, num_rows: int) -> jax.Array:
    lane_width = _sparsecore_lane_width()
    if row_indices.dtype != jnp.int32:
        row_indices = row_indices.astype(jnp.int32)
    row_indices_2d = jnp.broadcast_to(row_indices[:, None], (row_indices.shape[0], lane_width))

    num_updates, row_width = updates.shape

    @partial(
        pl.pallas_call,
        out_shape=jax.ShapeDtypeStruct((num_rows, row_width), updates.dtype),
        grid=(),
        in_specs=(
            pl.BlockSpec((num_updates, row_width), lambda: (0, 0)),
            pl.BlockSpec((num_updates, lane_width), lambda: (0, 0)),
        ),
        out_specs=pl.BlockSpec((num_rows, row_width), lambda: (0, 0), memory_space=pl.MemorySpace.ANY),
        compiler_params=pltpu.CompilerParams(
            kernel_type=pltpu.KernelType.SC_VECTOR_SUBCORE,
            dimension_semantics=(),
        ),
    )
    def kernel(updates_hbm, indices_hbm, out_hbm):
        @pl.loop(0, out_hbm.shape[0])
        def _(row):
            @pl.loop(0, out_hbm.shape[1])
            def _(col):
                out_hbm.at[row, col][...] = jnp.zeros((), dtype=out_hbm.dtype)

        def body(update_vmem, index_vmem):
            row_index = index_vmem.at[0, 0][...]
            pltpu.sync_copy(update_vmem.at[0], out_hbm.at[row_index])

        pltpu.emit_pipeline(
            body,
            grid=(num_updates,),
            in_specs=(
                pl.BlockSpec((1, row_width), lambda row: (row, 0)),
                pl.BlockSpec((1, lane_width), lambda row: (row, 0)),
            ),
            out_specs=(),
            dimension_semantics=(pltpu.PARALLEL,),
        )(updates_hbm, indices_hbm)

    return kernel(updates, row_indices_2d)


def _bench(fn, updates: jax.Array, row_indices: jax.Array, *, warmup: int, iters: int) -> tuple[float, float]:
    step = jax.jit(fn)

    start = time.perf_counter()
    out = step(updates, row_indices)
    jax.block_until_ready(out)
    compile_s = time.perf_counter() - start

    for _ in range(warmup):
        out = step(updates, row_indices)
        jax.block_until_ready(out)

    start = time.perf_counter()
    for _ in range(iters):
        out = step(updates, row_indices)
        jax.block_until_ready(out)
    steady_s = (time.perf_counter() - start) / iters
    return compile_s, steady_s


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("xla", "sparsecore"), required=True)
    parser.add_argument("--rows", type=int, default=163840)
    parser.add_argument("--updates", type=int, default=40960)
    parser.add_argument("--width", type=int, default=2048)
    parser.add_argument("--dtype", choices=("bf16", "f32", "i32"), default="bf16")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iters", type=int, default=3)
    args = parser.parse_args()

    if jax.default_backend() != "tpu":
        raise RuntimeError("This repro requires TPU backend")

    dtype = _parse_dtype(args.dtype)
    key_updates, key_perm = jax.random.split(jax.random.key(0))
    if dtype == jnp.int32:
        updates = jax.random.randint(key_updates, (args.updates, args.width), 0, 1024, dtype=dtype)
    else:
        updates = jax.random.normal(key_updates, (args.updates, args.width), dtype=dtype)
    row_indices = jax.random.permutation(key_perm, args.rows)[: args.updates].astype(jnp.int32)

    print("jax", jax.__version__)
    print("devices", jax.devices())
    print("mode", args.mode, "rows", args.rows, "updates", args.updates, "width", args.width, "dtype", dtype)

    if args.mode == "xla":
        fn = lambda x, i: _xla_scatter_set(x, i, num_rows=args.rows)
    else:
        fn = lambda x, i: _sc_scatter_set_unique(x, i, num_rows=args.rows)

    compile_s, steady_s = _bench(fn, updates, row_indices, warmup=args.warmup, iters=args.iters)
    print(f"{args.mode},compile_s={compile_s:.6f},steady_s={steady_s:.6f}")


if __name__ == "__main__":
    main()
