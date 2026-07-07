#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Small JAX GPU benchmark for GEMM, HBM bandwidth, and collective bus bandwidth."""

from __future__ import annotations

import argparse
import json
import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import Any

os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", "/tmp/jax-compilation-cache")

import jax
import jax.numpy as jnp
import numpy as np
from iris.runtime.jax_init import initialize_jax
from jax import lax


@dataclass(frozen=True)
class Timing:
    seconds: float
    iterations: int


def _now() -> float:
    return time.perf_counter()


def _block(value: Any) -> Any:
    return jax.block_until_ready(value)


def _time_call(fn: Callable[[], Any], *, warmup: int, iterations: int) -> Timing:
    for _ in range(warmup):
        _block(fn())

    start = _now()
    for _ in range(iterations):
        _block(fn())
    end = _now()
    return Timing(seconds=(end - start) / iterations, iterations=iterations)


def _parse_csv_ints(raw: str) -> list[int]:
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def _json_record(record: dict[str, Any]) -> None:
    print(json.dumps(record, sort_keys=True), flush=True)


def _initialize_distributed_if_needed() -> None:
    if "IRIS_TASK_ID" in os.environ or "IRIS_MULTIGPU_PROCESS_INDEX" in os.environ:
        initialize_jax()
        return

    coordinator = os.environ.get("JAX_COORDINATOR_ADDRESS")
    process_count = os.environ.get("JAX_PROCESS_COUNT")
    process_id = os.environ.get("JAX_PROCESS_ID")
    if coordinator and process_count and process_id:
        jax.distributed.initialize(coordinator, int(process_count), int(process_id))


def _device_summary() -> dict[str, Any]:
    devices = jax.devices()
    return {
        "jax_version": jax.__version__,
        "process_index": jax.process_index(),
        "process_count": jax.process_count(),
        "device_count": jax.device_count(),
        "local_device_count": jax.local_device_count(),
        "devices": [str(device) for device in devices],
        "platform": jax.default_backend(),
        "host": os.uname().nodename,
    }


def _gemm_case(dtype_name: str, size: int, warmup: int, iterations: int) -> dict[str, Any]:
    if dtype_name == "bf16":
        dtype = jnp.bfloat16
    elif dtype_name == "fp8_e4m3":
        dtype = jnp.float8_e4m3fn
    elif dtype_name == "fp8_e5m2":
        dtype = jnp.float8_e5m2
    else:
        raise ValueError(f"unknown GEMM dtype {dtype_name}")

    key_a, key_b = jax.random.split(jax.random.PRNGKey(size))
    a = jax.random.normal(key_a, (size, size), dtype=jnp.bfloat16).astype(dtype)
    b = jax.random.normal(key_b, (size, size), dtype=jnp.bfloat16).astype(dtype)

    @jax.jit
    def matmul(x: jax.Array, y: jax.Array) -> jax.Array:
        return x @ y

    timing = _time_call(lambda: matmul(a, b), warmup=warmup, iterations=iterations)
    flops = 2 * size * size * size
    out = _block(matmul(a, b))
    return {
        "kind": "gemm",
        "dtype": dtype_name,
        "size": size,
        "seconds": timing.seconds,
        "iterations": timing.iterations,
        "tflops": flops / timing.seconds / 1e12,
        "output_dtype": str(out.dtype),
    }


def _hbm_case(size_mb: int, warmup: int, iterations: int) -> dict[str, Any]:
    bytes_requested = size_mb * 1024 * 1024
    elems = bytes_requested // np.dtype(np.float32).itemsize
    x = jax.random.normal(jax.random.PRNGKey(size_mb), (elems,), dtype=jnp.float32)

    @jax.jit
    def stream(inp: jax.Array) -> jax.Array:
        return inp * jnp.float32(1.0001) + jnp.float32(1.0)

    timing = _time_call(lambda: stream(x), warmup=warmup, iterations=iterations)
    bytes_moved = 2 * x.size * x.dtype.itemsize
    return {
        "kind": "hbm_stream",
        "dtype": "float32",
        "size_mb": size_mb,
        "seconds": timing.seconds,
        "iterations": timing.iterations,
        "tb_per_s": bytes_moved / timing.seconds / 1e12,
    }


def _collective_logical_bytes(kind: str, shard_bytes: int, ranks: int) -> int:
    if kind == "all_reduce":
        return shard_bytes
    if kind in {"all_gather", "reduce_scatter"}:
        return shard_bytes * ranks
    raise ValueError(f"unknown collective {kind}")


def _collective_busbw(kind: str, algbw: float, ranks: int) -> float:
    if ranks <= 1:
        return 0.0
    if kind == "all_reduce":
        return algbw * (2 * (ranks - 1) / ranks)
    if kind in {"all_gather", "reduce_scatter"}:
        return algbw * ((ranks - 1) / ranks)
    raise ValueError(f"unknown collective {kind}")


def _collective_case(kind: str, message_mb: int, warmup: int, iterations: int) -> dict[str, Any]:
    local_devices = jax.local_device_count()
    ranks = jax.device_count()
    dtype = jnp.bfloat16
    itemsize = np.dtype(jax.dtypes.canonicalize_dtype(dtype)).itemsize
    message_bytes = message_mb * 1024 * 1024
    elems = max(1, message_bytes // itemsize)
    tokens = jnp.arange(local_devices)

    if kind == "all_reduce":

        @partial(jax.pmap, axis_name="rank")
        def collective(token: jax.Array) -> jax.Array:
            scale = token.astype(dtype) + jnp.array(1, dtype=dtype)
            v = jnp.ones((elems,), dtype=dtype) * scale
            return lax.psum(v, "rank")

        def fn() -> jax.Array:
            return collective(tokens)

    elif kind == "all_gather":

        @partial(jax.pmap, axis_name="rank")
        def collective(token: jax.Array) -> jax.Array:
            scale = token.astype(dtype) + jnp.array(1, dtype=dtype)
            v = jnp.ones((elems,), dtype=dtype) * scale
            return lax.all_gather(v, "rank", axis=0)

        def fn() -> jax.Array:
            return collective(tokens)

    elif kind == "reduce_scatter":

        @partial(jax.pmap, axis_name="rank")
        def collective(token: jax.Array) -> jax.Array:
            scale = token.astype(dtype) + jnp.array(1, dtype=dtype)
            v = jnp.ones((ranks, elems), dtype=dtype) * scale
            return lax.psum_scatter(v, "rank", scatter_dimension=0, tiled=True)

        def fn() -> jax.Array:
            return collective(tokens)

    else:
        raise ValueError(f"unknown collective {kind}")

    timing = _time_call(fn, warmup=warmup, iterations=iterations)
    logical_bytes = _collective_logical_bytes(kind, message_bytes, ranks)
    algbw = logical_bytes / timing.seconds / 1e9
    busbw = _collective_busbw(kind, algbw, ranks)
    return {
        "kind": "collective",
        "collective": kind,
        "dtype": "bf16",
        "message_mb": message_mb,
        "logical_mb": logical_bytes / 1024 / 1024,
        "ranks": ranks,
        "local_ranks": local_devices,
        "seconds": timing.seconds,
        "iterations": timing.iterations,
        "algbw_gb_s": algbw,
        "busbw_gb_s": busbw,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-gemm", action="store_true")
    parser.add_argument("--skip-hbm", action="store_true")
    parser.add_argument("--skip-collectives", action="store_true")
    parser.add_argument("--gemm-sizes", default="8192,16384")
    parser.add_argument("--gemm-dtypes", default="bf16,fp8_e4m3")
    parser.add_argument("--hbm-size-mb", type=int, default=2048)
    parser.add_argument("--collective-message-mb", default="1,8,64,256,512,1024")
    parser.add_argument("--collectives", default="all_reduce,all_gather,reduce_scatter")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    args = parser.parse_args()

    _initialize_distributed_if_needed()
    _json_record({"kind": "summary", **_device_summary()})

    if not args.skip_gemm:
        for size in _parse_csv_ints(args.gemm_sizes):
            for dtype_name in [part.strip() for part in args.gemm_dtypes.split(",") if part.strip()]:
                try:
                    _json_record(_gemm_case(dtype_name, size, args.warmup, args.iterations))
                except Exception as exc:
                    _json_record({"kind": "gemm_error", "dtype": dtype_name, "size": size, "error": repr(exc)})

    if not args.skip_hbm:
        try:
            _json_record(_hbm_case(args.hbm_size_mb, args.warmup, args.iterations))
        except Exception as exc:
            _json_record({"kind": "hbm_error", "size_mb": args.hbm_size_mb, "error": repr(exc)})

    if not args.skip_collectives and jax.device_count() > 1:
        for message_mb in _parse_csv_ints(args.collective_message_mb):
            for collective in [part.strip() for part in args.collectives.split(",") if part.strip()]:
                try:
                    _json_record(_collective_case(collective, message_mb, args.warmup, args.iterations))
                except Exception as exc:
                    _json_record(
                        {
                            "kind": "collective_error",
                            "collective": collective,
                            "message_mb": message_mb,
                            "error": repr(exc),
                        }
                    )


if __name__ == "__main__":
    main()
