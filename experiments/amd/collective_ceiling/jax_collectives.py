# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Measure the collective bandwidth JAX reaches across the GPUs of one node.

Times ``all_gather``, ``psum_scatter`` (reduce-scatter), ``psum`` (all-reduce) and ``all_to_all``
inside ``shard_map`` over a 1-D mesh of every local device, at a list of message sizes. Each case
compiles once, then makes one mandatory warm-up call. ``--extra-warmup-seconds`` adds back-to-back
warm-up calls after it; 0 skips them. The mean time of all warm-up calls sizes the timed windows,
and the script reports the median of ``--repeats`` windows of about ``--window-seconds``.

Sizes and bus bandwidth follow rccl-tests: the size is the full (gathered) buffer for all-gather and
reduce-scatter, the per-rank buffer for all-reduce, and the per-rank send buffer for all-to-all.
Bus bandwidth is ``size / time`` times ``(n-1)/n`` (all-gather, reduce-scatter, all-to-all) or
``2(n-1)/n`` (all-reduce).

``--overlap`` also times each collective alongside an independent bf16 matmul in the same program,
to show how much the two slow each other down when XLA runs the collective asynchronously.
"""

import argparse
import dataclasses
import json
import logging
import math
import os
import statistics
import time
from collections.abc import Callable
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

logger = logging.getLogger(__name__)

AXIS = "x"
OPS = ("all_gather", "reduce_scatter", "all_reduce", "all_to_all")
DTYPES = {"bfloat16": jnp.bfloat16, "float32": jnp.float32}
# Row width of the June model's EP all-gather output (262,144 tokens x 2,560 hidden at batch 64).
MODEL_HIDDEN_DIM = 2560
MATMUL_DIM = 8192
# Chained matmuls in the overlap test: about 4-5 ms on MI350X, close to a 1.3 GB all-gather.
MATMUL_CHAIN = 4


@dataclasses.dataclass(frozen=True)
class Timing:
    extra_warmup_seconds: float
    window_seconds: float
    repeats: int


@dataclasses.dataclass(frozen=True)
class OpResult:
    op: str
    dtype: str
    size_bytes: int
    overlap: bool
    iterations_per_window: int
    median_seconds: float
    algbw_gbps: float
    busbw_gbps: float
    matmul_alone_seconds: float | None


def _bus_factor(op: str, n: int) -> float:
    return 2 * (n - 1) / n if op == "all_reduce" else (n - 1) / n


def _collective(op: str) -> Callable[[jax.Array], jax.Array]:
    if op == "all_gather":
        return lambda x: jax.lax.all_gather(x, AXIS, tiled=True)
    if op == "reduce_scatter":
        return lambda x: jax.lax.psum_scatter(x, AXIS, scatter_dimension=0, tiled=True)
    if op == "all_reduce":
        return lambda x: jax.lax.psum(x, AXIS)
    if op == "all_to_all":
        return lambda x: jax.lax.all_to_all(x, AXIS, split_axis=0, concat_axis=0, tiled=True)
    raise ValueError(f"unknown op {op}")


def _input_rows_per_rank(op: str, size_bytes: int, row_bytes: int, n: int) -> int:
    """Rows of the per-rank input so the rccl-tests size convention holds; rounded to a multiple of n."""
    rows = size_bytes // row_bytes
    if op == "all_gather":
        rows = rows // n
    return max(n, rows - rows % n)


def _time(fn: Callable[[], object], timing: Timing) -> tuple[float, int]:
    jax.block_until_ready(fn())  # Compile.
    start = time.perf_counter()
    out = jax.block_until_ready(fn())  # Mandatory warm-up call; always defines the per-call time.
    extra_start = time.perf_counter()
    calls = 1
    while time.perf_counter() - extra_start < timing.extra_warmup_seconds:
        out = fn()
        calls += 1
    jax.block_until_ready(out)
    seconds_per_call = (time.perf_counter() - start) / calls
    iterations = max(5, math.ceil(timing.window_seconds / seconds_per_call))
    windows = []
    for _ in range(timing.repeats):
        start = time.perf_counter()
        for _ in range(iterations):
            out = fn()
        jax.block_until_ready(out)
        windows.append((time.perf_counter() - start) / iterations)
    return statistics.median(windows), iterations


def run_op(mesh: Mesh, op: str, dtype: str, size_bytes: int, *, overlap: bool, timing: Timing) -> OpResult:
    n = mesh.size
    hidden = MODEL_HIDDEN_DIM
    row_bytes = hidden * jnp.dtype(DTYPES[dtype]).itemsize
    rows = _input_rows_per_rank(op, size_bytes, row_bytes, n)
    spec = P(AXIS)
    sharding = NamedSharding(mesh, spec)
    x = jax.device_put(jnp.ones((rows * n, hidden), DTYPES[dtype]), sharding)
    collective = _collective(op)
    actual_bytes = rows * row_bytes * (n if op == "all_gather" else 1)

    matmul_alone = None
    if overlap:
        a = jax.device_put(jnp.zeros((MATMUL_DIM * n, MATMUL_DIM), jnp.bfloat16), sharding)
        b = jax.device_put(jnp.zeros((MATMUL_DIM * n, MATMUL_DIM), jnp.bfloat16), sharding)

        def chain(a, b):
            for _ in range(MATMUL_CHAIN):
                a = a @ b
            return a

        @partial(jax.shard_map, mesh=mesh, in_specs=(spec, spec, spec), out_specs=(spec, spec), check_vma=False)
        def both(x, a, b):
            return collective(x), chain(a, b)

        @partial(jax.shard_map, mesh=mesh, in_specs=(spec, spec), out_specs=spec, check_vma=False)
        def mm(a, b):
            return chain(a, b)

        both_jit = jax.jit(both)
        mm_jit = jax.jit(mm)
        matmul_alone, _ = _time(lambda: mm_jit(a, b), timing)
        seconds, iterations = _time(lambda: both_jit(x, a, b), timing)
    else:
        fn = jax.jit(jax.shard_map(collective, mesh=mesh, in_specs=spec, out_specs=spec, check_vma=False))
        seconds, iterations = _time(lambda: fn(x), timing)

    algbw = actual_bytes / seconds / 1e9
    return OpResult(
        op=op,
        dtype=dtype,
        size_bytes=actual_bytes,
        overlap=overlap,
        iterations_per_window=iterations,
        median_seconds=seconds,
        algbw_gbps=algbw,
        busbw_gbps=algbw * _bus_factor(op, n),
        matmul_alone_seconds=matmul_alone,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ops", nargs="+", default=list(OPS), choices=OPS)
    parser.add_argument("--dtypes", nargs="+", default=["bfloat16"], choices=sorted(DTYPES))
    parser.add_argument(
        "--sizes-mb",
        type=float,
        nargs="+",
        default=[8, 16, 32, 64, 128, 256, 512, 1024, 1342.177, 2048],
        help="Message sizes in MB (1e6 bytes); 1342.177 is the June EP all-gather at batch 64.",
    )
    parser.add_argument("--overlap", action="store_true", help="Also time each op next to an independent matmul.")
    parser.add_argument(
        "--extra-warmup-seconds",
        type=float,
        default=1.0,
        help=(
            "Back-to-back warm-up calls to add after the one mandatory warm-up call. The mandatory call "
            "always runs and, with these calls, measures the per-call time that sizes the timed windows. "
            "0 runs only the mandatory call."
        ),
    )
    parser.add_argument("--window-seconds", type=float, default=0.2)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--label", default="default", help="Name of the environment configuration.")
    parser.add_argument("--output", type=Path, required=True, help="JSON lines file to append to.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    mesh = Mesh(np.array(jax.devices()), (AXIS,))
    timing = Timing(
        extra_warmup_seconds=args.extra_warmup_seconds, window_seconds=args.window_seconds, repeats=args.repeats
    )
    header = {
        "label": args.label,
        "jax": jax.__version__,
        "devices": mesh.size,
        "device_kind": jax.devices()[0].device_kind,
        "xla_flags": os.environ.get("XLA_FLAGS", ""),
        "nccl_env": {k: v for k, v in os.environ.items() if k.startswith(("NCCL_", "RCCL_", "HSA_"))},
    }
    logger.info("%s", header)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("a") as out:
        out.write(json.dumps({"header": header}) + "\n")
        for dtype in args.dtypes:
            for op in args.ops:
                for size_mb in args.sizes_mb:
                    for overlap in [False, True] if args.overlap else [False]:
                        result = run_op(mesh, op, dtype, int(size_mb * 1e6), overlap=overlap, timing=timing)
                        out.write(json.dumps({"label": args.label, **dataclasses.asdict(result)}) + "\n")
                        out.flush()
                        logger.info(
                            "%s %-14s %-8s %8.1f MB overlap=%d %8.3f ms busbw %6.1f GB/s%s",
                            args.label,
                            op,
                            dtype,
                            result.size_bytes / 1e6,
                            overlap,
                            result.median_seconds * 1e3,
                            result.busbw_gbps,
                            (
                                ""
                                if result.matmul_alone_seconds is None
                                else f" (matmul alone {result.matmul_alone_seconds * 1e3:.3f} ms)"
                            ),
                        )


if __name__ == "__main__":
    main()
