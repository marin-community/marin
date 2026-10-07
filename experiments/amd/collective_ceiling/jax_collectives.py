# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Measure the collective bandwidth JAX reaches across the GPUs of one node.

Times ``all_gather``, ``psum_scatter`` (reduce-scatter), ``psum`` (all-reduce) and ``all_to_all``
inside ``shard_map`` over a 1-D mesh of every local device, at a list of message sizes. Each case
compiles once, then makes one mandatory warm-up call. ``--extra-warmup-seconds`` adds back-to-back
warm-up calls after it; 0 skips them. The mean time of all warm-up calls sizes the timed windows,
and the script reports the median of ``--repeats`` windows of about ``--window-seconds``.

Sizes and bus bandwidth follow the nccl-tests conventions, which rccl-tests shares
(https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md). The size is the full
(gathered) buffer for all-gather and reduce-scatter, the per-rank buffer for all-reduce, and the
per-rank send buffer for all-to-all. Bus bandwidth is ``size / time`` times ``(n-1)/n``
(all-gather, reduce-scatter, all-to-all) or ``2(n-1)/n`` (all-reduce).

``--overlap`` also times each collective alongside an independent bf16 matmul in the same program,
to show how much the two slow each other down when XLA runs the collective asynchronously. The matmul
is timed alone once per run. Overlap rows divide the size by the time of the whole program, collective
and matmul together, so their bandwidths are effective rates that include the matmul.
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
from enum import StrEnum
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

logger = logging.getLogger(__name__)

AXIS = "x"
DTYPES = {"bfloat16": jnp.bfloat16, "float32": jnp.float32}
# Row width of the EP all-gather output in the June 67B-A2B MoE model (experiments/june_tpu_67b_a2b):
# 262,144 tokens x 2,560 hidden at batch 64.
MODEL_HIDDEN_DIM = 2560
MATMUL_DIM = 8192
# Chained matmuls in the overlap test: about 4-5 ms on MI350X, close to a 1.3 GB all-gather.
MATMUL_CHAIN = 4


class Collective(StrEnum):
    ALL_GATHER = "all_gather"
    REDUCE_SCATTER = "reduce_scatter"
    ALL_REDUCE = "all_reduce"
    ALL_TO_ALL = "all_to_all"


@dataclasses.dataclass(frozen=True)
class Timing:
    extra_warmup_seconds: float
    window_seconds: float
    repeats: int


@dataclasses.dataclass(frozen=True)
class OpResult:
    op: Collective
    dtype: str
    size_bytes: int
    overlap: bool
    iterations_per_window: int
    median_seconds: float
    algbw_gbps: float
    busbw_gbps: float
    matmul_alone_seconds: float | None


def _bus_factor(op: Collective, n: int) -> float:
    return 2 * (n - 1) / n if op == Collective.ALL_REDUCE else (n - 1) / n


def _collective(op: Collective) -> Callable[[jax.Array], jax.Array]:
    if op == Collective.ALL_GATHER:
        return lambda x: jax.lax.all_gather(x, AXIS, tiled=True)
    if op == Collective.REDUCE_SCATTER:
        return lambda x: jax.lax.psum_scatter(x, AXIS, scatter_dimension=0, tiled=True)
    if op == Collective.ALL_REDUCE:
        return lambda x: jax.lax.psum(x, AXIS)
    if op == Collective.ALL_TO_ALL:
        return lambda x: jax.lax.all_to_all(x, AXIS, split_axis=0, concat_axis=0, tiled=True)
    raise ValueError(f"unknown op {op}")


def _input_rows_per_rank(op: Collective, size_bytes: int, row_bytes: int, n: int) -> int:
    """Rows of the per-rank input so the rccl-tests size convention holds; rounded to a multiple of n."""
    rows = size_bytes // row_bytes
    if op == Collective.ALL_GATHER:
        rows = rows // n
    return max(n, rows - rows % n)


def _time(fn: Callable[[], object], timing: Timing) -> tuple[float, int]:
    jax.block_until_ready(fn())  # Compile.
    start = time.perf_counter()
    jax.block_until_ready(fn())  # Mandatory warm-up call; always defines the per-call time.
    extra_start = time.perf_counter()
    calls = 1
    # Block on each extra call so the deadline counts finished device work, not queued dispatches.
    while time.perf_counter() - extra_start < timing.extra_warmup_seconds:
        jax.block_until_ready(fn())
        calls += 1
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


def _chained_matmul(a: jax.Array, b: jax.Array) -> jax.Array:
    for _ in range(MATMUL_CHAIN):
        a = a @ b
    return a


def _matmul_inputs(mesh: Mesh) -> tuple[jax.Array, jax.Array]:
    sharding = NamedSharding(mesh, P(AXIS))
    shape = (MATMUL_DIM * mesh.size, MATMUL_DIM)
    return (
        jax.device_put(jnp.zeros(shape, jnp.bfloat16), sharding),
        jax.device_put(jnp.zeros(shape, jnp.bfloat16), sharding),
    )


def time_matmul(mesh: Mesh, timing: Timing) -> float:
    """Median seconds per call of the overlap test's chained matmul, run alone."""
    spec = P(AXIS)
    mm = jax.jit(jax.shard_map(_chained_matmul, mesh=mesh, in_specs=(spec, spec), out_specs=spec, check_vma=False))
    a, b = _matmul_inputs(mesh)
    seconds, _ = _time(lambda: mm(a, b), timing)
    return seconds


def run_op(mesh: Mesh, op: Collective, dtype: str, size_bytes: int, *, overlap: bool, timing: Timing) -> OpResult:
    n = mesh.size
    hidden = MODEL_HIDDEN_DIM
    row_bytes = hidden * jnp.dtype(DTYPES[dtype]).itemsize
    rows = _input_rows_per_rank(op, size_bytes, row_bytes, n)
    spec = P(AXIS)
    sharding = NamedSharding(mesh, spec)
    x = jax.device_put(jnp.ones((rows * n, hidden), DTYPES[dtype]), sharding)
    collective = _collective(op)
    actual_bytes = rows * row_bytes * (n if op == Collective.ALL_GATHER else 1)

    if overlap:
        a, b = _matmul_inputs(mesh)

        @partial(jax.shard_map, mesh=mesh, in_specs=(spec, spec, spec), out_specs=(spec, spec), check_vma=False)
        def both(x, a, b):
            return collective(x), _chained_matmul(a, b)

        both_jit = jax.jit(both)
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
        matmul_alone_seconds=None,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ops", nargs="+", type=Collective, default=list(Collective), choices=list(Collective))
    parser.add_argument("--dtypes", nargs="+", default=["bfloat16"], choices=sorted(DTYPES))
    parser.add_argument(
        "--sizes-mb",
        type=float,
        nargs="+",
        default=[8, 16, 32, 64, 128, 256, 512, 1024, 1342.177, 2048],
        help="Message sizes in MB (1e6 bytes); 1342.177 is the June 67B-A2B EP all-gather at batch 64.",
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
    matmul_alone_seconds = time_matmul(mesh, timing) if args.overlap else None
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("a") as out:
        out.write(json.dumps({"header": header}) + "\n")
        for dtype in args.dtypes:
            for op in args.ops:
                for size_mb in args.sizes_mb:
                    for overlap in [False, True] if args.overlap else [False]:
                        result = run_op(mesh, op, dtype, int(size_mb * 1e6), overlap=overlap, timing=timing)
                        if overlap:
                            result = dataclasses.replace(result, matmul_alone_seconds=matmul_alone_seconds)
                        out.write(json.dumps({"label": args.label, **dataclasses.asdict(result)}) + "\n")
                        out.flush()
                        logger.info(
                            "%s %-14s %-8s %8.1f MB overlap=%d %8.3f ms %s %6.1f GB/s%s",
                            args.label,
                            op,
                            dtype,
                            result.size_bytes / 1e6,
                            result.overlap,
                            result.median_seconds * 1e3,
                            "effective busbw" if result.overlap else "busbw",
                            result.busbw_gbps,
                            (
                                ""
                                if result.matmul_alone_seconds is None
                                else f" (matmul alone {result.matmul_alone_seconds * 1e3:.3f} ms)"
                            ),
                        )


if __name__ == "__main__":
    main()
