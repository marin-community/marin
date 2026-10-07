# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Measure the dense matmul throughput JAX reaches on one accelerator.

The JAX counterpart to MAMF-finder: time ``jnp.matmul`` of an MxK by KxN matrix
for each requested shape and report the TFLOP/s it sustains. Shapes come from a
grid of ranges, a shapes file, or both. Each shape is compiled once (XLA
autotunes the GEMM during compilation), run for ``--warmup-seconds`` so clocks
settle, then timed over ``--repeats`` windows of about ``--window-seconds``.

The rate reported here is wall clock: it includes host dispatch and the idle GPU
time between calls. Each window's bounds are recorded on the clock rocprofv3
stamps kernels with, so ``kernel_trace.py`` can add the kernel-time rate from a
``rocprofv3 --kernel-trace`` of the same run; ``jax_matmul.sbatch`` does both.
"""

import argparse
import dataclasses
import itertools
import json
import logging
import math
import os
import socket
import statistics
import time
from pathlib import Path

import jax
import jax.numpy as jnp

logger = logging.getLogger(__name__)

DTYPES = {"bfloat16": jnp.bfloat16, "float16": jnp.float16}
# rocprofv3 timestamps kernels with CLOCK_BOOTTIME, so window bounds use it too.
TRACE_CLOCK = time.CLOCK_BOOTTIME


@dataclasses.dataclass(frozen=True)
class Shape:
    m: int
    n: int
    k: int

    @property
    def flops(self) -> int:
        return 2 * self.m * self.n * self.k

    def __str__(self) -> str:
        return f"{self.m}x{self.n}x{self.k}"


@dataclasses.dataclass(frozen=True)
class RunHeader:
    hostname: str
    jax: str
    device_kind: str
    platform: str
    xla_flags: str
    num_shapes: int


@dataclasses.dataclass(frozen=True)
class ShapeResult:
    shape: str
    dtype: str
    iterations_per_window: int
    window_tflops: list[float]
    median_tflops: float
    max_tflops: float
    window_bounds_ns: list[list[int]]
    """Start and end of each timed window on ``TRACE_CLOCK``."""
    kernel_window_tflops: list[float] | None = None
    """Rate of each window from GPU kernel time alone; None until kernel_trace.py attaches a trace."""
    kernel_median_tflops: float | None = None
    kernels: list[str] | None = None
    """Names of the kernels that ran in the timed windows, most frequent first."""


def parse_shape(text: str) -> Shape:
    m, n, k = (int(part) for part in text.lower().split("x"))
    return Shape(m, n, k)


def dim_values(values: list[int] | None, value_range: list[int] | None) -> list[int] | None:
    """Return the values given for one dimension, expanding ``START STOP STEP`` with STOP inclusive."""
    if value_range is None:
        return values
    start, stop, step = value_range
    return list(range(start, stop + 1, step))


def header_line(header: RunHeader) -> str:
    return json.dumps({"header": dataclasses.asdict(header)}) + "\n"


def shape_line(result: ShapeResult) -> str:
    return json.dumps(dataclasses.asdict(result)) + "\n"


def read_results(path: Path) -> tuple[RunHeader, list[ShapeResult]]:
    lines = [json.loads(line) for line in path.read_text().splitlines()]
    return RunHeader(**lines[0]["header"]), [ShapeResult(**row) for row in lines[1:]]


def read_shapes_file(path: Path) -> list[Shape]:
    lines = (line.split("#", 1)[0].strip() for line in path.read_text().splitlines())
    return [parse_shape(line) for line in lines if line]


def time_shape(shape: Shape, dtype: str, *, warmup_seconds: float, window_seconds: float, repeats: int) -> ShapeResult:
    key_a, key_b = jax.random.split(jax.random.key(0))
    a = jax.random.normal(key_a, (shape.m, shape.k), DTYPES[dtype])
    b = jax.random.normal(key_b, (shape.k, shape.n), DTYPES[dtype])
    matmul = jax.jit(jnp.matmul)
    matmul(a, b).block_until_ready()

    # Warm up for a fixed wall time so the GPU reaches its loaded clock before measuring.
    calls = 0
    start = time.perf_counter()
    while time.perf_counter() - start < warmup_seconds:
        out = matmul(a, b)
        calls += 1
    out.block_until_ready()
    seconds_per_call = (time.perf_counter() - start) / calls
    iterations = max(10, math.ceil(window_seconds / seconds_per_call))

    window_tflops = []
    window_bounds_ns = []
    for _ in range(repeats):
        start_ns = time.clock_gettime_ns(TRACE_CLOCK)
        for _ in range(iterations):
            out = matmul(a, b)
        out.block_until_ready()
        end_ns = time.clock_gettime_ns(TRACE_CLOCK)
        window_bounds_ns.append([start_ns, end_ns])
        window_tflops.append(shape.flops * iterations / ((end_ns - start_ns) / 1e9) / 1e12)

    return ShapeResult(
        shape=str(shape),
        dtype=dtype,
        iterations_per_window=iterations,
        window_tflops=window_tflops,
        median_tflops=statistics.median(window_tflops),
        max_tflops=max(window_tflops),
        window_bounds_ns=window_bounds_ns,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dtype", choices=sorted(DTYPES), required=True)
    for dim in ("m", "n", "k"):
        group = parser.add_mutually_exclusive_group()
        group.add_argument(f"--{dim}", type=int, nargs="+", help=f"{dim.upper()} values")
        group.add_argument(
            f"--{dim}-range",
            type=int,
            nargs=3,
            metavar=("START", "STOP", "STEP"),
            help=f"{dim.upper()} values from START to STOP inclusive",
        )
    parser.add_argument("--shapes-file", type=Path, help="file with one MxNxK shape per line; # starts a comment")
    parser.add_argument("--warmup-seconds", type=float, default=1.0)
    parser.add_argument("--window-seconds", type=float, default=0.2)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True, help="JSON lines file, one row per shape")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    shapes: list[Shape] = []
    dims = [dim_values(getattr(args, dim), getattr(args, f"{dim}_range")) for dim in ("m", "n", "k")]
    if any(dims):
        if not all(dims):
            parser.error("give M, N and K together, each with --<dim> or --<dim>-range")
        shapes.extend(Shape(m, n, k) for m, n, k in itertools.product(*dims))
    if args.shapes_file:
        shapes.extend(read_shapes_file(args.shapes_file))
    if not shapes:
        parser.error("give --m/--n/--k, --shapes-file, or both")
    shapes = list(dict.fromkeys(shapes))

    device = jax.devices()[0]
    header = RunHeader(
        hostname=socket.gethostname(),
        jax=jax.__version__,
        device_kind=device.device_kind,
        platform=device.platform,
        xla_flags=os.environ.get("XLA_FLAGS", ""),
        num_shapes=len(shapes),
    )
    logger.info("%s", header)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    best: ShapeResult | None = None
    with args.output.open("w") as out:
        out.write(header_line(header))
        for i, shape in enumerate(shapes):
            result = time_shape(
                shape,
                args.dtype,
                warmup_seconds=args.warmup_seconds,
                window_seconds=args.window_seconds,
                repeats=args.repeats,
            )
            out.write(shape_line(result))
            out.flush()
            if best is None or result.median_tflops > best.median_tflops:
                best = result
            logger.info(
                "[%d/%d] %s %s median %.1f max %.1f TFLOP/s (best so far %.1f @ %s)",
                i + 1,
                len(shapes),
                result.shape,
                result.dtype,
                result.median_tflops,
                result.max_tflops,
                best.median_tflops,
                best.shape,
            )

    assert best is not None
    logger.info("Best median: %.1f TFLOP/s @ %s (%s)", best.median_tflops, best.shape, best.dtype)


if __name__ == "__main__":
    main()
