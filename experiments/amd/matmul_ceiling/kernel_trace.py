# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Attach GPU kernel time from a rocprofv3 kernel trace to jax_matmul.py results.

jax_matmul.py's wall-clock rate includes host dispatch and the idle GPU time
between calls. MAMF-finder times each GEMM with GPU events around the op, so the
JAX number that compares with MAMF and MSMF is the time the GPU spends in
kernels. Each kernel whose start and end fall inside a timed window is assigned
to that window; warmup, compilation and autotuning fall outside every window.

Run from the checkout root after the traced run has exited:
python -m experiments.amd.matmul_ceiling.kernel_trace <results.jsonl> <rocprofv3 output dir>
"""

import argparse
import bisect
import collections
import csv
import dataclasses
import statistics
import sys
from pathlib import Path

from experiments.amd.matmul_ceiling.jax_matmul import ShapeResult, header_line, parse_shape, read_results, shape_line


@dataclasses.dataclass(frozen=True, slots=True)
class Kernel:
    name: str
    start_ns: int
    end_ns: int


def read_kernel_trace(trace_dir: Path) -> list[Kernel]:
    """Read the one ``*kernel_trace.csv`` rocprofv3 wrote under ``trace_dir``, sorted by start time."""
    traces = sorted(trace_dir.rglob("*kernel_trace.csv"))
    if len(traces) != 1:
        raise ValueError(f"expected one kernel trace under {trace_dir}, found {traces}")
    with traces[0].open(newline="") as f:
        kernels = [
            Kernel(sys.intern(row["Kernel_Name"]), int(row["Start_Timestamp"]), int(row["End_Timestamp"]))
            for row in csv.DictReader(f)
        ]
    kernels.sort(key=lambda kernel: kernel.start_ns)
    return kernels


def with_kernel_time(result: ShapeResult, kernels: list[Kernel], starts: list[int]) -> ShapeResult:
    """Return ``result`` with the rate of each timed window computed from kernel time alone.

    ``starts`` holds the start time of each kernel in ``kernels``, for bisection.
    """
    flops = parse_shape(result.shape).flops
    window_tflops = []
    names: collections.Counter[str] = collections.Counter()
    for start_ns, end_ns in result.window_bounds_ns:
        first, last = bisect.bisect_left(starts, start_ns), bisect.bisect_right(starts, end_ns)
        window = [kernel for kernel in kernels[first:last] if kernel.end_ns <= end_ns]
        if not window or len(window) % result.iterations_per_window:
            raise ValueError(
                f"{result.shape}: {len(window)} kernels in a window of {result.iterations_per_window} calls; "
                "the trace and the window bounds do not line up"
            )
        kernel_seconds = sum(kernel.end_ns - kernel.start_ns for kernel in window) / 1e9
        window_tflops.append(flops * result.iterations_per_window / kernel_seconds / 1e12)
        names.update(kernel.name for kernel in window)
    return dataclasses.replace(
        result,
        kernel_window_tflops=window_tflops,
        kernel_median_tflops=statistics.median(window_tflops),
        kernels=[name for name, _ in names.most_common()],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results", type=Path, help="jax_matmul.py output, rewritten in place")
    parser.add_argument("trace_dir", type=Path, help="rocprofv3 --output-directory of the same run")
    args = parser.parse_args()

    header, results = read_results(args.results)
    kernels = read_kernel_trace(args.trace_dir)
    starts = [kernel.start_ns for kernel in kernels]
    traced = [with_kernel_time(result, kernels, starts) for result in results]
    args.results.write_text(header_line(header) + "".join(shape_line(result) for result in traced))


if __name__ == "__main__":
    main()
