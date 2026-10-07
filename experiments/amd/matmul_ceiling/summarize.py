# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Summarize jax_matmul.py result files: the fastest shapes and the Snowball shapes.

Run from the checkout root: python -m experiments.amd.matmul_ceiling.summarize <results.jsonl> ...
"""

import argparse
from pathlib import Path

from experiments.amd.matmul_ceiling.jax_matmul import ShapeResult, read_results

SNOWBALL_SHAPES = Path(__file__).with_name("snowball_shapes.txt")
KERNEL_NAME_CHARS = 72


def snowball_shape_labels() -> dict[str, str]:
    """Map each Snowball shape to its labels, joining the labels of a shape listed on several lines."""
    labels: dict[str, list[str]] = {}
    for line in SNOWBALL_SHAPES.read_text().splitlines():
        shape, _, comment = line.partition("#")
        shape, comment = shape.strip(), comment.strip()
        if not shape:
            continue
        if not comment:
            raise ValueError(f"{SNOWBALL_SHAPES.name}: shape {shape} has no label")
        labels.setdefault(shape, []).append(comment)
    return {shape: ", ".join(parts) for shape, parts in labels.items()}


def rate_columns(result: ShapeResult) -> str:
    kernel = "      -" if result.kernel_median_tflops is None else f"{result.kernel_median_tflops:7.1f}"
    return f"wall {result.median_tflops:7.1f}  kernel {kernel}"


def kernel_label(result: ShapeResult) -> str:
    if not result.kernels:
        return ""
    more = f" (+{len(result.kernels) - 1} more)" if len(result.kernels) > 1 else ""
    return f"{result.kernels[0][:KERNEL_NAME_CHARS]}{more}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results", type=Path, nargs="+")
    parser.add_argument("--top", type=int, default=5)
    args = parser.parse_args()

    snowball = snowball_shape_labels()
    for path in args.results:
        header, results = read_results(path)
        xla_flags = header.xla_flags.strip()
        print(f"== {path.name}: {header.hostname} {header.device_kind} {results[0].dtype} XLA_FLAGS={xla_flags!r}")
        traced = all(r.kernel_median_tflops is not None for r in results)
        if traced:
            ranked = sorted(results, key=lambda r: r.kernel_median_tflops or 0.0, reverse=True)
        else:
            ranked = sorted(results, key=lambda r: r.median_tflops, reverse=True)
        print(f"  {len(results)} shapes; top {args.top} by median {'kernel' if traced else 'wall-clock'} TFLOP/s:")
        for r in ranked[: args.top]:
            print(f"    {r.shape:>18}  {rate_columns(r)}  {kernel_label(r)}")
        print("  Snowball shapes:")
        for r in results:
            if r.shape in snowball:
                print(f"    {r.shape:>18}  {rate_columns(r)}  {snowball[r.shape]}  [{kernel_label(r)}]")


if __name__ == "__main__":
    main()
