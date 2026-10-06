# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Summarize jax_matmul.py result files: the fastest shapes and the Snowball shapes.

Run from the checkout root: python -m experiments.amd_matmul_ceiling.summarize <results.jsonl> ...
"""

import argparse
import json
from pathlib import Path

from experiments.amd_matmul_ceiling.jax_matmul import RunHeader, ShapeResult

SNOWBALL_SHAPES = Path(__file__).with_name("snowball_shapes.txt")


def read_results(path: Path) -> tuple[RunHeader, list[ShapeResult]]:
    lines = [json.loads(line) for line in path.read_text().splitlines()]
    return RunHeader(**lines[0]["header"]), [ShapeResult(**row) for row in lines[1:]]


def snowball_shape_names() -> dict[str, str]:
    names = {}
    for line in SNOWBALL_SHAPES.read_text().splitlines():
        shape, _, comment = line.partition("#")
        if shape.strip():
            names.setdefault(shape.strip(), comment.strip() or "weight gradient")
    return names


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results", type=Path, nargs="+")
    parser.add_argument("--top", type=int, default=5)
    args = parser.parse_args()

    snowball = snowball_shape_names()
    for path in args.results:
        header, results = read_results(path)
        print(f"== {path.name}: {header.device_kind} {results[0].dtype} XLA_FLAGS={header.xla_flags.strip()!r}")
        ranked = sorted(results, key=lambda r: r.median_tflops, reverse=True)
        print(f"  {len(results)} shapes; top {args.top} by median TFLOP/s:")
        for r in ranked[: args.top]:
            print(f"    {r.shape:>18}  median {r.median_tflops:7.1f}  max {r.max_tflops:7.1f}")
        print("  Snowball shapes:")
        for r in results:
            if r.shape in snowball:
                print(f"    {r.shape:>18}  median {r.median_tflops:7.1f}  {snowball[r.shape]}")


if __name__ == "__main__":
    main()
