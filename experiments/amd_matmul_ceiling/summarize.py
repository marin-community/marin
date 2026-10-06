# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Summarize jax_matmul.py result files: the fastest shapes and the Snowball shapes."""

import argparse
import json
from pathlib import Path

SNOWBALL_SHAPES = Path(__file__).with_name("snowball_shapes.txt")


def read_rows(path: Path) -> tuple[dict, list[dict]]:
    lines = [json.loads(line) for line in path.read_text().splitlines()]
    return lines[0]["header"], lines[1:]


def snowball_shape_names() -> dict[str, str]:
    names = {}
    for line in SNOWBALL_SHAPES.read_text().splitlines():
        shape, _, comment = line.partition("#")
        if shape.strip():
            names.setdefault(shape.strip(), comment.strip() or "weight gradient")
    return names


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path, nargs="+")
    parser.add_argument("--top", type=int, default=5)
    args = parser.parse_args()

    snowball = snowball_shape_names()
    for path in args.results:
        header, rows = read_rows(path)
        dtype = rows[0]["dtype"]
        print(f"== {path.name}: {header['device_kind']} {dtype} XLA_FLAGS={header['xla_flags'].strip()!r}")
        ranked = sorted(rows, key=lambda r: r["median_tflops"], reverse=True)
        print(f"  {len(rows)} shapes; top {args.top} by median TFLOP/s:")
        for r in ranked[: args.top]:
            print(f"    {r['shape']:>18}  median {r['median_tflops']:7.1f}  max {r['max_tflops']:7.1f}")
        print("  Snowball shapes:")
        for r in rows:
            if r["shape"] in snowball:
                print(f"    {r['shape']:>18}  median {r['median_tflops']:7.1f}  {snowball[r['shape']]}")


if __name__ == "__main__":
    main()
