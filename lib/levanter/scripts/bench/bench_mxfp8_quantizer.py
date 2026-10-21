# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Time the MXFP8 dual-quantize producer against the XLA composition on one GPU.

Both paths turn the aligned receiver buffer ``x [Mp, H]`` the ragged backend lays out for the MXFP8
expert MLP into its four operands: ``dual_quantize_activation`` of ``x`` with its unreceived rows
zeroed. ``xla`` runs that composition, a zeroing pass and two quantization passes; ``triton`` runs
`dual_quantize_padded`, one kernel. The defaults are the hero's expert chunk: a capacity of 302,336
rows, H = 3072, three experts with skewed sizes, and non-finite rows in the gaps between experts and
past the last. The same width covers the down projection's cotangent, I = H = 3072.

Achieved bandwidth counts the minimum HBM traffic: one read of the active rows, both FP8 copies,
both scale arrays and the row map. Each timing reports the median over ``--repeats`` windows of
``--iters`` back-to-back calls. With ``--sweep``, the Triton kernel is also timed over a grid of
tile shapes. Rows are printed as JSON lines and optionally appended to ``--output``.

Example::

    python lib/levanter/scripts/bench/bench_mxfp8_quantizer.py --sweep --output quantizer.jsonl
"""

import argparse
import dataclasses
import itertools
import json
import statistics
import time
from collections.abc import Callable
from pathlib import Path

import jax
import jax.numpy as jnp

from levanter.grug._moe.ep_common import _aligned_group_sizes, _aligned_receiver_rows
from levanter.grug._moe.mxfp8.layout import GROUP_PAD, aligned_dispatch_layout, wgrad_block_perm
from levanter.grug._moe.mxfp8.quantize import dual_quantize_activation
from levanter.grug._moe.mxfp8.triton_quant import _DEFAULT_TILE, _quantize_padded, _TileConfig

SWEEP_GRID = {
    "block_m": (32, 64, 128, 256),
    "num_warps": (1, 2, 4, 8),
    "max_tiles_per_program": (1, 4, 8, 12),
}


@dataclasses.dataclass(frozen=True)
class BenchRow:
    path: str
    tile: str
    ms: float
    gbps: float
    speedup_vs_xla: float
    bitwise_equal: bool


def _time_ms(fn: Callable, args: tuple, *, iters: int, repeats: int) -> float:
    jax.block_until_ready(fn(*args))
    jax.block_until_ready(fn(*args))
    windows = []
    for _ in range(repeats):
        start = time.perf_counter()
        for _ in range(iters):
            out = fn(*args)
        jax.block_until_ready(out)
        windows.append((time.perf_counter() - start) / iters)
    return 1e3 * statistics.median(windows)


def _bitwise_equal(got, want) -> bool:
    return all(
        bool(jnp.array_equal(jax.lax.bitcast_convert_type(g, jnp.uint8), jax.lax.bitcast_convert_type(w, jnp.uint8)))
        for g, w in zip(got, want, strict=True)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--capacity", type=int, default=302_336)
    parser.add_argument("--hidden", type=int, default=3072)
    parser.add_argument("--active-sizes", type=int, nargs="+", default=[182_000, 0, 117_500])
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    active_rows = sum(args.active_sizes)
    if active_rows > args.capacity:
        raise ValueError(f"active sizes sum to {active_rows}, above the capacity {args.capacity}")
    padded_rows = _aligned_receiver_rows(args.capacity, len(args.active_sizes), GROUP_PAD)
    active = jnp.asarray(args.active_sizes, jnp.int32)
    spans = _aligned_group_sizes(active, GROUP_PAD)
    layout = aligned_dispatch_layout(spans.at[-1].add(padded_rows - jnp.sum(spans)), active, rows=padded_rows)
    block_perm = wgrad_block_perm(layout.padded_group_sizes, rows=args.hidden, total_tokens=padded_rows)
    x = jax.random.normal(jax.random.key(0), (padded_rows, args.hidden), jnp.float32).astype(jnp.bfloat16)
    x = jnp.where(layout.source_valid[:, None], x, jnp.nan).astype(jnp.bfloat16)

    hbm_bytes = active_rows * args.hidden * 2 + 2 * padded_rows * args.hidden * (1 + 1 / 32) + padded_rows * 4
    print(
        f"device={jax.devices()[0].device_kind} C={args.capacity} H={args.hidden} Mp={padded_rows} "
        f"active={args.active_sizes} min_hbm_bytes={hbm_bytes / 1e9:.3f} GB",
        flush=True,
    )

    xla = jax.jit(
        lambda x, layout, perm: dual_quantize_activation(jnp.where(layout.source_valid[:, None], x, 0), perm)
    )
    want = xla(x, layout, block_perm)
    xla_ms = _time_ms(xla, (x, layout, block_perm), iters=args.iters, repeats=args.repeats)
    rows = [BenchRow("xla", "-", xla_ms, hbm_bytes / xla_ms / 1e6, 1.0, True)]
    print(json.dumps(dataclasses.asdict(rows[0])), flush=True)

    tiles = [_DEFAULT_TILE]
    if args.sweep:
        tiles += [_TileConfig(*values) for values in itertools.product(*SWEEP_GRID.values())]
    seen = set()
    for tile in tiles:
        # Each warp owns whole 32-row groups.
        if tile in seen or tile.block_m % (32 * tile.num_warps) != 0:
            continue
        seen.add(tile)
        triton_fn = jax.jit(lambda x, layout, perm, tile=tile: _quantize_padded(x, layout, perm, tile))
        label = f"m{tile.block_m}_w{tile.num_warps}_t{tile.max_tiles_per_program}"
        try:
            equal = _bitwise_equal(triton_fn(x, layout, block_perm), want)
        except Exception as error:  # A sweep point can exceed the GPU's registers or shared memory.
            print(json.dumps({"path": "triton", "tile": label, "error": repr(error)[:300]}), flush=True)
            continue
        ms = _time_ms(triton_fn, (x, layout, block_perm), iters=args.iters, repeats=args.repeats)
        path = "triton" if tile == _DEFAULT_TILE else "triton_sweep"
        rows.append(BenchRow(path, label, ms, hbm_bytes / ms / 1e6, xla_ms / ms, equal))
        print(json.dumps(dataclasses.asdict(rows[-1])), flush=True)

    print(f"{'path':<14}{'tile':<16}{'ms':>9}{'GB/s':>9}{'vs xla':>9}  bitwise")
    for row in sorted(rows, key=lambda r: r.ms):
        print(
            f"{row.path:<14}{row.tile:<16}{row.ms:9.3f}{row.gbps:9.0f}{row.speedup_vs_xla:8.2f}x  {row.bitwise_equal}"
        )
    if args.output is not None:
        with args.output.open("a") as f:
            for row in rows:
                f.write(json.dumps(dataclasses.asdict(row)) + "\n")


if __name__ == "__main__":
    main()
