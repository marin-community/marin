# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tile sweep of the MXFP8 row-block (token-axis) quantizers at the GB200 S4 bridge's expert shapes: one chunk's
receiver buffer (301568 rows, 3 local experts) at the expert input width (3072, rows + token blocks) and the expert
width (4608, token blocks of bf16 rows and of MXFP8 rows)."""

import argparse
import time

import jax
import jax.numpy as jnp
import numpy as np

from levanter.grug._moe.quack_mxfp8 import token_scale_blocks
from levanter.grug.mxfp8 import mx_quantize_rows
from levanter.grug.mxfp8_pallas import _row_blocks_call


def _time(fn, iters: int) -> float:
    jax.block_until_ready(fn())
    jax.block_until_ready(fn())
    start = time.perf_counter()
    for _ in range(iters):
        out = fn()
    jax.block_until_ready(out)
    return (time.perf_counter() - start) / iters


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capacity", type=int, default=301_568)
    parser.add_argument("--rows", type=int, default=262_144)
    parser.add_argument("--experts", type=int, default=3)
    parser.add_argument("--iters", type=int, default=20)
    args = parser.parse_args()
    rng = np.random.default_rng(0)
    sizes = np.full(args.experts, args.rows // args.experts)
    cu = jnp.asarray(np.concatenate([[0], np.cumsum(sizes)]), jnp.int32)
    starts, counts = token_scale_blocks(cu, args.capacity)
    x3072 = jnp.asarray(rng.normal(size=(args.capacity, 3072)), jnp.bfloat16)
    x4608 = jnp.asarray(rng.normal(size=(args.capacity, 4608)), jnp.bfloat16)
    v4608, e4608 = jax.jit(mx_quantize_rows)(x4608)
    reference = None
    for cols, warps in ((128, 2), (128, 4), (256, 4), (256, 8), (512, 4), (512, 8)):
        cases = {
            "rows+tokens 3072": lambda: _row_blocks_call(
                x3072, None, starts, counts, with_rows=True, interpret=False, cols=cols, num_warps=warps
            ),
            "tokens 4608": lambda: _row_blocks_call(
                x4608, None, starts, counts, with_rows=False, interpret=False, cols=cols, num_warps=warps
            ),
            "requant 4608": lambda: _row_blocks_call(
                v4608, e4608, starts, counts, with_rows=False, interpret=False, cols=cols, num_warps=warps
            ),
        }
        outs = {name: jax.jit(fn)() for name, fn in cases.items()}
        if reference is None:
            reference = outs
        # Rows past the last expert's are unspecified; compare the routed rows and every exponent block.
        same = all(
            bool(jnp.all(a[: args.rows].view(jnp.uint8) == b[: args.rows].view(jnp.uint8)))
            for name in outs
            for a, b in zip(outs[name], reference[name], strict=True)
        )
        times = {name: 1e3 * _time(jax.jit(fn), args.iters) for name, fn in cases.items()}
        print(
            f"cols {cols} warps {warps}: "
            + " ".join(f"{name} {ms:.3f} ms" for name, ms in times.items())
            + f" total {sum(times.values()):.3f} bitwise_equal {same}"
        )


if __name__ == "__main__":
    main()
