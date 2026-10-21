# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Sweep tile shape, cluster shape and persistence of every QuACK block-scaled GEMM of the GB200 d6144 recipe.

Each case is one GEMM of the step at its per-GPU shape (EP64 routed chunk: 3 local experts, 262144 routed rows in a
301568-row receiver buffer, latent 3072, expert width 4608; dense projections over 65536 tokens). Every schedule
runs on the same pre-quantized operands and is checked against the default schedule's output; a schedule that fails
to compile or launch is reported and skipped. One JSON line per (case, schedule)::

    python lib/levanter/scripts/bench/bench_mxfp8_gemm_schedules.py --cases up_relu2,drelu2 --out /tmp/sweep.jsonl
"""

import argparse
import dataclasses
import json
import time
import traceback

import jax
import jax.numpy as jnp
import numpy as np

from levanter.grug._moe.quack_mxfp8 import (
    _K_MAJOR,
    _MN_MAJOR,
    GemmSchedule,
    _dense_gemm,
    _quantize_cols,
    _quantize_rows,
    blocked_scales,
    mx_weights,
    mxfp8_drelu2_grouped_gemm,
    mxfp8_grouped_gemm,
    mxfp8_relu2_grouped_gemm,
    mxfp8_token_grouped_gemm,
    padded_scale_rows,
    quantize_token_blocks,
)
from levanter.grug.mxfp8 import mx_quantize_cols, mx_quantize_rows
from levanter.grug.mxfp8_pallas import quantize_rows

CAPACITY = 301_568
ROUTED = 262_144
EXPERTS = 3
LATENT = 3_072
EXPERT_WIDTH = 4_608
TOKENS = 65_536
HIDDEN = 6_144
# (contraction, output) of the dense projections: KDA/attention, latent down/up, the two fused shared experts, MLA.
DENSE_FEATURES = ((6144, 6144), (6144, 3072), (3072, 6144), (6144, 9216), (9216, 6144), (6144, 512), (512, 6144))

SCHEDULES = (
    GemmSchedule((256, 256), (2, 1, 1)),
    GemmSchedule((256, 192), (2, 1, 1)),
    GemmSchedule((256, 128), (2, 1, 1)),
    GemmSchedule((256, 256), (2, 2, 1)),
    GemmSchedule((256, 192), (2, 2, 1)),
    GemmSchedule((256, 128), (2, 2, 1)),
    GemmSchedule((256, 256), (4, 1, 1)),
    GemmSchedule((256, 192), (4, 1, 1)),
    GemmSchedule((256, 128), (4, 1, 1)),
    GemmSchedule((256, 128), (2, 4, 1)),
    GemmSchedule((256, 256), (4, 2, 1)),
    GemmSchedule((128, 256), (1, 1, 1)),
    GemmSchedule((128, 192), (1, 1, 1)),
    GemmSchedule((128, 128), (1, 1, 1)),
    GemmSchedule((128, 256), (1, 2, 1)),
    GemmSchedule((128, 256), (2, 1, 1)),
)
RESERVES = (8, 16, 24, 32)


def _cu() -> jax.Array:
    sizes = np.full(EXPERTS, ROUTED // EXPERTS)
    sizes[-1] += ROUTED - sizes.sum()
    return jnp.asarray(np.concatenate([[0], np.cumsum(sizes)]), jnp.int32)


def _normal(key, shape, scale=1.0):
    return (scale * jax.random.normal(key, shape, jnp.float32)).astype(jnp.bfloat16)


def _rows_operand(key, rows, width, cu):
    values, exponents = quantize_rows(_normal(key, (rows, width)))
    return values, padded_scale_rows(exponents, cu)


def _routed_cases(cu):
    """(name, flops, fn(schedule) -> outputs) of the routed-expert GEMMs."""
    keys = jax.random.split(jax.random.key(0), 8)
    x = _rows_operand(keys[0], CAPACITY, LATENT, cu)
    post = _rows_operand(keys[1], CAPACITY, EXPERT_WIDTH, cu)
    dy = _rows_operand(keys[2], CAPACITY, LATENT, cu)
    pre = _normal(keys[3], (CAPACITY, EXPERT_WIDTH))
    w_up = _normal(keys[4], (EXPERTS, LATENT, EXPERT_WIDTH), LATENT**-0.5)
    w_down = _normal(keys[5], (EXPERTS, EXPERT_WIDTH, LATENT), EXPERT_WIDTH**-0.5)
    w_up_n, w_up_k = mx_weights(w_up, "n"), mx_weights(w_up, "k")
    w_down_n, w_down_k = mx_weights(w_down, "n"), mx_weights(w_down, "k")
    tokens = {
        name: quantize_token_blocks(_normal(k, (CAPACITY, width)), cu)
        for name, k, width in (
            ("x", keys[6], LATENT),
            ("dpre", keys[7], EXPERT_WIDTH),
            ("post", keys[1], EXPERT_WIDTH),
            ("dy", keys[2], LATENT),
        )
    }
    flops = 2 * ROUTED * LATENT * EXPERT_WIDTH
    return [
        (
            "up_relu2",
            flops,
            lambda s, a, sa, b, sb, cu: mxfp8_relu2_grouped_gemm(a, sa, b, sb, cu, b_major="n", schedule=s),
            (*x, *w_up_n, cu),
        ),
        (
            "grouped_n",
            flops,
            lambda s, a, sa, b, sb, cu: mxfp8_grouped_gemm(a, sa, b, sb, cu, b_major="n", schedule=s),
            (*post, *w_down_n, cu),
        ),
        (
            "drelu2",
            flops,
            lambda s, a, sa, b, sb, pre, cu: mxfp8_drelu2_grouped_gemm(a, sa, b, sb, pre, cu, schedule=s),
            (*dy, *w_down_k, pre, cu),
        ),
        (
            "grouped_k",
            flops,
            lambda s, a, sa, b, sb, cu: mxfp8_grouped_gemm(a, sa, b, sb, cu, b_major="k", schedule=s),
            (*post, *w_up_k, cu),
        ),
        (
            "wgrad",
            flops,
            lambda s, a, sa, b, sb, cu: mxfp8_token_grouped_gemm((a, sa), (b, sb), cu, schedule=s),
            (*tokens["x"], *tokens["dpre"], cu),
        ),
        (
            "wgrad_down",
            flops,
            lambda s, a, sa, b, sb, cu: mxfp8_token_grouped_gemm((a, sa), (b, sb), cu, schedule=s),
            (*tokens["post"], *tokens["dy"], cu),
        ),
    ]


def _dense_cases():
    cases = []
    for k_dim, n_dim in DENSE_FEATURES:
        keys = jax.random.split(jax.random.key(k_dim * 7 + n_dim), 4)
        x = _normal(keys[0], (TOKENS, k_dim))
        g = _normal(keys[1], (TOKENS, n_dim))
        w = _normal(keys[2], (k_dim, n_dim), k_dim**-0.5)
        qx, sx = _quantize_rows(x)
        wk_values, wk_exponents = mx_quantize_cols(w)
        wn_values, wn_exponents = mx_quantize_rows(w)
        qg, sg = _quantize_rows(g)
        qx_cols, sx_cols = _quantize_cols(x)
        qg_cols, sg_cols = _quantize_cols(g)
        flops = 2 * TOKENS * k_dim * n_dim
        sw_k = blocked_scales(wk_exponents.T[None])
        sw_n = blocked_scales(wn_exponents[None])

        def fwd(s, qx, sx, wv, sw):
            return _dense_gemm(
                qx, sx, wv, sw, a_spec=_K_MAJOR, b_spec=_MN_MAJOR, rows=TOKENS, cols=wv.shape[1], schedule=s
            )

        def dgrad(s, qg, sg, wv, sw):
            return _dense_gemm(
                qg, sg, wv, sw, a_spec=_K_MAJOR, b_spec=_K_MAJOR, rows=TOKENS, cols=wv.shape[0], schedule=s
            )

        def wgrad(s, qx, sx, qg, sg):
            return _dense_gemm(
                qx, sx, qg, sg, a_spec=_MN_MAJOR, b_spec=_MN_MAJOR, rows=qx.shape[1], cols=qg.shape[1], schedule=s
            )

        cases += [
            (f"dense_fwd_{k_dim}x{n_dim}", flops, fwd, (qx, sx, wk_values, sw_k)),
            (f"dense_dgrad_{n_dim}x{k_dim}", flops, dgrad, (qg, sg, wn_values, sw_n)),
            (f"dense_wgrad_{k_dim}x{n_dim}", flops, wgrad, (qx_cols, sx_cols, qg_cols, sg_cols)),
        ]
    return cases


def _time(fn, operands, iters: int, repeats: int = 3) -> float:
    jax.block_until_ready(fn(*operands))
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        for _ in range(iters):
            out = fn(*operands)
        jax.block_until_ready(out)
        times.append((time.perf_counter() - start) / iters)
    return float(np.median(times))


def _rel_err(a, b) -> float | None:
    """Relative L2 distance over the rows every schedule defines (routed rows past ``cu[-1]`` are unspecified, and
    blocked scale buffers carry padding tiles, so those are skipped)."""
    if a.ndim == 6:
        return None
    if a.shape[0] == CAPACITY:
        a, b = a[:ROUTED], b[:ROUTED]
    a = jnp.asarray(a).astype(jnp.float32)
    b = jnp.asarray(b).astype(jnp.float32)
    return float(jnp.linalg.norm(a - b) / jnp.maximum(jnp.linalg.norm(b), 1e-30))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cases", default="", help="comma-separated case-name prefixes (default: all)")
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--clc", action="store_true", help="also run every schedule with CLC persistence")
    parser.add_argument("--reserves", action="store_true", help="also run the default schedule with SM reserves")
    parser.add_argument("--schedules", default="", help="'tm,tn,cm,cn;...' instead of the built-in list")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    cu = _cu()
    cases = _routed_cases(cu) + _dense_cases()
    prefixes = [p for p in args.cases.split(",") if p]
    if prefixes:
        cases = [c for c in cases if any(c[0].startswith(p) for p in prefixes)]
    schedules = list(SCHEDULES)
    if args.schedules:
        schedules = [
            GemmSchedule((tm, tn), (cm, cn, 1))
            for tm, tn, cm, cn in (map(int, s.split(",")) for s in args.schedules.split(";"))
        ]
    if args.clc:
        schedules += [dataclasses.replace(s, clc=True) for s in schedules]
    if args.reserves:
        schedules += [GemmSchedule(sm_reserve=r) for r in RESERVES]
    out = open(args.out, "a") if args.out else None
    for name, flops, fn, operands in cases:
        reference = jax.tree.leaves(jax.block_until_ready(jax.jit(lambda *a: fn(GemmSchedule(), *a))(*operands)))
        for schedule in schedules:
            record = {"case": name, **dataclasses.asdict(schedule)}
            try:
                jitted = jax.jit(lambda *a, schedule=schedule: fn(schedule, *a))
                outputs = jax.tree.leaves(jax.block_until_ready(jitted(*operands)))
                record["rel_err"] = [_rel_err(o, r) for o, r in zip(outputs, reference) if o.shape == r.shape]
                seconds = _time(jitted, operands, args.iters)
                record |= {"ms": seconds * 1e3, "tflops": flops / seconds / 1e12}
            except Exception as e:  # a schedule the kernel rejects is a sweep result, not a failure
                record["error"] = f"{type(e).__name__}: {str(e)[:300]}"
                traceback.print_exc(limit=2)
            line = json.dumps(record)
            print("SWEEP " + line, flush=True)
            if out:
                out.write(line + "\n")
                out.flush()


if __name__ == "__main__":
    main()
