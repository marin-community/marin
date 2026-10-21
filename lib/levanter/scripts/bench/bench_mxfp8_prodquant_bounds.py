# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Feasibility bounds for quantizing the MXFP8 expert weight-gradient operands inside the GEMM (quantize-on-load).

The routed-expert backward runs standalone token-axis quantizers (``quantize_token_blocks`` and friends) whose only
consumer is the varlen-k MXFP8 weight-gradient GEMM. Quantizing in the GEMM's producer warps instead reads bf16 (or
row-quantized e4m3) tiles and requantizes every tile once per output tile that loads it, so it trades the standalone
pass for (a) more L2->SM bytes and (b) per-tile ALU work. This script measures, on one GPU at one hero EP64 chunk
(three local experts, 262144 routed rows in a 301568-row buffer, latent 3072, ReLU^2 width 4608):

- the pieces the fusion would replace: each standalone quantizer and each MXFP8 wgrad GEMM (several clusters);
- the bf16 wgrad GEMMs (a GEMM that reads bf16 operands);
- L2->SM and HBM read bandwidth (a streaming Pallas read over buffers of several sizes);
- the producer half alone: a Pallas kernel that loads every wgrad tile the way an on-load GEMM would (``reuse`` times
  per element) and quantizes it along the token axis, with no MMA.

    python lib/levanter/scripts/bench/bench_mxfp8_prodquant_bounds.py
"""

import argparse
import functools
import json
import statistics
import time

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plt

from levanter.grug._moe.quack_moe_cute import quack_grouped_wgrad
from levanter.grug._moe.quack_mxfp8 import (
    mxfp8_token_grouped_gemm,
    quantize_token_blocks,
    token_scale_blocks,
)
from levanter.grug.mxfp8 import MX_BLOCK
from levanter.grug.mxfp8_pallas import (
    quantize_row_blocks,
    quantize_rows,
    quantize_rows_and_row_blocks,
    requantize_row_blocks,
)

_E4M3_MAX = 448.0


def _time(fn, args, iters: int, reps: int = 5) -> float:
    jitted = jax.jit(fn)
    jax.block_until_ready(jitted(*args))
    samples = []
    for _ in range(reps):
        jax.block_until_ready(jitted(*args))
        start = time.perf_counter()
        for _ in range(iters):
            out = jitted(*args)
        jax.block_until_ready(out)
        samples.append((time.perf_counter() - start) / iters)
    return statistics.median(samples)


# ---------------------------------------------------------------------------------------------------------------
# Read bandwidth: each program streams [ROWS, COLS] bf16 blocks of a buffer (block b = pid + i * programs, mod the
# block count) and accumulates them elementwise, so the buffer size picks L2 or HBM residency.
_RB_ROWS = 16
_RB_COLS = 512


def _read_kernel(x_ref, o_ref, *, iters: int, blocks: int):
    pid = pl.program_id(0)
    programs = pl.num_programs(0)

    def body(i, acc):
        b = (pid + i * programs) % blocks
        return acc + x_ref[pl.ds(b * _RB_ROWS, _RB_ROWS), :].astype(jnp.float32)

    acc = jax.lax.fori_loop(0, iters, body, jnp.zeros((_RB_ROWS, _RB_COLS), jnp.float32))
    o_ref[...] = acc


@functools.partial(jax.jit, static_argnames=("programs", "iters", "num_warps"))
def _read_bandwidth_call(x, *, programs: int, iters: int, num_warps: int):
    blocks = x.shape[0] // _RB_ROWS
    return pl.pallas_call(
        functools.partial(_read_kernel, iters=iters, blocks=blocks),
        grid=(programs,),
        in_specs=[pl.no_block_spec],
        out_specs=pl.BlockSpec((_RB_ROWS, _RB_COLS), lambda i: (i, 0)),
        out_shape=jax.ShapeDtypeStruct((programs * _RB_ROWS, _RB_COLS), jnp.float32),
        compiler_params=plt.CompilerParams(num_warps=num_warps, num_stages=1),
        name="read_bandwidth",
    )(x)


# ---------------------------------------------------------------------------------------------------------------
# Producer half of an on-load wgrad GEMM, no MMA. Program (c, r) owns column tile c of a [T, C] operand and loops
# over all tokens in 32-token steps (one MXFP8 block), as the r-th of the ``reuse`` output tiles that load column tile c would. Each step
# loads a [32, COL_TILE] tile (bf16, or e4m3 + per-row exponents for the requantize variant), quantizes it along
# the tokens (amax, e8m0 exponent, e4m3 cast) and folds the e4m3 bytes into a running max so the
# work is not dead. ``quantize=False`` keeps the loads and the fold, dropping the quantization.
_K_TILE = MX_BLOCK
_COL_TILE = 128


def _token_exponent(amax):
    bits = jax.lax.bitcast_convert_type(amax / _E4M3_MAX, jnp.int32)
    exponent = ((bits >> 23) & 0xFF) + ((bits & 0x7FFFFF) != 0).astype(jnp.int32)
    return jnp.where(amax > 0, jnp.clip(exponent, 1, 254), 127)


def _producer_kernel(*refs, steps: int, quantize: bool, requantize: bool):
    if requantize:
        x_ref, e_ref, o_ref = refs
    else:
        x_ref, o_ref = refs
    c = pl.program_id(0)
    cols = pl.ds(c * _COL_TILE, _COL_TILE)

    def body(k, acc):
        tok = pl.ds(k * _K_TILE, _K_TILE)
        tile = x_ref[tok, cols].astype(jnp.float32)  # [K_TILE, COL_TILE]
        if requantize:
            sub = _COL_TILE // MX_BLOCK
            exps = e_ref[tok, pl.ds(c * sub, sub)].astype(jnp.int32)  # [K_TILE, COL_TILE / 32]
            scale = jax.lax.bitcast_convert_type(exps << 23, jnp.float32)
            tile = tile * jnp.broadcast_to(scale[:, :, None], (_K_TILE, sub, MX_BLOCK)).reshape(tile.shape)
        if not quantize:
            return jnp.maximum(acc, tile.astype(jnp.float8_e4m3fn).astype(jnp.float32))
        exponent = _token_exponent(jnp.max(jnp.abs(tile), axis=0))  # [COL_TILE]
        scale = jax.lax.bitcast_convert_type((254 - exponent) << 23, jnp.float32)
        values = (tile * scale[None, :]).astype(jnp.float8_e4m3fn)
        return jnp.maximum(acc, values.astype(jnp.float32))

    acc = jax.lax.fori_loop(0, steps, body, jnp.zeros((_K_TILE, _COL_TILE), jnp.float32))
    o_ref[...] = jnp.max(acc, axis=0, keepdims=True)


@functools.partial(jax.jit, static_argnames=("reuse", "quantize", "num_warps"))
def _producer_call(x, exponents, *, reuse: int, quantize: bool, num_warps: int):
    t, cols = x.shape
    steps = t // _K_TILE
    sources = (x,) if exponents is None else (x, exponents)
    return pl.pallas_call(
        functools.partial(_producer_kernel, steps=steps, quantize=quantize, requantize=exponents is not None),
        grid=(cols // _COL_TILE, reuse),
        in_specs=[pl.no_block_spec] * len(sources),
        out_specs=pl.BlockSpec((1, _COL_TILE), lambda c, r: (r, c)),
        out_shape=jax.ShapeDtypeStruct((reuse, cols), jnp.float32),
        compiler_params=plt.CompilerParams(num_warps=num_warps, num_stages=1),
        name="producer_quant",
    )(*sources)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rows", type=int, default=262_144)
    parser.add_argument("--capacity", type=int, default=301_568)
    parser.add_argument("--experts", type=int, default=3)
    parser.add_argument("--hidden", type=int, default=3_072)
    parser.add_argument("--intermediate", type=int, default=4_608)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--skip", type=str, default="", help="comma list of sections to skip: base,bw,prod")
    parser.add_argument("--json", type=str, default=None)
    args = parser.parse_args()
    skip = set(filter(None, args.skip.split(",")))
    print(jax.devices()[0].device_kind, flush=True)

    h, i, e = args.hidden, args.intermediate, args.experts
    key = jax.random.PRNGKey(0)
    kx, kd, kp, kdy = jax.random.split(key, 4)
    x = jax.random.normal(kx, (args.capacity, h), jnp.bfloat16)
    dy = jax.random.normal(kdy, (args.capacity, h), jnp.bfloat16)
    d_pre = jax.random.normal(kd, (args.capacity, i), jnp.bfloat16)
    post = jnp.square(jax.nn.relu(jax.random.normal(kp, (args.capacity, i), jnp.bfloat16)))
    sizes = np.full(e, args.rows // e)
    sizes[-1] += args.rows - sizes.sum()
    cu = jnp.asarray(np.concatenate([[0], np.cumsum(sizes)]), jnp.int32)
    blocks = token_scale_blocks(cu, args.capacity)
    d_pre_rows = jax.block_until_ready(jax.jit(quantize_rows)(d_pre))
    results: dict = {"shape": vars(args)}

    def report(name, seconds, flops=None, bytes_moved=None):
        line = f"{name:72s} {seconds * 1e3:8.3f} ms"
        if flops:
            line += f"  {flops / seconds / 1e15:5.2f} PFLOP/s"
        if bytes_moved:
            line += f"  {bytes_moved / seconds / 1e12:6.2f} TB/s"
        print(line, flush=True)
        results[name] = seconds * 1e3

    if "base" not in skip:
        tokens = jax.block_until_ready(
            jax.jit(lambda a, b, c, d: [quantize_token_blocks(t, cu) for t in (a, b, c, d)])(x, d_pre, post, dy)
        )
        qx, qd, qp, qdy = tokens
        t_bytes = lambda width, src: args.rows * width * (src + 1)  # noqa: E731
        report("quant x rows only (fwd primal need)", _time(quantize_rows, (x,), args.iters), None, t_bytes(h, 3))
        report(
            "quant x rows+tokens (fwd/recompute, also dy in bwd)",
            _time(lambda a: quantize_rows_and_row_blocks(a, *blocks), (x,), args.iters),
            None,
            t_bytes(h, 4),
        )
        report(
            "quant x tokens only",
            _time(lambda a: quantize_row_blocks(a, *blocks), (x,), args.iters),
            None,
            t_bytes(h, 2),
        )
        report(
            "quant post tokens (bf16 [T, I])",
            _time(lambda a: quantize_row_blocks(a, *blocks), (post,), args.iters),
            None,
            t_bytes(i, 2),
        )
        report(
            "requant d_pre rows -> tokens (e4m3 [T, I])",
            _time(lambda v, s: requantize_row_blocks(v, s, *blocks), d_pre_rows, args.iters),
            None,
            t_bytes(i, 1),
        )
        flops_up = 2 * args.rows * h * i
        for cluster in [(2, 1, 1), (2, 2, 1), (4, 1, 1), (4, 2, 1), (2, 4, 1)]:
            kw = dict(out_dtype=jnp.bfloat16, tile_mn=(256, 256), cluster_mnk=cluster)
            try:
                report(
                    f"mxfp8 wgrad dw_up = x^T d_pre  cluster {cluster}",
                    _time(lambda a, b: mxfp8_token_grouped_gemm(a, b, cu, **kw), (qx, qd), args.iters),
                    flops_up,
                )
                report(
                    f"mxfp8 wgrad dw_down = post^T dy cluster {cluster}",
                    _time(lambda a, b: mxfp8_token_grouped_gemm(a, b, cu, **kw), (qp, qdy), args.iters),
                    flops_up,
                )
            except Exception as exc:  # report and continue: some clusters may not be supported
                print(f"cluster {cluster} failed: {type(exc).__name__}: {str(exc)[:200]}", flush=True)
        for cluster in [(2, 1, 1), (2, 2, 1)]:
            kw = dict(tile_mn=(256, 256), cluster_mnk=cluster, use_clc_persistence=False)
            report(
                f"bf16 wgrad dw_up cluster {cluster}",
                _time(lambda a, b: quack_grouped_wgrad(a, b, cu, **kw), (x, d_pre), args.iters),
                flops_up,
            )
            report(
                f"bf16 wgrad dw_down cluster {cluster}",
                _time(lambda a, b: quack_grouped_wgrad(a, b, cu, **kw), (post, dy), args.iters),
                flops_up,
            )

    if "bw" not in skip:
        sms = jax.devices()[0].core_count if hasattr(jax.devices()[0], "core_count") else 148
        for mbytes in (8, 24, 48, 96, 4096):
            rows = mbytes * 2**20 // (2 * _RB_COLS)
            buf = jnp.ones((rows, _RB_COLS), jnp.bfloat16)
            for per_sm, warps in ((4, 4), (8, 4), (16, 4), (8, 8)):
                programs = sms * per_sm
                iters = max(1, int(4 * 2**30 // (programs * _RB_ROWS * _RB_COLS * 2)))
                moved = programs * iters * _RB_ROWS * _RB_COLS * 2
                fn = functools.partial(_read_bandwidth_call, programs=programs, iters=iters, num_warps=warps)
                report(
                    f"read {mbytes:5d} MB buffer, {per_sm:2d} programs/SM x {warps} warps",
                    _time(fn, (buf,), 3),
                    None,
                    moved,
                )

    if "prod" not in skip:
        # Per-element reuse of an on-load GEMM = output tiles along the other operand's width (256-wide tiles),
        # divided by the cluster sharing of the quantized tile. Run the no-sharing reuse; time scales linearly.
        for name, src, exps, reuse in (
            ("x (A of dw_up, bf16 [T, 3072])", x, None, i // 256),
            ("d_pre (B of dw_up, e4m3 rows [T, 4608])", d_pre_rows[0], d_pre_rows[1], h // 256),
            ("post (A of dw_down, bf16 [T, 4608])", post, None, h // 256),
        ):
            width = src.shape[1]
            elements = args.capacity * width * reuse
            for warps in (4, 8):
                for quantize in (False, True):
                    fn = functools.partial(_producer_call, reuse=reuse, quantize=quantize, num_warps=warps)
                    seconds = _time(fn, (src, exps), max(1, args.iters // 4), reps=3)
                    tag = "load+quant" if quantize else "load only"
                    report(
                        f"producer {tag:10s} {name} reuse {reuse} warps {warps}",
                        seconds,
                        None,
                        elements * src.dtype.itemsize,
                    )
                    print(f"    {elements / seconds / 1e12:6.2f} T elements/s", flush=True)

    if args.json:
        with open(args.json, "w") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
