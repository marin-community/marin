# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Upper bound of fusing the MXFP8 ReLU^2 expert MLP forward (up GEMM + ReLU^2 + MXFP8 quantize + down GEMM) into
one kernel, at the hero EP64 chunk shape.

Times, on pre-quantized operands (the fusion does not touch the row or weight quantizers):

- ``up_q``: today's up-projection, ``mxfp8_relu2_grouped_gemm`` (stores bf16 ``pre`` and MXFP8 ``post``);
- ``up_plain``: the same GEMM with a plain bf16 epilogue (``pre`` only), i.e. without the quantizing epilogue;
- ``up_q_nopre``: the ReLU^2 quantizing epilogue without the bf16 ``pre`` store (what a recompute-``pre`` design
  would pay);
- ``down``: the down projection on the quantized ``post``;
- ``fwd``: ``up_q`` then ``down`` in one program (today's forward);
- ``dense_peak``: a large dense MXFP8 GEMM, the achievable block-scaled MMA rate on this GPU.

A fused kernel cannot beat the forward FLOPs at ``dense_peak``, nor the sum of the two GEMMs' mainloops; the gap
between ``fwd`` and those bounds is everything fusion could save::

    python lib/levanter/scripts/bench/bench_mxfp8_fused_expert_fwd.py
"""

import argparse
import json
import time

import jax
import jax.numpy as jnp
import numpy as np

import cutlass
import cutlass.cute as cute
from cutlass import Float32
from quack.gemm_tvm_ffi_utils import make_scheduler_args, make_varlen_args

from levanter.cutlass_kernel_cache import cute_launcher_factory, cutlass_call
from levanter.grug._moe.quack_moe_cute import _max_active_clusters, _mint_epilogue_gemm
from levanter.grug._moe.quack_mxfp8 import (
    MXFP8_CLUSTER,
    _INDEX_SPEC,
    _K_MAJOR,
    _ROWS_SPEC,
    _SCALE_SPEC,
    _WEIGHT_SPECS,
    _dense_gemm,
    _gemm,
    _quantize_rows,
    _relu2_quant_epilogue,
    mx_weights,
    mxfp8_grouped_gemm,
    mxfp8_relu2_grouped_gemm,
    padded_scale_rows,
    varlen_output_scale_shape,
)
from levanter.grug.mxfp8_pallas import quantize_rows


@cute_launcher_factory
def _build_relu2_nopre_launcher(*, max_active_clusters, max_swizzle):
    gemm_type = _mint_epilogue_gemm(_relu2_quant_epilogue(), kind_sig=(), paired_acc=False)

    @cute.jit
    def launcher(stream, mA, mB, mSFA, mSFB, mCuSeqlens, mPostAct, mPostActSF):
        gemm = _gemm(gemm_type, cutlass.Float8E4M3FN)
        epi_args = gemm_type.EpilogueArguments(postact=mPostAct, postact_sf=mPostActSF, sfd_norm_const=Float32(1.0))
        scheduler_args = make_scheduler_args(max_active_clusters, max_swizzle, None)
        varlen_args = make_varlen_args(mCuSeqlens, None, None)
        gemm(mA, mB, None, None, epi_args, scheduler_args, varlen_args, stream, mSFA, mSFB)

    return launcher


def relu2_nopre_grouped_gemm(a, a_scales, b, b_scales, cu_seqlens, *, max_swizzle=8):
    rows, cols = a.shape[0], b.shape[2]
    launcher = _build_relu2_nopre_launcher(
        max_active_clusters=_max_active_clusters(MXFP8_CLUSTER), max_swizzle=max_swizzle
    )
    call = cutlass_call(
        launcher,
        output_shape_dtype=(
            jax.ShapeDtypeStruct((rows, cols), jnp.float8_e4m3fn),
            jax.ShapeDtypeStruct(varlen_output_scale_shape(rows, cols, cu_seqlens.shape[0] - 1), jnp.float8_e8m0fnu),
        ),
        input_spec=(_ROWS_SPEC, _WEIGHT_SPECS["n"], _SCALE_SPEC, _SCALE_SPEC, _INDEX_SPEC),
        output_spec=(_ROWS_SPEC, _SCALE_SPEC),
        use_static_tensors=False,
    )
    return call(a, b, a_scales, b_scales, cu_seqlens.astype(jnp.int32))


def _time(fn, args, iters: int, reps: int) -> float:
    """Median over ``reps`` of the mean time of ``iters`` back-to-back calls, in ms."""
    jitted = jax.jit(fn)
    jax.block_until_ready(jitted(*args))
    samples = []
    for _ in range(reps):
        jax.block_until_ready(jitted(*args))
        start = time.perf_counter()
        for _ in range(iters):
            out = jitted(*args)
        jax.block_until_ready(out)
        samples.append((time.perf_counter() - start) / iters * 1e3)
    return float(np.median(samples))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rows", type=int, default=262_144, help="routed rows (active)")
    parser.add_argument("--capacity", type=int, default=301_568, help="receiver buffer rows")
    parser.add_argument("--experts", type=int, default=3)
    parser.add_argument("--hidden", type=int, default=3_072, help="expert input width (latent)")
    parser.add_argument("--intermediate", type=int, default=4_608, help="expert width")
    parser.add_argument("--dense", type=int, default=16_384, help="square dense GEMM size for the peak rate")
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--json", type=str, default=None)
    args = parser.parse_args()

    rng = np.random.default_rng(0)
    h, i, e = args.hidden, args.intermediate, args.experts
    x = jnp.asarray(rng.normal(0, 1, (args.capacity, h)), jnp.bfloat16)
    w_up = jnp.asarray(rng.normal(0, h**-0.5, (e, h, i)), jnp.bfloat16)
    w_down = jnp.asarray(rng.normal(0, i**-0.5, (e, i, h)), jnp.bfloat16)
    sizes = np.full(e, args.rows // e)
    sizes[-1] += args.rows - sizes.sum()
    cu = jnp.asarray(np.concatenate([[0], np.cumsum(sizes)]), jnp.int32)

    values, exponents = jax.jit(quantize_rows)(x)
    x_scales = jax.jit(padded_scale_rows)(exponents, cu)
    wu, wu_s = jax.jit(mx_weights, static_argnums=1)(w_up, "n")
    wd, wd_s = jax.jit(mx_weights, static_argnums=1)(w_down, "n")
    _, post, post_s = jax.jit(lambda *a: mxfp8_relu2_grouped_gemm(*a, b_major="n"))(values, x_scales, wu, wu_s, cu)

    def up_q(v, s, w, ws, c):
        return mxfp8_relu2_grouped_gemm(v, s, w, ws, c, b_major="n")

    def up_plain(v, s, w, ws, c):
        return mxfp8_grouped_gemm(v, s, w, ws, c, b_major="n")

    def down(p, ps, w, ws, c):
        return mxfp8_grouped_gemm(p, ps, w, ws, c, b_major="n")

    def fwd(v, s, w1, w1s, w2, w2s, c):
        pre, p, ps = mxfp8_relu2_grouped_gemm(v, s, w1, w1s, c, b_major="n")
        return pre, mxfp8_grouped_gemm(p, ps, w2, w2s, c, b_major="n")

    gemm_flops = 2 * args.rows * h * i
    results: dict = {"shape": vars(args)}
    for name, fn, fargs, flops in [
        ("up_q", up_q, (values, x_scales, wu, wu_s, cu), gemm_flops),
        ("up_plain", up_plain, (values, x_scales, wu, wu_s, cu), gemm_flops),
        ("up_q_nopre", relu2_nopre_grouped_gemm, (values, x_scales, wu, wu_s, cu), gemm_flops),
        ("down", down, (post, post_s, wd, wd_s, cu), gemm_flops),
        ("fwd", fwd, (values, x_scales, wu, wu_s, wd, wd_s, cu), 2 * gemm_flops),
    ]:
        ms = _time(fn, fargs, args.iters, args.reps)
        results[name] = {"ms": ms, "tflops": flops / ms / 1e9}
        print(f"{name}: {json.dumps(results[name])}", flush=True)

    n = args.dense
    a = jnp.asarray(rng.normal(0, 1, (n, n)), jnp.bfloat16)
    b = jnp.asarray(rng.normal(0, 1, (n, n)), jnp.bfloat16)
    (qa, sa), (qb, sb) = jax.jit(_quantize_rows)(a), jax.jit(_quantize_rows)(b)

    def dense(qa, sa, qb, sb):
        return _dense_gemm(qa, sa, qb, sb, a_spec=_K_MAJOR, b_spec=_K_MAJOR, rows=n, cols=n)

    ms = _time(dense, (qa, sa, qb, sb), args.iters, args.reps)
    peak = 2 * n**3 / ms / 1e9
    results["dense_peak"] = {"ms": ms, "tflops": peak}
    print(f"dense_peak: {json.dumps(results['dense_peak'])}", flush=True)

    fwd_ms = results["fwd"]["ms"]
    bounds = {
        "fwd_at_dense_peak_ms": 2 * gemm_flops / peak / 1e9,
        "fwd_sum_ms": results["up_q"]["ms"] + results["down"]["ms"],
        "quant_epilogue_cost_ms": results["up_q"]["ms"] - results["up_plain"]["ms"],
        "pre_store_cost_ms": results["up_q"]["ms"] - results["up_q_nopre"]["ms"],
    }
    bounds["max_saving_vs_peak_ms"] = fwd_ms - bounds["fwd_at_dense_peak_ms"]
    results["bounds"] = bounds
    print(f"bounds: {json.dumps(bounds)}", flush=True)
    if args.json:
        with open(args.json, "w") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
