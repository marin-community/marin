# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""MXFP8-001: dense GEMM microbench on B200 — mxfp8 vs bf16 vs per-tensor fp8.

Benchmarks the grug row-13 dense GEMM shapes (attention q/k/v/o and the
shared-expert MLP) in three arms:

  bf16        jax.lax.dot_general in bf16 (baseline)
  fp8_tensor  haliax.quantization.Fp8DotGeneralOp (per-tensor delayed scaling,
              the Hopper recipe; lowers to cuBLASLt f8 on both archs)
  mxfp8       jax.nn.scaled_dot_general with the default mxfp8 configs
              (e4m3 fwd/bwd, e8m0 block-32 scales; cuDNN block-scaled path)

Reports fwd and fwd+bwd wall time (median over --iters after --warmup),
TF/s, speedup vs bf16, and rel-Frobenius error of output/grads vs an f32
reference. Asserts the mxfp8 arm actually lowered to the block-scaled custom
call — `configs=None` or an unsupported backend silently falls back to plain
dot_general, which would otherwise masquerade as a null result.

Single GPU. Run on a B200 (sm100); the mxfp8 arm is expected to fail at
compile time on Hopper.

Usage: python bench_mxfp8_dense.py --git-sha REV [--tokens 65536] [--iters 50]
"""

import argparse
import json
import os
import re
import statistics
import time

import jax
import jax.numpy as jnp
from haliax.quantization import Fp8DotGeneralOp
from jax._src.cudnn.scaled_matmul_stablehlo import quantize
from jax.nn import get_scaled_dot_general_config, scaled_dot_general, scaled_matmul

# (label, K, N, production weight). The production mix has five square
# projections (Q, O, shared gate/up/down) and two GQA K/V projections.
SHAPES = [
    ("q_o_shared_5120x5120", 5120, 5120, 5),
    ("kv_5120x1280", 5120, 1280, 2),
    ("qkvo_2560x2560", 2560, 2560, 0),
    ("shared_up_2560x1280", 2560, 1280, 0),
    ("shared_down_1280x2560", 1280, 2560, 0),
]
DNUMS = (((1,), (0,)), ((), ()))  # (T,K) x (K,N) -> (T,N)
MXFP8_CONFIGS = [get_scaled_dot_general_config("mxfp8") for _ in range(3)]


def dot_bf16(x, w):
    return jax.lax.dot_general(x, w, DNUMS, preferred_element_type=jnp.bfloat16)


def dot_fp8_tensor(x, w):
    # Fresh op state (scale=1) each trace: fine for perf, not a delayed-scaling
    # numerics test. State leaves are closure constants under jit.
    return Fp8DotGeneralOp.init()(x, w, DNUMS)


def dot_mxfp8(x, w):
    return scaled_dot_general(x, w, DNUMS, preferred_element_type=jnp.bfloat16, configs=MXFP8_CONFIGS)


ARMS = {"bf16": dot_bf16, "fp8_tensor": dot_fp8_tensor, "mxfp8": dot_mxfp8}


def fwd_bwd(dot):
    def loss(x, w):
        return jnp.sum(dot(x, w).astype(jnp.float32) ** 2)

    return jax.grad(loss, argnums=(0, 1))


def rel_frob(a, ref):
    a = jnp.asarray(a, jnp.float32)
    return float(jnp.linalg.norm(a - ref) / jnp.linalg.norm(ref))


def sample_statistics(samples):
    median = statistics.median(samples)
    mad = statistics.median(abs(sample - median) for sample in samples)
    return median, mad


def weighted_production_ratio(measurements):
    weighted_oracle = sum(weight * oracle for weight, oracle, _ in measurements)
    weighted_per_tensor = sum(weight * per_tensor for weight, _, per_tensor in measurements)
    assert weighted_per_tensor > 0
    return weighted_oracle / weighted_per_tensor


def custom_call_count(compiled_text, target):
    return compiled_text.count(f'custom_call_target="{target}"')


def linear_orientations(q_row, s_row_t, q_col, s_col):
    row_scale = jax.lax.bitcast_convert_type(s_row_t.T, jnp.float8_e8m0fnu)
    col_scale = jax.lax.bitcast_convert_type(s_col.T, jnp.float8_e8m0fnu)
    return q_row[None], row_scale[None], q_col.T[None], col_scale[None]


def byte_mismatch_count(actual, expected):
    actual_bits = jax.lax.bitcast_convert_type(actual, jnp.uint8)
    expected_bits = jax.lax.bitcast_convert_type(expected, jnp.uint8)
    return int(jnp.sum(actual_bits != expected_bits))


def timed_samples(fn, args, iters, warmup):
    for _ in range(warmup):
        jax.block_until_ready(fn(*args))
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        jax.block_until_ready(fn(*args))
        times.append(time.perf_counter() - t0)
    return sample_statistics(times)


def timed(fn, args, iters, warmup):
    median, _ = timed_samples(fn, args, iters, warmup)
    return median


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--git-sha", required=True)
    p.add_argument("--tokens", type=int, default=65536)
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--shape", action="append", choices=[label for label, _, _, _ in SHAPES])
    p.add_argument("--producer", choices=["none", "cute"], default="none")
    p.add_argument("--out", default="bench_mxfp8_dense.json")
    return p.parse_args(argv)


def main():
    a = parse_args()
    if a.producer == "cute":
        from experiments.grug.moe.standalone.mxfp8_grouped.quantize import (  # noqa: PLC0415
            quantize_mxfp8,
            quantize_mxfp8_tokens,
        )
        from experiments.grug.moe.standalone.mxfp8_grouped.quantize_cute import (  # noqa: PLC0415
            dual_quantize_mxfp8_cute,
        )

    dev = jax.devices()[0]
    print(f"device: {dev.device_kind}, jax {jax.__version__}")
    selected_shapes = [shape for shape in SHAPES if a.shape is None or shape[0] in a.shape]
    results = {
        "device": str(dev.device_kind),
        "backend": str(dev.client.platform_version),
        "jax": jax.__version__,
        "git_sha": a.git_sha,
        "xla_flags": os.environ.get("XLA_FLAGS", ""),
        "tokens": a.tokens,
        "warmup": a.warmup,
        "iters": a.iters,
        "shapes": {},
    }
    production_measurements = []

    for label, k, n, production_weight in selected_shapes:
        key = jax.random.PRNGKey(0)
        kx, kw = jax.random.split(key)
        x = jax.random.normal(kx, (a.tokens, k), dtype=jnp.bfloat16)
        w = jax.random.normal(kw, (k, n), dtype=jnp.bfloat16) / (k**0.5)
        flops_fwd = 2 * a.tokens * k * n
        ref_out = jax.lax.dot_general(x.astype(jnp.float32), w.astype(jnp.float32), DNUMS)
        ref_gx, ref_gw = fwd_bwd(lambda x, w: jax.lax.dot_general(x, w, DNUMS))(
            x.astype(jnp.float32), w.astype(jnp.float32)
        )
        g = (2 * ref_out).astype(jnp.bfloat16)
        oracle_ref_gx = jax.lax.dot_general(g.astype(jnp.float32), w.T.astype(jnp.float32), DNUMS)
        oracle_ref_gw = jax.lax.dot_general(x.T.astype(jnp.float32), g.astype(jnp.float32), DNUMS)
        shape_res = {}
        for arm, dot in ARMS.items():
            fwd = jax.jit(dot)
            bwd = jax.jit(fwd_bwd(dot))
            lowered_fwd = fwd.lower(x, w)
            lowered_bwd = bwd.lower(x, w)
            compile_started = time.perf_counter()
            compiled_fwd = lowered_fwd.compile()
            compiled_bwd = lowered_bwd.compile()
            compile_ms = (time.perf_counter() - compile_started) * 1e3
            if arm == "mxfp8":
                # The custom call must be present at lowering time; after XLA
                # optimization it may be rewritten (e.g. into a cuDNN fusion),
                # so log the compiled custom-call targets rather than assert.
                assert "block_scaled_dot" in lowered_fwd.as_text(), "mxfp8 arm did not lower to __op$block_scaled_dot"
                compiled_text = compiled_fwd.as_text()
                targets = sorted(set(re.findall(r'custom_call_target="([^"]+)"', compiled_text)))
                fp8_ops = sorted(set(re.findall(r"\b(\S*(?:block_scaled|blockScaled|f8)\S*)\b", compiled_text)))[:8]
                print(f"  [mxfp8] compiled custom calls: {targets}")
                print(f"  [mxfp8] fp8-ish compiled symbols: {fp8_ops}")
            t_fwd, mad_fwd = timed_samples(compiled_fwd, (x, w), a.iters, a.warmup)
            t_bwd, mad_bwd = timed_samples(compiled_bwd, (x, w), a.iters, a.warmup)
            gx, gw = compiled_bwd(x, w)
            shape_res[arm] = {
                "fwd_ms": t_fwd * 1e3,
                "fwd_mad_ms": mad_fwd * 1e3,
                "fwd_tfs": flops_fwd / t_fwd / 1e12,
                "fwdbwd_ms": t_bwd * 1e3,
                "fwdbwd_mad_ms": mad_bwd * 1e3,
                "fwdbwd_tfs": 3 * flops_fwd / t_bwd / 1e12,
                "compile_ms": compile_ms,
                "err_out": rel_frob(compiled_fwd(x, w), ref_out),
                "err_gx": rel_frob(gx, ref_gx),
                "err_gw": rel_frob(gw, ref_gw),
            }
        # MXFP8-001b/U001: quantize outside the timed region to isolate the
        # __cudnn$blockScaledDot calls. U001 prepares every orientation needed
        # by forward, dgrad, and wgrad, then times those three GEMMs together.
        x3 = x.reshape(1, a.tokens, k)
        wt3 = w.T.reshape(1, n, k)  # rhs is (B, N, K), contract dim last
        quant = jax.jit(lambda t: quantize(t, MXFP8_CONFIGS[0]))
        xq, xs = jax.block_until_ready(quant(x3))
        wq, ws = jax.block_until_ready(quant(wt3))
        pm = jax.jit(lambda xq, wq, xs, ws: scaled_matmul(xq, wq, xs, ws, preferred_element_type=jnp.bfloat16))
        t_pm = timed(pm, (xq, wq, xs, ws), a.iters, a.warmup)
        t_qx = timed(quant, (x3,), a.iters, a.warmup)
        out_pm = pm(xq, wq, xs, ws).reshape(a.tokens, n)
        shape_res["mxfp8_prequant"] = {
            "fwd_ms": t_pm * 1e3,
            "fwd_tfs": flops_fwd / t_pm / 1e12,
            "quantize_x_ms": t_qx * 1e3,
            "err_out": rel_frob(out_pm, ref_out),
        }

        def quantize_oracle_operands(x, w, g):
            return (
                *quantize(x[None], MXFP8_CONFIGS[0]),
                *quantize(w.T[None], MXFP8_CONFIGS[0]),
                *quantize(g[None], MXFP8_CONFIGS[0]),
                *quantize(w[None], MXFP8_CONFIGS[0]),
                *quantize(x.T[None], MXFP8_CONFIGS[0]),
                *quantize(g.T[None], MXFP8_CONFIGS[0]),
            )

        def prequant_fwdbwd(xq, xs, wtq, wts, gq, gs, wq, ws, xtq, xts, gtq, gts):
            y = scaled_matmul(xq, wtq, xs, wts, preferred_element_type=jnp.bfloat16)
            dx = scaled_matmul(gq, wq, gs, ws, preferred_element_type=jnp.bfloat16)
            dw = scaled_matmul(xtq, gtq, xts, gts, preferred_element_type=jnp.bfloat16)
            return y, dx, dw

        quantize_all = jax.jit(quantize_oracle_operands)
        oracle_operands = jax.block_until_ready(quantize_all(x, w, g))
        quantize_ms, quantize_mad_ms = timed_samples(quantize_all, (x, w, g), a.iters, a.warmup)
        oracle = jax.jit(prequant_fwdbwd)
        lowered_oracle = oracle.lower(*oracle_operands)
        compile_started = time.perf_counter()
        compiled_oracle = lowered_oracle.compile()
        compile_ms = (time.perf_counter() - compile_started) * 1e3
        compiled_oracle_text = compiled_oracle.as_text()
        oracle_call_count = custom_call_count(compiled_oracle_text, "__cudnn$blockScaledDot")
        assert oracle_call_count >= 3, "oracle did not compile all three block-scaled dots"
        oracle_targets = sorted(set(re.findall(r'custom_call_target="([^"]+)"', compiled_oracle_text)))
        oracle_ms, oracle_mad_ms = timed_samples(compiled_oracle, oracle_operands, a.iters, a.warmup)
        oracle_y, oracle_dx, oracle_dw = compiled_oracle(*oracle_operands)
        oracle_y = oracle_y.reshape(a.tokens, n)
        oracle_dx = oracle_dx.reshape(a.tokens, k)
        oracle_dw = oracle_dw.reshape(k, n)
        shape_res["mxfp8_prequant_fwdbwd"] = {
            "fwdbwd_ms": oracle_ms * 1e3,
            "fwdbwd_mad_ms": oracle_mad_ms * 1e3,
            "fwdbwd_tfs": 3 * flops_fwd / oracle_ms / 1e12,
            "quantize_all_ms": quantize_ms * 1e3,
            "quantize_all_mad_ms": quantize_mad_ms * 1e3,
            "compile_ms": compile_ms,
            "custom_call_targets": oracle_targets,
            "block_scaled_dot_call_count": oracle_call_count,
            "err_out": rel_frob(oracle_y, ref_out),
            "err_gx": rel_frob(oracle_dx, oracle_ref_gx),
            "err_gw": rel_frob(oracle_dw, oracle_ref_gw),
        }
        if a.producer == "cute":

            def cute_orientations(t):
                return linear_orientations(*dual_quantize_mxfp8_cute(t))

            def cute_quantize_reuse(x, w, g):
                return *cute_orientations(x), *cute_orientations(w), *cute_orientations(g)

            producer = jax.jit(cute_quantize_reuse).lower(x, w, g).compile()
            producer_outputs = jax.block_until_ready(producer(x, w, g))
            xqr, xrs, xqc, xcs, wqr, wrs, wqc, wcs, gqr, grs, gqc, gcs = producer_outputs
            cute_operands = (xqr, xrs, wqc, wcs, gqr, grs, wqr, wrs, xqc, xcs, gqc, gcs)

            def grouped_reference_orientations(t):
                q_row, s_row = quantize_mxfp8(t)
                q_col, s_col = quantize_mxfp8_tokens(t)
                return linear_orientations(q_row, s_row.T, q_col, s_col)

            grouped_reference = jax.jit(
                lambda x, w, g: (
                    *grouped_reference_orientations(x),
                    *grouped_reference_orientations(w),
                    *grouped_reference_orientations(g),
                )
            )(x, w, g)
            grouped_mismatches = [
                byte_mismatch_count(actual, expected)
                for actual, expected in zip(producer_outputs, grouped_reference, strict=True)
            ]
            assert not any(grouped_mismatches), f"CuTe operands differ from grouped reference: {grouped_mismatches}"
            dense_jax_mismatches = {
                name: byte_mismatch_count(actual, expected)
                for name, actual, expected in zip(
                    (
                        "x_row",
                        "x_row_scale",
                        "w_col",
                        "w_col_scale",
                        "g_row",
                        "g_row_scale",
                        "w_row",
                        "w_row_scale",
                        "x_col",
                        "x_col_scale",
                        "g_col",
                        "g_col_scale",
                    ),
                    cute_operands,
                    oracle_operands,
                    strict=True,
                )
            }
            producer_ms, producer_mad_ms = timed_samples(producer, (x, w, g), a.iters, a.warmup)

            def cute_reuse_fwdbwd(x, w, g):
                xqr, xrs, xqc, xcs, wqr, wrs, wqc, wcs, gqr, grs, gqc, gcs = cute_quantize_reuse(x, w, g)
                return prequant_fwdbwd(xqr, xrs, wqc, wcs, gqr, grs, wqr, wrs, xqc, xcs, gqc, gcs)

            def cute_unshared_fwdbwd(x, w, g):
                x_forward = cute_orientations(jax.lax.optimization_barrier(x))
                w_forward = cute_orientations(jax.lax.optimization_barrier(w))
                g_dgrad = cute_orientations(jax.lax.optimization_barrier(g))
                w_dgrad = cute_orientations(jax.lax.optimization_barrier(w))
                x_wgrad = cute_orientations(jax.lax.optimization_barrier(x))
                g_wgrad = cute_orientations(jax.lax.optimization_barrier(g))
                out_type = jnp.bfloat16
                y = scaled_matmul(
                    x_forward[0], w_forward[2], x_forward[1], w_forward[3], preferred_element_type=out_type
                )
                dx = scaled_matmul(g_dgrad[0], w_dgrad[0], g_dgrad[1], w_dgrad[1], preferred_element_type=out_type)
                dw = scaled_matmul(x_wgrad[2], g_wgrad[2], x_wgrad[3], g_wgrad[3], preferred_element_type=out_type)
                return y, dx, dw

            for producer_arm, fn in {
                "mxfp8_cute_reuse": cute_reuse_fwdbwd,
                "mxfp8_cute_unshared": cute_unshared_fwdbwd,
            }.items():
                lowered = jax.jit(fn).lower(x, w, g)
                compile_started = time.perf_counter()
                compiled = lowered.compile()
                producer_compile_ms = (time.perf_counter() - compile_started) * 1e3
                compiled_text = compiled.as_text()
                targets = sorted(set(re.findall(r'custom_call_target="([^"]+)"', compiled_text)))
                block_scaled_dot_calls = custom_call_count(compiled_text, "__cudnn$blockScaledDot")
                assert block_scaled_dot_calls >= 3
                elapsed, elapsed_mad = timed_samples(compiled, (x, w, g), a.iters, a.warmup)
                out, dx, dw = compiled(x, w, g)
                shape_res[producer_arm] = {
                    "fwdbwd_ms": elapsed * 1e3,
                    "fwdbwd_mad_ms": elapsed_mad * 1e3,
                    "fwdbwd_tfs": 3 * flops_fwd / elapsed / 1e12,
                    "compile_ms": producer_compile_ms,
                    "standalone_reuse_producer_ms": producer_ms * 1e3,
                    "standalone_reuse_producer_mad_ms": producer_mad_ms * 1e3,
                    "custom_call_targets": targets,
                    "block_scaled_dot_call_count": block_scaled_dot_calls,
                    "dense_jax_operand_byte_mismatches": dense_jax_mismatches,
                    "err_out": rel_frob(out.reshape(a.tokens, n), ref_out),
                    "err_gx": rel_frob(dx.reshape(a.tokens, k), oracle_ref_gx),
                    "err_gw": rel_frob(dw.reshape(k, n), oracle_ref_gw),
                }
        if production_weight:
            production_measurements.append((production_weight, oracle_ms, shape_res["fp8_tensor"]["fwdbwd_ms"] / 1e3))

        base = shape_res["bf16"]
        print(f"\n== {label} (T={a.tokens}) ==")
        for arm, r in shape_res.items():
            line = f"  {arm:24s}"
            if "fwd_ms" in r:
                line += f" fwd {r['fwd_ms']:7.3f} ms ({r['fwd_tfs']:7.1f} TF/s," f" {base['fwd_ms'] / r['fwd_ms']:.3f}x)"
            if "fwdbwd_ms" in r:
                line += (
                    f"  fwd+bwd {r['fwdbwd_ms']:7.3f} ms ({r['fwdbwd_tfs']:7.1f} TF/s,"
                    f" {base['fwdbwd_ms']/r['fwdbwd_ms']:.3f}x)"
                    f"  err out/gx/gw {r['err_out']:.2e}/{r['err_gx']:.2e}/{r['err_gw']:.2e}"
                )
            else:
                line += f"  quantize_x {r['quantize_x_ms']:.3f} ms  err out {r['err_out']:.2e}"
            print(line)
        results["shapes"][label] = shape_res

    covered_weight = sum(weight for weight, _, _ in production_measurements)
    results["covered_production_weight"] = covered_weight
    if covered_weight:
        results["weighted_production_ratio"] = weighted_production_ratio(production_measurements)
        results["complete_production_mix"] = covered_weight == 7
        print(
            f"\nprequant MXFP8 / per-tensor weighted production time: "
            f"{results['weighted_production_ratio']:.4f}x (weight {covered_weight}/7)"
        )

    with open(a.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
