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

Usage: python bench_mxfp8_dense.py [--tokens 65536] [--iters 50] [--out out.json]
"""

import argparse
import json
import re
import statistics
import time

import jax
import jax.numpy as jnp
from haliax.quantization import Fp8DotGeneralOp
from jax._src.cudnn.scaled_matmul_stablehlo import quantize
from jax.nn import get_scaled_dot_general_config, scaled_dot_general, scaled_matmul

# (label, K, N): row-13 d2560 dense GEMMs. q/k/v/o are all 2560x2560
# (20 heads x head_dim 128, MHA); shared expert is 2560<->1280.
SHAPES = [
    ("qkvo_2560x2560", 2560, 2560),
    ("shared_up_2560x1280", 2560, 1280),
    ("shared_down_1280x2560", 1280, 2560),
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


def timed(fn, args, iters, warmup):
    for _ in range(warmup):
        jax.block_until_ready(fn(*args))
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        jax.block_until_ready(fn(*args))
        times.append(time.perf_counter() - t0)
    return statistics.median(times)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tokens", type=int, default=65536)
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--out", default="bench_mxfp8_dense.json")
    a = p.parse_args()

    dev = jax.devices()[0]
    print(f"device: {dev.device_kind}, jax {jax.__version__}")
    results = {"device": str(dev.device_kind), "jax": jax.__version__, "tokens": a.tokens, "shapes": {}}

    for label, k, n in SHAPES:
        key = jax.random.PRNGKey(0)
        kx, kw = jax.random.split(key)
        x = jax.random.normal(kx, (a.tokens, k), dtype=jnp.bfloat16)
        w = jax.random.normal(kw, (k, n), dtype=jnp.bfloat16) / (k**0.5)
        flops_fwd = 2 * a.tokens * k * n
        ref_out = dot_bf16(x.astype(jnp.float32), w.astype(jnp.float32))
        ref_gx, ref_gw = fwd_bwd(lambda x, w: jax.lax.dot_general(x, w, DNUMS))(
            x.astype(jnp.float32), w.astype(jnp.float32)
        )
        shape_res = {}
        for arm, dot in ARMS.items():
            fwd = jax.jit(dot)
            bwd = jax.jit(fwd_bwd(dot))
            if arm == "mxfp8":
                lowered = fwd.lower(x, w)
                # The custom call must be present at lowering time; after XLA
                # optimization it may be rewritten (e.g. into a cuDNN fusion),
                # so log the compiled custom-call targets rather than assert.
                assert "block_scaled_dot" in lowered.as_text(), "mxfp8 arm did not lower to __op$block_scaled_dot"
                compiled = lowered.compile().as_text()
                targets = sorted(set(re.findall(r'custom_call_target="([^"]+)"', compiled)))
                fp8_ops = sorted(set(re.findall(r"\b(\S*(?:block_scaled|blockScaled|f8)\S*)\b", compiled)))[:8]
                print(f"  [mxfp8] compiled custom calls: {targets}")
                print(f"  [mxfp8] fp8-ish compiled symbols: {fp8_ops}")
            t_fwd = timed(fwd, (x, w), a.iters, a.warmup)
            t_bwd = timed(bwd, (x, w), a.iters, a.warmup)
            gx, gw = bwd(x, w)
            shape_res[arm] = {
                "fwd_ms": t_fwd * 1e3,
                "fwd_tfs": flops_fwd / t_fwd / 1e12,
                "fwdbwd_ms": t_bwd * 1e3,
                "fwdbwd_tfs": 3 * flops_fwd / t_bwd / 1e12,
                "err_out": rel_frob(fwd(x, w), ref_out),
                "err_gx": rel_frob(gx, ref_gx),
                "err_gw": rel_frob(gw, ref_gw),
            }
        # MXFP8-001b: GEMM-only throughput — scaled_matmul on operands quantized
        # OUTSIDE the timed region, to separate the __cudnn$blockScaledDot call
        # from XLA's quantize kernels. Also time the quantize step itself.
        cfg = MXFP8_CONFIGS[0]
        x3 = x.reshape(1, a.tokens, k)
        wt3 = jnp.ascontiguousarray(w.T).reshape(1, n, k)  # rhs is (B, N, K), contract dim last
        quant = jax.jit(lambda t: quantize(t, cfg))
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

        base = shape_res["bf16"]
        print(f"\n== {label} (T={a.tokens}) ==")
        for arm, r in shape_res.items():
            line = f"  {arm:14s} fwd {r['fwd_ms']:7.3f} ms ({r['fwd_tfs']:7.1f} TF/s, {base['fwd_ms']/r['fwd_ms']:.3f}x)"
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

    with open(a.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
