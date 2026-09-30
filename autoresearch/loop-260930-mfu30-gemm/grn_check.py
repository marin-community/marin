"""GPU correctness and timing for levanter's fused gated_rms_norm against the XLA reference.

For each shape: output and all four gradients (x, norm weight, w_down, w_up) of the Pallas path
and of the bf16 XLA reference, each compared with an f32 reference evaluated on the same bf16
inputs. "Within bf16 rounding" means the kernel's error against f32 is no larger than the bf16
reference's own error against f32 (ratio <= ~1.5). Also times forward and forward+backward and
reports compiled temp memory.

Usage: python grn_check.py [--iters 20] [--out result.json]
"""

import argparse
import json
import time

import jax
import jax.numpy as jnp
import numpy as np

from levanter.kernels.pallas.gated_rms_norm import GatedRmsNormBlockSizes, gated_rms_norm, gated_rms_norm_reference

EPS = 1e-5
SHAPES = [(16, 4096, 6144, 128), (3, 1000, 768, 128), (1, 100, 6144, 128)]


def inputs(shape, seed):
    b, s, d, r = shape
    k = jax.random.split(jax.random.key(seed), 5)
    std = 0.5 / np.sqrt(d)
    x = jax.random.normal(k[0], (b, s, d), jnp.bfloat16)
    w = (1.0 + 0.2 * jax.random.normal(k[1], (d,))).astype(jnp.bfloat16)
    wd = (4 * std * jax.random.normal(k[2], (d, r))).astype(jnp.bfloat16)
    wu = (4 * std * jax.random.normal(k[3], (r, d)) * np.sqrt(d / r)).astype(jnp.bfloat16)
    cot = jax.random.normal(k[4], (b, s, d), jnp.bfloat16)
    return x, w, wd, wu, cot


def fwd_bwd(fn):
    def run(x, w, wd, wu, cot):
        out, vjp = jax.vjp(fn, x, w, wd, wu)
        return out, *vjp(cot)

    return jax.jit(run)


def timed(f, args, iters):
    out = f(*args)
    jax.block_until_ready(out)
    for _ in range(3):
        out = f(*args)
    jax.block_until_ready(out)
    t0 = time.perf_counter()
    for _ in range(iters):
        out = f(*args)
    jax.block_until_ready(out)
    return (time.perf_counter() - t0) / iters * 1e3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--out", default="")
    ap.add_argument("--block", default="", help="t,stats_d,out_d,stats_warps,stats_stages,out_warps,out_stages")
    args = ap.parse_args()
    bs = GatedRmsNormBlockSizes()
    if args.block:
        v = [int(t) for t in args.block.split(",")]
        bs = GatedRmsNormBlockSizes(*v)
    ref = lambda x, w, wd, wu: gated_rms_norm_reference(x, w, wd, wu, eps=EPS)  # noqa: E731
    ker = lambda x, w, wd, wu: gated_rms_norm(x, w, wd, wu, eps=EPS, implementation="pallas_gpu", block_sizes=bs)  # noqa: E731
    results = []
    for shape in SHAPES:
        x, w, wd, wu, cot = inputs(shape, seed=sum(shape))
        f32 = [a.astype(jnp.float32) for a in (x, w, wd, wu, cot)]
        want32 = [np.asarray(a, np.float32) for a in fwd_bwd(ref)(*f32)]
        got_ref = [np.asarray(a, np.float32) for a in fwd_bwd(ref)(x, w, wd, wu, cot)]
        got_ker = [np.asarray(a, np.float32) for a in fwd_bwd(ker)(x, w, wd, wu, cot)]
        row = dict(shape=shape, block_sizes=str(bs))
        for name, a, b, c in zip(("out", "dx", "dnorm_weight", "dw_down", "dw_up"), got_ker, got_ref, want32):
            scale = float(np.sqrt(np.mean(c**2))) or 1.0
            row[name] = dict(
                max_abs_vs_ref=float(np.max(np.abs(a - b))),
                rel_rms_vs_ref=float(np.sqrt(np.mean((a - b) ** 2)) / scale),
                kernel_rel_rms_vs_f32=float(np.sqrt(np.mean((a - c) ** 2)) / scale),
                ref_rel_rms_vs_f32=float(np.sqrt(np.mean((b - c) ** 2)) / scale),
                max_abs_ref=float(np.max(np.abs(b))),
                finite=bool(np.all(np.isfinite(a))),
            )
            r = row[name]
            ratio = r["kernel_rel_rms_vs_f32"] / max(r["ref_rel_rms_vs_f32"], 1e-30)
            r["error_ratio_kernel_over_ref"] = ratio
            print(
                f"{str(shape):26s} {name:13s} max|k-r| {r['max_abs_vs_ref']:.3e} (|r|max {r['max_abs_ref']:.2e}) "
                f"rel-rms k-r {r['rel_rms_vs_ref']:.2e} | vs f32: kernel {r['kernel_rel_rms_vs_f32']:.2e} "
                f"ref {r['ref_rel_rms_vs_f32']:.2e} ratio {ratio:.2f} finite {r['finite']}",
                flush=True,
            )
        if shape[0] * shape[1] >= 65536:
            fk, fr = jax.jit(ker), jax.jit(ref)
            row["fwd_ms_kernel"] = timed(fk, (x, w, wd, wu), args.iters)
            row["fwd_ms_ref"] = timed(fr, (x, w, wd, wu), args.iters)
            bk, br = fwd_bwd(ker), fwd_bwd(ref)
            row["fwdbwd_ms_kernel"] = timed(bk, (x, w, wd, wu, cot), args.iters)
            row["fwdbwd_ms_ref"] = timed(br, (x, w, wd, wu, cot), args.iters)
            mk = bk.lower(x, w, wd, wu, cot).compile().memory_analysis()
            mr = br.lower(x, w, wd, wu, cot).compile().memory_analysis()
            row["temp_gib_kernel"] = mk.temp_size_in_bytes / 2**30
            row["temp_gib_ref"] = mr.temp_size_in_bytes / 2**30
            print(
                f"{str(shape):26s} fwd {row['fwd_ms_kernel']:.3f} ms vs ref {row['fwd_ms_ref']:.3f} ms; "
                f"fwd+bwd {row['fwdbwd_ms_kernel']:.3f} vs {row['fwdbwd_ms_ref']:.3f} ms; "
                f"temp {row['temp_gib_kernel']:.2f} vs {row['temp_gib_ref']:.2f} GiB",
                flush=True,
            )
        results.append(row)
    print("RESULT", json.dumps(results))
    if args.out:
        json.dump(results, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
