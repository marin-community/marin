#!/usr/bin/env python
# Copyright The Levanter Authors
#
# SPDX-License-Identifier: Apache-2.0
"""Microbenchmark: FP8 ``ragged_dot`` (MoE grouped matmul) vs the bf16 Triton baseline.

Run on an H100 machine with a GPU-enabled JAX environment::

    python lib/haliax/bench/bench_fp8_ragged_dot.py
    python lib/haliax/bench/bench_fp8_ragged_dot.py --iters 30

Reports, for realistic MoE expert GEMMs (hidden 2560, intermediate 1280) across
a sweep of ``E_local`` and ``tokens_per_expert``, both w13 and w2 gate+up / down
shapes:

  * forward TFLOP/s and fp8/bf16 speedup,
  * end-to-end fwd+bwd throughput speedup,
  * forward relative-Frobenius error of the FP8 path vs the bf16 baseline.

Baseline is ``haliax.nn.ragged_dot(..., implementation="triton")`` in bf16.

Timing methodology: the throughput timer enqueues N ``value_and_grad`` calls and
``block_until_ready``s once, representing a pipelined training step where the
host runs ahead of the device.  (A per-call sync would add an artificial stall
per step that penalizes FP8's higher kernel-launch count and understates the
speedup a training run sees.)  This harness is the perf acceptance gate for the
FP8 ragged path; wall-clock speedup assertions do not belong in the pytest suite.
"""

import argparse
import time

import haliax.nn as hnn
import jax
import jax.numpy as jnp
from haliax.quantization import Fp8RaggedDotOp, apply_updates, partition_for_grad_overwrite

# MoE with hidden D=2560, intermediate F=1280.  Per-device expert GEMMs:
#   w13 (gate+up): lhs[T, 2560] x rhs[E, 2560, 2560]  (out 2F = 2560)
#   w2  (down):    lhs[T, 1280] x rhs[E, 1280, 2560]
SHAPES = {"w13": (2560, 2560), "w2": (1280, 2560)}
TOKENS_PER_EXPERT = (512, 1024, 2048, 4096)
E_LOCALS = (16, 32, 64)
OPERATING_POINT = (64, 1024)  # (E_local, tokens_per_expert)


def nonuniform_group_sizes(e, total, key, skew=0.6):
    """Realistic skewed/random token counts per expert summing to ``total``."""
    w = jax.random.uniform(key, (e,), minval=1.0 - skew, maxval=1.0 + skew)
    w = w / w.sum()
    sizes = jnp.floor(w * total).astype(jnp.int32)
    sizes = sizes.at[-1].add(total - sizes.sum())
    return sizes


def make_inputs(e, tpe, k, n, dtype=jnp.bfloat16, seed=0):
    t = e * tpe
    key = jax.random.key(seed)
    k1, k2, k3 = jax.random.split(key, 3)
    lhs = (jax.random.normal(k1, (t, k)) * 0.1).astype(dtype)
    rhs = (jax.random.normal(k2, (e, k, n)) * 0.1).astype(dtype)
    group_sizes = nonuniform_group_sizes(e, t, k3)
    return lhs, rhs, group_sizes


def time_throughput_ms(fn, *args, iters=30, warmup=5):
    """Throughput timer: enqueue ``iters`` calls then block once.

    Represents a pipelined training step where the host overlaps host and device
    work.  Use this timer for performance comparisons and acceptance gates.
    """
    jfn = jax.jit(fn)
    # compile + prime
    jax.block_until_ready(jfn(*args))
    for _ in range(warmup):
        jax.block_until_ready(jfn(*args))
    t0 = time.perf_counter()
    for _ in range(iters):
        out = jfn(*args)
    jax.block_until_ready(out)
    return (time.perf_counter() - t0) / iters * 1e3


def relfrob(a, b):
    a = a.astype(jnp.float32)
    b = b.astype(jnp.float32)
    return float(jnp.linalg.norm(a - b) / jnp.linalg.norm(b))


def warmup_op_state(op, lhs, rhs, group_sizes, steps=4):
    """Run a few fwd/bwd steps so delayed-scaling amax histories adapt."""

    def loss(op_, lhs_, rhs_):
        return hnn.ragged_dot(lhs_, rhs_, group_sizes, op=op_).sum()

    grad_fn = jax.jit(jax.grad(loss, argnums=0))
    for _ in range(steps):
        g = grad_fn(op, lhs, rhs)
        overwrites, grads = partition_for_grad_overwrite(g)
        zero = jax.tree_util.tree_map(lambda x: x * 0.0 if x is not None else None, grads)
        op = apply_updates(op, zero, overwrites)
    return op


def bench_config(e, tpe, k, n, iters, rev_dtype):
    lhs, rhs, group_sizes = make_inputs(e, tpe, k, n)
    op = Fp8RaggedDotOp.init(amax_history_length=16, rev_dtype=rev_dtype)
    op = warmup_op_state(op, lhs, rhs, group_sizes)

    def fp8_fwd(l, r, g):
        return hnn.ragged_dot(l, r, g, op=op)

    def bf16_fwd(l, r, g):
        return hnn.ragged_dot(l, r, g, implementation="triton")

    def fp8_fb(l, r):
        return jax.value_and_grad(lambda l_, r_: hnn.ragged_dot(l_, r_, group_sizes, op=op).sum(), (0, 1))(l, r)

    def bf16_fb(l, r):
        return jax.value_and_grad(
            lambda l_, r_: hnn.ragged_dot(l_, r_, group_sizes, implementation="triton").sum(), (0, 1)
        )(l, r)

    flops_fwd = 2 * (e * tpe) * k * n
    f8 = time_throughput_ms(fp8_fwd, lhs, rhs, group_sizes, iters=iters)
    bf = time_throughput_ms(bf16_fwd, lhs, rhs, group_sizes, iters=iters)
    f8_fb = time_throughput_ms(fp8_fb, lhs, rhs, iters=iters)
    bf_fb = time_throughput_ms(bf16_fb, lhs, rhs, iters=iters)

    out_fp8 = jax.jit(fp8_fwd)(lhs, rhs, group_sizes)
    out_bf16 = jax.jit(bf16_fwd)(lhs, rhs, group_sizes)
    err = relfrob(out_fp8, out_bf16)

    return {
        "fp8_fwd_ms": f8,
        "bf16_fwd_ms": bf,
        "fp8_tflops": flops_fwd / (f8 * 1e-3) / 1e12,
        "bf16_tflops": flops_fwd / (bf * 1e-3) / 1e12,
        "fwd_speedup": bf / f8,
        "fwd_bwd_speedup": bf_fb / f8_fb,
        "relfrob": err,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--quick", action="store_true", help="operating point only")
    ap.add_argument(
        "--rev-dtype",
        choices=("e5m2", "e4m3"),
        default="e5m2",
        help="output-gradient FP8 dtype: e5m2 = genuine mixed backward (jax >= 0.11.0, "
        "jax-ml/jax#38859), e4m3 = uniform backward that also lowers on jax 0.10.x",
    )
    args = ap.parse_args()
    rev_dtype = {"e5m2": jnp.float8_e5m2, "e4m3": jnp.float8_e4m3fn}[args.rev_dtype]

    print(f"jax {jax.__version__}  device {jax.devices()[0].device_kind}")
    print(
        f"FP8 ragged_dot (E4M3 fwd / {args.rev_dtype.upper()} bwd, delayed per-tensor scaling) vs bf16 Triton baseline"
    )
    if args.rev_dtype == "e5m2":
        print("Backward: genuine mixed e5m2 x e4m3 wgmma (requires jax >= 0.11.0, jax-ml/jax#38859).")
    else:
        print("Backward: uniform e4m3 x e4m3 (approximate output-gradient dtype).")
    # Show that group_sizes are genuinely non-uniform and dynamic.
    e0, tpe0 = OPERATING_POINT
    gs0 = nonuniform_group_sizes(e0, e0 * tpe0, jax.random.split(jax.random.key(0), 3)[2])
    print(
        f"group_sizes: NON-UNIFORM (e.g. E={e0} avg {tpe0}/expert -> "
        f"min={int(gs0.min())} max={int(gs0.max())} sum={int(gs0.sum())})\n"
    )
    header = (
        f"{'shape':5s} {'E':>3s} {'tok/E':>6s} {'K':>5s} {'N':>5s} "
        f"{'fp8 TF':>7s} {'bf16 TF':>7s} {'fwd x':>6s} "
        f"{'fwd+bwd x':>9s} {'relfrob':>9s}"
    )
    print(header)
    print("-" * len(header))

    if args.quick:
        configs = [(name, *OPERATING_POINT) for name in SHAPES]
    else:
        configs = [(name, e, tpe) for name in SHAPES for tpe in TOKENS_PER_EXPERT for e in E_LOCALS]

    for name, e, tpe in configs:
        k, n = SHAPES[name]
        try:
            r = bench_config(e, tpe, k, n, args.iters, rev_dtype)
            star = "  <-- operating point" if (e, tpe) == OPERATING_POINT and name == "w13" else ""
            print(
                f"{name:5s} {e:3d} {tpe:6d} {k:5d} {n:5d} "
                f"{r['fp8_tflops']:7.0f} {r['bf16_tflops']:7.0f} "
                f"{r['fwd_speedup']:6.2f} "
                f"{r['fwd_bwd_speedup']:9.2f} "
                f"{r['relfrob']:9.2e}{star}"
            )
        except Exception as exc:  # surface OOM/compile failures inline
            print(f"{name:5s} {e:3d} {tpe:6d} {k:5d} {n:5d}  FAILED: {type(exc).__name__}: {str(exc)[:80]}")


if __name__ == "__main__":
    main()
