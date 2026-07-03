#!/usr/bin/env python
# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0
"""E2E grug MoE layer benchmark (fwd+bwd) on one 8xH100 node: bf16 vs FP8.

Spike harness for the FP8 MoE comms research (logbook
``.agents/logbooks/fp8-moe-comms.md``, issue #6911). Measures the full
``levanter.grug.grug_moe.moe_mlp`` layer — routing dispatch, EP collectives,
expert GEMMs, combine — under the real d2560 EP4 layout:

  mesh (replica_dcn=1, data=2, expert=4, model=1) on 8 GPUs
  D=2560, I=1280, E=256, K=4; T_local=16384 tokens/device (B = 8192*16)
  => per device: E_local=64, ~1024 avg tokens/expert, 65536 assignments

Modes:
  bf16     — production baseline (ragged_dot implementation="auto" => triton)
  fp8gemm  — Fp8RaggedDotOp expert GEMMs, bf16 wire
  fp8wire  — fp8gemm + dispatch/combine collectives carrying FP8 over the wire

Usage (on the 8-GPU pod):
  python lib/levanter/scripts/bench/bench_fp8_moe_layer.py --impl ring --modes bf16,fp8gemm
  python lib/levanter/scripts/bench/bench_fp8_moe_layer.py --impl ragged_all_to_all --profile /tmp/prof
"""

import argparse
import time

import jax
import jax.numpy as jnp
from haliax.quantization import Fp8RaggedDotOp, apply_updates, partition_for_grad_overwrite
from jax.sharding import NamedSharding, PartitionSpec as P

from levanter.grug._moe.common import MoeRaggedDotOps
from levanter.grug.grug_moe import moe_mlp
from levanter.grug.sharding import compact_grug_mesh

D, I, E, K = 2560, 1280, 256, 4
CAPACITY_FACTOR = 1.25


def make_routing(key, t, *, skew=0.6):
    """Token->expert assignments with realistic non-uniform expert popularity."""
    kb, kg = jax.random.split(key)
    bias = skew * jax.random.normal(kb, (E,), dtype=jnp.float32)
    logits = bias[None, :] + jax.random.gumbel(kg, (t, E), dtype=jnp.float32)
    weights, selected = jax.lax.top_k(logits, K)
    weights = jax.nn.softmax(weights, axis=-1)
    return selected.astype(jnp.int32), weights.astype(jnp.bfloat16)


def make_inputs(mesh, t_global, seed=0):
    key = jax.random.key(seed)
    kx, kr, k13, k2, kc = jax.random.split(key, 5)
    batch = P(("replica_dcn", "data", "expert"))
    with jax.set_mesh(mesh):
        x = jax.device_put(
            (jax.random.normal(kx, (t_global, D)) * 0.1).astype(jnp.bfloat16),
            NamedSharding(mesh, batch),
        )
        selected, weights = make_routing(kr, t_global)
        selected = jax.device_put(selected, NamedSharding(mesh, batch))
        weights = jax.device_put(weights, NamedSharding(mesh, batch))
        w13 = jax.device_put(
            (jax.random.normal(k13, (E, D, 2 * I)) * 0.02).astype(jnp.bfloat16),
            NamedSharding(mesh, P("expert", None, None)),
        )
        w2 = jax.device_put(
            (jax.random.normal(k2, (E, I, D)) * 0.02).astype(jnp.bfloat16),
            NamedSharding(mesh, P("expert", None, None)),
        )
        cot = jax.device_put(
            (jax.random.normal(kc, (t_global, D)) * 0.1).astype(jnp.bfloat16),
            NamedSharding(mesh, batch),
        )
    return x, selected, weights, w13, w2, cot


def time_throughput_ms(fn, *args, iters=20, warmup=3):
    """Enqueue ``iters`` calls then block once (pipelined-step methodology)."""
    jax.block_until_ready(fn(*args))
    for _ in range(warmup):
        jax.block_until_ready(fn(*args))
    t0 = time.perf_counter()
    for _ in range(iters):
        out = fn(*args)
    jax.block_until_ready(out)
    return (time.perf_counter() - t0) / iters * 1e3


def relfrob(a, b):
    a = jnp.asarray(a, jnp.float32)
    b = jnp.asarray(b, jnp.float32)
    return float(jnp.linalg.norm(a - b) / jnp.linalg.norm(b))


def make_ops(mode):
    if mode in ("bf16", "wireonly"):
        return None
    if mode in ("fp8gemm", "fp8wire"):
        return MoeRaggedDotOps(
            w13=Fp8RaggedDotOp.init(amax_history_length=16),
            w2=Fp8RaggedDotOp.init(amax_history_length=16),
        )
    raise ValueError(f"unknown mode {mode!r}")


def build_step(mesh, impl, sel, wts, cot, mode):
    """jitted fwd+bwd step: loss = <moe_mlp(x), cot>; grads wrt (x, w13, w2[, ops]).

    Modes: bf16 (baseline), fp8gemm (FP8 expert GEMMs), fp8wire (fp8gemm + FP8
    over-the-wire collectives), wireonly (bf16 GEMMs + FP8 wire, for attribution).
    """
    wire = mode in ("fp8wire", "wireonly")

    if mode in ("bf16", "wireonly"):

        def loss(x, w13, w2):
            out = moe_mlp(x, sel, wts, w13, w2, implementation=impl, mesh=mesh, fp8_wire=wire)
            return (out.astype(jnp.float32) * cot.astype(jnp.float32)).sum()

        return jax.jit(jax.value_and_grad(loss, argnums=(0, 1, 2)))

    def loss_fp8(x, w13, w2, ops):
        out = moe_mlp(x, sel, wts, w13, w2, implementation=impl, mesh=mesh, ragged_dot_ops=ops, fp8_wire=wire)
        return (out.astype(jnp.float32) * cot.astype(jnp.float32)).sum()

    return jax.jit(jax.value_and_grad(loss_fp8, argnums=(0, 1, 2, 3)))


def warmup_op_state(step, x, w13, w2, ops, steps=4):
    """Run fwd/bwd steps applying delayed-scaling overwrites so scales adapt."""
    for _ in range(steps):
        _, grads = step(x, w13, w2, ops)
        overwrites, op_grads = partition_for_grad_overwrite(grads[3])
        zero = jax.tree.map(lambda g: g * 0.0 if g is not None else None, op_grads)
        ops = apply_updates(ops, zero, overwrites)
    return ops


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--impl", default="ring", choices=["ring", "ragged_all_to_all"])
    ap.add_argument("--modes", default="bf16")
    ap.add_argument("--t-local", type=int, default=16384)
    ap.add_argument("--expert-axis", type=int, default=4)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--profile", default=None, help="capture a jax profiler trace to this dir")
    ap.add_argument("--check-drops", action="store_true")
    ap.add_argument("--distributed", action="store_true", help="join a multi-node iris job via iris jax_init")
    ap.add_argument("--coordinator", default=None, help="manual jax.distributed coordinator host:port")
    ap.add_argument("--num-processes", type=int, default=2)
    ap.add_argument("--process-id", type=int, default=0)
    args = ap.parse_args()

    if args.distributed:
        # Local import: iris is only needed for multi-node runs.
        from iris.runtime.jax_init import initialize_jax  # noqa: PLC0415

        initialize_jax()
    elif args.coordinator:
        jax.distributed.initialize(
            coordinator_address=args.coordinator, num_processes=args.num_processes, process_id=args.process_id
        )

    def pr(*a):
        if jax.process_index() == 0:
            print(*a, flush=True)

    n_dev = len(jax.devices())
    mesh = compact_grug_mesh(expert_axis_size=args.expert_axis)
    t_global = args.t_local * n_dev
    pr(f"jax {jax.__version__}  {n_dev}x {jax.devices()[0].device_kind}")
    pr(f"mesh {dict(mesh.shape)}  impl={args.impl}")
    pr(
        f"D={D} I={I} E={E} K={K}  T_local={args.t_local} T_global={t_global} "
        f"(E_local={E // args.expert_axis}, "
        f"~{args.t_local * n_dev // (mesh.shape['data'] * mesh.shape['replica_dcn']) * K // E} avg tok/expert/device)"
    )

    x, sel, wts, w13, w2, cot = make_inputs(mesh, t_global)

    if args.check_drops:
        with jax.set_mesh(mesh):
            _, dropped = moe_mlp(
                x, sel, wts, w13, w2, implementation=args.impl, mesh=mesh, report_capacity_overflow=True
            )
            total = t_global * K
            pr(f"dropped assignments: {int(dropped)} / {total} ({100 * int(dropped) / total:.3f}%)")

    results = {}
    for mode in args.modes.split(","):
        with jax.set_mesh(mesh):
            step = build_step(mesh, args.impl, sel, wts, cot, mode)
            ops = make_ops(mode)
            if ops is None:
                ms = time_throughput_ms(step, x, w13, w2, iters=args.iters)
                loss_val, grads = step(x, w13, w2)
            else:
                ops = warmup_op_state(step, x, w13, w2, ops)
                ms = time_throughput_ms(step, x, w13, w2, ops, iters=args.iters)
                loss_val, grads = step(x, w13, w2, ops)
        results[mode] = (ms, loss_val, grads)
        tok_s = t_global / (ms * 1e-3)
        pr(f"{mode:8s} {ms:8.2f} ms/step   {tok_s / 1e6:6.1f} Mtok/s")

    if "bf16" in results:
        base_ms = results["bf16"][0]
        for mode, (ms, loss_val, grads) in results.items():
            if mode == "bf16":
                continue
            gx = grads[0]
            gx_ref = results["bf16"][2][0]
            pr(
                f"{mode}: speedup vs bf16 = {base_ms / ms:.3f}x   "
                f"relfrob(loss)={abs(float(loss_val) - float(results['bf16'][1])) / abs(float(results['bf16'][1])):.2e} "
                f"relfrob(dx)={relfrob(gx, gx_ref):.2e}"
            )

    if args.profile:
        mode = args.modes.split(",")[0]
        with jax.set_mesh(mesh):
            step = build_step(mesh, args.impl, sel, wts, cot, mode)
            ops = make_ops(mode)
            call_args = (x, w13, w2) if ops is None else (x, w13, w2, ops)
            jax.block_until_ready(step(*call_args))
            with jax.profiler.trace(args.profile):
                for _ in range(3):
                    out = step(*call_args)
                jax.block_until_ready(out)
        pr(f"profile written to {args.profile}")


if __name__ == "__main__":
    main()
