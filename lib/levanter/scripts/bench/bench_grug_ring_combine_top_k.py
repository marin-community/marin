# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0
"""Time and check the ring-EP MoE combine, the dispatch backward and the routing top-k on one GPU.

Defaults are the June Snowball per-GPU shapes at global batch 64 on EP8: x_global 262,144 x 2560 bf16, top-4 of
256 experts, 32 local experts on expert shard 0, capacity 131,072; the router matrix is 32,768 x 256 f32 with k=5.
Compares the scatter-add combine against the gather combine in ``ep_ring`` and ``jax.lax.top_k`` against the
Pallas-Triton routing top-k, printing ``BENCH`` timings and ``DIFF``/``TOPK-CHECK`` numerics lines.

    uv run python lib/levanter/scripts/bench/bench_grug_ring_combine_top_k.py [--section topk|moe]
"""

import argparse
import json
import time

import jax
import jax.numpy as jnp
import numpy as np

from levanter.grug._moe.ep_common import _prefix_cap_counts
from levanter.grug._moe import ep_ring
from levanter.grug._moe.routing_top_k import _total_order_key, triton_top_k_indices
from levanter.utils.jax_utils import is_rocm_backend

p = argparse.ArgumentParser()
p.add_argument("--tokens", type=int, default=262144)
p.add_argument("--hidden", type=int, default=2560)
p.add_argument("--experts", type=int, default=256)
p.add_argument("--topk", type=int, default=4)
p.add_argument("--ep", type=int, default=8)
p.add_argument("--iters", type=int, default=20)
p.add_argument("--section", choices=("all", "topk", "moe"), default="all")
args = p.parse_args()
print("devices", jax.devices(), "rocm", is_rocm_backend(), flush=True)

T, H, E, K, EP = args.tokens, args.hidden, args.experts, args.topk, args.ep
LE = E // EP
A = T * K
PCAP = A // EP


def bench(name, fn, *xs, nbytes=None):
    out = fn(*xs)
    jax.block_until_ready(out)
    times = []
    for _ in range(5):
        t0 = time.perf_counter()
        for _ in range(args.iters):
            out = fn(*xs)
        jax.block_until_ready(out)
        times.append((time.perf_counter() - t0) / args.iters)
    ms = 1e3 * float(np.median(times))
    extra = f"  {nbytes / (ms * 1e-3) / 1e12:.2f} TB/s" if nbytes else ""
    print(f"BENCH {name:40s} {ms:8.3f} ms{extra}", flush=True)
    print("JSON", json.dumps({"name": name, "ms": ms}), flush=True)
    return out


def diff(name, a, b):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    d = np.abs(a - b)
    scale = np.max(np.abs(b)) + 1e-30
    print(
        f"DIFF {name:40s} max_abs={d.max():.3e} mean_abs={d.mean():.3e} max_rel_to_max={d.max() / scale:.3e}",
        flush=True,
    )


@jax.jit
def routing(selected):
    """Same dispatch indices as ep_ring for expert shard 0."""
    expert_flat = selected.reshape(A)
    local_expert = expert_flat
    local_mask = local_expert < LE
    local_expert = jnp.where(local_mask, local_expert, 0)
    ids = jnp.arange(LE, dtype=jnp.int32)
    counts = jnp.sum((local_expert[:, None] == ids[None, :]).astype(jnp.int32) * local_mask[:, None], axis=0)
    accepted = _prefix_cap_counts(counts, capacity=PCAP)
    total = jnp.sum(accepted)
    valid = jnp.arange(PCAP, dtype=jnp.int32) < total
    flat_pos = jnp.arange(A, dtype=jnp.int32)
    key = jnp.where(local_mask, LE * A - (local_expert * A + flat_pos), -1)
    _, local_idx = jax.lax.top_k(key, PCAP)
    return local_idx // K, local_idx, valid


def combine_scatter(rows, token):
    return jnp.zeros((T, H), rows.dtype).at[token].add(rows, mode="drop")


def combine_scatter_f32(rows, token):
    return jnp.zeros((T, H), jnp.float32).at[token].add(rows.astype(jnp.float32), mode="drop").astype(rows.dtype)


def combine_gather_onetake(rows, slots):
    g = jnp.take(rows, slots, axis=0, mode="fill", fill_value=0)
    return jnp.sum(g.astype(jnp.float32), axis=1).astype(rows.dtype)


def main_moe():
    key = jax.random.key(0)
    logits = jax.random.normal(key, (T, E), jnp.float32)
    _, selected = jax.lax.top_k(logits, K)
    token, local_idx, valid = routing(selected.astype(jnp.int32))
    print("valid slots", int(jnp.sum(valid)), "of", PCAP, flush=True)
    x = jax.random.normal(jax.random.key(1), (T, H), jnp.bfloat16)
    rows = jax.random.normal(jax.random.key(2), (PCAP, H), jnp.bfloat16) * valid[:, None].astype(jnp.bfloat16)
    gx = jax.random.normal(jax.random.key(3), (PCAP, H), jnp.bfloat16)
    gout = jax.random.normal(jax.random.key(4), (T, H), jnp.bfloat16)
    nbytes_combine = (T * H + PCAP * H) * 2

    slots_fn = jax.jit(lambda li, v: ep_ring._assignment_slots(li, v, tokens=T, topk=K))
    slots = bench("assignment_slots (inverse map)", slots_fn, local_idx, valid)
    bench(f"dispatch selection top_k ({A} -> {PCAP})", routing, selected.astype(jnp.int32))

    # Forward combine.
    base = bench(
        "combine fwd: scatter-add bf16 (baseline)", jax.jit(combine_scatter), rows, token, nbytes=nbytes_combine
    )
    bench("combine fwd: scatter-add f32", jax.jit(combine_scatter_f32), rows, token, nbytes=nbytes_combine)
    new = bench(
        "combine fwd: gather-sum (new)",
        jax.jit(lambda r, t, v, s: ep_ring._combine_rows(r, t, v, s)),
        rows,
        token,
        valid,
        slots,
        nbytes=nbytes_combine,
    )
    one = bench(
        "combine fwd: gather-sum one take", jax.jit(combine_gather_onetake), rows, slots, nbytes=nbytes_combine
    )
    ref = combine_scatter_f32(rows, token)
    diff("combine scatter bf16 vs f32", base, ref)
    diff("combine gather vs f32 scatter", new, ref)
    diff("combine gather onetake vs f32 scatter", one, ref)

    # Dispatch backward: transpose of the take.
    def base_dispatch(x):
        r = jnp.take(x, token, axis=0)
        return jnp.where(valid[:, None], r, jnp.zeros_like(r))

    base_bwd = jax.jit(lambda x, g: jax.vjp(base_dispatch, x)[1](g)[0])
    new_bwd = jax.jit(lambda x, g: jax.vjp(lambda x: ep_ring._dispatch_rows(x, token, valid, slots), x)[1](g)[0])
    b = bench("dispatch bwd: scatter-add (baseline)", base_bwd, x, gx, nbytes=nbytes_combine)
    n = bench("dispatch bwd: gather-sum (new)", new_bwd, x, gx, nbytes=nbytes_combine)
    ref = jnp.zeros((T, H), jnp.float32).at[token].add(jnp.where(valid[:, None], gx, 0).astype(jnp.float32))
    diff("dispatch bwd baseline vs f32 ref", b, ref)
    diff("dispatch bwd new vs f32 ref", n, ref)

    # Dispatch forward and combine backward (both gathers already).
    bench("dispatch fwd: baseline take", jax.jit(base_dispatch), x)
    bench("dispatch fwd: new", jax.jit(lambda x: ep_ring._dispatch_rows(x, token, valid, slots)), x)
    cb_base = jax.jit(lambda r, g: jax.vjp(lambda r: combine_scatter(r, token), r)[1](g)[0])
    cb_new = jax.jit(lambda r, g: jax.vjp(lambda r: ep_ring._combine_rows(r, token, valid, slots), r)[1](g)[0])
    a1 = bench("combine bwd: baseline", cb_base, rows, gout)
    a2 = bench("combine bwd: new", cb_new, rows, gout)
    diff("combine bwd new vs baseline (valid rows)", a2 * valid[:, None], a1 * valid[:, None])

    # End to end: weighted combine of dispatched rows, value + grads w.r.t. x, rows weight.
    w = jax.random.uniform(jax.random.key(5), (PCAP,), jnp.bfloat16)

    def e2e(use_gather):
        def f(x, w):
            if use_gather:
                xd = ep_ring._dispatch_rows(x, token, valid, slots)
                out = ep_ring._combine_rows(xd * w[:, None], token, valid, slots)
            else:
                xd = base_dispatch(x)
                out = jnp.zeros_like(x).at[token].add(xd * w[:, None], mode="drop")
            return jnp.sum(out.astype(jnp.float32) * gout.astype(jnp.float32))

        return jax.jit(jax.value_and_grad(f, argnums=(0, 1)))

    vb, (gxb, gwb) = bench("e2e fwd+bwd: baseline", e2e(False), x, w)
    vn, (gxn, gwn) = bench("e2e fwd+bwd: new", e2e(True), x, w)
    diff("e2e value", vn, vb)
    diff("e2e dx", gxn, gxb)
    diff("e2e dw", gwn, gwb)


def max_mask_top_k(x, k):
    """Pure-JAX k rounds of argmax over total-order keys."""
    key = _total_order_key(x)
    out = []
    taken = jnp.zeros(x.shape, bool)
    col = jnp.arange(x.shape[1], dtype=jnp.int32)[None, :]
    for _ in range(k):
        masked = jnp.where(taken, jnp.iinfo(jnp.int32).min, key)
        best = jnp.max(masked, axis=1, keepdims=True)
        idx = jnp.min(jnp.where((masked == best) & ~taken, col, x.shape[1]), axis=1)
        out.append(idx)
        taken = taken | (col == idx[:, None])
    idx = jnp.stack(out, axis=1)
    return jnp.take_along_axis(x, idx, axis=1), idx


def main_topk():
    tl = T // EP
    for name, x in [
        ("normal", jax.random.normal(jax.random.key(0), (tl, E), jnp.float32)),
        ("coarse ties", jnp.round(jax.random.normal(jax.random.key(1), (tl, E), jnp.float32) * 2) / 2),
        ("all equal", jnp.zeros((tl, E), jnp.float32)),
        (
            "specials",
            jnp.asarray(
                np.random.default_rng(0).choice(
                    np.array([0.0, -0.0, np.inf, -np.inf, 1.0, -1.0, np.nan, -np.nan, 1e-30], np.float32), size=(tl, E)
                )
            ),
        ),
    ]:
        for k in (K, K + 1):
            rv, ri = jax.jit(lambda x: jax.lax.top_k(x, k))(x)
            ti = jax.jit(lambda x: triton_top_k_indices(x, k))(x)
            mv, mi = jax.jit(lambda x: max_mask_top_k(x, k))(x)
            print(
                f"TOPK-CHECK {name:12s} k={k} triton_identical={bool(jnp.all(ti == ri))} "
                f"mismatch_rows={int(jnp.sum(jnp.any(ti != ri, axis=1)))} "
                f"maxmask_identical={bool(jnp.all(mi == ri))}",
                flush=True,
            )
    x = jax.random.normal(jax.random.key(0), (tl, E), jnp.float32)
    k = K + 1
    nbytes = tl * E * 4
    bench(f"routing top_k {tl}x{E} k={k}: lax.top_k", jax.jit(lambda x: jax.lax.top_k(x, k)), x, nbytes=nbytes)
    bench(
        f"routing top_k {tl}x{E} k={k}: triton",
        jax.jit(lambda x: (lambda i: (jnp.take_along_axis(x, i, 1), i))(triton_top_k_indices(x, k))),
        x,
        nbytes=nbytes,
    )
    bench(f"routing top_k {tl}x{E} k={k}: max-mask jax", jax.jit(lambda x: max_mask_top_k(x, k)), x, nbytes=nbytes)
    # Gradient through values must match lax.top_k.
    g = jax.random.normal(jax.random.key(9), (tl, k))
    gr = jax.grad(lambda x: jnp.sum(jax.lax.top_k(x, k)[0] * g))(x)
    gt = jax.grad(lambda x: jnp.sum(jnp.take_along_axis(x, triton_top_k_indices(x, k), 1) * g))(x)
    diff("top_k value-grad triton vs lax", gt, gr)
    qb = jax.random.normal(jax.random.key(3), (E, tl), jnp.float32)
    phys = max(1, tl * K // E)
    bench(f"qb top_k {E}x{tl} k={phys}: lax.top_k", jax.jit(lambda x: jax.lax.top_k(x, phys)[0]), qb)


if args.section in ("all", "topk"):
    main_topk()
if args.section in ("all", "moe"):
    main_moe()
print("DONE", flush=True)
