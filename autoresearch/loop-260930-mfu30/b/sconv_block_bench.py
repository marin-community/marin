# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Single-GPU hero Block benchmark: Pallas vs Triton short conv.

Adapted from C's `autoresearch/loop-260930-mfu30-gemm/block_bench.py` (branch research/mcwitt/mfu30-gemm):
the real `Block.__call__` (norms, gated norms, FA4 attention, the three short convs, shared experts) at the
hero per-GPU shape (16 x 4096 tokens, d6144), with a stand-in for the routed MoE, rematerialized and
differentiated like a hero layer. The two variants differ only in `sconv_implementation`. Prints the step
time per variant (interleaved repeats), the short-conv kernel time per step from a profile, and the gradient
differences: everything but the short-conv weight gradients should be bitwise equal, since the Triton
forward and dx match the reference's rounding and its dw sums fp32 partials in a different order.

Usage (one GB200): python autoresearch/loop-260930-mfu30/b/sconv_block_bench.py [--iters 30] [--reps 3]
    [--profile DIR] [--segments packed|none]
"""

import argparse
import dataclasses
import glob
import json
import os
import sys
import time

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import AxisType, Mesh, reshard
from jax.sharding import PartitionSpec as P

from experiments.grug.moe_hero_ep import model as hero
from experiments.grug.moe_hero_ep.heuristic import HERO_MODEL

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from overlap import load, plane_events

B, S = 16, 4096
AXES = ("replica_dcn", "data", "context", "expert", "model")
TOKEN_SPEC = P(hero._BATCH_AXES, None)
VARIANTS = ("pallas_gpu", "triton_gpu")


class MoeStandIn(eqx.Module):
    """Input and output side of the routed MoE without the experts or the transport."""

    router: jax.Array
    w_latent_down: jax.Array
    latent_norm: jax.Array
    w_latent_up: jax.Array

    def __call__(self, x, token_valid):
        b, s, d = x.shape
        x_flat = reshard(x.reshape(b * s, d), TOKEN_SPEC)
        logits = jnp.einsum("td,de->te", x_flat, self.router).astype(jnp.float32)
        probs = jax.nn.softmax(logits, axis=-1)
        latent = jnp.einsum("td,dl->tl", x_flat, self.w_latent_down)
        lf = latent.astype(jnp.float32)
        latent = (lf * jax.lax.rsqrt(jnp.mean(lf * lf, axis=-1, keepdims=True) + 1e-5) * self.latent_norm).astype(
            x.dtype
        )
        routed = jnp.einsum("tl,ld->td", latent, self.w_latent_up, out_sharding=TOKEN_SPEC)
        routed = routed * (1 + probs[:, :1]).astype(routed.dtype)
        return reshard(routed.reshape(b, s, d), P(hero._BATCH_AXES, None, None)), {}


def build_block(cfg, key):
    k = jax.random.split(key, 12)
    d, lat, e = cfg.hidden_dim, cfg.latent_dim, cfg.num_experts
    std = cfg.initializer_std
    fsdp = P(hero._FSDP_AXES, "model")
    shared = tuple(
        hero.DenseMLP.init(d, cfg.shared_expert_intermediate_dim, std, key=k[i]) for i in range(cfg.num_shared_experts)
    )
    stand_in = MoeStandIn(
        router=reshard(std * jax.random.normal(k[3], (d, e)), P(None, None)),
        w_latent_down=reshard(std * jax.random.normal(k[4], (d, lat)), fsdp),
        latent_norm=jnp.ones((lat,), jnp.float32),
        w_latent_up=reshard(std * jax.random.normal(k[5], (lat, d)), P("model", hero._FSDP_AXES)),
    )
    attn = hero.CausalSelfAttention.init(cfg, key=k[6])
    attn = eqx.tree_at(
        lambda a: a.attn_gate, attn, reshard(std * jax.random.normal(k[7], (d, cfg.num_heads)), P(None, None))
    )
    # Non-trivial short-conv taps, so every tap's gradient path carries signal.
    sconv_key = jax.random.split(k[10], 3)

    def sconv(channels, key):
        conv = hero.ShortConv.init(channels, cfg.sconv_kernel, cfg.sconv_implementation)
        return eqx.tree_at(lambda c: c.weight, conv, 0.5 * jax.random.normal(key, conv.weight.shape))

    attn = eqx.tree_at(lambda a: a.sconv_k, attn, sconv(attn.sconv_k.weight.shape[1], sconv_key[0]))
    return hero.Block(
        rms_attn=hero.RMSNorm.init(d, cfg.layer_norm_eps),
        attn_gated_norm=hero.GatedNorm.init(d, std, key=k[8]),
        attn=attn,
        rms_mlp=hero.RMSNorm.init(d, cfg.layer_norm_eps),
        mlp_gated_norm=hero.GatedNorm.init(d, std, key=k[9]),
        mlp=stand_in,
        shared=shared,
        sconv_attn=sconv(d, sconv_key[1]),
        sconv_mlp=sconv(d, sconv_key[2]),
    )


def to_compute(tree, dtype=jnp.bfloat16):
    return jax.tree.map(lambda a: a.astype(dtype) if eqx.is_inexact_array(a) else a, tree)


def make_step():
    def step(block, x, cotangent, mask):
        def loss(block, x):
            out, _ = jax.checkpoint(lambda blk, h: blk(h, mask, False, False))(to_compute(block), x)
            return jnp.sum(out.astype(jnp.float32) * cotangent)

        return jax.value_and_grad(loss, argnums=(0, 1))(block, x)

    return jax.jit(step)


def make_segments(kind, rng):
    if kind == "none":
        return np.zeros((B, S), np.int32)
    seg = np.zeros((B, S), np.int32)
    for b in range(B):
        cuts = np.sort(rng.choice(np.arange(1, S), size=6, replace=False))
        seg[b] = np.searchsorted(cuts, np.arange(S), side="right")
    return seg


def make_mask(cfg, seg_np):
    seg = reshard(jnp.asarray(seg_np), P(hero._BATCH_AXES, None))
    short_mask = hero.AttentionMask(is_causal=True, sliding_window=cfg.sliding_window, segment_ids=(seg, seg))
    long_mask = hero.AttentionMask(is_causal=True, sliding_window=None, segment_ids=(seg, seg))
    _, valid = hero.fa4_cute_segment_bounds(long_mask, batch_size=B, seq_len=S, sliding_window=None)
    short_lb, _ = hero.fa4_cute_segment_bounds(short_mask, batch_size=B, seq_len=S, sliding_window=cfg.sliding_window)
    return long_mask.with_fa4_bounds(short_lb, valid)


def sconv_kernel_ms(outdir, steps):
    """Device time per step of the short-conv kernels (Pallas `short_conv*` / Triton `_short_conv*`)."""
    path = sorted(glob.glob(f"{outdir}/**/*.xplane.pb", recursive=True))[-1]
    planes = {p.name: p for p in load(path).planes}
    total = 0.0
    step_total = 0.0
    names = {}
    for lname, name, s, e, st in plane_events(planes["/device:GPU:0"]):
        if not lname.startswith("Stream"):
            continue
        if not (st.get("hlo_module") or "").startswith("jit_step"):
            continue
        step_total += (e - s) * 1e-6
        if "short_conv" in name:
            total += (e - s) * 1e-6
            names.setdefault(name[:60], [0.0, 0])
            names[name[:60]][0] += (e - s) * 1e-6 / steps
            names[name[:60]][1] += 1
    return total / steps, step_total / steps, {k: (round(v[0], 3), v[1] // steps) for k, v in names.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--profile", default="")
    ap.add_argument("--segments", default="packed", choices=("packed", "none"))
    args = ap.parse_args()

    mesh = Mesh(np.asarray(jax.devices()[:1]).reshape(1, 1, 1, 1, 1), AXES, axis_types=(AxisType.Explicit,) * 5)
    seg_np = make_segments(args.segments, np.random.default_rng(0))
    results = {v: dict(step_ms=[]) for v in VARIANTS}
    compiled = {}
    grads = {}
    with jax.set_mesh(mesh):
        x = reshard(
            jax.random.normal(jax.random.key(1), (B, S, HERO_MODEL.hidden_dim), jnp.bfloat16),
            P(hero._BATCH_AXES, None, None),
        )
        cot = reshard(
            jax.random.normal(jax.random.key(2), (B, S, HERO_MODEL.hidden_dim), jnp.float32),
            P(hero._BATCH_AXES, None, None),
        )
        for impl in VARIANTS:
            cfg = dataclasses.replace(HERO_MODEL, sconv_implementation=impl)
            block = build_block(cfg, jax.random.key(0))
            mask = make_mask(cfg, seg_np)
            step = make_step()
            exe = step.lower(block, x, cot, mask).compile()
            mem = exe.memory_analysis()
            results[impl]["temp_gib"] = mem.temp_size_in_bytes / 2**30 if mem else None
            out = exe(block, x, cot, mask)
            grads[impl] = (float(out[0]), jax.tree.map(np.asarray, out[1]))
            compiled[impl] = (exe, block, mask)
            del out
        for _ in range(args.reps):
            for impl in VARIANTS:
                exe, block, mask = compiled[impl]
                for _ in range(3):
                    out = exe(block, x, cot, mask)
                jax.block_until_ready(out)
                t0 = time.perf_counter()
                for _ in range(args.iters):
                    out = exe(block, x, cot, mask)
                jax.block_until_ready(out)
                results[impl]["step_ms"].append((time.perf_counter() - t0) / args.iters * 1e3)
                print(f"{impl:11s} step {results[impl]['step_ms'][-1]:8.3f} ms", flush=True)
                del out
        if args.profile:
            for impl in VARIANTS:
                exe, block, mask = compiled[impl]
                pdir = f"{args.profile}/{impl}"
                jax.profiler.start_trace(pdir)
                for _ in range(4):
                    out = exe(block, x, cot, mask)
                jax.block_until_ready(out)
                jax.profiler.stop_trace()
                del out
                sconv_ms, kernel_ms, names = sconv_kernel_ms(pdir, 4)
                results[impl].update(sconv_kernel_ms=sconv_ms, kernel_sum_ms=kernel_ms, sconv_kernels=names)
                print(f"{impl:11s} short-conv kernels {sconv_ms:.3f} ms/step of {kernel_ms:.3f}: {names}", flush=True)
    for impl in VARIANTS:
        results[impl]["step_ms_median"] = float(np.median(results[impl]["step_ms"]))
    (la, ga), (lb, gb) = grads["pallas_gpu"], grads["triton_gpu"]
    diffs = {}
    for (path, a), (_, b) in zip(jax.tree_util.tree_leaves_with_path(ga), jax.tree_util.tree_leaves_with_path(gb)):
        a32, b32 = a.astype(np.float32), b.astype(np.float32)
        if np.array_equal(a32, b32):
            continue
        rms = float(np.sqrt(np.mean(a32**2))) or 1.0
        diffs[jax.tree_util.keystr(path)] = float(np.sqrt(np.mean((a32 - b32) ** 2)) / rms)
    print(f"loss pallas {la:.8e} triton {lb:.8e} equal {la == lb}")
    print("gradient leaves that differ (rel-rms):", json.dumps(diffs))
    print("RESULT", json.dumps(dict(results=results, loss_equal=la == lb, grad_diffs=diffs)))


if __name__ == "__main__":
    main()
