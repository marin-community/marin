"""Single-GPU component benchmark of one hero Block without the routed experts.

Runs the real `Block.__call__` (norms, gated norms, attention with FA4, short convs, shared experts)
from two model sources at the hero per-GPU shape (16 x 4096 tokens, d6144). A stand-in replaces the
routed MoE: router, latent down, latent RMSNorm and latent up GEMMs read the same `mlp_in` as the
real MoE input side, so the backward accumulates the same set of input gradients. The step is
`jax.checkpoint`-ed (recompute everything) and differentiated, like a hero layer.

Usage: python block_bench.py --baseline <model_baseline.py> [--iters 10] [--profile DIR] [--check]
Prints per-variant step time, compiled temp memory, and optionally kernel-time summaries.
"""

import argparse
import dataclasses
import glob
import importlib
import importlib.util
import json
import os
import sys
import time

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import AxisType, Mesh, NamedSharding, reshard
from jax.sharding import PartitionSpec as P

from experiments.grug.moe_hero_ep.heuristic import HERO_MODEL

B, S = 16, 4096
AXES = ("replica_dcn", "data", "context", "expert", "model")
BATCH_AXES = ("replica_dcn", "data", "expert")
TOKEN_SPEC = P(BATCH_AXES, None)


def load_model_file(path, name="hero_model_baseline"):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


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
        return reshard(routed.reshape(b, s, d), P(BATCH_AXES, None, None)), {}


def build_block(mod, cfg, key):
    k = jax.random.split(key, 12)
    d, lat, e = cfg.hidden_dim, cfg.latent_dim, cfg.num_experts
    std = cfg.initializer_std
    fsdp = P(mod._FSDP_AXES, "model")
    shared = tuple(
        mod.DenseMLP.init(d, cfg.shared_expert_intermediate_dim, std, key=k[i]) for i in range(cfg.num_shared_experts)
    )
    stand_in = MoeStandIn(
        router=reshard(std * jax.random.normal(k[3], (d, e)), P(None, None)),
        w_latent_down=reshard(std * jax.random.normal(k[4], (d, lat)), fsdp),
        latent_norm=jnp.ones((lat,), jnp.float32),
        w_latent_up=reshard(std * jax.random.normal(k[5], (lat, d)), P("model", mod._FSDP_AXES)),
    )
    attn = mod.CausalSelfAttention.init(cfg, key=k[6])
    # The attention gate is zero-initialized in the model; give it values so its gradient path is live.
    attn = eqx.tree_at(lambda a: a.attn_gate, attn, reshard(std * jax.random.normal(k[7], (d, cfg.num_heads)), P(None, None)))
    return mod.Block(
        rms_attn=mod.RMSNorm.init(d, cfg.layer_norm_eps),
        attn_gated_norm=mod.GatedNorm.init(d, std, key=k[8]),
        attn=attn,
        rms_mlp=mod.RMSNorm.init(d, cfg.layer_norm_eps),
        mlp_gated_norm=mod.GatedNorm.init(d, std, key=k[9]),
        mlp=stand_in,
        shared=shared,
        sconv_attn=mod.ShortConv.init(d, cfg.sconv_kernel) if cfg.sconv and "attn" in cfg.sconv_sites else None,
        sconv_mlp=mod.ShortConv.init(d, cfg.sconv_kernel) if cfg.sconv and "mlp" in cfg.sconv_sites else None,
    )


def to_compute(tree, dtype=jnp.bfloat16):
    return jax.tree.map(lambda a: a.astype(dtype) if eqx.is_inexact_array(a) else a, tree)


def make_step(mod, cfg, dtype=jnp.bfloat16):
    def step(block, x, cotangent, mask):
        def loss(block, x):
            out, _ = jax.checkpoint(lambda blk, h: blk(h, mask, False, False))(to_compute(block, dtype), x)
            return jnp.sum(out.astype(jnp.float32) * cotangent)

        return jax.value_and_grad(loss, argnums=(0, 1))(block, x)

    step.__name__ = f"block_step_{mod.__name__.split('.')[-1]}"
    return jax.jit(step)


def make_mask(mod, cfg):
    seg = reshard(jnp.zeros((B, S), jnp.int32), P(mod._BATCH_AXES, None))
    short_mask = mod.AttentionMask(is_causal=True, sliding_window=cfg.sliding_window, segment_ids=(seg, seg))
    long_mask = mod.AttentionMask(is_causal=True, sliding_window=None, segment_ids=(seg, seg))
    _, valid = mod.fa4_cute_segment_bounds(long_mask, batch_size=B, seq_len=S, sliding_window=None)
    short_lb, _ = mod.fa4_cute_segment_bounds(short_mask, batch_size=B, seq_len=S, sliding_window=cfg.sliding_window)
    return long_mask.with_fa4_bounds(short_lb, valid)


def hlo_shapes(hlo_text):
    shapes = {}
    for line in hlo_text.splitlines():
        line = line.strip()
        if " = " not in line:
            continue
        name, rest = line.split(" = ", 1)
        name = name.replace("ROOT ", "").strip()
        shapes[name] = rest.split(" ")[0][:60] if not rest.startswith("(") else rest[: rest.index(")") + 1][:90]
    return shapes


def kernel_summary(outdir, module_prefix, shapes=None):
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "loop-260930-mfu30"))
    from overlap import load, plane_events  # noqa: E402

    path = sorted(glob.glob(f"{outdir}/**/*.xplane.pb", recursive=True))[-1]
    planes = {p.name: p for p in load(path).planes}
    by = {}
    for lname, name, s, e, st in plane_events(planes["/device:GPU:0"]):
        if not lname.startswith("Stream"):
            continue
        if not (st.get("hlo_module") or "").startswith(module_prefix):
            continue
        hlo = st.get("hlo_op") or name
        base = f"{hlo} {shapes.get(hlo, '') if shapes else ''} {name[:34] if 'nvjet' in name or 'cublas' in name else ''}"
        by.setdefault(base, [0.0, 0])
        by[base][0] += (e - s) * 1e-9
        by[base][1] += 1
    return by


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--profile", default="")
    ap.add_argument("--check", action="store_true", help="compare loss and gradients between variants")
    ap.add_argument("--small", action="store_true", help="tiny shapes and reference attention (CPU smoke test)")
    ap.add_argument("--variant", action="append", default=[], help="extra model source as name=path")
    args = ap.parse_args()
    global B, S

    mesh = Mesh(np.asarray(jax.devices()[:1]).reshape(1, 1, 1, 1, 1), AXES, axis_types=(AxisType.Explicit,) * 5)
    cand = importlib.import_module("experiments.grug.moe_hero_ep.model")
    base = load_model_file(args.baseline)
    extra = [(v.split("=")[0], load_model_file(v.split("=")[1], f"hero_model_{v.split('=')[0]}")) for v in args.variant]
    cfg = HERO_MODEL
    if args.small:
        B, S = 2, 64
        cfg = dataclasses.replace(
            HERO_MODEL, hidden_dim=256, intermediate_dim=128, shared_expert_intermediate_dim=128, latent_dim=128,
            num_experts=16, num_heads=4, num_kv_heads=2, local_kv_heads=2, global_kv_heads=1, head_dim=64,
            max_seq_len=64, sliding_window=32, attention_implementation="reference",
        )
    results = {}
    grads = {}
    with jax.set_mesh(mesh):
        x = reshard(jax.random.normal(jax.random.key(1), (B, S, cfg.hidden_dim), jnp.bfloat16), P(cand._BATCH_AXES, None, None))
        cot = reshard(jax.random.normal(jax.random.key(2), (B, S, cfg.hidden_dim), jnp.float32), P(cand._BATCH_AXES, None, None))
        for name, mod in (("baseline", base), ("candidate", cand), *extra):
            block = build_block(mod, cfg, jax.random.key(0))
            mask = make_mask(mod, cfg)
            step = make_step(mod, cfg)
            compiled = step.lower(block, x, cot, mask).compile()
            mem = compiled.memory_analysis()
            out = step(block, x, cot, mask)
            jax.block_until_ready(out)
            for _ in range(3):
                out = step(block, x, cot, mask)
            jax.block_until_ready(out)
            t0 = time.perf_counter()
            for _ in range(args.iters):
                out = step(block, x, cot, mask)
            jax.block_until_ready(out)
            dt = (time.perf_counter() - t0) / args.iters
            results[name] = dict(step_ms=dt * 1e3, temp_gib=mem.temp_size_in_bytes / 2**30 if mem else None)
            print(f"{name:10s} step {dt*1e3:8.2f} ms  temp {results[name]['temp_gib']:.2f} GiB  loss {float(out[0]):.6e}", flush=True)
            if args.check:
                grads[name] = (float(out[0]), jax.tree.map(np.asarray, out[1]))
            if args.profile:
                pdir = f"{args.profile}/{name}"
                jax.profiler.start_trace(pdir)
                for _ in range(2):
                    out = step(block, x, cot, mask)
                jax.block_until_ready(out)
                jax.profiler.stop_trace()
                by = kernel_summary(pdir, "jit_block_step", hlo_shapes(compiled.as_text()))
                tot = sum(v[0] for v in by.values()) / 2
                results[name]["kernel_ms"] = tot
                results[name]["kernels"] = {k: (round(v[0] / 2, 3), v[1] // 2) for k, v in sorted(by.items(), key=lambda kv: -kv[1][0])}
                print(f"{name:10s} kernel sum {tot:.2f} ms/step; top:", flush=True)
                for k, (ms, n) in list(results[name]["kernels"].items())[:70]:
                    print(f"    {ms:8.3f} ms  n={n:3d}  {k}", flush=True)
            del out
    if args.check and args.small:
        # fp32 reference: the baseline code with fp32 parameters and activations (FA4 is bf16-only,
        # so only the small reference-attention configuration runs it).
        with jax.set_mesh(mesh):
            block = build_block(base, cfg, jax.random.key(0))
            mask = make_mask(base, cfg)
            ref = make_step(base, cfg, jnp.float32)(block, x.astype(jnp.float32), cot, mask)
            gr = (float(ref[0]), jax.tree.map(np.asarray, ref[1]))
            del ref
        lr_, gref = gr
        print(f"loss fp32 reference {lr_:.8e}")
    if args.check and args.small:
        for (pa, a), (_, c), (_, r) in zip(
            jax.tree_util.tree_leaves_with_path(grads["baseline"][1]),
            jax.tree_util.tree_leaves_with_path(grads["candidate"][1]),
            jax.tree_util.tree_leaves_with_path(gref),
        ):
            a32, c32, r32 = a.astype(np.float32), c.astype(np.float32), r.astype(np.float32)
            rr = float(np.sqrt(np.mean(r32**2))) or 1.0
            eb = float(np.sqrt(np.mean((a32 - r32) ** 2))) / rr
            ec = float(np.sqrt(np.mean((c32 - r32) ** 2))) / rr
            print(f"  vs fp32 {jax.tree_util.keystr(pa)[:50]:50s} baseline rel-rms err {eb:.3e}  candidate {ec:.3e}")
    if args.check:
        lb, gb = grads["baseline"]
        lc, gc = grads["candidate"]
        print(f"loss baseline {lb:.8e} candidate {lc:.8e} rel diff {abs(lc - lb) / abs(lb):.3e}")
        # Block trees differ only in code, not structure; compare leaves pairwise.
        for (pa, a), (_, c) in zip(jax.tree_util.tree_leaves_with_path(gb), jax.tree_util.tree_leaves_with_path(gc)):
            a32, c32 = a.astype(np.float32), c.astype(np.float32)
            rms = float(np.sqrt(np.mean(a32**2))) or 1.0
            print(f"  grad {jax.tree_util.keystr(pa)[:50]:50s} rms {rms:.3e} max|d| {np.max(np.abs(a32 - c32)):.3e} rel-rms {np.sqrt(np.mean((a32 - c32) ** 2)) / rms:.3e}")
    print("RESULT", json.dumps(results))


if __name__ == "__main__":
    main()
