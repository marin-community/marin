# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import argparse
import time
from pathlib import Path
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from levanter.kernels.pallas.moe.fused_moe import (
    MoEBlockSizes,
    _fused_moe_reference,
    fused_moe_fused_routing,
    fused_moe_staged,
)


def _time_fn(fn: Callable, *args, warmup: int = 2, iters: int = 5) -> float:
    compiled = jax.jit(fn)
    compiled(*args).block_until_ready()
    for _ in range(warmup):
        compiled(*args).block_until_ready()
    start = time.perf_counter()
    for _ in range(iters):
        compiled(*args).block_until_ready()
    return (time.perf_counter() - start) / iters


def _flops_per_token(hidden: int, mlp_dim: int, topk: int) -> float:
    return 6.0 * hidden * mlp_dim * topk


def _route_topk(
    gating_logits: jax.Array,
    *,
    topk: int,
    renormalize: bool,
    dtype: jnp.dtype,
) -> tuple[jax.Array, jax.Array]:
    scores = jax.nn.softmax(gating_logits, axis=-1)
    topk_weights, topk_idx = jax.lax.top_k(scores, topk)
    if renormalize:
        topk_weights = topk_weights / jnp.sum(topk_weights, axis=-1, keepdims=True)
    return topk_idx.astype(jnp.int32), topk_weights.astype(dtype)


def _build_ref_dp(
    mesh: Mesh,
    *,
    topk: int,
    renormalize: bool,
    dtype: jnp.dtype,
    block_sizes: MoEBlockSizes,
):
    @jax.shard_map(
        mesh=mesh,
        in_specs=(P("data", None), P("data", None), P(), P(), P()),
        out_specs=P("data", None),
        check_vma=False,
    )
    def ref_dp(x_in, gating_in, w1_in, w2_in, w3_in):
        idx_in, w_in = _route_topk(
            gating_in,
            topk=topk,
            renormalize=renormalize,
            dtype=dtype,
        )
        return _fused_moe_reference(x_in, idx_in, w_in, w1_in, w2_in, w3_in, block_sizes=block_sizes)

    return jax.jit(ref_dp)


def _build_staged_dp(
    mesh: Mesh,
    *,
    topk: int,
    renormalize: bool,
    dtype: jnp.dtype,
    block_sizes: MoEBlockSizes,
    parallel: bool,
):
    @jax.shard_map(
        mesh=mesh,
        in_specs=(P("data", None), P("data", None), P(), P(), P()),
        out_specs=P("data", None),
        check_vma=False,
    )
    def staged_dp(x_in, gating_in, w1_in, w2_in, w3_in):
        idx_in, w_in = _route_topk(
            gating_in,
            topk=topk,
            renormalize=renormalize,
            dtype=dtype,
        )
        return fused_moe_staged(
            x_in,
            idx_in,
            w_in,
            w1_in,
            w2_in,
            w3_in,
            block_sizes=block_sizes,
            parallel=parallel,
        )

    return jax.jit(staged_dp)


def _build_fused_ep(
    mesh: Mesh,
    *,
    topk: int,
    renormalize: bool,
    act_fn: str,
    scoring_fn: str,
    ep_axis_name: str,
):
    def _fn(x_in, gating_in, w1_in, w2_in, w3_in):
        return fused_moe_fused_routing(
            x_in,
            gating_in,
            w1_in,
            w2_in,
            w3_in,
            top_k=topk,
            renormalize_topk_logits=renormalize,
            act_fn=act_fn,
            scoring_fn=scoring_fn,
            mesh=mesh,
            ep_axis_name=ep_axis_name,
        )

    return jax.jit(_fn)


def _profile_run(
    label: str,
    trace_dir: Path,
    fn: Callable,
    *args,
    steps: int,
) -> None:
    trace_path = trace_dir / label
    trace_path.mkdir(parents=True, exist_ok=True)
    with jax.profiler.trace(str(trace_path), create_perfetto_link=True):
        for _ in range(steps):
            fn(*args).block_until_ready()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=32768)
    parser.add_argument("--hidden", type=int, default=2048)
    parser.add_argument("--mlp-dim", type=int, default=4096)
    parser.add_argument("--experts", type=int, default=8)
    parser.add_argument("--topk", type=int, default=4)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--renormalize-topk", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--trace-dir", type=str, default=".profiles/moe_mlp_profile")
    parser.add_argument("--trace-steps", type=int, default=3)
    parser.add_argument("--ep-axis-name", type=str, default="model")
    parser.add_argument("--staged-dp", action="store_true")
    parser.add_argument("--staged-parallel", action="store_true")
    args = parser.parse_args()

    dtype = jnp.dtype(args.dtype)
    tokens = args.tokens
    hidden = args.hidden
    mlp_dim = args.mlp_dim
    experts = args.experts
    topk = args.topk

    if tokens % jax.device_count() != 0:
        raise ValueError("tokens must be divisible by device_count for DP/EP runs.")
    if experts % jax.device_count() != 0:
        raise ValueError("experts must be divisible by device_count for EP runs.")

    key = jax.random.PRNGKey(args.seed)
    key_x, key_w1, key_w2, key_w3, key_gate = jax.random.split(key, 5)
    x = jax.random.normal(key_x, (tokens, hidden), dtype=dtype)
    w1 = jax.random.normal(key_w1, (experts, hidden, mlp_dim), dtype=dtype)
    w2 = jax.random.normal(key_w2, (experts, mlp_dim, hidden), dtype=dtype)
    w3 = jax.random.normal(key_w3, (experts, hidden, mlp_dim), dtype=dtype)
    gating_logits = jax.random.normal(key_gate, (tokens, experts), dtype=dtype)

    dp_mesh = Mesh(np.array(jax.devices()), ("data",))
    ep_mesh = Mesh(np.array(jax.devices()).reshape(1, -1), ("data", args.ep_axis_name))

    x_dp = jax.device_put(x, NamedSharding(dp_mesh, P("data", None)))
    gate_dp = jax.device_put(gating_logits, NamedSharding(dp_mesh, P("data", None)))

    x_ep = jax.device_put(x, NamedSharding(ep_mesh, P(args.ep_axis_name, None)))
    gate_ep = jax.device_put(gating_logits, NamedSharding(ep_mesh, P(args.ep_axis_name, None)))
    w1_ep = jax.device_put(w1, NamedSharding(ep_mesh, P(args.ep_axis_name, None, None)))
    w2_ep = jax.device_put(w2, NamedSharding(ep_mesh, P(args.ep_axis_name, None, None)))
    w3_ep = jax.device_put(w3, NamedSharding(ep_mesh, P(args.ep_axis_name, None, None)))

    block_sizes = MoEBlockSizes.get_default()

    ref_dp = _build_ref_dp(
        dp_mesh,
        topk=topk,
        renormalize=args.renormalize_topk,
        dtype=dtype,
        block_sizes=block_sizes,
    )
    staged_dp = None
    if args.staged_dp:
        staged_dp = _build_staged_dp(
            dp_mesh,
            topk=topk,
            renormalize=args.renormalize_topk,
            dtype=dtype,
            block_sizes=block_sizes,
            parallel=args.staged_parallel,
        )
    fused_ep = _build_fused_ep(
        ep_mesh,
        topk=topk,
        renormalize=args.renormalize_topk,
        act_fn="silu",
        scoring_fn="softmax",
        ep_axis_name=args.ep_axis_name,
    )

    dt_ref_dp = _time_fn(ref_dp, x_dp, gate_dp, w1, w2, w3, warmup=args.warmup, iters=args.iters)
    dt_fused_ep = _time_fn(fused_ep, x_ep, gate_ep, w1_ep, w2_ep, w3_ep, warmup=args.warmup, iters=args.iters)

    flops = _flops_per_token(hidden, mlp_dim, topk) * tokens
    dp_devices = jax.device_count()
    print("ref_dp.time_s", dt_ref_dp)
    print("ref_dp.tflops", flops / dt_ref_dp / 1e12)
    print("ref_dp.tflops_per_device", flops / dt_ref_dp / 1e12 / dp_devices)
    if staged_dp is not None:
        dt_staged_dp = _time_fn(staged_dp, x_dp, gate_dp, w1, w2, w3, warmup=args.warmup, iters=args.iters)
        print("staged_dp.time_s", dt_staged_dp)
        print("staged_dp.tflops", flops / dt_staged_dp / 1e12)
        print("staged_dp.tflops_per_device", flops / dt_staged_dp / 1e12 / dp_devices)
    print("fused_ep.time_s", dt_fused_ep)
    print("fused_ep.tflops", flops / dt_fused_ep / 1e12)
    print("fused_ep.tflops_per_device", flops / dt_fused_ep / 1e12 / dp_devices)

    if args.profile:
        trace_dir = Path(args.trace_dir)
        trace_dir.mkdir(parents=True, exist_ok=True)
        _profile_run("ref_dp", trace_dir, ref_dp, x_dp, gate_dp, w1, w2, w3, steps=args.trace_steps)
        if staged_dp is not None:
            _profile_run("staged_dp", trace_dir, staged_dp, x_dp, gate_dp, w1, w2, w3, steps=args.trace_steps)
        _profile_run("fused_ep", trace_dir, fused_ep, x_ep, gate_ep, w1_ep, w2_ep, w3_ep, steps=args.trace_steps)


if __name__ == "__main__":
    main()
