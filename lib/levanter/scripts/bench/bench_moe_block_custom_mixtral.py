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

from haliax.nn.linear import gmm_sharded
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


def _route_topk(
    router_probs: jax.Array,
    *,
    topk: int,
) -> tuple[jax.Array, jax.Array]:
    topk_weights, topk_idx = jax.lax.top_k(router_probs, topk)
    topk_weights = topk_weights / jnp.sum(topk_weights, axis=-1, keepdims=True)
    return topk_idx.astype(jnp.int32), topk_weights


def _ragged_dot(
    lhs: jax.Array,
    rhs: jax.Array,
    group_sizes: jax.Array,
) -> jax.Array:
    dim_numbers = jax.lax.RaggedDotDimensionNumbers(
        dot_dimension_numbers=(((1,), (1,)), ((), ())),
        lhs_ragged_dimensions=(0,),
        rhs_group_dimensions=(0,),
    )
    return jax.lax.ragged_dot_general(
        lhs=lhs,
        rhs=rhs,
        group_sizes=group_sizes,
        ragged_dot_dimension_numbers=dim_numbers,
    )


def _mixtral_moe_block(
    x: jax.Array,
    w_gate: jax.Array,
    w1: jax.Array,
    w2: jax.Array,
    w3: jax.Array,
    *,
    topk: int,
    use_gmm: bool,
) -> jax.Array:
    # Gate + route.
    router_logits = jax.lax.dot_general(
        x,
        w_gate,
        (((1,), (0,)), ((), ())),
        preferred_element_type=jnp.float32,
    )
    router_probs = jax.nn.softmax(router_logits, axis=-1)
    topk_idx, topk_weights = _route_topk(router_probs, topk=topk)

    # Permute: group by expert.
    tokens, hidden = x.shape
    topk_idx_flat = topk_idx.reshape(-1)
    sort_idx = jnp.argsort(topk_idx_flat, axis=0)
    x_repeat_sort = jnp.take(x, sort_idx // topk, axis=0)
    group_sizes = jnp.bincount(topk_idx_flat, length=w1.shape[0]).astype(jnp.int32)

    # Expert MLP via ragged dot or GMM.
    if use_gmm:
        w1_out = gmm_sharded(x_repeat_sort, w1, group_sizes)
        w3_out = gmm_sharded(x_repeat_sort, w3, group_sizes)
    else:
        w1_out = _ragged_dot(x_repeat_sort, w1, group_sizes)
        w3_out = _ragged_dot(x_repeat_sort, w3, group_sizes)
    gated = jax.nn.silu(w1_out) * w3_out
    if use_gmm:
        out_repeat_sort = gmm_sharded(gated, w2, group_sizes)
    else:
        out_repeat_sort = _ragged_dot(gated, w2, group_sizes)

    # Unpermute and combine top-k.
    inv_sort_idx = jnp.argsort(sort_idx, axis=0)
    out_repeat = jnp.take(out_repeat_sort, inv_sort_idx, axis=0)
    out_repeat = out_repeat.reshape(tokens, topk, hidden)
    out = jnp.sum(out_repeat * topk_weights[..., None], axis=1)
    return out.astype(x.dtype)


def _build_mixtral_dp(
    mesh: Mesh,
    *,
    topk: int,
    use_gmm: bool,
):
    @jax.shard_map(
        mesh=mesh,
        in_specs=(P("data", None), P(), P(), P(), P()),
        out_specs=P("data", None),
        check_vma=False,
    )
    def mixtral_dp(x_in, w_gate_in, w1_in, w2_in, w3_in):
        return _mixtral_moe_block(x_in, w_gate_in, w1_in, w2_in, w3_in, topk=topk, use_gmm=use_gmm)

    return jax.jit(mixtral_dp)


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
        in_specs=(P("data", None), P(), P(), P(), P()),
        out_specs=P("data", None),
        check_vma=False,
    )
    def ref_dp(x_in, w_gate_in, w1_in, w2_in, w3_in):
        router_logits = jax.lax.dot_general(
            x_in,
            w_gate_in,
            (((1,), (0,)), ((), ())),
            preferred_element_type=jnp.float32,
        )
        router_probs = jax.nn.softmax(router_logits, axis=-1)
        topk_idx, topk_weights = _route_topk(router_probs, topk=topk)
        if renormalize:
            topk_weights = topk_weights / jnp.sum(topk_weights, axis=-1, keepdims=True)
        return _fused_moe_reference(
            x_in,
            topk_idx,
            topk_weights.astype(dtype),
            w1_in,
            w2_in,
            w3_in,
            block_sizes=block_sizes,
        )

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
        in_specs=(P("data", None), P(), P(), P(), P()),
        out_specs=P("data", None),
        check_vma=False,
    )
    def staged_dp(x_in, w_gate_in, w1_in, w2_in, w3_in):
        router_logits = jax.lax.dot_general(
            x_in,
            w_gate_in,
            (((1,), (0,)), ((), ())),
            preferred_element_type=jnp.float32,
        )
        router_probs = jax.nn.softmax(router_logits, axis=-1)
        topk_idx, topk_weights = _route_topk(router_probs, topk=topk)
        if renormalize:
            topk_weights = topk_weights / jnp.sum(topk_weights, axis=-1, keepdims=True)
        return fused_moe_staged(
            x_in,
            topk_idx,
            topk_weights.astype(dtype),
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
    ep_axis_name: str,
):
    def _fn(x_in, w_gate_in, w1_in, w2_in, w3_in):
        router_logits = jax.lax.dot_general(
            x_in,
            w_gate_in,
            (((1,), (0,)), ((), ())),
            preferred_element_type=jnp.float32,
        )
        router_logits = router_logits.astype(x_in.dtype)
        return fused_moe_fused_routing(
            x_in,
            router_logits,
            w1_in,
            w2_in,
            w3_in,
            top_k=topk,
            renormalize_topk_logits=False,
            act_fn="silu",
            scoring_fn="softmax",
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
    # Compile outside the trace so the profile reflects execution time.
    fn(*args).block_until_ready()
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
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--trace-dir", type=str, default=".profiles/moe_block_profile")
    parser.add_argument("--trace-steps", type=int, default=3)
    parser.add_argument("--ep-axis-name", type=str, default="model")
    parser.add_argument("--use-gmm", action="store_true")
    parser.add_argument("--run-mixtral", action="store_true", default=True)
    parser.add_argument("--no-run-mixtral", action="store_false", dest="run_mixtral")
    parser.add_argument("--run-ref-dp", action="store_true")
    parser.add_argument("--run-staged-dp", action="store_true")
    parser.add_argument("--staged-parallel", action="store_true")
    parser.add_argument("--run-fused-ep", action="store_true")
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
    key_x, key_gate, key_w1, key_w2, key_w3 = jax.random.split(key, 5)
    x = jax.random.normal(key_x, (tokens, hidden), dtype=dtype)
    w_gate = jax.random.normal(key_gate, (hidden, experts), dtype=dtype)
    w1 = jax.random.normal(key_w1, (experts, hidden, mlp_dim), dtype=dtype)
    w2 = jax.random.normal(key_w2, (experts, mlp_dim, hidden), dtype=dtype)
    w3 = jax.random.normal(key_w3, (experts, hidden, mlp_dim), dtype=dtype)

    dp_mesh = Mesh(np.array(jax.devices()), ("data",))
    ep_mesh = Mesh(np.array(jax.devices()).reshape(1, -1), ("data", args.ep_axis_name))

    x_dp = jax.device_put(x, NamedSharding(dp_mesh, P("data", None)))
    x_ep = jax.device_put(x, NamedSharding(ep_mesh, P(args.ep_axis_name, None)))
    w_gate_rep = jax.device_put(w_gate, NamedSharding(ep_mesh, P(None, None)))
    w1_ep = jax.device_put(w1, NamedSharding(ep_mesh, P(args.ep_axis_name, None, None)))
    w2_ep = jax.device_put(w2, NamedSharding(ep_mesh, P(args.ep_axis_name, None, None)))
    w3_ep = jax.device_put(w3, NamedSharding(ep_mesh, P(args.ep_axis_name, None, None)))

    block_sizes = MoEBlockSizes.get_default()
    mixtral_dp = None
    ref_dp = None
    staged_dp = None
    fused_ep = None

    if args.run_mixtral:
        mixtral_dp = _build_mixtral_dp(dp_mesh, topk=topk, use_gmm=args.use_gmm)
    if args.run_ref_dp:
        ref_dp = _build_ref_dp(
            dp_mesh,
            topk=topk,
            renormalize=False,
            dtype=dtype,
            block_sizes=block_sizes,
        )
    if args.run_staged_dp:
        staged_dp = _build_staged_dp(
            dp_mesh,
            topk=topk,
            renormalize=False,
            dtype=dtype,
            block_sizes=block_sizes,
            parallel=args.staged_parallel,
        )
    if args.run_fused_ep:
        fused_ep = _build_fused_ep(ep_mesh, topk=topk, ep_axis_name=args.ep_axis_name)

    dt_mixtral_dp = None
    dt_ref_dp = None
    dt_staged_dp = None
    dt_fused_ep = None

    if mixtral_dp is not None:
        dt_mixtral_dp = _time_fn(mixtral_dp, x_dp, w_gate, w1, w2, w3, warmup=args.warmup, iters=args.iters)
    if ref_dp is not None:
        dt_ref_dp = _time_fn(ref_dp, x_dp, w_gate, w1, w2, w3, warmup=args.warmup, iters=args.iters)
    if staged_dp is not None:
        dt_staged_dp = _time_fn(staged_dp, x_dp, w_gate, w1, w2, w3, warmup=args.warmup, iters=args.iters)
    if fused_ep is not None:
        dt_fused_ep = _time_fn(fused_ep, x_ep, w_gate_rep, w1_ep, w2_ep, w3_ep, warmup=args.warmup, iters=args.iters)

    # include gate flops (H*E) + MLP flops
    flops_per_token = 2.0 * hidden * experts + 6.0 * hidden * mlp_dim * topk
    flops = flops_per_token * tokens
    devices = jax.device_count()
    if dt_mixtral_dp is not None:
        print("mixtral_dp.time_s", dt_mixtral_dp)
        print("mixtral_dp.tflops", flops / dt_mixtral_dp / 1e12)
        print("mixtral_dp.tflops_per_device", flops / dt_mixtral_dp / 1e12 / devices)
    if dt_ref_dp is not None:
        print("ref_dp.time_s", dt_ref_dp)
        print("ref_dp.tflops", flops / dt_ref_dp / 1e12)
        print("ref_dp.tflops_per_device", flops / dt_ref_dp / 1e12 / devices)
    if dt_staged_dp is not None:
        print("staged_dp.time_s", dt_staged_dp)
        print("staged_dp.tflops", flops / dt_staged_dp / 1e12)
        print("staged_dp.tflops_per_device", flops / dt_staged_dp / 1e12 / devices)
    if dt_fused_ep is not None:
        print("fused_ep.time_s", dt_fused_ep)
        print("fused_ep.tflops", flops / dt_fused_ep / 1e12)
        print("fused_ep.tflops_per_device", flops / dt_fused_ep / 1e12 / devices)

    if args.profile:
        trace_dir = Path(args.trace_dir)
        trace_dir.mkdir(parents=True, exist_ok=True)
        if mixtral_dp is not None:
            _profile_run("mixtral_dp", trace_dir, mixtral_dp, x_dp, w_gate, w1, w2, w3, steps=args.trace_steps)
        if ref_dp is not None:
            _profile_run("ref_dp", trace_dir, ref_dp, x_dp, w_gate, w1, w2, w3, steps=args.trace_steps)
        if staged_dp is not None:
            _profile_run("staged_dp", trace_dir, staged_dp, x_dp, w_gate, w1, w2, w3, steps=args.trace_steps)
        if fused_ep is not None:
            _profile_run(
                "fused_ep", trace_dir, fused_ep, x_ep, w_gate_rep, w1_ep, w2_ep, w3_ep, steps=args.trace_steps
            )


if __name__ == "__main__":
    main()
