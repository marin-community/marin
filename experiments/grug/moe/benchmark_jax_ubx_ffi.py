# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""One-process EP8 correctness gate for the raw JAX NCCL UB-X FFI."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import asdict
from functools import partial
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from levanter.kernels.ubx import (
    UbxRuntimeConfig,
    combine_push3_bf16,
    dispatch_topk_bf16,
    ensure_local_runtime,
    pool_layout,
    shutdown_local_runtime,
)

from experiments.grug.moe.benchmark_nccl_ubx import (
    BenchmarkConfig,
    RoutePlan,
    build_route_plan,
    reference_maps,
    route_plan_summary,
)

EP_SIZE = 8
RELATIVE_L2_LIMIT = 0.002


def _stack_maps(
    config: BenchmarkConfig,
) -> tuple[RoutePlan, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    plan = build_route_plan(config)
    maps = [reference_maps(plan, config, rank) for rank in range(EP_SIZE)]
    dispatch_expert = np.concatenate([value.dispatch_topk_expert for value in maps], axis=0)
    dispatch_slot = np.concatenate([value.dispatch_topk_slot for value in maps], axis=0)
    inverse_map = np.concatenate([value.inverse_map for value in maps], axis=0)
    topk_idx = np.concatenate([value.topk_idx for value in maps], axis=0)
    dispatch_valid = inverse_map[:, 3].astype(np.bool_)
    gate_weights = plan.gate_weights_by_expert.astype(np.float32)
    return plan, dispatch_expert, dispatch_slot, dispatch_valid, inverse_map, topk_idx, gate_weights


def _local_gate(
    x_local: jax.Array,
    dispatch_expert_local: jax.Array,
    dispatch_slot_local: jax.Array,
    dispatch_valid_local: jax.Array,
    inverse_map_local: jax.Array,
    topk_idx_local: jax.Array,
    gate_weights_local: jax.Array,
) -> jax.Array:
    dispatch0 = dispatch_topk_bf16(
        x_local,
        dispatch_expert_local,
        dispatch_slot_local,
        dispatch_valid_local,
    )
    combined0 = combine_push3_bf16(
        dispatch0,
        inverse_map_local,
        topk_idx_local,
        gate_weights_local,
    )
    dispatch1 = dispatch_topk_bf16(
        x_local,
        dispatch_expert_local,
        dispatch_slot_local,
        dispatch_valid_local,
    )
    combined1 = combine_push3_bf16(
        dispatch1,
        inverse_map_local,
        topk_idx_local,
        gate_weights_local,
    )

    x_global = jax.lax.all_gather(x_local, "expert", tiled=True)
    local_tokens = x_local.shape[0]
    source_token = inverse_map_local[:, 0] * local_tokens + inverse_map_local[:, 1]
    expected_dispatch = jnp.take(x_global, source_token, axis=0)
    expected_dispatch = jnp.where(dispatch_valid_local[:, None], expected_dispatch, 0)
    dispatch_exact = jnp.logical_and(
        jnp.all(dispatch0 == expected_dispatch),
        jnp.all(dispatch1 == expected_dispatch),
    )
    dispatch_exact = jax.lax.pmin(dispatch_exact.astype(jnp.int32), "expert")

    reference = x_local.astype(jnp.float32) * jnp.sum(gate_weights_local, axis=1, dtype=jnp.float32)[:, None]

    def output_metrics(output: jax.Array) -> tuple[jax.Array, jax.Array]:
        difference = output.astype(jnp.float32) - reference
        error_square = jax.lax.psum(jnp.sum(jnp.square(difference), dtype=jnp.float32), "expert")
        reference_square = jax.lax.psum(jnp.sum(jnp.square(reference), dtype=jnp.float32), "expert")
        relative_l2 = jnp.sqrt(error_square) / jnp.sqrt(reference_square)
        finite = jax.lax.pmin(
            jnp.logical_and(jnp.all(jnp.isfinite(output)), jnp.all(jnp.isfinite(reference))).astype(jnp.int32),
            "expert",
        )
        return relative_l2, finite

    relative_l2_0, finite0 = output_metrics(combined0)
    relative_l2_1, finite1 = output_metrics(combined1)
    return jnp.stack(
        (
            dispatch_exact.astype(jnp.float32),
            relative_l2_0,
            relative_l2_1,
            finite0.astype(jnp.float32),
            finite1.astype(jnp.float32),
        )
    )


def _timing_summary(samples: list[float]) -> dict[str, Any]:
    values = np.asarray(samples, dtype=np.float64)
    return {
        "samples": len(samples),
        "mean_ms": statistics.fmean(samples),
        "p10_ms": float(np.quantile(values, 0.1)),
        "p50_ms": float(np.quantile(values, 0.5)),
        "p90_ms": float(np.quantile(values, 0.9)),
        "min_ms": min(samples),
        "max_ms": max(samples),
        "operation": "two_consecutive_dispatch_combine_cycles",
    }


def run(config: BenchmarkConfig, *, source_root: Path, cuda_home: Path, warmup: int, iterations: int) -> dict[str, Any]:
    if config.slot_layout != "compact":
        raise ValueError("the raw JAX FFI gate requires compact slots")
    devices = jax.local_devices()
    if len(devices) != EP_SIZE or any(device.platform != "gpu" for device in devices):
        raise RuntimeError(f"the raw JAX FFI gate requires eight local GPUs, got {devices}")

    runtime_config = UbxRuntimeConfig(
        num_ranks=EP_SIZE,
        max_tokens_per_rank=config.capacity_per_expert_rank,
        max_local_tokens=config.tokens_per_rank,
        hidden_size=config.hidden_dim,
        top_k=config.top_k,
        experts_per_rank=config.experts_per_rank,
    )
    ensure_local_runtime(runtime_config, source_root=source_root, cuda_home=cuda_home)

    mesh = Mesh(np.asarray(devices), ("expert",))
    token_sharding = NamedSharding(mesh, P("expert", None))
    map_sharding = NamedSharding(mesh, P("expert", None))
    capacity_sharding = NamedSharding(mesh, P("expert"))
    inverse_sharding = NamedSharding(mesh, P("expert", None))

    plan, dispatch_expert, dispatch_slot, dispatch_valid, inverse_map, topk_idx, gate_weights = _stack_maps(config)
    dispatch_expert_array = jax.device_put(dispatch_expert, map_sharding)
    dispatch_slot_array = jax.device_put(dispatch_slot, map_sharding)
    dispatch_valid_array = jax.device_put(dispatch_valid, capacity_sharding)
    inverse_map_array = jax.device_put(inverse_map, inverse_sharding)
    topk_idx_array = jax.device_put(topk_idx, map_sharding)
    gate_weights_array = jax.device_put(gate_weights, map_sharding)

    @partial(jax.jit, out_shardings=token_sharding)
    def make_tokens(key: jax.Array) -> jax.Array:
        return jax.random.normal(
            key,
            (config.global_tokens, config.hidden_dim),
            dtype=jnp.bfloat16,
        )

    x = make_tokens(jax.random.PRNGKey(config.seed + 1000))
    gate = jax.shard_map(
        _local_gate,
        mesh=mesh,
        in_specs=(
            P("expert", None),
            P("expert", None),
            P("expert", None),
            P("expert"),
            P("expert", None),
            P("expert", None),
            P("expert", None),
        ),
        out_specs=P(),
        check_vma=False,
    )
    compiled = (
        jax.jit(gate)
        .lower(
            x,
            dispatch_expert_array,
            dispatch_slot_array,
            dispatch_valid_array,
            inverse_map_array,
            topk_idx_array,
            gate_weights_array,
        )
        .compile()
    )

    def execute() -> jax.Array:
        return compiled(
            x,
            dispatch_expert_array,
            dispatch_slot_array,
            dispatch_valid_array,
            inverse_map_array,
            topk_idx_array,
            gate_weights_array,
        )

    metrics = execute()
    metrics.block_until_ready()
    for _ in range(warmup):
        execute().block_until_ready()
    samples = []
    for _ in range(iterations):
        start = time.perf_counter()
        execute().block_until_ready()
        samples.append((time.perf_counter() - start) * 1000)

    dispatch_exact, relative_l2_0, relative_l2_1, finite0, finite1 = map(float, np.asarray(metrics))
    passed = (
        dispatch_exact == 1.0
        and finite0 == 1.0
        and finite1 == 1.0
        and relative_l2_0 <= RELATIVE_L2_LIMIT
        and relative_l2_1 <= RELATIVE_L2_LIMIT
    )
    layout = pool_layout(runtime_config)
    return {
        "event": "jax_nccl_ubx_raw_ffi_gate",
        "config": asdict(config),
        "runtime_config": asdict(runtime_config),
        "route_plan": route_plan_summary(plan, config),
        "correctness": {
            "dispatch_bitwise_exact_both_cycles": dispatch_exact == 1.0,
            "relative_l2_cycle_0": relative_l2_0,
            "relative_l2_cycle_1": relative_l2_1,
            "finite_cycle_0": finite0 == 1.0,
            "finite_cycle_1": finite1 == 1.0,
            "relative_l2_limit": RELATIVE_L2_LIMIT,
        },
        "memory": asdict(layout),
        "timing": _timing_summary(samples),
        "passed": passed,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--cuda-home", type=Path, required=True)
    parser.add_argument("--routing", choices=("balanced", "learned_skew"), required=True)
    parser.add_argument("--tokens-per-rank", type=int, default=16_384)
    parser.add_argument("--hidden-dim", type=int, default=2_560)
    parser.add_argument("--num-experts", type=int, default=64)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--capacity-factor", type=float, default=1.0)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.warmup < 0 or args.iterations <= 0:
        raise ValueError("warmup must be nonnegative and iterations must be positive")
    config = BenchmarkConfig(
        tokens_per_rank=args.tokens_per_rank,
        hidden_dim=args.hidden_dim,
        num_experts=args.num_experts,
        top_k=args.top_k,
        capacity_factor=args.capacity_factor,
        routing=args.routing,
        slot_layout="compact",
        warmup=args.warmup,
        iterations=args.iterations,
    )
    try:
        result = run(
            config,
            source_root=args.source_root,
            cuda_home=args.cuda_home,
            warmup=args.warmup,
            iterations=args.iterations,
        )
        print(json.dumps(result, sort_keys=True), flush=True)
        if not result["passed"]:
            raise SystemExit(2)
    finally:
        shutdown_local_runtime()


if __name__ == "__main__":
    main()
