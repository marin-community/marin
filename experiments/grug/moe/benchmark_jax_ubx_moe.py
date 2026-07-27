# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compare the full UB-X MoE value and VJP against exact Ring on one H100x8 node."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from functools import partial
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from levanter.grug._moe.ep_ring import _moe_mlp_ep_ring_local
from levanter.grug._moe.ep_ubx import _moe_mlp_ep_ubx_local
from levanter.kernels.ubx import UbxRuntimeConfig, ensure_local_runtime, shutdown_local_runtime

from experiments.grug.moe.benchmark_nccl_ubx import BenchmarkConfig, build_route_plan, route_plan_summary

EP_SIZE = 8
RELATIVE_L2_LIMIT = 0.002


def _relative_l2(actual: np.ndarray, reference: np.ndarray) -> dict[str, Any]:
    actual_f32 = np.asarray(actual, dtype=np.float32)
    reference_f32 = np.asarray(reference, dtype=np.float32)
    difference = actual_f32 - reference_f32
    reference_l2 = float(np.linalg.norm(reference_f32.reshape(-1).astype(np.float64)))
    absolute_l2 = float(np.linalg.norm(difference.reshape(-1).astype(np.float64)))
    relative_l2 = absolute_l2 / max(reference_l2, 1e-12)
    finite = bool(np.all(np.isfinite(actual_f32)) and np.all(np.isfinite(reference_f32)))
    return {
        "reference_l2": reference_l2,
        "absolute_l2": absolute_l2,
        "max_absolute_error": float(np.max(np.abs(difference), initial=0.0)),
        "relative_l2": relative_l2 if finite else float("inf"),
        "finite": finite,
        "passed": finite and relative_l2 <= RELATIVE_L2_LIMIT,
    }


def _timed(execute, *, warmup: int, iterations: int) -> dict[str, Any]:
    for _ in range(warmup):
        jax.block_until_ready(execute())
    samples = []
    for _ in range(iterations):
        start = time.perf_counter()
        jax.block_until_ready(execute())
        samples.append((time.perf_counter() - start) * 1000)
    values = np.asarray(samples, dtype=np.float64)
    return {
        "samples": len(samples),
        "mean_ms": statistics.fmean(samples),
        "p50_ms": float(np.quantile(values, 0.5)),
        "p90_ms": float(np.quantile(values, 0.9)),
        "min_ms": min(samples),
        "max_ms": max(samples),
        "operation": "full_moe_value_and_grad",
    }


def run(
    config: BenchmarkConfig,
    *,
    intermediate_dim: int,
    source_root: Path,
    cuda_home: Path,
    warmup: int,
    iterations: int,
) -> dict[str, Any]:
    devices = jax.local_devices()
    if len(devices) != EP_SIZE or any(device.platform != "gpu" for device in devices):
        raise RuntimeError(f"the UB-X MoE gate requires eight local GPUs, got {devices}")
    if intermediate_dim <= 0:
        raise ValueError(f"intermediate_dim must be positive, got {intermediate_dim}")

    ensure_local_runtime(
        UbxRuntimeConfig(
            num_ranks=EP_SIZE,
            max_tokens_per_rank=config.capacity_per_expert_rank,
            max_local_tokens=config.tokens_per_rank,
            hidden_size=config.hidden_dim,
            top_k=config.top_k,
            experts_per_rank=config.experts_per_rank,
        ),
        source_root=source_root,
        cuda_home=cuda_home,
    )

    plan = build_route_plan(config)
    rng = np.random.default_rng(config.seed + 2000)
    x = jnp.asarray(
        rng.normal(scale=0.5, size=(config.global_tokens, config.hidden_dim)).astype(np.float32),
        dtype=jnp.bfloat16,
    )
    combine_weights = plan.assignment_weights.astype(np.float32)
    w13 = jnp.asarray(
        rng.normal(
            scale=0.02,
            size=(config.num_experts, config.hidden_dim, 2 * intermediate_dim),
        ).astype(np.float32),
        dtype=jnp.bfloat16,
    )
    w2 = jnp.asarray(
        rng.normal(
            scale=0.02,
            size=(config.num_experts, intermediate_dim, config.hidden_dim),
        ).astype(np.float32),
        dtype=jnp.bfloat16,
    )
    selected_experts = plan.selected_experts.astype(np.int32)

    mesh = Mesh(np.asarray(devices), ("expert",))
    token_spec = P("expert", None)
    weight_spec = P("expert", None, None)
    token_sharding = NamedSharding(mesh, token_spec)
    weight_sharding = NamedSharding(mesh, weight_spec)
    x = jax.device_put(x, token_sharding)
    selected_experts = jax.device_put(selected_experts, token_sharding)
    combine_weights = jax.device_put(combine_weights, token_sharding)
    w13 = jax.device_put(w13, weight_sharding)
    w2 = jax.device_put(w2, weight_sharding)

    def runner(local_fn):
        return jax.shard_map(
            partial(
                local_fn,
                activation_fn=jax.nn.silu,
                num_experts=config.num_experts,
                capacity_factor=config.capacity_factor,
            ),
            mesh=mesh,
            in_specs=(token_spec, token_spec, token_spec, weight_spec, weight_spec),
            out_specs=(token_spec, P()),
            check_vma=False,
        )

    ring = runner(_moe_mlp_ep_ring_local)
    ubx = runner(_moe_mlp_ep_ubx_local)

    def value_and_grad(implementation, x, combine_weights, w13, w2):
        def loss_fn(x, combine_weights, w13, w2):
            output, dropped = implementation(x, selected_experts, combine_weights, w13, w2)
            output_f32 = output.astype(jnp.float32)
            loss = jnp.mean(jnp.square(output_f32), dtype=jnp.float32)
            return loss, (output, dropped)

        return jax.value_and_grad(loss_fn, argnums=(0, 1, 2, 3), has_aux=True)(
            x,
            combine_weights,
            w13,
            w2,
        )

    ring_compiled = jax.jit(partial(value_and_grad, ring)).lower(x, combine_weights, w13, w2).compile()
    ubx_compiled = jax.jit(partial(value_and_grad, ubx)).lower(x, combine_weights, w13, w2).compile()

    def execute_ring():
        return ring_compiled(x, combine_weights, w13, w2)

    def execute_ubx():
        return ubx_compiled(x, combine_weights, w13, w2)

    ring_result = jax.block_until_ready(execute_ring())
    ubx_result = jax.block_until_ready(execute_ubx())
    (ring_loss, (ring_output, ring_dropped)), ring_gradients = ring_result
    (ubx_loss, (ubx_output, ubx_dropped)), ubx_gradients = ubx_result

    output_parity = _relative_l2(np.asarray(ubx_output), np.asarray(ring_output))
    loss_parity = _relative_l2(np.asarray(ubx_loss), np.asarray(ring_loss))
    gradient_names = ("x", "combine_weights", "w13", "w2")
    gradient_parity = {
        name: _relative_l2(np.asarray(actual), np.asarray(reference))
        for name, actual, reference in zip(gradient_names, ubx_gradients, ring_gradients, strict=True)
    }
    expected_dropped = int(plan.drops_by_expert_rank.sum())
    drops_exact = int(ubx_dropped) == int(ring_dropped) == expected_dropped
    passed = (
        drops_exact
        and output_parity["passed"]
        and loss_parity["passed"]
        and all(value["passed"] for value in gradient_parity.values())
    )
    return {
        "event": "jax_nccl_ubx_moe_value_and_grad_gate",
        "config": {
            "tokens_per_rank": config.tokens_per_rank,
            "hidden_dim": config.hidden_dim,
            "intermediate_dim": intermediate_dim,
            "num_experts": config.num_experts,
            "top_k": config.top_k,
            "capacity_factor": config.capacity_factor,
            "routing": config.routing,
        },
        "route_plan": route_plan_summary(plan, config),
        "correctness": {
            "relative_l2_limit": RELATIVE_L2_LIMIT,
            "drops_exact": drops_exact,
            "expected_dropped": expected_dropped,
            "ring_dropped": int(ring_dropped),
            "ubx_dropped": int(ubx_dropped),
            "output": output_parity,
            "loss": loss_parity,
            "gradients": gradient_parity,
        },
        "timing": {
            "ring": _timed(execute_ring, warmup=warmup, iterations=iterations),
            "ubx": _timed(execute_ubx, warmup=warmup, iterations=iterations),
        },
        "passed": passed,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--cuda-home", type=Path, required=True)
    parser.add_argument("--routing", choices=("balanced", "learned_skew"), required=True)
    parser.add_argument("--tokens-per-rank", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--intermediate-dim", type=int, default=384)
    parser.add_argument("--num-experts", type=int, default=64)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--capacity-factor", type=float, default=1.0)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=5)
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
            intermediate_dim=args.intermediate_dim,
            source_root=args.source_root,
            cuda_home=args.cuda_home,
            warmup=args.warmup,
            iterations=args.iterations,
        )
        print(json.dumps(result, sort_keys=True))
        if not result["passed"]:
            raise SystemExit(1)
    finally:
        shutdown_local_runtime()


if __name__ == "__main__":
    main()
