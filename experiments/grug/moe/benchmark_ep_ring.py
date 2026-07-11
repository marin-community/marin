# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Benchmark the exact bulk and two-chunk EP ring prototypes on one EP8 node."""

import argparse
import json
import statistics
import time
from collections.abc import Callable
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import AxisType, Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from levanter.grug._moe.ep_ring import (
    _ep_ring_two_chunk_fast_path_local,
    _moe_mlp_ep_ring_local,
    _moe_mlp_ep_ring_two_chunk_local,
)

_EP_SIZE = 8
_BF16_RTOL = 0.1
_BF16_ATOL = 2e-4


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--microbatch-size", type=int, default=32)
    parser.add_argument("--sequence-length", type=int, default=4096)
    parser.add_argument("--hidden-dim", type=int, default=2560)
    parser.add_argument("--intermediate-dim", type=int, default=1280)
    parser.add_argument("--num-experts", type=int, default=256)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--capacity-factor", type=float, default=1.0)
    parser.add_argument("--routing", choices=("balanced", "one_half_fallback"), default="balanced")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--lower-only", action="store_true")
    parser.add_argument("--output", choices=("human", "json", "both"), default="both")
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    positive = {
        "microbatch_size": args.microbatch_size,
        "sequence_length": args.sequence_length,
        "hidden_dim": args.hidden_dim,
        "intermediate_dim": args.intermediate_dim,
        "num_experts": args.num_experts,
        "top_k": args.top_k,
        "iterations": args.iterations,
    }
    for name, value in positive.items():
        if value <= 0:
            raise ValueError(f"{name} must be positive, got {value}")
    if args.warmup < 0:
        raise ValueError(f"warmup must be non-negative, got {args.warmup}")
    if args.capacity_factor <= 0:
        raise ValueError(f"capacity_factor must be positive, got {args.capacity_factor}")
    if args.num_experts % _EP_SIZE:
        raise ValueError(f"num_experts={args.num_experts} must be divisible by EP size {_EP_SIZE}")
    if args.top_k > args.num_experts:
        raise ValueError(f"top_k={args.top_k} must be <= num_experts={args.num_experts}")
    tokens = args.microbatch_size * args.sequence_length
    if tokens % _EP_SIZE:
        raise ValueError(f"microbatch tokens={tokens} must be divisible by EP size {_EP_SIZE}")


def _mesh() -> Mesh:
    local_devices = jax.local_devices()
    if len(local_devices) < _EP_SIZE:
        raise RuntimeError(f"EP8 benchmark requires 8 local devices, found {len(local_devices)}")
    devices = np.asarray(local_devices[:_EP_SIZE]).reshape(1, _EP_SIZE, 1)
    return Mesh(
        devices,
        axis_names=("data", "expert", "model"),
        axis_types=(AxisType.Explicit, AxisType.Explicit, AxisType.Explicit),
    )


def _selected_experts(
    *,
    routing: str,
    tokens: int,
    tokens_per_shard: int,
    top_k: int,
    num_experts: int,
) -> jax.Array:
    token = jnp.arange(tokens, dtype=jnp.int32)[:, None]
    topk_offset = jnp.arange(top_k, dtype=jnp.int32)[None, :]
    balanced = (token * top_k + topk_offset) % num_experts
    if routing == "balanced":
        return balanced
    source_token = token % tokens_per_shard
    return jnp.where(source_token < (tokens_per_shard + 1) // 2, jnp.zeros_like(balanced), balanced)


def _compiled_functions(
    mesh: Mesh,
    *,
    num_experts: int,
    capacity_factor: float,
) -> tuple[dict[str, Callable[..., Any]], Callable[..., Any]]:
    batch_spec = P(("data", "expert"), None)
    expert_spec = P("expert", None, None)

    def runner(local_fn: Callable[..., Any]) -> Callable[..., Any]:
        mapped = jax.shard_map(
            partial(
                local_fn,
                activation_fn=jax.nn.silu,
                num_experts=num_experts,
                capacity_factor=capacity_factor,
            ),
            mesh=mesh,
            in_specs=(batch_spec, batch_spec, batch_spec, expert_spec, expert_spec),
            out_specs=(batch_spec, P()),
            check_vma=False,
        )
        return jax.jit(mapped)

    forwards = {
        "ring": runner(_moe_mlp_ep_ring_local),
        "two_chunk": runner(_moe_mlp_ep_ring_two_chunk_local),
    }
    local_experts = num_experts // _EP_SIZE
    gate = jax.jit(
        jax.shard_map(
            partial(
                _ep_ring_two_chunk_fast_path_local,
                local_experts=local_experts,
                num_experts=num_experts,
                capacity_factor=capacity_factor,
            ),
            mesh=mesh,
            in_specs=(batch_spec,),
            out_specs=P(),
            check_vma=False,
        )
    )
    return forwards, gate


def _loss_with_aux(
    forward: Callable[..., Any],
    x: jax.Array,
    selected_experts: jax.Array,
    combine_weights: jax.Array,
    w13: jax.Array,
    w2: jax.Array,
) -> tuple[jax.Array, tuple[jax.Array, jax.Array]]:
    out, dropped = forward(x, selected_experts, combine_weights, w13, w2)
    return jnp.mean(jnp.square(out.astype(jnp.float32))), (out, dropped)


def _parity_metrics(actual: jax.Array, expected: jax.Array) -> dict[str, float | bool]:
    actual_f32 = actual.astype(jnp.float32)
    expected_f32 = expected.astype(jnp.float32)
    difference = jnp.abs(actual_f32 - expected_f32)
    close = jnp.all(difference <= _BF16_ATOL + _BF16_RTOL * jnp.abs(expected_f32))
    max_abs, mean_abs, is_close = jax.device_get((jnp.max(difference), jnp.mean(difference), close))
    return {"max_abs": float(max_abs), "mean_abs": float(mean_abs), "allclose": bool(is_close)}


def _time(
    compiled: Callable[..., Any], args: tuple[jax.Array, ...], *, warmup: int, iterations: int
) -> dict[str, float]:
    for _ in range(warmup):
        jax.block_until_ready(compiled(*args))
    durations = []
    for _ in range(iterations):
        start = time.perf_counter()
        jax.block_until_ready(compiled(*args))
        durations.append(time.perf_counter() - start)
    return {
        "mean_ms": 1000.0 * statistics.fmean(durations),
        "median_ms": 1000.0 * statistics.median(durations),
        "min_ms": 1000.0 * min(durations),
        "max_ms": 1000.0 * max(durations),
    }


def _print_result(result: dict[str, Any], output: str) -> None:
    if output in ("human", "both"):
        if result.get("lower_only"):
            print(
                f"EP8 lowering smoke: backend={result['backend']}, tokens={result['tokens']}, "
                f"forward={','.join(result['lowered']['forward'])}, "
                f"value_and_grad={','.join(result['lowered']['value_and_grad'])}"
            )
            if output == "both":
                print(json.dumps(result, sort_keys=True))
            return
        print(
            f"EP8 {result['routing']} routing: two_chunk_path={result['two_chunk_path']}, "
            f"tokens={result['tokens']}, shape={result['hidden_dim']}x{result['intermediate_dim']}, "
            f"experts={result['num_experts']}, top_k={result['top_k']}"
        )
        for mode in ("forward", "value_and_grad"):
            ring = result["timings"][mode]["ring"]
            chunked = result["timings"][mode]["two_chunk"]
            speedup = ring["median_ms"] / chunked["median_ms"]
            print(
                f"{mode:>14}: ring={ring['median_ms']:.3f} ms, two_chunk={chunked['median_ms']:.3f} ms, "
                f"median_speedup={speedup:.3f}x"
            )
    if output in ("json", "both"):
        print(json.dumps(result, sort_keys=True))


def main() -> None:
    args = _parser().parse_args()
    _validate_args(args)
    mesh = _mesh()
    if not args.lower_only and jax.default_backend() != "gpu":
        raise RuntimeError("timing requires eight local GPUs; use --lower-only for a CPU lowering smoke")

    tokens = args.microbatch_size * args.sequence_length
    tokens_per_shard = tokens // _EP_SIZE
    batch_sharding = NamedSharding(mesh, P(("data", "expert"), None))
    expert_sharding = NamedSharding(mesh, P("expert", None, None))
    key_x, key_weights, key_w13, key_w2 = jax.random.split(jax.random.key(args.seed), 4)
    selected_experts = _selected_experts(
        routing=args.routing,
        tokens=tokens,
        tokens_per_shard=tokens_per_shard,
        top_k=args.top_k,
        num_experts=args.num_experts,
    )
    x = jax.random.normal(key_x, (tokens, args.hidden_dim), dtype=jnp.bfloat16)
    combine_weights = jax.nn.softmax(jax.random.normal(key_weights, (tokens, args.top_k), dtype=jnp.float32), axis=-1)
    w13 = 0.02 * jax.random.normal(
        key_w13,
        (args.num_experts, args.hidden_dim, 2 * args.intermediate_dim),
        dtype=jnp.bfloat16,
    )
    w2 = 0.02 * jax.random.normal(
        key_w2,
        (args.num_experts, args.intermediate_dim, args.hidden_dim),
        dtype=jnp.bfloat16,
    )
    inputs = (
        jax.device_put(x, batch_sharding),
        jax.device_put(selected_experts, batch_sharding),
        jax.device_put(combine_weights, batch_sharding),
        jax.device_put(w13, expert_sharding),
        jax.device_put(w2, expert_sharding),
    )

    with jax.set_mesh(mesh):
        forwards, gate = _compiled_functions(mesh, num_experts=args.num_experts, capacity_factor=args.capacity_factor)
        value_and_grads = {
            name: jax.jit(jax.value_and_grad(partial(_loss_with_aux, forward), argnums=(0, 2, 3, 4), has_aux=True))
            for name, forward in forwards.items()
        }
        lowered_forwards = {name: fn.lower(*inputs) for name, fn in forwards.items()}
        lowered_value_and_grads = {name: fn.lower(*inputs) for name, fn in value_and_grads.items()}
        gate_lowered = gate.lower(inputs[1])

        if args.lower_only:
            result = {
                "backend": jax.default_backend(),
                "devices": _EP_SIZE,
                "lower_only": True,
                "lowered": {
                    "forward": sorted(lowered_forwards),
                    "value_and_grad": sorted(lowered_value_and_grads),
                    "fast_path_gate": True,
                },
                "tokens": tokens,
            }
            _print_result(result, args.output)
            return

        compiled_forwards = {name: lowered.compile() for name, lowered in lowered_forwards.items()}
        compiled_value_and_grads = {name: lowered.compile() for name, lowered in lowered_value_and_grads.items()}
        use_fast_path = bool(jax.device_get(gate_lowered.compile()(inputs[1])))

        ring_forward = jax.block_until_ready(compiled_forwards["ring"](*inputs))
        chunked_forward = jax.block_until_ready(compiled_forwards["two_chunk"](*inputs))
        ring_vg = jax.block_until_ready(compiled_value_and_grads["ring"](*inputs))
        chunked_vg = jax.block_until_ready(compiled_value_and_grads["two_chunk"](*inputs))

        output_parity = _parity_metrics(chunked_forward[0], ring_forward[0])
        if int(chunked_forward[1]) != int(ring_forward[1]):
            raise AssertionError(f"drop mismatch: two_chunk={int(chunked_forward[1])}, ring={int(ring_forward[1])}")
        gradient_parity = [
            _parity_metrics(actual, expected) for actual, expected in zip(chunked_vg[1], ring_vg[1], strict=True)
        ]
        if not output_parity["allclose"] or not all(metric["allclose"] for metric in gradient_parity):
            raise AssertionError(f"parity failed: output={output_parity}, gradients={gradient_parity}")

        timings = {
            "forward": {
                name: _time(compiled, inputs, warmup=args.warmup, iterations=args.iterations)
                for name, compiled in compiled_forwards.items()
            },
            "value_and_grad": {
                name: _time(compiled, inputs, warmup=args.warmup, iterations=args.iterations)
                for name, compiled in compiled_value_and_grads.items()
            },
        }

    result = {
        "backend": jax.default_backend(),
        "device_kind": jax.devices()[0].device_kind,
        "devices": _EP_SIZE,
        "microbatch_size": args.microbatch_size,
        "sequence_length": args.sequence_length,
        "tokens": tokens,
        "hidden_dim": args.hidden_dim,
        "intermediate_dim": args.intermediate_dim,
        "num_experts": args.num_experts,
        "top_k": args.top_k,
        "capacity_factor": args.capacity_factor,
        "routing": args.routing,
        "two_chunk_path": "fast" if use_fast_path else "fallback",
        "warmup": args.warmup,
        "iterations": args.iterations,
        "parity": {
            "output": output_parity,
            "gradients": dict(zip(("x", "combine_weights", "w13", "w2"), gradient_parity, strict=True)),
            "dropped": int(ring_forward[1]),
        },
        "timings": timings,
    }
    _print_result(result, args.output)


if __name__ == "__main__":
    main()
