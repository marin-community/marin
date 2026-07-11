# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Benchmark exact bulk-ring expert compute variants on one EP8 node."""

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
    _moe_mlp_ep_ring_quack_local,
    _moe_mlp_ep_ring_two_chunk_local,
)
from levanter.grug._moe.sonic_quack import _require_quack

_EP_SIZE = 8
_BF16_RTOL = 0.1
_BF16_ATOL = 2e-4
_IMPLEMENTATIONS = {
    "ring": _moe_mlp_ep_ring_local,
    "ring_quack": _moe_mlp_ep_ring_quack_local,
    "two_chunk": _moe_mlp_ep_ring_two_chunk_local,
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--microbatch-size", type=int, default=32)
    parser.add_argument("--sequence-length", type=int, default=4096)
    parser.add_argument("--hidden-dim", type=int, default=2560)
    parser.add_argument("--intermediate-dim", type=int, default=1280)
    parser.add_argument("--num-experts", type=int, default=256)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--capacity-factor", type=float, default=1.0)
    parser.add_argument("--implementations", nargs="+", choices=tuple(_IMPLEMENTATIONS), default=("ring", "ring_quack"))
    parser.add_argument("--routing", choices=("balanced", "skew"), default="balanced")
    parser.add_argument("--skew-alpha", type=float, default=1.2)
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
    if args.skew_alpha <= 0:
        raise ValueError(f"skew_alpha must be positive, got {args.skew_alpha}")
    if args.num_experts % _EP_SIZE:
        raise ValueError(f"num_experts={args.num_experts} must be divisible by EP size {_EP_SIZE}")
    if args.top_k > args.num_experts:
        raise ValueError(f"top_k={args.top_k} must be <= num_experts={args.num_experts}")
    tokens = args.microbatch_size * args.sequence_length
    if tokens % _EP_SIZE:
        raise ValueError(f"microbatch tokens={tokens} must be divisible by EP size {_EP_SIZE}")
    if args.routing == "balanced" and tokens * args.top_k % args.num_experts:
        raise ValueError("balanced routing requires total assignments to be divisible by num_experts")
    if "two_chunk" in args.implementations and tokens // _EP_SIZE < 2:
        raise ValueError("two_chunk requires at least two tokens per EP shard")


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
    top_k: int,
    num_experts: int,
    seed: int,
    skew_alpha: float,
) -> jax.Array:
    assignments = tokens * top_k
    if routing == "balanced":
        return jnp.arange(assignments, dtype=jnp.int32).reshape(tokens, top_k) % num_experts

    ranks = np.arange(1, num_experts + 1, dtype=np.float64)
    probabilities = np.power(ranks, -skew_alpha)
    probabilities /= probabilities.sum()
    selected = np.random.default_rng(seed).choice(num_experts, size=assignments, p=probabilities)
    return jnp.asarray(selected.reshape(tokens, top_k), dtype=jnp.int32)


def _routing_statistics(
    selected_experts: jax.Array,
    *,
    num_experts: int,
    capacity_factor: float,
) -> dict[str, Any]:
    counts = np.bincount(np.asarray(selected_experts).reshape(-1), minlength=num_experts)
    local_experts = num_experts // _EP_SIZE
    local_capacity = max(local_experts, int(np.ceil(capacity_factor * selected_experts.size / _EP_SIZE)))
    accepted_by_rank = []
    groups_with_padding_by_rank = []
    padding_by_rank = []
    for rank in range(_EP_SIZE):
        local_counts = counts[rank * local_experts : (rank + 1) * local_experts]
        remaining = local_capacity
        accepted = []
        for count in local_counts:
            take = min(int(count), remaining)
            accepted.append(take)
            remaining -= take
        groups_with_padding = accepted.copy()
        groups_with_padding[-1] += remaining
        accepted_by_rank.append(accepted)
        groups_with_padding_by_rank.append(groups_with_padding)
        padding_by_rank.append(remaining)
    return {
        "expert_counts": counts.tolist(),
        "expert_count_min": int(counts.min()),
        "expert_count_max": int(counts.max()),
        "local_capacity": local_capacity,
        "accepted_group_counts_by_rank": accepted_by_rank,
        "quack_group_sizes_by_rank": groups_with_padding_by_rank,
        "padding_by_rank": padding_by_rank,
        "padding_total": int(sum(padding_by_rank)),
    }


def _compiled_functions(
    mesh: Mesh,
    *,
    implementations: tuple[str, ...],
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

    names = tuple(dict.fromkeys(("ring", *implementations)))
    forwards = {name: runner(_IMPLEMENTATIONS[name]) for name in names}
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
        print(
            f"groups: min={result['groups']['expert_count_min']}, max={result['groups']['expert_count_max']}, "
            f"local_capacity={result['groups']['local_capacity']}, padding={result['groups']['padding_by_rank']}"
        )
        for mode in ("forward", "value_and_grad"):
            fields = []
            ring_median = result["timings"][mode].get("ring", {}).get("median_ms")
            for name, timing in result["timings"][mode].items():
                field = f"{name}={timing['median_ms']:.3f} ms"
                if name != "ring" and ring_median is not None:
                    field += f" ({ring_median / timing['median_ms']:.3f}x vs ring)"
                fields.append(field)
            print(f"{mode:>14}: {', '.join(fields)}")
    if output in ("json", "both"):
        print(json.dumps(result, sort_keys=True))


def main() -> None:
    args = _parser().parse_args()
    _validate_args(args)
    implementations = tuple(dict.fromkeys(args.implementations))
    if "ring_quack" in implementations:
        try:
            _require_quack()
        except ImportError as error:
            raise RuntimeError(
                "ring_quack requires the marin-levanter gpu extra, including quack-kernels and jax-tvm-ffi"
            ) from error
    mesh = _mesh()
    if not args.lower_only and jax.default_backend() != "gpu":
        raise RuntimeError("timing requires eight local GPUs; use --lower-only for a CPU lowering smoke")

    tokens = args.microbatch_size * args.sequence_length
    batch_sharding = NamedSharding(mesh, P(("data", "expert"), None))
    expert_sharding = NamedSharding(mesh, P("expert", None, None))
    key_x, key_weights, key_w13, key_w2 = jax.random.split(jax.random.key(args.seed), 4)
    selected_experts = _selected_experts(
        routing=args.routing,
        tokens=tokens,
        top_k=args.top_k,
        num_experts=args.num_experts,
        seed=args.seed,
        skew_alpha=args.skew_alpha,
    )
    group_statistics = _routing_statistics(
        selected_experts,
        num_experts=args.num_experts,
        capacity_factor=args.capacity_factor,
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
        forwards, gate = _compiled_functions(
            mesh,
            implementations=implementations,
            num_experts=args.num_experts,
            capacity_factor=args.capacity_factor,
        )
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
                "groups": group_statistics,
            }
            _print_result(result, args.output)
            return

        compiled_forwards = {name: lowered.compile() for name, lowered in lowered_forwards.items()}
        compiled_value_and_grads = {name: lowered.compile() for name, lowered in lowered_value_and_grads.items()}
        use_fast_path = bool(jax.device_get(gate_lowered.compile()(inputs[1])))

        ring_forward = jax.block_until_ready(compiled_forwards["ring"](*inputs))
        ring_vg = jax.block_until_ready(compiled_value_and_grads["ring"](*inputs))
        parity = {}
        for name in implementations:
            actual_forward = jax.block_until_ready(compiled_forwards[name](*inputs))
            actual_vg = jax.block_until_ready(compiled_value_and_grads[name](*inputs))
            output_parity = _parity_metrics(actual_forward[0], ring_forward[0])
            if int(actual_forward[1]) != int(ring_forward[1]):
                raise AssertionError(f"drop mismatch: {name}={int(actual_forward[1])}, ring={int(ring_forward[1])}")
            gradient_parity = [
                _parity_metrics(actual, expected) for actual, expected in zip(actual_vg[1], ring_vg[1], strict=True)
            ]
            if not output_parity["allclose"] or not all(metric["allclose"] for metric in gradient_parity):
                raise AssertionError(f"{name} parity failed: output={output_parity}, gradients={gradient_parity}")
            parity[name] = {
                "output": output_parity,
                "gradients": dict(zip(("x", "combine_weights", "w13", "w2"), gradient_parity, strict=True)),
                "dropped": int(actual_forward[1]),
            }

        timings = {
            "forward": {
                name: _time(compiled, inputs, warmup=args.warmup, iterations=args.iterations)
                for name, compiled in compiled_forwards.items()
                if name in implementations
            },
            "value_and_grad": {
                name: _time(compiled, inputs, warmup=args.warmup, iterations=args.iterations)
                for name, compiled in compiled_value_and_grads.items()
                if name in implementations
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
        "seed": args.seed,
        "skew_alpha": args.skew_alpha,
        "implementations": implementations,
        "two_chunk_path": "fast" if use_fast_path else "fallback",
        "groups": group_statistics,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "parity": parity,
        "timings": timings,
    }
    _print_result(result, args.output)


if __name__ == "__main__":
    main()
