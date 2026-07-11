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
_QUANTILE_SAMPLE_SIZE = 65_536
_ERROR_QUANTILES = (0.0, 0.5, 0.9, 0.99, 0.999, 1.0)
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
    parser.add_argument(
        "--parity-mode",
        choices=("strict", "diagnostic"),
        default="strict",
        help="strict raises on parity failure; diagnostic records failure, marks the run non-promotable, and times it",
    )
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


def _sampled_quantiles(difference: jax.Array) -> dict[str, Any]:
    flat = difference.reshape(-1)
    sample_size = min(flat.size, _QUANTILE_SAMPLE_SIZE)
    if sample_size < flat.size:
        indices = jnp.linspace(0, flat.size - 1, sample_size, dtype=jnp.int32)
        sample = jnp.take(flat, indices)
    else:
        sample = flat
    values = np.asarray(jax.device_get(jnp.quantile(sample, jnp.asarray(_ERROR_QUANTILES, dtype=jnp.float32))))
    return {
        "values": {
            f"p{100 * quantile:g}": float(value) for quantile, value in zip(_ERROR_QUANTILES, values, strict=True)
        },
        "sample_size": sample_size,
        "total_size": flat.size,
        "exact": sample_size == flat.size,
    }


def _group_error_breakdown(
    actual_f32: jax.Array,
    expected_f32: jax.Array,
    *,
    group_ids: jax.Array,
    group_labels: tuple[str, ...],
) -> dict[str, dict[str, float | int]]:
    group_ids = jnp.asarray(group_ids, dtype=jnp.int32)
    if actual_f32.shape[: group_ids.ndim] != group_ids.shape:
        raise ValueError(f"group IDs with shape {group_ids.shape} do not prefix tensor shape {actual_f32.shape}")
    rows = group_ids.size
    actual_rows = actual_f32.reshape(rows, -1)
    expected_rows = expected_f32.reshape(rows, -1)
    difference = jnp.abs(actual_rows - expected_rows)
    mismatches = jnp.logical_not(difference <= _BF16_ATOL + _BF16_RTOL * jnp.abs(expected_rows))
    segment_ids = group_ids.reshape(-1)
    num_groups = len(group_labels)
    values_per_row = actual_rows.shape[1]

    mismatch_count = jax.ops.segment_sum(jnp.sum(mismatches, axis=1), segment_ids, num_segments=num_groups)
    abs_error_sum = jax.ops.segment_sum(jnp.sum(difference, axis=1), segment_ids, num_segments=num_groups)
    squared_error = jax.ops.segment_sum(jnp.sum(jnp.square(difference), axis=1), segment_ids, num_segments=num_groups)
    reference_squared = jax.ops.segment_sum(
        jnp.sum(jnp.square(expected_rows), axis=1), segment_ids, num_segments=num_groups
    )
    candidate_squared = jax.ops.segment_sum(
        jnp.sum(jnp.square(actual_rows), axis=1), segment_ids, num_segments=num_groups
    )
    max_abs = jax.ops.segment_max(jnp.max(difference, axis=1), segment_ids, num_segments=num_groups)
    row_count = jax.ops.segment_sum(jnp.ones(rows, dtype=jnp.int32), segment_ids, num_segments=num_groups)
    aggregates = jax.device_get(
        (mismatch_count, abs_error_sum, squared_error, reference_squared, candidate_squared, max_abs, row_count)
    )

    breakdown = {}
    for index, label in enumerate(group_labels):
        count = int(aggregates[6][index]) * values_per_row
        if count == 0:
            continue
        reference_l2 = float(np.sqrt(aggregates[3][index]))
        error_l2 = float(np.sqrt(aggregates[2][index]))
        breakdown[label] = {
            "element_count": count,
            "mismatch_count": int(aggregates[0][index]),
            "mismatch_fraction": float(aggregates[0][index] / count),
            "mean_abs": float(aggregates[1][index] / count),
            "max_abs": float(aggregates[5][index]),
            "reference_l2": reference_l2,
            "candidate_l2": float(np.sqrt(aggregates[4][index])),
            "relative_l2_error": error_l2 / reference_l2 if reference_l2 else (0.0 if error_l2 == 0.0 else float("inf")),
        }
    return breakdown


def _parity_metrics(
    actual: jax.Array,
    expected: jax.Array,
    *,
    group_ids: jax.Array | None = None,
    group_labels: tuple[str, ...] = (),
) -> dict[str, Any]:
    actual_f32 = actual.astype(jnp.float32)
    expected_f32 = expected.astype(jnp.float32)
    difference = jnp.abs(actual_f32 - expected_f32)
    mismatches = jnp.logical_not(difference <= _BF16_ATOL + _BF16_RTOL * jnp.abs(expected_f32))
    mismatch_count = jnp.sum(mismatches)
    worst_flat_index = jnp.argmax(difference)
    reference_l2 = jnp.linalg.norm(expected_f32.reshape(-1))
    candidate_l2 = jnp.linalg.norm(actual_f32.reshape(-1))
    error_l2 = jnp.linalg.norm(difference.reshape(-1))
    scalars = jax.device_get(
        (
            jnp.max(difference),
            jnp.mean(difference),
            mismatch_count,
            reference_l2,
            candidate_l2,
            error_l2,
            worst_flat_index,
            jnp.take(expected_f32.reshape(-1), worst_flat_index),
            jnp.take(actual_f32.reshape(-1), worst_flat_index),
        )
    )
    reference_l2_value = float(scalars[3])
    error_l2_value = float(scalars[5])
    mismatch_count_value = int(scalars[2])
    metrics = {
        "max_abs": float(scalars[0]),
        "mean_abs": float(scalars[1]),
        "allclose": mismatch_count_value == 0,
        "mismatch_count": mismatch_count_value,
        "mismatch_fraction": mismatch_count_value / actual.size,
        "reference_l2": reference_l2_value,
        "candidate_l2": float(scalars[4]),
        "relative_l2_error": (
            error_l2_value / reference_l2_value
            if reference_l2_value
            else (0.0 if error_l2_value == 0.0 else float("inf"))
        ),
        "worst_error": {
            "flat_index": int(scalars[6]),
            "index": [int(index) for index in np.unravel_index(int(scalars[6]), actual.shape)],
            "reference_magnitude": abs(float(scalars[7])),
            "candidate_magnitude": abs(float(scalars[8])),
        },
        "abs_error_quantiles": _sampled_quantiles(difference),
    }
    if group_ids is not None:
        metrics["error_by_group"] = _group_error_breakdown(
            actual_f32, expected_f32, group_ids=group_ids, group_labels=group_labels
        )
    return metrics


def _parity_failures(parity: dict[str, Any]) -> list[dict[str, str]]:
    failures = []
    for implementation, implementation_parity in parity.items():
        if not implementation_parity["dropped_matches"]:
            failures.append({"implementation": implementation, "tensor": "dropped"})
        if not implementation_parity["output"]["allclose"]:
            failures.append({"implementation": implementation, "tensor": "output"})
        for tensor, metrics in implementation_parity["gradients"].items():
            if not metrics["allclose"]:
                failures.append({"implementation": implementation, "tensor": f"gradient.{tensor}"})
    return failures


def _parity_status(parity: dict[str, Any], *, mode: str) -> dict[str, Any]:
    failures = _parity_failures(parity)
    if failures and mode == "strict":
        raise AssertionError(f"parity failed: {failures}")
    return {
        "mode": mode,
        "passed": not failures,
        "failures": failures,
        "promotable": mode == "strict" and not failures,
        "non_promotable_reason": None if mode == "strict" and not failures else "diagnostic parity mode",
    }


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
        if not result["promotable"]:
            print(
                f"NON-PROMOTABLE diagnostic run: parity_passed={result['parity_status']['passed']}, "
                f"failures={result['parity_status']['failures']}"
            )
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
        tokens_per_rank = tokens // _EP_SIZE
        token_owner_ids = jnp.arange(tokens, dtype=jnp.int32) // tokens_per_rank
        token_owner_labels = tuple(f"owner_rank={rank}" for rank in range(_EP_SIZE))
        expert_ids = jnp.arange(args.num_experts, dtype=jnp.int32)
        local_experts = args.num_experts // _EP_SIZE
        expert_labels = tuple(
            f"owner_rank={expert // local_experts},local_expert={expert % local_experts}"
            for expert in range(args.num_experts)
        )
        parity = {}
        for name in implementations:
            actual_forward = jax.block_until_ready(compiled_forwards[name](*inputs))
            actual_vg = jax.block_until_ready(compiled_value_and_grads[name](*inputs))
            output_parity = _parity_metrics(
                actual_forward[0],
                ring_forward[0],
                group_ids=token_owner_ids,
                group_labels=token_owner_labels,
            )
            dropped = int(actual_forward[1])
            reference_dropped = int(ring_forward[1])
            gradient_groups = (
                (token_owner_ids, token_owner_labels),
                (selected_experts, expert_labels),
                (expert_ids, expert_labels),
                (expert_ids, expert_labels),
            )
            gradient_parity = [
                _parity_metrics(actual, expected, group_ids=group_ids, group_labels=group_labels)
                for actual, expected, (group_ids, group_labels) in zip(
                    actual_vg[1], ring_vg[1], gradient_groups, strict=True
                )
            ]
            parity[name] = {
                "output": output_parity,
                "gradients": dict(zip(("x", "combine_weights", "w13", "w2"), gradient_parity, strict=True)),
                "dropped": dropped,
                "reference_dropped": reference_dropped,
                "dropped_matches": dropped == reference_dropped,
            }
        parity_status = _parity_status(parity, mode=args.parity_mode)

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
        "parity_status": parity_status,
        "promotable": parity_status["promotable"],
        "timings": timings,
    }
    _print_result(result, args.output)


if __name__ == "__main__":
    main()
