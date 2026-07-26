# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Falsify cross-microbatch phase overlap for the exact bulk-ring MoE."""

import argparse
import itertools
import json
import statistics
import time
from collections.abc import Callable
from functools import partial
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import AxisType, Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from levanter.grug._moe.ep_ring import (
    _BulkRingDispatchState,
    _BulkRingExpertState,
    _moe_mlp_ep_ring_combine_local,
    _moe_mlp_ep_ring_dispatch_local,
    _moe_mlp_ep_ring_expert_local,
    _moe_mlp_ep_ring_local,
)

_EP_SIZE = 8
_PARITY_RELATIVE_L2 = 0.002
_VAG_PROMOTION_SPEEDUP = 1.11
_FORWARD_ARM_NAMES = ("queued_full", "phased", "fused_full")
_VAG_ARM_NAMES = ("queued_full", "fused_full")


class _MicrobatchInputs(NamedTuple):
    x: jax.Array
    selected_experts: jax.Array
    combine_weights: jax.Array


class _CompiledArms(NamedTuple):
    full: Callable[..., Any]
    dispatch: Callable[..., Any]
    expert: Callable[..., Any]
    combine: Callable[..., Any]
    fused_full: Callable[..., Any]
    full_value_and_grad: Callable[..., Any]
    fused_full_value_and_grad: Callable[..., Any]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--microbatch-size", type=int, default=32)
    parser.add_argument("--sequence-length", type=int, default=4096)
    parser.add_argument("--hidden-dim", type=int, default=2560)
    parser.add_argument("--intermediate-dim", type=int, default=1280)
    parser.add_argument("--num-experts", type=int, default=64)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--capacity-factor", type=float, default=1.0)
    parser.add_argument("--group-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--lower-only", action="store_true")
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
    if args.group_size < 2:
        raise ValueError(f"group_size must be at least 2, got {args.group_size}")
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
    if tokens * args.top_k % args.num_experts:
        raise ValueError("balanced routing requires assignments to be divisible by num_experts")


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


def _balanced_routes(*, tokens: int, top_k: int, num_experts: int, offset: int) -> jax.Array:
    assignments = jnp.arange(tokens * top_k, dtype=jnp.int32)
    return ((assignments + offset) % num_experts).reshape(tokens, top_k)


def _inputs(
    mesh: Mesh,
    args: argparse.Namespace,
) -> tuple[tuple[_MicrobatchInputs, ...], jax.Array, jax.Array]:
    tokens = args.microbatch_size * args.sequence_length
    batch_sharding = NamedSharding(mesh, P(("data", "expert"), None))
    expert_sharding = NamedSharding(mesh, P("expert", None, None))

    make_x = jax.jit(
        lambda key: jax.random.normal(key, (tokens, args.hidden_dim), dtype=jnp.bfloat16),
        out_shardings=batch_sharding,
    )
    make_combine = jax.jit(
        lambda key: jax.nn.softmax(
            jax.random.normal(key, (tokens, args.top_k), dtype=jnp.float32),
            axis=-1,
        ),
        out_shardings=batch_sharding,
    )
    make_w13 = jax.jit(
        lambda key: 0.02
        * jax.random.normal(
            key,
            (args.num_experts, args.hidden_dim, 2 * args.intermediate_dim),
            dtype=jnp.bfloat16,
        ),
        out_shardings=expert_sharding,
    )
    make_w2 = jax.jit(
        lambda key: 0.02
        * jax.random.normal(
            key,
            (args.num_experts, args.intermediate_dim, args.hidden_dim),
            dtype=jnp.bfloat16,
        ),
        out_shardings=expert_sharding,
    )

    keys = jax.random.split(jax.random.key(args.seed), 2 * args.group_size + 2)
    microbatches = []
    for index in range(args.group_size):
        selected = _balanced_routes(
            tokens=tokens,
            top_k=args.top_k,
            num_experts=args.num_experts,
            offset=index * args.top_k,
        )
        selected = jax.device_put(selected, batch_sharding)
        microbatches.append(
            _MicrobatchInputs(
                x=make_x(keys[2 * index]),
                selected_experts=selected,
                combine_weights=make_combine(keys[2 * index + 1]),
            )
        )
    return tuple(microbatches), make_w13(keys[-2]), make_w2(keys[-1])


def _mapped_functions(
    mesh: Mesh,
    args: argparse.Namespace,
) -> tuple[Callable[..., Any], Callable[..., Any], Callable[..., Any], Callable[..., Any]]:
    batch_spec = P(("data", "expert"), None)
    expert_spec = P("expert", None, None)
    phase_vector_spec = P(("data", "expert"))
    dispatch_spec = _BulkRingDispatchState(
        x_dispatch=P(("data", "expert"), None),
        weight_dispatch=phase_vector_spec,
        token_global=phase_vector_spec,
        group_sizes=phase_vector_spec,
        dropped_local=phase_vector_spec,
    )
    expert_state_spec = _BulkRingExpertState(out_dispatch=P(("data", "expert"), None))
    local_experts = args.num_experts // _EP_SIZE
    tokens_per_shard = args.microbatch_size * args.sequence_length // _EP_SIZE

    full = jax.shard_map(
        partial(
            _moe_mlp_ep_ring_local,
            activation_fn=jax.nn.silu,
            num_experts=args.num_experts,
            capacity_factor=args.capacity_factor,
        ),
        mesh=mesh,
        in_specs=(batch_spec, batch_spec, batch_spec, expert_spec, expert_spec),
        out_specs=(batch_spec, P()),
        check_vma=False,
    )
    dispatch = jax.shard_map(
        partial(
            _moe_mlp_ep_ring_dispatch_local,
            local_experts=local_experts,
            num_experts=args.num_experts,
            capacity_factor=args.capacity_factor,
        ),
        mesh=mesh,
        in_specs=(batch_spec, batch_spec, batch_spec),
        out_specs=dispatch_spec,
        check_vma=False,
    )
    expert = jax.shard_map(
        partial(_moe_mlp_ep_ring_expert_local, activation_fn=jax.nn.silu),
        mesh=mesh,
        in_specs=(dispatch_spec, expert_spec, expert_spec),
        out_specs=expert_state_spec,
        check_vma=False,
    )
    combine = jax.shard_map(
        partial(
            _moe_mlp_ep_ring_combine_local,
            tokens_per_shard=tokens_per_shard,
            expert_axis_size=_EP_SIZE,
        ),
        mesh=mesh,
        in_specs=(dispatch_spec, expert_state_spec),
        out_specs=(batch_spec, P()),
        check_vma=False,
    )
    return full, dispatch, expert, combine


def _phase_abstracts(
    mesh: Mesh,
    args: argparse.Namespace,
) -> tuple[_BulkRingDispatchState, _BulkRingExpertState]:
    tokens = args.microbatch_size * args.sequence_length
    assignments = tokens * args.top_k
    local_experts = args.num_experts // _EP_SIZE
    local_capacity = max(local_experts, int(np.ceil(args.capacity_factor * assignments / _EP_SIZE)))
    phase_vector_sharding = NamedSharding(mesh, P(("data", "expert")))
    phase_matrix_sharding = NamedSharding(mesh, P(("data", "expert"), None))
    global_capacity = local_capacity * _EP_SIZE
    dispatch = _BulkRingDispatchState(
        x_dispatch=jax.ShapeDtypeStruct(
            (global_capacity, args.hidden_dim), jnp.bfloat16, sharding=phase_matrix_sharding
        ),
        weight_dispatch=jax.ShapeDtypeStruct((global_capacity,), jnp.bfloat16, sharding=phase_vector_sharding),
        token_global=jax.ShapeDtypeStruct((global_capacity,), jnp.int32, sharding=phase_vector_sharding),
        group_sizes=jax.ShapeDtypeStruct((args.num_experts,), jnp.int32, sharding=phase_vector_sharding),
        dropped_local=jax.ShapeDtypeStruct((_EP_SIZE,), jnp.int32, sharding=phase_vector_sharding),
    )
    expert = _BulkRingExpertState(
        out_dispatch=jax.ShapeDtypeStruct(
            (global_capacity, args.hidden_dim), jnp.bfloat16, sharding=phase_matrix_sharding
        )
    )
    return dispatch, expert


def _fused_full_group(
    full: Callable[..., Any],
    group: tuple[_MicrobatchInputs, ...],
    w13: jax.Array,
    w2: jax.Array,
) -> tuple[tuple[jax.Array, jax.Array], ...]:
    return tuple(
        full(microbatch.x, microbatch.selected_experts, microbatch.combine_weights, w13, w2) for microbatch in group
    )


def _training_loss(output: jax.Array) -> jax.Array:
    return jnp.mean(jnp.sum(jnp.square(output.astype(jnp.float32)), axis=-1))


def _full_loss_with_aux(
    full: Callable[..., Any],
    x: jax.Array,
    selected_experts: jax.Array,
    combine_weights: jax.Array,
    w13: jax.Array,
    w2: jax.Array,
) -> tuple[jax.Array, tuple[jax.Array, jax.Array]]:
    output, dropped = full(x, selected_experts, combine_weights, w13, w2)
    return _training_loss(output), (output, dropped)


def _fused_full_group_loss(
    full: Callable[..., Any],
    xs: tuple[jax.Array, ...],
    selected_experts: tuple[jax.Array, ...],
    combine_weights: tuple[jax.Array, ...],
    w13: jax.Array,
    w2: jax.Array,
) -> tuple[jax.Array, tuple[tuple[jax.Array, jax.Array, jax.Array], ...]]:
    values = tuple(
        _full_loss_with_aux(full, x, selected, weights, w13, w2)
        for x, selected, weights in zip(xs, selected_experts, combine_weights, strict=True)
    )
    losses = tuple(value[0] for value in values)
    auxiliary = tuple((loss, output, dropped) for loss, (output, dropped) in values)
    return sum(losses, start=jnp.asarray(0.0, dtype=jnp.float32)), auxiliary


def _lowered_arms(
    mesh: Mesh,
    args: argparse.Namespace,
    microbatches: tuple[_MicrobatchInputs, ...],
    w13: jax.Array,
    w2: jax.Array,
) -> dict[str, Any]:
    full_mapped, dispatch_mapped, expert_mapped, combine_mapped = _mapped_functions(mesh, args)
    first = microbatches[0]
    full = jax.jit(full_mapped)
    dispatch = jax.jit(dispatch_mapped)
    expert = jax.jit(expert_mapped)
    combine = jax.jit(combine_mapped)
    fused_full = jax.jit(partial(_fused_full_group, full_mapped))
    full_value_and_grad = jax.jit(
        jax.value_and_grad(
            partial(_full_loss_with_aux, full_mapped),
            argnums=(0, 2, 3, 4),
            has_aux=True,
        )
    )
    fused_full_value_and_grad = jax.jit(
        jax.value_and_grad(
            partial(_fused_full_group_loss, full_mapped),
            argnums=(0, 2, 3, 4),
            has_aux=True,
        )
    )
    xs = tuple(microbatch.x for microbatch in microbatches)
    selected_experts = tuple(microbatch.selected_experts for microbatch in microbatches)
    combine_weights = tuple(microbatch.combine_weights for microbatch in microbatches)

    dispatch_abstract, expert_abstract = _phase_abstracts(mesh, args)
    lowered = {
        "full": full.lower(first.x, first.selected_experts, first.combine_weights, w13, w2),
        "dispatch": dispatch.lower(first.x, first.selected_experts, first.combine_weights),
        "expert": expert.lower(dispatch_abstract, w13, w2),
        "combine": combine.lower(dispatch_abstract, expert_abstract),
        "fused_full": fused_full.lower(microbatches, w13, w2),
        "full_value_and_grad": full_value_and_grad.lower(
            first.x,
            first.selected_experts,
            first.combine_weights,
            w13,
            w2,
        ),
        "fused_full_value_and_grad": fused_full_value_and_grad.lower(
            xs,
            selected_experts,
            combine_weights,
            w13,
            w2,
        ),
    }
    return lowered


def _compile_arms(
    lowered: dict[str, Any],
) -> _CompiledArms:
    return _CompiledArms(
        full=lowered["full"].compile(),
        dispatch=lowered["dispatch"].compile(),
        expert=lowered["expert"].compile(),
        combine=lowered["combine"].compile(),
        fused_full=lowered["fused_full"].compile(),
        full_value_and_grad=lowered["full_value_and_grad"].compile(),
        fused_full_value_and_grad=lowered["fused_full_value_and_grad"].compile(),
    )


def _run_queued_full(
    compiled: _CompiledArms,
    microbatches: tuple[_MicrobatchInputs, ...],
    w13: jax.Array,
    w2: jax.Array,
) -> tuple[tuple[jax.Array, jax.Array], ...]:
    results = tuple(
        compiled.full(microbatch.x, microbatch.selected_experts, microbatch.combine_weights, w13, w2)
        for microbatch in microbatches
    )
    return jax.block_until_ready(results)


def _run_phased(
    compiled: _CompiledArms,
    microbatches: tuple[_MicrobatchInputs, ...],
    w13: jax.Array,
    w2: jax.Array,
) -> tuple[tuple[jax.Array, jax.Array], ...]:
    dispatches = tuple(
        compiled.dispatch(microbatch.x, microbatch.selected_experts, microbatch.combine_weights)
        for microbatch in microbatches
    )
    experts = tuple(compiled.expert(dispatch, w13, w2) for dispatch in dispatches)
    results = tuple(compiled.combine(dispatch, expert) for dispatch, expert in zip(dispatches, experts, strict=True))
    return jax.block_until_ready(results)


def _run_fused_full(
    compiled: _CompiledArms,
    microbatches: tuple[_MicrobatchInputs, ...],
    w13: jax.Array,
    w2: jax.Array,
) -> tuple[tuple[jax.Array, jax.Array], ...]:
    return jax.block_until_ready(compiled.fused_full(microbatches, w13, w2))


def _run_queued_full_value_and_grad(
    compiled: _CompiledArms,
    microbatches: tuple[_MicrobatchInputs, ...],
    w13: jax.Array,
    w2: jax.Array,
) -> tuple[Any, ...]:
    results = tuple(
        compiled.full_value_and_grad(
            microbatch.x,
            microbatch.selected_experts,
            microbatch.combine_weights,
            w13,
            w2,
        )
        for microbatch in microbatches
    )
    return jax.block_until_ready(results)


def _run_fused_full_value_and_grad(
    compiled: _CompiledArms,
    microbatches: tuple[_MicrobatchInputs, ...],
    w13: jax.Array,
    w2: jax.Array,
) -> Any:
    xs = tuple(microbatch.x for microbatch in microbatches)
    selected_experts = tuple(microbatch.selected_experts for microbatch in microbatches)
    combine_weights = tuple(microbatch.combine_weights for microbatch in microbatches)
    result = compiled.fused_full_value_and_grad(xs, selected_experts, combine_weights, w13, w2)
    return jax.block_until_ready(result)


def _tensor_parity_metrics(actual: jax.Array, reference: jax.Array) -> dict[str, Any]:
    actual_f32 = actual.astype(jnp.float32)
    reference_f32 = reference.astype(jnp.float32)
    difference = actual_f32 - reference_f32
    error_l2, reference_l2, finite, bitwise = jax.device_get(
        (
            jnp.linalg.norm(difference),
            jnp.linalg.norm(reference_f32),
            jnp.logical_and(jnp.all(jnp.isfinite(actual_f32)), jnp.all(jnp.isfinite(reference_f32))),
            jnp.all(actual == reference),
        )
    )
    error_l2_value = float(error_l2)
    reference_l2_value = float(reference_l2)
    relative_l2 = (
        error_l2_value / reference_l2_value if reference_l2_value else (0.0 if error_l2_value == 0.0 else float("inf"))
    )
    return {
        "relative_l2_error": relative_l2,
        "finite": bool(finite),
        "bitwise_equal": bool(bitwise),
        "accepted": bool(finite) and relative_l2 <= _PARITY_RELATIVE_L2,
    }


def _parity_metrics(
    actual: tuple[tuple[jax.Array, jax.Array], ...],
    reference: tuple[tuple[jax.Array, jax.Array], ...],
) -> dict[str, Any]:
    microbatches = []
    for actual_result, reference_result in zip(actual, reference, strict=True):
        actual_output, actual_dropped = actual_result
        reference_output, reference_dropped = reference_result
        output_metrics = _tensor_parity_metrics(actual_output, reference_output)
        dropped = int(actual_dropped)
        expected_dropped = int(reference_dropped)
        microbatches.append(
            {
                **output_metrics,
                "dropped": dropped,
                "reference_dropped": expected_dropped,
                "dropped_matches": dropped == expected_dropped,
                "accepted": output_metrics["accepted"] and dropped == expected_dropped,
            }
        )
    return {
        "microbatches": microbatches,
        "max_relative_l2_error": max(item["relative_l2_error"] for item in microbatches),
        "all_bitwise_equal": all(item["bitwise_equal"] for item in microbatches),
        "passed": all(item["accepted"] for item in microbatches),
        "relative_l2_threshold": _PARITY_RELATIVE_L2,
    }


def _ordered_sum(values: tuple[jax.Array, ...]) -> jax.Array:
    total = jnp.zeros_like(values[0])
    for value in values:
        total = total + value
    return total


def _value_and_grad_parity_metrics(fused: Any, queued: tuple[Any, ...]) -> dict[str, Any]:
    (fused_total_loss, fused_auxiliary), fused_gradients = fused
    fused_x_gradients, fused_combine_gradients, fused_w13_gradient, fused_w2_gradient = fused_gradients

    queued_losses = tuple(result[0][0] for result in queued)
    queued_outputs = tuple(result[0][1][0] for result in queued)
    queued_dropped = tuple(result[0][1][1] for result in queued)
    queued_x_gradients = tuple(result[1][0] for result in queued)
    queued_combine_gradients = tuple(result[1][1] for result in queued)
    queued_w13_gradients = tuple(result[1][2] for result in queued)
    queued_w2_gradients = tuple(result[1][3] for result in queued)

    fused_losses = tuple(auxiliary[0] for auxiliary in fused_auxiliary)
    fused_outputs = tuple(auxiliary[1] for auxiliary in fused_auxiliary)
    fused_dropped = tuple(auxiliary[2] for auxiliary in fused_auxiliary)
    queued_total_loss = _ordered_sum(queued_losses)

    loss_metrics = {
        "total": _tensor_parity_metrics(fused_total_loss, queued_total_loss),
        "microbatches": [
            _tensor_parity_metrics(actual, reference)
            for actual, reference in zip(fused_losses, queued_losses, strict=True)
        ],
    }
    output_metrics = []
    for actual, reference, dropped, reference_dropped in zip(
        fused_outputs,
        queued_outputs,
        fused_dropped,
        queued_dropped,
        strict=True,
    ):
        metrics = _tensor_parity_metrics(actual, reference)
        dropped_value = int(dropped)
        reference_dropped_value = int(reference_dropped)
        output_metrics.append(
            {
                **metrics,
                "dropped": dropped_value,
                "reference_dropped": reference_dropped_value,
                "dropped_matches": dropped_value == reference_dropped_value,
                "accepted": metrics["accepted"] and dropped_value == reference_dropped_value,
            }
        )
    gradient_metrics = {
        "x": [
            _tensor_parity_metrics(actual, reference)
            for actual, reference in zip(fused_x_gradients, queued_x_gradients, strict=True)
        ],
        "combine_weights": [
            _tensor_parity_metrics(actual, reference)
            for actual, reference in zip(fused_combine_gradients, queued_combine_gradients, strict=True)
        ],
        "w13": _tensor_parity_metrics(fused_w13_gradient, _ordered_sum(queued_w13_gradients)),
        "w2": _tensor_parity_metrics(fused_w2_gradient, _ordered_sum(queued_w2_gradients)),
    }
    required_metrics = [
        loss_metrics["total"],
        *loss_metrics["microbatches"],
        *output_metrics,
        *gradient_metrics["x"],
        *gradient_metrics["combine_weights"],
        gradient_metrics["w13"],
        gradient_metrics["w2"],
    ]
    return {
        "loss": loss_metrics,
        "outputs": output_metrics,
        "gradients": gradient_metrics,
        "passed": all(metrics["accepted"] for metrics in required_metrics),
        "max_relative_l2_error": max(metrics["relative_l2_error"] for metrics in required_metrics),
        "all_finite": all(metrics["finite"] for metrics in required_metrics),
        "relative_l2_threshold": _PARITY_RELATIVE_L2,
        "shared_gradient_reference": "ordered_sum_of_queued_microbatch_gradients",
    }


def _timing_summary(durations: list[float]) -> dict[str, Any]:
    quantiles = np.quantile(np.asarray(durations), (0.1, 0.5, 0.9))
    return {
        "samples": len(durations),
        "mean_ms": 1000.0 * statistics.fmean(durations),
        "p10_ms": 1000.0 * float(quantiles[0]),
        "p50_ms": 1000.0 * float(quantiles[1]),
        "p90_ms": 1000.0 * float(quantiles[2]),
    }


def _time_arms(
    runners: dict[str, Callable[[], Any]],
    *,
    arm_names: tuple[str, ...],
    warmup: int,
    iterations: int,
) -> tuple[dict[str, dict[str, Any]], list[list[str]]]:
    if set(runners) != set(arm_names):
        raise ValueError(f"runner arms {tuple(runners)} do not match timing arms {arm_names}")
    orders = tuple(itertools.permutations(arm_names))
    for index in range(warmup):
        for arm in orders[index % len(orders)]:
            runners[arm]()

    durations = {arm: [] for arm in arm_names}
    executed_orders = []
    for index in range(iterations):
        order = orders[index % len(orders)]
        executed_orders.append(list(order))
        for arm in order:
            start = time.perf_counter()
            runners[arm]()
            durations[arm].append(time.perf_counter() - start)
    return {arm: _timing_summary(samples) for arm, samples in durations.items()}, executed_orders


def _speedups_vs_queued_full(
    timings: dict[str, dict[str, Any]],
    candidates: tuple[str, ...],
) -> dict[str, float]:
    baseline = timings["queued_full"]["p50_ms"]
    return {arm: baseline / timings[arm]["p50_ms"] for arm in candidates}


def _promotion(
    value_and_grad_parity: dict[str, Any],
    value_and_grad_timings: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    speedup = _speedups_vs_queued_full(value_and_grad_timings, ("fused_full",))["fused_full"]
    passed = value_and_grad_parity["passed"] and speedup >= _VAG_PROMOTION_SPEEDUP
    return {
        "passed": passed,
        "promoted_arm": "fused_full" if passed else None,
        "basis": "value_and_grad_p50",
        "required_speedup": _VAG_PROMOTION_SPEEDUP,
        "speedup_vs_queued_full": speedup,
        "parity_passed": value_and_grad_parity["passed"],
        "phased_forward_can_promote": False,
    }


def main() -> None:
    args = _parser().parse_args()
    _validate_args(args)
    mesh = _mesh()
    if not args.lower_only and jax.default_backend() != "gpu":
        raise RuntimeError("timing requires eight local GPUs; use --lower-only for a CPU lowering smoke")

    with jax.set_mesh(mesh):
        microbatches, w13, w2 = _inputs(mesh, args)
        lowered = _lowered_arms(mesh, args, microbatches, w13, w2)
        if args.lower_only:
            print(
                json.dumps(
                    {
                        "event": "ep_ring_microbatch_overlap",
                        "backend": jax.default_backend(),
                        "devices": _EP_SIZE,
                        "group_size": args.group_size,
                        "lower_only": True,
                        "lowered": sorted(lowered),
                    },
                    sort_keys=True,
                )
            )
            return

        compiled = _compile_arms(lowered)
        queued_reference = _run_queued_full(compiled, microbatches, w13, w2)
        phased_result = _run_phased(compiled, microbatches, w13, w2)
        fused_result = _run_fused_full(compiled, microbatches, w13, w2)
        forward_parity = {
            "phased": _parity_metrics(phased_result, queued_reference),
            "fused_full": _parity_metrics(fused_result, queued_reference),
        }
        queued_value_and_grad = _run_queued_full_value_and_grad(compiled, microbatches, w13, w2)
        fused_value_and_grad = _run_fused_full_value_and_grad(compiled, microbatches, w13, w2)
        value_and_grad_parity = _value_and_grad_parity_metrics(fused_value_and_grad, queued_value_and_grad)

        forward_runners = {
            "queued_full": partial(_run_queued_full, compiled, microbatches, w13, w2),
            "phased": partial(_run_phased, compiled, microbatches, w13, w2),
            "fused_full": partial(_run_fused_full, compiled, microbatches, w13, w2),
        }
        forward_timings, forward_arm_orders = _time_arms(
            forward_runners,
            arm_names=_FORWARD_ARM_NAMES,
            warmup=args.warmup,
            iterations=args.iterations,
        )
        value_and_grad_runners = {
            "queued_full": partial(_run_queued_full_value_and_grad, compiled, microbatches, w13, w2),
            "fused_full": partial(_run_fused_full_value_and_grad, compiled, microbatches, w13, w2),
        }
        value_and_grad_timings, value_and_grad_arm_orders = _time_arms(
            value_and_grad_runners,
            arm_names=_VAG_ARM_NAMES,
            warmup=args.warmup,
            iterations=args.iterations,
        )
        promotion = _promotion(value_and_grad_parity, value_and_grad_timings)
        speedups = {
            "forward": _speedups_vs_queued_full(forward_timings, ("phased", "fused_full")),
            "value_and_grad": _speedups_vs_queued_full(value_and_grad_timings, ("fused_full",)),
        }

    result = {
        "event": "ep_ring_microbatch_overlap",
        "backend": jax.default_backend(),
        "device_kind": jax.devices()[0].device_kind,
        "devices": _EP_SIZE,
        "microbatch_size": args.microbatch_size,
        "sequence_length": args.sequence_length,
        "tokens_per_microbatch": args.microbatch_size * args.sequence_length,
        "group_size": args.group_size,
        "hidden_dim": args.hidden_dim,
        "intermediate_dim": args.intermediate_dim,
        "num_experts": args.num_experts,
        "top_k": args.top_k,
        "capacity_factor": args.capacity_factor,
        "dtype": "bfloat16",
        "combine_dtype": "bfloat16",
        "routing": "balanced_source_major",
        "inputs": "distinct_microbatches_shared_identically_across_arms",
        "seed": args.seed,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "loss": "mean_tokens(sum_hidden(square(output_fp32)))",
        "parity": {
            "forward": forward_parity,
            "value_and_grad": {"fused_full": value_and_grad_parity},
        },
        "timings": {
            "forward": forward_timings,
            "value_and_grad": value_and_grad_timings,
        },
        "speedup_vs_queued_full": speedups,
        "arm_orders": {
            "forward": forward_arm_orders,
            "value_and_grad": value_and_grad_arm_orders,
        },
        "promotion": promotion,
    }
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
