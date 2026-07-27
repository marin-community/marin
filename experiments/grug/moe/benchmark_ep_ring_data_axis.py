# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compare EP8 with replicated-expert layouts and timed FP32 accumulation."""

import argparse
import json
import re
import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from haliax.nn import ragged_dot
from jax.sharding import AxisType, Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from levanter.grug._moe.common import MoeRaggedDotOps
from levanter.grug._moe.ep_ring import (
    _moe_mlp_ep_ring_local,
    _moe_mlp_ep_ring_local_accumulating_weight_gradient,
)

from experiments.grug.moe.benchmark_ep_ring import _selected_experts

_DEVICE_COUNT = 8
_PARITY_RELATIVE_L2 = 0.002
_BASELINE_MFU = 18.2583
_BASELINE_STEP_SECONDS = 81.037785
_INTERSTAGE_FP8_SPEEDUP = 1.0179
_PROMOTION_MFU = 20.0
_LAYERS_PER_STAGE = 6
_MICROBATCHES_PER_STEP = 256
_COLLECTIVE_NAMES = ("all-gather", "all-reduce", "reduce-scatter", "collective-permute")


@dataclass(frozen=True)
class _Arm:
    name: str
    data_axis_size: int
    expert_axis_size: int
    mesh: Mesh
    batch_sharding: NamedSharding
    expert_sharding: NamedSharding
    storage_sharding: NamedSharding
    local_gradient_sharding: NamedSharding
    inputs: tuple[jax.Array, ...]
    storage_weights: tuple[jax.Array, jax.Array]
    initialize_accumulators: jax.stages.Compiled
    forward: jax.stages.Compiled
    value_and_grad: jax.stages.Compiled
    sync_gradients: jax.stages.Compiled
    materialize_weights: jax.stages.Compiled
    materialize_gradients: jax.stages.Compiled
    hlo: dict[str, Any]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--microbatch-size", type=int, default=32)
    parser.add_argument("--sequence-length", type=int, default=4096)
    parser.add_argument("--hidden-dim", type=int, default=2560)
    parser.add_argument("--intermediate-dim", type=int, default=1280)
    parser.add_argument("--num-experts", type=int, default=64)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--capacity-factor", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--treatment-data-axis-size", type=int, choices=(2, 4), default=2)
    parser.add_argument("--microbatches-per-step", type=int, default=_MICROBATCHES_PER_STEP)
    parser.add_argument("--layers-per-stage", type=int, default=_LAYERS_PER_STAGE)
    parser.add_argument("--baseline-mfu", type=float, default=_BASELINE_MFU)
    parser.add_argument("--baseline-step-seconds", type=float, default=_BASELINE_STEP_SECONDS)
    parser.add_argument("--interstage-speedup", type=float, default=_INTERSTAGE_FP8_SPEEDUP)
    parser.add_argument("--promotion-mfu", type=float, default=_PROMOTION_MFU)
    parser.add_argument("--fuse-fp32-weight-gradient-accumulation", action="store_true")
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
        "microbatches_per_step": args.microbatches_per_step,
        "layers_per_stage": args.layers_per_stage,
        "baseline_mfu": args.baseline_mfu,
        "baseline_step_seconds": args.baseline_step_seconds,
        "interstage_speedup": args.interstage_speedup,
        "promotion_mfu": args.promotion_mfu,
    }
    for name, value in positive.items():
        if value <= 0:
            raise ValueError(f"{name} must be positive, got {value}")
    if args.warmup < 0:
        raise ValueError(f"warmup must be non-negative, got {args.warmup}")
    if args.capacity_factor <= 0:
        raise ValueError(f"capacity_factor must be positive, got {args.capacity_factor}")
    if args.top_k > args.num_experts:
        raise ValueError(f"top_k={args.top_k} must be <= num_experts={args.num_experts}")
    tokens = args.microbatch_size * args.sequence_length
    if tokens % _DEVICE_COUNT:
        raise ValueError(f"microbatch tokens={tokens} must be divisible by {_DEVICE_COUNT}")
    treatment_expert_axis_size = _DEVICE_COUNT // args.treatment_data_axis_size
    for expert_axis_size in (8, treatment_expert_axis_size):
        if args.num_experts % expert_axis_size:
            raise ValueError(f"num_experts={args.num_experts} must be divisible by expert axis size={expert_axis_size}")
    if tokens * args.top_k % args.num_experts:
        raise ValueError("balanced routing requires total assignments to be divisible by num_experts")


def _mesh(data_axis_size: int) -> Mesh:
    if _DEVICE_COUNT % data_axis_size:
        raise ValueError(f"device count {_DEVICE_COUNT} must be divisible by data axis size {data_axis_size}")
    local_devices = jax.local_devices()
    if len(local_devices) < _DEVICE_COUNT:
        raise RuntimeError(f"benchmark requires eight local devices, found {len(local_devices)}")
    expert_axis_size = _DEVICE_COUNT // data_axis_size
    devices = np.asarray(local_devices[:_DEVICE_COUNT]).reshape(data_axis_size, expert_axis_size, 1)
    return Mesh(
        devices,
        axis_names=("data", "expert", "model"),
        axis_types=(AxisType.Explicit, AxisType.Explicit, AxisType.Explicit),
    )


def _local_forward(
    x_local: jax.Array,
    selected_experts_local: jax.Array,
    combine_weights_local: jax.Array,
    w13_local: jax.Array,
    w2_local: jax.Array,
    *,
    num_experts: int,
    capacity_factor: float,
    output_elements: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    output, dropped = _moe_mlp_ep_ring_local(
        x_local,
        selected_experts_local,
        combine_weights_local,
        w13_local,
        w2_local,
        activation_fn=jax.nn.silu,
        num_experts=num_experts,
        capacity_factor=capacity_factor,
    )
    local_loss = jnp.sum(jnp.square(output.astype(jnp.float32))) / output_elements
    return local_loss[None], output, dropped[None]


class _Fp32WeightGradientRaggedDotOp(eqx.Module):
    def __call__(
        self,
        lhs: jax.Array,
        rhs: jax.Array,
        group_sizes: jax.Array,
    ) -> jax.Array:
        return ragged_dot(
            lhs,
            rhs,
            group_sizes,
            implementation="triton",
            fp32_weight_gradient=True,
        )


_FP32_WEIGHT_GRADIENT_OPS = MoeRaggedDotOps(
    w13=_Fp32WeightGradientRaggedDotOp(),
    w2=_Fp32WeightGradientRaggedDotOp(),
)


def _local_value_and_grad(
    x_local: jax.Array,
    selected_experts_local: jax.Array,
    combine_weights_local: jax.Array,
    w13_local: jax.Array,
    w2_local: jax.Array,
    w13_accumulator: jax.Array,
    w2_accumulator: jax.Array,
    *,
    num_experts: int,
    capacity_factor: float,
    output_elements: int,
    ops: MoeRaggedDotOps | None,
    fuse_weight_gradient_accumulation: bool,
) -> tuple[
    tuple[jax.Array, jax.Array, jax.Array],
    tuple[jax.Array, jax.Array, jax.Array, jax.Array],
]:
    def loss(
        x: jax.Array,
        combine_weights: jax.Array,
        w13: jax.Array,
        w2: jax.Array,
    ) -> tuple[jax.Array, tuple[jax.Array, jax.Array]]:
        if fuse_weight_gradient_accumulation:
            output, dropped, accumulation_token = _moe_mlp_ep_ring_local_accumulating_weight_gradient(
                x,
                selected_experts_local,
                combine_weights,
                w13,
                w2,
                w13_accumulator[0],
                w2_accumulator[0],
                activation_fn=jax.nn.silu,
                num_experts=num_experts,
                capacity_factor=capacity_factor,
            )
        else:
            output, dropped = _moe_mlp_ep_ring_local(
                x,
                selected_experts_local,
                combine_weights,
                w13,
                w2,
                activation_fn=jax.nn.silu,
                num_experts=num_experts,
                capacity_factor=capacity_factor,
                ops=ops,
            )
            accumulation_token = jnp.zeros((), dtype=jnp.float32)
        local_loss = jnp.sum(jnp.square(output.astype(jnp.float32))) / output_elements + accumulation_token
        return local_loss, (output, dropped)

    (local_loss, (output, dropped)), gradients = jax.value_and_grad(
        loss,
        argnums=(0, 1, 2, 3),
        has_aux=True,
    )(x_local, combine_weights_local, w13_local, w2_local)
    x_gradient, combine_gradient, w13_gradient, w2_gradient = gradients
    if fuse_weight_gradient_accumulation:
        w13_accumulator = w13_gradient[None]
        w2_accumulator = w2_gradient[None]
    else:
        w13_accumulator = w13_accumulator + w13_gradient[None].astype(jnp.float32)
        w2_accumulator = w2_accumulator + w2_gradient[None].astype(jnp.float32)
    return (
        (local_loss[None], output, dropped[None]),
        (x_gradient, combine_gradient, w13_accumulator, w2_accumulator),
    )


def _sync_local_gradients(
    w13_gradient: jax.Array,
    w2_gradient: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    return (
        jax.lax.psum_scatter(
            w13_gradient[0],
            "data",
            scatter_dimension=1,
            tiled=True,
        ),
        jax.lax.psum_scatter(
            w2_gradient[0],
            "data",
            scatter_dimension=1,
            tiled=True,
        ),
    )


def _materialize_local_weights(
    w13_storage: jax.Array,
    w2_storage: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    return (
        jax.lax.all_gather(w13_storage, "data", axis=1, tiled=True),
        jax.lax.all_gather(w2_storage, "data", axis=1, tiled=True),
    )


def _build_arm(
    *,
    name: str,
    data_axis_size: int,
    source_inputs: tuple[jax.Array, ...],
    num_experts: int,
    capacity_factor: float,
    output_elements: int,
    fuse_weight_gradient_accumulation: bool,
) -> _Arm:
    mesh = _mesh(data_axis_size)
    expert_axis_size = _DEVICE_COUNT // data_axis_size
    batch_spec = P(("data", "expert"), None)
    rank_vector_spec = P(("data", "expert"))
    expert_spec = P("expert", None, None)
    local_gradient_spec = P("data", "expert", None, None)
    storage_spec = P("expert", "data", None)
    batch_sharding = NamedSharding(mesh, batch_spec)
    expert_sharding = NamedSharding(mesh, expert_spec)
    local_gradient_sharding = NamedSharding(mesh, local_gradient_spec)
    storage_sharding = NamedSharding(mesh, storage_spec)
    x, selected_experts, combine_weights, w13, w2 = source_inputs
    inputs = (
        jax.device_put(x, batch_sharding),
        jax.device_put(selected_experts, batch_sharding),
        jax.device_put(combine_weights, batch_sharding),
        jax.device_put(w13, expert_sharding),
        jax.device_put(w2, expert_sharding),
    )
    storage_weights = (
        jax.device_put(w13, storage_sharding),
        jax.device_put(w2, storage_sharding),
    )
    accumulator_shapes = (
        (data_axis_size, num_experts, w13.shape[1], w13.shape[2]),
        (data_axis_size, num_experts, w2.shape[1], w2.shape[2]),
    )

    local_forward = partial(
        _local_forward,
        num_experts=num_experts,
        capacity_factor=capacity_factor,
        output_elements=output_elements,
    )
    local_value_and_grad = partial(
        _local_value_and_grad,
        num_experts=num_experts,
        capacity_factor=capacity_factor,
        output_elements=output_elements,
        ops=_FP32_WEIGHT_GRADIENT_OPS if data_axis_size > 1 and not fuse_weight_gradient_accumulation else None,
        fuse_weight_gradient_accumulation=fuse_weight_gradient_accumulation,
    )
    with jax.set_mesh(mesh):
        initialize_accumulators_lowered = jax.jit(
            lambda: (
                jnp.zeros(accumulator_shapes[0], dtype=jnp.float32),
                jnp.zeros(accumulator_shapes[1], dtype=jnp.float32),
            ),
            out_shardings=(local_gradient_sharding, local_gradient_sharding),
        ).lower()
        initialize_accumulators = initialize_accumulators_lowered.compile()
        accumulators = initialize_accumulators()
        mapped_forward = jax.shard_map(
            local_forward,
            mesh=mesh,
            in_specs=(batch_spec, batch_spec, batch_spec, expert_spec, expert_spec),
            out_specs=(rank_vector_spec, batch_spec, rank_vector_spec),
            check_vma=False,
        )
        mapped_value_and_grad = jax.shard_map(
            local_value_and_grad,
            mesh=mesh,
            in_specs=(
                batch_spec,
                batch_spec,
                batch_spec,
                expert_spec,
                expert_spec,
                local_gradient_spec,
                local_gradient_spec,
            ),
            out_specs=(
                (rank_vector_spec, batch_spec, rank_vector_spec),
                (batch_spec, batch_spec, local_gradient_spec, local_gradient_spec),
            ),
            check_vma=False,
        )
        mapped_sync = jax.shard_map(
            _sync_local_gradients,
            mesh=mesh,
            in_specs=(local_gradient_spec, local_gradient_spec),
            out_specs=(storage_spec, storage_spec),
            check_vma=False,
        )
        mapped_materialize = jax.shard_map(
            _materialize_local_weights,
            mesh=mesh,
            in_specs=(storage_spec, storage_spec),
            out_specs=(expert_spec, expert_spec),
            check_vma=False,
        )
        forward_lowered = jax.jit(mapped_forward).lower(*inputs)
        value_and_grad_lowered = jax.jit(
            mapped_value_and_grad,
            donate_argnums=(5, 6),
        ).lower(*inputs, *accumulators)
        sync_lowered = jax.jit(mapped_sync).lower(
            jax.ShapeDtypeStruct(
                (data_axis_size, num_experts, w13.shape[1], w13.shape[2]),
                jnp.float32,
                sharding=local_gradient_sharding,
            ),
            jax.ShapeDtypeStruct(
                (data_axis_size, num_experts, w2.shape[1], w2.shape[2]),
                jnp.float32,
                sharding=local_gradient_sharding,
            ),
        )
        materialize_lowered = jax.jit(mapped_materialize).lower(*storage_weights)
        materialize_gradients_lowered = jax.jit(mapped_materialize).lower(
            jax.ShapeDtypeStruct(
                w13.shape,
                jnp.float32,
                sharding=storage_sharding,
            ),
            jax.ShapeDtypeStruct(
                w2.shape,
                jnp.float32,
                sharding=storage_sharding,
            ),
        )
        hlo = {
            "local_vag": _collective_summary(
                value_and_grad_lowered.as_text(),
                data_axis_size=data_axis_size,
                expert_axis_size=expert_axis_size,
            ),
            "sync": _collective_summary(
                sync_lowered.as_text(),
                data_axis_size=data_axis_size,
                expert_axis_size=expert_axis_size,
            ),
            "materialize": _collective_summary(
                materialize_lowered.as_text(),
                data_axis_size=data_axis_size,
                expert_axis_size=expert_axis_size,
            ),
        }
        forward = forward_lowered.compile()
        value_and_grad = value_and_grad_lowered.compile()
        sync_gradients = sync_lowered.compile()
        materialize_weights = materialize_lowered.compile()
        materialize_gradients = materialize_gradients_lowered.compile()

    return _Arm(
        name=name,
        data_axis_size=data_axis_size,
        expert_axis_size=expert_axis_size,
        mesh=mesh,
        batch_sharding=batch_sharding,
        expert_sharding=expert_sharding,
        storage_sharding=storage_sharding,
        local_gradient_sharding=local_gradient_sharding,
        inputs=inputs,
        storage_weights=storage_weights,
        initialize_accumulators=initialize_accumulators,
        forward=forward,
        value_and_grad=value_and_grad,
        sync_gradients=sync_gradients,
        materialize_weights=materialize_weights,
        materialize_gradients=materialize_gradients,
        hlo=hlo,
    )


def _time_alternating(
    control: tuple[Callable[..., Any], tuple[Any, ...]],
    treatment: tuple[Callable[..., Any], tuple[Any, ...]],
    *,
    warmup: int,
    iterations: int,
) -> dict[str, dict[str, float]]:
    arms = {"control": control, "treatment": treatment}
    for _ in range(warmup):
        for function, function_args in arms.values():
            jax.block_until_ready(function(*function_args))
    durations = {name: [] for name in arms}
    for iteration in range(iterations):
        order = tuple(arms) if iteration % 2 == 0 else tuple(reversed(arms))
        for name in order:
            function, function_args = arms[name]
            start = time.perf_counter()
            jax.block_until_ready(function(*function_args))
            durations[name].append(time.perf_counter() - start)
    return {
        name: {
            "mean_ms": 1000.0 * statistics.fmean(samples),
            "median_ms": 1000.0 * statistics.median(samples),
            "min_ms": 1000.0 * min(samples),
            "max_ms": 1000.0 * max(samples),
        }
        for name, samples in durations.items()
    }


def _time_alternating_accumulating(
    control: _Arm,
    treatment: _Arm,
    *,
    warmup: int,
    iterations: int,
) -> dict[str, dict[str, float]]:
    arms = {"control": control, "treatment": treatment}
    accumulators = {name: arm.initialize_accumulators() for name, arm in arms.items()}

    def run(name: str) -> None:
        arm = arms[name]
        result = arm.value_and_grad(*arm.inputs, *accumulators[name])
        jax.block_until_ready(result)
        accumulators[name] = (result[1][2], result[1][3])

    for _ in range(warmup):
        for name in arms:
            run(name)

    accumulators = {name: arm.initialize_accumulators() for name, arm in arms.items()}
    durations = {name: [] for name in arms}
    for iteration in range(iterations):
        order = tuple(arms) if iteration % 2 == 0 else tuple(reversed(arms))
        for name in order:
            start = time.perf_counter()
            run(name)
            durations[name].append(time.perf_counter() - start)
    return {
        name: {
            "mean_ms": 1000.0 * statistics.fmean(samples),
            "median_ms": 1000.0 * statistics.median(samples),
            "min_ms": 1000.0 * min(samples),
            "max_ms": 1000.0 * max(samples),
        }
        for name, samples in durations.items()
    }


def _time(
    function: Callable[..., Any],
    function_args: tuple[Any, ...],
    *,
    warmup: int,
    iterations: int,
) -> dict[str, float]:
    for _ in range(warmup):
        jax.block_until_ready(function(*function_args))
    durations = []
    for _ in range(iterations):
        start = time.perf_counter()
        jax.block_until_ready(function(*function_args))
        durations.append(time.perf_counter() - start)
    return {
        "mean_ms": 1000.0 * statistics.fmean(durations),
        "median_ms": 1000.0 * statistics.median(durations),
        "min_ms": 1000.0 * min(durations),
        "max_ms": 1000.0 * max(durations),
    }


def _tensor_metrics(actual: jax.Array, reference: jax.Array) -> dict[str, float | bool]:
    actual = jax.device_put(actual, reference.sharding)

    @jax.jit
    def metrics(candidate: jax.Array, expected: jax.Array):
        candidate_f32 = candidate.astype(jnp.float32)
        expected_f32 = expected.astype(jnp.float32)
        difference = candidate_f32 - expected_f32
        reference_l2 = jnp.linalg.norm(expected_f32)
        candidate_l2 = jnp.linalg.norm(candidate_f32)
        difference_l2 = jnp.linalg.norm(difference)
        relative_l2 = jnp.where(reference_l2 > 0, difference_l2 / reference_l2, difference_l2)
        return (
            jnp.all(jnp.isfinite(candidate_f32)),
            reference_l2,
            candidate_l2,
            difference_l2,
            relative_l2,
            jnp.max(jnp.abs(difference)),
        )

    finite, reference_l2, candidate_l2, difference_l2, relative_l2, max_absolute_error = jax.device_get(
        metrics(actual, reference)
    )
    relative_l2_value = float(relative_l2)
    return {
        "finite": bool(finite),
        "reference_l2": float(reference_l2),
        "candidate_l2": float(candidate_l2),
        "difference_l2": float(difference_l2),
        "relative_l2_error": relative_l2_value,
        "max_absolute_error": float(max_absolute_error),
        "passed": bool(finite) and relative_l2_value <= _PARITY_RELATIVE_L2,
    }


def _replica_groups(data_axis_size: int, expert_axis_size: int, *, axis: str) -> str:
    if axis == "expert":
        groups = [
            [data_index * expert_axis_size + expert_index for expert_index in range(expert_axis_size)]
            for data_index in range(data_axis_size)
        ]
    elif axis == "data":
        groups = [
            [data_index * expert_axis_size + expert_index for data_index in range(data_axis_size)]
            for expert_index in range(expert_axis_size)
        ]
    else:
        raise ValueError(f"unknown mesh axis {axis!r}")
    return "{" + ",".join("{" + ",".join(str(rank) for rank in group) + "}" for group in groups) + "}"


def _replica_groups_bracketed(data_axis_size: int, expert_axis_size: int, *, axis: str) -> str:
    return _replica_groups(data_axis_size, expert_axis_size, axis=axis).replace("{", "[").replace("}", "]")


def _collective_summary(hlo_text: str, *, data_axis_size: int, expert_axis_size: int) -> dict[str, Any]:
    lines = []
    for line in hlo_text.splitlines():
        normalized = re.sub(r"\s+", "", line).replace("-", "_")
        if any(name.replace("-", "_") in normalized for name in _COLLECTIVE_NAMES):
            lines.append(normalized)
    data_replica_groups = _replica_groups(data_axis_size, expert_axis_size, axis="data")
    expert_replica_groups = _replica_groups(data_axis_size, expert_axis_size, axis="expert")
    data_replica_groups_bracketed = _replica_groups_bracketed(data_axis_size, expert_axis_size, axis="data")
    expert_replica_groups_bracketed = _replica_groups_bracketed(data_axis_size, expert_axis_size, axis="expert")

    def uses_groups(line: str, curly: str, bracketed: str) -> bool:
        return f"replica_groups={curly}" in line or bracketed in line

    return {
        "counts": {name: sum(name.replace("-", "_") in line for line in lines) for name in _COLLECTIVE_NAMES},
        "data_axis_collective_count": sum(
            uses_groups(line, data_replica_groups, data_replica_groups_bracketed) for line in lines
        ),
        "expert_axis_collective_count": sum(
            uses_groups(line, expert_replica_groups, expert_replica_groups_bracketed) for line in lines
        ),
        "data_replica_groups": data_replica_groups,
        "expert_replica_groups": expert_replica_groups,
    }


def _memory_summary(compiled: jax.stages.Compiled) -> dict[str, int | None]:
    memory = compiled.memory_analysis()
    return {
        "argument_size_in_bytes": memory.argument_size_in_bytes,
        "output_size_in_bytes": memory.output_size_in_bytes,
        "alias_size_in_bytes": memory.alias_size_in_bytes,
        "temp_size_in_bytes": memory.temp_size_in_bytes,
        "peak_memory_in_bytes": memory.peak_memory_in_bytes,
        "generated_code_size_in_bytes": memory.generated_code_size_in_bytes,
    }


def _projection(
    *,
    control_vag_ms: float,
    treatment_vag_ms: float,
    sync_ms: float,
    materialize_ms: float,
    microbatches_per_step: int,
    layers_per_stage: int,
    baseline_step_seconds: float,
    baseline_mfu: float,
    interstage_speedup: float,
) -> dict[str, float]:
    amortized_overhead_ms = (sync_ms + materialize_ms) / microbatches_per_step
    treatment_amortized_vag_ms = treatment_vag_ms + amortized_overhead_ms
    saving_per_layer_microbatch_ms = control_vag_ms - treatment_amortized_vag_ms
    projected_step_seconds = baseline_step_seconds - (
        layers_per_stage * microbatches_per_step * saving_per_layer_microbatch_ms / 1000.0
    )
    projected_mfu = baseline_mfu * baseline_step_seconds / projected_step_seconds
    composed_step_seconds = projected_step_seconds / interstage_speedup
    composed_mfu = baseline_mfu * baseline_step_seconds / composed_step_seconds
    return {
        "amortized_step_boundary_overhead_ms": amortized_overhead_ms,
        "treatment_amortized_vag_ms": treatment_amortized_vag_ms,
        "saving_per_layer_microbatch_ms": saving_per_layer_microbatch_ms,
        "projected_step_seconds": projected_step_seconds,
        "projected_mfu": projected_mfu,
        "projected_speedup": baseline_step_seconds / projected_step_seconds,
        "interstage_speedup": interstage_speedup,
        "composed_step_seconds": composed_step_seconds,
        "composed_mfu": composed_mfu,
        "composed_speedup": baseline_step_seconds / composed_step_seconds,
    }


def _print_result(result: dict[str, Any], output: str) -> None:
    if output in ("human", "both"):
        if result.get("lower_only"):
            print(
                "Data-axis ring lowering: "
                f"local_vag_data_collectives={result['hlo']['treatment_local_vag']['data_axis_collective_count']}, "
                f"sync_data_collectives={result['hlo']['treatment_sync']['data_axis_collective_count']}, "
                f"materialize_data_collectives={result['hlo']['treatment_materialize']['data_axis_collective_count']}"
            )
        else:
            print(
                f"EP8 vs EP{result['treatment']['expert_axis_size']}/data{result['treatment']['data_axis_size']}: "
                f"parity={result['parity']['passed']}, "
                f"projected_mfu={result['projection']['projected_mfu']:.4f}, "
                f"composed_mfu={result['projection']['composed_mfu']:.4f}, "
                f"promotable={result['promotable']}"
            )
            print(
                "  forward median: "
                f"EP8={result['timings']['forward']['control']['median_ms']:.3f} ms, "
                f"treatment={result['timings']['forward']['treatment']['median_ms']:.3f} ms"
            )
            print(
                "  local VAG + FP32 accumulation median: "
                f"EP8={result['timings']['value_and_grad']['control']['median_ms']:.3f} ms, "
                f"treatment={result['timings']['value_and_grad']['treatment']['median_ms']:.3f} ms"
            )
            print(
                "  step boundary: "
                f"sync={result['timings']['sync_gradients']['median_ms']:.3f} ms, "
                f"materialize={result['timings']['materialize_weights']['median_ms']:.3f} ms, "
                f"amortized={result['projection']['amortized_step_boundary_overhead_ms']:.4f} ms/microbatch"
            )
    if output in ("json", "both"):
        print(json.dumps(result, sort_keys=True))


def main() -> None:
    args = _parser().parse_args()
    _validate_args(args)
    if not args.lower_only and jax.default_backend() != "gpu":
        raise RuntimeError("timing requires eight local GPUs; use --lower-only for a CPU lowering smoke")

    tokens = args.microbatch_size * args.sequence_length
    output_elements = tokens * args.hidden_dim
    keys = jax.random.split(jax.random.key(args.seed), 4)
    x = jax.random.normal(keys[0], (tokens, args.hidden_dim), dtype=jnp.bfloat16)
    selected_experts = _selected_experts(
        routing="balanced",
        tokens=tokens,
        top_k=args.top_k,
        num_experts=args.num_experts,
        seed=args.seed,
        skew_alpha=1.2,
    )
    combine_weights = jax.nn.softmax(
        jax.random.normal(keys[1], (tokens, args.top_k), dtype=jnp.float32),
        axis=-1,
    )
    w13 = 0.02 * jax.random.normal(
        keys[2],
        (args.num_experts, args.hidden_dim, 2 * args.intermediate_dim),
        dtype=jnp.bfloat16,
    )
    w2 = 0.02 * jax.random.normal(
        keys[3],
        (args.num_experts, args.intermediate_dim, args.hidden_dim),
        dtype=jnp.bfloat16,
    )
    source_inputs = (x, selected_experts, combine_weights, w13, w2)
    control = _build_arm(
        name="control_ep8",
        data_axis_size=1,
        source_inputs=source_inputs,
        num_experts=args.num_experts,
        capacity_factor=args.capacity_factor,
        output_elements=output_elements,
        fuse_weight_gradient_accumulation=False,
    )
    treatment = _build_arm(
        name="treatment",
        data_axis_size=args.treatment_data_axis_size,
        source_inputs=source_inputs,
        num_experts=args.num_experts,
        capacity_factor=args.capacity_factor,
        output_elements=output_elements,
        fuse_weight_gradient_accumulation=args.fuse_fp32_weight_gradient_accumulation,
    )

    hlo = {
        "control_local_vag": control.hlo["local_vag"],
        "treatment_local_vag": treatment.hlo["local_vag"],
        "treatment_sync": treatment.hlo["sync"],
        "treatment_materialize": treatment.hlo["materialize"],
    }
    memory = {
        "control_local_vag": _memory_summary(control.value_and_grad),
        "treatment_local_vag": _memory_summary(treatment.value_and_grad),
        "treatment_sync": _memory_summary(treatment.sync_gradients),
        "treatment_materialize": _memory_summary(treatment.materialize_weights),
    }
    if args.lower_only:
        result = {
            "backend": jax.default_backend(),
            "lower_only": True,
            "tokens": tokens,
            "hlo": hlo,
            "memory": memory,
        }
        _print_result(result, args.output)
        return

    control_forward = jax.block_until_ready(control.forward(*control.inputs))
    treatment_forward = jax.block_until_ready(treatment.forward(*treatment.inputs))
    control_vag = jax.block_until_ready(control.value_and_grad(*control.inputs, *control.initialize_accumulators()))
    treatment_vag = jax.block_until_ready(
        treatment.value_and_grad(*treatment.inputs, *treatment.initialize_accumulators())
    )
    treatment_synced_gradients = jax.block_until_ready(
        treatment.sync_gradients(treatment_vag[1][2], treatment_vag[1][3])
    )
    treatment_materialized_gradients = jax.block_until_ready(
        treatment.materialize_gradients(*treatment_synced_gradients)
    )

    control_loss = float(jax.device_get(jnp.sum(control_forward[0])))
    treatment_loss = float(jax.device_get(jnp.sum(treatment_forward[0])))
    loss_relative_error = abs(treatment_loss - control_loss) / max(abs(control_loss), np.finfo(np.float32).tiny)
    control_dropped = int(jax.device_get(jnp.sum(control_forward[2])))
    treatment_dropped = int(jax.device_get(jnp.sum(treatment_forward[2])))
    control_w13_gradient = control_vag[1][2][0]
    control_w2_gradient = control_vag[1][3][0]
    parity_tensors = {
        "output": _tensor_metrics(treatment_forward[1], control_forward[1]),
        "gradient.x": _tensor_metrics(treatment_vag[1][0], control_vag[1][0]),
        "gradient.combine_weights": _tensor_metrics(treatment_vag[1][1], control_vag[1][1]),
        "gradient.w13": _tensor_metrics(treatment_materialized_gradients[0], control_w13_gradient),
        "gradient.w2": _tensor_metrics(treatment_materialized_gradients[1], control_w2_gradient),
    }
    parity_passed = (
        loss_relative_error <= _PARITY_RELATIVE_L2
        and control_dropped == treatment_dropped
        and all(metrics["passed"] for metrics in parity_tensors.values())
    )
    parity = {
        "threshold_relative_l2": _PARITY_RELATIVE_L2,
        "loss": {
            "control": control_loss,
            "treatment": treatment_loss,
            "relative_error": loss_relative_error,
            "passed": loss_relative_error <= _PARITY_RELATIVE_L2,
        },
        "dropped": {
            "control": control_dropped,
            "treatment": treatment_dropped,
            "passed": control_dropped == treatment_dropped,
        },
        "tensors": parity_tensors,
        "passed": parity_passed,
    }

    forward_timings = _time_alternating(
        (control.forward, control.inputs),
        (treatment.forward, treatment.inputs),
        warmup=args.warmup,
        iterations=args.iterations,
    )
    vag_timings = _time_alternating_accumulating(
        control,
        treatment,
        warmup=args.warmup,
        iterations=args.iterations,
    )
    sync_timing = _time(
        treatment.sync_gradients,
        (treatment_vag[1][2], treatment_vag[1][3]),
        warmup=args.warmup,
        iterations=args.iterations,
    )
    materialize_timing = _time(
        treatment.materialize_weights,
        treatment.storage_weights,
        warmup=args.warmup,
        iterations=args.iterations,
    )
    projection = _projection(
        control_vag_ms=vag_timings["control"]["median_ms"],
        treatment_vag_ms=vag_timings["treatment"]["median_ms"],
        sync_ms=sync_timing["median_ms"],
        materialize_ms=materialize_timing["median_ms"],
        microbatches_per_step=args.microbatches_per_step,
        layers_per_stage=args.layers_per_stage,
        baseline_step_seconds=args.baseline_step_seconds,
        baseline_mfu=args.baseline_mfu,
        interstage_speedup=args.interstage_speedup,
    )
    no_data_collective_per_microbatch = hlo["treatment_local_vag"]["data_axis_collective_count"] == 0
    step_boundary_has_data_collectives = (
        hlo["treatment_sync"]["data_axis_collective_count"] > 0
        and hlo["treatment_materialize"]["data_axis_collective_count"] > 0
    )
    promotable = (
        parity_passed
        and no_data_collective_per_microbatch
        and step_boundary_has_data_collectives
        and projection["composed_mfu"] >= args.promotion_mfu
    )
    result = {
        "backend": jax.default_backend(),
        "device_kind": jax.devices()[0].device_kind,
        "tokens": tokens,
        "microbatch_size": args.microbatch_size,
        "sequence_length": args.sequence_length,
        "hidden_dim": args.hidden_dim,
        "intermediate_dim": args.intermediate_dim,
        "num_experts": args.num_experts,
        "top_k": args.top_k,
        "capacity_factor": args.capacity_factor,
        "control": {"data_axis_size": 1, "expert_axis_size": 8},
        "treatment": {
            "data_axis_size": treatment.data_axis_size,
            "expert_axis_size": treatment.expert_axis_size,
            "fused_fp32_weight_gradient_accumulation": args.fuse_fp32_weight_gradient_accumulation,
        },
        "parity": parity,
        "hlo": hlo,
        "memory": memory,
        "timings": {
            "forward": forward_timings,
            "value_and_grad": vag_timings,
            "sync_gradients": sync_timing,
            "materialize_weights": materialize_timing,
        },
        "projection": projection,
        "promotion_mfu": args.promotion_mfu,
        "promotable": promotable,
    }
    _print_result(result, args.output)
    if not parity_passed:
        raise AssertionError("EP4/data2 parity failed")
    if not no_data_collective_per_microbatch:
        raise AssertionError("EP4/data2 local VAG contains a data-axis collective")
    if not step_boundary_has_data_collectives:
        raise AssertionError("EP4/data2 step-boundary sync/materialization did not lower to data-axis collectives")


if __name__ == "__main__":
    main()
