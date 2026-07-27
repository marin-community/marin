# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""CPU reproducer for nested manual-VJP graph growth in the JaxPP MoE path.

This is self-contained apart from JAX and NumPy. It deliberately mirrors the
structure of moe_mlp_accumulating_weight_gradient without importing Marin:

* token routing is exact and integer-only;
* activations are sharded over data and expert axes;
* expert weights are sharded over expert and replicated over data;
* the local Ring-shaped graph uses all_gather and psum_scatter;
* accumulating grouped matmuls use an explicit custom VJP;
* the "full_vjp" backward invokes jax.vjp over all repeated local blocks from
  inside shard_map, matching the production formulation at f52c1bbf9b.

The "scan_vjp" alternative stages repeated blocks through lax.scan.
The "outlined_vjp" alternative places a non-inlined jit boundary around the
full local function before taking its VJP.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import platform
import re
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=4")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
from jax._src import core as jax_core
from jax.sharding import AxisType, Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

Formulation = Literal["full_vjp", "scan_vjp", "outlined_vjp"]
FLOAT_RELATIVE_L2_LIMIT = 0.002


@dataclass(frozen=True)
class Config:
    blocks: int
    tokens: int = 32
    hidden: int = 16
    intermediate: int = 12
    experts: int = 4
    topk: int = 2
    data_axis: int = 2
    expert_axis: int = 2
    capacity_factor: float = 1.25

    @property
    def devices(self) -> int:
        return self.data_axis * self.expert_axis


@dataclass(frozen=True)
class Measurement:
    formulation: str
    blocks: int
    lower_seconds: float
    compile_seconds: float
    execute_seconds: float
    stablehlo_bytes: int
    stablehlo_lines: int
    stablehlo_operations: dict[str, int]
    jaxpr_equations_recursive: int
    output_sum: float
    dropped: int


@dataclass(frozen=True)
class ForwardMeasurement:
    blocks: int
    lower_seconds: float
    compile_seconds: float
    stablehlo_bytes: int
    stablehlo_lines: int
    stablehlo_operations: dict[str, int]
    jaxpr_equations_recursive: int


def relative_l2(actual: np.ndarray, expected: np.ndarray) -> float:
    delta = np.asarray(actual, np.float64) - np.asarray(expected, np.float64)
    denominator = np.linalg.norm(np.asarray(expected, np.float64).ravel())
    return float(np.linalg.norm(delta.ravel()) / max(denominator, np.finfo(np.float64).tiny))


@jax.custom_vjp
def accumulating_grouped_dot(
    lhs: jax.Array,
    rhs: jax.Array,
    group_sizes: jax.Array,
    accumulator: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Small grouped matmul with the production accumulator-token contract."""
    del accumulator
    group_ends = jnp.cumsum(group_sizes)
    group_ids = jnp.sum(jnp.arange(lhs.shape[0])[:, None] >= group_ends[None, :], axis=1)
    grouped_rhs = rhs[group_ids]
    output = jnp.einsum("mi,mio->mo", lhs, grouped_rhs)
    return output, jnp.zeros((), dtype=jnp.float32)


def accumulating_grouped_dot_fwd(lhs, rhs, group_sizes, accumulator):
    output, token = accumulating_grouped_dot(lhs, rhs, group_sizes, accumulator)
    return (output, token), (lhs, rhs, group_sizes, accumulator)


def accumulating_grouped_dot_bwd(residuals, cotangents):
    lhs, rhs, group_sizes, accumulator = residuals
    output_cotangent, accumulation_scale = cotangents
    group_ends = jnp.cumsum(group_sizes)
    group_ids = jnp.sum(jnp.arange(lhs.shape[0])[:, None] >= group_ends[None, :], axis=1)
    grouped_rhs = rhs[group_ids]
    lhs_cotangent = jnp.einsum("mo,mio->mi", output_cotangent, grouped_rhs)
    updates = jnp.einsum("mi,mo->mio", lhs, output_cotangent).astype(jnp.float32)
    rhs_cotangent = jnp.zeros(rhs.shape, jnp.float32).at[group_ids].add(updates)
    rhs_cotangent = rhs_cotangent + accumulator * accumulation_scale.astype(jnp.float32)
    return lhs_cotangent, rhs_cotangent, None, None


accumulating_grouped_dot.defvjp(accumulating_grouped_dot_fwd, accumulating_grouped_dot_bwd)


def exact_routing(
    selected_experts_local: jax.Array,
    *,
    local_experts: int,
    num_experts: int,
    capacity_factor: float,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, int]:
    """Production-shaped source-major capacity routing with exact integer state."""
    selected_global = jax.lax.all_gather(selected_experts_local, "expert", tiled=True)
    tokens_per_data_replica = selected_global.shape[0]
    assignments = selected_global.size
    expert_flat = selected_global.reshape(-1)
    local_capacity = max(
        local_experts,
        int(np.ceil(capacity_factor * assignments / (num_experts // local_experts))),
    )
    expert_start = jax.lax.axis_index("expert") * local_experts
    local_expert = expert_flat - expert_start
    local_mask = (local_expert >= 0) & (local_expert < local_experts)
    local_expert = jnp.where(local_mask, local_expert, 0)
    expert_ids = jnp.arange(local_experts, dtype=jnp.int32)
    counts = jnp.sum(
        (local_expert[:, None] == expert_ids[None, :]).astype(jnp.int32) * local_mask[:, None].astype(jnp.int32),
        axis=0,
        dtype=jnp.int32,
    )
    remaining = jnp.maximum(local_capacity - jnp.cumsum(counts) + counts, 0)
    accepted_counts = jnp.minimum(counts, remaining)
    accepted_total = jnp.sum(accepted_counts, dtype=jnp.int32)
    valid = jnp.arange(local_capacity, dtype=jnp.int32) < accepted_total
    flat_position = jnp.arange(assignments, dtype=jnp.int32)
    order_key = local_expert * assignments + flat_position
    max_order_key = local_experts * assignments
    selection_key = jnp.where(local_mask, max_order_key - order_key, -1)
    _, assignment_indices = jax.lax.top_k(selection_key, local_capacity)
    dropped = jnp.sum(counts, dtype=jnp.int32) - accepted_total
    return assignment_indices, valid, accepted_counts, dropped, tokens_per_data_replica


def local_ring_block(
    x_local: jax.Array,
    selected_experts_local: jax.Array,
    combine_weights_local: jax.Array,
    w13_local: jax.Array,
    w2_local: jax.Array,
    w13_accumulator_local: jax.Array,
    w2_accumulator_local: jax.Array,
    *,
    config: Config,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """One compact Ring-shaped MoE block."""
    local_experts = w13_local.shape[0]
    assignment_indices, valid, accepted_counts, dropped, tokens_per_data_replica = exact_routing(
        selected_experts_local,
        local_experts=local_experts,
        num_experts=config.experts,
        capacity_factor=config.capacity_factor,
    )
    x_global = jax.lax.all_gather(x_local, "expert", tiled=True)
    combine_global = jax.lax.all_gather(combine_weights_local, "expert", tiled=True)
    token_global = assignment_indices // config.topk
    x_dispatch = jnp.where(valid[:, None], x_global[token_global], jnp.zeros_like(x_global[token_global]))
    weight_flat = combine_global.reshape(-1)
    dispatch_weights = jnp.where(valid, weight_flat[assignment_indices], 0).astype(x_local.dtype)
    group_sizes = accepted_counts.at[-1].add(valid.size - jnp.sum(accepted_counts, dtype=jnp.int32))

    w13_out, w13_token = accumulating_grouped_dot(
        x_dispatch,
        w13_local,
        group_sizes,
        w13_accumulator_local,
    )
    gate, up = jnp.split(w13_out, 2, axis=-1)
    hidden = jax.nn.silu(gate) * up
    expert_out, w2_token = accumulating_grouped_dot(
        hidden,
        w2_local,
        group_sizes,
        w2_accumulator_local,
    )
    out_global = (
        jnp.zeros((tokens_per_data_replica, config.hidden), dtype=x_local.dtype)
        .at[token_global]
        .add(expert_out.astype(x_local.dtype) * dispatch_weights[:, None], mode="drop")
    )
    output = jax.lax.psum_scatter(out_global, "expert", scatter_dimension=0, tiled=True)
    dropped_total = jax.lax.psum(dropped, ("data", "expert"))
    return output, dropped_total, w13_token + w2_token


def local_stack(
    x_local: jax.Array,
    selected_experts_local: jax.Array,
    combine_weights_local: jax.Array,
    w13_local: jax.Array,
    w2_local: jax.Array,
    w13_accumulator_local: jax.Array,
    w2_accumulator_local: jax.Array,
    *,
    config: Config,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    dropped = jnp.zeros((), jnp.int32)
    token = jnp.zeros((), jnp.float32)
    for block in range(config.blocks):
        update, block_dropped, block_token = local_ring_block(
            x_local,
            selected_experts_local,
            combine_weights_local,
            w13_local[block],
            w2_local[block],
            w13_accumulator_local[block],
            w2_accumulator_local[block],
            config=config,
        )
        x_local = (x_local + update).astype(jnp.bfloat16)
        dropped = dropped + block_dropped
        token = token + block_token
    return x_local, dropped, token


def local_stack_scan(
    x_local: jax.Array,
    selected_experts_local: jax.Array,
    combine_weights_local: jax.Array,
    w13_local: jax.Array,
    w2_local: jax.Array,
    w13_accumulator_local: jax.Array,
    w2_accumulator_local: jax.Array,
    *,
    config: Config,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    def body(carry, weights):
        activation, dropped, token = carry
        w13, w2, w13_accumulator, w2_accumulator = weights
        update, block_dropped, block_token = local_ring_block(
            activation,
            selected_experts_local,
            combine_weights_local,
            w13,
            w2,
            w13_accumulator,
            w2_accumulator,
            config=config,
        )
        return (
            (activation + update).astype(jnp.bfloat16),
            dropped + block_dropped,
            token + block_token,
        ), None

    initial = (x_local, jnp.zeros((), jnp.int32), jnp.zeros((), jnp.float32))
    result, _ = jax.lax.scan(
        body,
        initial,
        (w13_local, w2_local, w13_accumulator_local, w2_accumulator_local),
    )
    return result


def build_mapped(
    mesh: Mesh,
    config: Config,
    formulation: Formulation,
) -> Callable[..., tuple[jax.Array, jax.Array, jax.Array]]:
    batch_spec = P(("data", "expert"), None)
    route_spec = P(("data", "expert"), None)
    weight_spec = P(None, "expert", None, None)
    stack = local_stack_scan if formulation == "scan_vjp" else local_stack

    def local(*args):
        return stack(*args, config=config)

    shard_fn = jax.shard_map(
        local,
        mesh=mesh,
        in_specs=(
            batch_spec,
            route_spec,
            route_spec,
            weight_spec,
            weight_spec,
            weight_spec,
            weight_spec,
        ),
        out_specs=(batch_spec, P(), P()),
        check_vma=False,
    )

    @jax.custom_vjp
    def mapped_fn(x, selected, combine, w13, w2, w13_accumulator, w2_accumulator):
        return shard_fn(x, selected, combine, w13, w2, w13_accumulator, w2_accumulator)

    def mapped_fwd(x, selected, combine, w13, w2, w13_accumulator, w2_accumulator):
        outputs = shard_fn(x, selected, combine, w13, w2, w13_accumulator, w2_accumulator)
        return outputs, (x, selected, combine, w13, w2, w13_accumulator, w2_accumulator)

    def full_local_bwd(
        x_local,
        selected_local,
        combine_local,
        w13_local,
        w2_local,
        w13_accumulator_local,
        w2_accumulator_local,
        output_cotangent_local,
        token_cotangent_local,
    ):
        def differentiable(local_x, local_combine, local_w13, local_w2, local_w13_acc, local_w2_acc):
            out, _, token = stack(
                local_x,
                selected_local,
                local_combine,
                local_w13,
                local_w2,
                local_w13_acc,
                local_w2_acc,
                config=config,
            )
            return out, token

        differentiated = jax.jit(differentiable, inline=False) if formulation == "outlined_vjp" else differentiable
        _, pullback = jax.vjp(
            differentiated,
            x_local,
            combine_local,
            w13_local,
            w2_local,
            w13_accumulator_local,
            w2_accumulator_local,
        )
        return pullback((output_cotangent_local, token_cotangent_local))

    def mapped_bwd(residuals, cotangents):
        x, selected, combine, w13, w2, w13_accumulator, w2_accumulator = residuals
        output_cotangent, _, token_cotangent = cotangents
        gradients = jax.shard_map(
            full_local_bwd,
            mesh=mesh,
            in_specs=(
                batch_spec,
                route_spec,
                route_spec,
                weight_spec,
                weight_spec,
                weight_spec,
                weight_spec,
                batch_spec,
                P(),
            ),
            out_specs=(
                batch_spec,
                route_spec,
                weight_spec,
                weight_spec,
                weight_spec,
                weight_spec,
            ),
            check_vma=False,
        )(
            x,
            selected,
            combine,
            w13,
            w2,
            w13_accumulator,
            w2_accumulator,
            output_cotangent,
            token_cotangent,
        )
        x_gradient, combine_gradient, w13_gradient, w2_gradient, w13_acc_gradient, w2_acc_gradient = gradients
        return (
            x_gradient,
            None,
            combine_gradient,
            w13_gradient,
            w2_gradient,
            w13_acc_gradient,
            w2_acc_gradient,
        )

    mapped_fn.defvjp(mapped_fwd, mapped_bwd)
    return mapped_fn


def make_mesh(config: Config) -> Mesh:
    if len(jax.devices()) < config.devices:
        raise ValueError(
            f"need {config.devices} CPU devices, got {len(jax.devices())}; "
            "set XLA_FLAGS=--xla_force_host_platform_device_count=4 before importing JAX"
        )
    devices = np.asarray(jax.devices()[: config.devices], dtype=object).reshape(
        config.data_axis,
        config.expert_axis,
    )
    return Mesh(devices, ("data", "expert"), axis_types=(AxisType.Explicit, AxisType.Explicit))


def make_inputs(config: Config, mesh: Mesh) -> tuple[jax.Array, ...]:
    if config.topk != 2:
        raise ValueError("this minimized routing fixture requires topk=2")
    if config.experts < 2 or config.experts % config.expert_axis:
        raise ValueError("experts must be at least 2 and divisible by expert_axis")
    key = jax.random.key(17)
    keys = jax.random.split(key, 6)
    half_tokens = config.tokens // config.data_axis
    x_half = jax.random.normal(keys[0], (half_tokens, config.hidden), dtype=jnp.bfloat16) * 0.1
    x = jnp.concatenate([x_half] * config.data_axis, axis=0)
    token_indices = jnp.arange(half_tokens, dtype=jnp.int32)
    route_half = jnp.stack(
        (
            jnp.zeros(half_tokens, dtype=jnp.int32),
            1 + token_indices % (config.experts - 1),
        ),
        axis=1,
    )
    selected = jnp.concatenate([route_half] * config.data_axis, axis=0)
    combine_half = jax.nn.softmax(
        jax.random.normal(keys[1], (half_tokens, config.topk), dtype=jnp.float32),
        axis=-1,
    ).astype(jnp.bfloat16)
    combine = jnp.concatenate([combine_half] * config.data_axis, axis=0)
    w13 = (
        jax.random.normal(
            keys[2],
            (config.blocks, config.experts, config.hidden, 2 * config.intermediate),
            dtype=jnp.bfloat16,
        )
        * 0.03
    )
    w2 = (
        jax.random.normal(
            keys[3],
            (config.blocks, config.experts, config.intermediate, config.hidden),
            dtype=jnp.bfloat16,
        )
        * 0.03
    )
    w13_accumulator = jax.random.normal(keys[4], w13.shape, dtype=jnp.float32) * 0.001
    w2_accumulator = jax.random.normal(keys[5], w2.shape, dtype=jnp.float32) * 0.001

    batch_sharding = NamedSharding(mesh, P(("data", "expert"), None))
    weight_sharding = NamedSharding(mesh, P(None, "expert", None, None))
    return (
        jax.device_put(x, batch_sharding),
        jax.device_put(selected, batch_sharding),
        jax.device_put(combine, batch_sharding),
        jax.device_put(w13, weight_sharding),
        jax.device_put(w2, weight_sharding),
        jax.device_put(w13_accumulator, weight_sharding),
        jax.device_put(w2_accumulator, weight_sharding),
    )


def build_step(mesh: Mesh, config: Config, formulation: Formulation) -> Callable[..., Any]:
    mapped = build_mapped(mesh, config, formulation)

    def loss_fn(x, selected, combine, w13, w2, w13_accumulator, w2_accumulator):
        output, dropped, token = mapped(
            x,
            selected,
            combine,
            w13,
            w2,
            w13_accumulator,
            w2_accumulator,
        )
        loss = jnp.mean(output.astype(jnp.float32) ** 2) + token
        return loss, (output, dropped)

    return jax.jit(
        jax.value_and_grad(loss_fn, argnums=(0, 2, 3, 4, 5, 6), has_aux=True),
        in_shardings=tuple(value.sharding for value in make_inputs(config, mesh)),
    )


def recursive_jaxpr_equations(value: Any) -> int:
    seen: set[int] = set()

    def visit(node: Any) -> int:
        identity = id(node)
        if identity in seen:
            return 0
        if isinstance(node, jax_core.Jaxpr):
            seen.add(identity)
            total = len(node.eqns)
            for equation in node.eqns:
                total += sum(visit(parameter) for parameter in equation.params.values())
            return total
        if isinstance(node, dict):
            return sum(visit(item) for item in node.values())
        if isinstance(node, (tuple, list)):
            return sum(visit(item) for item in node)
        return 0

    return visit(value)


def stablehlo_operation_counts(text: str) -> dict[str, int]:
    operations = re.findall(r"\bstablehlo\.([a-zA-Z0-9_]+)\b", text)
    return dict(sorted(collections.Counter(operations).items()))


def measure(config: Config, formulation: Formulation) -> tuple[Measurement, Any]:
    mesh = make_mesh(config)
    inputs = make_inputs(config, mesh)
    step = build_step(mesh, config, formulation)
    started = time.perf_counter()
    lowered = step.lower(*inputs)
    lower_seconds = time.perf_counter() - started
    stablehlo_text = str(lowered.compiler_ir(dialect="stablehlo"))
    jaxpr = jax.make_jaxpr(step)(*inputs)
    started = time.perf_counter()
    compiled = lowered.compile()
    compile_seconds = time.perf_counter() - started
    started = time.perf_counter()
    result = compiled(*inputs)
    jax.block_until_ready(result)
    execute_seconds = time.perf_counter() - started
    (loss, (output, dropped)), gradients = result
    measurement = Measurement(
        formulation=formulation,
        blocks=config.blocks,
        lower_seconds=lower_seconds,
        compile_seconds=compile_seconds,
        execute_seconds=execute_seconds,
        stablehlo_bytes=len(stablehlo_text.encode()),
        stablehlo_lines=stablehlo_text.count("\n") + 1,
        stablehlo_operations=stablehlo_operation_counts(stablehlo_text),
        jaxpr_equations_recursive=recursive_jaxpr_equations(jaxpr),
        output_sum=float(np.asarray(output, dtype=np.float32).sum()),
        dropped=int(dropped),
    )
    return measurement, (np.asarray(loss), np.asarray(output), int(dropped), jax.tree.map(np.asarray, gradients))


def measure_forward(config: Config) -> ForwardMeasurement:
    mesh = make_mesh(config)
    inputs = make_inputs(config, mesh)
    forward = jax.jit(
        build_mapped(mesh, config, "full_vjp"),
        in_shardings=tuple(value.sharding for value in inputs),
    )
    started = time.perf_counter()
    lowered = forward.lower(*inputs)
    lower_seconds = time.perf_counter() - started
    stablehlo_text = str(lowered.compiler_ir(dialect="stablehlo"))
    jaxpr = jax.make_jaxpr(forward)(*inputs)
    started = time.perf_counter()
    lowered.compile()
    compile_seconds = time.perf_counter() - started
    return ForwardMeasurement(
        blocks=config.blocks,
        lower_seconds=lower_seconds,
        compile_seconds=compile_seconds,
        stablehlo_bytes=len(stablehlo_text.encode()),
        stablehlo_lines=stablehlo_text.count("\n") + 1,
        stablehlo_operations=stablehlo_operation_counts(stablehlo_text),
        jaxpr_equations_recursive=recursive_jaxpr_equations(jaxpr),
    )


def routing_policy(config: Config) -> dict[str, Any]:
    tokens_per_data_replica = config.tokens // config.data_axis
    selected = np.stack(
        (
            np.zeros(tokens_per_data_replica, dtype=np.int32),
            1 + np.arange(tokens_per_data_replica, dtype=np.int32) % (config.experts - 1),
        ),
        axis=1,
    )
    counts = np.bincount(selected.reshape(-1), minlength=config.experts)
    local_experts = config.experts // config.expert_axis
    assignments = selected.size
    local_capacity = max(
        local_experts,
        int(np.ceil(config.capacity_factor * assignments / config.expert_axis)),
    )
    accepted_counts = np.zeros_like(counts)
    dropped_per_data_replica = 0
    for shard in range(config.expert_axis):
        start = shard * local_experts
        stop = start + local_experts
        remaining = local_capacity
        for expert in range(start, stop):
            accepted_counts[expert] = min(counts[expert], remaining)
            remaining -= accepted_counts[expert]
        dropped_per_data_replica += int(np.sum(counts[start:stop] - accepted_counts[start:stop]))
    return {
        "selected_experts_per_data_replica": selected.tolist(),
        "counts_per_data_replica": counts.tolist(),
        "accepted_counts_per_data_replica": accepted_counts.tolist(),
        "local_capacity": local_capacity,
        "dropped_per_data_replica": dropped_per_data_replica,
        "expected_global_dropped": dropped_per_data_replica * config.data_axis,
    }


def parity_report(reference: Any, candidate: Any, *, expected_dropped: int) -> dict[str, Any]:
    reference_loss, reference_output, reference_dropped, reference_gradients = reference
    candidate_loss, candidate_output, candidate_dropped, candidate_gradients = candidate
    gradient_errors = [
        relative_l2(candidate_leaf, reference_leaf)
        for candidate_leaf, reference_leaf in zip(
            jax.tree.leaves(candidate_gradients),
            jax.tree.leaves(reference_gradients),
            strict=True,
        )
    ]
    floating_errors = {
        "loss": relative_l2(candidate_loss, reference_loss),
        "output": relative_l2(candidate_output, reference_output),
        "gradient_max": max(gradient_errors),
        "gradient_leaves": gradient_errors,
    }
    return {
        "floating_relative_l2_limit": FLOAT_RELATIVE_L2_LIMIT,
        "floating_relative_l2": floating_errors,
        "floating_passed": (
            max(
                floating_errors["loss"],
                floating_errors["output"],
                floating_errors["gradient_max"],
            )
            <= FLOAT_RELATIVE_L2_LIMIT
        ),
        "routing_counts_drops_exact": candidate_dropped == reference_dropped == expected_dropped,
        "expected_dropped": expected_dropped,
        "reference_dropped": reference_dropped,
        "candidate_dropped": candidate_dropped,
    }


def growth_report(
    measurements: list[Measurement],
    forward_measurements: list[ForwardMeasurement],
) -> dict[str, Any]:
    by_formulation = {
        formulation: sorted(
            (measurement for measurement in measurements if measurement.formulation == formulation),
            key=lambda measurement: measurement.blocks,
        )
        for formulation in ("full_vjp", "scan_vjp", "outlined_vjp")
    }
    report: dict[str, Any] = {}
    for formulation, values in by_formulation.items():
        if len(values) != 2:
            continue
        one, two = values
        operation_growth = {
            operation: two.stablehlo_operations.get(operation, 0) - one.stablehlo_operations.get(operation, 0)
            for operation in sorted(set(one.stablehlo_operations) | set(two.stablehlo_operations))
        }
        report[formulation] = {
            "stablehlo_bytes_ratio_2_to_1": two.stablehlo_bytes / one.stablehlo_bytes,
            "jaxpr_equations_ratio_2_to_1": two.jaxpr_equations_recursive / max(one.jaxpr_equations_recursive, 1),
            "compile_seconds_ratio_2_to_1": two.compile_seconds / max(one.compile_seconds, 1e-12),
            "stablehlo_operation_delta_2_minus_1": operation_growth,
        }
    forward_one, forward_two = sorted(forward_measurements, key=lambda measurement: measurement.blocks)
    full_one, full_two = by_formulation["full_vjp"]
    forward_bytes_delta = forward_two.stablehlo_bytes - forward_one.stablehlo_bytes
    full_bytes_delta = full_two.stablehlo_bytes - full_one.stablehlo_bytes
    operation_amplification = {}
    for operation in ("dot_general", "all_gather", "reduce_scatter", "gather", "scatter"):
        forward_delta = forward_two.stablehlo_operations.get(operation, 0) - forward_one.stablehlo_operations.get(
            operation, 0
        )
        full_delta = full_two.stablehlo_operations.get(operation, 0) - full_one.stablehlo_operations.get(operation, 0)
        operation_amplification[operation] = {
            "forward_delta": forward_delta,
            "full_vjp_train_delta": full_delta,
            "amplification": full_delta / max(forward_delta, 1),
        }
    report["nested_vjp_attribution"] = {
        "forward_stablehlo_bytes_delta_2_minus_1": forward_bytes_delta,
        "full_vjp_train_stablehlo_bytes_delta_2_minus_1": full_bytes_delta,
        "train_over_forward_block_delta": full_bytes_delta / forward_bytes_delta,
        "operation_amplification": operation_amplification,
    }
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("results.json"))
    parser.add_argument("--tokens", type=int, default=32)
    parser.add_argument("--hidden", type=int, default=16)
    parser.add_argument("--intermediate", type=int, default=12)
    parser.add_argument("--experts", type=int, default=4)
    parser.add_argument("--topk", type=int, default=2)
    parser.add_argument("--capacity-factor", type=float, default=1.25)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    measurements: list[Measurement] = []
    forward_measurements: list[ForwardMeasurement] = []
    results: dict[tuple[str, int], Any] = {}
    configs: dict[int, Config] = {}
    for blocks in (1, 2):
        config = Config(
            blocks=blocks,
            tokens=args.tokens,
            hidden=args.hidden,
            intermediate=args.intermediate,
            experts=args.experts,
            topk=args.topk,
            capacity_factor=args.capacity_factor,
        )
        configs[blocks] = config
        forward_measurement = measure_forward(config)
        forward_measurements.append(forward_measurement)
        print(json.dumps({"event": "forward_measurement", **asdict(forward_measurement)}, sort_keys=True), flush=True)
        for formulation in ("full_vjp", "scan_vjp", "outlined_vjp"):
            measurement, result = measure(config, formulation)
            measurements.append(measurement)
            results[(formulation, blocks)] = result
            print(json.dumps({"event": "measurement", **asdict(measurement)}, sort_keys=True), flush=True)

    parity = {
        f"{formulation}_blocks_{blocks}": parity_report(
            results[("full_vjp", blocks)],
            results[(formulation, blocks)],
            expected_dropped=routing_policy(configs[blocks])["expected_global_dropped"] * blocks,
        )
        for formulation in ("scan_vjp", "outlined_vjp")
        for blocks in (1, 2)
    }
    report = {
        "environment": {
            "python": platform.python_version(),
            "jax": jax.__version__,
            "backend": jax.default_backend(),
            "devices": [str(device) for device in jax.devices()],
            "xla_flags": os.environ.get("XLA_FLAGS", ""),
        },
        "command": "uv run python scratch/jaxpp_manual_vjp_compile_repro/repro.py",
        "routing_policy": routing_policy(configs[1]),
        "forward_measurements": [asdict(measurement) for measurement in forward_measurements],
        "measurements": [asdict(measurement) for measurement in measurements],
        "growth": growth_report(measurements, forward_measurements),
        "parity": parity,
    }
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"event": "result", "output": str(args.output), "growth": report["growth"], "parity": parity}))
    if not all(value["floating_passed"] and value["routing_counts_drops_exact"] for value in parity.values()):
        raise SystemExit("parity policy failed")


if __name__ == "__main__":
    main()
