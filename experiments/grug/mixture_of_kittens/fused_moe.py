# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Fused Mixture-of-Kittens forward with a JAX reference gradient."""

from collections.abc import Callable

import jax
import jax.numpy as jnp
from haliax.jax_utils import tree_checkpoint_name
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from levanter.grug._moe.common import _CHECKPOINT_MOE_OUTPUT, MoeImplementation
from levanter.grug.grug_moe import moe_mlp
from levanter.grug.sharding import _batch_spec_from_x, _reshard_for_shard_map
from levanter.kernels.mixture_of_kittens.forward_ffi import (
    MoKForwardConfig,
    MoKForwardContext,
    backward_bf16_local,
    forward_bf16_local,
    schedule_capacity,
)
from levanter.utils.activation import ActivationFunctionEnum

from experiments.grug.mixture_of_kittens.schedule import build_schedule

_EXPERT_AXIS = "expert"
_NUM_DEVICES = 4


def _validate_topology(mesh: jax.sharding.Mesh | jax.sharding.AbstractMesh) -> None:
    if mesh.empty or int(mesh.shape.get(_EXPERT_AXIS, 1)) != _NUM_DEVICES:
        raise ValueError("Mixture-of-Kittens requires an expert axis of size four")
    if jax.process_count() != 1 or jax.local_device_count() != _NUM_DEVICES:
        raise ValueError("Mixture-of-Kittens requires one JAX process with four visible GPUs")


def _fused_forward_with_context(
    x: jax.Array,
    selected_experts: jax.Array,
    combine_weights: jax.Array,
    w_gate: jax.Array,
    w_up: jax.Array,
    w_down: jax.Array,
    shared_gate: jax.Array,
    shared_up: jax.Array,
    shared_down: jax.Array,
    *,
    mesh: jax.sharding.Mesh | jax.sharding.AbstractMesh,
    config: MoKForwardConfig,
) -> tuple[jax.Array, jax.Array, MoKForwardContext]:
    _validate_topology(mesh)
    batch_spec = _batch_spec_from_x(x, mesh)
    expert_weight_spec = P(_EXPERT_AXIS, None, None)
    shared_weight_spec = P(None, None)

    x = _reshard_for_shard_map(x, mesh, batch_spec)
    selected_experts = _reshard_for_shard_map(selected_experts, mesh, batch_spec)
    combine_weights = _reshard_for_shard_map(combine_weights, mesh, batch_spec)
    w_gate = _reshard_for_shard_map(w_gate, mesh, expert_weight_spec)
    w_up = _reshard_for_shard_map(w_up, mesh, expert_weight_spec)
    w_down = _reshard_for_shard_map(w_down, mesh, expert_weight_spec)
    shared_gate = _reshard_for_shard_map(shared_gate, mesh, shared_weight_spec)
    shared_up = _reshard_for_shard_map(shared_up, mesh, shared_weight_spec)
    shared_down = _reshard_for_shard_map(shared_down, mesh, shared_weight_spec)

    def local_forward(
        local_x: jax.Array,
        local_selected_experts: jax.Array,
        local_combine_weights: jax.Array,
        local_w_gate: jax.Array,
        local_w_up: jax.Array,
        local_w_down: jax.Array,
        local_shared_gate: jax.Array,
        local_shared_up: jax.Array,
        local_shared_down: jax.Array,
    ) -> tuple[jax.Array, jax.Array, MoKForwardContext]:
        all_selected_experts = jax.lax.all_gather(local_selected_experts, _EXPERT_AXIS)
        rank = jax.lax.axis_index(_EXPERT_AXIS)
        local_experts = local_w_gate.shape[0]
        capacity = schedule_capacity(
            local_x.shape[0],
            local_selected_experts.shape[1],
            local_experts,
            config,
        )
        schedule = build_schedule(
            all_selected_experts,
            num_local_experts=local_experts,
            schedule_capacity=capacity,
            rank=rank,
        )
        output, context = forward_bf16_local(
            local_x,
            local_combine_weights,
            jnp.transpose(local_shared_gate),
            jnp.transpose(local_w_gate, (0, 2, 1)),
            jnp.transpose(local_shared_up),
            jnp.transpose(local_w_up, (0, 2, 1)),
            jnp.transpose(local_shared_down),
            jnp.transpose(local_w_down, (0, 2, 1)),
            schedule.peer_rank,
            schedule.peer_token_idx,
            schedule.num_tokens,
            schedule.tokens_per_expert,
            config=config,
        )
        dropped_assignments = jax.lax.psum(schedule.dropped_assignments, _EXPERT_AXIS)
        return output, dropped_assignments, context

    routed_context_spec = P(_EXPERT_AXIS, None)
    context_specs = MoKForwardContext(
        x_routed=routed_context_spec,
        gate_shared=batch_spec,
        gate_routed=routed_context_spec,
        up_shared=batch_spec,
        up_routed=routed_context_spec,
        hidden_shared=batch_spec,
        hidden_routed=routed_context_spec,
    )

    return jax.shard_map(
        local_forward,
        mesh=mesh,
        in_specs=(
            batch_spec,
            batch_spec,
            batch_spec,
            expert_weight_spec,
            expert_weight_spec,
            expert_weight_spec,
            shared_weight_spec,
            shared_weight_spec,
            shared_weight_spec,
        ),
        out_specs=(batch_spec, P(), context_specs),
        check_vma=False,
    )(x, selected_experts, combine_weights, w_gate, w_up, w_down, shared_gate, shared_up, shared_down)


def _fused_forward(
    x: jax.Array,
    selected_experts: jax.Array,
    combine_weights: jax.Array,
    w_gate: jax.Array,
    w_up: jax.Array,
    w_down: jax.Array,
    shared_gate: jax.Array,
    shared_up: jax.Array,
    shared_down: jax.Array,
    *,
    mesh: jax.sharding.Mesh | jax.sharding.AbstractMesh,
    config: MoKForwardConfig,
) -> tuple[jax.Array, jax.Array]:
    output, dropped_assignments, _ = _fused_forward_with_context(
        x,
        selected_experts,
        combine_weights,
        w_gate,
        w_up,
        w_down,
        shared_gate,
        shared_up,
        shared_down,
        mesh=mesh,
        config=config,
    )
    return output, dropped_assignments


def _fused_backward(
    output_gradient: jax.Array,
    x: jax.Array,
    selected_experts: jax.Array,
    combine_weights: jax.Array,
    w_gate: jax.Array,
    w_up: jax.Array,
    w_down: jax.Array,
    shared_gate: jax.Array,
    shared_up: jax.Array,
    shared_down: jax.Array,
    context: MoKForwardContext,
    *,
    mesh: jax.sharding.Mesh | jax.sharding.AbstractMesh,
    config: MoKForwardConfig,
) -> tuple[jax.Array, ...]:
    _validate_topology(mesh)
    batch_spec = _batch_spec_from_x(x, mesh)
    expert_weight_spec = P(_EXPERT_AXIS, None, None)
    shared_weight_spec = P(None, None)
    gradient_specs = tuple(
        getattr(jax.typeof(value).sharding, "spec", default)
        for value, default in (
            (x, batch_spec),
            (combine_weights, batch_spec),
            (w_gate, expert_weight_spec),
            (w_up, expert_weight_spec),
            (w_down, expert_weight_spec),
            (shared_gate, shared_weight_spec),
            (shared_up, shared_weight_spec),
            (shared_down, shared_weight_spec),
        )
    )
    routed_context_spec = P(_EXPERT_AXIS, None)
    context_specs = MoKForwardContext(
        x_routed=routed_context_spec,
        gate_shared=batch_spec,
        gate_routed=routed_context_spec,
        up_shared=batch_spec,
        up_routed=routed_context_spec,
        hidden_shared=batch_spec,
        hidden_routed=routed_context_spec,
    )

    output_gradient = _reshard_for_shard_map(output_gradient, mesh, batch_spec)
    x = _reshard_for_shard_map(x, mesh, batch_spec)
    selected_experts = _reshard_for_shard_map(selected_experts, mesh, batch_spec)
    combine_weights = _reshard_for_shard_map(combine_weights, mesh, batch_spec)
    w_gate = _reshard_for_shard_map(w_gate, mesh, expert_weight_spec)
    w_up = _reshard_for_shard_map(w_up, mesh, expert_weight_spec)
    w_down = _reshard_for_shard_map(w_down, mesh, expert_weight_spec)
    shared_gate = _reshard_for_shard_map(shared_gate, mesh, shared_weight_spec)
    shared_up = _reshard_for_shard_map(shared_up, mesh, shared_weight_spec)
    shared_down = _reshard_for_shard_map(shared_down, mesh, shared_weight_spec)
    context = MoKForwardContext(
        *(_reshard_for_shard_map(value, mesh, spec) for value, spec in zip(context, context_specs, strict=True))
    )

    def local_backward(
        local_output_gradient: jax.Array,
        local_x: jax.Array,
        local_selected_experts: jax.Array,
        local_combine_weights: jax.Array,
        local_w_gate: jax.Array,
        local_w_up: jax.Array,
        local_w_down: jax.Array,
        local_shared_gate: jax.Array,
        local_shared_up: jax.Array,
        local_shared_down: jax.Array,
        local_context: MoKForwardContext,
    ) -> tuple[jax.Array, ...]:
        all_selected_experts = jax.lax.all_gather(local_selected_experts, _EXPERT_AXIS)
        rank = jax.lax.axis_index(_EXPERT_AXIS)
        local_experts = local_w_gate.shape[0]
        capacity = schedule_capacity(
            local_x.shape[0],
            local_selected_experts.shape[1],
            local_experts,
            config,
        )
        schedule = build_schedule(
            all_selected_experts,
            num_local_experts=local_experts,
            schedule_capacity=capacity,
            rank=rank,
        )
        gradients = backward_bf16_local(
            local_output_gradient,
            local_x,
            local_combine_weights,
            jnp.transpose(local_shared_gate),
            jnp.transpose(local_w_gate, (0, 2, 1)),
            jnp.transpose(local_shared_up),
            jnp.transpose(local_w_up, (0, 2, 1)),
            jnp.transpose(local_shared_down),
            jnp.transpose(local_w_down, (0, 2, 1)),
            local_context,
            schedule.peer_rank,
            schedule.peer_token_idx,
            schedule.num_tokens,
            schedule.tokens_per_expert,
            config=config,
        )
        (
            d_x,
            d_combine_weights,
            d_w_gate,
            d_w_up,
            d_w_down,
            d_shared_gate,
            d_shared_up,
            d_shared_down,
        ) = gradients
        return (
            d_x.astype(local_x.dtype),
            d_combine_weights.astype(local_combine_weights.dtype),
            jnp.transpose(d_w_gate, (0, 2, 1)).astype(local_w_gate.dtype),
            jnp.transpose(d_w_up, (0, 2, 1)).astype(local_w_up.dtype),
            jnp.transpose(d_w_down, (0, 2, 1)).astype(local_w_down.dtype),
            jnp.transpose(jax.lax.psum(d_shared_gate.astype(jnp.float32), _EXPERT_AXIS)).astype(local_shared_gate.dtype),
            jnp.transpose(jax.lax.psum(d_shared_up.astype(jnp.float32), _EXPERT_AXIS)).astype(local_shared_up.dtype),
            jnp.transpose(jax.lax.psum(d_shared_down.astype(jnp.float32), _EXPERT_AXIS)).astype(local_shared_down.dtype),
        )

    gradients = jax.shard_map(
        local_backward,
        mesh=mesh,
        in_specs=(
            batch_spec,
            batch_spec,
            batch_spec,
            batch_spec,
            expert_weight_spec,
            expert_weight_spec,
            expert_weight_spec,
            shared_weight_spec,
            shared_weight_spec,
            shared_weight_spec,
            context_specs,
        ),
        out_specs=(
            batch_spec,
            batch_spec,
            expert_weight_spec,
            expert_weight_spec,
            expert_weight_spec,
            shared_weight_spec,
            shared_weight_spec,
            shared_weight_spec,
        ),
        check_vma=False,
    )(
        output_gradient,
        x,
        selected_experts,
        combine_weights,
        w_gate,
        w_up,
        w_down,
        shared_gate,
        shared_up,
        shared_down,
        context,
    )
    return tuple(
        _reshard_for_shard_map(gradient, mesh, spec) for gradient, spec in zip(gradients, gradient_specs, strict=True)
    )


def mixture_of_kittens_reference(
    x: jax.Array,
    selected_experts: jax.Array,
    combine_weights: jax.Array,
    w_gate: jax.Array,
    w_up: jax.Array,
    w_down: jax.Array,
    shared_gate: jax.Array,
    shared_up: jax.Array,
    shared_down: jax.Array,
    *,
    mesh: jax.sharding.Mesh | jax.sharding.AbstractMesh,
    config: MoKForwardConfig,
    fallback_implementation: MoeImplementation,
    ragged_all_to_all_splits_per_peer: int,
) -> jax.Array:
    """Run the JAX forward used to differentiate the fused call."""
    routed = moe_mlp(
        x,
        selected_experts,
        combine_weights,
        jnp.concatenate((w_gate, w_up), axis=-1),
        w_down,
        activation=ActivationFunctionEnum.silu,
        implementation=fallback_implementation,
        mesh=mesh,
        capacity_factor=(
            schedule_capacity(
                x.shape[0] // int(mesh.shape[_EXPERT_AXIS]),
                selected_experts.shape[1],
                w_gate.shape[0] // int(mesh.shape[_EXPERT_AXIS]),
                config,
            )
            / ((x.shape[0] // int(mesh.shape[_EXPERT_AXIS])) * selected_experts.shape[1])
        ),
        ragged_all_to_all_splits_per_peer=ragged_all_to_all_splits_per_peer,
    )
    if isinstance(routed, tuple):
        raise AssertionError("The fallback MoE returned capacity data when only output was requested")
    gate = jnp.einsum("td,di->ti", x, shared_gate)
    up = jnp.einsum("td,di->ti", x, shared_up)
    shared = jnp.einsum(
        "ti,id->td",
        jax.nn.silu(gate) * up,
        shared_down,
        out_sharding=NamedSharding(mesh, _batch_spec_from_x(x, mesh)),
    )
    return (routed.astype(jnp.float32) + shared.astype(jnp.float32)).astype(x.dtype)


def _custom_forward(
    *,
    mesh: jax.sharding.Mesh | jax.sharding.AbstractMesh,
    config: MoKForwardConfig,
) -> Callable[..., tuple[jax.Array, jax.Array]]:
    @jax.custom_vjp
    def fused(
        x: jax.Array,
        selected_experts: jax.Array,
        combine_weights: jax.Array,
        w_gate: jax.Array,
        w_up: jax.Array,
        w_down: jax.Array,
        shared_gate: jax.Array,
        shared_up: jax.Array,
        shared_down: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        return _fused_forward(
            x,
            selected_experts,
            combine_weights,
            w_gate,
            w_up,
            w_down,
            shared_gate,
            shared_up,
            shared_down,
            mesh=mesh,
            config=config,
        )

    def fused_fwd(
        *arguments: jax.Array,
    ) -> tuple[tuple[jax.Array, jax.Array], tuple[tuple[jax.Array, ...], MoKForwardContext]]:
        output, dropped_assignments, context = _fused_forward_with_context(*arguments, mesh=mesh, config=config)
        return (output, dropped_assignments), (arguments, context)

    def fused_bwd(
        residual: tuple[tuple[jax.Array, ...], MoKForwardContext],
        output_gradients: tuple[jax.Array, jax.Array],
    ) -> tuple[jax.Array | None, ...]:
        output_gradient, _ = output_gradients
        arguments, context = residual
        x, selected_experts, combine_weights, w_gate, w_up, w_down, shared_gate, shared_up, shared_down = arguments
        gradients = _fused_backward(
            output_gradient,
            x,
            selected_experts,
            combine_weights,
            w_gate,
            w_up,
            w_down,
            shared_gate,
            shared_up,
            shared_down,
            context,
            mesh=mesh,
            config=config,
        )
        return gradients[0], None, *gradients[1:]

    fused.defvjp(fused_fwd, fused_bwd)
    return fused


def mixture_of_kittens_mlp(
    x: jax.Array,
    selected_experts: jax.Array,
    combine_weights: jax.Array,
    w_gate: jax.Array,
    w_up: jax.Array,
    w_down: jax.Array,
    shared_gate: jax.Array,
    shared_up: jax.Array,
    shared_down: jax.Array,
    *,
    mesh: jax.sharding.Mesh | jax.sharding.AbstractMesh,
    config: MoKForwardConfig,
) -> tuple[jax.Array, jax.Array]:
    """Run the fused BF16 forward and backward."""
    forward = _custom_forward(
        mesh=mesh,
        config=config,
    )
    output, dropped_assignments = forward(
        x,
        selected_experts,
        combine_weights,
        w_gate,
        w_up,
        w_down,
        shared_gate,
        shared_up,
        shared_down,
    )
    return tree_checkpoint_name(output, _CHECKPOINT_MOE_OUTPUT), dropped_assignments
