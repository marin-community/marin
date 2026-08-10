# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Fused Mixture-of-Kittens forward with a JAX reference gradient."""

from collections.abc import Callable

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P
from levanter.grug._moe.common import MoeImplementation
from levanter.grug.grug_moe import moe_mlp
from levanter.grug.sharding import _batch_spec_from_x, _reshard_for_shard_map
from levanter.kernels.mixture_of_kittens.forward_ffi import (
    MoKForwardConfig,
    forward_bf16_local,
    schedule_capacity,
)
from levanter.utils.activation import ActivationFunctionEnum

from experiments.grug.mixture_of_kittens.schedule import build_schedule

_EXPERT_AXIS = "expert"
_NUM_DEVICES = 4


def _validate_topology(mesh: jax.sharding.Mesh | jax.sharding.AbstractMesh, config: MoKForwardConfig) -> None:
    if mesh.empty or int(mesh.shape.get(_EXPERT_AXIS, 1)) != _NUM_DEVICES:
        raise ValueError("Mixture-of-Kittens requires an expert axis of size four")
    if jax.process_count() != 1 or jax.local_device_count() != _NUM_DEVICES:
        raise ValueError("Mixture-of-Kittens requires one JAX process with four visible GPUs")
    if config.schedule_capacity_multiplier < _NUM_DEVICES:
        raise ValueError("schedule_capacity_multiplier must cover all four source ranks")


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
) -> jax.Array:
    _validate_topology(mesh, config)
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
    ) -> jax.Array:
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
        return forward_bf16_local(
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
        out_specs=batch_spec,
        check_vma=False,
    )(x, selected_experts, combine_weights, w_gate, w_up, w_down, shared_gate, shared_up, shared_down)


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
        capacity_factor=float(config.schedule_capacity_multiplier),
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
        out_sharding=_batch_spec_from_x(x, mesh),
    )
    return (routed.astype(jnp.float32) + shared.astype(jnp.float32)).astype(x.dtype)


def _custom_forward(
    *,
    mesh: jax.sharding.Mesh | jax.sharding.AbstractMesh,
    config: MoKForwardConfig,
    fallback_implementation: MoeImplementation,
    ragged_all_to_all_splits_per_peer: int,
) -> Callable[..., jax.Array]:
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
    ) -> jax.Array:
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

    def fused_fwd(*arguments: jax.Array) -> tuple[jax.Array, tuple[jax.Array, ...]]:
        output = _fused_forward(*arguments, mesh=mesh, config=config)
        return output, arguments

    def fused_bwd(
        residuals: tuple[jax.Array, ...],
        output_gradient: jax.Array,
    ) -> tuple[jax.Array | None, ...]:
        x, selected_experts, combine_weights, w_gate, w_up, w_down, shared_gate, shared_up, shared_down = residuals

        def reference(*differentiable: jax.Array) -> jax.Array:
            (
                reference_x,
                reference_combine_weights,
                reference_w_gate,
                reference_w_up,
                reference_w_down,
                reference_shared_gate,
                reference_shared_up,
                reference_shared_down,
            ) = differentiable
            return mixture_of_kittens_reference(
                reference_x,
                selected_experts,
                reference_combine_weights,
                reference_w_gate,
                reference_w_up,
                reference_w_down,
                reference_shared_gate,
                reference_shared_up,
                reference_shared_down,
                mesh=mesh,
                config=config,
                fallback_implementation=fallback_implementation,
                ragged_all_to_all_splits_per_peer=ragged_all_to_all_splits_per_peer,
            )

        differentiable = (x, combine_weights, w_gate, w_up, w_down, shared_gate, shared_up, shared_down)
        _, pullback = jax.vjp(reference, *differentiable)
        gradients = pullback(output_gradient)
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
    fallback_implementation: MoeImplementation,
    ragged_all_to_all_splits_per_peer: int,
) -> jax.Array:
    """Run the fused forward and differentiate through the JAX MoE path."""
    forward = _custom_forward(
        mesh=mesh,
        config=config,
        fallback_implementation=fallback_implementation,
        ragged_all_to_all_splits_per_peer=ragged_all_to_all_splits_per_peer,
    )
    return forward(
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
