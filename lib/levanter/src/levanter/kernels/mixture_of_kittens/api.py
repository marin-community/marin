# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Fused Mixture-of-Kittens forward and backward JAX integration."""

from collections.abc import Callable

import jax
import jax.numpy as jnp
from haliax.jax_utils import tree_checkpoint_name
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from levanter.grug._moe.common import _CHECKPOINT_MOE_OUTPUT, MoeImplementation
from levanter.grug.grug_moe import moe_mlp
from levanter.grug.sharding import _batch_spec_from_x, _reshard_for_shard_map
from levanter.kernels.mixture_of_kittens.config import (
    EXPERT_AXIS as _EXPERT_AXIS,
)
from levanter.kernels.mixture_of_kittens.config import _DEVICES_PER_NODE, MokLikeConfig, MokLikeWorkspaceTransport
from levanter.kernels.mixture_of_kittens.source import SUPPORTED_NUM_DEVICES
from levanter.kernels.mixture_of_kittens.ffi import (
    MokLikeForwardContext,
    backward_bf16_local,
    fence_mok_like_failure,
    forward_bf16_local,
)
from levanter.kernels.mixture_of_kittens.runtime import MokLikeRuntimeHandle, validate_mok_like_mesh_topology
from levanter.kernels.mixture_of_kittens.schedule import build_schedule, schedule_capacity
from levanter.utils.activation import ActivationFunctionEnum

_NUM_DEVICES = 4
MOK_CONTEXT_CHECKPOINT_NAME = "mixture_of_kittens_context"


def _failure_agreement_axes(mesh: jax.sharding.Mesh | jax.sharding.AbstractMesh) -> tuple[str, ...]:
    return tuple(axis_name for axis_name in mesh.axis_names if axis_name != _EXPERT_AXIS)


def _failure_outputs(marker: jax.Array, values: tuple[jax.Array, ...]) -> tuple[jax.Array, ...]:
    """Make one failure-fence result data-dependent on every branch output."""
    fenced_marker = fence_mok_like_failure(marker)
    return tuple(jnp.broadcast_to(fenced_marker.astype(value.dtype), value.shape) for value in values)


def validate_mok_like_inputs(
    x: jax.Array,
    selected_experts: jax.Array,
    combine_weights: jax.Array,
    w_gate: jax.Array,
    w_up: jax.Array,
    w_down: jax.Array,
    shared_gate: jax.Array,
    shared_up: jax.Array,
    shared_down: jax.Array,
    routed_x: jax.Array | None = None,
    latent_up: jax.Array | None = None,
) -> None:
    """Validate the canonical Grug leaves before tracing native work.

    Under a latent projection the routed and shared paths run at different widths: `routed_x` and
    the routed weights carry `latent_dim`, the shared weights keep `hidden_dim`, and `latent_up`
    spans the two. Without it both paths share `x`'s width.
    """

    if x.ndim != 2:
        raise ValueError(f"x must have shape [tokens, hidden], got {x.shape}")
    if (routed_x is None) != (latent_up is None):
        raise ValueError("routed_x and latent_up describe one latent projection and must be given together")
    if routed_x is not None and routed_x.ndim != 2:
        raise ValueError(f"routed_x must have shape [tokens, latent], got {routed_x.shape}")
    if routed_x is not None and routed_x.shape[0] != x.shape[0]:
        raise ValueError("routed_x token count must match x")
    if selected_experts.ndim != 2 or selected_experts.shape != combine_weights.shape:
        raise ValueError("selected_experts and combine_weights must have identical [tokens, top_k] shapes")
    if selected_experts.shape[0] != x.shape[0]:
        raise ValueError("routing token count must match x")
    if selected_experts.dtype != jnp.int32:
        raise TypeError(f"selected_experts must be int32, got {selected_experts.dtype}")
    if combine_weights.dtype != jnp.float32:
        raise TypeError(f"combine_weights must be float32, got {combine_weights.dtype}")
    arrays = (x, w_gate, w_up, w_down, shared_gate, shared_up, shared_down)
    if routed_x is not None:
        arrays += (routed_x, latent_up)
    if any(array.dtype != jnp.bfloat16 for array in arrays):
        dtypes = tuple(array.dtype for array in arrays)
        raise TypeError(f"mok_like activations and weights must be bfloat16, got {dtypes}")

    tokens, hidden_dim = x.shape
    routed_dim = hidden_dim if routed_x is None else routed_x.shape[1]
    if w_gate.ndim != 3 or w_up.shape != w_gate.shape:
        raise ValueError("routed gate and up weights must have identical [experts, hidden, intermediate] shapes")
    # The routed experts carry their own intermediate width. The hero widens them against the
    # shared expert, so this is a second axis on which the two paths differ.
    num_experts, routed_hidden, routed_intermediate_dim = w_gate.shape
    if num_experts <= 0 or routed_hidden != routed_dim:
        raise ValueError("routed gate/up hidden dimension must match the routed input")
    if w_down.shape != (num_experts, routed_intermediate_dim, routed_dim):
        raise ValueError("routed down weights must have shape [experts, intermediate, routed hidden]")
    if latent_up is not None and latent_up.shape != (routed_dim, hidden_dim):
        raise ValueError("latent_up must have shape [latent, hidden]")
    if shared_gate.ndim != 2 or shared_gate.shape[0] != hidden_dim or shared_up.shape != shared_gate.shape:
        raise ValueError("shared gate/up weights must have shape [hidden, intermediate]")
    intermediate_dim = shared_gate.shape[1]
    if shared_down.shape != (intermediate_dim, hidden_dim):
        raise ValueError("shared down weights must have shape [intermediate, hidden]")
    if (
        tokens <= 0
        or hidden_dim % 256 != 0
        or intermediate_dim % 256 != 0
        or routed_intermediate_dim % 256 != 0
        or routed_dim % 256 != 0
    ):
        raise ValueError("tokens must be positive and hidden/intermediate/routed dimensions must be divisible by 256")


def _validate_topology(
    mesh: jax.sharding.Mesh | jax.sharding.AbstractMesh,
    num_devices: int = _NUM_DEVICES,
    workspace_transport: MokLikeWorkspaceTransport = MokLikeWorkspaceTransport.IN_PROCESS_PEER,
) -> None:
    """Check the mesh against the expert-group size the adapter was built for.

    Groups at or below one node's device count must be process-local, because the
    space-0 peer paths only reach devices the calling process owns. Larger groups
    necessarily span processes and rely on collective memory instead, so the
    process-local check is skipped and `MokLikeConfig` enforces the storage mode.
    """

    if num_devices not in SUPPORTED_NUM_DEVICES:
        supported = ", ".join(str(value) for value in SUPPORTED_NUM_DEVICES)
        raise ValueError(f"num_devices must be one of {supported}, got {num_devices!r}")
    if mesh.empty or int(mesh.shape.get(_EXPERT_AXIS, 1)) != num_devices:
        raise ValueError(f"Mixture-of-Kittens requires an expert axis of size {num_devices}")
    if workspace_transport.crosses_processes:
        # One rank per process: the group is assembled from imported fabric arenas, so this
        # process owns a single device no matter how wide the expert axis is.
        if jax.local_device_count() != 1:
            raise ValueError(
                f"Mixture-of-Kittens under {workspace_transport.value} places one rank per "
                f"process, so each process must see exactly 1 GPU, got {jax.local_device_count()}"
            )
        return
    if num_devices <= _DEVICES_PER_NODE:
        if jax.local_device_count() != num_devices:
            raise ValueError(
                f"Mixture-of-Kittens at {num_devices} ranks requires {num_devices} "
                "visible local GPUs per JAX process"
            )
        if isinstance(mesh, jax.sharding.Mesh):
            validate_mok_like_mesh_topology(mesh)


def _fused_forward_with_context(
    x: jax.Array,
    routed_x: jax.Array,
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
    runtime: MokLikeRuntimeHandle,
    config: MokLikeConfig,
    collective_id: int,
) -> tuple[jax.Array, jax.Array, jax.Array, MokLikeForwardContext]:
    _validate_topology(mesh, config.num_devices, config.workspace_transport)
    batch_spec = _batch_spec_from_x(x, mesh)
    expert_weight_spec = P(_EXPERT_AXIS, None, None)
    shared_weight_spec = P(None, None)

    x = _reshard_for_shard_map(x, mesh, batch_spec)
    routed_x = _reshard_for_shard_map(routed_x, mesh, batch_spec)
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
        local_routed_x: jax.Array,
        local_selected_experts: jax.Array,
        local_combine_weights: jax.Array,
        local_w_gate: jax.Array,
        local_w_up: jax.Array,
        local_w_down: jax.Array,
        local_shared_gate: jax.Array,
        local_shared_up: jax.Array,
        local_shared_down: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array, MokLikeForwardContext]:
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
        routed_output, shared_output, context, local_failure = forward_bf16_local(
            local_x,
            local_routed_x,
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
            runtime=runtime,
            config=config,
            collective_id=collective_id,
        )
        # Native cancellation already makes the process-local four-device expert
        # island uniform. Reduce one scalar per expert lane across every remaining
        # (process-spanning) mesh axis instead of forming an all-device group.
        global_failure = jax.lax.pmax(local_failure, _failure_agreement_axes(mesh))

        def success(_: None) -> tuple[jax.Array, jax.Array, jax.Array, MokLikeForwardContext]:
            return (
                routed_output,
                shared_output,
                jax.lax.psum(schedule.dropped_assignments, _EXPERT_AXIS),
                context,
            )

        def failure(_: None) -> tuple[jax.Array, jax.Array, jax.Array, MokLikeForwardContext]:
            failed_routed, failed_shared, failed_dropped_assignments = _failure_outputs(
                global_failure,
                (routed_output, shared_output, schedule.dropped_assignments),
            )
            return failed_routed, failed_shared, failed_dropped_assignments, context

        return jax.lax.cond(global_failure == 0, success, failure, operand=None)

    routed_context_spec = P(_EXPERT_AXIS, None)
    context_specs = MokLikeForwardContext(
        x_routed=routed_context_spec,
        gate_shared=batch_spec,
        gate_routed=routed_context_spec,
        up_shared=batch_spec,
        up_routed=routed_context_spec,
        hidden_shared=batch_spec,
        hidden_routed=routed_context_spec,
        stamp_slot=P(_EXPERT_AXIS),
        stamp_generation_high=P(_EXPERT_AXIS),
        stamp_generation_low=P(_EXPERT_AXIS),
        stamp_runtime_epoch=P(_EXPERT_AXIS),
    )

    return jax.shard_map(
        local_forward,
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
        ),
        out_specs=(batch_spec, batch_spec, P(), context_specs),
        check_vma=False,
    )(
        x,
        routed_x,
        selected_experts,
        combine_weights,
        w_gate,
        w_up,
        w_down,
        shared_gate,
        shared_up,
        shared_down,
    )


def _fused_forward(
    x: jax.Array,
    routed_x: jax.Array,
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
    runtime: MokLikeRuntimeHandle,
    config: MokLikeConfig,
    collective_id: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    routed_output, shared_output, dropped_assignments, _ = _fused_forward_with_context(
        x,
        routed_x,
        selected_experts,
        combine_weights,
        w_gate,
        w_up,
        w_down,
        shared_gate,
        shared_up,
        shared_down,
        mesh=mesh,
        runtime=runtime,
        config=config,
        collective_id=collective_id,
    )
    return routed_output, shared_output, dropped_assignments


def _fused_backward(
    output_gradient: jax.Array,
    routed_output_gradient: jax.Array,
    x: jax.Array,
    routed_x: jax.Array,
    selected_experts: jax.Array,
    combine_weights: jax.Array,
    w_gate: jax.Array,
    w_up: jax.Array,
    w_down: jax.Array,
    shared_gate: jax.Array,
    shared_up: jax.Array,
    shared_down: jax.Array,
    context: MokLikeForwardContext,
    *,
    mesh: jax.sharding.Mesh | jax.sharding.AbstractMesh,
    runtime: MokLikeRuntimeHandle,
    config: MokLikeConfig,
    collective_id: int,
) -> tuple[jax.Array, ...]:
    _validate_topology(mesh, config.num_devices, config.workspace_transport)
    batch_spec = _batch_spec_from_x(x, mesh)
    expert_weight_spec = P(_EXPERT_AXIS, None, None)
    shared_weight_spec = P(None, None)
    gradient_specs = tuple(
        getattr(jax.typeof(value).sharding, "spec", default)
        for value, default in (
            (x, batch_spec),
            (routed_x, batch_spec),
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
    context_specs = MokLikeForwardContext(
        x_routed=routed_context_spec,
        gate_shared=batch_spec,
        gate_routed=routed_context_spec,
        up_shared=batch_spec,
        up_routed=routed_context_spec,
        hidden_shared=batch_spec,
        hidden_routed=routed_context_spec,
        stamp_slot=P(_EXPERT_AXIS),
        stamp_generation_high=P(_EXPERT_AXIS),
        stamp_generation_low=P(_EXPERT_AXIS),
        stamp_runtime_epoch=P(_EXPERT_AXIS),
    )

    output_gradient = _reshard_for_shard_map(output_gradient, mesh, batch_spec)
    routed_output_gradient = _reshard_for_shard_map(routed_output_gradient, mesh, batch_spec)
    x = _reshard_for_shard_map(x, mesh, batch_spec)
    routed_x = _reshard_for_shard_map(routed_x, mesh, batch_spec)
    selected_experts = _reshard_for_shard_map(selected_experts, mesh, batch_spec)
    combine_weights = _reshard_for_shard_map(combine_weights, mesh, batch_spec)
    w_gate = _reshard_for_shard_map(w_gate, mesh, expert_weight_spec)
    w_up = _reshard_for_shard_map(w_up, mesh, expert_weight_spec)
    w_down = _reshard_for_shard_map(w_down, mesh, expert_weight_spec)
    shared_gate = _reshard_for_shard_map(shared_gate, mesh, shared_weight_spec)
    shared_up = _reshard_for_shard_map(shared_up, mesh, shared_weight_spec)
    shared_down = _reshard_for_shard_map(shared_down, mesh, shared_weight_spec)
    context = MokLikeForwardContext(
        *(_reshard_for_shard_map(value, mesh, spec) for value, spec in zip(context, context_specs, strict=True))
    )

    def local_backward(
        local_output_gradient: jax.Array,
        local_routed_output_gradient: jax.Array,
        local_x: jax.Array,
        local_routed_x: jax.Array,
        local_selected_experts: jax.Array,
        local_combine_weights: jax.Array,
        local_w_gate: jax.Array,
        local_w_up: jax.Array,
        local_w_down: jax.Array,
        local_shared_gate: jax.Array,
        local_shared_up: jax.Array,
        local_shared_down: jax.Array,
        local_context: MokLikeForwardContext,
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
        gradients, local_failure = backward_bf16_local(
            local_output_gradient,
            local_routed_output_gradient,
            local_x,
            local_routed_x,
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
            runtime=runtime,
            config=config,
            collective_id=collective_id,
        )
        (
            d_x,
            d_routed_x,
            d_combine_weights,
            d_w_gate,
            d_w_up,
            d_w_down,
            d_shared_gate,
            d_shared_up,
            d_shared_down,
        ) = gradients
        global_failure = jax.lax.pmax(local_failure, _failure_agreement_axes(mesh))

        def success(_: None) -> tuple[jax.Array, ...]:
            return (
                d_x.astype(local_x.dtype),
                d_routed_x.astype(local_routed_x.dtype),
                d_combine_weights.astype(local_combine_weights.dtype),
                jnp.transpose(d_w_gate, (0, 2, 1)).astype(local_w_gate.dtype),
                jnp.transpose(d_w_up, (0, 2, 1)).astype(local_w_up.dtype),
                jnp.transpose(d_w_down, (0, 2, 1)).astype(local_w_down.dtype),
                jnp.transpose(jax.lax.psum(d_shared_gate.astype(jnp.float32), _EXPERT_AXIS)).astype(
                    local_shared_gate.dtype
                ),
                jnp.transpose(jax.lax.psum(d_shared_up.astype(jnp.float32), _EXPERT_AXIS)).astype(
                    local_shared_up.dtype
                ),
                jnp.transpose(jax.lax.psum(d_shared_down.astype(jnp.float32), _EXPERT_AXIS)).astype(
                    local_shared_down.dtype
                ),
            )

        def failure(_: None) -> tuple[jax.Array, ...]:
            # The process-spanning pmax makes this branch uniform across
            # processes. It contains no collective before the typed FFI error.
            return _failure_outputs(
                global_failure,
                (
                    local_x,
                    local_routed_x,
                    local_combine_weights,
                    local_w_gate,
                    local_w_up,
                    local_w_down,
                    local_shared_gate,
                    local_shared_up,
                    local_shared_down,
                ),
            )

        return jax.lax.cond(
            global_failure == 0,
            success,
            failure,
            operand=None,
        )

    gradients = jax.shard_map(
        local_backward,
        mesh=mesh,
        in_specs=(
            batch_spec,
            batch_spec,
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
        routed_output_gradient,
        x,
        routed_x,
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


def mok_like_reference(
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
    config: MokLikeConfig,
    fallback_implementation: MoeImplementation,
    routed_x: jax.Array | None = None,
    latent_up: jax.Array | None = None,
) -> jax.Array:
    """Run the JAX forward used to differentiate the fused call.

    LatentMoE narrows only the routed path. `routed_x` carries the `latent_dim`-wide rows the
    experts consume, while the shared expert keeps reading the full-width `x`, and `latent_up`
    expands the combined routed result back to that width before the two are summed. Omitting both
    reproduces the single-width behaviour, where one `x` feeds both paths and no expansion happens.
    """
    if (routed_x is None) != (latent_up is None):
        raise ValueError("routed_x and latent_up describe one latent projection and must be given together")
    routed_input = x if routed_x is None else routed_x
    expert_axis_size = int(mesh.shape[_EXPERT_AXIS])
    reference_capacity_factor = schedule_capacity(
        x.shape[0] // expert_axis_size,
        selected_experts.shape[1],
        w_gate.shape[0] // expert_axis_size,
        config,
    ) / ((x.shape[0] // expert_axis_size) * selected_experts.shape[1])
    if fallback_implementation == "ring":
        # Ring selects from all assignments on one expert rank; padding
        # capacity past that population is invalid for lax.top_k.
        reference_capacity_factor = min(reference_capacity_factor, float(expert_axis_size))
    routed = moe_mlp(
        routed_input,
        selected_experts,
        combine_weights,
        jnp.concatenate((w_gate, w_up), axis=-1),
        w_down,
        activation=ActivationFunctionEnum.silu,
        implementation=fallback_implementation,
        mesh=mesh,
        capacity_factor=reference_capacity_factor,
    )
    if isinstance(routed, tuple):
        raise AssertionError("The fallback MoE returned capacity data when only output was requested")
    if latent_up is not None:
        # Expand after the combine: the experts already produced the weight-summed latent vector,
        # which is what the up-projection acts on.
        routed = jnp.einsum(
            "tl,ld->td",
            routed,
            latent_up.astype(routed.dtype),
            out_sharding=NamedSharding(mesh, _batch_spec_from_x(x, mesh)),
        )
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
    runtime: MokLikeRuntimeHandle,
    config: MokLikeConfig,
    collective_id: int,
) -> Callable[..., tuple[jax.Array, jax.Array, jax.Array]]:
    @jax.custom_vjp
    def fused(
        x: jax.Array,
        routed_x: jax.Array,
        selected_experts: jax.Array,
        combine_weights: jax.Array,
        w_gate: jax.Array,
        w_up: jax.Array,
        w_down: jax.Array,
        shared_gate: jax.Array,
        shared_up: jax.Array,
        shared_down: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        return _fused_forward(
            x,
            routed_x,
            selected_experts,
            combine_weights,
            w_gate,
            w_up,
            w_down,
            shared_gate,
            shared_up,
            shared_down,
            mesh=mesh,
            runtime=runtime,
            config=config,
            collective_id=collective_id,
        )

    def fused_fwd(
        *arguments: jax.Array,
    ) -> tuple[tuple[jax.Array, jax.Array, jax.Array], tuple[tuple[jax.Array, ...], MokLikeForwardContext]]:
        routed_output, shared_output, dropped_assignments, context = _fused_forward_with_context(
            *arguments, mesh=mesh, runtime=runtime, config=config, collective_id=collective_id
        )
        context = tree_checkpoint_name(context, MOK_CONTEXT_CHECKPOINT_NAME)
        return (routed_output, shared_output, dropped_assignments), (arguments, context)

    def fused_bwd(
        residual: tuple[tuple[jax.Array, ...], MokLikeForwardContext],
        output_gradients: tuple[jax.Array, jax.Array, jax.Array],
    ) -> tuple[jax.Array | None, ...]:
        routed_output_gradient, output_gradient, _ = output_gradients
        arguments, context = residual
        (
            x,
            routed_x,
            selected_experts,
            combine_weights,
            w_gate,
            w_up,
            w_down,
            shared_gate,
            shared_up,
            shared_down,
        ) = arguments
        gradients = _fused_backward(
            output_gradient,
            routed_output_gradient,
            x,
            routed_x,
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
            runtime=runtime,
            config=config,
            collective_id=collective_id,
        )
        return gradients[0], gradients[1], None, *gradients[2:]

    fused.defvjp(fused_fwd, fused_bwd)
    return fused


def mok_like_mlp(
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
    runtime: MokLikeRuntimeHandle,
    config: MokLikeConfig,
    collective_id: int,
    routed_x: jax.Array | None = None,
    latent_up: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Run the fused BF16 forward and backward for one explicit collective site.

    Under a latent projection the experts consume the narrower `routed_x` while the shared expert
    keeps reading `x`, and `latent_up` expands the combined routed result back to the shared width.
    Omitting both runs one width through both paths.
    """
    validate_mok_like_inputs(
        x,
        selected_experts,
        combine_weights,
        w_gate,
        w_up,
        w_down,
        shared_gate,
        shared_up,
        shared_down,
        routed_x=routed_x,
        latent_up=latent_up,
    )
    forward = _custom_forward(
        mesh=mesh,
        runtime=runtime,
        config=config,
        collective_id=collective_id,
    )
    routed_output, shared_output, dropped_assignments = forward(
        x,
        x if routed_x is None else routed_x,
        selected_experts,
        combine_weights,
        w_gate,
        w_up,
        w_down,
        shared_gate,
        shared_up,
        shared_down,
    )
    if latent_up is not None:
        # Name the combine before expanding it. Its transpose is what `w_latent_up`'s gradient
        # contracts against, so leaving it anonymous makes a remat policy recompute the whole fused
        # call to recover it -- the forward then runs twice per layer. It is the routed width, so
        # saving it costs half of what the composed output already costs.
        routed_output = tree_checkpoint_name(routed_output, _CHECKPOINT_MOE_OUTPUT)
        # Expand after the combine, matching the reference: the routed result is already the
        # weight-summed latent vector, so one projection covers every selected expert.
        routed_output = jnp.einsum(
            "tl,ld->td",
            routed_output,
            latent_up.astype(routed_output.dtype),
            out_sharding=NamedSharding(mesh, _batch_spec_from_x(x, mesh)),
        )
    output = (routed_output.astype(jnp.float32) + shared_output.astype(jnp.float32)).astype(x.dtype)
    return (
        tree_checkpoint_name(output, _CHECKPOINT_MOE_OUTPUT),
        tree_checkpoint_name(dropped_assignments, _CHECKPOINT_MOE_OUTPUT),
    )
