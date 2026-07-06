# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Opt-in public-shape adapter for the source-push MGPU MoE forward prototype."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import numpy as np
from jax.sharding import AbstractMesh, AxisType, Mesh, NamedSharding, PartitionSpec as P
from jaxtyping import Array, Float, Int

from levanter.grug._moe.source_push_forward import (
    FORWARD_EXECUTION_STAGED_DEVICE_SYNC,
    FORWARD_EXECUTION_STAGED_HOST_SYNC,
    SourcePushForwardHostInputs,
    make_source_push_forward_inputs,
)
from levanter.grug._moe.source_push_inbox import AXIS, PushInboxConfig
from levanter.grug._moe.source_push_inbox_blackwell import BLACKWELL_SOURCE_PUSH_STRATEGY
from levanter.grug._moe.source_push_mlp import (
    SOURCE_PUSH_MLP_IMPLEMENTATION_BLACKWELL_STAGED,
    SOURCE_PUSH_MLP_IMPLEMENTATION_PALLAS_MGPU,
    SourcePushMlpRouteTable,
    source_push_mlp_route_table_from_plan,
    source_push_moe_mlp_from_plan,
)
from levanter.grug._moe.source_push_plan import build_source_push_plan
from levanter.utils.activation import ActivationFunctionEnum


SOURCE_PUSH_PUBLIC_IMPLEMENTATION = "pallas_mgpu_source_push"
SOURCE_PUSH_PUBLIC_IMPLEMENTATION_BLACKWELL = "pallas_mgpu_source_push_blackwell"
SOURCE_PUSH_PUBLIC_IMPLEMENTATIONS = (SOURCE_PUSH_PUBLIC_IMPLEMENTATION, SOURCE_PUSH_PUBLIC_IMPLEMENTATION_BLACKWELL)
_BLOCK_M = 64
_BLOCK_N = 128
_TARGET_BLOCK_K = 128
_SMALL_BLOCK_K = 64
_MAX_EP_SIZE = 8
_HOPPER_MAX_INBOX_SLOTS = 12
_BLACKWELL_MAX_INBOX_SLOTS = 24
_HOPPER_SEND_WORKER_PROGRAMS_PER_PEER = 2
_BLACKWELL_SEND_WORKER_PROGRAMS_PER_PEER = 4


@dataclass(frozen=True)
class SourcePushPublicPlan:
    """Reusable public-shape source-push plan for fixed route assignments."""

    config: PushInboxConfig
    host_inputs: SourcePushForwardHostInputs
    route_table: SourcePushMlpRouteTable
    mesh: Mesh
    batch_spec: P
    implementation: str
    x_shape: tuple[int, ...]
    selected_experts_shape: tuple[int, ...]
    combine_weights_shape: tuple[int, ...]
    w_up_gate_shape: tuple[int, ...]
    w_down_shape: tuple[int, ...]


def moe_mlp_ep_source_push_mgpu(
    x: Float[Array, "T D"],
    selected_experts: Int[Array, "T K"],
    combine_weights: Float[Array, "T K"],
    w_up_gate: Float[Array, "E D I2"],
    w_down: Float[Array, "E I D"],
    *,
    activation: ActivationFunctionEnum,
    mesh: Mesh | AbstractMesh,
    batch_spec: P,
    capacity_factor: float,
    implementation: str = SOURCE_PUSH_PUBLIC_IMPLEMENTATION,
) -> tuple[Float[Array, "T D"], Int[Array, ""]]:
    """Run the opt-in source-push MGPU backend from the public EP ``moe_mlp`` layout."""

    plan = prepare_moe_mlp_ep_source_push_mgpu_plan(
        x,
        selected_experts,
        combine_weights,
        w_up_gate,
        w_down,
        activation=activation,
        mesh=mesh,
        batch_spec=batch_spec,
        capacity_factor=capacity_factor,
        implementation=implementation,
    )
    return moe_mlp_ep_source_push_mgpu_from_plan(plan, x, combine_weights, w_up_gate, w_down)


def prepare_moe_mlp_ep_source_push_mgpu_plan(
    x: Float[Array, "T D"],
    selected_experts: Int[Array, "T K"],
    combine_weights: Float[Array, "T K"],
    w_up_gate: Float[Array, "E D I2"],
    w_down: Float[Array, "E I D"],
    *,
    activation: ActivationFunctionEnum,
    mesh: Mesh | AbstractMesh,
    batch_spec: P,
    capacity_factor: float,
    implementation: str = SOURCE_PUSH_PUBLIC_IMPLEMENTATION,
) -> SourcePushPublicPlan:
    """Prepare reusable source-push metadata from public EP ``moe_mlp`` inputs."""

    _validate_source_push_public_implementation(implementation)
    _validate_source_push_public_request(
        implementation=implementation,
        activation=activation,
        mesh=mesh,
        x=x,
        selected_experts=selected_experts,
        combine_weights=combine_weights,
        w_up_gate=w_up_gate,
        w_down=w_down,
    )

    ep_size = int(mesh.shape[AXIS])
    tokens_per_rank = x.shape[0] // ep_size
    experts_per_rank = w_up_gate.shape[0] // ep_size
    intermediate_dim = w_down.shape[1]
    hidden_dim = x.shape[1]
    topk = selected_experts.shape[1]

    selected_source = selected_experts.reshape(ep_size, tokens_per_rank, topk)
    combine_source = combine_weights.reshape(ep_size, tokens_per_rank, topk)

    config = _source_push_config_from_public_inputs(
        selected_source,
        combine_source,
        ep_size=ep_size,
        tokens_per_rank=tokens_per_rank,
        topk=topk,
        hidden_dim=hidden_dim,
        intermediate_dim=intermediate_dim,
        experts_per_rank=experts_per_rank,
        capacity_factor=capacity_factor,
        implementation=implementation,
    )
    if not isinstance(mesh, Mesh):
        raise ValueError(f"{implementation!r} requires a concrete Mesh")

    x_source, selected_source, combine_source, w_gate_up_source, w_down_source = _source_push_public_source_arrays(
        config,
        x,
        selected_experts,
        combine_weights,
        w_up_gate,
        w_down,
    )
    host_inputs = make_source_push_forward_inputs(
        config,
        x_source,
        selected_source,
        combine_source,
        w_gate_up_source,
        w_down_source,
    )
    route_table = source_push_mlp_route_table_from_plan(
        host_inputs.plan,
        src_base_by_expert=host_inputs.src_base_by_expert,
    )
    return SourcePushPublicPlan(
        config=config,
        host_inputs=host_inputs,
        route_table=route_table,
        mesh=mesh,
        batch_spec=batch_spec,
        implementation=implementation,
        x_shape=tuple(x.shape),
        selected_experts_shape=tuple(selected_experts.shape),
        combine_weights_shape=tuple(combine_weights.shape),
        w_up_gate_shape=tuple(w_up_gate.shape),
        w_down_shape=tuple(w_down.shape),
    )


def moe_mlp_ep_source_push_mgpu_from_plan(
    plan: SourcePushPublicPlan,
    x: Float[Array, "T D"],
    combine_weights: Float[Array, "T K"],
    w_up_gate: Float[Array, "E D I2"],
    w_down: Float[Array, "E I D"],
) -> tuple[Float[Array, "T D"], Int[Array, ""]]:
    """Run public EP source-push from a reusable route plan."""

    _validate_source_push_public_plan_call(plan, x, combine_weights, w_up_gate, w_down)
    x_source, combine_source, w_gate_up_source, w_down_source = _source_push_public_dynamic_source_arrays(
        plan.config,
        x,
        combine_weights,
        w_up_gate,
        w_down,
    )
    # The fixed plan gathers route weights by source/token/slot before it shards
    # stage inputs, so avoid inheriting public expert sharding for this tensor.
    combine_source = jax.device_put(combine_source, NamedSharding(plan.mesh, P(None, None, None)))
    with jax.set_mesh(plan.mesh):
        out_source, dropped = source_push_moe_mlp_from_plan(
            plan.config,
            plan.host_inputs,
            plan.route_table,
            x_source,
            combine_source,
            w_gate_up_source,
            w_down_source,
            implementation=_source_push_mlp_implementation(plan.implementation),
            execution_mode=_source_push_execution_mode(plan.implementation),
            mesh=plan.mesh,
        )
    out = out_source.reshape(plan.x_shape)
    return jax.device_put(out, NamedSharding(plan.mesh, plan.batch_spec)), dropped


def _source_push_public_source_arrays(
    config: PushInboxConfig,
    x: jax.Array,
    selected_experts: jax.Array,
    combine_weights: jax.Array,
    w_up_gate: jax.Array,
    w_down: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    ep_size = config.ep_size
    tokens_per_rank = config.tokens_per_rank
    hidden_dim = config.hidden_dim
    intermediate_dim = config.intermediate_dim
    experts_per_rank = config.experts_per_rank
    topk = config.topk
    x_source = x.reshape(ep_size, tokens_per_rank, hidden_dim)
    selected_source = selected_experts.reshape(ep_size, tokens_per_rank, topk)
    combine_source = combine_weights.reshape(ep_size, tokens_per_rank, topk)
    w_gate_up_source = w_up_gate.reshape(
        ep_size,
        experts_per_rank,
        hidden_dim,
        2 * intermediate_dim,
    )
    w_down_source = w_down.reshape(
        ep_size,
        experts_per_rank,
        intermediate_dim,
        hidden_dim,
    )
    return x_source, selected_source, combine_source, w_gate_up_source, w_down_source


def _source_push_public_dynamic_source_arrays(
    config: PushInboxConfig,
    x: jax.Array,
    combine_weights: jax.Array,
    w_up_gate: jax.Array,
    w_down: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    ep_size = config.ep_size
    tokens_per_rank = config.tokens_per_rank
    hidden_dim = config.hidden_dim
    intermediate_dim = config.intermediate_dim
    experts_per_rank = config.experts_per_rank
    topk = config.topk
    x_source = x.reshape(ep_size, tokens_per_rank, hidden_dim)
    combine_source = combine_weights.reshape(ep_size, tokens_per_rank, topk)
    w_gate_up_source = w_up_gate.reshape(
        ep_size,
        experts_per_rank,
        hidden_dim,
        2 * intermediate_dim,
    )
    w_down_source = w_down.reshape(
        ep_size,
        experts_per_rank,
        intermediate_dim,
        hidden_dim,
    )
    return x_source, combine_source, w_gate_up_source, w_down_source


def _validate_source_push_public_plan_call(
    plan: SourcePushPublicPlan,
    x: jax.Array,
    combine_weights: jax.Array,
    w_up_gate: jax.Array,
    w_down: jax.Array,
) -> None:
    if tuple(x.shape) != plan.x_shape:
        raise ValueError(f"x shape {x.shape} must match prepared source-push plan shape {plan.x_shape}")
    if tuple(combine_weights.shape) != plan.combine_weights_shape:
        raise ValueError(
            f"combine_weights shape {combine_weights.shape} must match prepared source-push plan shape "
            f"{plan.combine_weights_shape}"
        )
    if tuple(w_up_gate.shape) != plan.w_up_gate_shape:
        raise ValueError(
            f"w_up_gate shape {w_up_gate.shape} must match prepared source-push plan shape {plan.w_up_gate_shape}"
        )
    if tuple(w_down.shape) != plan.w_down_shape:
        raise ValueError(f"w_down shape {w_down.shape} must match prepared source-push plan shape {plan.w_down_shape}")


def _validate_source_push_public_request(
    *,
    implementation: str,
    activation: ActivationFunctionEnum,
    mesh: Mesh | AbstractMesh,
    x: jax.Array,
    selected_experts: jax.Array,
    combine_weights: jax.Array,
    w_up_gate: jax.Array,
    w_down: jax.Array,
) -> None:
    _validate_source_push_public_implementation(implementation)
    if activation is not ActivationFunctionEnum.silu:
        raise ValueError(f"{implementation!r} only supports ActivationFunctionEnum.silu")
    if isinstance(mesh, AbstractMesh):
        raise ValueError(f"{implementation!r} requires a concrete source-push MGPU mesh, got AbstractMesh")
    if AXIS not in mesh.shape:
        raise ValueError(f"{implementation!r} requires an {AXIS!r} mesh axis")
    ep_size = int(mesh.shape[AXIS])
    if ep_size <= 1 or ep_size > _MAX_EP_SIZE:
        raise ValueError(f"{implementation!r} supports 2..{_MAX_EP_SIZE} EP ranks, got {ep_size}")
    for axis_name, axis_size in mesh.shape.items():
        if axis_name != AXIS and int(axis_size) != 1:
            raise ValueError(
                f"{implementation!r} currently supports only the {AXIS!r} axis; "
                f"got nontrivial axis {axis_name!r}={axis_size}"
            )
    axis_type_by_name = dict(zip(mesh.axis_names, mesh.axis_types, strict=True))
    if axis_type_by_name.get(AXIS) is not AxisType.Explicit:
        raise ValueError(f"{implementation!r} requires an explicit {AXIS!r} mesh axis")

    devices = np.asarray(mesh.devices, dtype=object).reshape(-1)
    if not all(getattr(device, "platform", None) == "gpu" for device in devices):
        raise ValueError(f"{implementation!r} requires GPU devices")
    if not all(_source_push_device_matches_implementation(device, implementation) for device in devices):
        expected = "H100" if implementation == SOURCE_PUSH_PUBLIC_IMPLEMENTATION else "B200/B300/GB200/GB300"
        raise ValueError(f"{implementation!r} requires {expected} devices")

    if x.shape[0] % ep_size:
        raise ValueError(f"x token dimension {x.shape[0]} must be divisible by ep_size={ep_size}")
    if w_up_gate.shape[0] % ep_size:
        raise ValueError(f"w_up_gate expert dimension {w_up_gate.shape[0]} must be divisible by ep_size={ep_size}")
    if w_down.shape[0] != w_up_gate.shape[0]:
        raise ValueError(f"w_down expert dimension {w_down.shape[0]} must match w_up_gate {w_up_gate.shape[0]}")
    if w_up_gate.shape[1] != x.shape[1] or w_down.shape[2] != x.shape[1]:
        raise ValueError(
            f"source-push hidden dimensions must agree; got x={x.shape}, w_up_gate={w_up_gate.shape}, "
            f"w_down={w_down.shape}"
        )
    if w_up_gate.shape[2] != 2 * w_down.shape[1]:
        raise ValueError(
            f"w_up_gate output dimension must be 2 * w_down intermediate dim; got {w_up_gate.shape} and {w_down.shape}"
        )
    if selected_experts.shape != combine_weights.shape:
        raise ValueError(
            f"selected_experts shape {selected_experts.shape} must match combine_weights {combine_weights.shape}"
        )


def _source_push_config_from_public_inputs(
    selected_experts: Int[Array, "S T K"],
    combine_weights: Float[Array, "S T K"],
    *,
    ep_size: int,
    tokens_per_rank: int,
    topk: int,
    hidden_dim: int,
    intermediate_dim: int,
    experts_per_rank: int,
    capacity_factor: float,
    implementation: str = SOURCE_PUSH_PUBLIC_IMPLEMENTATION,
) -> PushInboxConfig:
    _validate_source_push_public_implementation(implementation)
    block_k = _source_push_block_k(hidden_dim)
    if intermediate_dim % _BLOCK_N:
        raise ValueError(
            f"{implementation!r} requires intermediate_dim divisible by {_BLOCK_N}, " f"got {intermediate_dim}"
        )

    probe_plan = build_source_push_plan(
        selected_experts,
        combine_weights,
        ep_size=ep_size,
        experts_per_rank=experts_per_rank,
        block_m=_BLOCK_M,
        capacity_factor=capacity_factor,
        entries_per_dst=None,
    )
    entries_per_rank = max(1, int(probe_plan.assignment_ids.shape[2]))
    n_work_groups = intermediate_dim // _BLOCK_N
    n_groups_per_job = min(2, n_work_groups)
    send_worker_programs_per_peer = _source_push_send_worker_programs_per_peer(implementation, entries_per_rank)
    worker_programs_per_peer = 8 if entries_per_rank <= 2 else 32
    return PushInboxConfig(
        ep_size=ep_size,
        entries_per_rank=entries_per_rank,
        inbox_slots=max(1, min(_source_push_max_inbox_slots(implementation), entries_per_rank)),
        hidden_dim=hidden_dim,
        intermediate_dim=intermediate_dim,
        block_m=_BLOCK_M,
        block_n=_BLOCK_N,
        block_k=block_k,
        n_group=1,
        n_groups_per_job=n_groups_per_job,
        experts_per_rank=experts_per_rank,
        send_worker_programs_per_peer=send_worker_programs_per_peer,
        worker_programs_per_peer=worker_programs_per_peer,
        send_pipeline_depth=1,
        routing="balanced",
        tokens_per_rank=tokens_per_rank,
        topk=topk,
        capacity_factor=capacity_factor,
    )


def _source_push_block_k(hidden_dim: int) -> int:
    if hidden_dim >= 256 and hidden_dim % _TARGET_BLOCK_K == 0:
        return _TARGET_BLOCK_K
    if hidden_dim % _SMALL_BLOCK_K == 0:
        return _SMALL_BLOCK_K
    raise ValueError(
        f"{SOURCE_PUSH_PUBLIC_IMPLEMENTATION!r} requires hidden_dim divisible by {_SMALL_BLOCK_K}, got {hidden_dim}"
    )


def is_source_push_public_implementation(implementation: str) -> bool:
    return implementation in SOURCE_PUSH_PUBLIC_IMPLEMENTATIONS


def _validate_source_push_public_implementation(implementation: str) -> None:
    if implementation not in SOURCE_PUSH_PUBLIC_IMPLEMENTATIONS:
        raise ValueError(
            "source-push public implementation must be one of "
            f"{SOURCE_PUSH_PUBLIC_IMPLEMENTATIONS}, got {implementation!r}"
        )


def _source_push_device_matches_implementation(device: jax.Device, implementation: str) -> bool:
    device_kind = getattr(device, "device_kind", "").upper()
    if implementation == SOURCE_PUSH_PUBLIC_IMPLEMENTATION:
        return "H100" in device_kind
    return any(name in device_kind for name in ("B200", "B300", "GB200", "GB300"))


def _source_push_max_inbox_slots(implementation: str) -> int:
    if implementation == SOURCE_PUSH_PUBLIC_IMPLEMENTATION_BLACKWELL:
        return _BLACKWELL_MAX_INBOX_SLOTS
    return _HOPPER_MAX_INBOX_SLOTS


def _source_push_send_worker_programs_per_peer(implementation: str, entries_per_rank: int) -> int:
    if entries_per_rank <= 2:
        return 1
    if implementation == SOURCE_PUSH_PUBLIC_IMPLEMENTATION_BLACKWELL:
        return _BLACKWELL_SEND_WORKER_PROGRAMS_PER_PEER
    return _HOPPER_SEND_WORKER_PROGRAMS_PER_PEER


def _source_push_mlp_implementation(implementation: str) -> str:
    if implementation == SOURCE_PUSH_PUBLIC_IMPLEMENTATION_BLACKWELL:
        assert BLACKWELL_SOURCE_PUSH_STRATEGY.value == "staged_copy_local_w13"
        return SOURCE_PUSH_MLP_IMPLEMENTATION_BLACKWELL_STAGED
    return SOURCE_PUSH_MLP_IMPLEMENTATION_PALLAS_MGPU


def _source_push_execution_mode(implementation: str) -> str:
    _validate_source_push_public_implementation(implementation)
    if implementation == SOURCE_PUSH_PUBLIC_IMPLEMENTATION_BLACKWELL:
        return FORWARD_EXECUTION_STAGED_DEVICE_SYNC
    return FORWARD_EXECUTION_STAGED_HOST_SYNC
