# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Opt-in public-shape adapter for the source-push MGPU MoE forward prototype."""

from __future__ import annotations

from typing import Any

import jax
import numpy as np
from jax.sharding import AbstractMesh, AxisType, Mesh, NamedSharding, PartitionSpec as P
from jaxtyping import Array, Float, Int

from levanter.grug._moe.source_push_forward import (
    FORWARD_EXECUTION_STAGED_HOST_SYNC,
    SOURCE_PUSH_FORWARD_IMPLEMENTATION_PALLAS_MGPU,
    source_push_forward,
)
from levanter.grug._moe.source_push_inbox import AXIS, PushInboxConfig
from levanter.grug._moe.source_push_plan import build_source_push_plan
from levanter.utils.activation import ActivationFunctionEnum


SOURCE_PUSH_PUBLIC_IMPLEMENTATION = "pallas_mgpu_source_push"
_BLOCK_M = 64
_BLOCK_N = 128
_TARGET_BLOCK_K = 128
_SMALL_BLOCK_K = 64
_MAX_EP_SIZE = 8


def moe_mlp_ep_source_push_mgpu(
    x: Float[Array, "T D"],
    selected_experts: Int[Array, "T K"],
    combine_weights: Float[Array, "T K"],
    w_up_gate: Float[Array, "E D I2"],
    w_down: Float[Array, "E I D"],
    *,
    activation: Any,
    mesh: Mesh | AbstractMesh,
    batch_spec: P,
    capacity_factor: float,
) -> tuple[Float[Array, "T D"], Int[Array, ""]]:
    """Run the staged source-push prototype from the public EP ``moe_mlp`` layout.

    This is deliberately narrow while source-push plan construction remains
    host-side. It gives users an explicit public implementation name for H100x8
    experiments without pretending the path is a general jittable backend.
    """

    _validate_source_push_public_request(
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
    )
    out_source, dropped = source_push_forward(
        config,
        x_source,
        selected_source,
        combine_source,
        w_gate_up_source,
        w_down_source,
        implementation=SOURCE_PUSH_FORWARD_IMPLEMENTATION_PALLAS_MGPU,
        execution_mode=FORWARD_EXECUTION_STAGED_HOST_SYNC,
        mesh=mesh,
    )
    out = out_source.reshape(x.shape)
    return jax.device_put(out, NamedSharding(mesh, batch_spec)), dropped


def _validate_source_push_public_request(
    *,
    activation: Any,
    mesh: Mesh | AbstractMesh,
    x: jax.Array,
    selected_experts: jax.Array,
    combine_weights: jax.Array,
    w_up_gate: jax.Array,
    w_down: jax.Array,
) -> None:
    if activation is not ActivationFunctionEnum.silu:
        raise ValueError(f"{SOURCE_PUSH_PUBLIC_IMPLEMENTATION!r} only supports ActivationFunctionEnum.silu")
    if isinstance(mesh, AbstractMesh):
        raise ValueError(f"{SOURCE_PUSH_PUBLIC_IMPLEMENTATION!r} requires a concrete H100 mesh, got AbstractMesh")
    if AXIS not in mesh.shape:
        raise ValueError(f"{SOURCE_PUSH_PUBLIC_IMPLEMENTATION!r} requires an {AXIS!r} mesh axis")
    ep_size = int(mesh.shape[AXIS])
    if ep_size <= 1 or ep_size > _MAX_EP_SIZE:
        raise ValueError(f"{SOURCE_PUSH_PUBLIC_IMPLEMENTATION!r} supports 2..{_MAX_EP_SIZE} EP ranks, got {ep_size}")
    for axis_name, axis_size in mesh.shape.items():
        if axis_name != AXIS and int(axis_size) != 1:
            raise ValueError(
                f"{SOURCE_PUSH_PUBLIC_IMPLEMENTATION!r} currently supports only the {AXIS!r} axis; "
                f"got nontrivial axis {axis_name!r}={axis_size}"
            )
    axis_type_by_name = dict(zip(mesh.axis_names, mesh.axis_types, strict=True))
    if axis_type_by_name.get(AXIS) is not AxisType.Explicit:
        raise ValueError(f"{SOURCE_PUSH_PUBLIC_IMPLEMENTATION!r} requires an explicit {AXIS!r} mesh axis")

    devices = np.asarray(mesh.devices, dtype=object).reshape(-1)
    if not all(getattr(device, "platform", None) == "gpu" for device in devices):
        raise ValueError(f"{SOURCE_PUSH_PUBLIC_IMPLEMENTATION!r} requires GPU devices")
    if not all("H100" in getattr(device, "device_kind", "").upper() for device in devices):
        raise ValueError(f"{SOURCE_PUSH_PUBLIC_IMPLEMENTATION!r} is currently restricted to H100 devices")

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
) -> PushInboxConfig:
    block_k = _source_push_block_k(hidden_dim)
    if intermediate_dim % _BLOCK_N:
        raise ValueError(
            f"{SOURCE_PUSH_PUBLIC_IMPLEMENTATION!r} requires intermediate_dim divisible by {_BLOCK_N}, "
            f"got {intermediate_dim}"
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
    send_worker_programs_per_peer = 1 if entries_per_rank <= 2 else 2
    worker_programs_per_peer = 8 if entries_per_rank <= 2 else 32
    return PushInboxConfig(
        ep_size=ep_size,
        entries_per_rank=entries_per_rank,
        inbox_slots=max(1, min(12, entries_per_rank)),
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
