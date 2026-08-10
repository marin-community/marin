# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Execute recovered collective Folds through JAX named-axis primitives."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from shuttle.ir import DType
from tile_lifetime.collective_transport import (
    CollectiveCompletionPlan,
    CollectiveReduction,
)
from tile_lifetime.event_dataflow import (
    EventMemoryScope,
    EventSchedulingMode,
    MemoryVisibility,
)
from tile_lifetime.event_dataflow_adapters import (
    CollectiveCompletionSchedule,
    CollectiveCompletionTaskDataflow,
    collective_completion_task_dataflow,
)
from tile_lifetime.plan import NumericalPolicy


@dataclass(frozen=True)
class JaxCollectiveExecutionPlan:
    """A JAX-owned transport boundary for one recovered collective Fold.

    ``device_id_by_axis_index`` is explicit because post-SPMD HLO replica
    groups may use global device IDs while JAX named-axis collectives use
    logical axis indices. This prototype lowers one complete value at a time;
    the data result of the named-axis primitive is the physical completion
    signal consumed downstream.
    """

    completion: CollectiveCompletionPlan
    dataflow: CollectiveCompletionTaskDataflow
    axis_name: str
    device_id_by_axis_index: tuple[int, ...]
    axis_index_groups: tuple[tuple[int, ...], ...]
    completion_visibility: MemoryVisibility


def build_jax_collective_execution_plan(
    completion: CollectiveCompletionPlan,
    *,
    axis_name: str,
    device_id_by_axis_index: tuple[int, ...],
    scheduling_mode: EventSchedulingMode = EventSchedulingMode.STATIC,
) -> JaxCollectiveExecutionPlan:
    """Bind a generic Fold/Transport plan to a JAX logical device axis.

    JAX and XLA retain ownership of automatic differentiation and the physical
    collective implementation. Shuttle supplies the recovered reducer,
    numerical contract, replica groups, and Event Tensor visibility contract.
    """
    if not axis_name:
        raise ValueError("collective axis name must be non-empty")
    if len(device_id_by_axis_index) != len(set(device_id_by_axis_index)):
        raise ValueError("collective axis device IDs must be unique")
    if not completion.transport.replica_domain.use_global_device_ids:
        raise ValueError(
            "JAX collective execution requires replica groups expressed as global device IDs; "
            "local replica-ID mapping is not implemented"
        )
    replica_groups = completion.transport.replica_domain.groups
    participating_devices = tuple(device for group in replica_groups for device in group)
    if set(participating_devices) != set(device_id_by_axis_index):
        raise ValueError(
            "collective axis device IDs must exactly match the recovered replica domain; "
            f"found axis devices {device_id_by_axis_index} and replica devices {participating_devices}"
        )
    if completion.fold.numerical_policy is not NumericalPolicy.ALLOW_ROUNDING_REORDER:
        raise ValueError(
            "JAX collective lowering requires allow_rounding_reorder because the physical reduction tree is not fixed"
        )
    if completion.fold.reduction is CollectiveReduction.PRODUCT:
        raise ValueError("JAX named-axis product reduction has no direct primitive in the supported JAX version")
    if completion.fold.dtype not in _JAX_DTYPES:
        raise ValueError(f"JAX collective lowering does not support dtype {completion.fold.dtype.value!r}")

    axis_index_by_device_id = {device_id: axis_index for axis_index, device_id in enumerate(device_id_by_axis_index)}
    axis_index_groups = tuple(
        tuple(axis_index_by_device_id[device_id] for device_id in group) for group in replica_groups
    )
    dataflow = collective_completion_task_dataflow(
        completion,
        schedule=CollectiveCompletionSchedule(tile_count=1, scheduling_mode=scheduling_mode),
    )
    completion_plan = dataflow.program.event_plans[-1]
    if completion_plan.memory_scope is not EventMemoryScope.SYSTEM:
        raise ValueError("placement-changing collective completion must have system visibility")
    return JaxCollectiveExecutionPlan(
        completion=completion,
        dataflow=dataflow,
        axis_name=axis_name,
        device_id_by_axis_index=device_id_by_axis_index,
        axis_index_groups=axis_index_groups,
        completion_visibility=completion_plan.visibility,
    )


def execute_jax_collective_completion(
    plan: JaxCollectiveExecutionPlan,
    partial_value: jax.Array,
) -> jax.Array:
    """Complete a placement-partial value with JAX-owned transport and AD."""
    expected_dtype = _JAX_DTYPES[plan.completion.fold.dtype]
    if np.dtype(partial_value.dtype) != np.dtype(expected_dtype):
        raise ValueError(f"collective partial must have dtype {np.dtype(expected_dtype)}, found {partial_value.dtype}")
    groups = tuple(tuple(group) for group in plan.axis_index_groups)
    with jax.named_scope("shuttle_collective_completion"):
        match plan.completion.fold.reduction:
            case CollectiveReduction.SUM:
                return jax.lax.psum(partial_value, plan.axis_name, axis_index_groups=groups)
            case CollectiveReduction.MAXIMUM:
                return jax.lax.pmax(partial_value, plan.axis_name, axis_index_groups=groups)
            case CollectiveReduction.MINIMUM:
                return jax.lax.pmin(partial_value, plan.axis_name, axis_index_groups=groups)
            case CollectiveReduction.PRODUCT:
                raise AssertionError("product reduction is rejected while constructing the execution plan")


_JAX_DTYPES = {
    DType.BF16: jnp.bfloat16,
    DType.FP32: jnp.float32,
    DType.FP64: jnp.float64,
    DType.INT32: jnp.int32,
}
