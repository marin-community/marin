# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exercise Shuttle-derived collective completion on a virtual CPU mesh."""

from __future__ import annotations

import hashlib
import json

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

from tile_lifetime.collective_transport import (
    CollectiveCompletionPlan,
    CollectiveFoldPlan,
    CollectiveReduction,
    PlacementTransitionPlan,
    ReplicaGroupDomain,
)
from tile_lifetime.event_dataflow import EventSchedulingMode
from tile_lifetime.ir import DType
from tile_lifetime.jax_collective_transport import (
    build_jax_collective_execution_plan,
    execute_jax_collective_completion,
)
from tile_lifetime.plan import NumericalPolicy

_AXIS_NAME = "collective"
_FEATURE_COUNT = 8


def _completion(
    reduction: CollectiveReduction,
    groups: tuple[tuple[int, ...], ...],
) -> CollectiveCompletionPlan:
    return CollectiveCompletionPlan(
        shape=f"f32[1,{_FEATURE_COUNT}]",
        fold=CollectiveFoldPlan(
            reduction=reduction,
            dtype=DType.FP32,
            numerical_policy=NumericalPolicy.ALLOW_ROUNDING_REORDER,
        ),
        transport=PlacementTransitionPlan(
            source_value="partial",
            destination_value="complete",
            replica_domain=ReplicaGroupDomain(groups=groups, use_global_device_ids=True),
            channel_id=1,
        ),
    )


def _reference_by_group(
    value: jax.Array,
    groups: tuple[tuple[int, ...], ...],
    reduction: CollectiveReduction,
) -> jax.Array:
    output = jnp.empty_like(value)
    for group in groups:
        group_values = value[jnp.asarray(group)]
        if reduction is CollectiveReduction.SUM:
            reduced = jnp.sum(group_values, axis=0)
        elif reduction is CollectiveReduction.MAXIMUM:
            reduced = jnp.max(group_values, axis=0)
        else:
            raise ValueError(f"unsupported reference reduction {reduction.value}")
        output = output.at[jnp.asarray(group)].set(reduced)
    return output


def main() -> None:
    devices = tuple(jax.devices("cpu"))
    if len(devices) < 2:
        raise RuntimeError("run with XLA_FLAGS=--xla_force_host_platform_device_count=N for N >= 2")
    device_ids = tuple(device.id for device in devices)
    full_group = (device_ids,)
    pair_groups = tuple(tuple(device_ids[index : index + 2]) for index in range(0, len(device_ids), 2))
    if any(len(group) != 2 for group in pair_groups):
        raise RuntimeError("the CPU replay requires an even virtual-device count")

    mesh = Mesh(np.asarray(devices), (_AXIS_NAME,))

    def mapped_collective(
        reduction: CollectiveReduction,
        groups: tuple[tuple[int, ...], ...],
    ):
        execution = build_jax_collective_execution_plan(
            _completion(reduction, groups),
            axis_name=_AXIS_NAME,
            device_id_by_axis_index=device_ids,
            scheduling_mode=EventSchedulingMode.STATIC,
        )
        mapped = jax.shard_map(
            lambda local: execute_jax_collective_completion(execution, local),
            mesh=mesh,
            in_specs=P(_AXIS_NAME),
            out_specs=P(_AXIS_NAME),
            check_vma=False,
        )
        return execution, jax.jit(mapped)

    sum_execution, sum_collective = mapped_collective(CollectiveReduction.SUM, full_group)
    maximum_execution, maximum_collective = mapped_collective(CollectiveReduction.MAXIMUM, pair_groups)
    value = jnp.arange(len(devices) * _FEATURE_COUNT, dtype=jnp.float32).reshape(len(devices), _FEATURE_COUNT)
    sum_output = sum_collective(value).block_until_ready()
    maximum_output = maximum_collective(value).block_until_ready()
    sum_reference = _reference_by_group(value, full_group, CollectiveReduction.SUM)
    maximum_reference = _reference_by_group(value, pair_groups, CollectiveReduction.MAXIMUM)

    def loss(input_value: jax.Array) -> jax.Array:
        output = sum_collective(input_value)
        return jnp.sum(output * output, dtype=jnp.float32) / 2

    def reference_loss(input_value: jax.Array) -> jax.Array:
        output = _reference_by_group(input_value, full_group, CollectiveReduction.SUM)
        return jnp.sum(output * output, dtype=jnp.float32) / 2

    gradient = jax.jit(jax.grad(loss))(value).block_until_ready()
    reference_gradient = jax.grad(reference_loss)(value)
    repeated = sum_collective(value).block_until_ready()
    forward_hlo = str(sum_collective.lower(value).compiler_ir(dialect="stablehlo"))
    gradient_hlo = str(jax.jit(jax.grad(loss)).lower(value).compiler_ir(dialect="stablehlo"))
    result = {
        "device_count": len(devices),
        "sum_axis_index_groups": sum_execution.axis_index_groups,
        "maximum_axis_index_groups": maximum_execution.axis_index_groups,
        "sum_event_initial_count": [
            {"coordinate": coordinate, "count": count}
            for coordinate, count in sum_execution.dataflow.program.event_plans[-1].initial_count.as_mapping().items()
        ],
        "completion_visibility": {
            "scope": sum_execution.dataflow.program.event_plans[-1].memory_scope.value,
            "release_on_notify": sum_execution.completion_visibility.release_on_notify,
            "acquire_before_consumer": sum_execution.completion_visibility.acquire_before_consumer,
        },
        "sum_max_abs_error": float(jnp.max(jnp.abs(sum_output - sum_reference))),
        "maximum_max_abs_error": float(jnp.max(jnp.abs(maximum_output - maximum_reference))),
        "gradient_max_abs_error": float(jnp.max(jnp.abs(gradient - reference_gradient))),
        "deterministic": bool(jnp.array_equal(sum_output, repeated)),
        "output_sha256": hashlib.sha256(np.asarray(sum_output).tobytes()).hexdigest(),
        "forward_all_reduce_count": forward_hlo.count("stablehlo.all_reduce"),
        "gradient_all_reduce_count": gradient_hlo.count("stablehlo.all_reduce"),
        "custom_call_count": forward_hlo.count("stablehlo.custom_call"),
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
