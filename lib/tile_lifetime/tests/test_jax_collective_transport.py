# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

from tile_lifetime.collective_transport import (
    CollectiveCompletionPlan,
    CollectiveFoldPlan,
    CollectiveReduction,
    PlacementTransitionPlan,
    ReplicaGroupDomain,
)
from tile_lifetime.event_dataflow import EventMemoryScope, EventSchedulingMode
from tile_lifetime.ir import DType
from tile_lifetime.jax_collective_transport import (
    build_jax_collective_execution_plan,
    execute_jax_collective_completion,
)
from tile_lifetime.plan import NumericalPolicy


def _completion(
    *,
    reduction: CollectiveReduction = CollectiveReduction.SUM,
    groups: tuple[tuple[int, ...], ...] = ((0,),),
) -> CollectiveCompletionPlan:
    return CollectiveCompletionPlan(
        shape="f32[1,4]",
        fold=CollectiveFoldPlan(
            reduction=reduction,
            dtype=DType.FP32,
            numerical_policy=NumericalPolicy.ALLOW_ROUNDING_REORDER,
        ),
        transport=PlacementTransitionPlan(
            source_value="partial",
            destination_value="complete",
            replica_domain=ReplicaGroupDomain(groups=groups, use_global_device_ids=True),
            channel_id=7,
        ),
    )


def test_jax_collective_completion_executes_and_differentiates_through_jax() -> None:
    device = jax.devices("cpu")[0]
    execution = build_jax_collective_execution_plan(
        _completion(groups=((device.id,),)),
        axis_name="collective",
        device_id_by_axis_index=(device.id,),
        scheduling_mode=EventSchedulingMode.DYNAMIC,
    )
    mesh = Mesh(np.asarray((device,)), ("collective",))
    collective = jax.shard_map(
        lambda local: execute_jax_collective_completion(execution, local),
        mesh=mesh,
        in_specs=P("collective"),
        out_specs=P("collective"),
        check_vma=False,
    )
    value = jnp.arange(4, dtype=jnp.float32).reshape(1, 4)

    output = jax.jit(collective)(value)
    gradient = jax.jit(jax.grad(lambda argument: jnp.sum(collective(argument))))(value)
    stablehlo = str(jax.jit(collective).lower(value).compiler_ir(dialect="stablehlo"))

    np.testing.assert_array_equal(output, value)
    np.testing.assert_array_equal(gradient, jnp.ones_like(value))
    assert stablehlo.count("stablehlo.all_reduce") == 1
    assert "stablehlo.custom_call" not in stablehlo
    completion_event = execution.dataflow.program.event_plans[-1]
    assert completion_event.memory_scope is EventMemoryScope.SYSTEM
    assert execution.completion_visibility.release_on_notify
    assert execution.completion_visibility.acquire_before_consumer


def test_jax_collective_completion_maps_global_device_ids_to_axis_indices() -> None:
    execution = build_jax_collective_execution_plan(
        _completion(groups=((7, 9), (3, 1))),
        axis_name="collective",
        device_id_by_axis_index=(7, 3, 9, 1),
    )

    assert execution.axis_index_groups == ((0, 2), (1, 3))
    assert execution.dataflow.program.event_plans[-1].initial_count.as_mapping() == {
        (0, 0): 2,
        (1, 0): 2,
    }


def test_jax_collective_completion_rejects_unimplemented_product_reducer() -> None:
    with pytest.raises(ValueError, match="product reduction"):
        build_jax_collective_execution_plan(
            _completion(reduction=CollectiveReduction.PRODUCT),
            axis_name="collective",
            device_id_by_axis_index=(0,),
        )


def test_jax_collective_completion_rejects_fixed_tree_numerical_contract() -> None:
    completion = _completion()
    completion = replace(
        completion,
        fold=replace(completion.fold, numerical_policy=NumericalPolicy.BITWISE_EXACT),
    )
    with pytest.raises(ValueError, match="physical reduction tree is not fixed"):
        build_jax_collective_execution_plan(
            completion,
            axis_name="collective",
            device_id_by_axis_index=(0,),
        )


def test_jax_collective_completion_rejects_local_replica_id_groups() -> None:
    completion = _completion()
    completion = replace(
        completion,
        transport=replace(
            completion.transport,
            replica_domain=replace(completion.transport.replica_domain, use_global_device_ids=False),
        ),
    )
    with pytest.raises(ValueError, match="global device IDs"):
        build_jax_collective_execution_plan(
            completion,
            axis_name="collective",
            device_id_by_axis_index=(0,),
        )
