# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from tile_lifetime.event_dataflow import (
    EventMemoryScope,
    EventSchedulingMode,
    execute_event_dataflow,
    verify_event_dataflow_program,
)
from tile_lifetime.ir import DType
from tile_lifetime.relation import RelationPlan, build_fixed_capacity_relation_plan, build_relation_plan
from tile_lifetime.relation_transport import (
    TransportCapacityMode,
    TransportMechanism,
    TransportPayloadDomain,
    TransportPayloadField,
    derive_relation_transport_metadata,
    derive_relation_transport_training_plan,
    execute_relation_dispatch,
    execute_relation_edge_dispatch,
    execute_relation_return,
)


def _dynamic_relation(*, route_slots: int = 3, weights: np.ndarray | None = None) -> RelationPlan:
    destinations = np.asarray(
        [
            [0, 2, 3],
            [1, 2, 0],
            [2, 2, 1],
            [3, 0, 2],
            [0, 3, 1],
            [2, 1, 3],
        ],
        dtype=np.int32,
    )[:, :route_slots]
    valid = np.asarray(
        [
            [True, True, False],
            [True, True, True],
            [True, False, True],
            [True, True, True],
            [True, False, True],
            [True, True, True],
        ],
        dtype=np.bool_,
    )[:, :route_slots]
    if weights is None:
        weights = np.arange(1, destinations.size + 1, dtype=np.float32).reshape(destinations.shape) / 10
    return build_relation_plan(
        destinations,
        weights,
        edge_valid=valid,
        destination_rank_by_item=np.asarray([0, 0, 1, 1], dtype=np.int32),
        destination_local_item_by_item=np.asarray([0, 1, 0, 1], dtype=np.int32),
        padding_quantum=4,
    )


def _field(
    name: str,
    width: int = 8,
    *,
    logical_domain: TransportPayloadDomain = TransportPayloadDomain.SOURCE_ITEM,
) -> TransportPayloadField:
    return TransportPayloadField(name, logical_domain, (width,), DType.BF16)


def test_runtime_relation_derives_rank_metadata_and_exact_readiness() -> None:
    relation = _dynamic_relation()
    source_ranks = np.asarray([0, 0, 0, 1, 1, 1], dtype=np.int32)

    plan = derive_relation_transport_training_plan(
        relation,
        source_rank_by_item=source_ranks,
        capacity_mode=TransportCapacityMode.DYNAMIC,
        primal_input_fields=(
            _field("input"),
            TransportPayloadField("edge_attribute", TransportPayloadDomain.RELATION_EDGE, (), DType.FP32),
        ),
        primal_return_fields=(_field("result", logical_domain=TransportPayloadDomain.RELATION_EDGE),),
        cotangent_input_fields=(_field("output_cotangent"),),
        cotangent_return_fields=(
            _field("input_cotangent", logical_domain=TransportPayloadDomain.RELATION_EDGE),
            TransportPayloadField("edge_cotangent", TransportPayloadDomain.RELATION_EDGE, (), DType.FP32),
        ),
    )

    assert plan.metadata.logical_edge_count == relation.route_count
    assert np.array_equal(plan.metadata.destination_group_count, relation.group_count)
    assert np.array_equal(
        plan.metadata.destination_physical_offset_by_rank,
        np.asarray([0, 8, 20], dtype=np.int32),
    )
    assert int(plan.metadata.logical_edge_count_by_rank_pair.sum()) == relation.route_count
    dispatch_counts = tuple(count.value for count in plan.primal.dispatch.readiness.initial_count.counts)
    return_counts = tuple(count.value for count in plan.primal.returned_edges.readiness.initial_count.counts)
    assert dispatch_counts == tuple(int(value) for value in relation.group_count)
    assert return_counts == tuple(int(value) for value in relation.edge_valid.sum(axis=1))
    assert plan.primal.dispatch.readiness.memory_scope is EventMemoryScope.SYSTEM
    assert plan.primal.dispatch.readiness.visibility.release_on_notify
    assert plan.primal.dispatch.readiness.visibility.acquire_before_consumer
    assert plan.primal.dispatch.runtime_inputs.event_generations == (0, 0, 0, 0)
    assert plan.primal.returned_edges.runtime_inputs.event_generations == (1,) * relation.source_item_count
    assert plan.cotangent.dispatch.runtime_inputs.event_generations == (2, 2, 2, 2)
    assert plan.cotangent.returned_edges.runtime_inputs.event_generations == (3,) * relation.source_item_count
    assert TransportMechanism.COALESCED_DISPATCH_AND_EXPAND in plan.primal.dispatch.mechanism_candidates
    assert TransportMechanism.COALESCED_DISPATCH_AND_EXPAND not in plan.primal.returned_edges.mechanism_candidates
    for leg in (
        plan.primal.dispatch,
        plan.primal.returned_edges,
        plan.cotangent.dispatch,
        plan.cotangent.returned_edges,
    ):
        verify_event_dataflow_program(leg.dataflow)


def test_payload_transport_preserves_edges_and_leaves_weighted_fold_external() -> None:
    relation = _dynamic_relation()
    source = np.arange(relation.source_item_count * 4, dtype=np.float32).reshape(relation.source_item_count, 4)

    dispatched = execute_relation_dispatch(relation, source)
    source_edge_attributes = relation.weight[..., None]
    dispatched_edge_attributes = execute_relation_edge_dispatch(relation, source_edge_attributes)
    destination_result = dispatched * 2 + np.arange(relation.destination_row_count, dtype=np.float32)[:, None]
    returned = execute_relation_return(relation, destination_result)

    expected_returned = np.zeros_like(returned)
    for row in np.flatnonzero(relation.row_valid):
        expected_returned[relation.row_source_item[row], relation.row_route_slot[row]] = destination_result[row]
    assert np.array_equal(returned, expected_returned)
    assert np.array_equal(dispatched_edge_attributes[relation.row_valid, 0], relation.row_weight[relation.row_valid])

    folded = np.zeros((relation.source_item_count, 4), dtype=np.float32)
    for source_item in range(relation.source_item_count):
        for slot in range(relation.route_slots):
            if relation.edge_valid[source_item, slot]:
                folded[source_item] += returned[source_item, slot] * relation.weight[source_item, slot]
    assert np.array_equal(folded, relation.weighted_merge(destination_result))

    mutated_weights = np.flip(relation.weight, axis=1).copy()
    mutated = _dynamic_relation(weights=mutated_weights)
    mutated_dispatched = execute_relation_dispatch(mutated, source)
    mutated_metadata = derive_relation_transport_metadata(
        mutated,
        source_rank_by_item=np.asarray([0, 0, 0, 1, 1, 1], dtype=np.int32),
        capacity_mode=TransportCapacityMode.DYNAMIC,
    )
    original_metadata = derive_relation_transport_metadata(
        relation,
        source_rank_by_item=np.asarray([0, 0, 0, 1, 1, 1], dtype=np.int32),
        capacity_mode=TransportCapacityMode.DYNAMIC,
    )
    assert np.array_equal(mutated_dispatched, dispatched)
    assert np.array_equal(mutated_metadata.edge_to_destination_row, original_metadata.edge_to_destination_row)
    assert not np.array_equal(mutated.weighted_merge(destination_result), folded)


def test_fixed_capacity_keeps_physical_shape_while_runtime_counts_include_empty_segments() -> None:
    relation = build_fixed_capacity_relation_plan(
        np.asarray([[0, 1], [0, 1], [1, 0]], dtype=np.int32),
        np.full((3, 2), 0.5, dtype=np.float32),
        destination_rank_by_item=np.asarray([0, 0, 1, 1], dtype=np.int32),
        destination_local_item_by_item=np.asarray([0, 1, 0, 1], dtype=np.int32),
        destination_capacity=3,
    )
    plan = derive_relation_transport_training_plan(
        relation,
        source_rank_by_item=np.asarray([0, 0, 1], dtype=np.int32),
        capacity_mode=TransportCapacityMode.FIXED,
        primal_input_fields=(_field("input"),),
        primal_return_fields=(_field("result", logical_domain=TransportPayloadDomain.RELATION_EDGE),),
        cotangent_input_fields=(_field("output_cotangent"),),
        cotangent_return_fields=(_field("input_cotangent", logical_domain=TransportPayloadDomain.RELATION_EDGE),),
    )

    assert plan.metadata.physical_destination_row_count == 12
    assert np.array_equal(plan.metadata.destination_group_capacity, np.asarray([3, 3, 3, 3], dtype=np.int32))
    assert tuple(count.value for count in plan.primal.dispatch.readiness.initial_count.counts) == (3, 3, 0, 0)
    assert plan.primal.dispatch.runtime_inputs.initially_ready_events == (2, 3)
    actions = {family.name: lambda _coordinate, _state: None for family in plan.primal.dispatch.dataflow.task_families}
    execution = execute_event_dataflow(
        plan.primal.dispatch.dataflow,
        actions=actions,
        state={},
        scheduling_mode=EventSchedulingMode.DYNAMIC,
        generation=plan.primal.dispatch.generation,
        random_seed=7,
    )
    assert len(execution.executed_tasks) == 16


def test_cotangent_round_trip_is_payload_only_and_preserves_arbitrary_route_slots() -> None:
    relation = _dynamic_relation(route_slots=2)
    plan = derive_relation_transport_training_plan(
        relation,
        source_rank_by_item=np.asarray([0, 0, 0, 1, 1, 1], dtype=np.int32),
        capacity_mode=TransportCapacityMode.DYNAMIC,
        primal_input_fields=(_field("input", 3),),
        primal_return_fields=(_field("result", 3, logical_domain=TransportPayloadDomain.RELATION_EDGE),),
        cotangent_input_fields=(_field("output_cotangent", 3),),
        cotangent_return_fields=(
            _field("input_cotangent", 3, logical_domain=TransportPayloadDomain.RELATION_EDGE),
            TransportPayloadField("edge_cotangent", TransportPayloadDomain.RELATION_EDGE, (), DType.FP32),
        ),
    )
    output_cotangent = np.arange(18, dtype=np.float32).reshape(6, 3)
    dispatched_cotangent = execute_relation_dispatch(relation, output_cotangent)
    edge_input_cotangent = dispatched_cotangent * 0.25
    edge_attribute_cotangent = dispatched_cotangent.sum(axis=1)

    returned_input = execute_relation_return(relation, edge_input_cotangent)
    returned_attribute = execute_relation_return(relation, edge_attribute_cotangent[:, None])[..., 0]

    assert returned_input.shape == (6, 2, 3)
    assert returned_attribute.shape == (6, 2)
    assert tuple(count.value for count in plan.cotangent.returned_edges.readiness.initial_count.counts) == tuple(
        int(value) for value in relation.edge_valid.sum(axis=1)
    )
    for source_item in range(relation.source_item_count):
        for slot in range(relation.route_slots):
            if not relation.edge_valid[source_item, slot]:
                assert np.array_equal(returned_input[source_item, slot], np.zeros(3, dtype=np.float32))
                assert returned_attribute[source_item, slot] == 0
