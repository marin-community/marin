# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from itertools import pairwise

import numpy as np
import pytest

from tile_lifetime.event_dataflow import (
    EventMemoryScope,
    EventSchedulingMode,
    execute_event_dataflow,
    verify_event_dataflow_program,
)
from tile_lifetime.ir import DType
from tile_lifetime.relation import RelationPlan, build_fixed_capacity_relation_plan, build_relation_plan
from tile_lifetime.relation_transport import (
    EpochResetKind,
    RelationTransportTemplate,
    TilePipelineEdge,
    TilePipelineGraph,
    TilePipelineStage,
    TransportCapacityMode,
    TransportDirection,
    TransportEpochProtocol,
    TransportKernelizationPolicy,
    TransportMechanism,
    TransportPayloadDomain,
    TransportPayloadField,
    TransportRowGranularity,
    bind_transport_epoch,
    derive_relation_tile_pipeline,
    derive_relation_transport_runtime_metadata,
    derive_relation_transport_training_plan,
    derive_transport_field_flow,
    execute_dispatch_field,
    execute_return_field,
    execute_transport_field_flow_reference,
)

_SOURCE_RANK = np.asarray([2, 0, 2, 0, 2, 0], dtype=np.int32)
_SOURCE_LOCAL = np.asarray([0, 0, 1, 1, 2, 2], dtype=np.int32)
_DESTINATION_RANK = np.asarray([0, 2, 2, 0], dtype=np.int32)
_DESTINATION_LOCAL = np.asarray([0, 0, 1, 1], dtype=np.int32)


def _relation(*, route_slots: int = 3, destinations: np.ndarray | None = None) -> RelationPlan:
    if destinations is None:
        destinations = np.asarray(
            [
                [0, 1, 2],
                [3, 1, 0],
                [1, 2, 3],
                [2, 0, 1],
                [0, 3, 2],
                [1, 3, 0],
            ],
            dtype=np.int32,
        )[:, :route_slots]
    weights = np.arange(1, destinations.size + 1, dtype=np.float32).reshape(destinations.shape) / 20
    return build_relation_plan(
        destinations,
        weights,
        destination_rank_by_item=_DESTINATION_RANK,
        destination_local_item_by_item=_DESTINATION_LOCAL,
        padding_quantum=2,
    )


def _matrix(value: int) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(value for _destination in range(4)) for _source in range(4))


def _template(
    *,
    capacity_mode: TransportCapacityMode = TransportCapacityMode.DYNAMIC,
    destination_capacity: tuple[int, ...] = (8, 8, 8, 8),
    coalesced_capacity: tuple[tuple[int, ...], ...] | None = None,
    exact_capacity: tuple[tuple[int, ...], ...] | None = None,
) -> RelationTransportTemplate:
    return RelationTransportTemplate(
        world_rank_count=4,
        source_rank_count=4,
        source_item_capacity_by_rank=(8, 0, 8, 0),
        destination_rank_by_item=tuple(int(value) for value in _DESTINATION_RANK),
        destination_local_item_by_item=tuple(int(value) for value in _DESTINATION_LOCAL),
        destination_row_capacity_by_item=destination_capacity,
        coalesced_capacity_by_rank_pair=_matrix(8) if coalesced_capacity is None else coalesced_capacity,
        exact_edge_capacity_by_rank_pair=_matrix(16) if exact_capacity is None else exact_capacity,
        capacity_mode=capacity_mode,
        tile_rows=2,
        macrobatch_rows=4,
        epoch_protocol=TransportEpochProtocol(phase_count=2, generation_modulus=2),
    )


def _runtime(relation: RelationPlan | None = None, *, template: RelationTransportTemplate | None = None):
    return derive_relation_transport_runtime_metadata(
        _relation() if relation is None else relation,
        template=_template() if template is None else template,
        source_rank_by_item=_SOURCE_RANK,
        source_local_item_by_item=_SOURCE_LOCAL,
    )


def _field(
    name: str,
    domain: TransportPayloadDomain,
    *,
    width: int = 3,
) -> TransportPayloadField:
    return TransportPayloadField(name, domain, (width,), DType.BF16)


def _epoch(epoch: int = 0):
    return bind_transport_epoch(
        TransportEpochProtocol(phase_count=2, generation_modulus=2),
        epoch,
        completed_epochs=frozenset(),
        completed_resets=frozenset(),
    )


def _pipeline_graph() -> TilePipelineGraph:
    names = ("receive_tile", "first_contract", "local_map", "second_contract", "return_ready")
    return TilePipelineGraph(
        stages=tuple(TilePipelineStage(name) for name in names),
        edges=tuple(TilePipelineEdge(source, target) for source, target in pairwise(names)),
        entry_stage="receive_tile",
        exit_stage="return_ready",
        tile_group_split_after="local_map",
    )


def test_runtime_metadata_has_exact_rank_pair_oracle_and_empty_world_ranks() -> None:
    runtime = _runtime()

    assert np.array_equal(runtime.source_item_count_by_rank, np.asarray([3, 0, 3, 0], dtype=np.int32))
    assert runtime.template.world_rank_count == 4
    assert runtime.template.source_rank_count == 4
    assert np.array_equal(runtime.source_item_offset_by_rank, np.asarray([0, 3, 3, 6, 6], dtype=np.int32))
    assert np.array_equal(runtime.destination_item_count_by_rank, np.asarray([2, 0, 2, 0], dtype=np.int32))
    assert np.array_equal(runtime.destination_item_offset_by_rank, np.asarray([0, 2, 2, 4, 4], dtype=np.int32))
    oracle = np.zeros((4, 4), dtype=np.int32)
    for edge in np.flatnonzero(runtime.edge_valid):
        oracle[runtime.edge_source_rank[edge], runtime.edge_destination_rank[edge]] += 1
    assert np.array_equal(runtime.exact_count_by_rank_pair, oracle)
    assert runtime.exact_count_offset_by_rank_pair.shape == (17,)
    assert runtime.exact_capacity_offset_by_rank_pair.shape == (17,)
    assert np.all(runtime.exact_transport_row_to_capacity_slot[:-1] < runtime.exact_transport_row_to_capacity_slot[1:])
    for transport_row, edge in enumerate(runtime.exact_edge_by_transport_row):
        capacity_slot = runtime.exact_transport_row_to_capacity_slot[transport_row]
        rank_pair = np.searchsorted(runtime.exact_capacity_offset_by_rank_pair, capacity_slot, side="right") - 1
        assert rank_pair // 4 == runtime.edge_source_rank[edge]
        assert rank_pair % 4 == runtime.edge_destination_rank[edge]
    assert runtime.source_rank_by_item[0] == 2
    assert runtime.source_local_item_by_item[0] == 0


def test_mixed_payload_domains_use_separate_flows_and_explicit_expand_join() -> None:
    runtime = _runtime()
    epoch = _epoch()
    source_field = _field("source_value", TransportPayloadDomain.SOURCE_ITEM)
    edge_field = _field("edge_value", TransportPayloadDomain.RELATION_EDGE)
    source_flow = derive_transport_field_flow(
        runtime,
        field=source_field,
        direction=TransportDirection.SOURCE_TO_DESTINATION,
        name="source_flow",
        epoch=epoch,
    )
    edge_flow = derive_transport_field_flow(
        runtime,
        field=edge_field,
        direction=TransportDirection.SOURCE_TO_DESTINATION,
        name="edge_flow",
        epoch=epoch,
    )

    assert source_flow.row_granularity is TransportRowGranularity.SOURCE_DESTINATION
    assert source_flow.mechanism_candidates == (
        TransportMechanism.COALESCED_DISPATCH,
        TransportMechanism.ALL_TO_ALL_V,
    )
    assert edge_flow.row_granularity is TransportRowGranularity.RELATION_EDGE
    assert TransportMechanism.COALESCED_DISPATCH not in edge_flow.mechanism_candidates
    assert source_flow.join_tasks.name.endswith("destination_edge_join")
    assert source_flow.readiness.owner_placement == "destination_rank_memory"
    assert source_flow.readiness.owner_rank_by_event == tuple(
        int(value) for value in runtime.destination_row_destination_rank
    )

    source = np.arange(18, dtype=np.float32).reshape(6, 3)
    edges = np.arange(54, dtype=np.float32).reshape(6, 3, 3)
    joined_source = execute_dispatch_field(runtime, source_field, source)
    joined_edges = execute_dispatch_field(runtime, edge_field, edges)
    for row in np.flatnonzero(runtime.destination_row_valid):
        source_item = runtime.destination_row_source_item[row]
        route_slot = runtime.destination_row_route_slot[row]
        assert np.array_equal(joined_source[row], source[source_item])
        assert np.array_equal(joined_edges[row], edges[source_item, route_slot])


def test_capacity_and_topology_contracts_reject_runtime_mutations() -> None:
    relation = _relation()
    exact_capacity = [list(row) for row in _matrix(16)]
    exact_capacity[0][0] = 0
    with pytest.raises(ValueError, match="rank-pair capacity"):
        _runtime(relation, template=_template(exact_capacity=tuple(tuple(row) for row in exact_capacity)))

    with pytest.raises(ValueError, match=r"cover \[0, count\)"):
        derive_relation_transport_runtime_metadata(
            relation,
            template=_template(),
            source_rank_by_item=_SOURCE_RANK,
            source_local_item_by_item=np.asarray([0, 0, 2, 1, 3, 2], dtype=np.int32),
        )

    changed_topology = build_relation_plan(
        np.asarray([[0, 1], [1, 0], [0, 1], [1, 0], [0, 1], [1, 0]], dtype=np.int32),
        np.full((6, 2), 0.5, dtype=np.float32),
        destination_rank_by_item=np.asarray([0, 0, 2, 2], dtype=np.int32),
        destination_local_item_by_item=np.asarray([0, 1, 0, 1], dtype=np.int32),
        padding_quantum=2,
    )
    with pytest.raises(ValueError, match="rank/local coordinates"):
        _runtime(changed_topology)


def test_runtime_route_mutation_changes_counts_without_changing_template_identity() -> None:
    template = _template(destination_capacity=(12, 12, 12, 12))
    primary = _runtime(template=template)
    mutated_destinations = np.tile(np.asarray([[0, 3, 0]], dtype=np.int32), (6, 1))
    mutated = _runtime(_relation(destinations=mutated_destinations), template=template)

    assert primary.template is mutated.template
    assert not np.array_equal(primary.exact_count_by_rank_pair, mutated.exact_count_by_rank_pair)
    assert not np.array_equal(primary.coalesced_count_by_rank_pair, mutated.coalesced_count_by_rank_pair)
    assert np.array_equal(primary.exact_capacity_offset_by_rank_pair, mutated.exact_capacity_offset_by_rank_pair)


def test_fixed_capacity_validates_rows_and_empty_destination_readiness() -> None:
    relation = build_fixed_capacity_relation_plan(
        np.asarray([[0, 1], [0, 1], [1, 0], [0, 1], [1, 0], [0, 1]], dtype=np.int32),
        np.full((6, 2), 0.5, dtype=np.float32),
        destination_rank_by_item=_DESTINATION_RANK,
        destination_local_item_by_item=_DESTINATION_LOCAL,
        destination_capacity=8,
    )
    runtime = _runtime(
        relation,
        template=_template(capacity_mode=TransportCapacityMode.FIXED),
    )
    flow = derive_transport_field_flow(
        runtime,
        field=_field("source_value", TransportPayloadDomain.SOURCE_ITEM),
        direction=TransportDirection.SOURCE_TO_DESTINATION,
        name="fixed",
        epoch=_epoch(),
    )

    counts = tuple(count.value for count in flow.readiness.plan.initial_count.counts)
    expected_ready = tuple(int(row) for row in np.flatnonzero(~runtime.destination_row_valid))
    assert tuple(index for index, count in enumerate(counts) if count == 0) == expected_ready
    assert flow.readiness.runtime_inputs.initially_ready_events == expected_ready
    assert flow.transfer_rows.axes[0].extent == sum(sum(row) for row in runtime.template.coalesced_capacity_by_rank_pair)
    source = np.arange(18, dtype=np.float32).reshape(6, 3)
    joined = execute_dispatch_field(runtime, flow.field, source)
    scheduled_joined = execute_transport_field_flow_reference(runtime, flow, source)
    assert np.array_equal(scheduled_joined, joined)
    for row in np.flatnonzero(runtime.destination_row_valid):
        assert np.array_equal(joined[row], source[runtime.destination_row_source_item[row]])

    return_flow = derive_transport_field_flow(
        runtime,
        field=_field("edge_result", TransportPayloadDomain.RELATION_EDGE),
        direction=TransportDirection.DESTINATION_TO_SOURCE,
        name="fixed_return",
        epoch=_epoch(),
    )
    fixed_pipeline = derive_relation_tile_pipeline(
        runtime,
        dispatch_flows=(flow,),
        return_flows=(return_flow,),
        graph=_pipeline_graph(),
        epoch=_epoch(),
    )
    assert fixed_pipeline.tile_domain.padding_is_masked
    assert any(not active for active in fixed_pipeline.tile_domain.active_by_task)
    fixed_entry = fixed_pipeline.stage_families[0]
    fixed_entry_events = tuple(
        owned for owned in fixed_pipeline.owned_events if owned.plan.trigger_relation.target == fixed_entry
    )
    assert fixed_entry_events
    assert all(count.value > 0 for owned in fixed_entry_events for count in owned.plan.initial_count.counts)
    fixed_state: dict[str, object] = {"active_stage_tasks": 0}
    fixed_actions = {family.name: lambda _coordinate, _state: None for family in fixed_pipeline.program.task_families}

    def execute_masked_tile(coordinate, state) -> None:
        if fixed_pipeline.tile_domain.active_by_task[coordinate[0]]:
            state["active_stage_tasks"] = int(state["active_stage_tasks"]) + 1

    for stage_family in fixed_pipeline.stage_families:
        fixed_actions[stage_family.name] = execute_masked_tile
    execute_event_dataflow(
        fixed_pipeline.program,
        actions=fixed_actions,
        state=fixed_state,
        scheduling_mode=EventSchedulingMode.DYNAMIC,
        generation=flow.epoch.stored_generation,
        random_seed=5,
    )
    assert fixed_state["active_stage_tasks"] == sum(fixed_pipeline.tile_domain.active_by_task) * len(
        fixed_pipeline.stage_families
    )


def test_epoch_protocol_requires_ordered_phase_and_wrap_resets() -> None:
    protocol = TransportEpochProtocol(phase_count=2, generation_modulus=2)
    first = bind_transport_epoch(
        protocol,
        0,
        completed_epochs=frozenset(),
        completed_resets=frozenset(),
    )
    assert first.phase == 0
    assert first.stored_generation == 0
    assert not first.reset_transitions
    second_phase = bind_transport_epoch(
        protocol,
        1,
        completed_epochs=frozenset(),
        completed_resets=frozenset(),
    )

    with pytest.raises(ValueError, match="live phase"):
        bind_transport_epoch(protocol, 2, completed_epochs=frozenset(), completed_resets=frozenset())
    reused = bind_transport_epoch(
        protocol,
        2,
        completed_epochs=frozenset({0}),
        completed_resets=frozenset({(EpochResetKind.PHASE_REUSE, 2)}),
    )
    assert reused.stored_generation == 1
    assert reused.reset_transitions[0].visibility.scope is EventMemoryScope.SYSTEM

    with pytest.raises(ValueError, match="generation-wrap"):
        bind_transport_epoch(
            protocol,
            4,
            completed_epochs=frozenset(range(4)),
            completed_resets=frozenset({(EpochResetKind.PHASE_REUSE, 4)}),
        )
    wrapped = bind_transport_epoch(
        protocol,
        4,
        completed_epochs=frozenset(range(4)),
        completed_resets=frozenset(
            {
                (EpochResetKind.PHASE_REUSE, 4),
                (EpochResetKind.GENERATION_WRAP, 4),
            }
        ),
    )
    assert wrapped.stored_generation == 0
    assert tuple(transition.kind for transition in wrapped.reset_transitions) == (
        EpochResetKind.PHASE_REUSE,
        EpochResetKind.GENERATION_WRAP,
    )
    assert all(transition.task_name.startswith("reset_") for transition in wrapped.reset_transitions)
    assert all(transition.ordered_before_initialization for transition in wrapped.reset_transitions)

    runtime = _runtime()
    source_field = _field("phase_source", TransportPayloadDomain.SOURCE_ITEM)
    result_field = _field("phase_result", TransportPayloadDomain.RELATION_EDGE)

    def field_pair(epoch_binding):
        return (
            derive_transport_field_flow(
                runtime,
                field=source_field,
                direction=TransportDirection.SOURCE_TO_DESTINATION,
                name="phase_dispatch",
                epoch=epoch_binding,
            ),
            derive_transport_field_flow(
                runtime,
                field=result_field,
                direction=TransportDirection.DESTINATION_TO_SOURCE,
                name="phase_return",
                epoch=epoch_binding,
            ),
        )

    dispatch0, return0 = field_pair(first)
    dispatch1, return1 = field_pair(second_phase)
    dispatch2, return2 = field_pair(reused)
    dispatch4, return4 = field_pair(wrapped)
    slots0 = dispatch0.readiness.runtime_inputs.event_storage_slots
    slots1 = dispatch1.readiness.runtime_inputs.event_storage_slots
    assert set(slots0).isdisjoint(slots1)
    assert dispatch0.readiness.storage_namespace != dispatch1.readiness.storage_namespace
    assert dispatch2.readiness.runtime_inputs.event_storage_slots == slots0
    assert dispatch2.readiness.storage_namespace == dispatch0.readiness.storage_namespace
    assert dispatch2.readiness.runtime_inputs.event_generations == (1,) * len(slots0)
    assert dispatch4.readiness.runtime_inputs.event_storage_slots == slots0
    assert dispatch4.readiness.storage_namespace == dispatch0.readiness.storage_namespace
    assert dispatch4.readiness.runtime_inputs.event_generations == (0,) * len(slots0)

    pipelines = tuple(
        derive_relation_tile_pipeline(
            runtime,
            dispatch_flows=(dispatch,),
            return_flows=(returned,),
            graph=_pipeline_graph(),
            epoch=epoch_binding,
        )
        for dispatch, returned, epoch_binding in (
            (dispatch0, return0, first),
            (dispatch1, return1, second_phase),
            (dispatch2, return2, reused),
            (dispatch4, return4, wrapped),
        )
    )
    owned_by_name = tuple({owned.plan.name: owned for owned in pipeline.owned_events} for pipeline in pipelines)
    for name in owned_by_name[0]:
        phase0 = owned_by_name[0][name].runtime_inputs
        phase1 = owned_by_name[1][name].runtime_inputs
        phase2 = owned_by_name[2][name].runtime_inputs
        phase4 = owned_by_name[3][name].runtime_inputs
        assert set(phase0.event_storage_slots).isdisjoint(phase1.event_storage_slots)
        assert owned_by_name[0][name].storage_namespace != owned_by_name[1][name].storage_namespace
        assert phase2.event_storage_slots == phase0.event_storage_slots
        assert owned_by_name[2][name].storage_namespace == owned_by_name[0][name].storage_namespace
        assert phase4.event_storage_slots == phase0.event_storage_slots
        assert owned_by_name[3][name].storage_namespace == owned_by_name[0][name].storage_namespace
        assert phase2.event_generations == (1,) * len(phase2.event_generations)
        assert phase4.event_generations == (0,) * len(phase4.event_generations)


def test_tile_pipeline_keeps_exact_graph_separate_from_kernelization_candidates() -> None:
    relation = _relation()
    epoch = _epoch()
    source_field = _field("source_value", TransportPayloadDomain.SOURCE_ITEM)
    edge_field = _field("edge_attribute", TransportPayloadDomain.RELATION_EDGE)
    result_field = _field("edge_result", TransportPayloadDomain.RELATION_EDGE)
    training = derive_relation_transport_training_plan(
        relation,
        template=_template(),
        source_rank_by_item=_SOURCE_RANK,
        source_local_item_by_item=_SOURCE_LOCAL,
        epoch=epoch,
        primal_dispatch_fields=(source_field, edge_field),
        primal_return_fields=(result_field,),
        cotangent_dispatch_fields=(source_field,),
        cotangent_return_fields=(result_field,),
    )
    pipeline = derive_relation_tile_pipeline(
        training.runtime,
        dispatch_flows=training.primal.dispatch_flows,
        return_flows=training.primal.return_flows,
        graph=_pipeline_graph(),
        epoch=epoch,
    )

    verify_event_dataflow_program(pipeline.program)
    pipeline_execution = execute_event_dataflow(
        pipeline.program,
        actions={family.name: lambda _coordinate, _state: None for family in pipeline.program.task_families},
        state={},
        scheduling_mode=EventSchedulingMode.DYNAMIC,
        generation=epoch.stored_generation,
        random_seed=11,
    )
    assert len(pipeline_execution.executed_tasks) == sum(
        len(family.coordinates) for family in pipeline.program.task_families
    )
    assert pipeline.stage_families[0].axes[0].name == "tile_task"
    assert not pipeline.tile_domain.padding_is_masked
    assert all(pipeline.tile_domain.active_by_task)
    assert all(
        pipeline.tile_domain.task_by_destination_row[row] == -1
        for row in np.flatnonzero(~training.runtime.destination_row_valid)
    )
    entry_readiness = tuple(
        owned for owned in pipeline.owned_events if owned.plan.trigger_relation.target == pipeline.stage_families[0]
    )
    assert entry_readiness
    assert all(count.value > 0 for owned in entry_readiness for count in owned.plan.initial_count.counts)
    assert all(not owned.runtime_inputs.initially_ready_events for owned in entry_readiness)
    compute_readiness = tuple(
        owned for owned in pipeline.owned_events if owned.plan.trigger_relation.target in pipeline.stage_families
    )
    assert all(count.value > 0 for owned in compute_readiness for count in owned.plan.initial_count.counts)
    candidates = {candidate.policy: candidate for candidate in pipeline.kernelization_candidates}
    assert set(candidates) == {
        TransportKernelizationPolicy.NONE,
        TransportKernelizationPolicy.TILE,
        TransportKernelizationPolicy.FULL,
    }
    assert len(candidates[TransportKernelizationPolicy.NONE].groups) > 2
    assert len(candidates[TransportKernelizationPolicy.TILE].groups) == 2
    assert len(candidates[TransportKernelizationPolicy.FULL].groups) == 1
    assert "local_map" in candidates[TransportKernelizationPolicy.TILE].groups[0]
    assert "second_contract" in candidates[TransportKernelizationPolicy.TILE].groups[1]
    fold_readiness = next(
        owned for owned in pipeline.owned_events if owned.plan.trigger_relation.target == pipeline.source_fold
    )
    assert fold_readiness.owner_rank_by_event == tuple(int(value) for value in _SOURCE_RANK)
    assert fold_readiness.owner_placement == "source_compute_workers"

    source = np.arange(18, dtype=np.float32).reshape(6, 3)
    edge_attribute = np.ones((6, 3, 3), dtype=np.float32)
    joined_source = execute_dispatch_field(training.runtime, source_field, source)
    joined_attribute = execute_dispatch_field(training.runtime, edge_field, edge_attribute)
    mutated_attribute = execute_dispatch_field(training.runtime, edge_field, edge_attribute * 3)
    assert not np.array_equal(joined_attribute, mutated_attribute)

    destination_result = joined_source + joined_attribute
    mutated_destination_result = joined_source + mutated_attribute
    returned = execute_return_field(training.runtime, result_field, destination_result)
    mutated_returned = execute_return_field(training.runtime, result_field, mutated_destination_result)
    returned_delta = execute_return_field(
        training.runtime,
        result_field,
        mutated_attribute - joined_attribute,
    )
    assert np.array_equal(mutated_returned - returned, returned_delta)
    direct = np.zeros_like(returned)
    for row in np.flatnonzero(training.runtime.destination_row_valid):
        direct[
            training.runtime.destination_row_source_item[row],
            training.runtime.destination_row_route_slot[row],
        ] = destination_result[row]
    assert np.array_equal(returned, direct)
    folded = np.sum(returned * relation.weight[..., None], axis=1)
    mutated_folded = np.sum(mutated_returned * relation.weight[..., None], axis=1)
    assert not np.array_equal(folded, mutated_folded)
