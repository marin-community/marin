# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Generic payload transport plans derived from runtime sparse relations.

This module owns the metadata and readiness boundary around a placement
transition.  It intentionally does not own any Map or Fold applied to the
payload.  In particular, returned relation edges retain their source-local
slot identity so a separately generated, numerically explicit Fold can merge
them after transport.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np

from tile_lifetime.event_dataflow import (
    EventDataflowProgram,
    EventGenerationPolicy,
    EventMemoryScope,
    EventSchedulingMode,
    EventTensorPlan,
    EventTensorRuntimeInputs,
    MemoryVisibility,
    TaskAxis,
    TaskDependence,
    TaskFamily,
    TaskRelation,
    derive_event_tensor_plan,
    event_tensor_runtime_inputs,
    phased_event_storage_binding,
    verify_event_dataflow_program,
)
from tile_lifetime.ir import DType
from tile_lifetime.relation import RelationPlan


class TransportDirection(StrEnum):
    """Direction of one payload-only placement transition."""

    SOURCE_TO_DESTINATION = "source_to_destination"
    DESTINATION_TO_SOURCE = "destination_to_source"


class TransportCapacityMode(StrEnum):
    """Whether physical destination storage follows runtime occupancy."""

    DYNAMIC = "dynamic"
    FIXED = "fixed"


class TransportMechanism(StrEnum):
    """Backend candidates whose interfaces move payload without reducing it."""

    SYMMETRIC_MEMORY_PULL = "symmetric_memory_pull"
    SYMMETRIC_MEMORY_PUSH = "symmetric_memory_push"
    ALL_TO_ALL = "all_to_all"
    COALESCED_DISPATCH_AND_EXPAND = "coalesced_dispatch_and_expand"


class TransportRowGranularity(StrEnum):
    """Logical identity retained by physical transport rows."""

    RELATION_EDGE = "relation_edge"
    SOURCE_DESTINATION = "source_destination"


class TransportPayloadDomain(StrEnum):
    """Logical leading domain of one payload independently of its storage."""

    SOURCE_ITEM = "source_item"
    RELATION_EDGE = "relation_edge"


@dataclass(frozen=True)
class TransportPayloadField:
    """One opaque payload field carried by a transport leg."""

    name: str
    logical_domain: TransportPayloadDomain
    trailing_shape: tuple[int, ...]
    dtype: DType

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("transport payload field name must be non-empty")
        if any(extent <= 0 for extent in self.trailing_shape):
            raise ValueError("transport payload trailing extents must be positive")


@dataclass(frozen=True)
class TransportOverlapHooks:
    """Schedule hooks exposed without selecting workers or synchronization."""

    producer_granularity: TransportRowGranularity
    consumer_granularity: str
    notify_per_completed_row: bool
    async_progress_eligible: bool


@dataclass(frozen=True)
class RelationTransportMetadataPlane:
    """Transport/index metadata kept separate from tensor payload values."""

    source_rank_count: int
    destination_rank_count: int
    capacity_mode: TransportCapacityMode
    source_rank_by_item: np.ndarray
    edge_source_item: np.ndarray
    edge_route_slot: np.ndarray
    edge_source_rank: np.ndarray
    edge_destination_rank: np.ndarray
    edge_destination_item: np.ndarray
    edge_valid: np.ndarray
    edge_to_destination_row: np.ndarray
    destination_row_to_edge: np.ndarray
    destination_row_valid: np.ndarray
    destination_group_count: np.ndarray
    destination_group_capacity: np.ndarray
    destination_group_offset: np.ndarray
    logical_edge_count_by_rank_pair: np.ndarray
    destination_physical_count_by_rank: np.ndarray
    destination_physical_offset_by_rank: np.ndarray
    coalesced_source_item: np.ndarray
    coalesced_source_rank: np.ndarray
    coalesced_destination_rank: np.ndarray
    edge_to_coalesced_row: np.ndarray

    @property
    def logical_edge_count(self) -> int:
        """Number of valid semantic relation edges."""
        return int(np.count_nonzero(self.edge_valid))

    @property
    def physical_destination_row_count(self) -> int:
        """Number of destination rows including bounded padding."""
        return int(self.destination_row_to_edge.shape[0])


@dataclass(frozen=True)
class RelationTransportLegPlan:
    """One payload-only relation transport plus derived readiness."""

    name: str
    direction: TransportDirection
    payload_fields: tuple[TransportPayloadField, ...]
    metadata: RelationTransportMetadataPlane
    row_granularity: TransportRowGranularity
    mechanism_candidates: tuple[TransportMechanism, ...]
    overlap: TransportOverlapHooks
    dataflow: EventDataflowProgram
    readiness: EventTensorPlan
    runtime_inputs: EventTensorRuntimeInputs
    generation: int


@dataclass(frozen=True)
class RelationTransportRoundTripPlan:
    """Primal or cotangent dispatch and exact-edge return transport."""

    dispatch: RelationTransportLegPlan
    returned_edges: RelationTransportLegPlan


@dataclass(frozen=True)
class RelationTransportTrainingPlan:
    """Transport ABI for a primal round trip and its JAX-derived cotangent."""

    metadata: RelationTransportMetadataPlane
    primal: RelationTransportRoundTripPlan
    cotangent: RelationTransportRoundTripPlan


def derive_relation_transport_metadata(
    relation: RelationPlan,
    *,
    source_rank_by_item: np.ndarray,
    capacity_mode: TransportCapacityMode,
) -> RelationTransportMetadataPlane:
    """Derive exact-edge and optional coalesced transport metadata."""
    if not isinstance(capacity_mode, TransportCapacityMode):
        raise TypeError("transport capacity mode must be explicit")
    source_rank_by_item = np.asarray(source_rank_by_item)
    if source_rank_by_item.shape != (relation.source_item_count,):
        raise ValueError(
            "source-rank ownership must have one entry per source item, " f"got {source_rank_by_item.shape}"
        )
    if not np.issubdtype(source_rank_by_item.dtype, np.integer):
        raise TypeError("source-rank ownership must use an integer dtype")
    if np.any(source_rank_by_item < 0):
        raise ValueError("source-rank ownership must be non-negative")
    source_rank_by_item = source_rank_by_item.astype(np.int32, copy=False)
    source_rank_count = int(np.max(source_rank_by_item, initial=-1)) + 1
    if source_rank_count == 0:
        raise ValueError("source-rank ownership must not be empty")

    edge_valid = relation.edge_valid.reshape(-1).copy()
    edge_source_rank = source_rank_by_item[relation.source_item]
    rank_pair_counts = np.zeros((source_rank_count, relation.destination_rank_count), dtype=np.int32)
    np.add.at(
        rank_pair_counts,
        (edge_source_rank[edge_valid], relation.destination_rank[edge_valid]),
        1,
    )
    physical_count_by_rank = np.bincount(
        relation.row_destination_rank,
        minlength=relation.destination_rank_count,
    ).astype(np.int32, copy=False)
    physical_offset_by_rank = np.concatenate(
        (np.zeros(1, dtype=np.int32), np.cumsum(physical_count_by_rank, dtype=np.int32))
    )
    coalesced_source_rank = source_rank_by_item[relation.exchange_source_item]
    return RelationTransportMetadataPlane(
        source_rank_count=source_rank_count,
        destination_rank_count=relation.destination_rank_count,
        capacity_mode=capacity_mode,
        source_rank_by_item=source_rank_by_item.copy(),
        edge_source_item=relation.source_item.copy(),
        edge_route_slot=relation.route_slot.copy(),
        edge_source_rank=edge_source_rank.copy(),
        edge_destination_rank=relation.destination_rank.copy(),
        edge_destination_item=relation.destination_item.copy(),
        edge_valid=edge_valid,
        edge_to_destination_row=relation.route_to_destination_row.copy(),
        destination_row_to_edge=relation.destination_row_to_route.copy(),
        destination_row_valid=relation.row_valid.copy(),
        destination_group_count=relation.group_count.copy(),
        destination_group_capacity=relation.group_padded_count.copy(),
        destination_group_offset=relation.group_offset.copy(),
        logical_edge_count_by_rank_pair=rank_pair_counts,
        destination_physical_count_by_rank=physical_count_by_rank,
        destination_physical_offset_by_rank=physical_offset_by_rank,
        coalesced_source_item=relation.exchange_source_item.copy(),
        coalesced_source_rank=coalesced_source_rank.copy(),
        coalesced_destination_rank=relation.exchange_destination_rank.copy(),
        edge_to_coalesced_row=relation.route_to_exchange_row.copy(),
    )


def derive_relation_transport_round_trip(
    relation: RelationPlan,
    *,
    metadata: RelationTransportMetadataPlane,
    dispatched_fields: tuple[TransportPayloadField, ...],
    returned_fields: tuple[TransportPayloadField, ...],
    name: str,
    generation_base: int,
) -> RelationTransportRoundTripPlan:
    """Build exact-edge dispatch/return legs without introducing a Fold."""
    dispatch = _derive_transport_leg(
        relation,
        metadata=metadata,
        direction=TransportDirection.SOURCE_TO_DESTINATION,
        payload_fields=dispatched_fields,
        name=f"{name}.dispatch",
        generation=generation_base,
    )
    returned_edges = _derive_transport_leg(
        relation,
        metadata=metadata,
        direction=TransportDirection.DESTINATION_TO_SOURCE,
        payload_fields=returned_fields,
        name=f"{name}.return",
        generation=generation_base + 1,
    )
    return RelationTransportRoundTripPlan(dispatch, returned_edges)


def derive_relation_transport_training_plan(
    relation: RelationPlan,
    *,
    source_rank_by_item: np.ndarray,
    capacity_mode: TransportCapacityMode,
    primal_input_fields: tuple[TransportPayloadField, ...],
    primal_return_fields: tuple[TransportPayloadField, ...],
    cotangent_input_fields: tuple[TransportPayloadField, ...],
    cotangent_return_fields: tuple[TransportPayloadField, ...],
) -> RelationTransportTrainingPlan:
    """Derive a transport-only primal/adjoint ABI while leaving AD to JAX."""
    metadata = derive_relation_transport_metadata(
        relation,
        source_rank_by_item=source_rank_by_item,
        capacity_mode=capacity_mode,
    )
    primal = derive_relation_transport_round_trip(
        relation,
        metadata=metadata,
        dispatched_fields=primal_input_fields,
        returned_fields=primal_return_fields,
        name="primal",
        generation_base=0,
    )
    cotangent = derive_relation_transport_round_trip(
        relation,
        metadata=metadata,
        dispatched_fields=cotangent_input_fields,
        returned_fields=cotangent_return_fields,
        name="cotangent",
        generation_base=2,
    )
    return RelationTransportTrainingPlan(metadata, primal, cotangent)


def execute_relation_dispatch(relation: RelationPlan, payload: np.ndarray) -> np.ndarray:
    """Reference payload-only source-to-destination permutation."""
    return relation.dispatch(payload)


def execute_relation_edge_dispatch(relation: RelationPlan, payload: np.ndarray) -> np.ndarray:
    """Reference permutation of source-local edge payload to destination rows."""
    if not isinstance(payload, np.ndarray):
        raise TypeError("relation-edge payload must be a NumPy array")
    expected_prefix = (relation.source_item_count, relation.route_slots)
    if payload.shape[:2] != expected_prefix:
        raise ValueError(f"relation-edge payload must begin with shape {expected_prefix}, got {payload.shape}")
    flat = payload.reshape(relation.source_item_count * relation.route_slots, *payload.shape[2:])
    output = np.zeros((relation.destination_row_count, *payload.shape[2:]), dtype=payload.dtype)
    valid_rows = np.flatnonzero(relation.row_valid)
    output[valid_rows] = flat[relation.destination_row_to_route[valid_rows]]
    return output


def execute_relation_return(
    relation: RelationPlan,
    destination_payload: np.ndarray,
    *,
    fill_value: int | float = 0,
) -> np.ndarray:
    """Reference exact inverse preserving source item and relation slot."""
    return relation.inverse_dispatch(destination_payload, fill_value=fill_value)


def _derive_transport_leg(
    relation: RelationPlan,
    *,
    metadata: RelationTransportMetadataPlane,
    direction: TransportDirection,
    payload_fields: tuple[TransportPayloadField, ...],
    name: str,
    generation: int,
) -> RelationTransportLegPlan:
    if not payload_fields:
        raise ValueError("a transport leg must carry at least one payload field")
    if generation < 0:
        raise ValueError("transport generation must be non-negative")
    allowed_domains = (
        {TransportPayloadDomain.SOURCE_ITEM, TransportPayloadDomain.RELATION_EDGE}
        if direction is TransportDirection.SOURCE_TO_DESTINATION
        else {TransportPayloadDomain.RELATION_EDGE}
    )
    invalid_fields = tuple(field.name for field in payload_fields if field.logical_domain not in allowed_domains)
    if invalid_fields:
        raise ValueError(f"payload fields have incompatible logical domains: {invalid_fields}")
    visibility = MemoryVisibility(EventMemoryScope.SYSTEM)
    physical_row_count = relation.destination_row_count
    transfer = TaskFamily(
        f"{name}.row_completion",
        (TaskAxis("physical_row", physical_row_count),),
        placement="transport_workers",
    )
    if direction is TransportDirection.SOURCE_TO_DESTINATION:
        consumer = TaskFamily(
            f"{name}.destination_segment",
            (TaskAxis("destination_segment", relation.destination_count),),
            placement="destination_compute_workers",
        )
        pairs = tuple(
            ((row,), (int(relation.row_destination_item[row]),))
            for row in range(physical_row_count)
            if relation.row_valid[row]
        )
        mechanisms = (
            TransportMechanism.SYMMETRIC_MEMORY_PULL,
            TransportMechanism.ALL_TO_ALL,
            TransportMechanism.COALESCED_DISPATCH_AND_EXPAND,
        )
        consumer_granularity = "destination_segment"
    else:
        consumer = TaskFamily(
            f"{name}.source_item",
            (TaskAxis("source_item", relation.source_item_count),),
            placement="source_compute_workers",
        )
        pairs = tuple(
            ((row,), (int(relation.row_source_item[row]),))
            for row in range(physical_row_count)
            if relation.row_valid[row]
        )
        mechanisms = (TransportMechanism.SYMMETRIC_MEMORY_PUSH, TransportMechanism.ALL_TO_ALL)
        consumer_granularity = "source_item"
    dependence = TaskDependence(TaskRelation.from_pairs(transfer, consumer, pairs), visibility)
    readiness = derive_event_tensor_plan(
        dependence,
        name=f"{name}.readiness",
        memory_scope=EventMemoryScope.SYSTEM,
        generation_policy=EventGenerationPolicy.PHASED,
        scheduling_mode=EventSchedulingMode.DYNAMIC,
    )
    storage = phased_event_storage_binding(
        readiness,
        slot=lambda coordinate: consumer.coordinates.index(coordinate),
        generation=lambda _coordinate: generation,
    )
    runtime_inputs = event_tensor_runtime_inputs(readiness, storage_binding=storage)
    dataflow = EventDataflowProgram((transfer, consumer), (dependence,), (readiness,))
    verify_event_dataflow_program(dataflow)
    return RelationTransportLegPlan(
        name=name,
        direction=direction,
        payload_fields=payload_fields,
        metadata=metadata,
        row_granularity=TransportRowGranularity.RELATION_EDGE,
        mechanism_candidates=mechanisms,
        overlap=TransportOverlapHooks(
            producer_granularity=TransportRowGranularity.RELATION_EDGE,
            consumer_granularity=consumer_granularity,
            notify_per_completed_row=True,
            async_progress_eligible=True,
        ),
        dataflow=dataflow,
        readiness=readiness,
        runtime_inputs=runtime_inputs,
        generation=generation,
    )
