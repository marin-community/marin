# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Payload-only placement transitions derived from sparse relations.

Static templates bound topology and storage. Runtime metadata carries route
occupancy, counts, offsets, and exact inverse identities as device operands.
Map, Contract, Fold, and automatic differentiation remain outside transport.
"""

from __future__ import annotations

from collections.abc import Callable, MutableMapping, Sequence
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
    execute_event_dataflow,
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
    """Whether physical row extents are invocation-sized or template-sized."""

    DYNAMIC = "dynamic"
    FIXED = "fixed"


class TransportMechanism(StrEnum):
    """Backend mechanisms that move payload without applying a Fold."""

    SYMMETRIC_MEMORY_PULL = "symmetric_memory_pull"
    SYMMETRIC_MEMORY_PUSH = "symmetric_memory_push"
    ALL_TO_ALL_V = "all_to_all_v"
    COALESCED_DISPATCH = "coalesced_dispatch"


class TransportRowGranularity(StrEnum):
    """Logical identity represented by one transport row."""

    RELATION_EDGE = "relation_edge"
    SOURCE_DESTINATION = "source_destination"


class TransportPayloadDomain(StrEnum):
    """Logical leading domain of a payload field."""

    SOURCE_ITEM = "source_item"
    RELATION_EDGE = "relation_edge"


class TransportKernelizationPolicy(StrEnum):
    """Optional fusion/materialization choices over an exact task graph."""

    NONE = "none"
    TILE = "tile"
    FULL = "full"


class EpochResetKind(StrEnum):
    """Ordered reset transitions required before event storage reuse."""

    PHASE_REUSE = "phase_reuse"
    GENERATION_WRAP = "generation_wrap"


@dataclass(frozen=True)
class TransportPayloadField:
    """One typed payload field carried without semantic transformation."""

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
class TransportEpochProtocol:
    """Bounded cyclic storage protocol for repeated invocations."""

    phase_count: int
    generation_modulus: int

    def __post_init__(self) -> None:
        if self.phase_count < 2:
            raise ValueError("transport event storage requires at least two phases")
        if self.generation_modulus < 2:
            raise ValueError("transport event generations require modulus at least two")


@dataclass(frozen=True)
class EpochResetTransition:
    """System-visible ordered task required before counter initialization."""

    kind: EpochResetKind
    source_epoch: int
    target_epoch: int
    task_name: str
    visibility: MemoryVisibility
    ordered_before_initialization: bool = True


@dataclass(frozen=True)
class TransportEpochBinding:
    """Physical phase/generation selected for one legal invocation epoch."""

    epoch: int
    phase: int
    absolute_generation: int
    stored_generation: int
    reset_transitions: tuple[EpochResetTransition, ...]


@dataclass(frozen=True)
class RelationTransportTemplate:
    """Compile-time topology, capacities, and tiling independent of routing."""

    world_rank_count: int
    source_rank_count: int
    source_item_capacity_by_rank: tuple[int, ...]
    destination_rank_by_item: tuple[int, ...]
    destination_local_item_by_item: tuple[int, ...]
    destination_row_capacity_by_item: tuple[int, ...]
    coalesced_capacity_by_rank_pair: tuple[tuple[int, ...], ...]
    exact_edge_capacity_by_rank_pair: tuple[tuple[int, ...], ...]
    capacity_mode: TransportCapacityMode
    tile_rows: int
    macrobatch_rows: int
    epoch_protocol: TransportEpochProtocol

    def __post_init__(self) -> None:
        if self.world_rank_count <= 0:
            raise ValueError("transport world must contain at least one rank")
        if self.source_rank_count <= 0 or self.source_rank_count > self.world_rank_count:
            raise ValueError("source rank count must be within the declared transport world")
        _validate_rank_vector(
            self.source_item_capacity_by_rank,
            self.source_rank_count,
            "source-item capacity",
        )
        _validate_ownership(
            np.asarray(self.destination_rank_by_item),
            np.asarray(self.destination_local_item_by_item),
            item_count=len(self.destination_rank_by_item),
            world_rank_count=self.world_rank_count,
            name="template destination",
        )
        _validate_rank_pair_matrix(
            self.coalesced_capacity_by_rank_pair,
            self.world_rank_count,
            "coalesced capacity",
        )
        _validate_rank_pair_matrix(
            self.exact_edge_capacity_by_rank_pair,
            self.world_rank_count,
            "exact-edge capacity",
        )
        if len(self.destination_row_capacity_by_item) != len(self.destination_rank_by_item):
            raise ValueError("destination-row capacity must cover every destination item")
        if any(capacity <= 0 for capacity in self.destination_row_capacity_by_item):
            raise ValueError("destination-row capacities must be positive")
        if self.tile_rows <= 0:
            raise ValueError("transport tile rows must be positive")
        if self.macrobatch_rows < self.tile_rows or self.macrobatch_rows % self.tile_rows:
            raise ValueError("transport macrobatch rows must be a positive multiple of tile rows")

    @property
    def destination_item_count_by_rank(self) -> tuple[int, ...]:
        """Static destination ownership counts including trailing empty ranks."""
        counts = np.bincount(
            np.asarray(self.destination_rank_by_item),
            minlength=self.world_rank_count,
        )
        return tuple(int(value) for value in counts)


@dataclass(frozen=True)
class RelationTransportRuntimeMetadata:
    """Invocation-specific device metadata derived from one RelationPlan."""

    template: RelationTransportTemplate
    source_item_count: int
    route_slots: int
    destination_item_count: int
    source_rank_by_item: np.ndarray
    source_local_item_by_item: np.ndarray
    source_item_count_by_rank: np.ndarray
    source_item_offset_by_rank: np.ndarray
    destination_rank_by_item: np.ndarray
    destination_local_item_by_item: np.ndarray
    destination_item_count_by_rank: np.ndarray
    destination_item_offset_by_rank: np.ndarray
    edge_source_item: np.ndarray
    edge_route_slot: np.ndarray
    edge_source_rank: np.ndarray
    edge_destination_rank: np.ndarray
    edge_destination_item: np.ndarray
    edge_valid: np.ndarray
    edge_to_destination_row: np.ndarray
    destination_row_to_edge: np.ndarray
    destination_row_valid: np.ndarray
    destination_row_source_item: np.ndarray
    destination_row_route_slot: np.ndarray
    destination_row_destination_item: np.ndarray
    destination_row_destination_rank: np.ndarray
    destination_group_count_by_item: np.ndarray
    destination_group_capacity_by_item: np.ndarray
    destination_group_offset_by_item: np.ndarray
    exact_edge_by_transport_row: np.ndarray
    edge_to_exact_transport_row: np.ndarray
    exact_count_by_rank_pair: np.ndarray
    exact_count_offset_by_rank_pair: np.ndarray
    exact_capacity_offset_by_rank_pair: np.ndarray
    exact_transport_row_to_capacity_slot: np.ndarray
    coalesced_source_item: np.ndarray
    coalesced_source_rank: np.ndarray
    coalesced_destination_rank: np.ndarray
    edge_to_coalesced_transport_row: np.ndarray
    coalesced_count_by_rank_pair: np.ndarray
    coalesced_count_offset_by_rank_pair: np.ndarray
    coalesced_capacity_offset_by_rank_pair: np.ndarray
    coalesced_transport_row_to_capacity_slot: np.ndarray

    @property
    def logical_edge_count(self) -> int:
        """Number of valid relation edges in this invocation."""
        return int(self.exact_edge_by_transport_row.shape[0])

    @property
    def destination_row_count(self) -> int:
        """Number of runtime destination rows including group padding."""
        return int(self.destination_row_to_edge.shape[0])


@dataclass(frozen=True)
class OwnedEventTensorPlan:
    """EventTensor plan with explicit owner rank and physical placement."""

    plan: EventTensorPlan
    runtime_inputs: EventTensorRuntimeInputs
    owner_rank_by_event: tuple[int, ...]
    owner_placement: str
    storage_namespace: str

    def __post_init__(self) -> None:
        if len(self.owner_rank_by_event) != len(self.plan.domain.coordinates):
            raise ValueError("event ownership must cover every event coordinate")
        if any(rank < 0 for rank in self.owner_rank_by_event):
            raise ValueError("event owner ranks must be non-negative")
        if not self.owner_placement:
            raise ValueError("event owner placement must be explicit")
        if not self.storage_namespace:
            raise ValueError("event storage namespace must be explicit")


@dataclass(frozen=True)
class TransportFieldFlowPlan:
    """One field-specific transport and explicit destination/source join."""

    name: str
    field: TransportPayloadField
    direction: TransportDirection
    row_granularity: TransportRowGranularity
    mechanism_candidates: tuple[TransportMechanism, ...]
    transfer_rows: TaskFamily
    join_tasks: TaskFamily
    dataflow: EventDataflowProgram
    readiness: OwnedEventTensorPlan
    epoch: TransportEpochBinding


@dataclass(frozen=True)
class TilePipelineStage:
    """One workload-independent tile stage in an exact dataflow graph."""

    name: str

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("tile pipeline stage name must be non-empty")


@dataclass(frozen=True)
class TilePipelineEdge:
    """One exact pointwise tile dependence between stage families."""

    source: str
    target: str


@dataclass(frozen=True)
class TilePipelineGraph:
    """Semantic stage dependencies, independent of kernel grouping."""

    stages: tuple[TilePipelineStage, ...]
    edges: tuple[TilePipelineEdge, ...]
    entry_stage: str
    exit_stage: str
    tile_group_split_after: str

    def __post_init__(self) -> None:
        names = tuple(stage.name for stage in self.stages)
        if not names or len(names) != len(set(names)):
            raise ValueError("tile pipeline stage names must be non-empty and unique")
        named = set(names)
        if {self.entry_stage, self.exit_stage, self.tile_group_split_after} - named:
            raise ValueError("pipeline entry, exit, and tile split must name declared stages")
        if any(edge.source not in named or edge.target not in named for edge in self.edges):
            raise ValueError("tile pipeline edge references an undeclared stage")
        position = {name: index for index, name in enumerate(names)}
        if any(position[edge.source] >= position[edge.target] for edge in self.edges):
            raise ValueError("tile pipeline stages must be declared in dependency order")
        if any(edge.target == self.entry_stage or edge.source == self.exit_stage for edge in self.edges):
            raise ValueError("tile pipeline entry/exit stages disagree with the dependency graph")


@dataclass(frozen=True)
class KernelGroupCandidate:
    """One kernelization of an unchanged exact task-dependence graph."""

    policy: TransportKernelizationPolicy
    groups: tuple[tuple[str, ...], ...]


@dataclass(frozen=True)
class RelationTilePipelinePlan:
    """Tile/macrobatch dataflow and independent kernelization candidates."""

    program: EventDataflowProgram
    owned_events: tuple[OwnedEventTensorPlan, ...]
    stage_families: tuple[TaskFamily, ...]
    source_fold: TaskFamily
    tile_domain: RelationTileTaskDomain
    kernelization_candidates: tuple[KernelGroupCandidate, ...]


@dataclass(frozen=True)
class RelationTileTaskDomain:
    """Runtime-valid dynamic tiles or explicitly masked fixed-capacity tiles."""

    logical_coordinate_by_task: tuple[tuple[int, int, int], ...]
    active_by_task: tuple[bool, ...]
    task_by_destination_row: tuple[int, ...]
    padding_is_masked: bool

    def __post_init__(self) -> None:
        if len(self.logical_coordinate_by_task) != len(self.active_by_task):
            raise ValueError("every tile task must have one logical coordinate and active predicate")
        if any(task < -1 or task >= len(self.active_by_task) for task in self.task_by_destination_row):
            raise ValueError("destination-row tile mapping is outside the tile task domain")
        represented_tasks = {task for task in self.task_by_destination_row if task >= 0}
        if represented_tasks != set(range(len(self.active_by_task))):
            raise ValueError("every tile task must represent at least one destination row")
        if not self.padding_is_masked and not all(self.active_by_task):
            raise ValueError("inactive tile tasks require an explicit padding mask")


@dataclass(frozen=True)
class RelationTransportInvocationPlan:
    """Field flows for one primal or cotangent invocation."""

    dispatch_flows: tuple[TransportFieldFlowPlan, ...]
    return_flows: tuple[TransportFieldFlowPlan, ...]


@dataclass(frozen=True)
class RelationTransportTrainingPlan:
    """Transport-only ABI for a JAX-owned primal and cotangent program."""

    template: RelationTransportTemplate
    runtime: RelationTransportRuntimeMetadata
    epoch: TransportEpochBinding
    primal: RelationTransportInvocationPlan
    cotangent: RelationTransportInvocationPlan


def bind_transport_epoch(
    protocol: TransportEpochProtocol,
    epoch: int,
    *,
    completed_epochs: frozenset[int],
    completed_resets: frozenset[tuple[EpochResetKind, int]],
) -> TransportEpochBinding:
    """Bind cyclic event storage after verifying ordered reset transitions."""
    if epoch < 0:
        raise ValueError("transport invocation epoch must be non-negative")
    visibility = MemoryVisibility(EventMemoryScope.SYSTEM)
    resets: list[EpochResetTransition] = []
    if epoch >= protocol.phase_count:
        predecessor = epoch - protocol.phase_count
        if predecessor not in completed_epochs:
            raise ValueError(f"epoch {epoch} reuses a live phase from epoch {predecessor}")
        if (EpochResetKind.PHASE_REUSE, epoch) not in completed_resets:
            raise ValueError(f"epoch {epoch} requires a completed phase-reset task")
        resets.append(
            EpochResetTransition(
                EpochResetKind.PHASE_REUSE,
                predecessor,
                epoch,
                "reset_reused_event_phase",
                visibility,
            )
        )
    cycle_length = protocol.phase_count * protocol.generation_modulus
    if epoch > 0 and epoch % cycle_length == 0:
        prior_cycle = range(epoch - cycle_length, epoch)
        if not set(prior_cycle).issubset(completed_epochs):
            raise ValueError(f"epoch {epoch} cannot wrap generation while the prior cycle is live")
        if (EpochResetKind.GENERATION_WRAP, epoch) not in completed_resets:
            raise ValueError(f"epoch {epoch} requires a completed generation-wrap reset task")
        resets.append(
            EpochResetTransition(
                EpochResetKind.GENERATION_WRAP,
                epoch - 1,
                epoch,
                "reset_wrapped_event_generations",
                visibility,
            )
        )
    absolute_generation = epoch // protocol.phase_count
    return TransportEpochBinding(
        epoch=epoch,
        phase=epoch % protocol.phase_count,
        absolute_generation=absolute_generation,
        stored_generation=absolute_generation % protocol.generation_modulus,
        reset_transitions=tuple(resets),
    )


def derive_relation_transport_runtime_metadata(
    relation: RelationPlan,
    *,
    template: RelationTransportTemplate,
    source_rank_by_item: np.ndarray,
    source_local_item_by_item: np.ndarray,
) -> RelationTransportRuntimeMetadata:
    """Derive and capacity-check invocation metadata from a RelationPlan."""
    source_rank_by_item = np.asarray(source_rank_by_item)
    source_local_item_by_item = np.asarray(source_local_item_by_item)
    source_counts, source_offsets = _validate_ownership(
        source_rank_by_item,
        source_local_item_by_item,
        item_count=relation.source_item_count,
        world_rank_count=template.source_rank_count,
        name="source",
    )
    if np.any(source_counts > np.asarray(template.source_item_capacity_by_rank)):
        raise ValueError("runtime source-item count exceeds the transport template capacity")

    destination_rank_by_item = np.empty(relation.destination_count, dtype=np.int32)
    destination_local_item_by_item = np.empty(relation.destination_count, dtype=np.int32)
    destination_rank_by_item[relation.group_destination_item] = relation.group_destination_rank
    destination_local_item_by_item[relation.group_destination_item] = relation.group_destination_local_item
    destination_counts, destination_offsets = _validate_ownership(
        destination_rank_by_item,
        destination_local_item_by_item,
        item_count=relation.destination_count,
        world_rank_count=template.world_rank_count,
        name="destination",
    )
    if tuple(int(value) for value in destination_counts) != template.destination_item_count_by_rank:
        raise ValueError("runtime destination ownership disagrees with the transport template")
    if not np.array_equal(destination_rank_by_item, np.asarray(template.destination_rank_by_item)) or not np.array_equal(
        destination_local_item_by_item,
        np.asarray(template.destination_local_item_by_item),
    ):
        raise ValueError("runtime destination rank/local coordinates disagree with the transport template")

    group_count_by_item = np.empty(relation.destination_count, dtype=np.int32)
    group_capacity_by_item = np.empty(relation.destination_count, dtype=np.int32)
    group_offset_by_item = np.empty(relation.destination_count, dtype=np.int32)
    group_count_by_item[relation.group_destination_item] = relation.group_count
    group_capacity_by_item[relation.group_destination_item] = relation.group_padded_count
    group_offset_by_item[relation.group_destination_item] = relation.group_offset
    row_destination_item = relation.row_destination_item.copy()
    for group_index, destination_item in enumerate(relation.group_destination_item):
        offset = int(relation.group_offset[group_index])
        capacity = int(relation.group_padded_count[group_index])
        row_destination_item[offset : offset + capacity] = destination_item
    template_group_capacity = np.asarray(template.destination_row_capacity_by_item, dtype=np.int32)
    if template.capacity_mode is TransportCapacityMode.FIXED:
        if not np.array_equal(group_capacity_by_item, template_group_capacity):
            raise ValueError("fixed transport rows must equal the declared destination capacities")
    elif np.any(group_capacity_by_item > template_group_capacity):
        raise ValueError("runtime destination rows exceed the transport template capacity")

    edge_valid = relation.edge_valid.reshape(-1).copy()
    valid_edges = np.flatnonzero(edge_valid)
    edge_source_rank = source_rank_by_item[relation.source_item].astype(np.int32, copy=False)
    exact_edges = _rank_pair_sorted_edges(
        valid_edges,
        source_rank=edge_source_rank,
        destination_rank=relation.destination_rank,
        source_local=source_local_item_by_item[relation.source_item],
        destination_local=relation.destination_local_item,
        route_slot=relation.route_slot,
    )
    edge_to_exact = np.full(edge_valid.shape[0], -1, dtype=np.int32)
    edge_to_exact[exact_edges] = np.arange(exact_edges.shape[0], dtype=np.int32)
    exact_counts = _rank_pair_counts(
        edge_source_rank[exact_edges],
        relation.destination_rank[exact_edges],
        template.world_rank_count,
    )
    exact_capacities = np.asarray(template.exact_edge_capacity_by_rank_pair, dtype=np.int32)
    if np.any(exact_counts > exact_capacities):
        raise ValueError("runtime exact-edge traffic exceeds a rank-pair capacity")
    exact_count_offsets = _flat_offsets(exact_counts)
    exact_capacity_offsets = _flat_offsets(exact_capacities)
    exact_capacity_slots = _capacity_slots(exact_counts, exact_capacity_offsets)

    coalesced_items, coalesced_destination_ranks, edge_to_coalesced = _coalesced_rows(
        relation,
        valid_edges=valid_edges,
        source_rank_by_item=source_rank_by_item,
        source_local_item_by_item=source_local_item_by_item,
    )
    coalesced_source_ranks = source_rank_by_item[coalesced_items].astype(np.int32, copy=False)
    coalesced_counts = _rank_pair_counts(
        coalesced_source_ranks,
        coalesced_destination_ranks,
        template.world_rank_count,
    )
    coalesced_capacities = np.asarray(template.coalesced_capacity_by_rank_pair, dtype=np.int32)
    if np.any(coalesced_counts > coalesced_capacities):
        raise ValueError("runtime coalesced traffic exceeds a rank-pair capacity")
    coalesced_count_offsets = _flat_offsets(coalesced_counts)
    coalesced_capacity_offsets = _flat_offsets(coalesced_capacities)
    coalesced_capacity_slots = _capacity_slots(coalesced_counts, coalesced_capacity_offsets)

    return RelationTransportRuntimeMetadata(
        template=template,
        source_item_count=relation.source_item_count,
        route_slots=relation.route_slots,
        destination_item_count=relation.destination_count,
        source_rank_by_item=source_rank_by_item.astype(np.int32, copy=True),
        source_local_item_by_item=source_local_item_by_item.astype(np.int32, copy=True),
        source_item_count_by_rank=source_counts,
        source_item_offset_by_rank=source_offsets,
        destination_rank_by_item=destination_rank_by_item,
        destination_local_item_by_item=destination_local_item_by_item,
        destination_item_count_by_rank=destination_counts,
        destination_item_offset_by_rank=destination_offsets,
        edge_source_item=relation.source_item.copy(),
        edge_route_slot=relation.route_slot.copy(),
        edge_source_rank=edge_source_rank.copy(),
        edge_destination_rank=relation.destination_rank.copy(),
        edge_destination_item=relation.destination_item.copy(),
        edge_valid=edge_valid,
        edge_to_destination_row=relation.route_to_destination_row.copy(),
        destination_row_to_edge=relation.destination_row_to_route.copy(),
        destination_row_valid=relation.row_valid.copy(),
        destination_row_source_item=relation.row_source_item.copy(),
        destination_row_route_slot=relation.row_route_slot.copy(),
        destination_row_destination_item=row_destination_item,
        destination_row_destination_rank=relation.row_destination_rank.copy(),
        destination_group_count_by_item=group_count_by_item,
        destination_group_capacity_by_item=group_capacity_by_item,
        destination_group_offset_by_item=group_offset_by_item,
        exact_edge_by_transport_row=exact_edges,
        edge_to_exact_transport_row=edge_to_exact,
        exact_count_by_rank_pair=exact_counts,
        exact_count_offset_by_rank_pair=exact_count_offsets,
        exact_capacity_offset_by_rank_pair=exact_capacity_offsets,
        exact_transport_row_to_capacity_slot=exact_capacity_slots,
        coalesced_source_item=coalesced_items,
        coalesced_source_rank=coalesced_source_ranks,
        coalesced_destination_rank=coalesced_destination_ranks,
        edge_to_coalesced_transport_row=edge_to_coalesced,
        coalesced_count_by_rank_pair=coalesced_counts,
        coalesced_count_offset_by_rank_pair=coalesced_count_offsets,
        coalesced_capacity_offset_by_rank_pair=coalesced_capacity_offsets,
        coalesced_transport_row_to_capacity_slot=coalesced_capacity_slots,
    )


def derive_transport_field_flow(
    runtime: RelationTransportRuntimeMetadata,
    *,
    field: TransportPayloadField,
    direction: TransportDirection,
    name: str,
    epoch: TransportEpochBinding,
) -> TransportFieldFlowPlan:
    """Derive a field-specific transport and explicit expand/join task."""
    if (
        direction is TransportDirection.DESTINATION_TO_SOURCE
        and field.logical_domain is not TransportPayloadDomain.RELATION_EDGE
    ):
        raise ValueError("return transport requires exact relation-edge payload identity")
    if (
        direction is TransportDirection.SOURCE_TO_DESTINATION
        and field.logical_domain is TransportPayloadDomain.SOURCE_ITEM
    ):
        granularity = TransportRowGranularity.SOURCE_DESTINATION
        compact_row_count = runtime.coalesced_source_item.shape[0]
        capacity_offsets = runtime.coalesced_capacity_offset_by_rank_pair
        compact_to_capacity = runtime.coalesced_transport_row_to_capacity_slot
        mechanisms = (TransportMechanism.COALESCED_DISPATCH, TransportMechanism.ALL_TO_ALL_V)
    else:
        granularity = TransportRowGranularity.RELATION_EDGE
        compact_row_count = runtime.exact_edge_by_transport_row.shape[0]
        capacity_offsets = runtime.exact_capacity_offset_by_rank_pair
        compact_to_capacity = runtime.exact_transport_row_to_capacity_slot
        mechanisms = (
            (TransportMechanism.SYMMETRIC_MEMORY_PULL, TransportMechanism.ALL_TO_ALL_V)
            if direction is TransportDirection.SOURCE_TO_DESTINATION
            else (TransportMechanism.SYMMETRIC_MEMORY_PUSH, TransportMechanism.ALL_TO_ALL_V)
        )
    row_count = (
        int(capacity_offsets[-1]) if runtime.template.capacity_mode is TransportCapacityMode.FIXED else compact_row_count
    )
    transfer_rows = TaskFamily(
        f"{name}.transport_row",
        (TaskAxis("transport_row", row_count),),
        placement="transport_workers",
    )
    if direction is TransportDirection.SOURCE_TO_DESTINATION:
        join_tasks = TaskFamily(
            f"{name}.destination_edge_join",
            (TaskAxis("destination_row", runtime.destination_row_count),),
            placement="destination_memory",
        )
        pairs = []
        owner_ranks = []
        for destination_row in range(runtime.destination_row_count):
            edge = int(runtime.destination_row_to_edge[destination_row])
            owner_ranks.append(int(runtime.destination_row_destination_rank[destination_row]))
            if edge < 0:
                continue
            transport_row = (
                runtime.edge_to_coalesced_transport_row[edge]
                if granularity is TransportRowGranularity.SOURCE_DESTINATION
                else runtime.edge_to_exact_transport_row[edge]
            )
            if runtime.template.capacity_mode is TransportCapacityMode.FIXED:
                transport_row = compact_to_capacity[transport_row]
            pairs.append(((int(transport_row),), (destination_row,)))
        owner_placement = "destination_rank_memory"
    else:
        join_tasks = TaskFamily(
            f"{name}.source_edge_join",
            (TaskAxis("source_item", runtime.source_item_count), TaskAxis("route_slot", runtime.route_slots)),
            placement="source_memory",
        )
        pairs = tuple(
            (
                (
                    (
                        int(runtime.exact_transport_row_to_capacity_slot[transport_row])
                        if runtime.template.capacity_mode is TransportCapacityMode.FIXED
                        else transport_row
                    ),
                ),
                (
                    int(runtime.edge_source_item[edge]),
                    int(runtime.edge_route_slot[edge]),
                ),
            )
            for transport_row, edge in enumerate(runtime.exact_edge_by_transport_row)
        )
        owner_ranks = [
            int(runtime.source_rank_by_item[source_item])
            for source_item in range(runtime.source_item_count)
            for _route_slot in range(runtime.route_slots)
        ]
        owner_placement = "source_rank_memory"
    visibility = MemoryVisibility(EventMemoryScope.SYSTEM)
    dependence = TaskDependence(TaskRelation.from_pairs(transfer_rows, join_tasks, tuple(pairs)), visibility)
    plan = derive_event_tensor_plan(
        dependence,
        name=f"{name}.join_readiness",
        memory_scope=EventMemoryScope.SYSTEM,
        generation_policy=EventGenerationPolicy.PHASED,
        scheduling_mode=EventSchedulingMode.DYNAMIC,
    )
    binding = _phase_strided_event_storage_binding(plan, epoch)
    runtime_inputs = event_tensor_runtime_inputs(plan, storage_binding=binding)
    dataflow = EventDataflowProgram((transfer_rows, join_tasks), (dependence,), (plan,))
    verify_event_dataflow_program(dataflow)
    owned = OwnedEventTensorPlan(
        plan,
        runtime_inputs,
        tuple(owner_ranks),
        owner_placement,
        f"{plan.name}.phase{epoch.phase}",
    )
    return TransportFieldFlowPlan(
        name=name,
        field=field,
        direction=direction,
        row_granularity=granularity,
        mechanism_candidates=mechanisms,
        transfer_rows=transfer_rows,
        join_tasks=join_tasks,
        dataflow=dataflow,
        readiness=owned,
        epoch=epoch,
    )


def derive_relation_transport_training_plan(
    relation: RelationPlan,
    *,
    template: RelationTransportTemplate,
    source_rank_by_item: np.ndarray,
    source_local_item_by_item: np.ndarray,
    epoch: TransportEpochBinding,
    primal_dispatch_fields: tuple[TransportPayloadField, ...],
    primal_return_fields: tuple[TransportPayloadField, ...],
    cotangent_dispatch_fields: tuple[TransportPayloadField, ...],
    cotangent_return_fields: tuple[TransportPayloadField, ...],
) -> RelationTransportTrainingPlan:
    """Derive a transport-only ABI around a JAX-owned AD program."""
    runtime = derive_relation_transport_runtime_metadata(
        relation,
        template=template,
        source_rank_by_item=source_rank_by_item,
        source_local_item_by_item=source_local_item_by_item,
    )
    primal = _derive_invocation(
        runtime,
        dispatch_fields=primal_dispatch_fields,
        return_fields=primal_return_fields,
        name="primal",
        epoch=epoch,
    )
    cotangent = _derive_invocation(
        runtime,
        dispatch_fields=cotangent_dispatch_fields,
        return_fields=cotangent_return_fields,
        name="cotangent",
        epoch=epoch,
    )
    return RelationTransportTrainingPlan(template, runtime, epoch, primal, cotangent)


def derive_relation_tile_pipeline(
    runtime: RelationTransportRuntimeMetadata,
    *,
    dispatch_flows: tuple[TransportFieldFlowPlan, ...],
    return_flows: tuple[TransportFieldFlowPlan, ...],
    graph: TilePipelineGraph,
    epoch: TransportEpochBinding,
) -> RelationTilePipelinePlan:
    """Instantiate tile/macrobatch tasks while retaining exact dependencies."""
    if not dispatch_flows or not return_flows:
        raise ValueError("a relation tile pipeline requires dispatch and return field flows")
    if any(flow.direction is not TransportDirection.SOURCE_TO_DESTINATION for flow in dispatch_flows):
        raise ValueError("pipeline dispatch flows must move toward destination work")
    if any(flow.direction is not TransportDirection.DESTINATION_TO_SOURCE for flow in return_flows):
        raise ValueError("pipeline return flows must preserve exact source-edge identity")
    tile_domain = _derive_relation_tile_task_domain(runtime)
    tile_axes = (TaskAxis("tile_task", len(tile_domain.logical_coordinate_by_task)),)
    stage_family_by_name = {
        stage.name: TaskFamily(stage.name, tile_axes, placement="destination_compute_workers") for stage in graph.stages
    }
    source_fold = TaskFamily(
        "source_ordered_fold",
        (TaskAxis("source_item", runtime.source_item_count),),
        placement="source_compute_workers",
    )
    visibility = MemoryVisibility(EventMemoryScope.SYSTEM)
    dependences: list[TaskDependence] = []
    task_families: list[TaskFamily] = []
    owner_for_target: dict[TaskFamily, Callable[[tuple[int, ...]], int]] = {}

    for flow in (*dispatch_flows, *return_flows):
        task_families.extend((flow.transfer_rows, flow.join_tasks))
        dependences.extend(flow.dataflow.dependences)
        if flow.direction is TransportDirection.SOURCE_TO_DESTINATION:
            owner_for_target[flow.join_tasks] = lambda coordinate, runtime=runtime: int(
                runtime.destination_row_destination_rank[coordinate[0]]
            )
        else:
            owner_for_target[flow.join_tasks] = lambda coordinate, runtime=runtime: int(
                runtime.source_rank_by_item[coordinate[0]]
            )

    entry = stage_family_by_name[graph.entry_stage]
    exit_stage = stage_family_by_name[graph.exit_stage]
    for flow in dispatch_flows:
        pairs = tuple(
            ((row,), (tile_task,)) for row, tile_task in enumerate(tile_domain.task_by_destination_row) if tile_task >= 0
        )
        dependences.append(TaskDependence(TaskRelation.from_pairs(flow.join_tasks, entry, pairs), visibility))

    tile_coordinates = entry.coordinates
    for edge in graph.edges:
        source = stage_family_by_name[edge.source]
        target = stage_family_by_name[edge.target]
        dependences.append(
            TaskDependence(
                TaskRelation.from_pairs(
                    source,
                    target,
                    tuple((coordinate, coordinate) for coordinate in tile_coordinates),
                ),
                visibility,
            )
        )

    for flow in return_flows:
        pairs = tuple(
            (
                (tile_domain.task_by_destination_row[int(runtime.edge_to_destination_row[edge])],),
                (
                    (
                        int(runtime.exact_transport_row_to_capacity_slot[transport_row])
                        if runtime.template.capacity_mode is TransportCapacityMode.FIXED
                        else transport_row
                    ),
                ),
            )
            for transport_row, edge in enumerate(runtime.exact_edge_by_transport_row)
        )
        dependences.append(TaskDependence(TaskRelation.from_pairs(exit_stage, flow.transfer_rows, pairs), visibility))
        fold_pairs = tuple(
            ((source_item, route_slot), (source_item,))
            for source_item in range(runtime.source_item_count)
            for route_slot in range(runtime.route_slots)
        )
        dependences.append(TaskDependence(TaskRelation.from_pairs(flow.join_tasks, source_fold, fold_pairs), visibility))

    stage_families = tuple(stage_family_by_name[stage.name] for stage in graph.stages)
    task_families.extend((*stage_families, source_fold))
    for family in stage_families:
        owner_for_target[family] = lambda coordinate, runtime=runtime, tile_domain=tile_domain: int(
            runtime.destination_rank_by_item[tile_domain.logical_coordinate_by_task[coordinate[0]][0]]
        )
    owner_for_target[source_fold] = lambda coordinate, runtime=runtime: int(runtime.source_rank_by_item[coordinate[0]])
    for flow in return_flows:
        owner_for_target[flow.transfer_rows] = lambda coordinate, runtime=runtime: _exact_transport_row_destination_rank(
            runtime,
            coordinate[0],
        )

    event_plans = tuple(
        derive_event_tensor_plan(
            dependence,
            name=f"{dependence.relation.source.name}_to_{dependence.relation.target.name}",
            memory_scope=EventMemoryScope.SYSTEM,
            generation_policy=EventGenerationPolicy.PHASED,
            scheduling_mode=EventSchedulingMode.DYNAMIC,
        )
        for dependence in dependences
    )
    program = EventDataflowProgram(tuple(task_families), tuple(dependences), event_plans)
    verify_event_dataflow_program(program)
    owned_event_list = []
    for plan in event_plans:
        target = plan.trigger_relation.target
        if not isinstance(target, TaskFamily):
            raise TypeError("a pipeline EventTensor must trigger a TaskFamily")
        owned_event_list.append(_own_event_plan(plan, owner_for_target[target], epoch))
    owned_events = tuple(owned_event_list)
    sequence = tuple(
        (
            *(flow.transfer_rows.name for flow in dispatch_flows),
            *(flow.join_tasks.name for flow in dispatch_flows),
            *(stage.name for stage in graph.stages),
            *(flow.transfer_rows.name for flow in return_flows),
            *(flow.join_tasks.name for flow in return_flows),
            source_fold.name,
        )
    )
    split_index = sequence.index(graph.tile_group_split_after) + 1
    candidates = (
        KernelGroupCandidate(TransportKernelizationPolicy.NONE, tuple((name,) for name in sequence)),
        KernelGroupCandidate(TransportKernelizationPolicy.TILE, (sequence[:split_index], sequence[split_index:])),
        KernelGroupCandidate(TransportKernelizationPolicy.FULL, (sequence,)),
    )
    return RelationTilePipelinePlan(program, owned_events, stage_families, source_fold, tile_domain, candidates)


def execute_dispatch_field(
    runtime: RelationTransportRuntimeMetadata,
    field: TransportPayloadField,
    payload: np.ndarray,
) -> np.ndarray:
    """Execute transport plus explicit destination-edge join on CPU."""
    if field.logical_domain is TransportPayloadDomain.SOURCE_ITEM:
        _validate_payload(payload, (runtime.source_item_count,), field.trailing_shape, field.name)
        transported = _physical_transport_rows(
            payload[runtime.coalesced_source_item],
            runtime.coalesced_transport_row_to_capacity_slot,
            int(runtime.coalesced_capacity_offset_by_rank_pair[-1]),
            runtime.template.capacity_mode,
        )
        row_for_edge = runtime.edge_to_coalesced_transport_row
        compact_to_capacity = runtime.coalesced_transport_row_to_capacity_slot
    else:
        _validate_payload(
            payload,
            (runtime.source_item_count, runtime.route_slots),
            field.trailing_shape,
            field.name,
        )
        flat = payload.reshape(runtime.source_item_count * runtime.route_slots, *payload.shape[2:])
        transported = _physical_transport_rows(
            flat[runtime.exact_edge_by_transport_row],
            runtime.exact_transport_row_to_capacity_slot,
            int(runtime.exact_capacity_offset_by_rank_pair[-1]),
            runtime.template.capacity_mode,
        )
        row_for_edge = runtime.edge_to_exact_transport_row
        compact_to_capacity = runtime.exact_transport_row_to_capacity_slot
    output = np.zeros((runtime.destination_row_count, *field.trailing_shape), dtype=payload.dtype)
    valid_rows = np.flatnonzero(runtime.destination_row_valid)
    edges = runtime.destination_row_to_edge[valid_rows]
    transport_rows = row_for_edge[edges]
    if runtime.template.capacity_mode is TransportCapacityMode.FIXED:
        transport_rows = compact_to_capacity[transport_rows]
    output[valid_rows] = transported[transport_rows]
    return output


def execute_return_field(
    runtime: RelationTransportRuntimeMetadata,
    field: TransportPayloadField,
    destination_payload: np.ndarray,
) -> np.ndarray:
    """Execute exact-edge return and source-slot join on CPU."""
    if field.logical_domain is not TransportPayloadDomain.RELATION_EDGE:
        raise ValueError("returned payload must retain relation-edge identity")
    _validate_payload(
        destination_payload,
        (runtime.destination_row_count,),
        field.trailing_shape,
        field.name,
    )
    compact_rows = destination_payload[runtime.edge_to_destination_row[runtime.exact_edge_by_transport_row]]
    transported = _physical_transport_rows(
        compact_rows,
        runtime.exact_transport_row_to_capacity_slot,
        int(runtime.exact_capacity_offset_by_rank_pair[-1]),
        runtime.template.capacity_mode,
    )
    exact_rows = (
        transported[runtime.exact_transport_row_to_capacity_slot]
        if runtime.template.capacity_mode is TransportCapacityMode.FIXED
        else transported
    )
    output = np.zeros(
        (runtime.source_item_count, runtime.route_slots, *field.trailing_shape),
        dtype=destination_payload.dtype,
    )
    edges = runtime.exact_edge_by_transport_row
    output[runtime.edge_source_item[edges], runtime.edge_route_slot[edges]] = exact_rows
    return output


def execute_transport_field_flow_reference(
    runtime: RelationTransportRuntimeMetadata,
    flow: TransportFieldFlowPlan,
    payload: np.ndarray,
) -> np.ndarray:
    """Execute numerical gather/scatter actions through the EventDataflow interpreter."""
    field = flow.field
    if flow.direction is TransportDirection.SOURCE_TO_DESTINATION:
        leading_shape = (
            (runtime.source_item_count,)
            if field.logical_domain is TransportPayloadDomain.SOURCE_ITEM
            else (runtime.source_item_count, runtime.route_slots)
        )
        output_shape = (runtime.destination_row_count, *field.trailing_shape)
    else:
        leading_shape = (runtime.destination_row_count,)
        output_shape = (runtime.source_item_count, runtime.route_slots, *field.trailing_shape)
    _validate_payload(payload, leading_shape, field.trailing_shape, field.name)

    compact_to_physical = (
        runtime.coalesced_transport_row_to_capacity_slot
        if flow.row_granularity is TransportRowGranularity.SOURCE_DESTINATION
        else runtime.exact_transport_row_to_capacity_slot
    )
    if runtime.template.capacity_mode is TransportCapacityMode.DYNAMIC:
        compact_to_physical = np.arange(compact_to_physical.shape[0], dtype=np.int32)
    physical_to_compact = np.full(flow.transfer_rows.axes[0].extent, -1, dtype=np.int32)
    physical_to_compact[compact_to_physical] = np.arange(compact_to_physical.shape[0], dtype=np.int32)
    transported = np.zeros((flow.transfer_rows.axes[0].extent, *field.trailing_shape), dtype=payload.dtype)
    output = np.zeros(output_shape, dtype=payload.dtype)

    def transfer_action(coordinate: tuple[int, ...], _state: MutableMapping[str, object]) -> None:
        physical_row = coordinate[0]
        compact_row = int(physical_to_compact[physical_row])
        if compact_row < 0:
            return
        if flow.direction is TransportDirection.DESTINATION_TO_SOURCE:
            edge = int(runtime.exact_edge_by_transport_row[compact_row])
            transported[physical_row] = payload[runtime.edge_to_destination_row[edge]]
        elif field.logical_domain is TransportPayloadDomain.SOURCE_ITEM:
            transported[physical_row] = payload[runtime.coalesced_source_item[compact_row]]
        else:
            edge = int(runtime.exact_edge_by_transport_row[compact_row])
            transported[physical_row] = payload[
                runtime.edge_source_item[edge],
                runtime.edge_route_slot[edge],
            ]

    def join_action(coordinate: tuple[int, ...], _state: MutableMapping[str, object]) -> None:
        if flow.direction is TransportDirection.SOURCE_TO_DESTINATION:
            destination_row = coordinate[0]
            edge = int(runtime.destination_row_to_edge[destination_row])
            if edge < 0:
                return
            compact_row = (
                runtime.edge_to_coalesced_transport_row[edge]
                if flow.row_granularity is TransportRowGranularity.SOURCE_DESTINATION
                else runtime.edge_to_exact_transport_row[edge]
            )
            output[destination_row] = transported[compact_to_physical[compact_row]]
            return
        source_item, route_slot = coordinate
        edge = source_item * runtime.route_slots + route_slot
        if not runtime.edge_valid[edge]:
            return
        compact_row = runtime.edge_to_exact_transport_row[edge]
        output[source_item, route_slot] = transported[compact_to_physical[compact_row]]

    execute_event_dataflow(
        flow.dataflow,
        actions={
            flow.transfer_rows.name: transfer_action,
            flow.join_tasks.name: join_action,
        },
        state={},
        scheduling_mode=EventSchedulingMode.DYNAMIC,
        generation=flow.epoch.stored_generation,
        random_seed=17,
    )
    return output


def _derive_invocation(
    runtime: RelationTransportRuntimeMetadata,
    *,
    dispatch_fields: tuple[TransportPayloadField, ...],
    return_fields: tuple[TransportPayloadField, ...],
    name: str,
    epoch: TransportEpochBinding,
) -> RelationTransportInvocationPlan:
    if not dispatch_fields or not return_fields:
        raise ValueError("a transport invocation requires dispatch and return fields")
    dispatch = tuple(
        derive_transport_field_flow(
            runtime,
            field=field,
            direction=TransportDirection.SOURCE_TO_DESTINATION,
            name=f"{name}.dispatch.{field.name}",
            epoch=epoch,
        )
        for field in dispatch_fields
    )
    returned = tuple(
        derive_transport_field_flow(
            runtime,
            field=field,
            direction=TransportDirection.DESTINATION_TO_SOURCE,
            name=f"{name}.return.{field.name}",
            epoch=epoch,
        )
        for field in return_fields
    )
    return RelationTransportInvocationPlan(dispatch, returned)


def _own_event_plan(
    plan: EventTensorPlan,
    owner: Callable[[tuple[int, ...]], int],
    epoch: TransportEpochBinding,
) -> OwnedEventTensorPlan:
    binding = _phase_strided_event_storage_binding(plan, epoch)
    runtime_inputs = event_tensor_runtime_inputs(plan, storage_binding=binding)
    return OwnedEventTensorPlan(
        plan,
        runtime_inputs,
        tuple(owner(coordinate) for coordinate in plan.domain.coordinates),
        plan.trigger_relation.target.placement or "unspecified",
        f"{plan.name}.phase{epoch.phase}",
    )


def _phase_strided_event_storage_binding(
    plan: EventTensorPlan,
    epoch: TransportEpochBinding,
):
    coordinates = plan.domain.coordinates
    coordinate_index = {coordinate: index for index, coordinate in enumerate(coordinates)}
    phase_offset = epoch.phase * len(coordinates)
    return phased_event_storage_binding(
        plan,
        slot=lambda coordinate: phase_offset + coordinate_index[coordinate],
        generation=lambda _coordinate: epoch.stored_generation,
    )


def _derive_relation_tile_task_domain(
    runtime: RelationTransportRuntimeMetadata,
) -> RelationTileTaskDomain:
    task_by_row = np.full(runtime.destination_row_count, -1, dtype=np.int32)
    logical_coordinates: list[tuple[int, int, int]] = []
    active: list[bool] = []
    if runtime.template.capacity_mode is TransportCapacityMode.DYNAMIC:
        task_by_logical_coordinate: dict[tuple[int, int, int], int] = {}
        for destination_row in np.flatnonzero(runtime.destination_row_valid):
            logical_coordinate = _logical_tile_coordinate(runtime, int(destination_row))
            tile_task = task_by_logical_coordinate.setdefault(logical_coordinate, len(task_by_logical_coordinate))
            if tile_task == len(logical_coordinates):
                logical_coordinates.append(logical_coordinate)
                active.append(True)
            task_by_row[destination_row] = tile_task
        return RelationTileTaskDomain(
            tuple(logical_coordinates),
            tuple(active),
            tuple(int(value) for value in task_by_row),
            padding_is_masked=False,
        )

    for destination_item, capacity in enumerate(runtime.template.destination_row_capacity_by_item):
        group_offset = int(runtime.destination_group_offset_by_item[destination_item])
        for tile_start in range(0, capacity, runtime.template.tile_rows):
            logical_coordinate = (
                destination_item,
                tile_start // runtime.template.macrobatch_rows,
                tile_start % runtime.template.macrobatch_rows // runtime.template.tile_rows,
            )
            tile_task = len(logical_coordinates)
            logical_coordinates.append(logical_coordinate)
            tile_stop = min(tile_start + runtime.template.tile_rows, capacity)
            rows = np.arange(group_offset + tile_start, group_offset + tile_stop, dtype=np.int32)
            task_by_row[rows] = tile_task
            active.append(bool(np.any(runtime.destination_row_valid[rows])))
    return RelationTileTaskDomain(
        tuple(logical_coordinates),
        tuple(active),
        tuple(int(value) for value in task_by_row),
        padding_is_masked=True,
    )


def _logical_tile_coordinate(
    runtime: RelationTransportRuntimeMetadata,
    destination_row: int,
) -> tuple[int, int, int]:
    destination_item = int(runtime.destination_row_destination_item[destination_row])
    within_group = destination_row - int(runtime.destination_group_offset_by_item[destination_item])
    macrobatch = within_group // runtime.template.macrobatch_rows
    tile = within_group % runtime.template.macrobatch_rows // runtime.template.tile_rows
    return destination_item, macrobatch, tile


def _exact_transport_row_destination_rank(
    runtime: RelationTransportRuntimeMetadata,
    physical_row: int,
) -> int:
    if runtime.template.capacity_mode is TransportCapacityMode.DYNAMIC:
        edge = runtime.exact_edge_by_transport_row[physical_row]
        return int(runtime.edge_destination_rank[edge])
    pair = int(np.searchsorted(runtime.exact_capacity_offset_by_rank_pair, physical_row, side="right") - 1)
    return pair % runtime.template.world_rank_count


def _physical_transport_rows(
    compact_payload: np.ndarray,
    compact_to_capacity: np.ndarray,
    capacity: int,
    capacity_mode: TransportCapacityMode,
) -> np.ndarray:
    if capacity_mode is TransportCapacityMode.DYNAMIC:
        return compact_payload
    output = np.zeros((capacity, *compact_payload.shape[1:]), dtype=compact_payload.dtype)
    output[compact_to_capacity] = compact_payload
    return output


def _validate_ownership(
    rank_by_item: np.ndarray,
    local_item_by_item: np.ndarray,
    *,
    item_count: int,
    world_rank_count: int,
    name: str,
) -> tuple[np.ndarray, np.ndarray]:
    if rank_by_item.shape != (item_count,) or local_item_by_item.shape != (item_count,):
        raise ValueError(f"{name} ownership must have one rank/local coordinate per item")
    if not np.issubdtype(rank_by_item.dtype, np.integer) or not np.issubdtype(local_item_by_item.dtype, np.integer):
        raise TypeError(f"{name} ownership coordinates must use integer dtypes")
    if np.any(rank_by_item < 0) or np.any(rank_by_item >= world_rank_count):
        raise ValueError(f"{name} rank is outside the declared transport world")
    if np.any(local_item_by_item < 0):
        raise ValueError(f"{name} local item coordinates must be non-negative")
    pairs = np.stack((rank_by_item, local_item_by_item), axis=1)
    if np.unique(pairs, axis=0).shape[0] != item_count:
        raise ValueError(f"{name} rank/local coordinates must be unique")
    counts = np.bincount(rank_by_item, minlength=world_rank_count).astype(np.int32, copy=False)
    for rank, count in enumerate(counts):
        actual = np.sort(local_item_by_item[rank_by_item == rank])
        expected = np.arange(count, dtype=actual.dtype)
        if not np.array_equal(actual, expected):
            raise ValueError(f"{name} local coordinates for rank {rank} must cover [0, count)")
    return counts, np.concatenate((np.zeros(1, dtype=np.int32), np.cumsum(counts, dtype=np.int32)))


def _rank_pair_sorted_edges(
    edges: np.ndarray,
    *,
    source_rank: np.ndarray,
    destination_rank: np.ndarray,
    source_local: np.ndarray,
    destination_local: np.ndarray,
    route_slot: np.ndarray,
) -> np.ndarray:
    if not edges.shape[0]:
        return edges.astype(np.int32, copy=True)
    order = np.lexsort(
        (
            route_slot[edges],
            destination_local[edges],
            source_local[edges],
            destination_rank[edges],
            source_rank[edges],
        )
    )
    return edges[order].astype(np.int32, copy=False)


def _coalesced_rows(
    relation: RelationPlan,
    *,
    valid_edges: np.ndarray,
    source_rank_by_item: np.ndarray,
    source_local_item_by_item: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pairs = {(int(relation.source_item[edge]), int(relation.destination_rank[edge])) for edge in valid_edges}
    ordered = sorted(
        pairs,
        key=lambda pair: (
            int(source_rank_by_item[pair[0]]),
            pair[1],
            int(source_local_item_by_item[pair[0]]),
        ),
    )
    row_by_pair = {pair: row for row, pair in enumerate(ordered)}
    edge_to_row = np.full(relation.source_item.shape[0], -1, dtype=np.int32)
    for edge in valid_edges:
        edge_to_row[edge] = row_by_pair[(int(relation.source_item[edge]), int(relation.destination_rank[edge]))]
    if not ordered:
        return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32), edge_to_row
    return (
        np.asarray([pair[0] for pair in ordered], dtype=np.int32),
        np.asarray([pair[1] for pair in ordered], dtype=np.int32),
        edge_to_row,
    )


def _rank_pair_counts(source_rank: np.ndarray, destination_rank: np.ndarray, world: int) -> np.ndarray:
    counts = np.zeros((world, world), dtype=np.int32)
    np.add.at(counts, (source_rank, destination_rank), 1)
    return counts


def _flat_offsets(counts: np.ndarray) -> np.ndarray:
    flat = counts.reshape(-1)
    return np.concatenate((np.zeros(1, dtype=np.int32), np.cumsum(flat, dtype=np.int32)))


def _capacity_slots(counts: np.ndarray, capacity_offsets: np.ndarray) -> np.ndarray:
    slots = []
    for pair, count in enumerate(counts.reshape(-1)):
        slots.extend(range(int(capacity_offsets[pair]), int(capacity_offsets[pair]) + int(count)))
    return np.asarray(slots, dtype=np.int32)


def _validate_rank_vector(values: Sequence[int], world: int, name: str) -> None:
    if len(values) != world or any(value < 0 for value in values):
        raise ValueError(f"{name} must contain one non-negative value per world rank")


def _validate_rank_pair_matrix(values: Sequence[Sequence[int]], world: int, name: str) -> None:
    if len(values) != world or any(len(row) != world for row in values):
        raise ValueError(f"{name} must be a square world-rank matrix")
    if any(value < 0 for row in values for value in row):
        raise ValueError(f"{name} entries must be non-negative")


def _validate_payload(
    payload: np.ndarray,
    leading_shape: tuple[int, ...],
    trailing_shape: tuple[int, ...],
    name: str,
) -> None:
    if not isinstance(payload, np.ndarray):
        raise TypeError(f"{name} payload must be a NumPy array")
    expected_shape = (*leading_shape, *trailing_shape)
    if payload.shape != expected_shape:
        raise ValueError(f"{name} payload must have shape {expected_shape}, got {payload.shape}")
