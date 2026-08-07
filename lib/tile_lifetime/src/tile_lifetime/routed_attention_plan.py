# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Inspectable physical candidates for routed sparse attention."""

from dataclasses import dataclass
from enum import StrEnum

import numpy as np

from tile_lifetime.plan import (
    BoundedBuffer,
    PersistentTaskPlacement,
    PersistentTaskRole,
    PersistentWorkerRole,
    ReadinessEvent,
)
from tile_lifetime.relation import RelationPlan


class RoutedAttentionOrientation(StrEnum):
    """Side of the sparse relation that owns the outer physical traversal."""

    QUERY_MAJOR = "query_major"
    KV_MAJOR = "kv_major"
    KV_MAJOR_SLOT_WAVES = "kv_major_slot_waves"


@dataclass(frozen=True)
class RoutedAttentionPlanConfig:
    """Explicit physical assumptions shared by the initial two candidates."""

    query_block_size: int
    key_value_block_size: int
    query_heads: int
    key_value_heads: int
    head_dimension: int
    value_dimension: int
    buffer_depth: int
    transfer_workers: int
    matrix_workers: int
    reduction_workers: int
    input_element_bytes: int = 2
    state_element_bytes: int = 4


@dataclass(frozen=True)
class RoutedAttentionPhysicalPlan:
    """One bounded schedule over a shared query-block/KV-block relation."""

    orientation: RoutedAttentionOrientation
    relation: RelationPlan
    config: RoutedAttentionPlanConfig
    task_roles: tuple[PersistentTaskRole, ...]
    worker_roles: tuple[PersistentWorkerRole, ...]
    readiness_events: tuple[ReadinessEvent, ...]
    buffers: tuple[BoundedBuffer, ...]
    score_tile_bytes: int
    partial_state_materialization_bytes: int
    online_state_materialization_bytes: int
    output_materialization_bytes: int
    kernel_regions: tuple[tuple[str, ...], ...]
    numerical_policy: str

    @property
    def sequence_squared_materialization_bytes(self) -> int:
        """No candidate may allocate a full query-by-key score or probability tensor."""
        return 0

    def event(self, name: str) -> ReadinessEvent:
        """Return a named readiness event."""
        matches = tuple(event for event in self.readiness_events if event.name == name)
        if len(matches) != 1:
            raise KeyError(f"expected one event named {name}, found {len(matches)}")
        return matches[0]

    def dump(self) -> str:
        """Render schedule choices and derived resource costs without backend source."""
        lines = [
            f"RoutedAttention orientation={self.orientation.value}",
            (
                f"  relation: query_blocks={self.relation.source_item_count} "
                f"selected_edges={self.relation.route_count} kv_blocks={self.relation.destination_count}"
            ),
            (
                f"  tiles: Q={self.config.query_block_size} KV={self.config.key_value_block_size} "
                f"heads={self.config.query_heads}/{self.config.key_value_heads} "
                f"dimension={self.config.head_dimension}"
            ),
            "  tasks: " + " -> ".join(task.name for task in self.task_roles),
        ]
        for event in self.readiness_events:
            lines.append(
                f"  event {event.name}: arrivals={_format_arrivals(event.required_arrivals)} "
                f"granularity={event.granularity}"
            )
        for buffer in self.buffers:
            lines.append(
                f"  buffer {buffer.name}: capacity={buffer.capacity_items} bytes={buffer.size_bytes} "
                f"reuse_after={buffer.reuse_after}"
            )
        lines.extend(
            (
                f"  score_tile_bytes: {self.score_tile_bytes} (internal only)",
                f"  partial_state_materialization_bytes: {self.partial_state_materialization_bytes}",
                f"  online_state_materialization_bytes: {self.online_state_materialization_bytes}",
                "  sequence_squared_materialization_bytes: 0",
                f"  numerical_policy: {self.numerical_policy}",
            )
        )
        return "\n".join(lines)


def compile_routed_attention_candidates(
    relation: RelationPlan, config: RoutedAttentionPlanConfig
) -> tuple[RoutedAttentionPhysicalPlan, RoutedAttentionPhysicalPlan]:
    """Emit query-major and KV-major candidates from the same relation."""
    _validate_config(relation, config)
    return _query_major_plan(relation, config), _kv_major_plan(relation, config)


def _query_major_plan(relation: RelationPlan, config: RoutedAttentionPlanConfig) -> RoutedAttentionPhysicalPlan:
    source_degree = tuple(int(value) for value in np.count_nonzero(relation.edge_valid, axis=1))
    query_state_bytes = (
        config.buffer_depth
        * config.query_block_size
        * config.query_heads
        * (config.value_dimension + 2)
        * config.state_element_bytes
    )
    staged_kv_bytes = (
        config.buffer_depth
        * config.key_value_block_size
        * config.key_value_heads
        * (config.head_dimension + config.value_dimension)
        * config.input_element_bytes
    )
    tasks = (
        PersistentTaskRole(
            name="load_query_block",
            placement=PersistentTaskPlacement.CTA_LOCAL,
            inputs=("query",),
            outputs=("resident_query",),
            signals=("query_resident",),
        ),
        PersistentTaskRole(
            name="stream_selected_kv",
            placement=PersistentTaskPlacement.CLUSTER,
            inputs=("relation", "key", "value"),
            outputs=("staged_kv",),
            waits_for=("query_resident",),
            signals=("kv_tile_ready",),
        ),
        PersistentTaskRole(
            name="qk_online_update_pv",
            placement=PersistentTaskPlacement.CLUSTER,
            inputs=("resident_query", "staged_kv", "online_state"),
            outputs=("online_state",),
            waits_for=("kv_tile_ready",),
            signals=("query_selected_edges_complete",),
        ),
        PersistentTaskRole(
            name="finalize_query",
            placement=PersistentTaskPlacement.CTA_LOCAL,
            inputs=("online_state",),
            outputs=("attention_output",),
            waits_for=("query_selected_edges_complete",),
        ),
    )
    return RoutedAttentionPhysicalPlan(
        orientation=RoutedAttentionOrientation.QUERY_MAJOR,
        relation=relation,
        config=config,
        task_roles=tasks,
        worker_roles=_worker_roles(config),
        readiness_events=(
            ReadinessEvent(
                name="query_selected_edges_complete",
                producers=("qk_online_update_pv",),
                consumers=("finalize_query",),
                granularity="query_block",
                required_arrivals=source_degree,
            ),
        ),
        buffers=(
            BoundedBuffer(
                name="staged_kv",
                item_domain="selected KV block",
                capacity_items=config.buffer_depth,
                size_bytes=staged_kv_bytes,
                producer="stream_selected_kv",
                consumers=("qk_online_update_pv",),
                reuse_after="qk_online_update_pv completion for buffer generation",
                placement="shared_memory",
            ),
            BoundedBuffer(
                name="online_state",
                item_domain="resident query block",
                capacity_items=config.buffer_depth,
                size_bytes=query_state_bytes,
                producer="qk_online_update_pv",
                consumers=("qk_online_update_pv", "finalize_query"),
                reuse_after="finalize_query",
                placement="register_or_shared_memory",
            ),
        ),
        score_tile_bytes=_score_tile_bytes(config),
        partial_state_materialization_bytes=0,
        online_state_materialization_bytes=0,
        output_materialization_bytes=_output_bytes(relation, config),
        kernel_regions=(("load_query_block", "stream_selected_kv", "qk_online_update_pv", "finalize_query"),),
        numerical_policy="stable selected-slot FP32 online update; bounded reassociation within QK/PV tiles",
    )


def _kv_major_plan(relation: RelationPlan, config: RoutedAttentionPlanConfig) -> RoutedAttentionPhysicalPlan:
    source_degree = tuple(int(value) for value in np.count_nonzero(relation.edge_valid, axis=1))
    destination_degree = tuple(int(value) for value in relation.group_count)
    one_partial_bytes = (
        config.query_block_size * config.query_heads * (config.value_dimension + 2) * config.state_element_bytes
    )
    partial_bytes = relation.route_count * one_partial_bytes
    staged_kv_bytes = (
        config.buffer_depth
        * config.key_value_block_size
        * config.key_value_heads
        * (config.head_dimension + config.value_dimension)
        * config.input_element_bytes
    )
    tasks = (
        PersistentTaskRole(
            name="stage_kv_block",
            placement=PersistentTaskPlacement.CLUSTER,
            inputs=("relation", "key", "value"),
            outputs=("resident_kv",),
            signals=("kv_block_ready",),
        ),
        PersistentTaskRole(
            name="grouped_query_block_partial",
            placement=PersistentTaskPlacement.CLUSTER,
            inputs=("resident_kv", "query"),
            outputs=("edge_partial_state",),
            waits_for=("kv_block_ready",),
            signals=("kv_incident_queries_complete", "edge_partial_ready"),
        ),
        PersistentTaskRole(
            name="inverse_route_partial_state",
            placement=PersistentTaskPlacement.CTA_LOCAL,
            inputs=("relation", "edge_partial_state"),
            outputs=("query_slot_partial_state",),
            waits_for=("edge_partial_ready",),
            signals=("query_partials_ready",),
        ),
        PersistentTaskRole(
            name="stable_query_partial_merge",
            placement=PersistentTaskPlacement.CTA_LOCAL,
            inputs=("query_slot_partial_state",),
            outputs=("attention_output",),
            waits_for=("query_partials_ready",),
        ),
    )
    return RoutedAttentionPhysicalPlan(
        orientation=RoutedAttentionOrientation.KV_MAJOR,
        relation=relation,
        config=config,
        task_roles=tasks,
        worker_roles=_worker_roles(config),
        readiness_events=(
            ReadinessEvent(
                name="kv_incident_queries_complete",
                producers=("grouped_query_block_partial",),
                consumers=("stage_kv_block",),
                granularity="KV block",
                required_arrivals=destination_degree,
                generation="staged_kv_buffer_generation",
            ),
            ReadinessEvent(
                name="query_partials_ready",
                producers=("inverse_route_partial_state",),
                consumers=("stable_query_partial_merge",),
                granularity="query_block",
                required_arrivals=source_degree,
            ),
        ),
        buffers=(
            BoundedBuffer(
                name="resident_kv",
                item_domain="KV block",
                capacity_items=config.buffer_depth,
                size_bytes=staged_kv_bytes,
                producer="stage_kv_block",
                consumers=("grouped_query_block_partial",),
                reuse_after="kv_incident_queries_complete for buffer generation",
                placement="shared_memory",
            ),
            BoundedBuffer(
                name="edge_partial_state",
                item_domain="valid relation edge",
                capacity_items=relation.route_count,
                size_bytes=partial_bytes,
                producer="grouped_query_block_partial",
                consumers=("inverse_route_partial_state", "stable_query_partial_merge"),
                reuse_after="stable_query_partial_merge",
                placement="global_memory",
            ),
        ),
        score_tile_bytes=_score_tile_bytes(config),
        partial_state_materialization_bytes=partial_bytes,
        online_state_materialization_bytes=0,
        output_materialization_bytes=_output_bytes(relation, config),
        kernel_regions=(
            ("stage_kv_block", "grouped_query_block_partial"),
            ("inverse_route_partial_state", "stable_query_partial_merge"),
        ),
        numerical_policy="source query block, then selected-slot FP32 partial merge without atomics",
    )


def compile_bounded_kv_major_candidate(
    relation: RelationPlan, config: RoutedAttentionPlanConfig
) -> RoutedAttentionPhysicalPlan:
    """Emit deterministic KV-major slot waves with one writer per query state."""
    _validate_config(relation, config)
    state_bytes = (
        relation.source_item_count
        * config.query_block_size
        * config.query_heads
        * (config.value_dimension + 2)
        * config.state_element_bytes
    )
    staged_kv_bytes = (
        config.buffer_depth
        * config.key_value_block_size
        * config.key_value_heads
        * (config.head_dimension + config.value_dimension)
        * config.input_element_bytes
    )
    task_roles: list[PersistentTaskRole] = [
        PersistentTaskRole(
            name="initialize_query_online_state",
            placement=PersistentTaskPlacement.CTA_LOCAL,
            inputs=(),
            outputs=("query_online_state",),
            signals=("slot_0_state_ready",),
        )
    ]
    readiness_events: list[ReadinessEvent] = [
        ReadinessEvent(
            name="slot_0_state_ready",
            producers=("initialize_query_online_state",),
            consumers=("slot_0_kv_major_qk_pv_merge",),
            granularity="query_block",
            required_arrivals=relation.source_item_count,
        )
    ]
    kernel_regions: list[tuple[str, ...]] = [("initialize_query_online_state",)]
    for selected_slot in range(relation.route_slots):
        valid_slot_edges = relation.edge_valid[:, selected_slot]
        slot_edge_count = int(np.count_nonzero(valid_slot_edges))
        slot_group_counts = _slot_group_counts(relation, selected_slot)
        task_name = f"slot_{selected_slot}_kv_major_qk_pv_merge"
        input_event = f"slot_{selected_slot}_state_ready"
        output_event = f"slot_{selected_slot + 1}_state_ready"
        task_roles.append(
            PersistentTaskRole(
                name=task_name,
                placement=PersistentTaskPlacement.CLUSTER,
                inputs=("relation", "query", "key", "value", "query_online_state"),
                outputs=("query_online_state",),
                waits_for=(input_event,),
                signals=(output_event,),
            )
        )
        readiness_events.extend(
            (
                ReadinessEvent(
                    name=f"slot_{selected_slot}_kv_groups_complete",
                    producers=(task_name,),
                    consumers=(task_name,),
                    granularity="KV block",
                    required_arrivals=slot_group_counts,
                    generation=f"selected_slot_{selected_slot}",
                ),
                ReadinessEvent(
                    name=output_event,
                    producers=(task_name,),
                    consumers=(
                        (
                            f"slot_{selected_slot + 1}_kv_major_qk_pv_merge"
                            if selected_slot + 1 < relation.route_slots
                            else "finalize_query"
                        ),
                    ),
                    granularity="selected-slot wave",
                    required_arrivals=slot_edge_count,
                    generation=f"selected_slot_{selected_slot}",
                ),
            )
        )
        kernel_regions.append((task_name,))
    task_roles.append(
        PersistentTaskRole(
            name="finalize_query",
            placement=PersistentTaskPlacement.CTA_LOCAL,
            inputs=("query_online_state",),
            outputs=("attention_output",),
            waits_for=(f"slot_{relation.route_slots}_state_ready",),
        )
    )
    kernel_regions.append(("finalize_query",))
    return RoutedAttentionPhysicalPlan(
        orientation=RoutedAttentionOrientation.KV_MAJOR_SLOT_WAVES,
        relation=relation,
        config=config,
        task_roles=tuple(task_roles),
        worker_roles=_worker_roles(config),
        readiness_events=tuple(readiness_events),
        buffers=(
            BoundedBuffer(
                name="staged_kv",
                item_domain="KV block within selected-slot wave",
                capacity_items=config.buffer_depth,
                size_bytes=staged_kv_bytes,
                producer="slot wave KV staging",
                consumers=("slot wave QK/PV",),
                reuse_after="all incident queries for the KV block and slot complete",
                placement="shared_memory",
            ),
            BoundedBuffer(
                name="query_online_state",
                item_domain="query block",
                capacity_items=relation.source_item_count,
                size_bytes=state_bytes,
                producer="selected-slot wave",
                consumers=("following selected-slot wave", "finalize_query"),
                reuse_after="finalize_query",
                placement="global_memory",
            ),
        ),
        score_tile_bytes=_score_tile_bytes(config),
        partial_state_materialization_bytes=0,
        online_state_materialization_bytes=state_bytes,
        output_materialization_bytes=_output_bytes(relation, config),
        kernel_regions=tuple(kernel_regions),
        numerical_policy=(
            "ascending selected-slot FP32 online updates; one query-state writer per wave; no atomic accumulation"
        ),
    )


def _slot_group_counts(relation: RelationPlan, selected_slot: int) -> tuple[int, ...]:
    counts = np.zeros(relation.destination_count, dtype=np.int32)
    valid_sources = np.flatnonzero(relation.edge_valid[:, selected_slot])
    destinations = relation.destination_item[valid_sources * relation.route_slots + selected_slot]
    np.add.at(counts, destinations, 1)
    return tuple(int(value) for value in counts)


def _worker_roles(config: RoutedAttentionPlanConfig) -> tuple[PersistentWorkerRole, ...]:
    return (
        PersistentWorkerRole(
            name="transfer_workers",
            count=config.transfer_workers,
            responsibilities=("stage Q/K/V tiles",),
        ),
        PersistentWorkerRole(
            name="matrix_workers",
            count=config.matrix_workers,
            responsibilities=("QK contraction", "PV contraction"),
        ),
        PersistentWorkerRole(
            name="reduction_workers",
            count=config.reduction_workers,
            responsibilities=("online state update", "partial-state merge", "finalization"),
        ),
    )


def _score_tile_bytes(config: RoutedAttentionPlanConfig) -> int:
    return config.query_block_size * config.query_heads * config.key_value_block_size * config.state_element_bytes


def _output_bytes(relation: RelationPlan, config: RoutedAttentionPlanConfig) -> int:
    return (
        relation.source_item_count
        * config.query_block_size
        * config.query_heads
        * config.value_dimension
        * config.input_element_bytes
    )


def _validate_config(relation: RelationPlan, config: RoutedAttentionPlanConfig) -> None:
    positive = {
        "query block size": config.query_block_size,
        "key/value block size": config.key_value_block_size,
        "query heads": config.query_heads,
        "key/value heads": config.key_value_heads,
        "head dimension": config.head_dimension,
        "value dimension": config.value_dimension,
        "buffer depth": config.buffer_depth,
        "transfer workers": config.transfer_workers,
        "matrix workers": config.matrix_workers,
        "reduction workers": config.reduction_workers,
    }
    invalid = tuple(name for name, value in positive.items() if value <= 0)
    if invalid:
        raise ValueError(f"routed-attention physical values must be positive: {', '.join(invalid)}")
    if config.query_heads % config.key_value_heads:
        raise ValueError("query heads must map evenly onto key/value heads")
    if np.any(np.count_nonzero(relation.edge_valid, axis=1) == 0):
        raise ValueError("every query block must select at least one KV block")


def _format_arrivals(arrivals: int | tuple[int, ...] | str) -> str:
    if not isinstance(arrivals, tuple) or len(arrivals) <= 16:
        return str(arrivals)
    values = np.asarray(arrivals)
    return (
        f"count={values.size}, min={int(np.min(values))}, median={float(np.median(values)):.1f}, "
        f"max={int(np.max(values))}, total={int(np.sum(values))}"
    )
