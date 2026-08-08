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
from tile_lifetime.streaming_attention import AttentionScoreAxis, StreamingAttentionProgram


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


@dataclass(frozen=True)
class RoutedStreamingAttentionCompilation:
    """One semantic online-attention program scheduled over a sparse relation."""

    program: StreamingAttentionProgram
    relation: RelationPlan
    candidates: tuple[RoutedAttentionPhysicalPlan, RoutedAttentionPhysicalPlan]


@dataclass(frozen=True)
class QueryMajorBlockIndexPlan:
    """Compact block lists derived from the source-major relation orientation."""

    block_count: np.ndarray
    block_index: np.ndarray


@dataclass(frozen=True)
class BoundedKVReuseWave:
    """Fixed-capacity right-major tasks for one source-order relation slot."""

    selected_slot: int
    key_value_block: np.ndarray
    query_block: np.ndarray
    query_count: np.ndarray

    @property
    def task_count(self) -> int:
        """Number of independently schedulable right-resource tasks."""
        return int(self.key_value_block.size)

    @property
    def edge_count(self) -> int:
        """Number of relation edges covered by the tasks."""
        return int(np.sum(self.query_count))


@dataclass(frozen=True)
class BoundedKVReusePlan:
    """Deterministic slot waves with bounded query consumers per staged KV block."""

    source_item_count: int
    destination_count: int
    route_slots: int
    query_capacity_per_task: int
    waves: tuple[BoundedKVReuseWave, ...]

    @property
    def task_count(self) -> int:
        """Total physical task count across every selected-slot wave."""
        return sum(wave.task_count for wave in self.waves)

    @property
    def edge_count(self) -> int:
        """Total relation edges covered by every selected-slot wave."""
        return sum(wave.edge_count for wave in self.waves)


def bounded_kv_reuse_plan(relation: RelationPlan, *, query_capacity_per_task: int) -> BoundedKVReusePlan:
    """Group each relation slot by right resource without changing source order.

    Slot waves serialize the only updates that may target the same query state.
    Within one wave every source occurs at most once, so right-major tasks can
    write their query states directly without atomics.  Large right-side groups
    are split into bounded tasks; each task may stage its right resource once
    and reuse it for at most ``query_capacity_per_task`` consumers.
    """
    if query_capacity_per_task <= 0:
        raise ValueError("query capacity per KV-reuse task must be positive")

    waves = []
    for selected_slot in range(relation.route_slots):
        source = np.flatnonzero(relation.edge_valid[:, selected_slot]).astype(np.int32, copy=False)
        route = source * relation.route_slots + selected_slot
        destination = relation.destination_item[route].astype(np.int32, copy=False)
        order = np.lexsort((source, destination))
        source = source[order]
        destination = destination[order]

        task_destinations: list[int] = []
        task_queries: list[np.ndarray] = []
        start = 0
        while start < destination.size:
            stop = start + 1
            while stop < destination.size and destination[stop] == destination[start]:
                stop += 1
            for chunk_start in range(start, stop, query_capacity_per_task):
                chunk = source[chunk_start : min(chunk_start + query_capacity_per_task, stop)]
                task_destinations.append(int(destination[start]))
                task_queries.append(chunk)
            start = stop

        query_block = np.full(
            (len(task_queries), query_capacity_per_task),
            -1,
            dtype=np.int32,
        )
        query_count = np.empty(len(task_queries), dtype=np.int32)
        for task_index, chunk in enumerate(task_queries):
            query_block[task_index, : chunk.size] = chunk
            query_count[task_index] = chunk.size
        wave = BoundedKVReuseWave(
            selected_slot=selected_slot,
            key_value_block=np.asarray(task_destinations, dtype=np.int32),
            query_block=query_block,
            query_count=query_count,
        )
        if wave.edge_count != source.size:
            raise ValueError(f"slot {selected_slot} lost relation edges: {wave.edge_count} != {source.size}")
        valid_queries = wave.query_block[wave.query_block >= 0]
        if np.unique(valid_queries).size != valid_queries.size:
            raise ValueError(f"slot {selected_slot} assigns one query state to multiple physical tasks")
        waves.append(wave)

    plan = BoundedKVReusePlan(
        source_item_count=relation.source_item_count,
        destination_count=relation.destination_count,
        route_slots=relation.route_slots,
        query_capacity_per_task=query_capacity_per_task,
        waves=tuple(waves),
    )
    if plan.edge_count != relation.route_count:
        raise ValueError(f"bounded KV reuse lost relation edges: {plan.edge_count} != {relation.route_count}")
    return plan


def query_major_block_index_plan(relation: RelationPlan) -> QueryMajorBlockIndexPlan:
    """Lower a generic relation to compact per-source destination block lists.

    This is an index-plane lowering only.  It preserves source-local route-slot
    order and does not attach attention semantics or select a named kernel.
    """
    destination = relation.destination_item.reshape(relation.source_item_count, relation.route_slots)
    block_count = np.count_nonzero(relation.edge_valid, axis=1).astype(np.int32)
    block_index = np.zeros_like(destination, dtype=np.int32)
    for source in range(relation.source_item_count):
        valid_destinations = destination[source, relation.edge_valid[source]]
        block_index[source, : valid_destinations.shape[0]] = valid_destinations
    return QueryMajorBlockIndexPlan(block_count=block_count, block_index=block_index)


def compile_routed_streaming_attention_candidates(
    program: StreamingAttentionProgram,
    relation: RelationPlan,
    config: RoutedAttentionPlanConfig,
) -> RoutedStreamingAttentionCompilation:
    """Schedule derived Contract/Map/Fold attention in both relation orientations."""
    query = program.qk.inputs[0]
    key = program.qk.inputs[1]
    value = program.pv.inputs[1]
    query_axis = next((axis for axis in query.axes if axis.label == AttentionScoreAxis.QUERY.value), None)
    key_axis = next((axis for axis in key.axes if axis.label == AttentionScoreAxis.KEY.value), None)
    query_head_axis = next((axis for axis in query.axes if axis.label == AttentionScoreAxis.HEAD.value), None)
    key_value_head_axis = next((axis for axis in key.axes if axis.label == "key_value_head"), None)
    if query_axis is None or key_axis is None or query_head_axis is None or key_value_head_axis is None:
        raise ValueError("routed attention requires explicit query, key, and head axes")
    expected_query_length = relation.source_item_count * config.query_block_size
    minimum_key_length = relation.destination_count * config.key_value_block_size
    reasons = []
    if query_axis.extent != expected_query_length:
        reasons.append(f"query extent {query_axis.extent} does not equal relation blocks x tile {expected_query_length}")
    if key_axis.extent < minimum_key_length:
        reasons.append(f"key extent {key_axis.extent} is smaller than relation destination span {minimum_key_length}")
    if query_head_axis.extent != config.query_heads:
        reasons.append("semantic query-head count does not match the physical candidate")
    if key_value_head_axis.extent != config.key_value_heads:
        reasons.append("semantic key/value-head count does not match the physical candidate")
    if query.axes[-1].extent != config.head_dimension or key.axes[-1].extent != config.head_dimension:
        reasons.append("semantic Q/K feature dimensions do not match the physical candidate")
    if value.axes[-1].extent != config.value_dimension:
        reasons.append("semantic value dimension does not match the physical candidate")
    if program.schedule.query_tile_size != config.query_block_size:
        reasons.append("semantic query tile does not match the relation block size")
    if program.schedule.key_value_tile_size != config.key_value_block_size:
        reasons.append("semantic K/V tile does not match the relation block size")
    if reasons:
        raise ValueError("; ".join(reasons))
    candidates = compile_routed_attention_candidates(relation, config)
    return RoutedStreamingAttentionCompilation(program=program, relation=relation, candidates=candidates)


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
    relation: RelationPlan,
    config: RoutedAttentionPlanConfig,
    *,
    query_capacity_per_task: int = 2,
) -> RoutedAttentionPhysicalPlan:
    """Emit deterministic KV-major slot waves with bounded right-resource reuse."""
    _validate_config(relation, config)
    reuse_plan = bounded_kv_reuse_plan(relation, query_capacity_per_task=query_capacity_per_task)
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
        wave = reuse_plan.waves[selected_slot]
        slot_group_counts = tuple(
            int(np.count_nonzero(wave.key_value_block == destination))
            for destination in range(relation.destination_count)
        )
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
                    required_arrivals=wave.task_count,
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
            "ascending selected-slot FP32 online updates; bounded right-resource reuse; "
            "one query-state writer per wave; no atomic accumulation"
        ),
    )


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
