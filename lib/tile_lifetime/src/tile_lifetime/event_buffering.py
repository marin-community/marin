# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bounded buffer reuse and Event Tensor realization audits.

This module stays below tensor semantics.  It takes scheduled task families,
their exact dependences, and a chosen finite buffer assignment.  From those
objects it derives the last-consumer-to-next-producer dependences needed to
reuse physical slots safely.  It also records whether each logical event is
realized physically or erased by a proven execution order.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from enum import StrEnum
from itertools import pairwise

from tile_lifetime.event_dataflow import (
    Coordinate,
    EventDataflowError,
    EventDataflowProgram,
    EventTensorPlan,
    MemoryVisibility,
    TaskDependence,
    TaskFamily,
    TaskInstance,
    TaskRelation,
)


class EventRealizationKind(StrEnum):
    """How one logical Event Tensor is implemented after worker assignment."""

    ERASED_PROGRAM_ORDER = "erased_program_order"
    ERASED_STREAM_ORDER = "erased_stream_order"
    PHYSICAL = "physical"


@dataclass(frozen=True)
class EventRealization:
    """One auditable implementation choice for an Event Tensor plan."""

    plan_name: str
    kind: EventRealizationKind
    mechanism: str
    reason: str
    ordering: TaskRelation | None = None

    def __post_init__(self) -> None:
        if not self.plan_name or not self.mechanism or not self.reason:
            raise ValueError("event realization fields must be non-empty")
        if self.kind is EventRealizationKind.PHYSICAL and self.ordering is not None:
            raise ValueError("a physical event realization must not claim an erasing order")
        if self.kind is not EventRealizationKind.PHYSICAL and self.ordering is None:
            raise ValueError("an erased event realization requires an explicit ordering relation")


@dataclass(frozen=True)
class EventRealizationAudit:
    """Verified realization choices for every logical Event Tensor."""

    entries: tuple[EventRealization, ...]

    @property
    def erased(self) -> tuple[EventRealization, ...]:
        """Return plans erased by a proven execution order."""
        return tuple(entry for entry in self.entries if entry.kind is not EventRealizationKind.PHYSICAL)

    @property
    def physical(self) -> tuple[EventRealization, ...]:
        """Return plans that require a physical synchronization primitive."""
        return tuple(entry for entry in self.entries if entry.kind is EventRealizationKind.PHYSICAL)


@dataclass(frozen=True)
class BoundedBufferPlan:
    """Finite storage assignment plus mechanically derived safe-reuse edges.

    ``producer`` coordinates identify logical buffer items. ``uses`` map each
    item to all tasks that read it. ``slots`` and ``generations`` are schedule
    choices over the producer domain. ``reuse_dependences`` are derived from
    the unique last consumer of each prior slot generation.
    """

    name: str
    producer: TaskFamily
    uses: tuple[TaskRelation, ...]
    capacity: int
    slots: tuple[int, ...]
    generations: tuple[int, ...]
    last_consumers: tuple[tuple[Coordinate, TaskInstance], ...]
    reuse_dependences: tuple[TaskDependence, ...]

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("bounded buffer name must be non-empty")
        item_count = len(self.producer.coordinates)
        if self.capacity <= 0:
            raise ValueError("bounded buffer capacity must be positive")
        if len(self.slots) != item_count or len(self.generations) != item_count:
            raise ValueError("bounded buffer assignments must cover every logical item")
        if any(slot < 0 or slot >= self.capacity for slot in self.slots):
            raise ValueError("bounded buffer slot is outside its capacity")
        if any(generation < 0 for generation in self.generations):
            raise ValueError("bounded buffer generations must be non-negative")
        if len(set(zip(self.slots, self.generations, strict=True))) != item_count:
            raise ValueError("one physical slot/generation pair may identify only one logical item")


def derive_bounded_buffer_plan(
    *,
    name: str,
    program: EventDataflowProgram,
    producer: TaskFamily,
    uses: tuple[TaskRelation, ...],
    capacity: int,
    slot_for: dict[Coordinate, int],
    generation_for: dict[Coordinate, int],
    visibility: MemoryVisibility,
) -> BoundedBufferPlan:
    """Derive last-consumer reuse edges from exact task dataflow.

    A last consumer is not supplied by the workload.  It is the unique maximal
    task, under the scheduled dependency graph, among all readers of one
    logical item.  Reuse is legal only when such a task exists and precedes the
    next producer assigned to the same physical slot.
    """
    if producer not in program.task_families:
        raise EventDataflowError(("bounded buffer producer is outside the task program",))
    for relation in uses:
        if relation.source != producer:
            raise EventDataflowError(("every bounded buffer use must originate at its producer family",))
    coordinates = producer.coordinates
    if set(slot_for) != set(coordinates) or set(generation_for) != set(coordinates):
        raise EventDataflowError(("bounded buffer slot and generation maps must cover the producer domain",))

    reachability = _task_reachability(program)
    readers_by_item: dict[Coordinate, set[TaskInstance]] = defaultdict(set)
    for relation in uses:
        assert isinstance(relation.target, TaskFamily)
        for pair in relation.pairs:
            readers_by_item[pair.source].add(TaskInstance(relation.target.name, pair.target))

    last_consumers: dict[Coordinate, TaskInstance] = {}
    for coordinate in coordinates:
        readers = readers_by_item[coordinate]
        if not readers:
            raise EventDataflowError((f"bounded buffer item {coordinate} has no consumer",))
        maxima = tuple(
            reader for reader in readers if all(other == reader or reader in reachability[other] for other in readers)
        )
        if len(maxima) != 1:
            raise EventDataflowError(
                (f"bounded buffer item {coordinate} has no unique last consumer: {sorted(readers)}",)
            )
        last_consumers[coordinate] = maxima[0]

    by_slot: dict[int, list[Coordinate]] = defaultdict(list)
    for coordinate in coordinates:
        by_slot[slot_for[coordinate]].append(coordinate)
    reuse_pairs_by_families: dict[tuple[str, str], list[tuple[Coordinate, Coordinate]]] = defaultdict(list)
    family_by_name = {family.name: family for family in program.task_families}
    for slot, items in by_slot.items():
        ordered = sorted(items, key=lambda coordinate: generation_for[coordinate])
        generations = [generation_for[coordinate] for coordinate in ordered]
        if generations != list(range(generations[0], generations[0] + len(generations))):
            raise EventDataflowError((f"buffer slot {slot} generations must be contiguous",))
        for prior, following in pairwise(ordered):
            last = last_consumers[prior]
            reuse_pairs_by_families[(last.family, producer.name)].append((last.coordinate, following))

    reuse_dependences = []
    for (source_name, target_name), pairs in sorted(reuse_pairs_by_families.items()):
        source = family_by_name[source_name]
        target = family_by_name[target_name]
        reuse_dependences.append(TaskDependence(TaskRelation.from_pairs(source, target, pairs), visibility))
    return BoundedBufferPlan(
        name=name,
        producer=producer,
        uses=uses,
        capacity=capacity,
        slots=tuple(slot_for[coordinate] for coordinate in coordinates),
        generations=tuple(generation_for[coordinate] for coordinate in coordinates),
        last_consumers=tuple((coordinate, last_consumers[coordinate]) for coordinate in coordinates),
        reuse_dependences=tuple(reuse_dependences),
    )


def verify_event_realizations(
    program: EventDataflowProgram,
    realizations: tuple[EventRealization, ...],
) -> EventRealizationAudit:
    """Verify every event is either physical or erased by a covering order."""
    by_name = {entry.plan_name: entry for entry in realizations}
    if len(by_name) != len(realizations):
        raise EventDataflowError(("event realization plan names must be unique",))
    expected = {plan.name for plan in program.event_plans}
    if set(by_name) != expected:
        raise EventDataflowError(
            (f"event realizations must cover every plan exactly; expected={sorted(expected)}, found={sorted(by_name)}",)
        )
    for plan in program.event_plans:
        realization = by_name[plan.name]
        if realization.kind is EventRealizationKind.PHYSICAL:
            continue
        assert realization.ordering is not None
        required = plan.scheduled_dependence
        if realization.ordering.source != required.source or realization.ordering.target != required.target:
            raise EventDataflowError((f"erasing order for {plan.name} has incompatible task domains",))
        missing = set(required.pairs) - set(realization.ordering.pairs)
        if missing:
            raise EventDataflowError((f"erasing order for {plan.name} omits scheduled edges {sorted(missing)}",))
    return EventRealizationAudit(tuple(by_name[plan.name] for plan in program.event_plans))


def physical_event_realization(plan: EventTensorPlan, *, mechanism: str, reason: str) -> EventRealization:
    """Build an explicit physical realization record."""
    return EventRealization(plan.name, EventRealizationKind.PHYSICAL, mechanism, reason)


def erased_event_realization(
    plan: EventTensorPlan,
    *,
    kind: EventRealizationKind,
    mechanism: str,
    reason: str,
    ordering: TaskRelation | None = None,
) -> EventRealization:
    """Build an erased realization whose order covers the scheduled relation."""
    if kind is EventRealizationKind.PHYSICAL:
        raise ValueError("use physical_event_realization for physical synchronization")
    return EventRealization(
        plan.name,
        kind,
        mechanism,
        reason,
        ordering if ordering is not None else plan.scheduled_dependence,
    )


def _task_reachability(program: EventDataflowProgram) -> dict[TaskInstance, set[TaskInstance]]:
    tasks = {
        TaskInstance(family.name, coordinate) for family in program.task_families for coordinate in family.coordinates
    }
    outgoing: dict[TaskInstance, set[TaskInstance]] = defaultdict(set)
    for plan in program.event_plans:
        for pair in plan.scheduled_dependence.pairs:
            outgoing[TaskInstance(plan.notify_relation.source.name, pair.source)].add(
                TaskInstance(plan.trigger_relation.target.name, pair.target)
            )
    reachability = {task: set() for task in tasks}
    for source in tasks:
        frontier = deque(outgoing[source])
        while frontier:
            target = frontier.popleft()
            if target in reachability[source]:
                continue
            reachability[source].add(target)
            frontier.extend(outgoing[target])
    return reachability
