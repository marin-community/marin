# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Schedule-level readiness derived from exact tiled task dependencies."""

from __future__ import annotations

import random
from collections import defaultdict, deque
from collections.abc import Callable, Mapping, MutableMapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from itertools import pairwise, product

Coordinate = tuple[int, ...]


class EventDataflowError(ValueError):
    """A structured rejection of an invalid dependency or event plan."""

    def __init__(self, reasons: Sequence[str]):
        self.reasons = tuple(reasons)
        super().__init__("; ".join(self.reasons))


class EventMemoryScope(StrEnum):
    """Minimum address-space scope across which readiness must be visible."""

    CTA = "cta"
    CLUSTER = "cluster"
    DEVICE = "device"
    SYSTEM = "system"


class EventGenerationPolicy(StrEnum):
    """How logical event storage is distinguished across repeated executions."""

    PER_INVOCATION = "per_invocation"
    PHASED = "phased"


class EventSchedulingMode(StrEnum):
    """Reference scheduling policies available for one event plan."""

    STATIC = "static"
    DYNAMIC = "dynamic"


class ImperativeEventOpKind(StrEnum):
    """Backend-neutral imperative operations produced by event lowering."""

    INITIALIZE = "initialize"
    WAIT = "wait"
    NOTIFY = "notify"
    TRIGGER_ENQUEUE = "trigger_enqueue"


@dataclass(frozen=True)
class TaskAxis:
    """One concrete task-grid axis after schedule decomposition."""

    name: str
    extent: int

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("task axis name must be non-empty")
        if self.extent < 0:
            raise ValueError(f"task axis extent must be non-negative, got {self.extent}")


@dataclass(frozen=True)
class CoordinateDomain:
    """A named rectangular coordinate space."""

    name: str
    axes: tuple[TaskAxis, ...]

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("coordinate domain name must be non-empty")
        axis_names = tuple(axis.name for axis in self.axes)
        if len(axis_names) != len(set(axis_names)):
            raise ValueError(f"coordinate domain {self.name!r} has duplicate axis names")

    @property
    def coordinates(self) -> tuple[Coordinate, ...]:
        """Enumerate the concrete task or event coordinates deterministically."""
        if not self.axes:
            return ((),)
        return tuple(product(*(range(axis.extent) for axis in self.axes)))

    def contains(self, coordinate: Coordinate) -> bool:
        """Return whether ``coordinate`` is in this rectangular domain."""
        return len(coordinate) == len(self.axes) and all(
            0 <= value < axis.extent for value, axis in zip(coordinate, self.axes, strict=True)
        )


@dataclass(frozen=True)
class TaskFamily(CoordinateDomain):
    """A rectangular family of physical tasks in the chosen decomposition."""

    placement: str | None = None


@dataclass(frozen=True)
class EventDomain(CoordinateDomain):
    """A virtual family of counted readiness coordinates."""


@dataclass(frozen=True, order=True)
class IndexPair:
    """One edge in a concrete coordinate relation."""

    source: Coordinate
    target: Coordinate


@dataclass(frozen=True)
class TaskRelation:
    """A finite index relation between two task or event domains."""

    source: CoordinateDomain
    target: CoordinateDomain
    pairs: tuple[IndexPair, ...]

    def __post_init__(self) -> None:
        if len(self.pairs) != len(set(self.pairs)):
            raise EventDataflowError(("task relation contains duplicate logical edges",))
        normalized = tuple(sorted(self.pairs))
        object.__setattr__(self, "pairs", normalized)
        reasons: list[str] = []
        for pair in normalized:
            if not self.source.contains(pair.source):
                reasons.append(f"source coordinate {pair.source} is outside {self.source.name}")
            if not self.target.contains(pair.target):
                reasons.append(f"target coordinate {pair.target} is outside {self.target.name}")
        if reasons:
            raise EventDataflowError(reasons)

    @classmethod
    def from_pairs(
        cls,
        source: CoordinateDomain,
        target: CoordinateDomain,
        pairs: Sequence[tuple[Coordinate, Coordinate]],
    ) -> TaskRelation:
        """Build a normalized finite relation from coordinate pairs."""
        return cls(
            source,
            target,
            tuple(IndexPair(source_coordinate, target_coordinate) for source_coordinate, target_coordinate in pairs),
        )

    def sources_for(self, target: Coordinate) -> tuple[Coordinate, ...]:
        """Return the source coordinates incident on one target coordinate."""
        return tuple(pair.source for pair in self.pairs if pair.target == target)

    def targets_for(self, source: Coordinate) -> tuple[Coordinate, ...]:
        """Return the target coordinates incident on one source coordinate."""
        return tuple(pair.target for pair in self.pairs if pair.source == source)


@dataclass(frozen=True)
class MemoryVisibility:
    """Release/acquire contract carried by a task dependence."""

    scope: EventMemoryScope
    release_on_notify: bool = True
    acquire_before_consumer: bool = True


@dataclass(frozen=True)
class TaskDependence:
    """Exact required task-to-task data-dependence relation."""

    relation: TaskRelation
    visibility: MemoryVisibility

    def __post_init__(self) -> None:
        if not isinstance(self.relation.source, TaskFamily) or not isinstance(self.relation.target, TaskFamily):
            raise TypeError("a TaskDependence must connect TaskFamily domains")


@dataclass(frozen=True, order=True)
class EventCount:
    """Initial outstanding notification count for one event coordinate."""

    coordinate: Coordinate
    value: int


@dataclass(frozen=True)
class EventCountExpression:
    """Concrete instantiation of a possibly runtime-derived event count tensor."""

    counts: tuple[EventCount, ...]
    provenance: str

    def __post_init__(self) -> None:
        coordinates = tuple(count.coordinate for count in self.counts)
        if len(coordinates) != len(set(coordinates)):
            raise ValueError("an event count expression may define each coordinate once")
        if any(count.value < 0 for count in self.counts):
            raise ValueError("event counts must be non-negative")

    def as_mapping(self) -> dict[Coordinate, int]:
        """Return the concrete event counter initialization."""
        return {count.coordinate: count.value for count in self.counts}


@dataclass(frozen=True)
class EventFactorization:
    """Schedule choice mapping every consumer task to one event coordinate."""

    domain: EventDomain
    consumer_to_event: TaskRelation

    def __post_init__(self) -> None:
        if not isinstance(self.consumer_to_event.source, TaskFamily):
            raise TypeError("event factorization must originate at a TaskFamily")
        if self.consumer_to_event.target != self.domain:
            raise ValueError("event factorization relation must target its declared event domain")


@dataclass(frozen=True)
class EventTensorPlan:
    """One producer-event-consumer factorization of an exact dependence."""

    name: str
    required_dependence: TaskDependence
    domain: EventDomain
    notify_relation: TaskRelation
    trigger_relation: TaskRelation
    initial_count: EventCountExpression
    memory_scope: EventMemoryScope
    generation_policy: EventGenerationPolicy
    visibility: MemoryVisibility
    scheduling_mode: EventSchedulingMode | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("event tensor plan name must be non-empty")

    @property
    def scheduled_dependence(self) -> TaskRelation:
        """Compose notify and trigger relations, including legal false edges."""
        producer_by_event: dict[Coordinate, set[Coordinate]] = defaultdict(set)
        consumer_by_event: dict[Coordinate, set[Coordinate]] = defaultdict(set)
        for pair in self.notify_relation.pairs:
            producer_by_event[pair.target].add(pair.source)
        for pair in self.trigger_relation.pairs:
            consumer_by_event[pair.source].add(pair.target)
        pairs = (
            (producer, consumer)
            for event in self.domain.coordinates
            for producer in producer_by_event[event]
            for consumer in consumer_by_event[event]
        )
        required = self.required_dependence.relation
        return TaskRelation.from_pairs(required.source, required.target, tuple(pairs))


@dataclass(frozen=True)
class EventTensorRuntimeInputs:
    """Flat runtime tables used to initialize and traverse one event plan.

    Coordinates are linearized in their declared domain order. The CSR tables
    are derived views of the generic notify and trigger relations; they are not
    a second source of dependency truth. Backends may upload these tables,
    generate them on device, or erase them when a static schedule makes them
    unnecessary.
    """

    event_initial_counts: tuple[int, ...]
    event_source_offsets: tuple[int, ...]
    event_sources: tuple[int, ...]
    source_event_offsets: tuple[int, ...]
    source_events: tuple[int, ...]
    event_trigger_offsets: tuple[int, ...]
    event_consumers: tuple[int, ...]
    initially_ready_events: tuple[int, ...]
    event_storage_slots: tuple[int, ...]
    event_generations: tuple[int, ...]

    def __post_init__(self) -> None:
        event_count = len(self.event_initial_counts)
        _verify_csr_offsets(self.event_source_offsets, len(self.event_sources), event_count, "event-source")
        _verify_csr_offsets(self.event_trigger_offsets, len(self.event_consumers), event_count, "event-trigger")
        if not self.source_event_offsets:
            raise ValueError("source-event offsets must contain at least the zero sentinel")
        _verify_csr_offsets(
            self.source_event_offsets,
            len(self.source_events),
            len(self.source_event_offsets) - 1,
            "source-event",
        )
        if any(count < 0 for count in self.event_initial_counts):
            raise ValueError("runtime event counts must be non-negative")
        represented_counts = tuple(right - left for left, right in pairwise(self.event_source_offsets))
        if represented_counts != self.event_initial_counts:
            raise ValueError("runtime event counts must equal event-source CSR row lengths")
        source_count = len(self.source_event_offsets) - 1
        if any(source < 0 or source >= source_count for source in self.event_sources):
            raise ValueError("event-source values must index the producer domain")
        if any(event < 0 or event >= event_count for event in self.source_events):
            raise ValueError("source-event values must index the event domain")
        expected_ready = tuple(index for index, count in enumerate(self.event_initial_counts) if count == 0)
        if self.initially_ready_events != expected_ready:
            raise ValueError("initially-ready events must be exactly the zero-count event coordinates")
        if len(self.event_storage_slots) != event_count or len(self.event_generations) != event_count:
            raise ValueError("every logical event must have one physical slot and generation")
        if any(slot < 0 for slot in self.event_storage_slots):
            raise ValueError("physical event slots must be non-negative")
        if any(generation < 0 for generation in self.event_generations):
            raise ValueError("event generations must be non-negative")
        if len(set(zip(self.event_storage_slots, self.event_generations, strict=True))) != event_count:
            raise ValueError("a physical slot/generation pair may identify only one logical event")


@dataclass(frozen=True)
class EventStorageBinding:
    """Physical slot and epoch selected for every logical event coordinate."""

    domain: EventDomain
    slots: tuple[int, ...]
    generations: tuple[int, ...]

    def __post_init__(self) -> None:
        event_count = len(self.domain.coordinates)
        if len(self.slots) != event_count or len(self.generations) != event_count:
            raise ValueError("event storage binding must cover the complete event domain")


def phased_event_storage_binding(
    plan: EventTensorPlan,
    *,
    slot: Callable[[Coordinate], int],
    generation: Callable[[Coordinate], int],
) -> EventStorageBinding:
    """Choose reusable physical event slots for a phased logical plan."""
    if plan.generation_policy is not EventGenerationPolicy.PHASED:
        raise ValueError("reusable event storage requires the phased generation policy")
    coordinates = plan.domain.coordinates
    return EventStorageBinding(
        plan.domain,
        tuple(slot(coordinate) for coordinate in coordinates),
        tuple(generation(coordinate) for coordinate in coordinates),
    )


def _verify_csr_offsets(offsets: tuple[int, ...], value_count: int, row_count: int, name: str) -> None:
    if len(offsets) != row_count + 1:
        raise ValueError(f"{name} offsets must contain one sentinel per row plus one")
    if offsets[0] != 0 or offsets[-1] != value_count:
        raise ValueError(f"{name} offsets must span exactly all values")
    if any(left > right for left, right in pairwise(offsets)):
        raise ValueError(f"{name} offsets must be monotonically non-decreasing")


def event_tensor_runtime_inputs(
    plan: EventTensorPlan,
    *,
    storage_binding: EventStorageBinding | None = None,
) -> EventTensorRuntimeInputs:
    """Linearize a verified plan into runtime count and relation inputs."""
    verify_event_tensor_plan(plan)
    if storage_binding is None:
        if plan.generation_policy is EventGenerationPolicy.PHASED:
            raise ValueError("phased event plans require an explicit physical storage binding")
        storage_binding = EventStorageBinding(
            plan.domain,
            tuple(range(len(plan.domain.coordinates))),
            (0,) * len(plan.domain.coordinates),
        )
    elif storage_binding.domain != plan.domain:
        raise ValueError("event storage binding domain does not match the event plan")
    source = plan.notify_relation.source
    consumer = plan.trigger_relation.target
    source_linear = {coordinate: index for index, coordinate in enumerate(source.coordinates)}
    event_linear = {coordinate: index for index, coordinate in enumerate(plan.domain.coordinates)}
    consumer_linear = {coordinate: index for index, coordinate in enumerate(consumer.coordinates)}

    event_sources_by_event: dict[int, list[int]] = defaultdict(list)
    source_events_by_source: dict[int, list[int]] = defaultdict(list)
    for pair in plan.notify_relation.pairs:
        source_index = source_linear[pair.source]
        event_index = event_linear[pair.target]
        event_sources_by_event[event_index].append(source_index)
        source_events_by_source[source_index].append(event_index)

    consumers_by_event: dict[int, list[int]] = defaultdict(list)
    for pair in plan.trigger_relation.pairs:
        consumers_by_event[event_linear[pair.source]].append(consumer_linear[pair.target])

    event_source_offsets, event_sources = _csr_rows(
        tuple(tuple(sorted(event_sources_by_event[index])) for index in range(len(event_linear)))
    )
    source_event_offsets, source_events = _csr_rows(
        tuple(tuple(sorted(source_events_by_source[index])) for index in range(len(source_linear)))
    )
    event_trigger_offsets, event_consumers = _csr_rows(
        tuple(tuple(sorted(consumers_by_event[index])) for index in range(len(event_linear)))
    )
    count_by_coordinate = plan.initial_count.as_mapping()
    counts = tuple(count_by_coordinate[coordinate] for coordinate in plan.domain.coordinates)
    return EventTensorRuntimeInputs(
        event_initial_counts=counts,
        event_source_offsets=event_source_offsets,
        event_sources=event_sources,
        source_event_offsets=source_event_offsets,
        source_events=source_events,
        event_trigger_offsets=event_trigger_offsets,
        event_consumers=event_consumers,
        initially_ready_events=tuple(index for index, count in enumerate(counts) if count == 0),
        event_storage_slots=storage_binding.slots,
        event_generations=storage_binding.generations,
    )


def _csr_rows(rows: tuple[tuple[int, ...], ...]) -> tuple[tuple[int, ...], tuple[int, ...]]:
    offsets = [0]
    values: list[int] = []
    for row in rows:
        values.extend(row)
        offsets.append(len(values))
    return tuple(offsets), tuple(values)


@dataclass(frozen=True)
class ImperativeEventOperation:
    """One logical counter/wait/queue operation before target legalization."""

    kind: ImperativeEventOpKind
    event: Coordinate
    task: Coordinate | None
    count: int | None
    memory_scope: EventMemoryScope
    visibility: MemoryVisibility


def lower_event_tensor_plan(
    plan: EventTensorPlan,
    *,
    scheduling_mode: EventSchedulingMode,
) -> tuple[ImperativeEventOperation, ...]:
    """Expose the minimal imperative correspondence without choosing GPU primitives."""
    operations: list[ImperativeEventOperation] = [
        ImperativeEventOperation(
            ImperativeEventOpKind.INITIALIZE,
            coordinate,
            None,
            count,
            plan.memory_scope,
            plan.visibility,
        )
        for coordinate, count in plan.initial_count.as_mapping().items()
    ]
    if scheduling_mode is EventSchedulingMode.STATIC:
        operations.extend(
            ImperativeEventOperation(
                ImperativeEventOpKind.WAIT,
                pair.source,
                pair.target,
                None,
                plan.memory_scope,
                plan.visibility,
            )
            for pair in plan.trigger_relation.pairs
        )
    operations.extend(
        ImperativeEventOperation(
            ImperativeEventOpKind.NOTIFY,
            pair.target,
            pair.source,
            None,
            plan.memory_scope,
            plan.visibility,
        )
        for pair in plan.notify_relation.pairs
    )
    if scheduling_mode is EventSchedulingMode.DYNAMIC:
        operations.extend(
            ImperativeEventOperation(
                ImperativeEventOpKind.TRIGGER_ENQUEUE,
                pair.source,
                pair.target,
                None,
                plan.memory_scope,
                plan.visibility,
            )
            for pair in plan.trigger_relation.pairs
        )
    return tuple(operations)


def exact_event_factorization(dependence: TaskDependence, *, name: str) -> EventFactorization:
    """Create one event per consumer coordinate without adding dependencies."""
    target = dependence.relation.target
    assert isinstance(target, TaskFamily)
    domain = EventDomain(name, target.axes)
    pairs = tuple((coordinate, coordinate) for coordinate in target.coordinates)
    return EventFactorization(domain, TaskRelation.from_pairs(target, domain, pairs))


def projected_event_factorization(
    dependence: TaskDependence,
    *,
    domain: EventDomain,
    project: Callable[[Coordinate], Coordinate],
) -> EventFactorization:
    """Coarsen consumers by a coordinate projection chosen by the scheduler."""
    target = dependence.relation.target
    assert isinstance(target, TaskFamily)
    pairs = tuple((coordinate, project(coordinate)) for coordinate in target.coordinates)
    return EventFactorization(domain, TaskRelation.from_pairs(target, domain, pairs))


def derive_event_tensor_plan(
    dependence: TaskDependence,
    *,
    name: str,
    factorization: EventFactorization | None = None,
    memory_scope: EventMemoryScope | None = None,
    generation_policy: EventGenerationPolicy = EventGenerationPolicy.PER_INVOCATION,
    scheduling_mode: EventSchedulingMode | None = None,
) -> EventTensorPlan:
    """Mechanically derive notify, trigger, and count tensors from exact dependencies."""
    required = dependence.relation
    target = required.target
    assert isinstance(target, TaskFamily)
    if factorization is None:
        factorization = exact_event_factorization(dependence, name=f"{name}.events")
    if factorization.consumer_to_event.source != target:
        raise EventDataflowError(("event factorization does not cover the dependence consumer family",))

    event_by_consumer: dict[Coordinate, Coordinate] = {}
    for pair in factorization.consumer_to_event.pairs:
        if pair.source in event_by_consumer:
            raise EventDataflowError((f"consumer {pair.source} maps to more than one event",))
        event_by_consumer[pair.source] = pair.target
    missing = set(target.coordinates) - set(event_by_consumer)
    if missing:
        raise EventDataflowError((f"event factorization omits consumers {sorted(missing)}",))

    notify_pairs = {IndexPair(pair.source, event_by_consumer[pair.target]) for pair in required.pairs}
    trigger_pairs = {IndexPair(event, consumer) for consumer, event in event_by_consumer.items()}
    notify = TaskRelation(required.source, factorization.domain, tuple(notify_pairs))
    trigger = TaskRelation(factorization.domain, target, tuple(trigger_pairs))
    indegree = {coordinate: 0 for coordinate in factorization.domain.coordinates}
    for pair in notify.pairs:
        indegree[pair.target] += 1
    counts = EventCountExpression(
        tuple(EventCount(coordinate, indegree[coordinate]) for coordinate in factorization.domain.coordinates),
        provenance="notify-relation indegree",
    )
    scope = memory_scope if memory_scope is not None else dependence.visibility.scope
    plan = EventTensorPlan(
        name=name,
        required_dependence=dependence,
        domain=factorization.domain,
        notify_relation=notify,
        trigger_relation=trigger,
        initial_count=counts,
        memory_scope=scope,
        generation_policy=generation_policy,
        visibility=dependence.visibility,
        scheduling_mode=scheduling_mode,
    )
    verify_event_tensor_plan(plan)
    return plan


def coarsen_event_tensor_plan(
    plan: EventTensorPlan,
    *,
    domain: EventDomain,
    project: Callable[[Coordinate], Coordinate],
    name: str,
) -> EventTensorPlan:
    """Project consumer coordinates onto fewer events, adding only false dependencies."""
    factorization = projected_event_factorization(plan.required_dependence, domain=domain, project=project)
    return derive_event_tensor_plan(
        plan.required_dependence,
        name=name,
        factorization=factorization,
        memory_scope=plan.memory_scope,
        generation_policy=plan.generation_policy,
        scheduling_mode=plan.scheduling_mode,
    )


_SCOPE_RANK = {
    EventMemoryScope.CTA: 0,
    EventMemoryScope.CLUSTER: 1,
    EventMemoryScope.DEVICE: 2,
    EventMemoryScope.SYSTEM: 3,
}


def verify_event_tensor_plan(plan: EventTensorPlan) -> None:
    """Verify one factorization covers every required edge and has sound counters."""
    reasons: list[str] = []
    required_relation = plan.required_dependence.relation
    relation_domains_match = True
    if plan.notify_relation.source != required_relation.source:
        reasons.append("notify relation source does not match the required producer family")
        relation_domains_match = False
    if plan.notify_relation.target != plan.domain:
        reasons.append("notify relation target does not match the event domain")
        relation_domains_match = False
    if plan.trigger_relation.source != plan.domain:
        reasons.append("trigger relation source does not match the event domain")
        relation_domains_match = False
    if plan.trigger_relation.target != required_relation.target:
        reasons.append("trigger relation target does not match the required consumer family")
        relation_domains_match = False
    if relation_domains_match:
        required = set(required_relation.pairs)
        scheduled = set(plan.scheduled_dependence.pairs)
        missing = required - scheduled
        if missing:
            reasons.append(f"scheduled dependence omits required edges {sorted(missing)}")

    expected_counts = {coordinate: 0 for coordinate in plan.domain.coordinates}
    for pair in plan.notify_relation.pairs:
        expected_counts[pair.target] += 1
    if plan.initial_count.as_mapping() != expected_counts:
        reasons.append("event initial counts do not equal notify-relation indegrees")

    trigger_sources = {pair.source for pair in plan.trigger_relation.pairs}
    unused_events = set(plan.domain.coordinates) - trigger_sources
    if unused_events:
        reasons.append(f"event coordinates have no triggered consumers: {sorted(unused_events)}")
    target_counts: dict[Coordinate, int] = defaultdict(int)
    for pair in plan.trigger_relation.pairs:
        target_counts[pair.target] += 1
    bad_targets = {coordinate: count for coordinate, count in target_counts.items() if count != 1}
    missing_targets = set(plan.required_dependence.relation.target.coordinates) - set(target_counts)
    if bad_targets or missing_targets:
        reasons.append(
            f"each consumer must be triggered by exactly one event; bad={bad_targets}, missing={sorted(missing_targets)}"
        )

    required_visibility = plan.required_dependence.visibility
    if _SCOPE_RANK[plan.memory_scope] < _SCOPE_RANK[required_visibility.scope]:
        reasons.append(
            f"event scope {plan.memory_scope.value} is weaker than required {required_visibility.scope.value}"
        )
    if required_visibility.release_on_notify and not plan.visibility.release_on_notify:
        reasons.append("event lowering omits the required release on notify")
    if required_visibility.acquire_before_consumer and not plan.visibility.acquire_before_consumer:
        reasons.append("event lowering omits the required acquire before consumer")
    if reasons:
        raise EventDataflowError(reasons)


@dataclass(frozen=True, order=True)
class TaskInstance:
    """One concrete coordinate in a task family."""

    family: str
    coordinate: Coordinate


@dataclass(frozen=True)
class EventDataflowProgram:
    """Concrete task graph plus one derived eventization per exact dependence."""

    task_families: tuple[TaskFamily, ...]
    dependences: tuple[TaskDependence, ...]
    event_plans: tuple[EventTensorPlan, ...]


class TraceKind(StrEnum):
    """Stable event-interpreter trace kinds."""

    TASK_READY = "task_ready"
    TASK_EXECUTE = "task_execute"
    NOTIFY = "notify"
    EVENT_READY = "event_ready"


@dataclass(frozen=True)
class EventTraceEntry:
    """One observable reference-interpreter transition."""

    step: int
    kind: TraceKind
    subject: str
    coordinate: Coordinate
    generation: int
    remaining: int | None = None


@dataclass(frozen=True)
class EventExecutionResult:
    """Reference execution state and trace."""

    state: Mapping[str, object]
    trace: tuple[EventTraceEntry, ...]
    executed_tasks: tuple[TaskInstance, ...]
    ready_events: frozenset[tuple[str, Coordinate, int]]


TaskAction = Callable[[Coordinate, MutableMapping[str, object]], None]


def verify_event_dataflow_program(program: EventDataflowProgram) -> None:
    """Verify plan coverage, domains, and acyclicity of a concrete task graph."""
    reasons: list[str] = []
    families_by_name = {family.name: family for family in program.task_families}
    if len(families_by_name) != len(program.task_families):
        reasons.append("task family names must be unique")
    family_set = set(program.task_families)
    for dependence in program.dependences:
        if dependence.relation.source not in family_set or dependence.relation.target not in family_set:
            reasons.append("every dependence endpoint must be present in the program task families")
    expected_dependences = {dependence.relation for dependence in program.dependences}
    planned_dependences = {plan.required_dependence.relation for plan in program.event_plans}
    if len(expected_dependences) != len(program.dependences):
        reasons.append("exact dependences must be unique")
    if expected_dependences != planned_dependences or len(program.dependences) != len(program.event_plans):
        reasons.append("event plans must cover every exact dependence exactly once")
    plan_names = tuple(plan.name for plan in program.event_plans)
    if len(plan_names) != len(set(plan_names)):
        reasons.append("event tensor plan names must be unique within a program")
    for plan in program.event_plans:
        try:
            verify_event_tensor_plan(plan)
        except EventDataflowError as error:
            reasons.extend(f"{plan.name}: {reason}" for reason in error.reasons)

    instances = {
        TaskInstance(family.name, coordinate) for family in program.task_families for coordinate in family.coordinates
    }
    outgoing: dict[TaskInstance, set[TaskInstance]] = defaultdict(set)
    indegree = {instance: 0 for instance in instances}
    for plan in program.event_plans:
        source_name = plan.notify_relation.source.name
        target_name = plan.trigger_relation.target.name
        for pair in plan.scheduled_dependence.pairs:
            source = TaskInstance(source_name, pair.source)
            target = TaskInstance(target_name, pair.target)
            if source not in instances or target not in instances:
                continue
            if target not in outgoing[source]:
                outgoing[source].add(target)
                indegree[target] += 1
    frontier = deque(sorted(instance for instance, degree in indegree.items() if degree == 0))
    visited = 0
    while frontier:
        source = frontier.popleft()
        visited += 1
        for target in sorted(outgoing[source]):
            indegree[target] -= 1
            if indegree[target] == 0:
                frontier.append(target)
    if visited != len(instances):
        reasons.append("scheduled task dependencies contain an impossible cycle")
    if reasons:
        raise EventDataflowError(reasons)


def execute_event_dataflow(
    program: EventDataflowProgram,
    *,
    actions: Mapping[str, TaskAction],
    state: MutableMapping[str, object],
    scheduling_mode: EventSchedulingMode,
    generation: int = 0,
    random_seed: int | None = None,
) -> EventExecutionResult:
    """Execute a deterministic or randomized legal reference event schedule."""
    if generation < 0:
        raise ValueError("event generation must be non-negative")
    verify_event_dataflow_program(program)
    family_by_name = {family.name: family for family in program.task_families}
    missing_actions = set(family_by_name) - set(actions)
    if missing_actions:
        raise EventDataflowError((f"missing task actions for {sorted(missing_actions)}",))

    event_counts: dict[tuple[str, Coordinate, int], int] = {}
    ready_events: set[tuple[str, Coordinate, int]] = set()
    event_targets: dict[tuple[str, Coordinate, int], list[TaskInstance]] = defaultdict(list)
    task_events: dict[TaskInstance, set[tuple[str, Coordinate, int]]] = defaultdict(set)
    task_notifications: dict[TaskInstance, list[tuple[str, Coordinate, int]]] = defaultdict(list)
    trace: list[EventTraceEntry] = []
    step = 0
    for plan in program.event_plans:
        for coordinate, count in plan.initial_count.as_mapping().items():
            key = (plan.name, coordinate, generation)
            event_counts[key] = count
            if count == 0:
                ready_events.add(key)
                trace.append(EventTraceEntry(step, TraceKind.EVENT_READY, plan.name, coordinate, generation, 0))
                step += 1
        for pair in plan.trigger_relation.pairs:
            key = (plan.name, pair.source, generation)
            task = TaskInstance(plan.trigger_relation.target.name, pair.target)
            event_targets[key].append(task)
            task_events[task].add(key)
        for pair in plan.notify_relation.pairs:
            task = TaskInstance(plan.notify_relation.source.name, pair.source)
            task_notifications[task].append((plan.name, pair.target, generation))

    all_tasks = tuple(
        TaskInstance(family.name, coordinate) for family in program.task_families for coordinate in family.coordinates
    )
    static_priority = {task: index for index, task in enumerate(all_tasks)}
    runnable: list[TaskInstance] = []
    queued: set[TaskInstance] = set()
    executed: set[TaskInstance] = set()

    def maybe_enqueue(task: TaskInstance) -> None:
        nonlocal step
        if task in queued or task in executed:
            return
        if not task_events[task].issubset(ready_events):
            return
        runnable.append(task)
        queued.add(task)
        trace.append(EventTraceEntry(step, TraceKind.TASK_READY, task.family, task.coordinate, generation))
        step += 1

    for task in all_tasks:
        maybe_enqueue(task)

    rng = random.Random(random_seed)
    execution_order: list[TaskInstance] = []
    while runnable:
        if random_seed is not None:
            selected_index = rng.randrange(len(runnable))
        elif scheduling_mode is EventSchedulingMode.STATIC:
            selected_index = min(range(len(runnable)), key=lambda index: static_priority[runnable[index]])
        else:
            selected_index = 0
        task = runnable.pop(selected_index)
        queued.remove(task)
        actions[task.family](task.coordinate, state)
        executed.add(task)
        execution_order.append(task)
        trace.append(EventTraceEntry(step, TraceKind.TASK_EXECUTE, task.family, task.coordinate, generation))
        step += 1

        newly_ready: list[tuple[str, Coordinate, int]] = []
        for key in sorted(task_notifications[task]):
            remaining = event_counts[key]
            if remaining <= 0:
                raise EventDataflowError((f"duplicate or excess notification for event {key}",))
            remaining -= 1
            event_counts[key] = remaining
            trace.append(EventTraceEntry(step, TraceKind.NOTIFY, key[0], key[1], generation, remaining))
            step += 1
            if remaining == 0:
                if key in ready_events:
                    raise EventDataflowError((f"event {key} reached zero more than once",))
                ready_events.add(key)
                newly_ready.append(key)
                trace.append(EventTraceEntry(step, TraceKind.EVENT_READY, key[0], key[1], generation, 0))
                step += 1
        triggered = {consumer for event in newly_ready for consumer in event_targets[event]}
        for consumer in sorted(triggered):
            maybe_enqueue(consumer)

    if len(executed) != len(all_tasks):
        blocked = sorted(set(all_tasks) - executed)
        raise EventDataflowError((f"event schedule deadlocked with blocked tasks {blocked}",))
    if any(remaining != 0 for remaining in event_counts.values()):
        pending = {key: value for key, value in event_counts.items() if value != 0}
        raise EventDataflowError((f"execution completed with pending event counts {pending}",))
    return EventExecutionResult(dict(state), tuple(trace), tuple(execution_order), frozenset(ready_events))
