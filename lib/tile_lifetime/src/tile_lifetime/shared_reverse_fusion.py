# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Plan owner-computes fusion for reverse contractions with shared tile work."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from enum import StrEnum

from tile_lifetime.event_dataflow import Coordinate, TaskRelation

_RelationComponent = tuple[
    tuple[Coordinate, ...],
    tuple[Coordinate, ...],
    tuple[tuple[Coordinate, Coordinate], ...],
]


class OwnerComputeTraversal(StrEnum):
    """Deterministic endpoint order used by one fused component task."""

    SOURCE_MAJOR = "source_major"
    TARGET_MAJOR = "target_major"


class SharedReverseFusionDisposition(StrEnum):
    """Whether a relation component fits one owner-computes task."""

    FUSED_LOCAL = "fused_local"
    REJECTED_LOCAL_CAPACITY = "rejected_local_capacity"


@dataclass(frozen=True)
class OwnerComputeComponent:
    """One connected relation component and its local accumulator frontier."""

    source_coordinates: tuple[Coordinate, ...]
    target_coordinates: tuple[Coordinate, ...]
    edge_count: int
    source_major_peak_elements: int
    target_major_peak_elements: int
    selected_traversal: OwnerComputeTraversal
    selected_peak_elements: int


@dataclass(frozen=True)
class SharedReverseFusionPlan:
    """A bounded fusion decision for two output Folds sharing edge-local work."""

    relation: TaskRelation
    components: tuple[OwnerComputeComponent, ...]
    disposition: SharedReverseFusionDisposition
    source_accumulator_elements: int
    target_accumulator_elements: int
    transient_edge_elements: int
    accumulator_bytes_per_element: int
    local_capacity_bytes: int
    required_local_bytes: int
    baseline_contract_invocations: int
    fused_contract_invocations: int
    reasons: tuple[str, ...]

    @property
    def physical_contract_reduction(self) -> float:
        """Return the baseline-to-fused Contract invocation ratio."""
        return self.baseline_contract_invocations / self.fused_contract_invocations


def plan_shared_producer_reverse_fusion(
    relation: TaskRelation,
    *,
    source_accumulator_elements: int,
    target_accumulator_elements: int,
    transient_edge_elements: int,
    accumulator_bytes_per_element: int,
    local_capacity_bytes: int,
    baseline_contracts_per_edge: int,
    fused_contracts_per_edge: int,
) -> SharedReverseFusionPlan:
    """Fuse edge work only when connected output owners fit in local state.

    Each relation edge represents tile-local work shared by two output Folds.
    A no-atomic owner-computes task may fuse the work only when it owns a whole
    connected component. Splitting a connected component would cut at least one
    output Fold and require external partials or ordered cross-task updates.
    """
    positive_values = {
        "source accumulator elements": source_accumulator_elements,
        "target accumulator elements": target_accumulator_elements,
        "transient edge elements": transient_edge_elements,
        "accumulator bytes per element": accumulator_bytes_per_element,
        "local capacity bytes": local_capacity_bytes,
        "baseline Contracts per edge": baseline_contracts_per_edge,
        "fused Contracts per edge": fused_contracts_per_edge,
    }
    invalid = tuple(name for name, value in positive_values.items() if value <= 0)
    if invalid:
        raise ValueError(f"shared reverse fusion parameters must be positive: {', '.join(invalid)}")
    if not relation.pairs:
        raise ValueError("shared reverse fusion requires at least one relation edge")
    if fused_contracts_per_edge >= baseline_contracts_per_edge:
        raise ValueError("fused reverse traversal must remove at least one Contract per edge")

    components = tuple(
        _component_estimate(
            source_coordinates,
            target_coordinates,
            edge_pairs,
            source_accumulator_elements=source_accumulator_elements,
            target_accumulator_elements=target_accumulator_elements,
            transient_edge_elements=transient_edge_elements,
        )
        for source_coordinates, target_coordinates, edge_pairs in _relation_components(relation)
    )
    required_local_bytes = max(component.selected_peak_elements for component in components) * (
        accumulator_bytes_per_element
    )
    disposition = (
        SharedReverseFusionDisposition.FUSED_LOCAL
        if required_local_bytes <= local_capacity_bytes
        else SharedReverseFusionDisposition.REJECTED_LOCAL_CAPACITY
    )
    reasons: tuple[str, ...] = ()
    if disposition is SharedReverseFusionDisposition.REJECTED_LOCAL_CAPACITY:
        reasons = (
            "owner-computes fusion requires one task per connected relation component",
            (
                "the smaller source-major/target-major deterministic accumulator frontier requires "
                f"{required_local_bytes} bytes, "
                f"exceeding the {local_capacity_bytes}-byte local capacity"
            ),
            "splitting the component requires an external partial Fold, ordered cross-task updates, or atomics",
        )
    edge_count = len(relation.pairs)
    return SharedReverseFusionPlan(
        relation=relation,
        components=components,
        disposition=disposition,
        source_accumulator_elements=source_accumulator_elements,
        target_accumulator_elements=target_accumulator_elements,
        transient_edge_elements=transient_edge_elements,
        accumulator_bytes_per_element=accumulator_bytes_per_element,
        local_capacity_bytes=local_capacity_bytes,
        required_local_bytes=required_local_bytes,
        baseline_contract_invocations=edge_count * baseline_contracts_per_edge,
        fused_contract_invocations=edge_count * fused_contracts_per_edge,
        reasons=reasons,
    )


def _relation_components(
    relation: TaskRelation,
) -> tuple[_RelationComponent, ...]:
    source_neighbors: dict[Coordinate, set[Coordinate]] = defaultdict(set)
    target_neighbors: dict[Coordinate, set[Coordinate]] = defaultdict(set)
    for pair in relation.pairs:
        source_neighbors[pair.source].add(pair.target)
        target_neighbors[pair.target].add(pair.source)

    unseen_sources = set(source_neighbors)
    components = []
    while unseen_sources:
        first_source = min(unseen_sources)
        queue = deque(((True, first_source),))
        sources: set[Coordinate] = set()
        targets: set[Coordinate] = set()
        while queue:
            is_source, coordinate = queue.popleft()
            if is_source:
                if coordinate in sources:
                    continue
                sources.add(coordinate)
                unseen_sources.discard(coordinate)
                queue.extend((False, target) for target in source_neighbors[coordinate])
                continue
            if coordinate in targets:
                continue
            targets.add(coordinate)
            queue.extend((True, source) for source in target_neighbors[coordinate])
        edges = tuple(
            (pair.source, pair.target) for pair in relation.pairs if pair.source in sources and pair.target in targets
        )
        components.append((tuple(sorted(sources)), tuple(sorted(targets)), edges))
    return tuple(components)


def _component_estimate(
    source_coordinates: tuple[Coordinate, ...],
    target_coordinates: tuple[Coordinate, ...],
    edges: tuple[tuple[Coordinate, Coordinate], ...],
    *,
    source_accumulator_elements: int,
    target_accumulator_elements: int,
    transient_edge_elements: int,
) -> OwnerComputeComponent:
    source_major_peak = _major_order_peak(
        source_coordinates,
        edges,
        major_accumulator_elements=source_accumulator_elements,
        minor_accumulator_elements=target_accumulator_elements,
        transient_edge_elements=transient_edge_elements,
    )
    transposed_edges = tuple((target, source) for source, target in edges)
    target_major_peak = _major_order_peak(
        target_coordinates,
        transposed_edges,
        major_accumulator_elements=target_accumulator_elements,
        minor_accumulator_elements=source_accumulator_elements,
        transient_edge_elements=transient_edge_elements,
    )
    if source_major_peak <= target_major_peak:
        selected_traversal = OwnerComputeTraversal.SOURCE_MAJOR
        selected_peak = source_major_peak
    else:
        selected_traversal = OwnerComputeTraversal.TARGET_MAJOR
        selected_peak = target_major_peak
    return OwnerComputeComponent(
        source_coordinates=source_coordinates,
        target_coordinates=target_coordinates,
        edge_count=len(edges),
        source_major_peak_elements=source_major_peak,
        target_major_peak_elements=target_major_peak,
        selected_traversal=selected_traversal,
        selected_peak_elements=selected_peak,
    )


def _major_order_peak(
    major_coordinates: tuple[Coordinate, ...],
    edges: tuple[tuple[Coordinate, Coordinate], ...],
    *,
    major_accumulator_elements: int,
    minor_accumulator_elements: int,
    transient_edge_elements: int,
) -> int:
    minor_by_major: dict[Coordinate, tuple[Coordinate, ...]] = {}
    last_major_index: dict[Coordinate, int] = {}
    for major_index, major in enumerate(major_coordinates):
        minors = tuple(sorted(minor for edge_major, minor in edges if edge_major == major))
        minor_by_major[major] = minors
        for minor in minors:
            last_major_index[minor] = major_index

    peak = 0
    live_minors: set[Coordinate] = set()
    for major_index, major in enumerate(major_coordinates):
        for minor in minor_by_major[major]:
            live_minors.add(minor)
            peak = max(
                peak,
                major_accumulator_elements + len(live_minors) * minor_accumulator_elements + transient_edge_elements,
            )
            if last_major_index[minor] == major_index:
                live_minors.remove(minor)
    return peak
