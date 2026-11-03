"""Deterministic logical-teacher routing for distillation trajectories."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass

from marinskyrl.distillation import DistillationPlan, TeacherEvidenceKind
from skyrl_train.batch_sampling import RowOwnership, filter_trajectory_batch
from skyrl_train.trajectory_runners.types import TrajectoryBatch


ROUTE_ROWS_METRIC_PREFIX = "distillation/route_rows/"


def route_row_metrics(routes: Iterable[TeacherRoute]) -> dict[str, float]:
    """Count the trained trajectories each logical route contributed to one optimizer step."""
    return {
        f"{ROUTE_ROWS_METRIC_PREFIX}{route_id}": float(count)
        for route_id, count in Counter(r.route_id for r in routes).items()
    }


@dataclass(frozen=True)
class TeacherRoute:
    """Versioned logical route resolved from immutable trajectory metadata."""

    route_id: str
    teacher_id: str
    objective_id: str
    weight: float
    plan_version: str


class PlanTeacherRouter:
    """Resolve explicit route keys without inspecting or classifying prompt text."""

    def __init__(self, plan: DistillationPlan) -> None:
        if not plan.teachers:
            raise ValueError("teacher routing requires at least one teacher")
        evidence_kinds = {teacher.evidence for teacher in plan.teachers}
        if len(evidence_kinds) != 1:
            raise ValueError("all teachers in one routing plan must provide the same evidence kind")
        if len({route.key for route in plan.routing.routes}) != len(plan.routing.routes):
            raise ValueError("teacher routing plan contains duplicate route keys")
        self._routes = {
            route.key: TeacherRoute(
                route_id=route.key,
                teacher_id=route.teacher_id,
                objective_id=plan.objective.value,
                weight=route.weight,
                plan_version=plan.routing.revision,
            )
            for route in plan.routing.routes
        }
        self.coefficient = plan.coefficient
        self.evidence = next(iter(evidence_kinds))
        self.plan_version = plan.routing.revision

    def resolve(self, route_key: str) -> TeacherRoute:
        if not isinstance(route_key, str) or not route_key.strip():
            raise ValueError("teacher route key must be a non-empty string")
        try:
            return self._routes[route_key]
        except KeyError as error:
            raise ValueError(f"unknown teacher route {route_key!r} in plan {self.plan_version!r}") from error


@dataclass(frozen=True)
class RoutedTrajectoryPartition:
    """Read-only borrowed rows for one teacher plus original batch coordinates."""

    teacher_id: str
    original_indices: tuple[int, ...]
    routes: tuple[TeacherRoute, ...]
    trajectory_batch: TrajectoryBatch


@dataclass(frozen=True)
class RoutedTrajectoryBatch:
    """A mixed batch partitioned by logical teacher with exact row provenance."""

    trajectory_ids: tuple[str, ...]
    response_lengths: tuple[int, ...]
    routes: tuple[TeacherRoute, ...]
    partitions: tuple[RoutedTrajectoryPartition, ...]
    coefficient: float
    evidence: TeacherEvidenceKind
    plan_version: str


def route_trajectory_batch(
    trajectory_batch: TrajectoryBatch,
    *,
    route_keys: tuple[str, ...],
    router: PlanTeacherRouter,
) -> RoutedTrajectoryBatch:
    """Resolve and partition a batch, preserving enough provenance to restore row order."""
    trajectory_ids = trajectory_batch.get("trajectory_ids")
    if trajectory_ids is None:
        raise ValueError("teacher routing requires stable trajectory_ids")
    row_count = len(trajectory_batch["response_ids"])
    aligned_fields = {
        "trajectory_ids": len(trajectory_ids),
        "route_keys": len(route_keys),
    }
    if row_count == 0:
        raise ValueError("teacher routing requires at least one trajectory")
    if any(length != row_count for length in aligned_fields.values()):
        raise ValueError(f"teacher routing fields must align with {row_count} trajectories: {aligned_fields}")

    resolved_routes = tuple(router.resolve(route_key) for route_key in route_keys)
    indices_by_teacher: dict[str, list[int]] = {}
    for index, route in enumerate(resolved_routes):
        indices_by_teacher.setdefault(route.teacher_id, []).append(index)

    partitions = []
    for teacher_id, indices in indices_by_teacher.items():
        original_indices = tuple(indices)
        partitions.append(
            RoutedTrajectoryPartition(
                teacher_id=teacher_id,
                original_indices=original_indices,
                routes=tuple(resolved_routes[index] for index in indices),
                trajectory_batch=filter_trajectory_batch(
                    trajectory_batch, indices, row_ownership=RowOwnership.BORROWED
                ),
            )
        )

    serialized_trajectory_ids = tuple(trajectory_id.to_string() for trajectory_id in trajectory_ids)
    if len(set(serialized_trajectory_ids)) != row_count:
        raise ValueError("teacher routing requires unique trajectory_ids")
    return RoutedTrajectoryBatch(
        trajectory_ids=serialized_trajectory_ids,
        response_lengths=tuple(len(response) for response in trajectory_batch["response_ids"]),
        routes=resolved_routes,
        partitions=tuple(partitions),
        coefficient=router.coefficient,
        evidence=router.evidence,
        plan_version=router.plan_version,
    )
