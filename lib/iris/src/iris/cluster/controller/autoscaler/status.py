# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Autoscaler status and pending-diagnostic helpers."""

from collections import Counter, defaultdict
from collections.abc import Mapping
from dataclasses import dataclass, replace
from enum import StrEnum

from iris.backends.status import (
    AutoscalerStatus,
    DemandEntryStatus,
    GroupRoutingStatus,
    ResourceStatus,
    RoutingStatus,
    UnmetDemandStatus,
)
from iris.cluster.controller.autoscaler.models import DemandEntry, RoutingDecision
from iris.cluster.controller.autoscaler.routing import format_variants
from iris.cluster.controller.autoscaler.scaling_group import SliceLifecycleState
from iris.cluster.types import JobName, WorkerId, WorkerUsability
from iris.resources.execution import ResourceSpec, get_gpu_count, get_tpu_count


@dataclass(frozen=True)
class PendingHint:
    """Autoscaler-derived hint for a pending job."""

    message: str
    is_scaling_up: bool


class SliceCapacityStatus(StrEnum):
    """Placement status of a ready slice (slice-granular). A fully-booked healthy
    slice is ``IN_USE``, not free; non-ready slices carry no capacity status."""

    AVAILABLE = "available"  # all hosts healthy, no tasks: free to place on now
    IN_USE = "in_use"  # all hosts healthy, at least one task running
    IDLE = "idle"  # all hosts healthy, no tasks, idle past scale-down threshold
    DEGRADED = "degraded"  # no hosts, or any host unhealthy: not a placement target


def slice_capacity_status(
    *,
    is_ready: bool,
    host_count: int,
    healthy_hosts: int,
    running_tasks: int,
    idle: bool,
) -> str:
    """Classify a ready slice by placement readiness; "" for non-ready slices.

    DEGRADED if it has no hosts or any host is not HEALTHY (it cannot take a gang
    job even if some hosts are fine); otherwise split by occupancy.
    """
    if not is_ready:
        return ""
    if host_count == 0 or healthy_hosts < host_count:
        return SliceCapacityStatus.DEGRADED
    if running_tasks > 0:
        return SliceCapacityStatus.IN_USE
    if idle:
        return SliceCapacityStatus.IDLE
    return SliceCapacityStatus.AVAILABLE


def overlay_worker_usability(
    status: AutoscalerStatus,
    usability_by_id: dict[str, WorkerUsability],
    running: dict[WorkerId, set[JobName]],
) -> AutoscalerStatus:
    """Return status with current worker usability and slice capacity overlaid.

    worker_id/running_task_count are always set; usability/worker_healthy only when
    the worker is in the liveness roster (else left empty rather than mislabelled).
    Per ready slice we derive a slice-granular ``capacity_status`` from host health
    and occupancy, plus ``degraded_slot_count`` for detail.
    """
    groups = []
    for group in status.groups:
        slices = []
        for slice_info in group.slices:
            healthy_hosts = 0
            degraded_hosts = 0
            running_tasks = 0
            vms = []
            for vm in slice_info.vms:
                running_task_count = len(running.get(WorkerId(vm.vm_id), set()))
                running_tasks += running_task_count
                usability = usability_by_id.get(vm.vm_id)
                if usability is None:
                    vms.append(replace(vm, worker_id=vm.vm_id, running_task_count=running_task_count))
                    continue
                if usability is WorkerUsability.HEALTHY:
                    healthy_hosts += 1
                elif usability is WorkerUsability.DEGRADED:
                    degraded_hosts += 1
                vms.append(
                    replace(
                        vm,
                        worker_id=vm.vm_id,
                        running_task_count=running_task_count,
                        usability=str(usability),
                        worker_healthy=usability is not WorkerUsability.DEAD,
                    )
                )
            slices.append(
                replace(
                    slice_info,
                    vms=tuple(vms),
                    degraded_slot_count=degraded_hosts,
                    capacity_status=slice_capacity_status(
                        is_ready=slice_info.state == SliceLifecycleState.READY,
                        host_count=len(slice_info.vms),
                        healthy_hosts=healthy_hosts,
                        running_tasks=running_tasks,
                        idle=slice_info.idle,
                    ),
                )
            )
        groups.append(replace(group, slices=tuple(slices)))
    return replace(status, groups=tuple(groups))


def _resource_status(resources: ResourceSpec) -> ResourceStatus:
    return ResourceStatus(
        cpu_millicores=resources.cpu_millicores,
        memory_bytes=resources.memory,
        disk_bytes=resources.disk,
        gpu_count=get_gpu_count(resources.device) if resources.device else 0,
        tpu_count=get_tpu_count(resources.device) if resources.device else 0,
    )


def _entry_to_status(entry: DemandEntry) -> DemandEntryStatus:
    normalized = entry.normalized
    return DemandEntryStatus(
        task_ids=tuple(entry.task_ids),
        coschedule_group_id=entry.coschedule_group_id or "",
        device_type=normalized.device_type.value if normalized.device_type else "",
        device_variant=format_variants(normalized.device_variants),
        preemptible=bool(normalized.preemptible),
        resources=_resource_status(entry.resources),
    )


def routing_decision_to_status(
    decision: RoutingDecision,
    group_to_launch: Mapping[str, int],
) -> RoutingStatus:
    """Project an internal routing decision into native status.

    The ``group_to_launch`` argument is the capacity/rate-clamped launch count and
    overrides ``decision.group_to_launch`` (the raw demand-derived count).
    """
    launch_counts = dict(group_to_launch)

    return RoutingStatus(
        group_to_launch=launch_counts,
        group_reasons=dict(decision.group_reasons),
        routed_entries={
            name: tuple(_entry_to_status(entry) for entry in entries)
            for name, entries in decision.routed_entries.items()
        },
        unmet_entries=tuple(
            UnmetDemandStatus(entry=_entry_to_status(unmet.entry), reason=unmet.reason)
            for unmet in decision.unmet_entries
        ),
        group_statuses=tuple(
            GroupRoutingStatus(
                group=status.group,
                priority=status.priority,
                assigned=status.assigned,
                launch=launch_counts.get(status.group, 0),
                decision=status.decision,
                reason=status.reason,
            )
            for status in decision.group_statuses
        ),
    )


def _task_id_to_job_id(task_id: str) -> str | None:
    """Return parent job wire id for a task id, or None for invalid input."""

    try:
        task_name = JobName.from_wire(task_id)
    except ValueError:
        return None
    parent = task_name.parent
    if parent is None:
        return None
    return parent.to_wire()


def _group_status_detail(routing: RoutingStatus, group_name: str) -> str:
    """Extract decision and reason for a given group from routing status."""

    for group_status in routing.group_statuses:
        if group_status.group != group_name:
            continue
        if group_status.reason:
            return f"{group_status.decision}: {group_status.reason}"
        return group_status.decision
    return ""


def build_job_pending_hints(routing: RoutingStatus | None) -> dict[str, PendingHint]:
    """Build autoscaler pending hints keyed by job id."""

    if routing is None:
        return {}

    routed_counts_by_job: dict[str, Counter[str]] = defaultdict(Counter)
    unmet_reasons_by_job: dict[str, Counter[str]] = defaultdict(Counter)

    for group_name, entry_list in routing.routed_entries.items():
        for entry in entry_list:
            for task_id in entry.task_ids:
                job_id = _task_id_to_job_id(task_id)
                if job_id is not None:
                    routed_counts_by_job[job_id][group_name] += 1

    for unmet in routing.unmet_entries:
        reason = unmet.reason or "unknown"
        for task_id in unmet.entry.task_ids:
            job_id = _task_id_to_job_id(task_id)
            if job_id is not None:
                unmet_reasons_by_job[job_id][reason] += 1

    hints: dict[str, PendingHint] = {}

    for job_id, group_counts in routed_counts_by_job.items():
        ranked_groups = sorted(group_counts.items(), key=lambda item: (-item[1], item[0]))
        launch_groups = [(name, count) for name, count in ranked_groups if routing.group_to_launch.get(name, 0) > 0]

        if launch_groups:
            group_name, _ = launch_groups[0]
            launch_count = routing.group_to_launch.get(group_name, 0)
            hints[job_id] = PendingHint(
                message=f"Waiting for worker scale-up in scale group '{group_name}' ({launch_count} slice(s) requested)",
                is_scaling_up=True,
            )
            continue

        primary_group, _ = ranked_groups[0]
        status_detail = _group_status_detail(routing, primary_group)
        suffix = f" ({status_detail})" if status_detail else ""
        hints[job_id] = PendingHint(
            message=f"Waiting for workers in scale group '{primary_group}' to become ready{suffix}",
            is_scaling_up=False,
        )

    for job_id, reason_counts in unmet_reasons_by_job.items():
        if job_id in hints:
            continue
        reason, _ = reason_counts.most_common(1)[0]
        hints[job_id] = PendingHint(
            message=f"Unsatisfied autoscaler demand: {reason}",
            is_scaling_up=False,
        )

    return hints
