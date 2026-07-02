# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Autoscaler status and pending-diagnostic helpers."""

from collections import Counter, defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum

from iris.cluster.controller.autoscaler.models import DemandEntry, RoutingDecision
from iris.cluster.controller.autoscaler.routing import format_variants
from iris.cluster.controller.autoscaler.scaling_group import SliceLifecycleState
from iris.cluster.types import JobName, WorkerId, WorkerUsability, get_gpu_count, get_tpu_count
from iris.rpc import job_pb2, vm_pb2


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
    status: vm_pb2.AutoscalerStatus,
    usability_by_id: dict[str, WorkerUsability],
    running: dict[WorkerId, set[JobName]],
) -> None:
    """Overlay per-VM usability and stamp each ready slice's capacity status in place.

    worker_id/running_task_count are always set; usability/worker_healthy only when
    the worker is in the liveness roster (else left empty rather than mislabelled).
    Per ready slice we derive a slice-granular ``capacity_status`` from host health
    and occupancy, plus ``degraded_slot_count`` for detail.
    """
    for group in status.groups:
        for slice_info in group.slices:
            healthy_hosts = 0
            degraded_hosts = 0
            running_tasks = 0
            for vm in slice_info.vms:
                vm.worker_id = vm.vm_id
                vm.running_task_count = len(running.get(WorkerId(vm.vm_id), set()))
                running_tasks += vm.running_task_count
                usability = usability_by_id.get(vm.vm_id)
                if usability is None:
                    continue
                vm.usability = str(usability)
                vm.worker_healthy = usability is not WorkerUsability.DEAD
                if usability is WorkerUsability.HEALTHY:
                    healthy_hosts += 1
                elif usability is WorkerUsability.DEGRADED:
                    degraded_hosts += 1
            slice_info.degraded_slot_count = degraded_hosts
            slice_info.capacity_status = slice_capacity_status(
                is_ready=slice_info.state == SliceLifecycleState.READY,
                host_count=len(slice_info.vms),
                healthy_hosts=healthy_hosts,
                running_tasks=running_tasks,
                idle=slice_info.idle,
            )


def _resource_spec_proto(resources: job_pb2.ResourceSpecProto) -> vm_pb2.ResourceSpec:
    return vm_pb2.ResourceSpec(
        cpu_millicores=resources.cpu_millicores,
        memory_bytes=resources.memory_bytes,
        disk_bytes=resources.disk_bytes,
        gpu_count=get_gpu_count(resources.device),
        tpu_count=get_tpu_count(resources.device),
    )


def _entry_to_proto(entry: DemandEntry) -> vm_pb2.DemandEntryStatus:
    normalized = entry.normalized
    return vm_pb2.DemandEntryStatus(
        task_ids=entry.task_ids,
        coschedule_group_id=entry.coschedule_group_id or "",
        device_type=normalized.device_type.value if normalized.device_type else "",
        device_variant=format_variants(normalized.device_variants),
        preemptible=bool(normalized.preemptible),
        resources=_resource_spec_proto(entry.resources),
    )


def routing_decision_to_proto(
    decision: RoutingDecision,
    group_to_launch: Mapping[str, int],
) -> vm_pb2.RoutingDecision:
    """Convert an internal routing decision into the status proto.

    The ``group_to_launch`` argument is the capacity/rate-clamped launch count and
    overrides ``decision.group_to_launch`` (the raw demand-derived count) in the proto.
    """

    routed_entries = {
        name: vm_pb2.DemandEntryStatusList(entries=[_entry_to_proto(entry) for entry in entries])
        for name, entries in decision.routed_entries.items()
    }
    unmet_entries = [
        vm_pb2.UnmetDemand(entry=_entry_to_proto(unmet.entry), reason=unmet.reason) for unmet in decision.unmet_entries
    ]
    launch_counts = dict(group_to_launch)

    return vm_pb2.RoutingDecision(
        group_to_launch=launch_counts,
        group_reasons=decision.group_reasons,
        routed_entries=routed_entries,
        unmet_entries=unmet_entries,
        group_statuses=[
            vm_pb2.GroupRoutingStatus(
                group=status.group,
                priority=status.priority,
                assigned=status.assigned,
                launch=launch_counts.get(status.group, 0),
                decision=status.decision,
                reason=status.reason,
            )
            for status in decision.group_statuses
        ],
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


def _group_status_detail(routing: vm_pb2.RoutingDecision, group_name: str) -> str:
    """Extract decision and reason for a given group from routing status."""

    for group_status in routing.group_statuses:
        if group_status.group != group_name:
            continue
        if group_status.reason:
            return f"{group_status.decision}: {group_status.reason}"
        return group_status.decision
    return ""


def build_job_pending_hints(routing: vm_pb2.RoutingDecision | None) -> dict[str, PendingHint]:
    """Build autoscaler pending hints keyed by job id."""

    if routing is None:
        return {}

    routed_counts_by_job: dict[str, Counter[str]] = defaultdict(Counter)
    unmet_reasons_by_job: dict[str, Counter[str]] = defaultdict(Counter)

    for group_name, entry_list in routing.routed_entries.items():
        for entry in entry_list.entries:
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
