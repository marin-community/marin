# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Autoscaler status and pending-diagnostic helpers."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping
from dataclasses import dataclass

from iris.cluster.controller.autoscaler.models import DemandEntry, RoutingDecision
from iris.cluster.controller.autoscaler.routing import _format_variants
from iris.cluster.types import JobName
from iris.rpc import job_pb2, vm_pb2


@dataclass(frozen=True)
class PendingHint:
    """Autoscaler-derived hint for a pending job."""

    message: str
    is_scaling_up: bool


def _resource_spec_proto(resources: job_pb2.ResourceSpecProto) -> vm_pb2.ResourceSpec:
    gpu_count = 0
    tpu_count = 0
    if resources.HasField("device"):
        if resources.device.HasField("gpu"):
            gpu_count = resources.device.gpu.count or 1
        if resources.device.HasField("tpu"):
            tpu_count = resources.device.tpu.count or 0
    return vm_pb2.ResourceSpec(
        cpu_millicores=resources.cpu_millicores,
        memory_bytes=resources.memory_bytes,
        disk_bytes=resources.disk_bytes,
        gpu_count=gpu_count,
        tpu_count=tpu_count,
    )


def _entry_to_proto(entry: DemandEntry) -> vm_pb2.DemandEntryStatus:
    normalized = entry.normalized
    return vm_pb2.DemandEntryStatus(
        task_ids=entry.task_ids,
        coschedule_group_id=entry.coschedule_group_id or "",
        device_type=normalized.device_type.value if normalized.device_type else "",
        device_variant=_format_variants(normalized.device_variants),
        preemptible=bool(normalized.preemptible),
        resources=_resource_spec_proto(entry.resources),
    )


def routing_decision_to_proto(
    decision: RoutingDecision,
    group_to_launch: Mapping[str, int] | None = None,
) -> vm_pb2.RoutingDecision:
    """Convert an internal routing decision into the status proto."""

    routed_entries = {
        name: vm_pb2.DemandEntryStatusList(entries=[_entry_to_proto(entry) for entry in entries])
        for name, entries in decision.routed_entries.items()
    }
    unmet_entries = [
        vm_pb2.UnmetDemand(entry=_entry_to_proto(unmet.entry), reason=unmet.reason) for unmet in decision.unmet_entries
    ]
    launch_counts = dict(decision.group_to_launch if group_to_launch is None else group_to_launch)

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
