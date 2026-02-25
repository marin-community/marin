# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Helpers for composing autoscaler-aware pending diagnostics.

This module keeps scheduler and autoscaler concerns separate:
- Scheduler explains *why* no worker can be assigned right now.
- Autoscaler explains whether capacity is being created for pending demand.

Controller service can compose these signals into user-facing diagnostics
without coupling scheduler internals to autoscaler routing.
"""

from __future__ import annotations

from collections import Counter, defaultdict

from iris.cluster.types import JobName
from iris.rpc import vm_pb2

ACTIVE_HINT_PREFIXES = (
    "Waiting for worker scale-up in scale group",
    "Waiting for workers in scale group",
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


def build_job_pending_hints(routing: vm_pb2.RoutingDecision | None) -> dict[str, str]:
    """Build autoscaler pending hints keyed by job id.

    Args:
        routing: Latest autoscaler routing decision, if available.

    Returns:
        Map of job_id wire format -> concise pending diagnostic message.
    """
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

    hints: dict[str, str] = {}

    for job_id, group_counts in routed_counts_by_job.items():
        ranked_groups = sorted(group_counts.items(), key=lambda x: (-x[1], x[0]))
        launch_groups = [(name, count) for name, count in ranked_groups if routing.group_to_launch.get(name, 0) > 0]

        if launch_groups:
            group_name, _ = launch_groups[0]
            launch_count = routing.group_to_launch.get(group_name, 0)
            hints[job_id] = (
                f"Waiting for worker scale-up in scale group '{group_name}' ({launch_count} slice(s) requested)"
            )
            continue

        # Demand is routed but no new slices requested right now (for example
        # existing in-flight slices are expected to satisfy demand).
        primary_group, _ = ranked_groups[0]
        hints[job_id] = f"Waiting for workers in scale group '{primary_group}' to become ready"

    for job_id, reason_counts in unmet_reasons_by_job.items():
        if job_id in hints:
            continue
        reason, _ = reason_counts.most_common(1)[0]
        hints[job_id] = f"Unsatisfied autoscaler demand: {reason}"

    return hints


def is_active_scale_up_hint(hint: str) -> bool:
    """Whether a hint indicates the autoscaler is actively provisioning capacity."""
    return any(hint.startswith(prefix) for prefix in ACTIVE_HINT_PREFIXES)
