# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from iris.cluster.controller.pending_diagnostics import build_job_pending_hints
from iris.cluster.types import JobName
from iris.rpc import vm_pb2


def _task(job: str, idx: int) -> str:
    return JobName.root("test-user", job).task(idx).to_wire()


def test_build_job_pending_hints_reports_scale_up_group() -> None:
    routing = vm_pb2.RoutingDecision(
        group_to_launch={"tpu_v5e_32": 1},
        routed_entries={
            "tpu_v5e_32": vm_pb2.DemandEntryStatusList(entries=[vm_pb2.DemandEntryStatus(task_ids=[_task("job-a", 0)])])
        },
    )

    hints = build_job_pending_hints(routing)

    assert hints[JobName.root("test-user", "job-a").to_wire()] == (
        "Waiting for worker scale-up in scale group 'tpu_v5e_32' (1 slice(s) requested)"
    )


def test_build_job_pending_hints_reports_waiting_ready_when_no_launch() -> None:
    routing = vm_pb2.RoutingDecision(
        group_to_launch={"tpu_v5e_32": 0},
        routed_entries={
            "tpu_v5e_32": vm_pb2.DemandEntryStatusList(
                entries=[vm_pb2.DemandEntryStatus(task_ids=[_task("job-b", 0), _task("job-b", 1)])]
            )
        },
    )

    hints = build_job_pending_hints(routing)

    assert hints[JobName.root("test-user", "job-b").to_wire()] == (
        "Waiting for workers in scale group 'tpu_v5e_32' to become ready"
    )


def test_build_job_pending_hints_reports_unmet_when_not_routed() -> None:
    routing = vm_pb2.RoutingDecision(
        unmet_entries=[
            vm_pb2.UnmetDemand(
                entry=vm_pb2.DemandEntryStatus(task_ids=[_task("job-c", 0)]),
                reason="no_matching_group: need device=tpu:v5p-8",
            )
        ]
    )

    hints = build_job_pending_hints(routing)

    assert hints[JobName.root("test-user", "job-c").to_wire()] == (
        "Unsatisfied autoscaler demand: no_matching_group: need device=tpu:v5p-8"
    )
