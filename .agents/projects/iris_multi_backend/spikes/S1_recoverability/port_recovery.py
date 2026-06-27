# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""S1 spike: the port-recovery hole, demonstrated against the REAL adopt() path.

design.md / research.md "What surprised us": port reservations have no substrate
footprint today. ``TaskAttempt.adopt`` (worker/task_attempt.py:303) rebuilds with
``ports={}`` and never re-reserves in ``PortAllocator``, and ``DiscoveredContainer``
has no port field -- so after a worker restart the recovered attempt's ports are
forgotten and the allocator will hand the same ports to a new task (double-alloc).

This module exercises the production classes directly (no model) to prove the
hole, then shows the fix direction working: stamp ports on the substrate object
and re-reserve them in ``PortAllocator`` on adopt.
"""

from __future__ import annotations

from dataclasses import dataclass

from iris.cluster.runtime.types import DiscoveredContainer, ExecutionStage
from iris.cluster.worker.port_allocator import PortAllocator
from iris.cluster.worker.task_attempt import TaskAttempt


@dataclass
class _FakeHandle:
    """Minimal ContainerHandle stand-in. adopt() only stores the handle."""

    container_id: str = "container-abc"


def _discovered(attempt_uid: str, worker_id: str) -> DiscoveredContainer:
    return DiscoveredContainer(
        container_id="container-abc",
        task_id="/alice/job/0",
        attempt_id=0,
        attempt_uid=attempt_uid,
        job_id="/alice/job",
        worker_id=worker_id,
        phase=ExecutionStage.RUN,
        running=True,
        exit_code=None,
        started_at="2026-01-01T00:00:00Z",
        workdir_host_path="/var/lib/iris/tasks/t0/app",
    )


@dataclass(frozen=True)
class AdoptResult:
    attempt_ports: dict[str, int]
    allocator_reserved: tuple[int, ...]


def adopt_today(stamped_ports: dict[str, int]) -> AdoptResult:
    """Today's behaviour: a worker restarts and adopts via the REAL adopt().

    ``stamped_ports`` is what the original task had allocated; it is passed only
    to show it is *not* recovered -- DiscoveredContainer cannot carry it, so the
    real adopt() rebuilds ports={} and the fresh allocator stays empty.
    """
    allocator = PortAllocator()
    attempt = TaskAttempt.adopt(
        discovered=_discovered("uid-1", "w1"),
        container_handle=_FakeHandle(),
        log_client=None,
        port_allocator=allocator,
    )
    return AdoptResult(attempt_ports=dict(attempt.ports), allocator_reserved=tuple(sorted(allocator._allocated)))


def adopt_with_recovery(stamped_ports: dict[str, int]) -> AdoptResult:
    """Fix direction: ports are stamped on the substrate and re-reserved on adopt.

    We still call the real adopt() (so the production construction path runs),
    then apply what the fix must do: restore the stamped ports onto the attempt
    and re-reserve them in the allocator before any new work is scheduled.
    """
    allocator = PortAllocator()
    attempt = TaskAttempt.adopt(
        discovered=_discovered("uid-1", "w1"),
        container_handle=_FakeHandle(),
        log_client=None,
        port_allocator=allocator,
    )
    # The fix: DiscoveredContainer gains a ``ports`` field (stamped from Docker
    # labels), adopt() restores it onto the attempt, and PortAllocator.reserve()
    # re-marks the exact numbers as taken.
    attempt.ports = dict(stamped_ports)
    with allocator._lock:  # noqa: SLF001 -- stand-in for PortAllocator.reserve()
        allocator._allocated.update(stamped_ports.values())
    return AdoptResult(attempt_ports=dict(attempt.ports), allocator_reserved=tuple(sorted(allocator._allocated)))
