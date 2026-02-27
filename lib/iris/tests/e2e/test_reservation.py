# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""E2E tests for the reservation system.

Tests verify both sides of the reservation gate:
- Unsatisfiable reservations (requiring devices that don't exist) block
  scheduling while regular jobs on the same cluster proceed normally.
- Satisfiable reservations allow the job to schedule and complete.
- Dashboard renders reservation status correctly.
"""

import time

import pytest
from iris.cluster.types import ReservationEntry, ResourceSpec, gpu_device
from iris.rpc import cluster_pb2

from .conftest import (
    _is_noop_page,
    assert_visible,
    dashboard_goto,
    wait_for_dashboard_ready,
)
from .helpers import _quick

pytestmark = pytest.mark.e2e


def _unsatisfiable_reservation() -> list[ReservationEntry]:
    """A reservation entry requesting a GPU that doesn't exist in the local cluster."""
    return [ReservationEntry(resources=ResourceSpec(cpu=1, memory="1g", device=gpu_device("H100", 8)))]


def test_reservation_gates_scheduling_while_regular_jobs_proceed(cluster):
    """An unsatisfiable reservation blocks scheduling; regular jobs still complete.

    The reservation requests GPUs that don't exist in the local test cluster,
    so the claiming system never finds a matching worker and the gate stays
    closed. A regular job submitted to the same cluster schedules and completes
    normally, proving the gate is selective.
    """
    reserved_job = cluster.submit(
        _quick,
        "reservation-gated",
        reservation=_unsatisfiable_reservation(),
    )

    regular_job = cluster.submit(_quick, "no-reservation")

    regular_status = cluster.wait(regular_job, timeout=30)
    assert regular_status.state == cluster_pb2.JOB_STATE_SUCCEEDED

    reserved_status = cluster.status(reserved_job)
    assert reserved_status.state == cluster_pb2.JOB_STATE_PENDING

    assert reserved_status.HasField("reservation")
    assert reserved_status.reservation.total_entries == 1
    assert not reserved_status.reservation.satisfied

    cluster.kill(reserved_job)


def test_reservation_satisfied_allows_scheduling(cluster):
    """A satisfiable reservation allows the job to complete.

    The reservation requests CPU resources that the local cluster can provide.
    The claiming system finds a matching worker, satisfies the gate, and the
    job schedules and runs to completion.
    """
    reserved_job = cluster.submit(
        _quick,
        "reservation-satisfied",
        reservation=[ReservationEntry(resources=ResourceSpec(cpu=1, memory="1g"))],
    )

    status = cluster.wait(reserved_job, timeout=30)
    assert status.state == cluster_pb2.JOB_STATE_SUCCEEDED
    assert status.HasField("reservation")
    assert status.reservation.satisfied


def test_reservation_pending_reason_shows_fulfillment(cluster):
    """Pending reason for an unsatisfied reservation reports fulfillment progress."""
    reserved_job = cluster.submit(
        _quick,
        "reservation-pending-reason",
        reservation=[
            ReservationEntry(resources=ResourceSpec(cpu=1, memory="1g", device=gpu_device("H100", 8))),
            ReservationEntry(resources=ResourceSpec(cpu=1, memory="1g", device=gpu_device("H100", 8))),
        ],
    )

    time.sleep(1.0)

    status = cluster.status(reserved_job)
    assert status.state == cluster_pb2.JOB_STATE_PENDING
    assert "Waiting for reservation" in status.pending_reason
    assert "0/2 entries fulfilled" in status.pending_reason

    cluster.kill(reserved_job)


def test_reservation_dashboard_shows_provisioning_status(cluster, page, screenshot):
    """Job detail page renders reservation progress bar and entry table."""
    reserved_job = cluster.submit(
        _quick,
        "reservation-dashboard",
        reservation=[
            ReservationEntry(resources=ResourceSpec(cpu=1, memory="1g", device=gpu_device("H100", 8))),
            ReservationEntry(resources=ResourceSpec(cpu=1, memory="1g", device=gpu_device("H100", 8))),
        ],
    )

    time.sleep(1.0)

    dashboard_goto(page, f"{cluster.url}/job/{reserved_job.job_id.to_wire()}")
    wait_for_dashboard_ready(page)

    if not _is_noop_page(page):
        page.wait_for_function(
            "() => document.querySelector('.reservation-section') !== null",
            timeout=10000,
        )

    assert_visible(page, ".reservation-section")
    assert_visible(page, "text=Reservation")
    assert_visible(page, "text=workers provisioned")

    screenshot("job-detail-reservation-pending")

    cluster.kill(reserved_job)


def test_reservation_dashboard_absent_for_regular_job(cluster, page, screenshot):
    """Job detail page does NOT show a reservation section for regular jobs."""
    job = cluster.submit(_quick, "no-reservation-dashboard")
    cluster.wait(job, timeout=30)

    dashboard_goto(page, f"{cluster.url}/job/{job.job_id.to_wire()}")
    wait_for_dashboard_ready(page)

    if not _is_noop_page(page):
        page.wait_for_function(
            "() => document.querySelector('.info-grid') !== null",
            timeout=10000,
        )
        count = page.locator(".reservation-section").count()
        assert count == 0, "Reservation section should not appear for regular jobs"

    screenshot("job-detail-no-reservation")
