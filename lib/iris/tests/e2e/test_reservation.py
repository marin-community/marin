# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""E2E tests for the reservation system.

Tests prove that the scheduling gate actually works: a job with a reservation
stays PENDING while regular jobs on the same cluster schedule and complete,
the pending_reason diagnostic reports fulfillment progress, and the dashboard
renders the reservation section.
"""

import time

import pytest
from iris.cluster.types import ReservationEntry, ResourceSpec
from iris.rpc import cluster_pb2

from .conftest import (
    _is_noop_page,
    assert_visible,
    dashboard_goto,
    wait_for_dashboard_ready,
)
from .helpers import _quick

pytestmark = pytest.mark.e2e


def test_reservation_gates_scheduling_while_regular_jobs_proceed(cluster):
    """A job with a reservation stays PENDING while regular jobs complete.

    This is the core behavioral test for the reservation gate. In the local
    platform, workers don't carry reservation tags, so the gate is never
    satisfied. A regular job submitted to the same cluster schedules and
    completes normally, proving the gate is selective.
    """
    # Submit a job with a reservation — should be blocked
    reserved_job = cluster.submit(
        _quick,
        "reservation-gated",
        reservation=[ReservationEntry(resources=ResourceSpec(cpu=1, memory="1g"))],
    )

    # Submit a regular job — should schedule normally
    regular_job = cluster.submit(_quick, "no-reservation")

    # The regular job should complete
    regular_status = cluster.wait(regular_job, timeout=30)
    assert regular_status.state == cluster_pb2.JOB_STATE_SUCCEEDED

    # The reserved job should still be PENDING (gate unsatisfied)
    reserved_status = cluster.status(reserved_job)
    assert reserved_status.state == cluster_pb2.JOB_STATE_PENDING

    # The reserved job's status should include reservation info
    assert reserved_status.HasField("reservation")
    assert reserved_status.reservation.total_entries == 1
    assert not reserved_status.reservation.satisfied

    # Clean up
    cluster.kill(reserved_job)


def test_reservation_pending_reason_shows_fulfillment(cluster):
    """Pending reason for a reservation-gated job reports fulfillment progress."""
    reserved_job = cluster.submit(
        _quick,
        "reservation-pending-reason",
        reservation=[
            ReservationEntry(resources=ResourceSpec(cpu=1, memory="1g")),
            ReservationEntry(resources=ResourceSpec(cpu=1, memory="1g")),
        ],
    )

    # Give the controller time to process
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
            ReservationEntry(resources=ResourceSpec(cpu=1, memory="1g")),
            ReservationEntry(resources=ResourceSpec(cpu=1, memory="1g")),
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
