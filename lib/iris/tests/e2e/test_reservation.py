# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""E2E tests for the reservation system.

Tests verify that unsatisfiable reservations (requiring devices that don't
exist) block scheduling while regular jobs on the same cluster proceed normally.
"""

import pytest
from iris.cluster.types import ReservationEntry, ResourceSpec, gpu_device
from iris.rpc import cluster_pb2

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

    cluster.kill(reserved_job)
