# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""VM lifecycle tests: quota exhaustion, stuck init, preemption.

These tests use GcpWorkerProvider with InMemoryGcpService(DRY_RUN) and don't
need a full cluster fixture.
"""

import unittest.mock

import pytest
from iris.cluster.config import GcpPlatformConfig, GcpSliceConfig, SliceConfig, WorkerConfig
from iris.cluster.platforms.gcp.fake import InMemoryGcpService
from iris.cluster.platforms.gcp.workers import GcpWorkerProvider
from iris.cluster.platforms.types import CloudSliceState, QuotaExhaustedError
from iris.cluster.service_mode import ServiceMode
from iris.cluster.types import AcceleratorType


def _make_gcp_config(zone: str = "us-central2-b") -> GcpPlatformConfig:
    return GcpPlatformConfig(project_id="test-project", zones=[zone])


def _make_slice_config(name: str = "test", zone: str = "us-central2-b") -> SliceConfig:
    return SliceConfig(
        name_prefix=name,
        accelerator_type=AcceleratorType.TPU,
        accelerator_variant="v5litepod-8",
        gcp=GcpSliceConfig(zone=zone, runtime_version="v2-alpha-tpuv5-lite"),
    )


def test_tpu_quota_exceeded_retry():
    """TPU creation fails with quota exceeded, retries successfully after clearing."""
    service = InMemoryGcpService(mode=ServiceMode.DRY_RUN, project_id="test-project")
    service.set_zone_quota("us-central2-b", 0)
    platform = GcpWorkerProvider(_make_gcp_config(), label_prefix="test", worker_port=10001, gcp_service=service)

    with pytest.raises(QuotaExhaustedError):
        platform.create_slice(_make_slice_config())

    # Clear quota restriction and retry
    service.set_zone_quota("us-central2-b", 100)
    handle = platform.create_slice(_make_slice_config())

    # Advance TPU to READY so describe() returns workers
    tpu_name = handle.slice_id
    service.advance_tpu_state(tpu_name, "us-central2-b", "READY")

    status = handle.describe()
    assert status.state == CloudSliceState.READY
    assert status.worker_count > 0


def test_tpu_init_stuck():
    """A monitored TPU whose workers never become healthy stays non-READY (CREATING)."""
    service = InMemoryGcpService(mode=ServiceMode.DRY_RUN, project_id="test-project")
    platform = GcpWorkerProvider(_make_gcp_config(), label_prefix="test", worker_port=10001, gcp_service=service)

    # Bootstrap-monitored slice (worker_config) with the bootstrap thread
    # suppressed: bootstrap never reports healthy, so the slice must not present
    # as READY while the node sits in CREATING.
    wc = WorkerConfig(docker_image="img:latest", port=10001, controller_address="c:10000", cache_dir="/var/cache/iris")
    with unittest.mock.patch("iris.cluster.platforms.gcp.workers.threading.Thread"):
        handle = platform.create_slice(_make_slice_config("stuck"), worker_config=wc)

    status = handle.describe()
    assert status.state == CloudSliceState.CREATING


def test_tpu_preempted():
    """TPU reaches READY, then termination destroys the slice."""
    service = InMemoryGcpService(mode=ServiceMode.DRY_RUN, project_id="test-project")
    platform = GcpWorkerProvider(_make_gcp_config(), label_prefix="test", worker_port=10001, gcp_service=service)
    handle = platform.create_slice(_make_slice_config("preempt"))

    # Advance to READY
    service.advance_tpu_state(handle.slice_id, "us-central2-b", "READY")

    status = handle.describe()
    assert status.state == CloudSliceState.READY
    assert len(status.workers) > 0
    assert all(vm.internal_address for vm in status.workers)

    handle.terminate()

    # After deletion, the TPU is removed from in-memory state; describe returns UNKNOWN
    status = handle.describe()
    assert status.state == CloudSliceState.UNKNOWN
