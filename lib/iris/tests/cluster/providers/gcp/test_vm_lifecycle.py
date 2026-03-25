# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""VM lifecycle tests: quota exhaustion, stuck init, preemption.

These tests use GcpWorkerProvider with InMemoryGcpService(DRY_RUN) and don't
need a full cluster fixture.
"""

import pytest
from iris.cluster.providers.gcp.fake import InMemoryGcpService
from iris.cluster.providers.gcp.workers import GcpWorkerProvider
from iris.cluster.providers.types import CloudSliceState, QuotaExhaustedError
from iris.cluster.service_mode import ServiceMode
from iris.rpc import config_pb2


def _make_gcp_config(zone: str = "us-central2-b") -> config_pb2.GcpPlatformConfig:
    return config_pb2.GcpPlatformConfig(project_id="test-project", zones=[zone])


def _make_slice_config(name: str = "test", zone: str = "us-central2-b") -> config_pb2.SliceConfig:
    return config_pb2.SliceConfig(
        name_prefix=name,
        accelerator_type=config_pb2.ACCELERATOR_TYPE_TPU,
        accelerator_variant="v5litepod-8",
        gcp=config_pb2.GcpSliceConfig(zone=zone, runtime_version="v2-alpha-tpuv5-lite"),
    )


def test_tpu_quota_exceeded_retry():
    """TPU creation fails with quota exceeded, retries successfully after clearing."""
    service = InMemoryGcpService(mode=ServiceMode.DRY_RUN, project_id="test-project")
    service.set_zone_quota("us-central2-b", 0)
    platform = GcpWorkerProvider(_make_gcp_config(), label_prefix="test", gcp_service=service)

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
    """TPU never becomes READY (stuck in CREATING)."""
    service = InMemoryGcpService(mode=ServiceMode.DRY_RUN, project_id="test-project")
    platform = GcpWorkerProvider(_make_gcp_config(), label_prefix="test", gcp_service=service)
    handle = platform.create_slice(_make_slice_config("stuck"))

    # TPU stays in CREATING (we never advance it to READY)
    status = handle.describe()
    assert status.state == CloudSliceState.CREATING


def test_tpu_preempted():
    """TPU reaches READY, then termination destroys the slice."""
    service = InMemoryGcpService(mode=ServiceMode.DRY_RUN, project_id="test-project")
    platform = GcpWorkerProvider(_make_gcp_config(), label_prefix="test", gcp_service=service)
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
