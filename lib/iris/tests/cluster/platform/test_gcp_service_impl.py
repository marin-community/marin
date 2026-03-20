# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for GcpServiceImpl DRY_RUN and LOCAL modes.

Covers validation, in-memory state, failure injection, and quota enforcement.
"""

from __future__ import annotations

import pytest

from iris.cluster.platform.base import PlatformError, QuotaExhaustedError, ResourceNotFoundError
from iris.cluster.platform.gcp_service import TpuCreateRequest, VmCreateRequest
from iris.cluster.platform.gcp_service_impl import GcpServiceImpl
from iris.cluster.service_mode import ServiceMode


@pytest.fixture
def svc() -> GcpServiceImpl:
    return GcpServiceImpl(mode=ServiceMode.DRY_RUN, project_id="test-project")


def _tpu_request(
    name: str = "my-tpu",
    zone: str = "us-central2-b",
    accelerator_type: str = "v4-8",
    runtime_version: str = "tpu-ubuntu2204-base",
    labels: dict[str, str] | None = None,
) -> TpuCreateRequest:
    return TpuCreateRequest(
        name=name,
        zone=zone,
        accelerator_type=accelerator_type,
        runtime_version=runtime_version,
        labels=labels or {},
    )


def _vm_request(
    name: str = "my-vm",
    zone: str = "us-central2-b",
    machine_type: str = "n1-standard-4",
    labels: dict[str, str] | None = None,
) -> VmCreateRequest:
    return VmCreateRequest(
        name=name,
        zone=zone,
        machine_type=machine_type,
        labels=labels or {},
    )


# ========================================================================
# TPU validation
# ========================================================================


class TestTpuValidation:
    def test_invalid_zone(self, svc: GcpServiceImpl) -> None:
        with pytest.raises(PlatformError, match="not available"):
            svc.tpu_create(_tpu_request(zone="mars-west1-a"))

    def test_invalid_accelerator_type(self, svc: GcpServiceImpl) -> None:
        with pytest.raises(ResourceNotFoundError, match="Unknown accelerator"):
            svc.tpu_create(_tpu_request(accelerator_type="v99-9999"))

    def test_name_too_long(self, svc: GcpServiceImpl) -> None:
        with pytest.raises(ValueError, match="exceeds 63"):
            svc.tpu_create(_tpu_request(name="a" * 64))

    def test_name_invalid_chars(self, svc: GcpServiceImpl) -> None:
        with pytest.raises(ValueError, match="Invalid TPU name"):
            svc.tpu_create(_tpu_request(name="MyTPU"))

    def test_name_starts_with_digit(self, svc: GcpServiceImpl) -> None:
        with pytest.raises(ValueError, match="Invalid TPU name"):
            svc.tpu_create(_tpu_request(name="1-bad-name"))

    def test_name_ends_with_hyphen(self, svc: GcpServiceImpl) -> None:
        with pytest.raises(ValueError, match="Invalid TPU name"):
            svc.tpu_create(_tpu_request(name="bad-name-"))

    def test_invalid_label_key(self, svc: GcpServiceImpl) -> None:
        with pytest.raises(ValueError, match="Invalid label key"):
            svc.tpu_create(_tpu_request(labels={"Bad-Key": "value"}))

    def test_invalid_label_value(self, svc: GcpServiceImpl) -> None:
        with pytest.raises(ValueError, match="Invalid label value"):
            svc.tpu_create(_tpu_request(labels={"key": "BAD VALUE"}))

    def test_empty_runtime_version(self, svc: GcpServiceImpl) -> None:
        with pytest.raises(ValueError, match="runtime_version"):
            svc.tpu_create(_tpu_request(runtime_version=""))


# ========================================================================
# VM validation
# ========================================================================


class TestVmValidation:
    def test_invalid_zone(self, svc: GcpServiceImpl) -> None:
        with pytest.raises(PlatformError, match="not available"):
            svc.vm_create(_vm_request(zone="nowhere-1-a"))

    def test_name_too_long(self, svc: GcpServiceImpl) -> None:
        with pytest.raises(ValueError, match="exceeds 63"):
            svc.vm_create(_vm_request(name="a" * 64))

    def test_invalid_disk_size(self, svc: GcpServiceImpl) -> None:
        req = VmCreateRequest(
            name="my-vm",
            zone="us-central2-b",
            machine_type="n1-standard-4",
            disk_size_gb=0,
        )
        with pytest.raises(ValueError, match="disk_size_gb must be positive"):
            svc.vm_create(req)

    def test_invalid_label_key(self, svc: GcpServiceImpl) -> None:
        with pytest.raises(ValueError, match="Invalid label key"):
            svc.vm_create(_vm_request(labels={"123bad": "v"}))


# ========================================================================
# DRY_RUN in-memory state: TPUs
# ========================================================================


class TestDryRunTpuState:
    def test_create_and_describe(self, svc: GcpServiceImpl) -> None:
        info = svc.tpu_create(_tpu_request(name="tpu-a"))
        assert info.name == "tpu-a"
        assert info.state == "CREATING"
        assert info.accelerator_type == "v4-8"
        assert info.zone == "us-central2-b"

        svc.advance_tpu_state("tpu-a", "us-central2-b", "READY")
        described = svc.tpu_describe("tpu-a", "us-central2-b")
        assert described is not None
        assert described.name == "tpu-a"
        assert described.state == "READY"

    def test_describe_nonexistent(self, svc: GcpServiceImpl) -> None:
        assert svc.tpu_describe("no-such-tpu", "us-central2-b") is None

    def test_create_and_list(self, svc: GcpServiceImpl) -> None:
        svc.tpu_create(_tpu_request(name="tpu-a", labels={"env": "test"}))
        svc.tpu_create(_tpu_request(name="tpu-b", labels={"env": "prod"}))

        all_tpus = svc.tpu_list(zones=["us-central2-b"])
        assert len(all_tpus) == 2

        filtered = svc.tpu_list(zones=["us-central2-b"], labels={"env": "test"})
        assert len(filtered) == 1
        assert filtered[0].name == "tpu-a"

    def test_list_filters_by_zone(self, svc: GcpServiceImpl) -> None:
        svc.tpu_create(_tpu_request(name="tpu-a", zone="us-central2-b"))
        svc.tpu_create(_tpu_request(name="tpu-b", zone="us-central1-a"))

        assert len(svc.tpu_list(zones=["us-central2-b"])) == 1
        assert len(svc.tpu_list(zones=["us-central1-a"])) == 1
        assert len(svc.tpu_list(zones=["us-central2-b", "us-central1-a"])) == 2

    def test_create_and_delete(self, svc: GcpServiceImpl) -> None:
        svc.tpu_create(_tpu_request(name="tpu-a"))
        assert svc.tpu_describe("tpu-a", "us-central2-b") is not None

        svc.tpu_delete("tpu-a", "us-central2-b")
        assert svc.tpu_describe("tpu-a", "us-central2-b") is None

    def test_delete_nonexistent_is_noop(self, svc: GcpServiceImpl) -> None:
        svc.tpu_delete("no-such-tpu", "us-central2-b")

    def test_network_endpoints_from_topology(self, svc: GcpServiceImpl) -> None:
        info = svc.tpu_create(_tpu_request(accelerator_type="v4-8"))
        # v4-8 has vm_count=1 per TPU_TOPOLOGIES
        assert len(info.network_endpoints) == 1

        svc_2 = GcpServiceImpl(mode=ServiceMode.DRY_RUN)
        info_multi = svc_2.tpu_create(_tpu_request(accelerator_type="v4-16"))
        # v4-16 has vm_count=2
        assert len(info_multi.network_endpoints) == 2

    def test_labels_and_metadata_preserved(self, svc: GcpServiceImpl) -> None:
        info = svc.tpu_create(_tpu_request(labels={"env": "test"}))
        assert info.labels == {"env": "test"}


# ========================================================================
# DRY_RUN in-memory state: VMs
# ========================================================================


class TestDryRunVmState:
    def test_create_and_describe(self, svc: GcpServiceImpl) -> None:
        info = svc.vm_create(_vm_request(name="vm-a"))
        assert info.name == "vm-a"
        assert info.status == "RUNNING"
        assert info.zone == "us-central2-b"
        assert info.internal_ip.startswith("10.1.")

        described = svc.vm_describe("vm-a", "us-central2-b")
        assert described is not None
        assert described.name == "vm-a"

    def test_describe_nonexistent(self, svc: GcpServiceImpl) -> None:
        assert svc.vm_describe("no-vm", "us-central2-b") is None

    def test_create_and_list(self, svc: GcpServiceImpl) -> None:
        svc.vm_create(_vm_request(name="vm-a", labels={"role": "ctrl"}))
        svc.vm_create(_vm_request(name="vm-b", labels={"role": "worker"}))

        all_vms = svc.vm_list(zones=["us-central2-b"])
        assert len(all_vms) == 2

        filtered = svc.vm_list(zones=["us-central2-b"], labels={"role": "ctrl"})
        assert len(filtered) == 1
        assert filtered[0].name == "vm-a"

    def test_create_and_delete(self, svc: GcpServiceImpl) -> None:
        svc.vm_create(_vm_request(name="vm-a"))
        svc.vm_delete("vm-a", "us-central2-b")
        assert svc.vm_describe("vm-a", "us-central2-b") is None

    def test_update_labels(self, svc: GcpServiceImpl) -> None:
        svc.vm_create(_vm_request(name="vm-a", labels={"env": "test"}))
        svc.vm_update_labels("vm-a", "us-central2-b", {"version": "2"})

        info = svc.vm_describe("vm-a", "us-central2-b")
        assert info is not None
        assert info.labels == {"env": "test", "version": "2"}

    def test_update_labels_nonexistent_raises(self, svc: GcpServiceImpl) -> None:
        with pytest.raises(PlatformError, match="not found"):
            svc.vm_update_labels("no-vm", "us-central2-b", {"x": "y"})

    def test_set_metadata(self, svc: GcpServiceImpl) -> None:
        svc.vm_create(_vm_request(name="vm-a"))
        svc.vm_set_metadata("vm-a", "us-central2-b", {"startup-script": "echo hi"})

        info = svc.vm_describe("vm-a", "us-central2-b")
        assert info is not None
        assert info.metadata["startup-script"] == "echo hi"

    def test_set_metadata_nonexistent_raises(self, svc: GcpServiceImpl) -> None:
        with pytest.raises(PlatformError, match="not found"):
            svc.vm_set_metadata("no-vm", "us-central2-b", {"k": "v"})

    def test_serial_port_output_returns_empty(self, svc: GcpServiceImpl) -> None:
        assert svc.vm_get_serial_port_output("any-vm", "us-central2-b") == ""


# ========================================================================
# Quota enforcement
# ========================================================================


class TestQuotaEnforcement:
    def test_zone_quota_exhausted(self, svc: GcpServiceImpl) -> None:
        svc.set_zone_quota("us-central2-b", max_tpus=1)
        svc.tpu_create(_tpu_request(name="tpu-a"))

        with pytest.raises(QuotaExhaustedError, match="Quota exhausted"):
            svc.tpu_create(_tpu_request(name="tpu-b"))

    def test_zone_quota_zero(self, svc: GcpServiceImpl) -> None:
        svc.set_zone_quota("us-central2-b", max_tpus=0)
        with pytest.raises(QuotaExhaustedError):
            svc.tpu_create(_tpu_request(name="tpu-a"))

    def test_quota_per_zone(self, svc: GcpServiceImpl) -> None:
        svc.set_zone_quota("us-central2-b", max_tpus=0)
        # Different zone still has default quota
        info = svc.tpu_create(_tpu_request(name="tpu-a", zone="us-central1-a"))
        assert info.zone == "us-central1-a"

    def test_quota_freed_after_delete(self, svc: GcpServiceImpl) -> None:
        svc.set_zone_quota("us-central2-b", max_tpus=1)
        svc.tpu_create(_tpu_request(name="tpu-a"))

        svc.tpu_delete("tpu-a", "us-central2-b")
        # Now quota is available again
        info = svc.tpu_create(_tpu_request(name="tpu-c"))
        assert info.name == "tpu-c"


# ========================================================================
# Failure injection
# ========================================================================


class TestFailureInjection:
    def test_inject_tpu_create_failure(self, svc: GcpServiceImpl) -> None:
        svc.inject_failure("tpu_create", PlatformError("simulated"))
        with pytest.raises(PlatformError, match="simulated"):
            svc.tpu_create(_tpu_request())

    def test_injected_failure_auto_clears(self, svc: GcpServiceImpl) -> None:
        svc.inject_failure("tpu_create", PlatformError("once"))
        with pytest.raises(PlatformError):
            svc.tpu_create(_tpu_request())

        # Second call succeeds (failure was one-shot)
        info = svc.tpu_create(_tpu_request(name="tpu-ok"))
        assert info.name == "tpu-ok"

    def test_inject_quota_exhausted(self, svc: GcpServiceImpl) -> None:
        svc.inject_failure("tpu_create", QuotaExhaustedError("zone full"))
        with pytest.raises(QuotaExhaustedError, match="zone full"):
            svc.tpu_create(_tpu_request())

    def test_inject_tpu_delete_failure(self, svc: GcpServiceImpl) -> None:
        svc.tpu_create(_tpu_request(name="tpu-a"))
        svc.inject_failure("tpu_delete", PlatformError("delete failed"))
        with pytest.raises(PlatformError, match="delete failed"):
            svc.tpu_delete("tpu-a", "us-central2-b")

    def test_inject_vm_create_failure(self, svc: GcpServiceImpl) -> None:
        svc.inject_failure("vm_create", PlatformError("vm boom"))
        with pytest.raises(PlatformError, match="vm boom"):
            svc.vm_create(_vm_request())

    def test_inject_tpu_list_failure(self, svc: GcpServiceImpl) -> None:
        svc.inject_failure("tpu_list", PlatformError("list failed"))
        with pytest.raises(PlatformError, match="list failed"):
            svc.tpu_list(zones=["us-central2-b"])


# ========================================================================
# set_tpu_type_unavailable / add_tpu_type
# ========================================================================


class TestAcceleratorTypeManagement:
    def test_remove_type_makes_create_fail(self, svc: GcpServiceImpl) -> None:
        svc.set_tpu_type_unavailable("v4-8")
        with pytest.raises(ResourceNotFoundError, match="Unknown accelerator"):
            svc.tpu_create(_tpu_request(accelerator_type="v4-8"))

    def test_add_custom_type(self, svc: GcpServiceImpl) -> None:
        svc.add_tpu_type("v99-magic-256")
        info = svc.tpu_create(_tpu_request(accelerator_type="v99-magic-256"))
        assert info.accelerator_type == "v99-magic-256"

    def test_remove_then_add_back(self, svc: GcpServiceImpl) -> None:
        svc.set_tpu_type_unavailable("v4-8")
        with pytest.raises(ResourceNotFoundError):
            svc.tpu_create(_tpu_request(accelerator_type="v4-8"))

        svc.add_tpu_type("v4-8")
        info = svc.tpu_create(_tpu_request(accelerator_type="v4-8"))
        assert info.accelerator_type == "v4-8"


# ========================================================================
# Mode property
# ========================================================================


def test_mode_property() -> None:
    svc = GcpServiceImpl(mode=ServiceMode.DRY_RUN)
    assert svc.mode == ServiceMode.DRY_RUN


def test_cloud_mode_property() -> None:
    svc = GcpServiceImpl(mode=ServiceMode.CLOUD, project_id="p")
    assert svc.mode == ServiceMode.CLOUD
    assert svc.project_id == "p"


# ========================================================================
# Duplicate detection
# ========================================================================


class TestDuplicateDetection:
    def test_tpu_create_duplicate_raises(self, svc: GcpServiceImpl) -> None:
        svc.tpu_create(_tpu_request(name="tpu-a", zone="us-central2-b"))
        with pytest.raises(PlatformError, match="already exists"):
            svc.tpu_create(_tpu_request(name="tpu-a", zone="us-central2-b"))

    def test_vm_create_duplicate_raises(self, svc: GcpServiceImpl) -> None:
        svc.vm_create(_vm_request(name="vm-a", zone="us-central2-b"))
        with pytest.raises(PlatformError, match="already exists"):
            svc.vm_create(_vm_request(name="vm-a", zone="us-central2-b"))


# ========================================================================
# TPU state transitions
# ========================================================================


class TestTpuStateTransitions:
    def test_tpu_initial_state_creating(self, svc: GcpServiceImpl) -> None:
        info = svc.tpu_create(_tpu_request(name="tpu-a"))
        assert info.state == "CREATING"

        described = svc.tpu_describe("tpu-a", "us-central2-b")
        assert described is not None
        assert described.state == "CREATING"

    def test_advance_tpu_state(self, svc: GcpServiceImpl) -> None:
        svc.tpu_create(_tpu_request(name="tpu-a"))
        svc.advance_tpu_state("tpu-a", "us-central2-b", "READY")

        described = svc.tpu_describe("tpu-a", "us-central2-b")
        assert described is not None
        assert described.state == "READY"

    def test_advance_nonexistent_raises(self, svc: GcpServiceImpl) -> None:
        with pytest.raises(ValueError, match="not found"):
            svc.advance_tpu_state("no-such-tpu", "us-central2-b")


# ========================================================================
# Per-type-per-zone availability
# ========================================================================


class TestTypeZoneAvailability:
    def test_tpu_type_zone_restriction(self, svc: GcpServiceImpl) -> None:
        svc.set_available_types_by_zone({"us-central2-b": {"v4-8"}})

        # Allowed type succeeds
        info = svc.tpu_create(_tpu_request(name="tpu-ok", accelerator_type="v4-8"))
        assert info.accelerator_type == "v4-8"

        # Disallowed type raises
        with pytest.raises(QuotaExhaustedError, match="not available"):
            svc.tpu_create(_tpu_request(name="tpu-bad", accelerator_type="v4-16"))

    def test_zone_not_in_mapping_rejects_all(self, svc: GcpServiceImpl) -> None:
        svc.set_available_types_by_zone({"us-central1-a": {"v4-8"}})
        with pytest.raises(QuotaExhaustedError, match="not available"):
            svc.tpu_create(_tpu_request(name="tpu-a", zone="us-central2-b", accelerator_type="v4-8"))


# ========================================================================
# VM quota enforcement
# ========================================================================


class TestVmQuotaEnforcement:
    def test_vm_quota_enforcement(self, svc: GcpServiceImpl) -> None:
        svc.set_vm_zone_quota("us-central2-b", max_vms=2)
        svc.vm_create(_vm_request(name="vm-a"))
        svc.vm_create(_vm_request(name="vm-b"))

        with pytest.raises(QuotaExhaustedError, match="VM quota exhausted"):
            svc.vm_create(_vm_request(name="vm-c"))

    def test_vm_quota_freed_after_delete(self, svc: GcpServiceImpl) -> None:
        svc.set_vm_zone_quota("us-central2-b", max_vms=1)
        svc.vm_create(_vm_request(name="vm-a"))
        svc.vm_delete("vm-a", "us-central2-b")
        info = svc.vm_create(_vm_request(name="vm-b"))
        assert info.name == "vm-b"
