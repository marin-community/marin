# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for GCP service-level validation and behavior.

Validates the GcpService contract using InMemoryGcpService in DRY_RUN mode.
These tests cover:
- Input validation (names, zones, labels, accelerator types)
- Quota enforcement (per-zone TPU and VM limits)
- Duplicate detection
- Failure injection (one-shot error injection for testing error paths)
"""

import time
from collections.abc import Callable
from unittest.mock import MagicMock, patch

import httpx
import pytest
from iris.cluster.platforms.gcp.fake import InMemoryGcpService
from iris.cluster.platforms.gcp.handles import GcpVmSliceHandle
from iris.cluster.platforms.gcp.service import (
    CloudGcpService,
    TpuCreateRequest,
    VmCreateRequest,
    VmInfo,
)
from iris.cluster.platforms.types import (
    CloudSliceState,
    InfraError,
    InfraUnavailableError,
    QuotaExhaustedError,
    ResourceNotFoundError,
)
from iris.cluster.service_mode import ServiceMode
from iris.cluster.types import CapacityType
from rigging.timing import Timestamp


@pytest.fixture
def svc() -> InMemoryGcpService:
    return InMemoryGcpService(mode=ServiceMode.DRY_RUN, project_id="test-project")


def _tpu_request(
    name: str = "my-tpu",
    zone: str = "us-central2-b",
    accelerator_type: str = "v4-8",
    runtime_version: str = "tpu-ubuntu2204-base",
    labels: dict[str, str] | None = None,
    capacity_type: CapacityType | None = CapacityType.PREEMPTIBLE,
) -> TpuCreateRequest:
    return TpuCreateRequest(
        name=name,
        zone=zone,
        accelerator_type=accelerator_type,
        runtime_version=runtime_version,
        capacity_type=capacity_type,
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
    def test_invalid_zone(self, svc: InMemoryGcpService) -> None:
        with pytest.raises(InfraError, match="not available"):
            svc.tpu_create(_tpu_request(zone="mars-west1-a"))

    def test_invalid_accelerator_type(self, svc: InMemoryGcpService) -> None:
        with pytest.raises(ResourceNotFoundError, match="Unknown accelerator"):
            svc.tpu_create(_tpu_request(accelerator_type="v99-9999"))

    def test_name_too_long(self, svc: InMemoryGcpService) -> None:
        with pytest.raises(ValueError, match="exceeds 63"):
            svc.tpu_create(_tpu_request(name="a" * 64))

    def test_name_invalid_chars(self, svc: InMemoryGcpService) -> None:
        with pytest.raises(ValueError, match="Invalid TPU name"):
            svc.tpu_create(_tpu_request(name="MyTPU"))

    def test_name_starts_with_digit(self, svc: InMemoryGcpService) -> None:
        with pytest.raises(ValueError, match="Invalid TPU name"):
            svc.tpu_create(_tpu_request(name="1-bad-name"))

    def test_name_ends_with_hyphen(self, svc: InMemoryGcpService) -> None:
        with pytest.raises(ValueError, match="Invalid TPU name"):
            svc.tpu_create(_tpu_request(name="bad-name-"))

    def test_invalid_label_key(self, svc: InMemoryGcpService) -> None:
        with pytest.raises(ValueError, match="Invalid label key"):
            svc.tpu_create(_tpu_request(labels={"Bad-Key": "value"}))

    def test_invalid_label_value(self, svc: InMemoryGcpService) -> None:
        with pytest.raises(ValueError, match="Invalid label value"):
            svc.tpu_create(_tpu_request(labels={"key": "BAD VALUE"}))

    def test_empty_runtime_version(self, svc: InMemoryGcpService) -> None:
        with pytest.raises(ValueError, match="runtime_version"):
            svc.tpu_create(_tpu_request(runtime_version=""))


# ========================================================================
# VM validation
# ========================================================================


class TestVmValidation:
    def test_invalid_zone(self, svc: InMemoryGcpService) -> None:
        with pytest.raises(InfraError, match="not available"):
            svc.vm_create(_vm_request(zone="nowhere-1-a"))

    def test_name_too_long(self, svc: InMemoryGcpService) -> None:
        with pytest.raises(ValueError, match="exceeds 63"):
            svc.vm_create(_vm_request(name="a" * 64))

    def test_invalid_disk_size(self, svc: InMemoryGcpService) -> None:
        req = VmCreateRequest(
            name="my-vm",
            zone="us-central2-b",
            machine_type="n1-standard-4",
            disk_size_gb=0,
        )
        with pytest.raises(ValueError, match="disk_size_gb must be positive"):
            svc.vm_create(req)

    def test_invalid_label_key(self, svc: InMemoryGcpService) -> None:
        with pytest.raises(ValueError, match="Invalid label key"):
            svc.vm_create(_vm_request(labels={"123bad": "v"}))


# ========================================================================
# Duplicate detection
# ========================================================================


def test_tpu_create_duplicate_raises(svc: InMemoryGcpService) -> None:
    svc.tpu_create(_tpu_request(name="tpu-a", zone="us-central2-b"))
    with pytest.raises(InfraError, match="already exists"):
        svc.tpu_create(_tpu_request(name="tpu-a", zone="us-central2-b"))


def test_vm_create_duplicate_raises(svc: InMemoryGcpService) -> None:
    svc.vm_create(_vm_request(name="vm-a", zone="us-central2-b"))
    with pytest.raises(InfraError, match="already exists"):
        svc.vm_create(_vm_request(name="vm-a", zone="us-central2-b"))


# ========================================================================
# TPU quota enforcement
# ========================================================================


def test_tpu_zone_quota_exhausted(svc: InMemoryGcpService) -> None:
    svc.set_zone_quota("us-central2-b", max_tpus=1)
    svc.tpu_create(_tpu_request(name="tpu-a"))

    with pytest.raises(QuotaExhaustedError, match="Quota exhausted"):
        svc.tpu_create(_tpu_request(name="tpu-b"))


def test_tpu_zone_quota_zero(svc: InMemoryGcpService) -> None:
    svc.set_zone_quota("us-central2-b", max_tpus=0)
    with pytest.raises(QuotaExhaustedError):
        svc.tpu_create(_tpu_request(name="tpu-a"))


def test_tpu_quota_per_zone(svc: InMemoryGcpService) -> None:
    svc.set_zone_quota("us-central2-b", max_tpus=0)
    info = svc.tpu_create(_tpu_request(name="tpu-a", zone="us-central1-a"))
    assert info.zone == "us-central1-a"


def test_tpu_quota_freed_after_delete(svc: InMemoryGcpService) -> None:
    svc.set_zone_quota("us-central2-b", max_tpus=1)
    svc.tpu_create(_tpu_request(name="tpu-a"))

    svc.tpu_delete("tpu-a", "us-central2-b")
    info = svc.tpu_create(_tpu_request(name="tpu-c"))
    assert info.name == "tpu-c"


# ========================================================================
# VM quota enforcement
# ========================================================================


def test_vm_quota_enforcement(svc: InMemoryGcpService) -> None:
    svc.set_vm_zone_quota("us-central2-b", max_vms=2)
    svc.vm_create(_vm_request(name="vm-a"))
    svc.vm_create(_vm_request(name="vm-b"))

    with pytest.raises(QuotaExhaustedError, match="VM quota exhausted"):
        svc.vm_create(_vm_request(name="vm-c"))


def test_vm_quota_freed_after_delete(svc: InMemoryGcpService) -> None:
    svc.set_vm_zone_quota("us-central2-b", max_vms=1)
    svc.vm_create(_vm_request(name="vm-a"))
    svc.vm_delete("vm-a", "us-central2-b")
    info = svc.vm_create(_vm_request(name="vm-b"))
    assert info.name == "vm-b"


def test_service_account_is_preserved_on_created_resources(svc: InMemoryGcpService) -> None:
    tpu_request = _tpu_request(name="tpu-sa", labels={"env": "test"})
    tpu_request.service_account = "iris-worker@test-project.iam.gserviceaccount.com"
    tpu_info = svc.tpu_create(tpu_request)

    vm_request = _vm_request(name="vm-sa", labels={"env": "test"})
    vm_request.service_account = "iris-controller@test-project.iam.gserviceaccount.com"
    vm_info = svc.vm_create(vm_request)

    assert tpu_info.service_account == "iris-worker@test-project.iam.gserviceaccount.com"
    assert vm_info.service_account == "iris-controller@test-project.iam.gserviceaccount.com"


# ========================================================================
# Failure injection
# ========================================================================


def test_inject_tpu_create_failure(svc: InMemoryGcpService) -> None:
    svc.inject_failure("tpu_create", InfraError("simulated"))
    with pytest.raises(InfraError, match="simulated"):
        svc.tpu_create(_tpu_request())


def test_injected_failure_auto_clears(svc: InMemoryGcpService) -> None:
    svc.inject_failure("tpu_create", InfraError("once"))
    with pytest.raises(InfraError):
        svc.tpu_create(_tpu_request())

    info = svc.tpu_create(_tpu_request(name="tpu-ok"))
    assert info.name == "tpu-ok"


def test_inject_quota_exhausted(svc: InMemoryGcpService) -> None:
    svc.inject_failure("tpu_create", QuotaExhaustedError("zone full"))
    with pytest.raises(QuotaExhaustedError, match="zone full"):
        svc.tpu_create(_tpu_request())


def test_inject_tpu_delete_failure(svc: InMemoryGcpService) -> None:
    svc.tpu_create(_tpu_request(name="tpu-a"))
    svc.inject_failure("tpu_delete", InfraError("delete failed"))
    with pytest.raises(InfraError, match="delete failed"):
        svc.tpu_delete("tpu-a", "us-central2-b")


def test_inject_vm_create_failure(svc: InMemoryGcpService) -> None:
    svc.inject_failure("vm_create", InfraError("vm boom"))
    with pytest.raises(InfraError, match="vm boom"):
        svc.vm_create(_vm_request())


def test_inject_tpu_list_failure(svc: InMemoryGcpService) -> None:
    svc.inject_failure("tpu_list", InfraError("list failed"))
    with pytest.raises(InfraError, match="list failed"):
        svc.tpu_list(zones=["us-central2-b"])


# ========================================================================
# Behavioral tests: TPU topology-based endpoint generation
# ========================================================================


def test_network_endpoints_match_topology(svc: InMemoryGcpService) -> None:
    """TPU create returns network endpoints matching the accelerator topology."""
    info_v4_8 = svc.tpu_create(_tpu_request(name="tpu-a", accelerator_type="v4-8"))
    assert len(info_v4_8.network_endpoints) == 1

    svc2 = InMemoryGcpService(mode=ServiceMode.DRY_RUN)
    info_v4_16 = svc2.tpu_create(_tpu_request(name="tpu-b", accelerator_type="v4-16"))
    assert len(info_v4_16.network_endpoints) == 2


# ========================================================================
# Behavioral: list filtering, delete-then-describe
# ========================================================================


def test_tpu_list_filters_by_zone(svc: InMemoryGcpService) -> None:
    svc.tpu_create(_tpu_request(name="tpu-a", zone="us-central2-b"))
    svc.tpu_create(_tpu_request(name="tpu-b", zone="us-central1-a"))

    assert len(svc.tpu_list(zones=["us-central2-b"])) == 1
    assert len(svc.tpu_list(zones=["us-central1-a"])) == 1
    assert len(svc.tpu_list(zones=["us-central2-b", "us-central1-a"])) == 2


def test_tpu_list_filters_by_labels(svc: InMemoryGcpService) -> None:
    svc.tpu_create(_tpu_request(name="tpu-a", labels={"env": "test"}))
    svc.tpu_create(_tpu_request(name="tpu-b", labels={"env": "prod"}))

    filtered = svc.tpu_list(zones=["us-central2-b"], labels={"env": "test"})
    assert len(filtered) == 1
    assert filtered[0].name == "tpu-a"


def test_tpu_describe_nonexistent(svc: InMemoryGcpService) -> None:
    assert svc.tpu_describe("no-such-tpu", "us-central2-b") is None


def test_tpu_delete_nonexistent_is_noop(svc: InMemoryGcpService) -> None:
    """Deleting a nonexistent TPU does not raise."""
    svc.tpu_delete("no-such-tpu", "us-central2-b")


def test_tpu_delete_makes_describe_return_none(svc: InMemoryGcpService) -> None:
    svc.tpu_create(_tpu_request(name="tpu-a"))
    svc.tpu_delete("tpu-a", "us-central2-b")
    assert svc.tpu_describe("tpu-a", "us-central2-b") is None


def test_vm_describe_nonexistent(svc: InMemoryGcpService) -> None:
    assert svc.vm_describe("no-vm", "us-central2-b") is None


def test_vm_delete_makes_describe_return_none(svc: InMemoryGcpService) -> None:
    svc.vm_create(_vm_request(name="vm-a"))
    svc.vm_delete("vm-a", "us-central2-b")
    assert svc.vm_describe("vm-a", "us-central2-b") is None


def test_vm_update_labels_nonexistent_raises(svc: InMemoryGcpService) -> None:
    with pytest.raises(InfraError, match="not found"):
        svc.vm_update_labels("no-vm", "us-central2-b", {"x": "y"})


def test_vm_set_metadata_nonexistent_raises(svc: InMemoryGcpService) -> None:
    with pytest.raises(InfraError, match="not found"):
        svc.vm_set_metadata("no-vm", "us-central2-b", {"k": "v"})


# ========================================================================
# Zone extraction from full resource names (wildcard listing)
# When zones=[] the GCP API is called with locations/- and the real zone
# must be parsed from the returned full resource name, not left as "-".
# ========================================================================


def test_queued_resource_list_extracts_zone_from_wildcard() -> None:
    """queued_resource_list must parse the real zone from the full resource name."""
    svc = CloudGcpService(project_id="test-project")

    fake_qr = MagicMock()
    fake_qr.name = "projects/test-project/locations/us-central2-b/queuedResources/my-reserved-tpu"
    fake_qr.state.state.name = "PROVISIONING"
    fake_qr.tpu.node_spec = []

    mock_client = MagicMock()
    mock_client.list_queued_resources.return_value = [fake_qr]
    svc._tpu_alpha_client_cached = mock_client

    results = svc.queued_resource_list(zones=[])

    assert len(results) == 1
    assert results[0].name == "my-reserved-tpu"
    assert results[0].zone == "us-central2-b"


def test_queued_resource_list_preserves_explicit_zone() -> None:
    """queued_resource_list must not override zone when one is explicitly provided."""
    svc = CloudGcpService(project_id="test-project")

    fake_qr = MagicMock()
    fake_qr.name = "projects/test-project/locations/us-central2-b/queuedResources/my-reserved-tpu"
    fake_qr.state.state.name = "PROVISIONING"
    fake_qr.tpu.node_spec = []

    mock_client = MagicMock()
    mock_client.list_queued_resources.return_value = [fake_qr]
    svc._tpu_alpha_client_cached = mock_client

    results = svc.queued_resource_list(zones=["us-central2-b"])

    assert len(results) == 1
    assert results[0].zone == "us-central2-b"


def test_tpu_list_extracts_zone_from_wildcard() -> None:
    """tpu_list must parse the real zone from the full resource name."""
    svc = CloudGcpService(project_id="test-project")

    fake_tpu = {
        "name": "projects/test-project/locations/us-central2-b/nodes/my-tpu",
        "state": "READY",
        "acceleratorType": "v4-8",
        "runtimeVersion": "tpu-ubuntu2204-base",
        "labels": {},
        "networkEndpoints": [],
    }

    with patch.object(svc, "_paginate", return_value=[fake_tpu]):
        results = svc.tpu_list(zones=[])

    assert len(results) == 1
    assert results[0].name == "my-tpu"
    assert results[0].zone == "us-central2-b"


# ========================================================================
# Transient-error retry and propagation (HTTP read path)
# ========================================================================


def _gcp_error_response(code: int, *, status: str = "", message: str = "boom") -> httpx.Response:
    return httpx.Response(code, json={"error": {"code": code, "status": status, "message": message}})


def _mock_service(handler: Callable[[httpx.Request], httpx.Response]) -> CloudGcpService:
    """Build a CloudGcpService whose HTTP client is driven by ``handler``.

    The token is pre-seeded so ``_headers`` never triggers a real google.auth
    credential refresh.
    """
    client = httpx.Client(transport=httpx.MockTransport(handler))
    svc = CloudGcpService(project_id="test-project", http_client=client)
    svc._token = "fake-token"
    svc._expires_at = time.monotonic() + 3600
    return svc


def test_vm_list_recovers_after_transient_blip(monkeypatch: pytest.MonkeyPatch) -> None:
    """The ferry discovery path: a one-off 503 is retried, then returns the controller VM.

    Drives the full public path (classify -> retry -> paginate -> parse) so the
    returned VM distinguishes genuine recovery from a swallow-to-[].
    """
    monkeypatch.setattr("rigging.timing.time.sleep", lambda _: None)
    calls = {"n": 0}
    vm = {
        "name": "iris-controller",
        "status": "RUNNING",
        "zone": "https://www.googleapis.com/compute/v1/projects/p/zones/us-central1-a",
        "networkInterfaces": [{"networkIP": "10.0.0.5"}],
        "labels": {"iris-x-controller": "true"},
    }

    def handler(_request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return _gcp_error_response(503, message="Internal error. Please try again")
        return httpx.Response(200, json={"items": {"zones/us-central1-a": {"instances": [vm]}}})

    svc = _mock_service(handler)
    result = svc.vm_list(zones=[], labels={"iris-x-controller": "true"})
    assert [v.name for v in result] == ["iris-controller"]
    assert calls["n"] == 2  # one retried failure, then success


@pytest.mark.parametrize(
    ("code", "expected"),
    [
        (503, InfraUnavailableError),  # transient: retried, then surfaces after exhaustion
        (403, InfraError),  # non-transient: surfaces immediately
    ],
)
def test_vm_list_propagates_api_errors(monkeypatch: pytest.MonkeyPatch, code: int, expected: type[InfraError]) -> None:
    """vm_list never degrades an API error to []: a swallowed error reads as 'no VMs',

    which on the controller-discovery path the canary ferry depends on becomes a
    spurious "No controller VM found".
    """
    monkeypatch.setattr("rigging.timing.time.sleep", lambda _: None)
    svc = _mock_service(lambda _r: _gcp_error_response(code, message="boom"))
    with pytest.raises(expected):
        svc.vm_list(zones=[], labels={"iris-x-controller": "true"})


# ========================================================================
# Non-blocking vm_create
# ========================================================================

# A long-running zone operation that should never be awaited on the autoscaler path.
_OP_NAME = "operation-123"
_VM_ZONE = "us-central1-a"
_VM_NAME = "my-vm"


def _make_recording_service(op_name: str = _OP_NAME) -> tuple[CloudGcpService, list[httpx.Request]]:
    """Build a CloudGcpService that records every HTTP request it issues.

    POST /instances returns an operation name; GET /operations/... returns DONE;
    GET /instances/<name> returns a running instance so wait=True describe succeeds.
    The token is pre-seeded so _headers() never triggers a real credential refresh.
    """
    recorded: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        recorded.append(request)
        if request.method == "POST":
            return httpx.Response(200, json={"name": op_name, "status": "RUNNING"})
        if "/operations/" in str(request.url):
            return httpx.Response(200, json={"name": op_name, "status": "DONE"})
        # GET /instances/<name>
        return httpx.Response(
            200,
            json={
                "name": _VM_NAME,
                "status": "RUNNING",
                "zone": f"https://www.googleapis.com/compute/v1/projects/test-project/zones/{_VM_ZONE}",
                "networkInterfaces": [{"networkIP": "10.0.0.1"}],
                "labels": {},
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    svc = CloudGcpService(project_id="test-project", http_client=client)
    svc._token = "fake-token"
    svc._expires_at = time.monotonic() + 3600
    return svc, recorded


def test_vm_create_default_issues_no_operation_poll() -> None:
    """vm_create(wait=False) issues only the insert POST; no zone-operation GET is sent.

    The observable contract at the HTTP boundary: blocking on the operation poll
    (up to 600 s) inside the single control tick would starve reconcile and
    worker keep-alive for GCP_SLICE_MODE_VM slices.
    """
    svc, recorded = _make_recording_service()

    req = VmCreateRequest(name=_VM_NAME, zone=_VM_ZONE, machine_type="n2-standard-4")
    info = svc.vm_create(req)  # wait=False by default

    methods = [r.method for r in recorded]
    urls = [str(r.url) for r in recorded]

    assert methods == ["POST"], f"Expected only the insert POST, got: {list(zip(methods, urls, strict=True))}"
    assert info.name == _VM_NAME
    assert info.status == "PROVISIONING"
    assert info.internal_ip == ""  # IP not yet assigned; describe loop will supply it


def test_vm_create_wait_true_polls_operation_and_describes() -> None:
    """vm_create(wait=True) polls the zone operation then describes the instance.

    This is the controller-VM bootstrap path that needs internal_ip immediately.
    The observable contract: a POST (insert), a GET to operations, and a GET to
    the instance are all issued.
    """
    svc, recorded = _make_recording_service()

    req = VmCreateRequest(name=_VM_NAME, zone=_VM_ZONE, machine_type="n2-standard-4")
    info = svc.vm_create(req, wait=True)

    methods = [r.method for r in recorded]
    urls = [str(r.url) for r in recorded]

    assert methods[0] == "POST", f"First call should be the insert POST, got: {list(zip(methods, urls, strict=True))}"
    assert any(
        "/operations/" in u for u in urls
    ), f"Expected a zone-operation poll GET, got: {list(zip(methods, urls, strict=True))}"
    assert info.status == "RUNNING"
    assert info.internal_ip == "10.0.0.1"


# ========================================================================
# GcpVmSliceHandle state machine: None-describe → CREATING → READY
# ========================================================================


def _make_vm_slice_handle(gcp_service: InMemoryGcpService, slice_id: str, vm_name: str, zone: str) -> GcpVmSliceHandle:
    """Build a GcpVmSliceHandle in the bootstrapping state (bootstrap_state=None)."""
    return GcpVmSliceHandle(
        _slice_id=slice_id,
        _vm_name=vm_name,
        _zone=zone,
        _project_id="test-project",
        _labels={},
        _created_at=Timestamp.now(),
        _label_prefix="iris",
        _worker_port=10001,
        _gcp_service=gcp_service,
        _bootstrapping=True,  # bootstrap_state=None → still booting
    )


def test_vm_slice_describe_unknown_when_vm_not_yet_visible() -> None:
    """describe() returns UNKNOWN when vm_describe returns None immediately after insert.

    The slice must NOT be treated as FAILED during the boot grace window: UNKNOWN
    is the correct state for a VM that has been inserted but is not yet visible to
    the describe API (a race that was always possible but is now more common with
    the non-blocking insert path).
    """
    gcp_service = InMemoryGcpService(mode=ServiceMode.DRY_RUN, project_id="test-project")
    handle = _make_vm_slice_handle(gcp_service, "slice-abc", "vm-abc", "us-central2-b")

    # VM not in the service yet (as if the insert just returned but describe hasn't caught up)
    status = handle.describe()
    assert status.state == CloudSliceState.UNKNOWN, (
        f"Expected UNKNOWN while VM is not visible, got {status.state}. "
        "FAILED here would cause an immediate spurious boot failure."
    )


def test_vm_slice_advances_from_creating_to_ready() -> None:
    """Slice state machine: not-visible → CREATING (PROVISIONING) → READY (RUNNING).

    This is the normal boot path for the non-blocking create: the autoscaler's
    refresh loop drives state advancement purely through vm_describe() calls.
    No zone-operation wait is involved.
    """
    gcp_service = InMemoryGcpService(mode=ServiceMode.DRY_RUN, project_id="test-project")

    # Create VM in PROVISIONING state by adding it directly to the service
    zone = "us-central2-b"
    vm_name = "vm-boot-test"
    provisioning_info = VmInfo(
        name=vm_name,
        status="PROVISIONING",
        zone=zone,
        internal_ip="",
        external_ip=None,
        labels={},
        metadata={},
        service_account=None,
        created_at=Timestamp.now(),
    )
    gcp_service._vms[(vm_name, zone)] = provisioning_info

    handle = _make_vm_slice_handle(gcp_service, "slice-boot", vm_name, zone)

    # PROVISIONING maps to CREATING in the state machine
    status = handle.describe()
    assert status.state == CloudSliceState.CREATING, f"Expected CREATING while VM is PROVISIONING, got {status.state}"

    # Advance to RUNNING (as refresh() would see after a few seconds)
    gcp_service.advance_vm_state(vm_name, zone, "RUNNING")

    # Still BOOTSTRAPPING because the bootstrap thread hasn't completed yet
    status = handle.describe()
    assert (
        status.state == CloudSliceState.BOOTSTRAPPING
    ), f"Expected BOOTSTRAPPING after VM is RUNNING but before bootstrap completes, got {status.state}"

    # Simulate bootstrap thread completing
    with handle._bootstrap_lock:
        handle._bootstrap_state = CloudSliceState.READY

    status = handle.describe()
    assert status.state == CloudSliceState.READY
    assert status.worker_count == 1
    assert status.workers[0].internal_address == gcp_service._vms[(vm_name, zone)].internal_ip
