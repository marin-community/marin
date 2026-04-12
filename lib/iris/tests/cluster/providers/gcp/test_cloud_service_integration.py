# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Integration test that exercises the full CloudGcpService TPU/VM lifecycle
with a mock HTTP backend, logging every request/response for debugging.

This test simulates the controller's view: create VMs and TPUs, describe them,
list them with label filters, set metadata/labels, and read logs. Every HTTP
call is recorded so we can compare the REST API behavior against what the old
gcloud CLI produced.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

import httpx
import pytest

from iris.cluster.providers.gcp import service as gcp_service
from iris.cluster.providers.gcp.service import (
    CloudGcpService,
    TpuCreateRequest,
    VmCreateRequest,
)
from iris.cluster.providers.types import QuotaExhaustedError
from iris.rpc import config_pb2

logger = logging.getLogger(__name__)

PROJECT = "test-project"
ZONE_EU = "europe-west4-b"
ZONE_US = "us-west4-a"


@dataclass
class HttpLog:
    method: str
    url: str
    request_body: dict | None
    status: int
    response_body: dict


@dataclass
class GcpFakeBackend:
    """Simulates GCP APIs at the HTTP level, tracking all instances and TPUs."""

    project: str = PROJECT
    vms: dict[tuple[str, str], dict] = field(default_factory=dict)  # (name, zone) -> vm_data
    tpus: dict[tuple[str, str], dict] = field(default_factory=dict)  # (name, zone) -> tpu_data
    operations: dict[str, dict] = field(default_factory=dict)  # op_name -> op_data
    http_log: list[HttpLog] = field(default_factory=list)
    # Number of remaining 429 responses to return for DELETE requests
    delete_429_remaining: int = 0

    def handle(self, request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        method = request.method
        req_body = None
        if request.content:
            try:
                req_body = json.loads(request.content)
            except (json.JSONDecodeError, UnicodeDecodeError):
                pass

        resp = self._route(method, url, req_body)
        resp_body = {}
        try:
            resp_body = resp.json() if resp.content else {}
        except Exception:
            pass

        self.http_log.append(
            HttpLog(
                method=method,
                url=url,
                request_body=req_body,
                status=resp.status_code,
                response_body=resp_body,
            )
        )
        return resp

    def _route(self, method: str, url: str, body: dict | None) -> httpx.Response:
        # --- TPU operations ---
        if "tpu.googleapis.com" in url:
            return self._handle_tpu(method, url, body)

        # --- Compute operations ---
        if "compute.googleapis.com" in url:
            return self._handle_compute(method, url, body)

        # --- Logging ---
        if "logging.googleapis.com" in url:
            return httpx.Response(200, json={"entries": []})

        return httpx.Response(404, json={"error": {"code": 404, "message": f"Unknown URL: {url}"}})

    def _handle_tpu(self, method: str, url: str, body: dict | None) -> httpx.Response:
        # POST .../nodes?nodeId=NAME — create TPU
        if method == "POST" and "/nodes" in url and "nodeId=" in url:
            node_id = url.split("nodeId=")[1].split("&")[0]
            zone = self._extract_tpu_zone(url)
            op_name = f"projects/{self.project}/locations/{zone}/operations/op-tpu-{node_id}"
            tpu_data = {
                "name": f"projects/{self.project}/locations/{zone}/nodes/{node_id}",
                "state": "CREATING",
                "acceleratorType": (body or {}).get("acceleratorType", ""),
                "runtimeVersion": (body or {}).get("runtimeVersion", ""),
                "labels": (body or {}).get("labels", {}),
                "metadata": (body or {}).get("metadata", {}),
                "networkEndpoints": [],
                "serviceAccount": (body or {}).get("serviceAccount"),
                "createTime": "2026-01-01T00:00:00Z",
            }
            if (body or {}).get("networkConfig"):
                tpu_data["networkConfig"] = body["networkConfig"]
            self.tpus[(node_id, zone)] = tpu_data
            # Mark operation as pending, will complete on poll
            self.operations[op_name] = {"name": op_name, "done": False, "tpu_key": (node_id, zone)}
            return httpx.Response(200, json={"name": op_name, "done": False})

        # GET .../operations/op-* — poll TPU operation
        if method == "GET" and "/operations/" in url:
            op_name = url.split("tpu.googleapis.com/v2/")[1].split("?")[0]
            op = self.operations.get(op_name)
            if op is None:
                return httpx.Response(404, json={"error": {"code": 404, "message": "Operation not found"}})
            # If the operation has an injected error, surface it
            if "error" in op:
                return httpx.Response(200, json={"name": op_name, "done": True, "error": op["error"]})
            # Complete the operation and make TPU READY with endpoints
            tpu_key = op.get("tpu_key")
            if tpu_key and tpu_key in self.tpus:
                tpu = self.tpus[tpu_key]
                tpu["state"] = "READY"
                tpu["networkEndpoints"] = [{"ipAddress": f"10.128.0.{i}", "port": 8470} for i in range(4)]
            return httpx.Response(200, json={"name": op_name, "done": True})

        # GET .../nodes/NAME — describe TPU
        if method == "GET" and "/nodes/" in url and "/operations/" not in url:
            parts = url.split("/nodes/")
            if len(parts) == 2:
                node_name = parts[1].split("?")[0]
                zone = self._extract_tpu_zone(url)
                key = (node_name, zone)
                if key in self.tpus:
                    return httpx.Response(200, json=self.tpus[key])
                return httpx.Response(404, json={"error": {"code": 404, "message": "Not found", "status": "NOT_FOUND"}})

        # GET .../nodes — list TPUs
        if (method == "GET" and url.endswith("/nodes")) or ("/nodes?" in url and "nodeId" not in url):
            zone = self._extract_tpu_zone(url)
            nodes = [t for (_, z), t in self.tpus.items() if z == zone]
            return httpx.Response(200, json={"nodes": nodes})

        # DELETE .../nodes/NAME
        if method == "DELETE" and "/nodes/" in url:
            if self.delete_429_remaining > 0:
                self.delete_429_remaining -= 1
                return httpx.Response(
                    429,
                    json={
                        "error": {
                            "code": 429,
                            "message": "Quota exceeded for DeleteNode requests per minute.",
                            "status": "RESOURCE_EXHAUSTED",
                        }
                    },
                )
            parts = url.split("/nodes/")
            node_name = parts[1].split("?")[0]
            zone = self._extract_tpu_zone(url)
            self.tpus.pop((node_name, zone), None)
            return httpx.Response(200, json={})

        return httpx.Response(404, json={"error": {"code": 404, "message": f"Unhandled TPU URL: {url}"}})

    def _handle_compute(self, method: str, url: str, body: dict | None) -> httpx.Response:
        # POST .../instances — create VM
        if method == "POST" and url.endswith("/instances"):
            zone = self._extract_compute_zone(url)
            name = (body or {}).get("name", "unknown")
            op_name = f"op-vm-{name}"
            vm_data = {
                "name": name,
                "status": "RUNNING",
                "zone": f"projects/{self.project}/zones/{zone}",
                "networkInterfaces": [
                    {"networkIP": "10.164.0.42", "accessConfigs": [{"natIP": "35.1.2.3", "type": "ONE_TO_ONE_NAT"}]}
                ],
                "labels": (body or {}).get("labels", {}),
                "metadata": (body or {}).get("metadata", {}),
                "serviceAccounts": (body or {}).get("serviceAccounts", []),
                "labelFingerprint": "abc123",
                "creationTimestamp": "2026-01-01T00:00:00Z",
            }
            self.vms[(name, zone)] = vm_data
            self.operations[op_name] = {"name": op_name, "status": "DONE"}
            return httpx.Response(200, json={"name": op_name, "status": "RUNNING"})

        # GET .../operations/OP — poll compute operation
        if method == "GET" and "/operations/" in url:
            op_name = url.split("/operations/")[1].split("?")[0]
            op = self.operations.get(op_name, {"name": op_name, "status": "DONE"})
            op["status"] = "DONE"  # Always complete immediately
            return httpx.Response(200, json=op)

        # GET .../serialPort
        if method == "GET" and "/serialPort" in url:
            return httpx.Response(200, json={"contents": "serial output", "next": 100})

        # GET .../instances/NAME — describe VM (must come after serialPort/setLabels/etc)
        if method == "GET" and "/instances/" in url and "/instances?" not in url:
            zone = self._extract_compute_zone(url)
            name = url.split("/instances/")[1].split("?")[0].split("/")[0]
            key = (name, zone)
            if key in self.vms:
                return httpx.Response(200, json=self.vms[key])
            return httpx.Response(404, json={"error": {"code": 404, "message": "Not found", "status": "NOT_FOUND"}})

        # GET .../instances (list, zone-scoped)
        if method == "GET" and "/instances" in url and "/instances/" not in url:
            if "/aggregated/" in url:
                # aggregatedList
                result: dict = {}
                for (_name, zone), vm in self.vms.items():
                    scope_key = f"zones/{zone}"
                    if scope_key not in result:
                        result[scope_key] = {"instances": []}
                    result[scope_key]["instances"].append(vm)
                return httpx.Response(200, json={"items": result})
            else:
                zone = self._extract_compute_zone(url)
                filter_str = ""
                if "filter=" in url:
                    filter_str = url.split("filter=")[1].split("&")[0]
                vms = [v for (_, z), v in self.vms.items() if z == zone]
                if filter_str:
                    vms = [v for v in vms if self._matches_label_filter(v.get("labels", {}), filter_str)]
                return httpx.Response(200, json={"items": vms})

        # DELETE .../instances/NAME
        if method == "DELETE" and "/instances/" in url:
            zone = self._extract_compute_zone(url)
            name = url.split("/instances/")[1].split("?")[0]
            self.vms.pop((name, zone), None)
            return httpx.Response(200, json={})

        # POST .../setLabels
        if method == "POST" and "/setLabels" in url:
            zone = self._extract_compute_zone(url)
            name = url.split("/instances/")[1].split("/setLabels")[0]
            if (name, zone) in self.vms:
                self.vms[(name, zone)]["labels"] = (body or {}).get("labels", {})
                self.vms[(name, zone)]["labelFingerprint"] = "updated"
            return httpx.Response(200, json={"name": f"op-labels-{name}", "status": "DONE"})

        # POST .../setMetadata
        if method == "POST" and "/setMetadata" in url:
            zone = self._extract_compute_zone(url)
            name = url.split("/instances/")[1].split("/setMetadata")[0]
            if (name, zone) in self.vms:
                self.vms[(name, zone)]["metadata"] = body
            return httpx.Response(200, json={"name": f"op-meta-{name}", "status": "DONE"})

        # POST .../reset
        if method == "POST" and "/reset" in url:
            return httpx.Response(200, json={"name": "op-reset", "status": "DONE"})

        return httpx.Response(404, json={"error": {"code": 404, "message": f"Unhandled compute URL: {url}"}})

    def _extract_tpu_zone(self, url: str) -> str:
        if "/locations/" in url:
            return url.split("/locations/")[1].split("/")[0]
        return "unknown"

    def _extract_compute_zone(self, url: str) -> str:
        if "/zones/" in url:
            return url.split("/zones/")[1].split("/")[0]
        return "unknown"

    def _matches_label_filter(self, labels: dict[str, str], filter_str: str) -> bool:
        import urllib.parse

        decoded = urllib.parse.unquote(filter_str)
        for part in decoded.split(" AND "):
            part = part.strip()
            if part.startswith("labels."):
                kv = part[len("labels.") :]
                if "=" in kv:
                    k, v = kv.split("=", 1)
                    if labels.get(k) != v:
                        return False
        return True


@pytest.fixture
def backend() -> GcpFakeBackend:
    return GcpFakeBackend()


@pytest.fixture
def svc(backend: GcpFakeBackend) -> CloudGcpService:
    client = httpx.Client(transport=httpx.MockTransport(backend.handle), timeout=10)
    s = CloudGcpService(PROJECT, http_client=client)
    s._token = "fake-token"
    s._expires_at = float("inf")
    return s


def _dump_http_log(log: list[HttpLog]) -> str:
    lines = []
    for i, entry in enumerate(log):
        lines.append(f"[{i}] {entry.method} {entry.url}")
        if entry.request_body:
            lines.append(f"     REQ: {json.dumps(entry.request_body, indent=2)[:500]}")
        lines.append(f"     RSP ({entry.status}): {json.dumps(entry.response_body, indent=2)[:500]}")
    return "\n".join(lines)


# ========================================================================
# Full lifecycle: VM create → describe → set labels → set metadata → list
# ========================================================================


def test_vm_full_lifecycle(svc: CloudGcpService, backend: GcpFakeBackend):
    vm = svc.vm_create(
        VmCreateRequest(
            name="ctrl-vm",
            zone=ZONE_EU,
            machine_type="e2-standard-4",
            labels={"iris-managed": "true"},
            startup_script="#!/bin/bash\necho hello",
            service_account="sa@test.iam.gserviceaccount.com",
        )
    )

    assert vm.name == "ctrl-vm", f"Expected ctrl-vm, got {vm.name}\n{_dump_http_log(backend.http_log)}"
    assert vm.internal_ip == "10.164.0.42", f"Expected IP, got {vm.internal_ip!r}\n{_dump_http_log(backend.http_log)}"
    assert vm.status == "RUNNING"

    # Set labels (read-modify-write)
    svc.vm_update_labels("ctrl-vm", ZONE_EU, {"controller": "true"})
    assert backend.vms[("ctrl-vm", ZONE_EU)]["labels"]["controller"] == "true"

    # Set metadata (read-modify-write)
    svc.vm_set_metadata("ctrl-vm", ZONE_EU, {"controller-address": "http://10.164.0.42:10000"})

    # List VMs with label filter
    vms = svc.vm_list(zones=[ZONE_EU], labels={"iris-managed": "true"})
    assert len(vms) >= 1, f"Expected >=1 VM, got {len(vms)}\n{_dump_http_log(backend.http_log)}"

    # List VMs project-wide
    all_vms = svc.vm_list(zones=[])
    assert len(all_vms) >= 1

    # Describe
    described = svc.vm_describe("ctrl-vm", ZONE_EU)
    assert described is not None
    assert described.internal_ip == "10.164.0.42"


# ========================================================================
# Full lifecycle: TPU create → describe → list → bootstrap health polling
# ========================================================================


def test_tpu_full_lifecycle(svc: CloudGcpService, backend: GcpFakeBackend):
    tpu = svc.tpu_create(
        TpuCreateRequest(
            name="test-slice",
            zone=ZONE_EU,
            accelerator_type="v5litepod-16",
            runtime_version="v2-alpha-tpuv5-lite",
            capacity_type=config_pb2.CAPACITY_TYPE_PREEMPTIBLE,
            labels={"iris-managed": "true", "env": "test"},
            metadata={"startup-script": "#!/bin/bash\necho bootstrap"},
            service_account="worker@test.iam.gserviceaccount.com",
        )
    )

    log_str = _dump_http_log(backend.http_log)

    assert tpu.name == "test-slice", f"Expected test-slice, got {tpu.name}\n{log_str}"
    assert tpu.state == "READY", f"Expected READY, got {tpu.state}\n{log_str}"
    assert len(tpu.network_endpoints) == 4, f"Expected 4 endpoints, got {tpu.network_endpoints}\n{log_str}"

    # Verify the create request body was correct
    create_req = next(e for e in backend.http_log if e.method == "POST" and "nodeId=test-slice" in e.url)
    assert create_req.request_body is not None
    assert create_req.request_body["acceleratorType"] == "v5litepod-16"
    assert create_req.request_body["runtimeVersion"] == "v2-alpha-tpuv5-lite"
    assert create_req.request_body["labels"] == {"iris-managed": "true", "env": "test"}
    assert create_req.request_body["metadata"] == {"startup-script": "#!/bin/bash\necho bootstrap"}
    assert create_req.request_body["schedulingConfig"] == {"preemptible": True}
    assert create_req.request_body["serviceAccount"] == {"email": "worker@test.iam.gserviceaccount.com"}
    assert create_req.request_body["networkConfig"]["enableExternalIps"] is True

    # Describe should return the same TPU
    described = svc.tpu_describe("test-slice", ZONE_EU)
    assert described is not None
    assert described.state == "READY"

    # List with label filter
    results = svc.tpu_list(zones=[ZONE_EU], labels={"iris-managed": "true"})
    assert len(results) == 1, f"Expected 1 TPU, got {len(results)}\n{_dump_http_log(backend.http_log)}"
    assert results[0].name == "test-slice"

    # List with non-matching label
    no_match = svc.tpu_list(zones=[ZONE_EU], labels={"iris-managed": "false"})
    assert len(no_match) == 0


def test_tpu_create_across_zones(svc: CloudGcpService, backend: GcpFakeBackend):
    """Controller creates TPU slices in multiple zones and lists all of them."""
    svc.tpu_create(
        TpuCreateRequest(
            name="slice-eu",
            zone=ZONE_EU,
            accelerator_type="v5litepod-16",
            runtime_version="v2-alpha-tpuv5-lite",
            capacity_type=config_pb2.CAPACITY_TYPE_PREEMPTIBLE,
            labels={"managed": "true"},
        )
    )
    svc.tpu_create(
        TpuCreateRequest(
            name="slice-us",
            zone=ZONE_US,
            accelerator_type="v5litepod-16",
            runtime_version="v2-alpha-tpuv5-lite",
            capacity_type=config_pb2.CAPACITY_TYPE_PREEMPTIBLE,
            labels={"managed": "true"},
        )
    )

    # List both zones
    all_tpus = svc.tpu_list(zones=[ZONE_EU, ZONE_US], labels={"managed": "true"})
    log_str = _dump_http_log(backend.http_log)
    assert len(all_tpus) == 2, f"Expected 2 TPUs, got {len(all_tpus)}\n{log_str}"

    names = {t.name for t in all_tpus}
    assert names == {"slice-eu", "slice-us"}, f"Got names {names}\n{log_str}"

    # Verify zones were correctly extracted
    for tpu in all_tpus:
        if tpu.name == "slice-eu":
            assert tpu.zone == ZONE_EU
        else:
            assert tpu.zone == ZONE_US


def test_tpu_metadata_and_network_config(svc: CloudGcpService, backend: GcpFakeBackend):
    """Verify metadata (startup-script) and network config are passed correctly."""
    large_script = "#!/bin/bash\n" + "echo line\n" * 200  # >256 chars

    svc.tpu_create(
        TpuCreateRequest(
            name="net-tpu",
            zone=ZONE_EU,
            accelerator_type="v5litepod-16",
            runtime_version="v2-alpha-tpuv5-lite",
            capacity_type=config_pb2.CAPACITY_TYPE_ON_DEMAND,
            labels={},
            metadata={"startup-script": large_script, "other-key": "other-value"},
            network="projects/test/global/networks/default",
            subnetwork="projects/test/regions/europe-west4/subnetworks/default",
        )
    )

    create_req = next(e for e in backend.http_log if e.method == "POST" and "nodeId=net-tpu" in e.url)
    body = create_req.request_body

    # Metadata must be a flat dict (not items array like Compute)
    assert body["metadata"]["startup-script"] == large_script
    assert body["metadata"]["other-key"] == "other-value"

    # Network config
    assert body["networkConfig"]["network"] == "projects/test/global/networks/default"
    assert body["networkConfig"]["subnetwork"] == "projects/test/regions/europe-west4/subnetworks/default"

    # No schedulingConfig for on-demand
    assert "schedulingConfig" not in body


def test_vm_create_startup_script_in_metadata(svc: CloudGcpService, backend: GcpFakeBackend):
    """Verify VM startup script is passed as metadata items (not flat dict like TPU)."""
    svc.vm_create(
        VmCreateRequest(
            name="boot-vm",
            zone=ZONE_EU,
            machine_type="e2-standard-4",
            startup_script="#!/bin/bash\necho hello",
        )
    )

    create_req = next(e for e in backend.http_log if e.method == "POST" and e.url.endswith("/instances"))
    body = create_req.request_body

    # VM metadata uses {"items": [...]} format
    metadata = body.get("metadata", {})
    assert "items" in metadata, f"VM metadata should use items format, got: {metadata}"
    items = {item["key"]: item["value"] for item in metadata["items"]}
    assert items["startup-script"] == "#!/bin/bash\necho hello"


def test_logging_read(svc: CloudGcpService, backend: GcpFakeBackend):
    entries = svc.logging_read('resource.type="gce_instance"', limit=100)
    assert entries == []  # Backend returns empty

    # Verify the request was correct
    log_req = next(e for e in backend.http_log if "logging.googleapis.com" in e.url)
    assert log_req.method == "POST"


def test_serial_port_output(svc: CloudGcpService, backend: GcpFakeBackend):
    # Create a VM first
    svc.vm_create(VmCreateRequest(name="serial-vm", zone=ZONE_EU, machine_type="e2-standard-4"))

    output = svc.vm_get_serial_port_output("serial-vm", ZONE_EU, start=0)
    assert output == "serial output"


# ========================================================================
# TPU operation error classification (issue #4664)
# ========================================================================


def test_tpu_create_lro_resource_exhausted_raises_quota_error(svc: CloudGcpService, backend: GcpFakeBackend):
    """LRO finishing with code=8 (RESOURCE_EXHAUSTED) raises QuotaExhaustedError,
    not generic InfraError. This lets the autoscaler log it without a stack trace."""
    # Inject a pending operation whose poll returns an error with code 8
    op_name = f"projects/{PROJECT}/locations/{ZONE_EU}/operations/op-tpu-stockout"
    backend.operations[op_name] = {
        "name": op_name,
        "done": True,
        "error": {
            "code": 8,
            "message": 'There is no more capacity in the zone "europe-west4-b"',
        },
    }

    # Patch the POST to return this operation name
    orig_handle = backend._handle_tpu

    def _patched_tpu(method, url, body):
        if method == "POST" and "/nodes" in url and "nodeId=" in url:
            return httpx.Response(200, json={"name": op_name, "done": False})
        return orig_handle(method, url, body)

    backend._handle_tpu = _patched_tpu

    with pytest.raises(QuotaExhaustedError, match="no more capacity"):
        svc.tpu_create(
            TpuCreateRequest(
                name="stockout-tpu",
                zone=ZONE_EU,
                accelerator_type="v5litepod-16",
                runtime_version="v2-alpha-tpuv5-lite",
                capacity_type=config_pb2.CAPACITY_TYPE_PREEMPTIBLE,
            )
        )


def test_tpu_delete_retries_on_quota(svc: CloudGcpService, backend: GcpFakeBackend, monkeypatch: pytest.MonkeyPatch):
    """tpu_delete retries when GCP returns 429 on the DeleteNode quota limit."""
    monkeypatch.setattr(gcp_service, "_TPU_DELETE_RETRY_BACKOFF", 0.0)

    # Create a TPU then arm the backend to return 429 on the first delete attempt
    svc.tpu_create(
        TpuCreateRequest(
            name="retry-tpu",
            zone=ZONE_EU,
            accelerator_type="v5litepod-16",
            runtime_version="v2-alpha-tpuv5-lite",
            capacity_type=config_pb2.CAPACITY_TYPE_PREEMPTIBLE,
        )
    )
    backend.delete_429_remaining = 1

    # Delete should succeed after the retry
    svc.tpu_delete("retry-tpu", ZONE_EU)

    # Verify it made 2 DELETE requests (1 rejected + 1 succeeded)
    delete_requests = [e for e in backend.http_log if e.method == "DELETE" and "/nodes/" in e.url]
    assert len(delete_requests) == 2
    assert delete_requests[0].status == 429
    assert delete_requests[1].status == 200


def test_tpu_delete_raises_after_exhausted_retries(
    svc: CloudGcpService, backend: GcpFakeBackend, monkeypatch: pytest.MonkeyPatch
):
    """tpu_delete raises QuotaExhaustedError when all retry attempts fail."""
    monkeypatch.setattr(gcp_service, "_TPU_DELETE_RETRY_BACKOFF", 0.0)

    svc.tpu_create(
        TpuCreateRequest(
            name="fail-tpu",
            zone=ZONE_EU,
            accelerator_type="v5litepod-16",
            runtime_version="v2-alpha-tpuv5-lite",
            capacity_type=config_pb2.CAPACITY_TYPE_PREEMPTIBLE,
        )
    )
    # More 429s than the retry budget
    backend.delete_429_remaining = 10

    with pytest.raises(QuotaExhaustedError, match="Quota exceeded"):
        svc.tpu_delete("fail-tpu", ZONE_EU)
